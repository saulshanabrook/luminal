pub use crate::typed_reference::dynamic_rules::binary_unroll_rules as binary_op_unroll_rules;

use std::fmt::Display;
use std::{fmt::Debug, sync::Arc};

use crate::egglog_utils::DecodeContext;
use crate::op::*;
use crate::prelude::*;
use crate::typed_reference::{
    lower::{dimension, dimensions},
    schema::{self, DimExpr, EList, Ir, OpKind},
};
use egglog_experimental::typed::{TypedError, builtins as egg, get_args};

use as_any::AsAny;
use itertools::Itertools;
use num_traits::Float;
use petgraph::{Direction, algo::toposort, prelude::StableGraph, visit::EdgeRef};
use rustc_hash::{FxHashMap, FxHashSet};
use tracing::info_span;

// Core dtype, loop, unroll, and dimension rules are installed once by the
// common driver. Backends run before these ordered fallback decoders.
pub const DECODERS: &[DecodeFn] = &[
    LoopInput::decode,
    LoopInputStatic::decode,
    LoopOutput::decode,
    LoopOutputSelect::decode,
    Constant::decode,
    Cast::decode,
    Iota::decode,
    Exp2::decode,
    Log2::decode,
    Sin::decode,
    Recip::decode,
    Sqrt::decode,
    Add::decode,
    Mul::decode,
    Mod::decode,
    LessThan::decode,
    Gather::decode,
    Scatter::decode,
    SumReduce::decode,
    MaxReduce::decode,
];

/// Core alternatives removable when another eligible implementation survives.
pub fn cleanup(kind: &OpKind) -> Result<bool, TypedError> {
    Ok(schema::ConstantArgs::get_args(kind)?.is_some()
        || schema::CastArgs::get_args(kind)?.is_some()
        || schema::IotaArgs::get_args(kind)?.is_some()
        || schema::Exp2Args::get_args(kind)?.is_some()
        || schema::Log2Args::get_args(kind)?.is_some()
        || schema::SinArgs::get_args(kind)?.is_some()
        || schema::RecipArgs::get_args(kind)?.is_some()
        || schema::SqrtArgs::get_args(kind)?.is_some()
        || schema::AddArgs::get_args(kind)?.is_some()
        || schema::MulArgs::get_args(kind)?.is_some()
        || schema::ModArgs::get_args(kind)?.is_some()
        || schema::LessThanArgs::get_args(kind)?.is_some()
        || schema::GatherArgs::get_args(kind)?.is_some()
        || schema::ScatterArgs::get_args(kind)?.is_some()
        || schema::SumArgs::get_args(kind)?.is_some()
        || schema::MaxArgs::get_args(kind)?.is_some())
}

#[derive(Default, Debug, Clone)]
pub struct Input {
    pub node: usize,
    pub label: String,
    pub dtype: DType,
}

impl Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Input({}{})",
            if self.label.is_empty() {
                "".to_string()
            } else {
                format!("{}; ", self.label)
            },
            self.dtype
        )
    }
}

impl HLIROp for Input {
    fn to_ir(&self, _: &[&Ir]) -> Ir {
        Ir::Input(self.node as i64, self.label.as_str(), self.dtype)
    }
}

impl ReferenceOp for Input {
    fn execute(&self, _: Vec<&ReferenceData>, _: &FxHashMap<char, usize>) -> ReferenceData {
        unimplemented!()
    }
}

#[derive(Default, Debug, Clone)]
pub struct Output {
    pub node: usize,
}

impl Display for Output {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Output")
    }
}

impl HLIROp for Output {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::Output(inputs[0], self.node as i64)
    }
}

impl ReferenceOp for Output {
    fn execute(&self, _: Vec<&ReferenceData>, _: &FxHashMap<char, usize>) -> ReferenceData {
        unimplemented!()
    }
}

#[derive(Default, Debug, Clone)]
pub struct CustomOpKind {
    pub id: usize,
    pub dtype: DType,
}

impl Display for CustomOpKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CustomOp({})", self.dtype)
    }
}

impl HLIROp for CustomOpKind {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::Op(OpKind::CustomOpKind(self.id as i64, self.dtype), inputs)
    }
}

impl ReferenceOp for CustomOpKind {
    fn execute(&self, _: Vec<&ReferenceData>, _: &FxHashMap<char, usize>) -> ReferenceData {
        unimplemented!()
    }
}

// --- Loop ops ---------------------------------------------------------------
//
// Automatic loop-rolling replaces N unrolled copies of a repeating body with
// a single body plus structural marker ops. All four ops in one loop share a
// `loop_id`. `iters` lives on `LoopStart` only; every other op references the
// same loop via `loop_id`.
//
//   LoopStart   — one per loop-carried slot; takes the initial value, yields
//                 the current iteration's value into the body.
//   LoopEnd     — mirror of LoopStart; takes the body's final value for the
//                 slot, yields the post-loop value.
//   LoopInput   — OpKind (variable-arity). Takes N input tensors (one per
//                 iteration) and yields the current iteration's tensor.
//   LoopOutput  — OpKind (variable-arity, sink). Takes the body's value + N
//                 target tensors; writes body[i] -> target[i] each iteration.
//
// Execution semantics and iteration driving live in the runtime compilation
// step; these ops just carry the structure through HLIR/egglog/LLIR.

#[derive(Default, Debug, Clone)]
pub struct LoopStart {
    pub loop_id: usize,
    pub slot_idx: usize,
    pub iters: Expression,
    pub dtype: DType,
}

impl Display for LoopStart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LoopStart(id={}, slot={}, iters={:?}, {})",
            self.loop_id, self.slot_idx, self.iters, self.dtype
        )
    }
}

impl HLIROp for LoopStart {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::LoopStart(
            inputs[0],
            self.loop_id as i64,
            self.slot_idx as i64,
            dimension(&self.iters, true),
            self.dtype,
        )
    }
}

impl ReferenceOp for LoopStart {
    fn execute(&self, _: Vec<&ReferenceData>, _: &FxHashMap<char, usize>) -> ReferenceData {
        unimplemented!("LoopStart is driven by the runtime loop compiler")
    }
}

#[derive(Default, Debug, Clone)]
pub struct LoopEnd {
    pub loop_id: usize,
    pub slot_idx: usize,
    pub dtype: DType,
}

impl Display for LoopEnd {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LoopEnd(id={}, slot={}, {})",
            self.loop_id, self.slot_idx, self.dtype
        )
    }
}

impl HLIROp for LoopEnd {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::LoopEnd(
            inputs[0],
            self.loop_id as i64,
            self.slot_idx as i64,
            self.dtype,
        )
    }
}

impl ReferenceOp for LoopEnd {
    fn execute(&self, _: Vec<&ReferenceData>, _: &FxHashMap<char, usize>) -> ReferenceData {
        unimplemented!("LoopEnd is driven by the runtime loop compiler")
    }
}

#[derive(Default, Debug, Clone)]
pub struct LoopInput {
    pub loop_id: usize,
    pub stream_id: usize,
    pub dtype: DType,
}

impl Display for LoopInput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LoopInput(id={}, stream={}, {})",
            self.loop_id, self.stream_id, self.dtype
        )
    }
}

impl LoopInput {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((loop_id, stream_id, dtype)) = get_args(
            kind,
            |loop_id: &egg::I64, stream_id: &egg::I64, dtype: &schema::DType| {
                OpKind::LoopInput(loop_id, stream_id, dtype)
            },
        )?
        else {
            return Ok(None);
        };
        Ok(Some(LLIROp::new::<LoopInput>(Box::new(Self {
            loop_id: usize::try_from(i64::try_from(&loop_id)?)
                .map_err(|_| TypedError::Decode("LoopInput loop_id must be nonnegative".into()))?,
            stream_id: usize::try_from(i64::try_from(&stream_id)?).map_err(|_| {
                TypedError::Decode("LoopInput stream_id must be nonnegative".into())
            })?,
            dtype: context.dtype(&dtype)?,
        }))))
    }
}

impl HLIROp for LoopInput {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::Op(
            OpKind::LoopInput(self.loop_id as i64, self.stream_id as i64, self.dtype),
            inputs,
        )
    }
}

impl ReferenceOp for LoopInput {
    fn execute(&self, _: Vec<&ReferenceData>, _: &FxHashMap<char, usize>) -> ReferenceData {
        unimplemented!("LoopInput is driven by the runtime loop compiler")
    }
}

/// Iteration-independent boundary input: the same value flows into every
/// iteration of a loop. Structurally a `LoopInput` whose per-iteration
/// sources have all been proven equal (via the `identical_inputs` egglog
/// relation) collapses into `LoopInputStatic` with a single-element IList,
/// and that in turn collapses via a further rewrite into just its inner
/// value — so egglog search can explore any of the three representations.
/// At unroll time `LoopInputStatic` lowers to a plain edge: every cloned
/// body node in every iteration references the single shared source.
#[derive(Default, Debug, Clone)]
pub struct LoopInputStatic {
    pub loop_id: usize,
    pub stream_id: usize,
    pub dtype: DType,
}

impl Display for LoopInputStatic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LoopInputStatic(id={}, stream={}, {})",
            self.loop_id, self.stream_id, self.dtype
        )
    }
}

impl LoopInputStatic {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((loop_id, stream_id, dtype)) = get_args(
            kind,
            |loop_id: &egg::I64, stream_id: &egg::I64, dtype: &schema::DType| {
                OpKind::LoopInputStatic(loop_id, stream_id, dtype)
            },
        )?
        else {
            return Ok(None);
        };
        Ok(Some(LLIROp::new::<LoopInputStatic>(Box::new(Self {
            loop_id: usize::try_from(i64::try_from(&loop_id)?).map_err(|_| {
                TypedError::Decode("LoopInputStatic loop_id must be nonnegative".into())
            })?,
            stream_id: usize::try_from(i64::try_from(&stream_id)?).map_err(|_| {
                TypedError::Decode("LoopInputStatic stream_id must be nonnegative".into())
            })?,
            dtype: context.dtype(&dtype)?,
        }))))
    }
}

impl HLIROp for LoopInputStatic {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::Op(
            OpKind::LoopInputStatic(self.loop_id as i64, self.stream_id as i64, self.dtype),
            inputs,
        )
    }
}

impl ReferenceOp for LoopInputStatic {
    fn execute(&self, _: Vec<&ReferenceData>, _: &FxHashMap<char, usize>) -> ReferenceData {
        unimplemented!("LoopInputStatic is driven by the runtime loop compiler")
    }
}

/// Marker for the per-iter output stream of a rolled loop. Mirrors `LoopInput`
/// in reverse: a single body producer (one incoming edge) feeds the marker, and
/// `LoopOutputSelect(i)` nodes hang off it to pluck iteration `i`'s value for
/// downstream consumers (any post-region op — `Output` HLIR, downstream
/// computation, etc.).
#[derive(Default, Debug, Clone)]
pub struct LoopOutput {
    pub loop_id: usize,
    pub stream_id: usize,
    pub dtype: DType,
}

impl Display for LoopOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LoopOutput(id={}, stream={}, {})",
            self.loop_id, self.stream_id, self.dtype
        )
    }
}

impl LoopOutput {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((loop_id, stream_id, dtype)) = get_args(
            kind,
            |loop_id: &egg::I64, stream_id: &egg::I64, dtype: &schema::DType| {
                OpKind::LoopOutput(loop_id, stream_id, dtype)
            },
        )?
        else {
            return Ok(None);
        };
        Ok(Some(LLIROp::new::<LoopOutput>(Box::new(Self {
            loop_id: usize::try_from(i64::try_from(&loop_id)?)
                .map_err(|_| TypedError::Decode("LoopOutput loop_id must be nonnegative".into()))?,
            stream_id: usize::try_from(i64::try_from(&stream_id)?).map_err(|_| {
                TypedError::Decode("LoopOutput stream_id must be nonnegative".into())
            })?,
            dtype: context.dtype(&dtype)?,
        }))))
    }
}

impl HLIROp for LoopOutput {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::Op(
            OpKind::LoopOutput(self.loop_id as i64, self.stream_id as i64, self.dtype),
            inputs,
        )
    }
}

impl ReferenceOp for LoopOutput {
    fn execute(&self, _: Vec<&ReferenceData>, _: &FxHashMap<char, usize>) -> ReferenceData {
        unimplemented!("LoopOutput is driven by the runtime loop compiler")
    }
}

/// Per-iteration extractor for a `LoopOutput` stream. Mirrors a per-iter
/// `LoopInput` source slot in reverse: every cross-region edge that originally
/// went from iteration `i`'s body producer to a post-region consumer is
/// rewired through `LoopOutputSelect { iter: i, ... }`. At unroll time
/// `Select(i)` lowers to the iter-`i` body clone's producer; at collapse time
/// every Select lowers to iter-0's producer.
#[derive(Default, Debug, Clone)]
pub struct LoopOutputSelect {
    pub loop_id: usize,
    pub stream_id: usize,
    pub iter: usize,
    pub dtype: DType,
}

impl Display for LoopOutputSelect {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LoopOutputSelect(id={}, stream={}, iter={}, {})",
            self.loop_id, self.stream_id, self.iter, self.dtype
        )
    }
}

impl LoopOutputSelect {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((loop_id, stream_id, iter, dtype)) = get_args(
            kind,
            |loop_id: &egg::I64, stream_id: &egg::I64, iter: &egg::I64, dtype: &schema::DType| {
                OpKind::LoopOutputSelect(loop_id, stream_id, iter, dtype)
            },
        )?
        else {
            return Ok(None);
        };
        Ok(Some(LLIROp::new::<LoopOutputSelect>(Box::new(Self {
            loop_id: usize::try_from(i64::try_from(&loop_id)?).map_err(|_| {
                TypedError::Decode("LoopOutputSelect loop_id must be nonnegative".into())
            })?,
            stream_id: usize::try_from(i64::try_from(&stream_id)?).map_err(|_| {
                TypedError::Decode("LoopOutputSelect stream_id must be nonnegative".into())
            })?,
            iter: usize::try_from(i64::try_from(&iter)?).map_err(|_| {
                TypedError::Decode("LoopOutputSelect iter must be nonnegative".into())
            })?,
            dtype: context.dtype(&dtype)?,
        }))))
    }
}

impl HLIROp for LoopOutputSelect {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::Op(
            OpKind::LoopOutputSelect(
                self.loop_id as i64,
                self.stream_id as i64,
                self.iter as i64,
                self.dtype,
            ),
            inputs,
        )
    }
}

impl ReferenceOp for LoopOutputSelect {
    fn execute(&self, _: Vec<&ReferenceData>, _: &FxHashMap<char, usize>) -> ReferenceData {
        unimplemented!("LoopOutputSelect is driven by the runtime loop compiler")
    }
}

/// Produces a single number constant from an expression or a float
#[derive(Clone, PartialEq, Default)]
pub struct Constant(pub f32);
impl Debug for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Constant({:?})", self.0)
    }
}

impl Display for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.0)
    }
}

impl HLIROp for Constant {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        // Widening the f32 is exact. Scaling by 10^6 needs at most 38
        // significand bits, so ties-to-even matches six-decimal formatting.
        // Formatting and parsing canonicalized every NaN payload and sign.
        let value = if self.0.is_nan() {
            f64::NAN
        } else {
            (f64::from(self.0) * 1_000_000.0).round_ties_even() / 1_000_000.0
        };
        Ir::Op(OpKind::Constant(value), inputs)
    }
}

impl Constant {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        _context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((value,)) = get_args(kind, |value: &egg::F64| OpKind::Constant(value))? else {
            return Ok(None);
        };
        Ok(Some(LLIROp::new::<dyn ReferenceOp>(Box::new(Self(
            f64::try_from(&value)? as f32,
        )))))
    }
}

impl ReferenceOp for Constant {
    fn execute(&self, _: Vec<&ReferenceData>, _: &FxHashMap<char, usize>) -> ReferenceData {
        ReferenceData::F32(vec![self.0])
    }
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct Iota(pub Expression, pub Expression);
impl Display for Iota {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Iota({}; {})", self.0, self.1)
    }
}
impl HLIROp for Iota {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::Op(
            OpKind::Iota(dimension(&self.0, true), dimension(&self.1, true)),
            inputs,
        )
    }
}
impl Iota {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((expr, range)) = get_args(kind, |expr: &DimExpr, range: &DimExpr| {
            OpKind::Iota(expr, range)
        })?
        else {
            return Ok(None);
        };
        Ok(Some(LLIROp::new::<dyn ReferenceOp>(Box::new(Self(
            context.expression(&expr)?,
            context.expression(&range)?,
        )))))
    }
}
impl ReferenceOp for Iota {
    fn execute(&self, _: Vec<&ReferenceData>, dyn_map: &FxHashMap<char, usize>) -> ReferenceData {
        let length = self.1.exec(dyn_map).unwrap();
        let expr = self.0.resolve_vars(dyn_map);
        ReferenceData::Int(
            (0..length)
                .map(|i| expr.exec_single_var(i) as i32)
                .collect(),
        )
    }
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct Cast(pub Expression, pub DType);
impl Display for Cast {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cast({})", self.1)
    }
}
impl HLIROp for Cast {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::Op(OpKind::Cast(dimension(&self.0, true), self.1), inputs)
    }
}
impl Cast {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((size, dtype)) = get_args(kind, |size: &DimExpr, dtype: &schema::DType| {
            OpKind::Cast(size, dtype)
        })?
        else {
            return Ok(None);
        };
        Ok(Some(LLIROp::new::<dyn ReferenceOp>(Box::new(Self(
            context.expression(&size)?,
            context.dtype(&dtype)?,
        )))))
    }
}
impl ReferenceOp for Cast {
    fn execute(&self, input: Vec<&ReferenceData>, _: &FxHashMap<char, usize>) -> ReferenceData {
        match self.1 {
            DType::F32 => ReferenceData::F32(match &input[0] {
                ReferenceData::F32(f) => f.clone(),
                ReferenceData::F16(f) => f.iter().map(|f| f.to_f32()).collect(),
                ReferenceData::Bf16(f) => f.iter().map(|f| f.to_f32()).collect(),
                ReferenceData::Int(i) => i.iter().map(|i| *i as f32).collect(),
                ReferenceData::I64(i) => i.iter().map(|i| *i as f32).collect(),
                ReferenceData::F64(f) => f.iter().map(|f| *f as f32).collect(),
                ReferenceData::Bool(b) => b.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect(),
            }),
            DType::F64 => ReferenceData::F64(match &input[0] {
                ReferenceData::F64(f) => f.clone(),
                ReferenceData::F32(f) => f.iter().map(|f| *f as f64).collect(),
                ReferenceData::F16(f) => f.iter().map(|f| f.to_f32() as f64).collect(),
                ReferenceData::Bf16(f) => f.iter().map(|f| f.to_f32() as f64).collect(),
                ReferenceData::Int(i) => i.iter().map(|i| *i as f64).collect(),
                ReferenceData::I64(i) => i.iter().map(|i| *i as f64).collect(),
                ReferenceData::Bool(b) => b.iter().map(|b| if *b { 1.0 } else { 0.0 }).collect(),
            }),
            DType::Int => ReferenceData::Int(match &input[0] {
                ReferenceData::F32(f) => f.iter().map(|f| *f as i32).collect(),
                ReferenceData::F16(f) => f.iter().map(|f| f.to_f32() as i32).collect(),
                ReferenceData::Bf16(f) => f.iter().map(|f| f.to_f32() as i32).collect(),
                ReferenceData::Int(i) => i.clone(),
                // Saturating `Cast(I64 -> Int)`. This is the explicit
                // graph node — `Cast` IS the user/translator-emitted
                // operation, not an implicit bridge. Values outside the
                // i32 range wrap via Rust's `as i32`, matching
                // `tensor.to(torch.int32)` semantics on overflow.
                ReferenceData::I64(i) => i.iter().map(|i| *i as i32).collect(),
                ReferenceData::F64(f) => f.iter().map(|f| *f as i32).collect(),
                ReferenceData::Bool(b) => b.iter().map(|b| if *b { 1 } else { 0 }).collect(),
            }),
            DType::I64 => ReferenceData::I64(match &input[0] {
                ReferenceData::I64(i) => i.clone(),
                ReferenceData::Int(i) => i.iter().map(|i| *i as i64).collect(),
                ReferenceData::F32(f) => f.iter().map(|f| *f as i64).collect(),
                ReferenceData::F64(f) => f.iter().map(|f| *f as i64).collect(),
                ReferenceData::F16(f) => f.iter().map(|f| f.to_f32() as i64).collect(),
                ReferenceData::Bf16(f) => f.iter().map(|f| f.to_f32() as i64).collect(),
                ReferenceData::Bool(b) => b.iter().map(|b| if *b { 1 } else { 0 }).collect(),
            }),
            DType::F16 => ReferenceData::F16(match &input[0] {
                ReferenceData::F32(f) => f.iter().copied().map(f16::from_f32).collect(),
                ReferenceData::F16(f) => f.clone(),
                ReferenceData::Bf16(f) => f.iter().map(|f| f16::from_f32(f.to_f32())).collect(),
                ReferenceData::Int(i) => i.iter().map(|i| f16::from_f32(*i as f32)).collect(),
                ReferenceData::I64(i) => i.iter().map(|i| f16::from_f32(*i as f32)).collect(),
                ReferenceData::F64(f) => f.iter().map(|f| f16::from_f64(*f)).collect(),
                ReferenceData::Bool(b) => b
                    .iter()
                    .map(|b| f16::from_f32(if *b { 1.0 } else { 0.0 }))
                    .collect(),
            }),
            DType::Bf16 => ReferenceData::Bf16(match &input[0] {
                ReferenceData::F32(f) => f.iter().copied().map(bf16::from_f32).collect(),
                ReferenceData::F16(f) => f.iter().map(|f| bf16::from_f32(f.to_f32())).collect(),
                ReferenceData::Bf16(f) => f.clone(),
                ReferenceData::Int(i) => i.iter().map(|i| bf16::from_f32(*i as f32)).collect(),
                ReferenceData::I64(i) => i.iter().map(|i| bf16::from_f32(*i as f32)).collect(),
                ReferenceData::F64(f) => f.iter().map(|f| bf16::from_f64(*f)).collect(),
                ReferenceData::Bool(b) => b
                    .iter()
                    .map(|b| bf16::from_f32(if *b { 1.0 } else { 0.0 }))
                    .collect(),
            }),
            DType::Bool => ReferenceData::Bool(match &input[0] {
                ReferenceData::F32(f) => f.iter().map(|f| *f != 0.0).collect(),
                ReferenceData::F16(f) => f.iter().map(|f| f.to_f32() != 0.0).collect(),
                ReferenceData::Bf16(f) => f.iter().map(|f| f.to_f32() != 0.0).collect(),
                ReferenceData::Int(i) => i.iter().map(|i| *i != 0).collect(),
                ReferenceData::I64(i) => i.iter().map(|i| *i != 0).collect(),
                ReferenceData::F64(f) => f.iter().map(|f| *f != 0.0).collect(),
                ReferenceData::Bool(b) => b.clone(),
            }),
            other => {
                unimplemented!("Cast to {other} is not yet supported in reference interpreter")
            }
        }
    }
}

// Unary Op (A -> A)

fn unary_impl(
    inp: &ReferenceData,
    shape: &[Expression],
    strides: &[Expression],
    dyn_map: &FxHashMap<char, usize>,
    f32_fn: fn(f32) -> f32,
    f16_fn: fn(f16) -> f16,
    bf16_fn: fn(bf16) -> bf16,
) -> ReferenceData {
    let ind = StridedIterator::new(shape, strides, dyn_map);
    match &inp {
        ReferenceData::F32(f) => ReferenceData::F32(ind.map(|i| f32_fn(f[i])).collect()),
        ReferenceData::F16(f) => ReferenceData::F16(ind.map(|i| f16_fn(f[i])).collect()),
        ReferenceData::Bf16(f) => ReferenceData::Bf16(ind.map(|i| bf16_fn(f[i])).collect()),
        ReferenceData::Int(_) => panic!("unary_impl: no Int kernel — cast to F32 at the call site"),
        ReferenceData::I64(_) => panic!("unary_impl: no I64 kernel — cast to F32 at the call site"),
        // No F64 transcendental kernel. Refuse loudly rather than
        // silently bridging through F32 — the caller asked for double
        // precision and that's not what an F32 bridge delivers. Fix at
        // the call site: cast inputs to F32 (`x.to(torch.float32)`) and
        // accept the precision, or wait for a native F64 transcendental
        // kernel.
        ReferenceData::F64(_) => panic!(
            "unary_impl: no F64 transcendental kernel — cast inputs to F32 \
             at the call site (`x.to(torch.float32)`), or wait for the F64 \
             transcendental kernel follow-up. Silent F32 bridging is \
             intentionally rejected: it would hide a precision downgrade \
             behind an `F64` dtype tag."
        ),
        ReferenceData::Bool(_) => {
            panic!("unary_impl: no Bool kernel — cast to F32 at the call site")
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Log2 {
    pub shape: Vec<Expression>,
    pub strides: Vec<Expression>,
    pub input_shape: ShapeTracker,
}
impl Display for Log2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Log2")
    }
}
impl HLIROp for Log2 {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::Op(
            OpKind::Log2(
                dimensions(&self.input_shape.dims),
                dimensions(&self.input_shape.strides),
                dimensions(&self.input_shape.contiguous().strides),
            ),
            inputs,
        )
    }
}
impl Log2 {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((shape, strides, _out_strides)) = get_args(
            kind,
            |shape: &EList, strides: &EList, _out_strides: &EList| {
                OpKind::Log2(shape, strides, _out_strides)
            },
        )?
        else {
            return Ok(None);
        };
        let shape = context.expressions(&shape)?;
        let strides = context.expressions(&strides)?;
        Ok(Some(LLIROp::new::<dyn ReferenceOp>(Box::new(Self {
            shape,
            strides,
            ..Default::default()
        }))))
    }
}
impl ReferenceOp for Log2 {
    fn execute(
        &self,
        inputs: Vec<&ReferenceData>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> ReferenceData {
        unary_impl(
            inputs[0],
            &self.shape,
            &self.strides,
            dyn_map,
            |f| f.log2(),
            |f| f.log2(),
            |f| f.log2(),
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Exp2 {
    pub shape: Vec<Expression>,
    pub strides: Vec<Expression>,
    pub input_shape: ShapeTracker,
}
impl Display for Exp2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Exp2")
    }
}
impl HLIROp for Exp2 {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::Op(
            OpKind::Exp2(
                dimensions(&self.input_shape.dims),
                dimensions(&self.input_shape.strides),
                dimensions(&self.input_shape.contiguous().strides),
            ),
            inputs,
        )
    }
}
impl Exp2 {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((shape, strides, _out_strides)) = get_args(
            kind,
            |shape: &EList, strides: &EList, _out_strides: &EList| {
                OpKind::Exp2(shape, strides, _out_strides)
            },
        )?
        else {
            return Ok(None);
        };
        let shape = context.expressions(&shape)?;
        let strides = context.expressions(&strides)?;
        Ok(Some(LLIROp::new::<dyn ReferenceOp>(Box::new(Self {
            shape,
            strides,
            ..Default::default()
        }))))
    }
}
impl ReferenceOp for Exp2 {
    fn execute(
        &self,
        inputs: Vec<&ReferenceData>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> ReferenceData {
        unary_impl(
            inputs[0],
            &self.shape,
            &self.strides,
            dyn_map,
            |f| f.exp2(),
            |f| f.exp2(),
            |f| f.exp2(),
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sin {
    pub shape: Vec<Expression>,
    pub strides: Vec<Expression>,
    pub input_shape: ShapeTracker,
}
impl Display for Sin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sin")
    }
}
impl HLIROp for Sin {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::Op(
            OpKind::Sin(
                dimensions(&self.input_shape.dims),
                dimensions(&self.input_shape.strides),
                dimensions(&self.input_shape.contiguous().strides),
            ),
            inputs,
        )
    }
}

impl Sin {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((shape, strides, _out_strides)) = get_args(
            kind,
            |shape: &EList, strides: &EList, _out_strides: &EList| {
                OpKind::Sin(shape, strides, _out_strides)
            },
        )?
        else {
            return Ok(None);
        };
        let shape = context.expressions(&shape)?;
        let strides = context.expressions(&strides)?;
        Ok(Some(LLIROp::new::<dyn ReferenceOp>(Box::new(Self {
            shape,
            strides,
            ..Default::default()
        }))))
    }
}
impl ReferenceOp for Sin {
    fn execute(
        &self,
        inputs: Vec<&ReferenceData>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> ReferenceData {
        unary_impl(
            inputs[0],
            &self.shape,
            &self.strides,
            dyn_map,
            |f| f.sin(),
            |f| f.sin(),
            |f| f.sin(),
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Recip {
    pub shape: Vec<Expression>,
    pub strides: Vec<Expression>,
    pub input_shape: ShapeTracker,
}
impl Display for Recip {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Recip")
    }
}
impl HLIROp for Recip {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::Op(
            OpKind::Recip(
                dimensions(&self.input_shape.dims),
                dimensions(&self.input_shape.strides),
                dimensions(&self.input_shape.contiguous().strides),
            ),
            inputs,
        )
    }
}

impl Recip {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((shape, strides, _out_strides)) = get_args(
            kind,
            |shape: &EList, strides: &EList, _out_strides: &EList| {
                OpKind::Recip(shape, strides, _out_strides)
            },
        )?
        else {
            return Ok(None);
        };
        let shape = context.expressions(&shape)?;
        let strides = context.expressions(&strides)?;
        Ok(Some(LLIROp::new::<dyn ReferenceOp>(Box::new(Self {
            shape,
            strides,
            ..Default::default()
        }))))
    }
}
impl ReferenceOp for Recip {
    fn execute(
        &self,
        inputs: Vec<&ReferenceData>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> ReferenceData {
        unary_impl(
            inputs[0],
            &self.shape,
            &self.strides,
            dyn_map,
            |f| f.recip(),
            |f| f.recip(),
            |f| f.recip(),
        )
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sqrt {
    pub shape: Vec<Expression>,
    pub strides: Vec<Expression>,
    pub input_shape: ShapeTracker,
}
impl Display for Sqrt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Sqrt")
    }
}
impl HLIROp for Sqrt {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::Op(
            OpKind::Sqrt(
                dimensions(&self.input_shape.dims),
                dimensions(&self.input_shape.strides),
                dimensions(&self.input_shape.contiguous().strides),
            ),
            inputs,
        )
    }
}

impl Sqrt {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((shape, strides, _out_strides)) = get_args(
            kind,
            |shape: &EList, strides: &EList, _out_strides: &EList| {
                OpKind::Sqrt(shape, strides, _out_strides)
            },
        )?
        else {
            return Ok(None);
        };
        let shape = context.expressions(&shape)?;
        let strides = context.expressions(&strides)?;
        Ok(Some(LLIROp::new::<dyn ReferenceOp>(Box::new(Self {
            shape,
            strides,
            ..Default::default()
        }))))
    }
}
impl ReferenceOp for Sqrt {
    fn execute(
        &self,
        inputs: Vec<&ReferenceData>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> ReferenceData {
        unary_impl(
            inputs[0],
            &self.shape,
            &self.strides,
            dyn_map,
            |f| f.sqrt(),
            |f| f.sqrt(),
            |f| f.sqrt(),
        )
    }
}

// Binary Ops (A x A -> A)

fn bin_fn<A: Copy>(
    a_ind: StridedIterator,
    a: &[A],
    b_ind: StridedIterator,
    b: &[A],
    op: impl Fn(A, A) -> A,
) -> Vec<A> {
    let a_shape = a_ind.shape.clone();
    let a_strides = a_ind.strides.clone();
    let b_shape = b_ind.shape.clone();
    let b_strides = b_ind.strides.clone();
    a_ind
        .zip(b_ind)
        .map(|(i, j)| {
            assert!(
                i < a.len(),
                "bin_fn: a index {i} out of bounds (a.len={}), shape={a_shape:?}, strides={a_strides:?}",
                a.len(),
            );
            assert!(
                j < b.len(),
                "bin_fn: b index {j} out of bounds (b.len={}), shape={b_shape:?}, strides={b_strides:?}",
                b.len(),
            );
            op(a[i], b[j])
        })
        .collect()
}

fn bin_cmp_fn<A: Copy>(
    a_ind: StridedIterator,
    a: &[A],
    b_ind: StridedIterator,
    b: &[A],
    op: impl Fn(A, A) -> bool,
) -> Vec<bool> {
    let a_shape = a_ind.shape.clone();
    let a_strides = a_ind.strides.clone();
    let b_shape = b_ind.shape.clone();
    let b_strides = b_ind.strides.clone();
    a_ind
        .zip(b_ind)
        .map(|(i, j)| {
            assert!(
                i < a.len(),
                "bin_cmp_fn: a index {i} out of bounds (a.len={}), shape={a_shape:?}, strides={a_strides:?}",
                a.len(),
            );
            assert!(
                j < b.len(),
                "bin_cmp_fn: b index {j} out of bounds (b.len={}), shape={b_shape:?}, strides={b_strides:?}",
                b.len(),
            );
            op(a[i], b[j])
        })
        .collect()
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Add {
    pub shape: Vec<Expression>,
    pub a_strides: Vec<Expression>,
    pub b_strides: Vec<Expression>,
    pub input_shapes: Vec<ShapeTracker>,
}
impl Display for Add {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Add")
    }
}
impl HLIROp for Add {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::Op(
            OpKind::Add(
                dimensions(&self.input_shapes[0].dims),
                dimensions(&self.input_shapes[0].strides),
                dimensions(&self.input_shapes[1].strides),
                dimensions(&self.input_shapes[0].contiguous().strides),
            ),
            inputs,
        )
    }
}

impl Add {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((shape, a_strides, b_strides, _out_strides)) = get_args(
            kind,
            |shape: &EList, a_strides: &EList, b_strides: &EList, _out_strides: &EList| {
                OpKind::Add(shape, a_strides, b_strides, _out_strides)
            },
        )?
        else {
            return Ok(None);
        };
        let shape = context.expressions(&shape)?;
        let a_strides = context.expressions(&a_strides)?;
        let b_strides = context.expressions(&b_strides)?;
        Ok(Some(LLIROp::new::<dyn ReferenceOp>(Box::new(Self {
            shape,
            a_strides,
            b_strides,
            ..Default::default()
        }))))
    }
}

impl ReferenceOp for Add {
    fn execute(
        &self,
        inputs: Vec<&ReferenceData>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> ReferenceData {
        let (a, b) = (inputs[0], inputs[1]);
        let (a_ind, b_ind) = (
            StridedIterator::new(&self.shape, &self.a_strides, dyn_map),
            StridedIterator::new(&self.shape, &self.b_strides, dyn_map),
        );
        match (a, b) {
            (ReferenceData::F32(a), ReferenceData::F32(b)) => {
                ReferenceData::F32(bin_fn(a_ind, a, b_ind, b, |x, y| x + y))
            }
            (ReferenceData::F16(a), ReferenceData::F16(b)) => {
                ReferenceData::F16(bin_fn(a_ind, a, b_ind, b, |x, y| x + y))
            }
            (ReferenceData::Bf16(a), ReferenceData::Bf16(b)) => {
                ReferenceData::Bf16(bin_fn(a_ind, a, b_ind, b, |x, y| x + y))
            }
            (ReferenceData::Int(a), ReferenceData::Int(b)) => {
                ReferenceData::Int(bin_fn(a_ind, a, b_ind, b, |x, y| x + y))
            }
            (ReferenceData::I64(a), ReferenceData::I64(b)) => {
                ReferenceData::I64(bin_fn(a_ind, a, b_ind, b, |x, y| x + y))
            }
            (ReferenceData::F64(a), ReferenceData::F64(b)) => {
                ReferenceData::F64(bin_fn(a_ind, a, b_ind, b, |x, y| x + y))
            }
            (ReferenceData::Bool(_), ReferenceData::Bool(_)) => {
                panic!("Cannot add Bool tensors, cast to F32 first")
            }
            _ => panic!("Add inputs must have the same dtype"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mul {
    pub shape: Vec<Expression>,
    pub a_strides: Vec<Expression>,
    pub b_strides: Vec<Expression>,
    pub input_shapes: Vec<ShapeTracker>,
}
impl Display for Mul {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mul")
    }
}
impl HLIROp for Mul {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::Op(
            OpKind::Mul(
                dimensions(&self.input_shapes[0].dims),
                dimensions(&self.input_shapes[0].strides),
                dimensions(&self.input_shapes[1].strides),
                dimensions(&self.input_shapes[0].contiguous().strides),
            ),
            inputs,
        )
    }
}

impl Mul {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((shape, a_strides, b_strides, _out_strides)) = get_args(
            kind,
            |shape: &EList, a_strides: &EList, b_strides: &EList, _out_strides: &EList| {
                OpKind::Mul(shape, a_strides, b_strides, _out_strides)
            },
        )?
        else {
            return Ok(None);
        };
        let shape = context.expressions(&shape)?;
        let a_strides = context.expressions(&a_strides)?;
        let b_strides = context.expressions(&b_strides)?;
        Ok(Some(LLIROp::new::<dyn ReferenceOp>(Box::new(Self {
            shape,
            a_strides,
            b_strides,
            ..Default::default()
        }))))
    }
}

impl ReferenceOp for Mul {
    fn execute(
        &self,
        inputs: Vec<&ReferenceData>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> ReferenceData {
        let (a, b) = (inputs[0], inputs[1]);
        let (a_ind, b_ind) = (
            StridedIterator::new(&self.shape, &self.a_strides, dyn_map),
            StridedIterator::new(&self.shape, &self.b_strides, dyn_map),
        );
        match (a, b) {
            (ReferenceData::F32(a), ReferenceData::F32(b)) => {
                ReferenceData::F32(bin_fn(a_ind, a, b_ind, b, |x, y| x * y))
            }
            (ReferenceData::F16(a), ReferenceData::F16(b)) => {
                ReferenceData::F16(bin_fn(a_ind, a, b_ind, b, |x, y| x * y))
            }
            (ReferenceData::Bf16(a), ReferenceData::Bf16(b)) => {
                ReferenceData::Bf16(bin_fn(a_ind, a, b_ind, b, |x, y| x * y))
            }
            (ReferenceData::Int(a), ReferenceData::Int(b)) => {
                ReferenceData::Int(bin_fn(a_ind, a, b_ind, b, |x, y| x * y))
            }
            (ReferenceData::I64(a), ReferenceData::I64(b)) => {
                ReferenceData::I64(bin_fn(a_ind, a, b_ind, b, |x, y| x * y))
            }
            (ReferenceData::F64(a), ReferenceData::F64(b)) => {
                ReferenceData::F64(bin_fn(a_ind, a, b_ind, b, |x, y| x * y))
            }
            (ReferenceData::Bool(_), ReferenceData::Bool(_)) => {
                panic!("Cannot multiply Bool tensors, cast to F32 first")
            }
            _ => panic!("Mul inputs must have the same dtype"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mod {
    pub shape: Vec<Expression>,
    pub a_strides: Vec<Expression>,
    pub b_strides: Vec<Expression>,
    pub input_shapes: Vec<ShapeTracker>,
}
impl Display for Mod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Mod")
    }
}
impl HLIROp for Mod {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::Op(
            OpKind::Mod(
                dimensions(&self.input_shapes[0].dims),
                dimensions(&self.input_shapes[0].strides),
                dimensions(&self.input_shapes[1].strides),
                dimensions(&self.input_shapes[0].contiguous().strides),
            ),
            inputs,
        )
    }
}

impl Mod {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((shape, a_strides, b_strides, _out_strides)) = get_args(
            kind,
            |shape: &EList, a_strides: &EList, b_strides: &EList, _out_strides: &EList| {
                OpKind::Mod(shape, a_strides, b_strides, _out_strides)
            },
        )?
        else {
            return Ok(None);
        };
        let shape = context.expressions(&shape)?;
        let a_strides = context.expressions(&a_strides)?;
        let b_strides = context.expressions(&b_strides)?;
        Ok(Some(LLIROp::new::<dyn ReferenceOp>(Box::new(Self {
            shape,
            a_strides,
            b_strides,
            ..Default::default()
        }))))
    }
}

impl ReferenceOp for Mod {
    fn execute(
        &self,
        inputs: Vec<&ReferenceData>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> ReferenceData {
        let (a, b) = (inputs[0], inputs[1]);
        let (a_ind, b_ind) = (
            StridedIterator::new(&self.shape, &self.a_strides, dyn_map),
            StridedIterator::new(&self.shape, &self.b_strides, dyn_map),
        );
        match (a, b) {
            (ReferenceData::F32(a), ReferenceData::F32(b)) => {
                ReferenceData::F32(bin_fn(a_ind, a, b_ind, b, |x, y| x % y))
            }
            (ReferenceData::F16(a), ReferenceData::F16(b)) => {
                ReferenceData::F16(bin_fn(a_ind, a, b_ind, b, |x, y| x % y))
            }
            (ReferenceData::Bf16(a), ReferenceData::Bf16(b)) => {
                ReferenceData::Bf16(bin_fn(a_ind, a, b_ind, b, |x, y| x % y))
            }
            (ReferenceData::Int(a), ReferenceData::Int(b)) => {
                ReferenceData::Int(bin_fn(a_ind, a, b_ind, b, |x, y| x % y))
            }
            (ReferenceData::I64(a), ReferenceData::I64(b)) => {
                ReferenceData::I64(bin_fn(a_ind, a, b_ind, b, |x, y| x % y))
            }
            (ReferenceData::F64(a), ReferenceData::F64(b)) => {
                ReferenceData::F64(bin_fn(a_ind, a, b_ind, b, |x, y| x % y))
            }
            (ReferenceData::Bool(_), ReferenceData::Bool(_)) => panic!("Cannot mod Bool tensors"),
            _ => panic!("Mod inputs must have the same dtype"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LessThan {
    pub shape: Vec<Expression>,
    pub a_strides: Vec<Expression>,
    pub b_strides: Vec<Expression>,
    pub input_shapes: Vec<ShapeTracker>,
}
impl Display for LessThan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LessThan")
    }
}
impl HLIROp for LessThan {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::Op(
            OpKind::LessThan(
                dimensions(&self.input_shapes[0].dims),
                dimensions(&self.input_shapes[0].strides),
                dimensions(&self.input_shapes[1].strides),
                dimensions(&self.input_shapes[0].contiguous().strides),
            ),
            inputs,
        )
    }
}

impl LessThan {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((shape, a_strides, b_strides, _out_strides)) = get_args(
            kind,
            |shape: &EList, a_strides: &EList, b_strides: &EList, _out_strides: &EList| {
                OpKind::LessThan(shape, a_strides, b_strides, _out_strides)
            },
        )?
        else {
            return Ok(None);
        };
        let shape = context.expressions(&shape)?;
        let a_strides = context.expressions(&a_strides)?;
        let b_strides = context.expressions(&b_strides)?;
        Ok(Some(LLIROp::new::<dyn ReferenceOp>(Box::new(Self {
            shape,
            a_strides,
            b_strides,
            ..Default::default()
        }))))
    }
}

impl ReferenceOp for LessThan {
    fn execute(
        &self,
        inputs: Vec<&ReferenceData>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> ReferenceData {
        let (a, b) = (inputs[0], inputs[1]);
        let (a_ind, b_ind) = (
            StridedIterator::new(&self.shape, &self.a_strides, dyn_map),
            StridedIterator::new(&self.shape, &self.b_strides, dyn_map),
        );
        match (a, b) {
            (ReferenceData::F32(a), ReferenceData::F32(b)) => {
                ReferenceData::Bool(bin_cmp_fn(a_ind, a, b_ind, b, |x, y| x < y))
            }
            (ReferenceData::F16(a), ReferenceData::F16(b)) => {
                ReferenceData::Bool(bin_cmp_fn(a_ind, a, b_ind, b, |x, y| x < y))
            }
            (ReferenceData::Bf16(a), ReferenceData::Bf16(b)) => {
                ReferenceData::Bool(bin_cmp_fn(a_ind, a, b_ind, b, |x, y| x < y))
            }
            (ReferenceData::Int(a), ReferenceData::Int(b)) => {
                ReferenceData::Bool(bin_cmp_fn(a_ind, a, b_ind, b, |x, y| x < y))
            }
            (ReferenceData::I64(a), ReferenceData::I64(b)) => {
                ReferenceData::Bool(bin_cmp_fn(a_ind, a, b_ind, b, |x, y| x < y))
            }
            (ReferenceData::F64(a), ReferenceData::F64(b)) => {
                ReferenceData::Bool(bin_cmp_fn(a_ind, a, b_ind, b, |x, y| x < y))
            }
            (ReferenceData::Bool(a), ReferenceData::Bool(b)) => {
                ReferenceData::Bool(bin_cmp_fn(a_ind, a, b_ind, b, |x, y| !x & y))
            }
            _ => panic!("LessThan inputs must have the same dtype"),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Gather {
    pub index_shape: Vec<Expression>,
    pub data_shape: Vec<Expression>,
    pub index_strides: Vec<Expression>,
    pub data_strides: Vec<Expression>,
    pub input_shapes: Vec<ShapeTracker>,
}
impl Display for Gather {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gather")
    }
}
impl HLIROp for Gather {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::Op(
            OpKind::Gather(
                dimensions(&self.input_shapes[0].dims),
                dimensions(&self.input_shapes[0].strides),
                dimensions(&self.input_shapes[1].dims),
                dimensions(&self.input_shapes[1].strides),
            ),
            inputs,
        )
    }
}

impl Gather {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((index_shape, index_strides, data_shape, data_strides)) = get_args(
            kind,
            |index_shape: &EList,
             index_strides: &EList,
             data_shape: &EList,
             data_strides: &EList| {
                OpKind::Gather(index_shape, index_strides, data_shape, data_strides)
            },
        )?
        else {
            return Ok(None);
        };
        let index_shape = context.expressions(&index_shape)?;
        let index_strides = context.expressions(&index_strides)?;
        let data_shape = context.expressions(&data_shape)?;
        let data_strides = context.expressions(&data_strides)?;
        Ok(Some(LLIROp::new::<dyn ReferenceOp>(Box::new(Self {
            index_shape,
            index_strides,
            data_shape,
            data_strides,
            ..Default::default()
        }))))
    }
}
impl ReferenceOp for Gather {
    fn execute(
        &self,
        inputs: Vec<&ReferenceData>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> ReferenceData {
        let (indexes, data) = (inputs[0], inputs[1]);
        let indexes_ind = StridedIterator::new(&self.index_shape, &self.index_strides, dyn_map);
        let data_ind =
            StridedIterator::new(&self.data_shape, &self.data_strides, dyn_map).collect_vec();
        let ReferenceData::Int(indexes) = indexes else {
            panic!("indexes must be int!")
        };
        match data {
            ReferenceData::F32(a) => ReferenceData::F32(
                indexes_ind
                    .map(|i| a[data_ind[indexes[i] as usize]])
                    .collect(),
            ),
            ReferenceData::F16(a) => ReferenceData::F16(
                indexes_ind
                    .map(|i| a[data_ind[indexes[i] as usize]])
                    .collect(),
            ),
            ReferenceData::Bf16(a) => ReferenceData::Bf16(
                indexes_ind
                    .map(|i| a[data_ind[indexes[i] as usize]])
                    .collect(),
            ),
            ReferenceData::Int(a) => ReferenceData::Int(
                indexes_ind
                    .map(|i| a[data_ind[indexes[i] as usize]])
                    .collect(),
            ),
            ReferenceData::I64(a) => ReferenceData::I64(
                indexes_ind
                    .map(|i| a[data_ind[indexes[i] as usize]])
                    .collect(),
            ),
            ReferenceData::F64(a) => ReferenceData::F64(
                indexes_ind
                    .map(|i| a[data_ind[indexes[i] as usize]])
                    .collect(),
            ),
            ReferenceData::Bool(a) => ReferenceData::Bool(
                indexes_ind
                    .map(|i| a[data_ind[indexes[i] as usize]])
                    .collect(),
            ),
        }
    }
}

// Scatter Op (inverse of Gather)

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Scatter {
    pub dest_shape: Vec<Expression>,
    pub dest_strides: Vec<Expression>,
    pub index_shape: Vec<Expression>,
    pub index_strides: Vec<Expression>,
    pub src_strides: Vec<Expression>,
}
impl Display for Scatter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Scatter")
    }
}
impl HLIROp for Scatter {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        Ir::Op(
            OpKind::Scatter(
                dimensions(&self.dest_shape),
                dimensions(&self.dest_strides),
                dimensions(&self.index_shape),
                dimensions(&self.index_strides),
                dimensions(&self.src_strides),
            ),
            inputs,
        )
    }
}

impl Scatter {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((dest_shape, dest_strides, index_shape, index_strides, src_strides)) = get_args(
            kind,
            |dest_shape: &EList,
             dest_strides: &EList,
             index_shape: &EList,
             index_strides: &EList,
             src_strides: &EList| {
                OpKind::Scatter(
                    dest_shape,
                    dest_strides,
                    index_shape,
                    index_strides,
                    src_strides,
                )
            },
        )?
        else {
            return Ok(None);
        };
        let dest_shape = context.expressions(&dest_shape)?;
        let dest_strides = context.expressions(&dest_strides)?;
        let index_shape = context.expressions(&index_shape)?;
        let index_strides = context.expressions(&index_strides)?;
        let src_strides = context.expressions(&src_strides)?;
        Ok(Some(LLIROp::new::<dyn ReferenceOp>(Box::new(Self {
            dest_shape,
            dest_strides,
            index_shape,
            index_strides,
            src_strides,
        }))))
    }
}
impl ReferenceOp for Scatter {
    fn execute(
        &self,
        inputs: Vec<&ReferenceData>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> ReferenceData {
        let (dest, indexes, src) = (inputs[0], inputs[1], inputs[2]);
        let dest_ind =
            StridedIterator::new(&self.dest_shape, &self.dest_strides, dyn_map).collect_vec();
        let index_ind = StridedIterator::new(&self.index_shape, &self.index_strides, dyn_map);
        let src_ind =
            StridedIterator::new(&self.index_shape, &self.src_strides, dyn_map).collect_vec();
        let ReferenceData::Int(indexes) = indexes else {
            panic!("indexes must be int!")
        };
        macro_rules! scatter_impl {
            ($variant:ident, $dest_data:expr, $src_data:expr) => {{
                let mut output: Vec<_> = dest_ind.iter().map(|&i| $dest_data[i]).collect();
                for (src_idx, flat_i) in index_ind.enumerate() {
                    let idx = indexes[flat_i] as usize;
                    if idx < output.len() {
                        output[idx] = $src_data[src_ind[src_idx]];
                    }
                }
                ReferenceData::$variant(output)
            }};
        }
        match (dest, src) {
            (ReferenceData::F32(d), ReferenceData::F32(s)) => scatter_impl!(F32, d, s),
            (ReferenceData::F64(d), ReferenceData::F64(s)) => scatter_impl!(F64, d, s),
            (ReferenceData::F16(d), ReferenceData::F16(s)) => scatter_impl!(F16, d, s),
            (ReferenceData::Bf16(d), ReferenceData::Bf16(s)) => scatter_impl!(Bf16, d, s),
            (ReferenceData::Int(d), ReferenceData::Int(s)) => scatter_impl!(Int, d, s),
            (ReferenceData::I64(d), ReferenceData::I64(s)) => scatter_impl!(I64, d, s),
            (ReferenceData::Bool(d), ReferenceData::Bool(s)) => scatter_impl!(Bool, d, s),
            _ => panic!("dest and src must have the same dtype!"),
        }
    }
}

// Reduce Ops (A -> B (different shape))

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SumReduce {
    pub dim: usize,
    pub shape: Vec<Expression>,
    pub strides: Vec<Expression>,
    pub iters: Expression,
    pub iter_stride: Expression,
    pub input_shape: ShapeTracker,
}
impl Display for SumReduce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SumReduce(dim={})", self.dim)
    }
}
impl HLIROp for SumReduce {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        let mut reduced_shape = self.input_shape;
        reduced_shape.remove_dim(self.dim);
        let mut reduced_strides = self.input_shape.strides;
        reduced_strides.remove(self.dim);
        Ir::Op(
            OpKind::Sum(
                dimensions(&reduced_shape.dims),
                dimension(&self.input_shape.dims[self.dim], true),
                dimensions(&reduced_strides),
                dimension(&self.input_shape.strides[self.dim], true),
                dimensions(&reduced_shape.contiguous().strides),
            ),
            inputs,
        )
    }
}

impl SumReduce {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((shape, iters, strides, iter_stride, _out_strides)) = get_args(
            kind,
            |shape: &EList,
             iters: &DimExpr,
             strides: &EList,
             iter_stride: &DimExpr,
             _out_strides: &EList| {
                OpKind::Sum(shape, iters, strides, iter_stride, _out_strides)
            },
        )?
        else {
            return Ok(None);
        };
        let shape = context.expressions(&shape)?;
        let strides = context.expressions(&strides)?;
        Ok(Some(LLIROp::new::<dyn ReferenceOp>(Box::new(Self {
            dim: 0,
            shape,
            iters: context.expression(&iters)?,
            strides,
            iter_stride: context.expression(&iter_stride)?,
            ..Default::default()
        }))))
    }
}

impl ReferenceOp for SumReduce {
    fn execute(
        &self,
        inputs: Vec<&ReferenceData>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> ReferenceData {
        let ind = StridedIterator::new(&self.shape, &self.strides, dyn_map);
        // Resolve dyn vars in iter_stride, then evaluate z-stride at each iteration
        let mut resolved_stride = self.iter_stride;
        for (&var, &val) in dyn_map {
            resolved_stride = resolved_stride.substitute(var, Expression::from(val as i32));
        }
        let iters = self.iters.exec(dyn_map).unwrap();
        match inputs[0] {
            ReferenceData::F32(a) => ReferenceData::F32(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .sum()
                })
                .collect(),
            ),
            ReferenceData::F16(a) => ReferenceData::F16(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .sum()
                })
                .collect(),
            ),
            ReferenceData::Bf16(a) => ReferenceData::Bf16(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .sum()
                })
                .collect(),
            ),
            ReferenceData::Int(a) => ReferenceData::Int(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .sum()
                })
                .collect(),
            ),
            ReferenceData::I64(a) => ReferenceData::I64(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .sum::<i64>()
                })
                .collect(),
            ),
            ReferenceData::F64(a) => ReferenceData::F64(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .sum::<f64>()
                })
                .collect(),
            ),
            ReferenceData::Bool(_) => panic!("Cannot sum Bool tensors, cast to F32 first"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct MaxReduce {
    pub dim: usize,
    pub shape: Vec<Expression>,
    pub strides: Vec<Expression>,
    pub iters: Expression,
    pub iter_stride: Expression,
    pub input_shape: ShapeTracker,
}
impl Display for MaxReduce {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MaxReduce(dim={})", self.dim)
    }
}
impl HLIROp for MaxReduce {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        let mut reduced_shape = self.input_shape;
        reduced_shape.remove_dim(self.dim);
        let mut reduced_strides = self.input_shape.strides;
        reduced_strides.remove(self.dim);
        Ir::Op(
            OpKind::Max(
                dimensions(&reduced_shape.dims),
                dimension(&self.input_shape.dims[self.dim], true),
                dimensions(&reduced_strides),
                dimension(&self.input_shape.strides[self.dim], true),
                dimensions(&reduced_shape.contiguous().strides),
            ),
            inputs,
        )
    }
}

impl MaxReduce {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((shape, iters, strides, iter_stride, _out_strides)) = get_args(
            kind,
            |shape: &EList,
             iters: &DimExpr,
             strides: &EList,
             iter_stride: &DimExpr,
             _out_strides: &EList| {
                OpKind::Max(shape, iters, strides, iter_stride, _out_strides)
            },
        )?
        else {
            return Ok(None);
        };
        let shape = context.expressions(&shape)?;
        let strides = context.expressions(&strides)?;
        Ok(Some(LLIROp::new::<dyn ReferenceOp>(Box::new(Self {
            dim: 0,
            shape,
            iters: context.expression(&iters)?,
            strides,
            iter_stride: context.expression(&iter_stride)?,
            ..Default::default()
        }))))
    }
}

impl ReferenceOp for MaxReduce {
    fn execute(
        &self,
        inputs: Vec<&ReferenceData>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> ReferenceData {
        let ind = StridedIterator::new(&self.shape, &self.strides, dyn_map);
        // Resolve dyn vars in iter_stride, then evaluate z-stride at each iteration
        let mut resolved_stride = self.iter_stride;
        for (&var, &val) in dyn_map {
            resolved_stride = resolved_stride.substitute(var, Expression::from(val as i32));
        }
        let iters = self.iters.exec(dyn_map).unwrap();
        match inputs[0] {
            ReferenceData::F32(a) => ReferenceData::F32(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .max_by(|a, b| a.total_cmp(b))
                        .unwrap_or_default()
                })
                .collect(),
            ),
            ReferenceData::F16(a) => ReferenceData::F16(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .max_by(|a, b| a.total_cmp(b))
                        .unwrap_or_default()
                })
                .collect(),
            ),
            ReferenceData::Bf16(a) => ReferenceData::Bf16(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .max_by(|a, b| a.total_cmp(b))
                        .unwrap_or_default()
                })
                .collect(),
            ),
            ReferenceData::Int(a) => ReferenceData::Int(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .max()
                        .unwrap_or_default()
                })
                .collect(),
            ),
            ReferenceData::I64(a) => ReferenceData::I64(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .max()
                        .unwrap_or_default()
                })
                .collect(),
            ),
            ReferenceData::F64(a) => ReferenceData::F64(
                ind.map(|start| {
                    (0..iters)
                        .map(|i| a[start + resolved_stride.exec_single_var(i)])
                        .max_by(|a, b| a.total_cmp(b))
                        .unwrap_or_default()
                })
                .collect(),
            ),
            ReferenceData::Bool(_) => panic!("Cannot max-reduce Bool tensors"),
        }
    }
}

pub trait ReferenceOp: Debug + AsAny + Send + Sync {
    fn execute(
        &self,
        inputs: Vec<&ReferenceData>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> ReferenceData;
}

#[derive(Debug, Clone)]
pub enum ReferenceData {
    F32(Vec<f32>),
    F16(Vec<f16>),
    Bf16(Vec<bf16>),
    Int(Vec<i32>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
}

impl ReferenceData {
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn len(&self) -> usize {
        match self {
            ReferenceData::F32(v) => v.len(),
            ReferenceData::F16(v) => v.len(),
            ReferenceData::Bf16(v) => v.len(),
            ReferenceData::Int(v) => v.len(),
            ReferenceData::I64(v) => v.len(),
            ReferenceData::F64(v) => v.len(),
            ReferenceData::Bool(v) => v.len(),
        }
    }
    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            ReferenceData::F32(v) => v.clone(),
            ReferenceData::F16(v) => v.iter().map(|v| v.to_f32()).collect(),
            ReferenceData::Bf16(v) => v.iter().map(|v| v.to_f32()).collect(),
            ReferenceData::Int(v) => v.iter().map(|v| *v as f32).collect(),
            ReferenceData::I64(v) => v.iter().map(|v| *v as f32).collect(),
            ReferenceData::F64(v) => v.iter().map(|v| *v as f32).collect(),
            ReferenceData::Bool(v) => v.iter().map(|v| if *v { 1.0 } else { 0.0 }).collect(),
        }
    }

    pub fn to_f16_vec(&self) -> Vec<f16> {
        match self {
            ReferenceData::F32(v) => v.iter().copied().map(f16::from_f32).collect(),
            ReferenceData::F16(v) => v.clone(),
            ReferenceData::Bf16(v) => v.iter().map(|v| f16::from_f32(v.to_f32())).collect(),
            ReferenceData::Int(v) => v.iter().map(|v| f16::from_f32(*v as f32)).collect(),
            ReferenceData::I64(v) => v.iter().map(|v| f16::from_f32(*v as f32)).collect(),
            ReferenceData::F64(v) => v.iter().map(|v| f16::from_f64(*v)).collect(),
            ReferenceData::Bool(v) => v
                .iter()
                .map(|v| f16::from_f32(if *v { 1.0 } else { 0.0 }))
                .collect(),
        }
    }

    pub fn to_i32_vec(&self) -> Vec<i32> {
        match self {
            ReferenceData::F32(v) => v.iter().map(|v| *v as i32).collect(),
            ReferenceData::F16(v) => v.iter().map(|v| v.to_f32() as i32).collect(),
            ReferenceData::Bf16(v) => v.iter().map(|v| v.to_f32() as i32).collect(),
            ReferenceData::Int(v) => v.clone(),
            ReferenceData::I64(v) => v.iter().map(|v| *v as i32).collect(),
            ReferenceData::F64(v) => v.iter().map(|v| *v as i32).collect(),
            ReferenceData::Bool(v) => v.iter().map(|v| if *v { 1 } else { 0 }).collect(),
        }
    }

    pub fn to_bool_vec(&self) -> Vec<bool> {
        match self {
            ReferenceData::F32(v) => v.iter().map(|v| *v != 0.0).collect(),
            ReferenceData::F16(v) => v.iter().map(|v| v.to_f32() != 0.0).collect(),
            ReferenceData::Bf16(v) => v.iter().map(|v| v.to_f32() != 0.0).collect(),
            ReferenceData::Int(v) => v.iter().map(|v| *v != 0).collect(),
            ReferenceData::I64(v) => v.iter().map(|v| *v != 0).collect(),
            ReferenceData::F64(v) => v.iter().map(|v| *v != 0.0).collect(),
            ReferenceData::Bool(v) => v.clone(),
        }
    }
}

impl From<Vec<f32>> for ReferenceData {
    fn from(value: Vec<f32>) -> Self {
        ReferenceData::F32(value)
    }
}
impl From<Vec<f16>> for ReferenceData {
    fn from(value: Vec<f16>) -> Self {
        ReferenceData::F16(value)
    }
}
impl From<Vec<bf16>> for ReferenceData {
    fn from(value: Vec<bf16>) -> Self {
        ReferenceData::Bf16(value)
    }
}
impl From<Vec<i32>> for ReferenceData {
    fn from(value: Vec<i32>) -> Self {
        ReferenceData::Int(value)
    }
}
impl From<Vec<i64>> for ReferenceData {
    fn from(value: Vec<i64>) -> Self {
        ReferenceData::I64(value)
    }
}
// No `From<Vec<f64>> for ReferenceData` impl. Adding it makes plain
// float literals (`vec![1.0, 2.0, 3.0]` passed to `set_data`)
// ambiguous between `Vec<f32>` and `Vec<f64>` and forces every test
// site to spell out `Vec::<f32>::from([...])`. Callers that need to
// construct an F64 buffer can do `ReferenceData::F64(my_vec)` directly.
impl From<Vec<bool>> for ReferenceData {
    fn from(value: Vec<bool>) -> Self {
        ReferenceData::Bool(value)
    }
}

macro_rules! impl_reference_data_from_ref {
    ($ty:ty, $variant:ident) => {
        impl From<&[$ty]> for ReferenceData {
            fn from(value: &[$ty]) -> Self {
                ReferenceData::$variant(value.to_vec())
            }
        }

        impl From<&Vec<$ty>> for ReferenceData {
            fn from(value: &Vec<$ty>) -> Self {
                ReferenceData::$variant(value.clone())
            }
        }
    };
}

macro_rules! impl_reference_data_from_array_ref {
    ($ty:ty, $variant:ident) => {
        impl<const N: usize> From<&[$ty; N]> for ReferenceData {
            fn from(value: &[$ty; N]) -> Self {
                ReferenceData::$variant(value.to_vec())
            }
        }
    };
}

impl_reference_data_from_ref!(f32, F32);
impl_reference_data_from_ref!(f16, F16);
impl_reference_data_from_ref!(bf16, Bf16);
impl_reference_data_from_ref!(i32, Int);
impl_reference_data_from_ref!(bool, Bool);

impl_reference_data_from_array_ref!(f32, F32);
impl_reference_data_from_array_ref!(f16, F16);
impl_reference_data_from_array_ref!(bf16, Bf16);
impl_reference_data_from_array_ref!(i32, Int);
impl_reference_data_from_array_ref!(bool, Bool);

#[derive(Default)]
pub struct ReferenceRuntime {
    pub buffers: FxHashMap<NodeIndex, ReferenceData>,
    pub graph: StableGraph<Arc<Box<dyn ReferenceOp>>, ()>,
}

impl ReferenceRuntime {
    pub fn set_data(&mut self, id: impl ToId, data: impl Into<ReferenceData>) {
        let id = id.to_id();
        let local_id = self
            .graph
            .node_indices()
            .find(|n| {
                if let Some(Input { node, .. }) = (**self.graph[*n]).as_any().downcast_ref() {
                    *node == id.index()
                } else {
                    false
                }
            })
            .unwrap_or_else(|| panic!("{id:?} is not an Input node in the graph"));
        self.buffers.insert(local_id, data.into());
    }
}

impl Runtime for ReferenceRuntime {
    type CompileArg = ();
    type ExecReturn = ();
    type ProfileMetric = usize;

    fn initialize(_: Self::CompileArg) -> Self {
        Self {
            buffers: Default::default(),
            graph: Default::default(),
        }
    }

    fn profile(
        &mut self,
        _: &LLIRGraph,
        _: &FxHashMap<char, usize>,
        _: usize,
        _: Option<std::time::Duration>,
    ) -> (Self::ProfileMetric, String) {
        (0, "0 ms".to_string())
    }

    fn aggregate_profile_metrics(metrics: &[Self::ProfileMetric]) -> Self::ProfileMetric {
        metrics.iter().copied().sum()
    }

    fn load_llir(&mut self, llir_graph: &LLIRGraph) {
        // Extract reference-op graph
        let mut graph = StableGraph::new();
        for node in llir_graph.node_weights() {
            if let Some(op) = node.to_dialect::<dyn ReferenceOp>() {
                graph.add_node(op.clone());
            } else if let Some(input) = node.to_op::<Input>() {
                graph.add_node(Arc::new(Box::new(input.clone())));
            } else {
                let output = node.to_op::<Output>().unwrap();
                graph.add_node(Arc::new(Box::new(output.clone())));
            }
        }
        for edge in llir_graph.edge_indices() {
            let (start, end) = llir_graph.edge_endpoints(edge).unwrap();
            graph.add_edge(start, end, ());
        }

        self.graph = graph;
    }

    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) -> Self::ExecReturn {
        for node in toposort(&self.graph, None).unwrap() {
            if (**self.graph[node]).as_any().is::<Input>() {
                continue;
            }

            if (**self.graph[node]).as_any().is::<Output>() {
                // Copy source buffer into Output node's own slot
                let source = self
                    .graph
                    .edges_directed(node, Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .next()
                    .unwrap()
                    .source();
                let data = self.buffers[&source].clone();
                self.buffers.insert(node, data);
                continue;
            }

            let span = info_span!("reference_op", op = %format!("{:?}", self.graph[node]));
            let _entered = span.enter();
            let inputs = self
                .graph
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|e| e.id())
                .map(|e| &self.buffers[&e.source()])
                .collect_vec();
            let output = self.graph[node].execute(inputs, dyn_map);
            self.buffers.insert(node, output);
        }

        // Consume all non-Output buffers (inputs + intermediates)
        let output_nodes: FxHashSet<NodeIndex> = self
            .graph
            .node_indices()
            .filter(|n| (**self.graph[*n]).as_any().is::<Output>())
            .collect();
        self.buffers.retain(|k, _| output_nodes.contains(k));
    }
}

impl ReferenceRuntime {
    pub fn get_f32(&self, id: impl ToId) -> &Vec<f32> {
        let id = id.to_id();
        let output_id = self
            .graph
            .node_indices()
            .find(|n| {
                if let Some(Output { node }) = (**self.graph[*n]).as_any().downcast_ref::<Output>()
                {
                    *node == id.index()
                } else {
                    false
                }
            })
            .unwrap();
        let ReferenceData::F32(f) = self.buffers.get(&output_id).unwrap() else {
            panic!()
        };
        f
    }
}

struct StridedIterator {
    shape: Vec<usize>,
    strides: Vec<Expression>,
    index: Vec<usize>,
    done: bool,
}

impl StridedIterator {
    fn new(shape: &[Expression], strides: &[Expression], dyn_map: &FxHashMap<char, usize>) -> Self {
        let shape: Vec<usize> = shape.iter().map(|e| e.exec(dyn_map).unwrap()).collect();
        // Resolve dynamic vars in strides but keep 'z' as a variable
        let strides: Vec<Expression> = strides
            .iter()
            .copied()
            .map(|e| e.resolve_vars(dyn_map))
            .collect();
        Self {
            index: vec![0; shape.len()],
            strides,
            done: shape.contains(&0),
            shape,
        }
    }
}

impl Iterator for StridedIterator {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let fin = self
            .strides
            .iter()
            .zip(self.index.iter())
            .map(|(s, &idx)| s.exec_single_var(idx))
            .sum();

        for i in (0..self.shape.len()).rev() {
            self.index[i] += 1;
            if self.index[i] < self.shape[i] {
                return Some(fin);
            }
            self.index[i] = 0;
        }

        self.done = true;
        Some(fin)
    }
}
