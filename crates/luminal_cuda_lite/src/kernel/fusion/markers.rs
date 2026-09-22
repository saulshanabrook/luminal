use egglog_experimental::typed::{self as typed, TypedError, builtins as egg};
use luminal::egglog_utils::DecodeContext;
use luminal::typed_reference::schema::{DimExpr, EList, IList, Ir, OpKind};
// =========================================================================
// Fusion boundary markers — FusionStart and FusionEnd.
//
// Tag-like LLIR ops that bracket a region of elementwise ops destined to
// be emitted as a single CUDA kernel:
//   - N FusionStart nodes per region (one per FS leaf — distinct external
//     reads),
//   - exactly 1 FusionEnd per region.
//
// `FusionEnd::rewrites()` carries the seven rule families that build and
// extend regions (pair-fuse / grow / merge); the actual single-kernel
// codegen lives in `region_codegen`. Both markers' `compile()` is
// `unreachable!()` — region codegen folds them away
// before kernel_to_host's compile loop reaches an interior node.
// =========================================================================

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{op::*, prelude::*};

use crate::kernel::KernelOp;

type CompileOut = (
    CudaFunction,
    Arc<CudaModule>,
    String,
    (Expression, Expression, Expression),
    (Expression, Expression, Expression),
    Expression,
    FxHashMap<char, CudaSlice<u8>>,
);

// =========================================================================
// FusionStart
// =========================================================================

#[derive(Default, Debug, Clone)]
pub struct FusionStart {
    pub(crate) shape: Vec<Expression>,
    pub(crate) strides: Vec<Expression>,
    pub(crate) dtype: DType,
}

impl FusionStart {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_shape, field_strides, field_dtype)) = typed::get_args(
            kind,
            |shape: &EList, strides: &EList, dtype: &luminal::typed_reference::schema::DType| {
                crate::typed_rules::fusion_start(shape, strides, dtype)
            },
        )?
        else {
            return Ok(None);
        };
        let decoded_shape = context.expressions(&field_shape)?;
        let decoded_strides = context.expressions(&field_strides)?;
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            shape: decoded_shape,
            strides: decoded_strides,
            dtype: context.dtype(&field_dtype)?,
        }))))
    }
}

impl KernelOp for FusionStart {
    fn compile(
        &self,
        _stream: &Arc<CudaStream>,
        _compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> CompileOut {
        unreachable!("FusionStart must be compiled through fusion region codegen")
    }
    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }
    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }
    fn output_dtype(&self) -> DType {
        self.dtype
    }
    fn kernel_name(&self) -> &'static str {
        "FusionStart"
    }
    fn output_aliases_input(&self) -> Option<usize> {
        Some(0)
    }
}

// =========================================================================
// FusionEnd
// =========================================================================

#[derive(Default, Debug, Clone)]
pub struct FusionEnd {
    pub(crate) shape: Vec<Expression>,
    pub(crate) strides: Vec<Expression>,
    pub(crate) dtype: DType,
}

pub(crate) mod fusion_end {
    use crate::typed_rules::*;
    use egglog_experimental::typed::builtins as egg;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    // No idempotence or dissolve rule: unifying nested FusionStart markers,
    // or FS(FE(x)) with x, creates eclass cycles after growth. The grow rules
    // already compose adjacent regions; extra identity layers are bounded by
    // the run schedule.

    #[ruleset]
    pub(crate) fn fusion_grow(
        shape: &EList,
        s: &EList,
        dt: &DType,
        inner: &Ir,
        fe: &Ir,
        u: &Ir,
        dt_in: &DType,
        size: &DimExpr,
        dt_out: &DType,
        cast: &Ir,
        x: &Ir,
        fs_c: &Ir,
        a_s: &EList,
        inner_a: &Ir,
        b_s: &EList,
        out_s: &EList,
        b: &Ir,
        bin: &Ir,
        inner_b: &Ir,
        a: &Ir,
        fs_u: &Ir,
        fs_bin: &Ir,
    ) -> Vec<Rule> {
        let unaries = [
            ("Sin", OpKind::Sin(shape, s, s)),
            ("Sqrt", OpKind::Sqrt(shape, s, s)),
            ("Exp2", OpKind::Exp2(shape, s, s)),
            ("Log2", OpKind::Log2(shape, s, s)),
            ("Recip", OpKind::Recip(shape, s, s)),
        ];
        let binaries = [
            ("Add", OpKind::Add(shape, a_s, b_s, out_s)),
            ("Mul", OpKind::Mul(shape, a_s, b_s, out_s)),
        ];
        let fs_b = &Ir::Op(fusion_start(shape, b_s, dt), [b]);
        let fs_a = &Ir::Op(fusion_start(shape, a_s, dt), [a]);
        let fs_x = &Ir::Op(fusion_start(shape, s, dt), [x]);
        let mut rules = Vec::new();
        for (opcode, hlir) in &unaries {
            let element = &Ir::Op(cuda_unary_elementwise(*opcode, shape, s, s, dt), [inner]);
            let end = &Ir::Op(fusion_end(shape, s, dt), [element]);
            rules.push(
                rule(
                    (
                        eq(fe, Ir::Op(fusion_end(shape, s, dt), [inner])),
                        eq(u, Ir::Op(hlir, [fe])),
                    ),
                    (element, end, union(u, end), set(end.dtype(), dt)),
                )
                .label(format!("grow-FE-U-{opcode}")),
            );
        }
        let cast_op = &Ir::Op(cuda_unary_elementwise("Cast", shape, s, s, dt_out), [inner]);
        let cast_end = &Ir::Op(fusion_end(shape, s, dt_out), [cast_op]);
        let cast_start = &Ir::Op(fusion_start(shape, s, dt_in), [x]);
        let cast_started = &Ir::Op(
            cuda_unary_elementwise("Cast", shape, s, s, dt_out),
            [cast_start],
        );
        rules.extend([
            rule(
                (
                    eq(fe, Ir::Op(fusion_end(shape, s, dt_in), [inner])),
                    eq(cast, Ir::Op(OpKind::Cast(size, dt_out), [fe])),
                ),
                (
                    cast_op,
                    cast_end,
                    union(cast, cast_end),
                    set(cast_end.dtype(), dt_out),
                ),
            )
            .label("grow-FE-Cast"),
            rule(
                (
                    eq(cast, Ir::Op(OpKind::Cast(size, dt_out), [x])),
                    eq(fs_c, Ir::Op(fusion_start(shape, s, dt_out), [cast])),
                    eq(dt_in, x.dtype()),
                ),
                (cast_start, cast_started, union(fs_c, cast_started)),
            )
            .label("grow-Cast-FS"),
        ]);
        for (opcode, hlir) in &binaries {
            let lhs = &Ir::Op(
                cuda_binary_elementwise(*opcode, shape, a_s, b_s, out_s, dt),
                [inner_a, fs_b],
            );
            let lhs_end = &Ir::Op(fusion_end(shape, out_s, dt), [lhs]);
            let rhs = &Ir::Op(
                cuda_binary_elementwise(*opcode, shape, a_s, b_s, out_s, dt),
                [fs_a, inner_b],
            );
            let rhs_end = &Ir::Op(fusion_end(shape, out_s, dt), [rhs]);
            rules.extend([
                rule(
                    (
                        eq(fe, Ir::Op(fusion_end(shape, a_s, dt), [inner_a])),
                        eq(bin, Ir::Op(hlir, [fe, b])),
                    ),
                    (
                        fs_b,
                        lhs,
                        lhs_end,
                        union(bin, lhs_end),
                        set(lhs_end.dtype(), dt),
                    ),
                )
                .label(format!("grow-FE-B-lhs-{opcode}")),
                rule(
                    (
                        eq(fe, Ir::Op(fusion_end(shape, b_s, dt), [inner_b])),
                        eq(bin, Ir::Op(hlir, [a, fe])),
                    ),
                    (
                        fs_a,
                        rhs,
                        rhs_end,
                        union(bin, rhs_end),
                        set(rhs_end.dtype(), dt),
                    ),
                )
                .label(format!("grow-FE-B-rhs-{opcode}")),
            ]);
        }
        for (opcode, hlir) in &unaries {
            let element = &Ir::Op(cuda_unary_elementwise(*opcode, shape, s, s, dt), [fs_x]);
            rules.push(
                rule(
                    (
                        eq(u, Ir::Op(hlir, [x])),
                        eq(fs_u, Ir::Op(fusion_start(shape, s, dt), [u])),
                    ),
                    (fs_x, element, union(fs_u, element)),
                )
                .label(format!("grow-U-FS-{opcode}")),
            );
        }
        for (opcode, hlir) in &binaries {
            let element = &Ir::Op(
                cuda_binary_elementwise(*opcode, shape, a_s, b_s, out_s, dt),
                [fs_a, fs_b],
            );
            rules.push(
                rule(
                    (
                        eq(bin, Ir::Op(hlir, [a, b])),
                        eq(fs_bin, Ir::Op(fusion_start(shape, out_s, dt), [bin])),
                    ),
                    (fs_a, fs_b, element, union(fs_bin, element)),
                )
                .label(format!("grow-B-FS-{opcode}")),
            );
        }
        rules
    }

    #[ruleset]
    pub(crate) fn cleanup(
        shape: &EList,
        s: &EList,
        dt_in: &DType,
        inner: &Ir,
        inner_fe: &Ir,
        bad_fs: &Ir,
        dt_out: &DType,
        bad_elem: &Ir,
        bad_fe: &Ir,
        good_elem: &Ir,
        good_fe: &Ir,
        size: &DimExpr,
        x: &Ir,
        cast: &Ir,
        fs_c: &Ir,
        is: &EList,
        os: &EList,
        fs_x: &Ir,
        dt: &DType,
        a_s: &EList,
        inner_a: &Ir,
        b_s: &EList,
        b: &Ir,
        fs_b: &Ir,
        out_s: &EList,
        inner_b: &Ir,
        a: &Ir,
        fs_a: &Ir,
        fe_shape: &EList,
        fe_s: &EList,
        op: &egg::String,
        inner_shape: &EList,
        inner_in_s: &EList,
        inner_s: &EList,
        inner_inputs: &IList,
    ) -> Vec<Rule> {
        let mut rules = vec![
            rule(
                (
                    // Cast growth can merge FS with the FE it wraps. The
                    // fused survivor below proves deleting the redundant
                    // FS-wrapping-FE breaks the cycle without emptying it.
                    eq(inner_fe, Ir::Op(fusion_end(shape, s, dt_in), [inner])),
                    eq(bad_fs, Ir::Op(fusion_start(shape, s, dt_in), [inner_fe])),
                    eq(
                        bad_elem,
                        Ir::Op(
                            cuda_unary_elementwise("Cast", shape, s, s, dt_out),
                            [bad_fs],
                        ),
                    ),
                    eq(bad_fe, Ir::Op(fusion_end(shape, s, dt_out), [bad_elem])),
                    eq(
                        good_elem,
                        Ir::Op(cuda_unary_elementwise("Cast", shape, s, s, dt_out), [inner]),
                    ),
                    eq(good_fe, Ir::Op(fusion_end(shape, s, dt_out), [good_elem])),
                    eq(bad_fe, good_fe),
                ),
                (delete(Ir::Op(fusion_start(shape, s, dt_in), [inner_fe])),),
            )
            .label("cleanup-nested-FS-FE-cast"),
            // Genome freeze: bare-FS and absorbed-Cast variants move the same
            // bytes but double the search genome and produce fresh kernels.
            // Cleanup runs after growth, so the deleted variant is not rederived.
            rule(
                (
                    eq(cast, Ir::Op(OpKind::Cast(size, dt_out), [x])),
                    eq(fs_c, Ir::Op(fusion_start(shape, s, dt_out), [cast])),
                    eq(
                        fs_c,
                        Ir::Op(
                            cuda_unary_elementwise("Cast", shape, is, os, dt_out),
                            [fs_x],
                        ),
                    ),
                ),
                (delete(Ir::Op(fusion_start(shape, s, dt_out), [cast])),),
            )
            .label("cleanup-FS-with-absorbed-cast"),
        ];
        for opcode in ["Sin", "Sqrt", "Exp2", "Log2", "Recip"] {
            rules.push(
                rule(
                    (
                        eq(inner_fe, Ir::Op(fusion_end(shape, s, dt), [inner])),
                        eq(bad_fs, Ir::Op(fusion_start(shape, s, dt), [inner_fe])),
                        eq(
                            bad_elem,
                            Ir::Op(cuda_unary_elementwise(opcode, shape, s, s, dt), [bad_fs]),
                        ),
                        eq(bad_fe, Ir::Op(fusion_end(shape, s, dt), [bad_elem])),
                        eq(
                            good_elem,
                            Ir::Op(cuda_unary_elementwise(opcode, shape, s, s, dt), [inner]),
                        ),
                        eq(good_fe, Ir::Op(fusion_end(shape, s, dt), [good_elem])),
                        eq(bad_fe, good_fe),
                    ),
                    (delete(Ir::Op(fusion_start(shape, s, dt), [inner_fe])),),
                )
                .label(format!("cleanup-nested-FS-FE-unary-{opcode}")),
            );
        }
        for opcode in ["Add", "Mul"] {
            rules.push(
                rule(
                    (
                        eq(inner_fe, Ir::Op(fusion_end(shape, a_s, dt), [inner_a])),
                        eq(bad_fs, Ir::Op(fusion_start(shape, a_s, dt), [inner_fe])),
                        eq(fs_b, Ir::Op(fusion_start(shape, b_s, dt), [b])),
                        eq(
                            bad_elem,
                            Ir::Op(
                                cuda_binary_elementwise(opcode, shape, a_s, b_s, out_s, dt),
                                [bad_fs, fs_b],
                            ),
                        ),
                        eq(bad_fe, Ir::Op(fusion_end(shape, out_s, dt), [bad_elem])),
                        eq(
                            good_elem,
                            Ir::Op(
                                cuda_binary_elementwise(opcode, shape, a_s, b_s, out_s, dt),
                                [inner_a, fs_b],
                            ),
                        ),
                        eq(good_fe, Ir::Op(fusion_end(shape, out_s, dt), [good_elem])),
                        eq(bad_fe, good_fe),
                    ),
                    (delete(Ir::Op(fusion_start(shape, a_s, dt), [inner_fe])),),
                )
                .label(format!("cleanup-nested-FS-FE-binary-lhs-{opcode}")),
            );
            rules.push(
                rule(
                    (
                        eq(inner_fe, Ir::Op(fusion_end(shape, b_s, dt), [inner_b])),
                        eq(bad_fs, Ir::Op(fusion_start(shape, b_s, dt), [inner_fe])),
                        eq(fs_a, Ir::Op(fusion_start(shape, a_s, dt), [a])),
                        eq(
                            bad_elem,
                            Ir::Op(
                                cuda_binary_elementwise(opcode, shape, a_s, b_s, out_s, dt),
                                [fs_a, bad_fs],
                            ),
                        ),
                        eq(bad_fe, Ir::Op(fusion_end(shape, out_s, dt), [bad_elem])),
                        eq(
                            good_elem,
                            Ir::Op(
                                cuda_binary_elementwise(opcode, shape, a_s, b_s, out_s, dt),
                                [fs_a, inner_b],
                            ),
                        ),
                        eq(good_fe, Ir::Op(fusion_end(shape, out_s, dt), [good_elem])),
                        eq(bad_fe, good_fe),
                    ),
                    (delete(Ir::Op(fusion_start(shape, b_s, dt), [inner_fe])),),
                )
                .label(format!("cleanup-nested-FS-FE-binary-rhs-{opcode}")),
            );
        }
        rules.extend([
            rule(
                (
                    Ir::Op(fusion_end(fe_shape, fe_s, dt), [inner]),
                    eq(
                        inner,
                        Ir::Op(
                            cuda_unary_elementwise(op, inner_shape, inner_in_s, inner_s, dt),
                            inner_inputs,
                        ),
                    ),
                    ne(fe_shape, inner_shape),
                ),
                (delete(Ir::Op(fusion_end(fe_shape, fe_s, dt), [inner])),),
            )
            .label("delete-malformed-FE-unary-shape"),
            rule(
                (
                    Ir::Op(fusion_end(fe_shape, fe_s, dt), [inner]),
                    eq(
                        inner,
                        Ir::Op(
                            cuda_unary_elementwise(op, inner_shape, inner_in_s, inner_s, dt),
                            inner_inputs,
                        ),
                    ),
                    ne(fe_s, inner_s),
                ),
                (delete(Ir::Op(fusion_end(fe_shape, fe_s, dt), [inner])),),
            )
            .label("delete-malformed-FE-unary-strides"),
            rule(
                (
                    Ir::Op(fusion_end(fe_shape, fe_s, dt), [inner]),
                    eq(
                        inner,
                        Ir::Op(
                            CudaBinaryElementwiseArgs {
                                shape: inner_shape.into(),
                                dtype: dt.into(),
                                ..CudaBinaryElementwiseArgs::fresh()
                            },
                            inner_inputs,
                        ),
                    ),
                    ne(fe_shape, inner_shape),
                ),
                (delete(Ir::Op(fusion_end(fe_shape, fe_s, dt), [inner])),),
            )
            .label("delete-malformed-FE-binary-shape"),
            rule(
                (
                    Ir::Op(fusion_end(fe_shape, fe_s, dt), [inner]),
                    eq(
                        inner,
                        Ir::Op(
                            CudaBinaryElementwiseArgs {
                                out_strides: inner_s.into(),
                                dtype: dt.into(),
                                ..CudaBinaryElementwiseArgs::fresh()
                            },
                            inner_inputs,
                        ),
                    ),
                    ne(fe_s, inner_s),
                ),
                (delete(Ir::Op(fusion_end(fe_shape, fe_s, dt), [inner])),),
            )
            .label("delete-malformed-FE-binary-strides"),
            rule(
                (
                    Ir::Op(fusion_end(fe_shape, fe_s, dt), [inner]),
                    eq(
                        inner,
                        Ir::Op(fusion_end(inner_shape, inner_s, dt), inner_inputs),
                    ),
                    ne(fe_shape, inner_shape),
                ),
                (delete(Ir::Op(fusion_end(fe_shape, fe_s, dt), [inner])),),
            )
            .label("delete-malformed-FE-nested-shape"),
            rule(
                (
                    Ir::Op(fusion_end(fe_shape, fe_s, dt), [inner]),
                    eq(
                        inner,
                        Ir::Op(fusion_end(inner_shape, inner_s, dt), inner_inputs),
                    ),
                    ne(fe_s, inner_s),
                ),
                (delete(Ir::Op(fusion_end(fe_shape, fe_s, dt), [inner])),),
            )
            .label("delete-malformed-FE-nested-strides"),
        ]);
        rules
    }

    #[ruleset]
    pub(crate) fn fusion_merge(
        shape: &EList,
        a_s: &EList,
        dt: &DType,
        inner_a: &Ir,
        b_s: &EList,
        inner_b: &Ir,
        out_s: &EList,
        bin: &Ir,
    ) -> Vec<Rule> {
        [
            ("Add", OpKind::Add(shape, a_s, b_s, out_s)),
            ("Mul", OpKind::Mul(shape, a_s, b_s, out_s)),
        ]
        .into_iter()
        .map(|(opcode, hlir)| {
            let element = &Ir::Op(
                cuda_binary_elementwise(opcode, shape, a_s, b_s, out_s, dt),
                [inner_a, inner_b],
            );
            let fused = &Ir::Op(fusion_end(shape, out_s, dt), [element]);
            let fe_a = &Ir::Op(fusion_end(shape, a_s, dt), [inner_a]);
            let fe_b = &Ir::Op(fusion_end(shape, b_s, dt), [inner_b]);
            rule(
                (eq(bin, Ir::Op(hlir, [fe_a, fe_b])),),
                (element, fused, union(bin, fused), set(fused.dtype(), dt)),
            )
            .label(format!("merge-FE-FE-{opcode}"))
        })
        .collect()
    }
}

impl FusionEnd {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_shape, field_strides, field_dtype)) = typed::get_args(
            kind,
            |shape: &EList, strides: &EList, dtype: &luminal::typed_reference::schema::DType| {
                crate::typed_rules::fusion_end(shape, strides, dtype)
            },
        )?
        else {
            return Ok(None);
        };
        let decoded_shape = context.expressions(&field_shape)?;
        let decoded_strides = context.expressions(&field_strides)?;
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            shape: decoded_shape,
            strides: decoded_strides,
            dtype: context.dtype(&field_dtype)?,
        }))))
    }
}

impl KernelOp for FusionEnd {
    fn compile(
        &self,
        _stream: &Arc<CudaStream>,
        _compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> CompileOut {
        unreachable!("FusionEnd must be compiled through fusion region codegen")
    }
    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }
    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }
    fn output_dtype(&self) -> DType {
        self.dtype
    }
    fn kernel_name(&self) -> &'static str {
        "FusionEnd"
    }
}
