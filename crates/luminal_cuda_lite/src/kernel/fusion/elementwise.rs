use egglog_experimental::typed::{self as typed, TypedError, builtins as egg};
use luminal::egglog_utils::DecodeContext;
use luminal::typed_reference::schema::{DimExpr, EList, IList, Ir, OpKind};
// =========================================================================
// Generic CUDA elementwise ops used inside FusionStart/FusionEnd regions.
//
// CUDA elementwise execution is represented as a FusionEnd-rooted region even
// for a single op. These ops are therefore region-internal only; standalone
// compilation is intentionally unsupported.
// =========================================================================

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{op::*, prelude::*};

use crate::kernel::KernelOp;

#[cfg(test)]
mod decoder_tests {
    use super::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::DType as IrDType;

    #[test]
    fn unknown_opcodes_are_rejected_before_codegen() -> Result<(), TypedError> {
        let empty = EList::ENil();
        let cases: [(OpKind, DecodeFn, usize); 2] = [
            (
                crate::typed_rules::cuda_unary_elementwise(
                    "unknown",
                    &empty,
                    &empty,
                    &empty,
                    IrDType::F32(),
                ),
                CudaUnaryElementwise::decode,
                1,
            ),
            (
                crate::typed_rules::cuda_binary_elementwise(
                    "unknown",
                    &empty,
                    &empty,
                    &empty,
                    &empty,
                    IrDType::F32(),
                ),
                CudaBinaryElementwise::decode,
                2,
            ),
        ];
        for (kind, decoder, arity) in cases {
            let mut graph = EGraph::new(EGraphOptions::default());
            let capture = let_("kind", kind);
            graph.register(&capture)?;
            let frozen = graph.freeze()?;
            let observed = frozen.lookup(&capture)?;
            let selected = frozen.nodes(&observed)?.into_iter().next().unwrap();
            let choices = Default::default();
            let mut context = DecodeContext::new(&frozen, &choices);
            let inputs = vec![Ir::Input(0, "input", IrDType::F32()); arity];
            let error = decoder(&selected, &mut inputs.clone(), &mut context).unwrap_err();
            assert!(matches!(error, TypedError::Decode(message) if message.contains("opcode")));
        }
        Ok(())
    }
}

type CompileOut = (
    CudaFunction,
    Arc<CudaModule>,
    String,
    (Expression, Expression, Expression),
    (Expression, Expression, Expression),
    Expression,
    FxHashMap<char, CudaSlice<u8>>,
);

#[derive(Default, Debug, Clone)]
pub struct CudaUnaryElementwise {
    pub(crate) op: String,
    pub(crate) shape: Vec<Expression>,
    pub(crate) in_strides: Vec<Expression>,
    pub(crate) out_strides: Vec<Expression>,
    pub(crate) dtype: DType,
}

pub(crate) mod cuda_unary_elementwise {
    use crate::typed_rules::*;
    use egglog_experimental::typed::builtins as egg;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[sort]
    pub(crate) struct CudaSigmoidScaledState;

    #[constructor]
    pub(crate) fn mk_cuda_sigmoid_scaled_state(
        a0: Ir,
        a1: EList,
        a2: EList,
        a3: DType,
    ) -> CudaSigmoidScaledState;

    #[function(merge = |_old: CudaSigmoidScaledState, new: CudaSigmoidScaledState| new)]
    pub(crate) fn cuda_sigmoid_scaled(a0: Ir) -> CudaSigmoidScaledState;

    #[ruleset]
    pub(crate) fn kernel_lower(
        shape: &EList,
        s: &EList,
        out_s: &EList,
        x: &Ir,
        u: &Ir,
        dt: &DType,
        x_stride: &EList,
        sqrt_stride: &EList,
        out_stride: &EList,
        recip: &Ir,
    ) -> Vec<Rule> {
        let fs = &Ir::Op(fusion_start(shape, s, dt), [x]);
        let sin = &Ir::Op(cuda_unary_elementwise("Sin", shape, s, out_s, dt), [fs]);
        let fused_sin = &Ir::Op(fusion_end(shape, out_s, dt), [sin]);
        let sqrt_op = &Ir::Op(cuda_unary_elementwise("Sqrt", shape, s, out_s, dt), [fs]);
        let fused_sqrt = &Ir::Op(fusion_end(shape, out_s, dt), [sqrt_op]);
        let exp2 = &Ir::Op(cuda_unary_elementwise("Exp2", shape, s, out_s, dt), [fs]);
        let fused_exp2 = &Ir::Op(fusion_end(shape, out_s, dt), [exp2]);
        let log2 = &Ir::Op(cuda_unary_elementwise("Log2", shape, s, out_s, dt), [fs]);
        let fused_log2 = &Ir::Op(fusion_end(shape, out_s, dt), [log2]);
        let recip_op = &Ir::Op(cuda_unary_elementwise("Recip", shape, s, out_s, dt), [fs]);
        let fused_recip = &Ir::Op(fusion_end(shape, out_s, dt), [recip_op]);
        let rsqrt_start = &Ir::Op(fusion_start(shape, x_stride, dt), [x]);
        let rsqrt = &Ir::Op(
            cuda_unary_elementwise("Rsqrt", shape, x_stride, out_stride, dt),
            [rsqrt_start],
        );
        let fused_rsqrt = &Ir::Op(fusion_end(shape, out_stride, dt), [rsqrt]);
        let sqrt = &Ir::Op(OpKind::Sqrt(shape, x_stride, sqrt_stride), [x]);
        vec![
            rule(
                (
                    eq(u, Ir::Op(OpKind::Sin(shape, s, out_s), [x])),
                    eq(dt, u.dtype()),
                ),
                (
                    fs,
                    sin,
                    fused_sin,
                    union(u, fused_sin),
                    set(fused_sin.dtype(), dt),
                ),
            )
            .label("cuda-elem-singleton-Sin"),
            rule(
                (
                    eq(u, Ir::Op(OpKind::Sqrt(shape, s, out_s), [x])),
                    eq(dt, u.dtype()),
                ),
                (
                    fs,
                    sqrt_op,
                    fused_sqrt,
                    union(u, fused_sqrt),
                    set(fused_sqrt.dtype(), dt),
                ),
            )
            .label("cuda-elem-singleton-Sqrt"),
            rule(
                (
                    eq(u, Ir::Op(OpKind::Exp2(shape, s, out_s), [x])),
                    eq(dt, u.dtype()),
                ),
                (
                    fs,
                    exp2,
                    fused_exp2,
                    union(u, fused_exp2),
                    set(fused_exp2.dtype(), dt),
                ),
            )
            .label("cuda-elem-singleton-Exp2"),
            rule(
                (
                    eq(u, Ir::Op(OpKind::Log2(shape, s, out_s), [x])),
                    eq(dt, u.dtype()),
                ),
                (
                    fs,
                    log2,
                    fused_log2,
                    union(u, fused_log2),
                    set(fused_log2.dtype(), dt),
                ),
            )
            .label("cuda-elem-singleton-Log2"),
            rule(
                (
                    eq(u, Ir::Op(OpKind::Recip(shape, s, out_s), [x])),
                    eq(dt, u.dtype()),
                ),
                (
                    fs,
                    recip_op,
                    fused_recip,
                    union(u, fused_recip),
                    set(fused_recip.dtype(), dt),
                ),
            )
            .label("cuda-elem-singleton-Recip"),
            rule(
                (
                    eq(
                        recip,
                        Ir::Op(OpKind::Recip(shape, sqrt_stride, out_stride), [sqrt]),
                    ),
                    eq(dt, recip.dtype()),
                ),
                (
                    rsqrt_start,
                    rsqrt,
                    fused_rsqrt,
                    union(recip, fused_rsqrt),
                    set(fused_rsqrt.dtype(), dt),
                ),
            )
            .label("cuda-elem-rsqrt-from-sqrt-recip"),
        ]
    }

    #[ruleset]
    pub(crate) fn direct_kernel(
        shape: &EList,
        x_stride: &EList,
        const_stride: &EList,
        inter_stride: &EList,
        x: &Ir,
        exp_const: &Ir,
        out_stride: &EList,
        exp2: &Ir,
        dt: &DType,
        val: &egg::F64,
        nv: &egg::F64,
        neg_stride: &EList,
        neg_out_stride: &EList,
        lv: &egg::F64,
        scaled_stride: &EList,
        scaled: &Ir,
        scaled_state: &CudaSigmoidScaledState,
        exp_stride: &EList,
        ov: &egg::F64,
        one_stride: &EList,
        add_stride: &EList,
        sig_out: &Ir,
    ) -> Vec<Rule> {
        let fs = &Ir::Op(fusion_start(shape, x_stride, dt), [x]);
        let exp = &Ir::Op(
            cuda_unary_elementwise("Exp", shape, x_stride, out_stride, dt),
            [fs],
        );
        let fused_exp = &Ir::Op(fusion_end(shape, out_stride, dt), [exp]);
        let sigmoid = &Ir::Op(
            cuda_unary_elementwise("Sigmoid", shape, x_stride, out_stride, dt),
            [fs],
        );
        let fused_sigmoid = &Ir::Op(fusion_end(shape, out_stride, dt), [sigmoid]);
        let mul = &Ir::Op(
            OpKind::Mul(shape, x_stride, const_stride, inter_stride),
            [x, exp_const],
        );
        let cv = &Ir::Op(OpKind::Constant(val), IList::INil());
        let neg1 = &Ir::Op(OpKind::Constant(nv), IList::INil());
        let log2e = &Ir::Op(OpKind::Constant(lv), IList::INil());
        let one = &Ir::Op(OpKind::Constant(ov), IList::INil());
        let neg_x = &Ir::Op(
            OpKind::Mul(shape, x_stride, neg_stride, neg_out_stride),
            [x, neg1],
        );
        let plus_one = &Ir::Op(
            OpKind::Add(shape, exp_stride, one_stride, add_stride),
            [exp2, one],
        );
        vec![
            rule(
                (
                    eq(
                        exp2,
                        Ir::Op(OpKind::Exp2(shape, inter_stride, out_stride), [mul]),
                    ),
                    eq(dt, x.dtype()),
                    eq(exp_const, cv),
                    val.gt(1.44),
                    val.lt(1.45),
                ),
                (
                    fs,
                    exp,
                    fused_exp,
                    union(exp2, fused_exp),
                    set(fused_exp.dtype(), dt),
                ),
            )
            .label("direct-exp-region"),
            rule(
                (
                    nv.lt(-0.99),
                    nv.gt(-1.01),
                    lv.gt(1.44),
                    lv.lt(1.45),
                    eq(
                        scaled,
                        Ir::Op(
                            MulArgs {
                                shape: shape.into(),
                                a_strides: neg_out_stride.into(),
                                ..MulArgs::fresh()
                            },
                            [neg_x, log2e],
                        ),
                    ),
                    eq(dt, x.dtype()),
                ),
                (set(
                    cuda_sigmoid_scaled(scaled),
                    mk_cuda_sigmoid_scaled_state(x, shape, x_stride, dt),
                ),),
            )
            .label("direct-sigmoid-scaled-region-marker"),
            rule(
                (
                    eq(scaled_state, cuda_sigmoid_scaled(scaled)),
                    eq(
                        scaled_state,
                        mk_cuda_sigmoid_scaled_state(x, shape, x_stride, dt),
                    ),
                    eq(
                        exp2,
                        Ir::Op(OpKind::Exp2(shape, scaled_stride, exp_stride), [scaled]),
                    ),
                    ov.gt(0.99),
                    ov.lt(1.01),
                    eq(
                        sig_out,
                        Ir::Op(OpKind::Recip(shape, add_stride, out_stride), [plus_one]),
                    ),
                ),
                (
                    fs,
                    sigmoid,
                    fused_sigmoid,
                    union(sig_out, fused_sigmoid),
                    set(fused_sigmoid.dtype(), dt),
                ),
            )
            .label("direct-sigmoid-region"),
        ]
    }
}

impl CudaUnaryElementwise {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_op, field_shape, field_strides, field_out_strides, field_dtype)) =
            typed::get_args(
                kind,
                |op: &egg::String,
                 shape: &EList,
                 strides: &EList,
                 out_strides: &EList,
                 dtype: &luminal::typed_reference::schema::DType| {
                    crate::typed_rules::cuda_unary_elementwise(
                        op,
                        shape,
                        strides,
                        out_strides,
                        dtype,
                    )
                },
            )?
        else {
            return Ok(None);
        };
        let decoded_shape = context.expressions(&field_shape)?;
        let decoded_strides = context.expressions(&field_strides)?;
        let decoded_out_strides = context.expressions(&field_out_strides)?;
        let op = String::try_from(&field_op)?;
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            op,
            shape: decoded_shape,
            in_strides: decoded_strides,
            out_strides: decoded_out_strides,
            dtype: context.dtype(&field_dtype)?,
        }))))
    }
}

impl KernelOp for CudaUnaryElementwise {
    fn compile(
        &self,
        _stream: &Arc<CudaStream>,
        _compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> CompileOut {
        unreachable!("CudaUnaryElementwise must be compiled through fusion region codegen")
    }

    fn output_size(&self) -> Expression {
        self.shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_bytes()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.output_size()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "CudaUnaryElementwise"
    }
}

#[derive(Default, Debug, Clone)]
pub struct CudaBinaryElementwise {
    pub(crate) op: String,
    pub(crate) out_shape: Vec<Expression>,
    pub(crate) a_stride: Vec<Expression>,
    pub(crate) b_stride: Vec<Expression>,
    pub(crate) out_stride: Vec<Expression>,
    pub(crate) dtype: DType,
}

pub(crate) mod cuda_binary_elementwise {
    use crate::typed_rules::*;
    use egglog_experimental::typed::builtins as egg;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(
        shape: &EList,
        a_s: &EList,
        b_s: &EList,
        out_s: &EList,
        a: &Ir,
        b: &Ir,
        bin: &Ir,
        dt: &DType,
    ) -> Vec<Rule> {
        let fs_a = &Ir::Op(fusion_start(shape, a_s, dt), [a]);
        let fs_b = &Ir::Op(fusion_start(shape, b_s, dt), [b]);
        let add_element = &Ir::Op(
            cuda_binary_elementwise("Add", shape, a_s, b_s, out_s, dt),
            [fs_a, fs_b],
        );
        let fused_add = &Ir::Op(fusion_end(shape, out_s, dt), [add_element]);
        let mul_element = &Ir::Op(
            cuda_binary_elementwise("Mul", shape, a_s, b_s, out_s, dt),
            [fs_a, fs_b],
        );
        let fused_mul = &Ir::Op(fusion_end(shape, out_s, dt), [mul_element]);
        vec![
            rule(
                (
                    eq(bin, Ir::Op(OpKind::Add(shape, a_s, b_s, out_s), [a, b])),
                    eq(dt, bin.dtype()),
                ),
                (
                    fs_a,
                    fs_b,
                    add_element,
                    fused_add,
                    union(bin, fused_add),
                    set(fused_add.dtype(), dt),
                ),
            )
            .label("cuda-elem-singleton-Add"),
            rule(
                (
                    eq(bin, Ir::Op(OpKind::Mul(shape, a_s, b_s, out_s), [a, b])),
                    eq(dt, a.dtype()),
                ),
                (
                    fs_a,
                    fs_b,
                    mul_element,
                    fused_mul,
                    union(bin, fused_mul),
                    set(fused_mul.dtype(), dt),
                ),
            )
            .label("cuda-elem-singleton-Mul"),
        ]
    }
}

impl CudaBinaryElementwise {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((
            field_op,
            field_shape,
            field_a_strides,
            field_b_strides,
            field_out_strides,
            field_dtype,
        )) = typed::get_args(
            kind,
            |op: &egg::String,
             shape: &EList,
             a_strides: &EList,
             b_strides: &EList,
             out_strides: &EList,
             dtype: &luminal::typed_reference::schema::DType| {
                crate::typed_rules::cuda_binary_elementwise(
                    op,
                    shape,
                    a_strides,
                    b_strides,
                    out_strides,
                    dtype,
                )
            },
        )?
        else {
            return Ok(None);
        };
        let decoded_shape = context.expressions(&field_shape)?;
        let decoded_a_strides = context.expressions(&field_a_strides)?;
        let decoded_b_strides = context.expressions(&field_b_strides)?;
        let decoded_out_strides = context.expressions(&field_out_strides)?;
        let op = String::try_from(&field_op)?;
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            op,
            out_shape: decoded_shape,
            a_stride: decoded_a_strides,
            b_stride: decoded_b_strides,
            out_stride: decoded_out_strides,
            dtype: context.dtype(&field_dtype)?,
        }))))
    }
}

impl KernelOp for CudaBinaryElementwise {
    fn compile(
        &self,
        _stream: &Arc<CudaStream>,
        _compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> CompileOut {
        unreachable!("CudaBinaryElementwise must be compiled through fusion region codegen")
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_bytes() * 2
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.output_size()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "CudaBinaryElementwise"
    }
}
