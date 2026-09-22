//! CUDA conv2d-with-bias backend rewrite.
//!
//! `KernelConv2D` is selected by egglog from pure HLIR conv graphs and lowers
//! to a one-thread-per-output CUDA kernel. It avoids materializing unfold/im2col
//! intermediates while keeping model code free of custom ops.

use egglog_experimental::typed::{self as typed, TypedError, builtins as egg};
use luminal::egglog_utils::DecodeContext;
use luminal::typed_reference::schema::{DimExpr, EList, IList, Ir, OpKind};
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::prelude::FxHashMap;
use luminal::{
    dtype::DType,
    op::LLIROp,
    prelude::FxHashSet,
    shape::{Expression, flatten_strides},
};

use crate::compile_module_image_for_current_device;
use crate::kernel::{KernelOp, hlir::generate_dyn_dims_defines};

#[derive(Default, Debug, Clone)]
pub struct KernelConv2D {
    out_shape: Vec<Expression>,
    input_shape: Vec<Expression>,
    input_stride: Vec<Expression>,
    weight_co_stride: Expression,
    weight_inner_stride: Expression,
    bias_c_stride: Expression,
    out_stride: Vec<Expression>,
    kernel_h: Expression,
    kernel_w: Expression,
    stride_h: Expression,
    stride_w: Expression,
    dilation_h: Expression,
    dilation_w: Expression,
    pad_h: Expression,
    pad_w: Expression,
    dtype: DType,
}

pub(crate) mod kernel_conv2_d {
    use crate::typed_rules::*;
    use egglog_experimental::typed::builtins as egg;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    // 1x1 convolutions use the same Mul -> KernelSum -> Add skeleton without
    // unfolding; FusionStart reads the original input. General convolutions
    // consume the pre-unfold (possibly already padded) input, avoiding the
    // im2col window and product tensors. These rules target stride=1,
    // dilation=1; padding is represented by the input tensor itself.
    //
    // CUDA normalization does not reliably prove M=H_out*W_out, K=C_in*KH*KW,
    // or equality of separately derived strides. Keep the stable six-axis
    // window, [M,C_out,K] product, K reduction, output view and channel bias
    // anchors until expression/list canonicalization can prove more.

    #[ruleset]
    pub(crate) fn kernel_lower(
        out_shape: &EList,
        out_stride: &EList,
        add_elem: &Ir,
        out: &Ir,
        sum_add_stride: &EList,
        bias_add_stride: &EList,
        sum: &Ir,
        bias: &Ir,
        matmul_out_shape: &EList,
        c_in: &DimExpr,
        sum_in_stride: &EList,
        k_stride: &DimExpr,
        sum_out_stride: &EList,
        mul_shape: &EList,
        mul_out_stride: &EList,
        mul_elem: &Ir,
        input_1x1_stride: &EList,
        weight_stride: &EList,
        input: &Ir,
        weight: &Ir,
        c_out: &DimExpr,
        h_out: &DimExpr,
        w_out: &DimExpr,
        m: &DimExpr,
        flat_stride: &DimExpr,
        input_c_stride: &DimExpr,
        sum_m_stride: &DimExpr,
        sum_c_stride: &DimExpr,
        sum_out_m_stride: &DimExpr,
        sum_out_c_stride: &DimExpr,
        sum_add_c_stride: &DimExpr,
        sum_add_h_stride: &DimExpr,
        sum_add_w_stride: &DimExpr,
        weight_co_stride: &DimExpr,
        weight_inner_stride: &DimExpr,
        bias_c_stride: &DimExpr,
        k_dim: &DimExpr,
        patch_stride: &EList,
        idx_shape: &EList,
        idx_stride: &EList,
        input_shape: &EList,
        input_stride: &EList,
        gather_out_stride: &EList,
        indices: &Ir,
        h_in: &DimExpr,
        w_in: &DimExpr,
        kernel_h: &DimExpr,
        kernel_w: &DimExpr,
    ) -> Vec<Rule> {
        let pointwise_conv = &Ir::Op(
            kernel_conv2_d(
                out_shape,
                [c_in, h_out, w_out],
                [input_c_stride, &(w_out * flat_stride), flat_stride],
                weight_co_stride,
                weight_inner_stride,
                bias_c_stride,
                out_stride,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                DType::F32(),
            ),
            [input, weight, bias],
        );
        let source = &Ir::Op(fusion_end(out_shape, out_stride, DType::F32()), [add_elem]);
        let unfold_conv = &Ir::Op(
            kernel_conv2_d(
                out_shape,
                input_shape,
                input_stride,
                weight_co_stride,
                weight_inner_stride,
                bias_c_stride,
                out_stride,
                kernel_h,
                kernel_w,
                1,
                1,
                1,
                1,
                0,
                0,
                DType::F32(),
            ),
            [input, weight, bias],
        );
        let sum_fs = &Ir::Op(fusion_start(out_shape, sum_add_stride, DType::F32()), [sum]);
        let bias_fs = &Ir::Op(
            fusion_start(out_shape, bias_add_stride, DType::F32()),
            [bias],
        );
        let mul_fe = &Ir::Op(
            fusion_end(mul_shape, mul_out_stride, DType::F32()),
            [mul_elem],
        );
        let input_fs = &Ir::Op(
            fusion_start(mul_shape, input_1x1_stride, DType::F32()),
            [input],
        );
        let weight_fs = &Ir::Op(
            fusion_start(mul_shape, weight_stride, DType::F32()),
            [weight],
        );
        let patches = &Ir::Op(
            kernel_gather(
                idx_shape,
                idx_stride,
                input_shape,
                input_stride,
                gather_out_stride,
                DType::F32(),
            ),
            [indices, input],
        );
        let patch_fs = &Ir::Op(
            fusion_start(mul_shape, patch_stride, DType::F32()),
            [patches],
        );
        vec![
            rule(
                (
                    eq(out, source),
                    eq(
                        add_elem,
                        Ir::Op(
                            cuda_binary_elementwise(
                                "Add",
                                out_shape,
                                sum_add_stride,
                                bias_add_stride,
                                out_stride,
                                DType::F32(),
                            ),
                            [sum_fs, bias_fs],
                        ),
                    ),
                    eq(
                        sum,
                        Ir::Op(
                            kernel_sum(
                                matmul_out_shape,
                                c_in,
                                sum_in_stride,
                                k_stride,
                                sum_out_stride,
                                DType::F32(),
                            ),
                            [mul_fe],
                        ),
                    ),
                    eq(
                        mul_elem,
                        Ir::Op(
                            cuda_binary_elementwise(
                                "Mul",
                                mul_shape,
                                input_1x1_stride,
                                weight_stride,
                                mul_out_stride,
                                DType::F32(),
                            ),
                            [input_fs, weight_fs],
                        ),
                    ),
                    eq(out_shape, [c_out, h_out, w_out]),
                    eq(matmul_out_shape, [m, c_out]),
                    eq(mul_shape, [m, c_out, c_in]),
                    eq(
                        input_1x1_stride,
                        [flat_stride, &(DimExpr::MNum(0)), input_c_stride],
                    ),
                    eq(flat_stride, DimExpr::MIter()),
                    eq(k_stride, DimExpr::MIter()),
                    eq(sum_in_stride, [sum_m_stride, sum_c_stride]),
                    eq(sum_out_stride, [sum_out_m_stride, sum_out_c_stride]),
                    eq(
                        sum_add_stride,
                        [sum_add_c_stride, sum_add_h_stride, sum_add_w_stride],
                    ),
                    eq(weight_co_stride, weight_stride.nth_from_end(1_i64)),
                    eq(weight_inner_stride, weight_stride.nth_from_end(0_i64)),
                    eq(DimExpr::MNum(0), weight_stride.nth_from_end(2_i64)),
                    eq(
                        bias_add_stride,
                        [bias_c_stride, &(DimExpr::MNum(0)), &(DimExpr::MNum(0))],
                    ),
                ),
                (
                    pointwise_conv,
                    union(out, pointwise_conv),
                    subsume(source),
                    set(pointwise_conv.dtype(), DType::F32()),
                ),
            )
            .label("kernel conv2d 1x1 from cuda lowered matmul bias"),
            rule(
                (
                    eq(out, source),
                    eq(
                        add_elem,
                        Ir::Op(
                            cuda_binary_elementwise(
                                "Add",
                                out_shape,
                                bias_add_stride,
                                sum_add_stride,
                                out_stride,
                                DType::F32(),
                            ),
                            [bias_fs, sum_fs],
                        ),
                    ),
                    eq(
                        sum,
                        Ir::Op(
                            kernel_sum(
                                matmul_out_shape,
                                c_in,
                                sum_in_stride,
                                k_stride,
                                sum_out_stride,
                                DType::F32(),
                            ),
                            [mul_fe],
                        ),
                    ),
                    eq(
                        mul_elem,
                        Ir::Op(
                            cuda_binary_elementwise(
                                "Mul",
                                mul_shape,
                                input_1x1_stride,
                                weight_stride,
                                mul_out_stride,
                                DType::F32(),
                            ),
                            [input_fs, weight_fs],
                        ),
                    ),
                    eq(out_shape, [c_out, h_out, w_out]),
                    eq(matmul_out_shape, [m, c_out]),
                    eq(mul_shape, [m, c_out, c_in]),
                    eq(
                        input_1x1_stride,
                        [flat_stride, &(DimExpr::MNum(0)), input_c_stride],
                    ),
                    eq(flat_stride, DimExpr::MIter()),
                    eq(k_stride, DimExpr::MIter()),
                    eq(sum_in_stride, [sum_m_stride, sum_c_stride]),
                    eq(sum_out_stride, [sum_out_m_stride, sum_out_c_stride]),
                    eq(
                        sum_add_stride,
                        [sum_add_c_stride, sum_add_h_stride, sum_add_w_stride],
                    ),
                    eq(weight_co_stride, weight_stride.nth_from_end(1_i64)),
                    eq(weight_inner_stride, weight_stride.nth_from_end(0_i64)),
                    eq(DimExpr::MNum(0), weight_stride.nth_from_end(2_i64)),
                    eq(
                        bias_add_stride,
                        [bias_c_stride, &(DimExpr::MNum(0)), &(DimExpr::MNum(0))],
                    ),
                ),
                (
                    pointwise_conv,
                    union(out, pointwise_conv),
                    subsume(source),
                    set(pointwise_conv.dtype(), DType::F32()),
                ),
            )
            .label("kernel conv2d 1x1 from cuda lowered bias matmul"),
            rule(
                (
                    eq(out, source),
                    eq(
                        add_elem,
                        Ir::Op(
                            cuda_binary_elementwise(
                                "Add",
                                out_shape,
                                sum_add_stride,
                                bias_add_stride,
                                out_stride,
                                DType::F32(),
                            ),
                            [sum_fs, bias_fs],
                        ),
                    ),
                    eq(
                        sum,
                        Ir::Op(
                            kernel_sum(
                                matmul_out_shape,
                                k_dim,
                                sum_in_stride,
                                k_stride,
                                sum_out_stride,
                                DType::F32(),
                            ),
                            [mul_fe],
                        ),
                    ),
                    eq(
                        mul_elem,
                        Ir::Op(
                            cuda_binary_elementwise(
                                "Mul",
                                mul_shape,
                                patch_stride,
                                weight_stride,
                                mul_out_stride,
                                DType::F32(),
                            ),
                            [patch_fs, weight_fs],
                        ),
                    ),
                    eq(out_shape, [c_out, h_out, w_out]),
                    eq(input_shape, [c_in, h_in, w_in]),
                    eq(
                        idx_shape,
                        [c_in, h_out, w_out, &(DimExpr::MNum(1)), kernel_h, kernel_w],
                    ),
                    eq(matmul_out_shape, [m, c_out]),
                    eq(mul_shape, [m, c_out, k_dim]),
                    eq(k_stride, DimExpr::MIter()),
                    eq(sum_in_stride, [sum_m_stride, sum_c_stride]),
                    eq(sum_out_stride, [sum_out_m_stride, sum_out_c_stride]),
                    eq(
                        sum_add_stride,
                        [sum_add_c_stride, sum_add_h_stride, sum_add_w_stride],
                    ),
                    eq(weight_co_stride, weight_stride.nth_from_end(1_i64)),
                    eq(weight_inner_stride, weight_stride.nth_from_end(0_i64)),
                    eq(DimExpr::MNum(0), weight_stride.nth_from_end(2_i64)),
                    eq(
                        bias_add_stride,
                        [bias_c_stride, &(DimExpr::MNum(0)), &(DimExpr::MNum(0))],
                    ),
                ),
                (
                    unfold_conv,
                    union(out, unfold_conv),
                    subsume(source),
                    set(unfold_conv.dtype(), DType::F32()),
                ),
            )
            .label("kernel conv2d from cuda lowered unfold matmul bias"),
            rule(
                (
                    eq(out, source),
                    eq(
                        add_elem,
                        Ir::Op(
                            cuda_binary_elementwise(
                                "Add",
                                out_shape,
                                bias_add_stride,
                                sum_add_stride,
                                out_stride,
                                DType::F32(),
                            ),
                            [bias_fs, sum_fs],
                        ),
                    ),
                    eq(
                        sum,
                        Ir::Op(
                            kernel_sum(
                                matmul_out_shape,
                                k_dim,
                                sum_in_stride,
                                k_stride,
                                sum_out_stride,
                                DType::F32(),
                            ),
                            [mul_fe],
                        ),
                    ),
                    eq(
                        mul_elem,
                        Ir::Op(
                            cuda_binary_elementwise(
                                "Mul",
                                mul_shape,
                                patch_stride,
                                weight_stride,
                                mul_out_stride,
                                DType::F32(),
                            ),
                            [patch_fs, weight_fs],
                        ),
                    ),
                    eq(out_shape, [c_out, h_out, w_out]),
                    eq(input_shape, [c_in, h_in, w_in]),
                    eq(
                        idx_shape,
                        [c_in, h_out, w_out, &(DimExpr::MNum(1)), kernel_h, kernel_w],
                    ),
                    eq(matmul_out_shape, [m, c_out]),
                    eq(mul_shape, [m, c_out, k_dim]),
                    eq(k_stride, DimExpr::MIter()),
                    eq(sum_in_stride, [sum_m_stride, sum_c_stride]),
                    eq(sum_out_stride, [sum_out_m_stride, sum_out_c_stride]),
                    eq(
                        sum_add_stride,
                        [sum_add_c_stride, sum_add_h_stride, sum_add_w_stride],
                    ),
                    eq(weight_co_stride, weight_stride.nth_from_end(1_i64)),
                    eq(weight_inner_stride, weight_stride.nth_from_end(0_i64)),
                    eq(DimExpr::MNum(0), weight_stride.nth_from_end(2_i64)),
                    eq(
                        bias_add_stride,
                        [bias_c_stride, &(DimExpr::MNum(0)), &(DimExpr::MNum(0))],
                    ),
                ),
                (
                    unfold_conv,
                    union(out, unfold_conv),
                    subsume(source),
                    set(unfold_conv.dtype(), DType::F32()),
                ),
            )
            .label("kernel conv2d from cuda lowered bias unfold matmul"),
        ]
    }

    #[ruleset]
    pub(crate) fn kernel_specialize(
        out_shape: &EList,
        sum_add_stride: &EList,
        bias_add_stride: &EList,
        add_out_stride: &EList,
        sum: &Ir,
        bias: &Ir,
        add: &Ir,
        matmul_out_shape: &EList,
        k_dim: &DimExpr,
        k_stride: &DimExpr,
        weight_stride: &EList,
        weight: &Ir,
        idx_shape: &EList,
        input_shape: &EList,
        input_stride: &EList,
        indices: &Ir,
        input: &Ir,
        c_out: &DimExpr,
        h_out: &DimExpr,
        w_out: &DimExpr,
        c_in: &DimExpr,
        h_in: &DimExpr,
        w_in: &DimExpr,
        kernel_h: &DimExpr,
        kernel_w: &DimExpr,
        m: &DimExpr,
        weight_co_stride: &DimExpr,
        weight_inner_stride: &DimExpr,
        bias_c_stride: &DimExpr,
    ) -> Vec<Rule> {
        let conv = &Ir::Op(
            kernel_conv2_d(
                out_shape,
                input_shape,
                input_stride,
                weight_co_stride,
                weight_inner_stride,
                bias_c_stride,
                add_out_stride,
                kernel_h,
                kernel_w,
                1,
                1,
                1,
                1,
                0,
                0,
                DType::F32(),
            ),
            [input, weight, bias],
        );
        let patches = &Ir::Op(
            GatherArgs {
                index_shape: idx_shape.into(),
                data_shape: input_shape.into(),
                data_strides: input_stride.into(),
                ..GatherArgs::fresh()
            },
            [indices, input],
        );
        let mul = &Ir::Op(
            MulArgs {
                b_strides: weight_stride.into(),
                ..MulArgs::fresh()
            },
            [patches, weight],
        );
        vec![
            rule(
                (
                    eq(
                        add,
                        Ir::Op(
                            OpKind::Add(out_shape, sum_add_stride, bias_add_stride, add_out_stride),
                            [sum, bias],
                        ),
                    ),
                    eq(
                        sum,
                        Ir::Op(
                            SumArgs {
                                shape: matmul_out_shape.into(),
                                iters: k_dim.into(),
                                iter_stride: k_stride.into(),
                                ..SumArgs::fresh()
                            },
                            [mul],
                        ),
                    ),
                    eq(out_shape, [c_out, h_out, w_out]),
                    eq(input_shape, [c_in, h_in, w_in]),
                    eq(
                        idx_shape,
                        [c_in, h_out, w_out, &(DimExpr::MNum(1)), kernel_h, kernel_w],
                    ),
                    eq(matmul_out_shape, [m, c_out]),
                    eq(h_out, (h_in - kernel_h) + 1),
                    eq(w_out, (w_in - kernel_w) + 1),
                    eq(m, h_out * w_out),
                    eq(k_dim, c_in * (kernel_h * kernel_w)),
                    eq(k_stride, DimExpr::MIter()),
                    eq(weight_co_stride, weight_stride.nth_from_end(1_i64)),
                    eq(weight_inner_stride, weight_stride.nth_from_end(0_i64)),
                    eq(DimExpr::MNum(0), weight_stride.nth_from_end(2_i64)),
                    eq(
                        bias_add_stride,
                        [bias_c_stride, &(DimExpr::MNum(0)), &(DimExpr::MNum(0))],
                    ),
                    eq(DType::F32(), input.dtype()),
                    eq(DType::F32(), weight.dtype()),
                    eq(DType::F32(), bias.dtype()),
                ),
                (
                    conv,
                    union(add, conv),
                    subsume(Ir::Op(
                        OpKind::Add(out_shape, sum_add_stride, bias_add_stride, add_out_stride),
                        [sum, bias],
                    )),
                    set(conv.dtype(), DType::F32()),
                ),
            )
            .label("kernel conv2d from unfold matmul bias"),
            rule(
                (
                    eq(
                        add,
                        Ir::Op(
                            OpKind::Add(out_shape, bias_add_stride, sum_add_stride, add_out_stride),
                            [bias, sum],
                        ),
                    ),
                    eq(
                        sum,
                        Ir::Op(
                            SumArgs {
                                shape: matmul_out_shape.into(),
                                iters: k_dim.into(),
                                iter_stride: k_stride.into(),
                                ..SumArgs::fresh()
                            },
                            [mul],
                        ),
                    ),
                    eq(out_shape, [c_out, h_out, w_out]),
                    eq(input_shape, [c_in, h_in, w_in]),
                    eq(
                        idx_shape,
                        [c_in, h_out, w_out, &(DimExpr::MNum(1)), kernel_h, kernel_w],
                    ),
                    eq(matmul_out_shape, [m, c_out]),
                    eq(h_out, (h_in - kernel_h) + 1),
                    eq(w_out, (w_in - kernel_w) + 1),
                    eq(m, h_out * w_out),
                    eq(k_dim, c_in * (kernel_h * kernel_w)),
                    eq(k_stride, DimExpr::MIter()),
                    eq(weight_co_stride, weight_stride.nth_from_end(1_i64)),
                    eq(weight_inner_stride, weight_stride.nth_from_end(0_i64)),
                    eq(DimExpr::MNum(0), weight_stride.nth_from_end(2_i64)),
                    eq(
                        bias_add_stride,
                        [bias_c_stride, &(DimExpr::MNum(0)), &(DimExpr::MNum(0))],
                    ),
                    eq(DType::F32(), input.dtype()),
                    eq(DType::F32(), weight.dtype()),
                    eq(DType::F32(), bias.dtype()),
                ),
                (
                    conv,
                    union(add, conv),
                    subsume(Ir::Op(
                        OpKind::Add(out_shape, bias_add_stride, sum_add_stride, add_out_stride),
                        [bias, sum],
                    )),
                    set(conv.dtype(), DType::F32()),
                ),
            )
            .label("kernel conv2d from bias unfold matmul"),
        ]
    }

    #[ruleset]
    pub(crate) fn cleanup(
        shape: &EList,
        as_value: &EList,
        bs: &EList,
        os: &EList,
        inputs: &IList,
        add: &Ir,
        out_shape: &EList,
        input_shape: &EList,
        input_stride: &EList,
        wco: &DimExpr,
        wi: &DimExpr,
        bc: &DimExpr,
        out_stride: &EList,
        kh: &DimExpr,
        kw: &DimExpr,
        sh: &DimExpr,
        sw: &DimExpr,
        dh: &DimExpr,
        dw: &DimExpr,
        ph: &DimExpr,
        pw: &DimExpr,
        dt: &DType,
        conv_inputs: &IList,
        fe: &Ir,
        conv_dt: &DType,
    ) -> Vec<Rule> {
        let add_source = &Ir::Op(OpKind::Add(shape, as_value, bs, os), inputs);
        let fusion_source = &Ir::Op(fusion_end(shape, os, dt), inputs);
        vec![
            rule(
                (
                    eq(add, add_source),
                    eq(
                        add,
                        Ir::Op(
                            kernel_conv2_d(
                                out_shape,
                                input_shape,
                                input_stride,
                                wco,
                                wi,
                                bc,
                                out_stride,
                                kh,
                                kw,
                                sh,
                                sw,
                                dh,
                                dw,
                                ph,
                                pw,
                                dt,
                            ),
                            conv_inputs,
                        ),
                    ),
                ),
                (delete(add_source),),
            ),
            rule(
                (
                    eq(fe, fusion_source),
                    eq(
                        fe,
                        Ir::Op(
                            kernel_conv2_d(
                                out_shape,
                                input_shape,
                                input_stride,
                                wco,
                                wi,
                                bc,
                                out_stride,
                                kh,
                                kw,
                                sh,
                                sw,
                                dh,
                                dw,
                                ph,
                                pw,
                                conv_dt,
                            ),
                            conv_inputs,
                        ),
                    ),
                ),
                (delete(fusion_source),),
            ),
        ]
    }
}

impl KernelConv2D {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((
            field_out_shape,
            field_input_shape,
            field_input_stride,
            field_weight_co_stride,
            field_weight_inner_stride,
            field_bias_c_stride,
            field_out_stride,
            field_kernel_h,
            field_kernel_w,
            field_stride_h,
            field_stride_w,
            field_dilation_h,
            field_dilation_w,
            field_pad_h,
            field_pad_w,
            field_dtype,
        )) = typed::get_args(
            kind,
            |out_shape: &EList,
             input_shape: &EList,
             input_stride: &EList,
             weight_co_stride: &DimExpr,
             weight_inner_stride: &DimExpr,
             bias_c_stride: &DimExpr,
             out_stride: &EList,
             kernel_h: &DimExpr,
             kernel_w: &DimExpr,
             stride_h: &DimExpr,
             stride_w: &DimExpr,
             dilation_h: &DimExpr,
             dilation_w: &DimExpr,
             pad_h: &DimExpr,
             pad_w: &DimExpr,
             dtype: &luminal::typed_reference::schema::DType| {
                crate::typed_rules::kernel_conv2_d(
                    out_shape,
                    input_shape,
                    input_stride,
                    weight_co_stride,
                    weight_inner_stride,
                    bias_c_stride,
                    out_stride,
                    kernel_h,
                    kernel_w,
                    stride_h,
                    stride_w,
                    dilation_h,
                    dilation_w,
                    pad_h,
                    pad_w,
                    dtype,
                )
            },
        )?
        else {
            return Ok(None);
        };
        let decoded_out_shape = context.expressions(&field_out_shape)?;
        let decoded_input_shape = context.expressions(&field_input_shape)?;
        let decoded_input_stride = context.expressions(&field_input_stride)?;
        let decoded_out_stride = context.expressions(&field_out_stride)?;
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            out_shape: decoded_out_shape,
            input_shape: decoded_input_shape,
            input_stride: decoded_input_stride,
            weight_co_stride: context.expression(&field_weight_co_stride)?,
            weight_inner_stride: context.expression(&field_weight_inner_stride)?,
            bias_c_stride: context.expression(&field_bias_c_stride)?,
            out_stride: decoded_out_stride,
            kernel_h: context.expression(&field_kernel_h)?,
            kernel_w: context.expression(&field_kernel_w)?,
            stride_h: context.expression(&field_stride_h)?,
            stride_w: context.expression(&field_stride_w)?,
            dilation_h: context.expression(&field_dilation_h)?,
            dilation_w: context.expression(&field_dilation_w)?,
            pad_h: context.expression(&field_pad_h)?,
            pad_w: context.expression(&field_pad_w)?,
            dtype: context.dtype(&field_dtype)?,
        })
            as Box<dyn KernelOp>)))
    }
}

impl KernelOp for KernelConv2D {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        assert_eq!(self.dtype, DType::F32, "KernelConv2D currently emits F32");

        let vars: FxHashSet<char> = self
            .out_shape
            .iter()
            .chain(&self.input_shape)
            .chain(&self.input_stride)
            .chain(&self.out_stride)
            .flat_map(|e| e.dyn_vars())
            .chain(self.weight_co_stride.dyn_vars())
            .chain(self.weight_inner_stride.dyn_vars())
            .chain(self.bias_c_stride.dyn_vars())
            .chain(self.kernel_h.dyn_vars())
            .chain(self.kernel_w.dyn_vars())
            .chain(self.stride_h.dyn_vars())
            .chain(self.stride_w.dyn_vars())
            .chain(self.dilation_h.dyn_vars())
            .chain(self.dilation_w.dyn_vars())
            .chain(self.pad_h.dyn_vars())
            .chain(self.pad_w.dyn_vars())
            .collect();

        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let c_out = self.out_shape[0].to_kernel();
        let h_out = self.out_shape[1].to_kernel();
        let w_out = self.out_shape[2].to_kernel();
        let c_in = self.input_shape[0].to_kernel();
        let h_in = self.input_shape[1].to_kernel();
        let w_in = self.input_shape[2].to_kernel();
        let weight_co_stride = self
            .weight_co_stride
            .substitute('z', Expression::from(1))
            .simplify()
            .to_kernel();
        let weight_inner_stride = self
            .weight_inner_stride
            .substitute('z', Expression::from(1))
            .simplify()
            .to_kernel();
        let bias_c_stride = self
            .bias_c_stride
            .substitute('z', Expression::from(1))
            .simplify()
            .to_kernel();
        let kh = self.kernel_h.to_kernel();
        let kw = self.kernel_w.to_kernel();
        let stride_h = self.stride_h.to_kernel();
        let stride_w = self.stride_w.to_kernel();
        let dilation_h = self.dilation_h.to_kernel();
        let dilation_w = self.dilation_w.to_kernel();
        let pad_h = self.pad_h.to_kernel();
        let pad_w = self.pad_w.to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let input_idx = flatten_strides(&self.input_shape, &self.input_stride)
            .to_kernel()
            .replace("const_z", "input_linear");
        let n_outputs: Expression = self.out_shape.iter().copied().product();

        let kernel = format!(
            "
{dyn_defines}
extern \"C\" {{
    __global__ void generic_conv2d_bias(
        float* __restrict__ out,
        const float* __restrict__ input,
        const float* __restrict__ weight,
        const float* __restrict__ bias{dyn_dims_param}
    ) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        const long long total = {total};
        if (const_z >= total) return;

        const long long COUT = {c_out};
        const long long HOUT = {h_out};
        const long long WOUT = {w_out};
        const long long CIN = {c_in};
        const long long HIN = {h_in};
        const long long WIN = {w_in};
        const long long KH = {kh};
        const long long KW = {kw};
        const long long SH = {stride_h};
        const long long SW = {stride_w};
        const long long DH = {dilation_h};
        const long long DW = {dilation_w};
        const long long PH = {pad_h};
        const long long PW = {pad_w};
        const long long W_CO_STRIDE = {weight_co_stride};
        const long long W_INNER_STRIDE = {weight_inner_stride};
        const long long BIAS_C_STRIDE = {bias_c_stride};

        long long co = const_z / (HOUT * WOUT);
        long long rem = const_z - co * HOUT * WOUT;
        long long oh = rem / WOUT;
        long long ow = rem - oh * WOUT;

        float acc = bias[co * BIAS_C_STRIDE];
        for (long long ci = 0; ci < CIN; ++ci) {{
            for (long long r = 0; r < KH; ++r) {{
                long long ih = oh * SH + r * DH - PH;
                if (ih < 0 || ih >= HIN) continue;
                for (long long s = 0; s < KW; ++s) {{
                    long long iw = ow * SW + s * DW - PW;
                    if (iw < 0 || iw >= WIN) continue;
                    long long input_linear = (ci * HIN + ih) * WIN + iw;
                    long long input_idx = {input_idx};
                    long long inner = (ci * KH + r) * KW + s;
                    long long weight_idx = co * W_CO_STRIDE + inner * W_INNER_STRIDE;
                    acc += input[input_idx] * weight[weight_idx];
                }}
            }}
        }}
        out[{out_idx}] = acc;
    }}
}}",
            total = n_outputs.to_kernel(),
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("generic_conv2d_bias").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_outputs.ceil_div(256), 1.into(), 1.into()),
            (n_outputs.min(256), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn all_dyn_vars(&self) -> FxHashSet<char> {
        self.out_shape
            .iter()
            .chain(&self.input_shape)
            .chain(&self.input_stride)
            .chain(&self.out_stride)
            .flat_map(|e| e.dyn_vars())
            .chain(self.weight_co_stride.dyn_vars())
            .chain(self.weight_inner_stride.dyn_vars())
            .chain(self.bias_c_stride.dyn_vars())
            .chain(self.kernel_h.dyn_vars())
            .chain(self.kernel_w.dyn_vars())
            .chain(self.stride_h.dyn_vars())
            .chain(self.stride_w.dyn_vars())
            .chain(self.dilation_h.dyn_vars())
            .chain(self.dilation_w.dyn_vars())
            .chain(self.pad_h.dyn_vars())
            .chain(self.pad_w.dyn_vars())
            .collect()
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn bytes_loaded(&self) -> Expression {
        let c_in = self.input_shape[0];
        self.output_size() * self.kernel_h * self.kernel_w * c_in * 2 * 4 + self.output_size() * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        let c_in = self.input_shape[0];
        self.output_size() * self.kernel_h * self.kernel_w * c_in * 2
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "GenericConv2D"
    }
}
