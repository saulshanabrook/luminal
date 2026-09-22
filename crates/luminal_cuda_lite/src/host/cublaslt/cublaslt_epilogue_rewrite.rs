// cuBLASLt epilogue rewrites.
//
// ReLU in the frontend lowers through maximum_f32(0.0):
//
//   (matmul < 0) * 0 + cast(cast((-cast(matmul < 0) + 1) as bool) as f32) * matmul
//
// These rules fuse that expression back into CUBLASLT_EPILOGUE_RELU.

use crate::typed_rules::*;
use egglog_experimental::typed::prelude::*;
use luminal::typed_reference::schema::*;

#[allow(
    clippy::approx_constant,
    reason = "Exact pattern literals must match the authored graph."
)]
#[ruleset]
pub(crate) fn matmul_backend(
    m: &DimExpr,
    n: &DimExpr,
    batch: &DimExpr,
    a: &Ir,
    b: &Ir,
    matmul_tail: &IList,
    matmul: &Ir,
    shape: &EList,
    matmul_strides: &EList,
    mask_strides: &EList,
    lt: &Ir,
    size: &DimExpr,
    zeroed_strides: &EList,
    zeroed: &Ir,
    neg_mask_strides: &EList,
    neg_mask: &Ir,
    not_mask_f32_strides: &EList,
    not_mask_f: &Ir,
    positive_strides: &EList,
    out_strides: &EList,
    relu: &Ir,
    gelu_out: &Ir,
    d_dtype: &DType,
    matmul_add_strides: &EList,
    bias_add_strides: &EList,
    add_out_strides: &EList,
    bias: &Ir,
    add: &Ir,
) -> Vec<Rule> {
    let plain_args = CuBlasLtArgs {
        d_dtype: DType::F32(),
        beta: 0.0.into(),
        epilogue: "DEFAULT".into(),
        ..CuBlasLtArgs::fresh()
    };
    let biased_args = CuBlasLtArgs {
        d_dtype: DType::F32(),
        beta: 0.0.into(),
        epilogue: "BIAS".into(),
        ..CuBlasLtArgs::fresh()
    };
    let column_bias_args = CuBlasLtArgs {
        m: m.into(),
        n: n.into(),
        d_order: "COL".into(),
        batch_count: 1.into(),
        d_dtype: d_dtype.into(),
        beta: 0.0.into(),
        epilogue: "DEFAULT".into(),
        ..CuBlasLtArgs::fresh()
    };
    let batched_column_bias_args = CuBlasLtArgs {
        m: m.into(),
        n: n.into(),
        d_order: "COL".into(),
        batch_count: batch.into(),
        d_dtype: d_dtype.into(),
        beta: 0.0.into(),
        epilogue: "DEFAULT".into(),
        ..CuBlasLtArgs::fresh()
    };

    let relu_matmul = &Ir::Op(
        CuBlasLtArgs {
            epilogue: "RELU".into(),
            ..plain_args.clone()
        },
        IList::ICons(a, IList::ICons(b, matmul_tail)),
    );
    let relu_bias_matmul = &Ir::Op(
        CuBlasLtArgs {
            epilogue: "RELU_BIAS".into(),
            ..biased_args.clone()
        },
        IList::ICons(a, IList::ICons(b, matmul_tail)),
    );
    let gelu_matmul = &Ir::Op(
        CuBlasLtArgs {
            epilogue: "GELU".into(),
            ..plain_args.clone()
        },
        IList::ICons(a, IList::ICons(b, matmul_tail)),
    );
    let gelu_bias_matmul = &Ir::Op(
        CuBlasLtArgs {
            epilogue: "GELU_BIAS".into(),
            ..biased_args.clone()
        },
        IList::ICons(a, IList::ICons(b, matmul_tail)),
    );
    let bias_matmul = &Ir::Op(
        CuBlasLtArgs {
            epilogue: "BIAS".into(),
            ..column_bias_args.clone()
        },
        [a, b, bias],
    );
    let batched_bias_matmul = &Ir::Op(
        CuBlasLtArgs {
            epilogue: "BIAS".into(),
            ..batched_column_bias_args.clone()
        },
        [a, b, bias],
    );
    let zero = &Ir::Op(OpKind::Constant(0.0), IList::INil());
    let neg_one = &Ir::Op(OpKind::Constant(-1.0), IList::INil());
    let one = &Ir::Op(OpKind::Constant(1.0), IList::INil());
    let lt_f = &Ir::Op(OpKind::Cast(size, DType::F32()), [lt]);
    let not_mask_bool = &Ir::Op(OpKind::Cast(size, DType::Bool()), [not_mask_f]);
    let gelu_coeff_inner = &Ir::Op(OpKind::Constant(0.044715), IList::INil());
    let gelu_one = &Ir::Op(OpKind::Constant(1.0), IList::INil());
    let gelu_coeff_outer = &Ir::Op(OpKind::Constant(1.595769), IList::INil());
    let neg = &Ir::Op(OpKind::Constant(-1.0), IList::INil());
    let log2e = &Ir::Op(OpKind::Constant(1.442695), IList::INil());
    let not_mask = &Ir::Op(OpKind::Cast(size, DType::F32()), [not_mask_bool]);
    let gelu_inner_scaled = &Ir::Op(MulArgs::fresh(), [matmul, gelu_coeff_inner]);
    let gelu_outer_scaled = &Ir::Op(MulArgs::fresh(), [matmul, gelu_coeff_outer]);
    let positive = &Ir::Op(
        OpKind::Mul(
            shape,
            not_mask_f32_strides,
            matmul_strides,
            positive_strides,
        ),
        [not_mask, matmul],
    );
    let gelu_inner_quad = &Ir::Op(MulArgs::fresh(), [gelu_inner_scaled, matmul]);
    let gelu_poly = &Ir::Op(AddArgs::fresh(), [gelu_inner_quad, gelu_one]);
    let gelu_scaled = &Ir::Op(MulArgs::fresh(), [gelu_outer_scaled, gelu_poly]);
    let gelu_neg = &Ir::Op(MulArgs::fresh(), [gelu_scaled, neg]);
    let gelu_exp_scaled = &Ir::Op(MulArgs::fresh(), [gelu_neg, log2e]);
    let gelu_exp2_val = &Ir::Op(Exp2Args::fresh(), [gelu_exp_scaled]);
    let gelu_plus = &Ir::Op(AddArgs::fresh(), [gelu_exp2_val, gelu_one]);
    let gelu_sigmoid = &Ir::Op(RecipArgs::fresh(), [gelu_plus]);
    vec![
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        plain_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                eq(
                    lt,
                    Ir::Op(
                        OpKind::LessThan(shape, matmul_strides, [0, 0], mask_strides),
                        [matmul, zero],
                    ),
                ),
                eq(
                    zeroed,
                    Ir::Op(
                        OpKind::Mul(shape, mask_strides, [0, 0], zeroed_strides),
                        [lt_f, zero],
                    ),
                ),
                eq(
                    neg_mask,
                    Ir::Op(
                        OpKind::Mul(shape, mask_strides, [0, 0], neg_mask_strides),
                        [lt_f, neg_one],
                    ),
                ),
                eq(
                    not_mask_f,
                    Ir::Op(
                        OpKind::Add(shape, neg_mask_strides, [0, 0], not_mask_f32_strides),
                        [neg_mask, one],
                    ),
                ),
                eq(
                    relu,
                    Ir::Op(
                        OpKind::Add(shape, zeroed_strides, positive_strides, out_strides),
                        [zeroed, positive],
                    ),
                ),
            ),
            (
                relu_matmul,
                union(relu, relu_matmul),
                set(relu_matmul.dtype(), DType::F32()),
            ),
        )
        .label("cublaslt 2d relu epilogue"),
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        plain_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                eq(
                    lt,
                    Ir::Op(
                        OpKind::LessThan(shape, matmul_strides, [0, 0, 0], mask_strides),
                        [matmul, zero],
                    ),
                ),
                eq(
                    zeroed,
                    Ir::Op(
                        OpKind::Mul(shape, mask_strides, [0, 0, 0], zeroed_strides),
                        [lt_f, zero],
                    ),
                ),
                eq(
                    neg_mask,
                    Ir::Op(
                        OpKind::Mul(shape, mask_strides, [0, 0, 0], neg_mask_strides),
                        [lt_f, neg_one],
                    ),
                ),
                eq(
                    not_mask_f,
                    Ir::Op(
                        OpKind::Add(shape, neg_mask_strides, [0, 0, 0], not_mask_f32_strides),
                        [neg_mask, one],
                    ),
                ),
                eq(
                    relu,
                    Ir::Op(
                        OpKind::Add(shape, zeroed_strides, positive_strides, out_strides),
                        [zeroed, positive],
                    ),
                ),
            ),
            (
                relu_matmul,
                union(relu, relu_matmul),
                set(relu_matmul.dtype(), DType::F32()),
            ),
        )
        .label("cublaslt batched relu epilogue"),
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        biased_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                eq(
                    lt,
                    Ir::Op(
                        OpKind::LessThan(shape, matmul_strides, [0, 0], mask_strides),
                        [matmul, zero],
                    ),
                ),
                eq(
                    zeroed,
                    Ir::Op(
                        OpKind::Mul(shape, mask_strides, [0, 0], zeroed_strides),
                        [lt_f, zero],
                    ),
                ),
                eq(
                    neg_mask,
                    Ir::Op(
                        OpKind::Mul(shape, mask_strides, [0, 0], neg_mask_strides),
                        [lt_f, neg_one],
                    ),
                ),
                eq(
                    not_mask_f,
                    Ir::Op(
                        OpKind::Add(shape, neg_mask_strides, [0, 0], not_mask_f32_strides),
                        [neg_mask, one],
                    ),
                ),
                eq(
                    relu,
                    Ir::Op(
                        OpKind::Add(shape, zeroed_strides, positive_strides, out_strides),
                        [zeroed, positive],
                    ),
                ),
            ),
            (
                relu_bias_matmul,
                union(relu, relu_bias_matmul),
                set(relu_bias_matmul.dtype(), DType::F32()),
            ),
        )
        .label("cublaslt 2d relu bias epilogue"),
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        biased_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                eq(
                    lt,
                    Ir::Op(
                        OpKind::LessThan(shape, matmul_strides, [0, 0, 0], mask_strides),
                        [matmul, zero],
                    ),
                ),
                eq(
                    zeroed,
                    Ir::Op(
                        OpKind::Mul(shape, mask_strides, [0, 0, 0], zeroed_strides),
                        [lt_f, zero],
                    ),
                ),
                eq(
                    neg_mask,
                    Ir::Op(
                        OpKind::Mul(shape, mask_strides, [0, 0, 0], neg_mask_strides),
                        [lt_f, neg_one],
                    ),
                ),
                eq(
                    not_mask_f,
                    Ir::Op(
                        OpKind::Add(shape, neg_mask_strides, [0, 0, 0], not_mask_f32_strides),
                        [neg_mask, one],
                    ),
                ),
                eq(
                    relu,
                    Ir::Op(
                        OpKind::Add(shape, zeroed_strides, positive_strides, out_strides),
                        [zeroed, positive],
                    ),
                ),
            ),
            (
                relu_bias_matmul,
                union(relu, relu_bias_matmul),
                set(relu_bias_matmul.dtype(), DType::F32()),
            ),
        )
        .label("cublaslt batched relu bias epilogue"),
        // Canonical tanh-approx GELU can also appear directly as:
        // x * sigmoid(1.5957691216 * x * (1 + 0.044715 * x * x))
        // Match that sigmoid form and fuse it into the cuBLASLt GELU epilogues.
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        plain_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                eq(gelu_out, Ir::Op(MulArgs::fresh(), [matmul, gelu_sigmoid])),
            ),
            (
                gelu_matmul,
                union(gelu_out, gelu_matmul),
                set(gelu_matmul.dtype(), DType::F32()),
            ),
        )
        .label("cublaslt gelu epilogue"),
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        biased_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                eq(gelu_out, Ir::Op(MulArgs::fresh(), [matmul, gelu_sigmoid])),
            ),
            (
                gelu_bias_matmul,
                union(gelu_out, gelu_bias_matmul),
                set(gelu_bias_matmul.dtype(), DType::F32()),
            ),
        )
        .label("cublaslt gelu bias epilogue"),
        // This first slice fuses column-bias adds into CUBLASLT_EPILOGUE_BIAS for the
        // older COL-ordered output view. In that view Luminal's logical [m,n] output is
        // represented as a cuBLASLt [n,m] matrix, so cuBLASLt's row-broadcast bias maps
        // to the common logical column bias of length n.
        rule(
            (
                eq(matmul, Ir::Op(column_bias_args.clone(), [a, b])),
                eq(
                    add,
                    Ir::Op(
                        OpKind::Add(
                            [n, m],
                            matmul_add_strides,
                            bias_add_strides,
                            add_out_strides,
                        ),
                        [matmul, bias],
                    ),
                ),
                eq(bias_add_strides, [&(DimExpr::MNum(0)), &(DimExpr::MIter())]),
                eq(matmul_add_strides, add_out_strides),
                eq(d_dtype, bias.dtype()),
            ),
            (
                bias_matmul,
                union(add, bias_matmul),
                set(bias_matmul.dtype(), d_dtype),
            ),
        )
        .label("cublaslt 2d matmul plus column bias epilogue"),
        rule(
            (
                eq(matmul, Ir::Op(column_bias_args.clone(), [a, b])),
                eq(
                    add,
                    Ir::Op(
                        OpKind::Add(
                            [n, m],
                            bias_add_strides,
                            matmul_add_strides,
                            add_out_strides,
                        ),
                        [bias, matmul],
                    ),
                ),
                eq(bias_add_strides, [&(DimExpr::MNum(0)), &(DimExpr::MIter())]),
                eq(matmul_add_strides, add_out_strides),
                eq(d_dtype, bias.dtype()),
            ),
            (
                bias_matmul,
                union(add, bias_matmul),
                set(bias_matmul.dtype(), d_dtype),
            ),
        )
        .label("cublaslt 2d column bias plus matmul epilogue"),
        rule(
            (
                eq(matmul, Ir::Op(batched_column_bias_args.clone(), [a, b])),
                eq(
                    add,
                    Ir::Op(
                        OpKind::Add(
                            [batch, n, m],
                            matmul_add_strides,
                            bias_add_strides,
                            add_out_strides,
                        ),
                        [matmul, bias],
                    ),
                ),
                eq(
                    bias_add_strides,
                    [
                        &(DimExpr::MNum(0)),
                        &(DimExpr::MNum(0)),
                        &(DimExpr::MIter()),
                    ],
                ),
                eq(matmul_add_strides, add_out_strides),
                eq(d_dtype, bias.dtype()),
            ),
            (
                batched_bias_matmul,
                union(add, batched_bias_matmul),
                set(batched_bias_matmul.dtype(), d_dtype),
            ),
        )
        .label("cublaslt batched matmul plus column bias epilogue"),
        rule(
            (
                eq(matmul, Ir::Op(batched_column_bias_args.clone(), [a, b])),
                eq(
                    add,
                    Ir::Op(
                        OpKind::Add(
                            [batch, n, m],
                            bias_add_strides,
                            matmul_add_strides,
                            add_out_strides,
                        ),
                        [bias, matmul],
                    ),
                ),
                eq(
                    bias_add_strides,
                    [
                        &(DimExpr::MNum(0)),
                        &(DimExpr::MNum(0)),
                        &(DimExpr::MIter()),
                    ],
                ),
                eq(matmul_add_strides, add_out_strides),
                eq(d_dtype, bias.dtype()),
            ),
            (
                batched_bias_matmul,
                union(add, batched_bias_matmul),
                set(batched_bias_matmul.dtype(), d_dtype),
            ),
        )
        .label("cublaslt batched column bias plus matmul epilogue"),
    ]
}
