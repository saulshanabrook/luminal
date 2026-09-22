// Scalar alpha/beta rewrites for cuBLASLt. These rules target scalar constants
// expanded across the matmul/add shape, i.e. zero strides on every logical axis.

use crate::typed_rules::*;
use egglog_experimental::typed::builtins as egg;
use egglog_experimental::typed::prelude::*;
use luminal::typed_reference::schema::*;

#[ruleset]
pub(crate) fn matmul_backend(
    m: &DimExpr,
    n: &DimExpr,
    batch: &DimExpr,
    c_dtype: &DType,
    d_dtype: &DType,
    a: &Ir,
    b: &Ir,
    matmul_tail: &IList,
    matmul: &Ir,
    alpha: &egg::F64,
    shape: &EList,
    matmul_strides: &EList,
    scaled_out_strides: &EList,
    scaled: &Ir,
    shape1: &EList,
    beta: &egg::F64,
    c_strides: &EList,
    scaled_c_out_strides: &EList,
    c: &Ir,
    scaled_c: &Ir,
    matmul_add_strides: &EList,
    scaled_c_add_strides: &EList,
    add_out_strides: &EList,
    add: &Ir,
    d_row_stride: &DimExpr,
    d_col_stride: &DimExpr,
    c_row_stride: &DimExpr,
    c_col_stride: &DimExpr,
    d_batch_stride: &DimExpr,
    c_batch_stride: &DimExpr,
) -> Vec<Rule> {
    let matmul_args = CuBlasLtArgs {
        d_dtype: d_dtype.into(),
        alpha: 1.0.into(),
        beta: 0.0.into(),
        epilogue: "DEFAULT".into(),
        ..CuBlasLtArgs::fresh()
    };
    let beta_args = CuBlasLtArgs {
        m: m.into(),
        n: n.into(),
        d_order: "ROW".into(),
        batch_count: 1.into(),
        c_dtype: c_dtype.into(),
        d_dtype: d_dtype.into(),
        beta: 0.0.into(),
        ..CuBlasLtArgs::fresh()
    };
    let batched_beta_args = CuBlasLtArgs {
        m: m.into(),
        n: n.into(),
        d_order: "ROW".into(),
        batch_count: batch.into(),
        c_dtype: c_dtype.into(),
        d_dtype: d_dtype.into(),
        beta: 0.0.into(),
        ..CuBlasLtArgs::fresh()
    };

    let scaled_matmul = &Ir::Op(
        CuBlasLtArgs {
            alpha: alpha.into(),
            ..matmul_args.clone()
        },
        IList::ICons(a, IList::ICons(b, matmul_tail)),
    );
    let scaled_beta = &Ir::Op(
        CuBlasLtArgs {
            c_order: "ROW".into(),
            ldc: c_row_stride.into(),
            stride_c: 0.into(),
            beta: beta.into(),
            ..beta_args.clone()
        },
        IList::ICons(a, IList::ICons(b, IList::ICons(c, matmul_tail))),
    );
    let batched_scaled_beta = &Ir::Op(
        CuBlasLtArgs {
            c_order: "ROW".into(),
            ldc: c_row_stride.into(),
            stride_c: c_batch_stride.into(),
            beta: beta.into(),
            ..batched_beta_args.clone()
        },
        IList::ICons(a, IList::ICons(b, IList::ICons(c, matmul_tail))),
    );
    let scale = &Ir::Op(OpKind::Constant(alpha), IList::INil());
    let beta_node = &Ir::Op(OpKind::Constant(beta), IList::INil());
    vec![
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        matmul_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                // alpha=1.0 hash-conses fused == matmul; the union merges
                // Mul into matmul's eclass and saturation diverges.
                // alpha=1.0 hash-conses ?fused == ?matmul; the union merges Mul into ?matmul's eclass and saturate diverges.
                ne(alpha, 1.0),
                eq(
                    scaled,
                    Ir::Op(
                        OpKind::Mul(shape, matmul_strides, [0, 0], scaled_out_strides),
                        [matmul, scale],
                    ),
                ),
                eq(matmul_strides, scaled_out_strides),
            ),
            (
                scaled_matmul,
                union(scaled, scaled_matmul),
                set(scaled_matmul.dtype(), d_dtype),
            ),
        )
        .label("cublaslt 2d alpha scale"),
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        matmul_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                // See 2d alpha scale: alpha=1.0 makes saturation diverge.
                // See 2d alpha scale: alpha=1.0 makes (saturate ...) diverge.
                ne(alpha, 1.0),
                eq(
                    scaled,
                    Ir::Op(
                        OpKind::Mul(shape1, matmul_strides, [0, 0, 0], scaled_out_strides),
                        [matmul, scale],
                    ),
                ),
                eq(matmul_strides, scaled_out_strides),
            ),
            (
                scaled_matmul,
                union(scaled, scaled_matmul),
                set(scaled_matmul.dtype(), d_dtype),
            ),
        )
        .label("cublaslt batched alpha scale"),
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        beta_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                eq(
                    scaled_c,
                    Ir::Op(
                        OpKind::Mul([m, n], c_strides, [0, 0], scaled_c_out_strides),
                        [c, beta_node],
                    ),
                ),
                eq(
                    add,
                    Ir::Op(
                        OpKind::Add(
                            [m, n],
                            matmul_add_strides,
                            scaled_c_add_strides,
                            add_out_strides,
                        ),
                        [matmul, scaled_c],
                    ),
                ),
                eq(matmul_add_strides, [d_row_stride, d_col_stride]),
                eq(c_strides, [c_row_stride, c_col_stride]),
                eq(add_out_strides, [d_row_stride, d_col_stride]),
                eq(scaled_c_add_strides, scaled_c_out_strides),
                eq(c_col_stride, DimExpr::MIter()),
                ne(c_row_stride, 0),
                eq(matmul_add_strides, add_out_strides),
                eq(c_dtype, c.dtype()),
            ),
            (
                scaled_beta,
                union(add, scaled_beta),
                set(scaled_beta.dtype(), d_dtype),
            ),
        )
        .label("cublaslt row-order 2d scaled c beta"),
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        beta_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                eq(
                    scaled_c,
                    Ir::Op(
                        OpKind::Mul([m, n], c_strides, [0, 0], scaled_c_out_strides),
                        [c, beta_node],
                    ),
                ),
                eq(
                    add,
                    Ir::Op(
                        OpKind::Add(
                            [m, n],
                            scaled_c_add_strides,
                            matmul_add_strides,
                            add_out_strides,
                        ),
                        [scaled_c, matmul],
                    ),
                ),
                eq(matmul_add_strides, [d_row_stride, d_col_stride]),
                eq(c_strides, [c_row_stride, c_col_stride]),
                eq(add_out_strides, [d_row_stride, d_col_stride]),
                eq(scaled_c_add_strides, scaled_c_out_strides),
                eq(c_col_stride, DimExpr::MIter()),
                ne(c_row_stride, 0),
                eq(matmul_add_strides, add_out_strides),
                eq(c_dtype, c.dtype()),
            ),
            (
                scaled_beta,
                union(add, scaled_beta),
                set(scaled_beta.dtype(), d_dtype),
            ),
        )
        .label("cublaslt row-order 2d scaled c plus matmul beta"),
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        batched_beta_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                eq(
                    scaled_c,
                    Ir::Op(
                        OpKind::Mul([batch, m, n], c_strides, [0, 0, 0], scaled_c_out_strides),
                        [c, beta_node],
                    ),
                ),
                eq(
                    add,
                    Ir::Op(
                        OpKind::Add(
                            [batch, m, n],
                            matmul_add_strides,
                            scaled_c_add_strides,
                            add_out_strides,
                        ),
                        [matmul, scaled_c],
                    ),
                ),
                eq(
                    matmul_add_strides,
                    [d_batch_stride, d_row_stride, d_col_stride],
                ),
                eq(c_strides, [c_batch_stride, c_row_stride, c_col_stride]),
                eq(
                    add_out_strides,
                    [d_batch_stride, d_row_stride, d_col_stride],
                ),
                eq(scaled_c_add_strides, scaled_c_out_strides),
                eq(c_col_stride, DimExpr::MIter()),
                ne(c_row_stride, 0),
                eq(matmul_add_strides, add_out_strides),
                eq(c_dtype, c.dtype()),
            ),
            (
                batched_scaled_beta,
                union(add, batched_scaled_beta),
                set(batched_scaled_beta.dtype(), d_dtype),
            ),
        )
        .label("cublaslt row-order batched scaled c beta"),
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        batched_beta_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                eq(
                    scaled_c,
                    Ir::Op(
                        OpKind::Mul([batch, m, n], c_strides, [0, 0, 0], scaled_c_out_strides),
                        [c, beta_node],
                    ),
                ),
                eq(
                    add,
                    Ir::Op(
                        OpKind::Add(
                            [batch, m, n],
                            scaled_c_add_strides,
                            matmul_add_strides,
                            add_out_strides,
                        ),
                        [scaled_c, matmul],
                    ),
                ),
                eq(
                    matmul_add_strides,
                    [d_batch_stride, d_row_stride, d_col_stride],
                ),
                eq(c_strides, [c_batch_stride, c_row_stride, c_col_stride]),
                eq(
                    add_out_strides,
                    [d_batch_stride, d_row_stride, d_col_stride],
                ),
                eq(scaled_c_add_strides, scaled_c_out_strides),
                eq(c_col_stride, DimExpr::MIter()),
                ne(c_row_stride, 0),
                eq(matmul_add_strides, add_out_strides),
                eq(c_dtype, c.dtype()),
            ),
            (
                batched_scaled_beta,
                union(add, batched_scaled_beta),
                set(batched_scaled_beta.dtype(), d_dtype),
            ),
        )
        .label("cublaslt row-order batched scaled c plus matmul beta"),
    ]
}
