// Fuse a row-major Add on top of an existing cuBLASLt matmul into
// D = alpha * A * B + beta * C.
//
// The existing matmul rewrites view Luminal's row-major output [m,n] as a
// column-major cuBLASLt matrix [n,m]. A row-major C input with logical strides
// [row_stride, 1] therefore maps to ldc=row_stride. This lets a C slice from a
// wider parent tensor use a larger ldc while D keeps the matmul output layout.
// cuBLASLt requires out-of-place C and D to have the same matrix order, so these
// beta rules only fuse C layouts that map to the current COL-ordered D layout.

use crate::typed_rules::*;
use egglog_experimental::typed::builtins as egg;
use egglog_experimental::typed::prelude::*;
use luminal::typed_reference::schema::*;

#[ruleset]
pub(crate) fn matmul_backend(
    m: &DimExpr,
    n: &DimExpr,
    c_dtype: &DType,
    d_dtype: &DType,
    epilogue: &egg::String,
    a: &Ir,
    b: &Ir,
    matmul_tail: &IList,
    matmul: &Ir,
    matmul_add_strides: &EList,
    c_add_strides: &EList,
    add_out_strides: &EList,
    c: &Ir,
    add: &Ir,
    d_row_stride: &DimExpr,
    d_col_stride: &DimExpr,
    c_row_stride: &DimExpr,
    c_col_stride: &DimExpr,
    batch: &DimExpr,
    d_batch_stride: &DimExpr,
    c_batch_stride: &DimExpr,
) -> Vec<Rule> {
    let column_args = CuBlasLtArgs {
        m: m.into(),
        n: n.into(),
        d_order: "COL".into(),
        batch_count: 1.into(),
        c_dtype: c_dtype.into(),
        d_dtype: d_dtype.into(),
        beta: 0.0.into(),
        epilogue: epilogue.into(),
        ..CuBlasLtArgs::fresh()
    };
    let batched_column_args = CuBlasLtArgs {
        m: m.into(),
        n: n.into(),
        d_order: "COL".into(),
        batch_count: batch.into(),
        c_dtype: c_dtype.into(),
        d_dtype: d_dtype.into(),
        beta: 0.0.into(),
        epilogue: epilogue.into(),
        ..CuBlasLtArgs::fresh()
    };
    let row_args = CuBlasLtArgs {
        m: m.into(),
        n: n.into(),
        d_order: "ROW".into(),
        batch_count: 1.into(),
        c_dtype: c_dtype.into(),
        d_dtype: d_dtype.into(),
        beta: 0.0.into(),
        epilogue: epilogue.into(),
        ..CuBlasLtArgs::fresh()
    };
    let batched_row_args = CuBlasLtArgs {
        m: m.into(),
        n: n.into(),
        d_order: "ROW".into(),
        batch_count: batch.into(),
        c_dtype: c_dtype.into(),
        d_dtype: d_dtype.into(),
        beta: 0.0.into(),
        epilogue: epilogue.into(),
        ..CuBlasLtArgs::fresh()
    };

    let column_beta = &Ir::Op(
        CuBlasLtArgs {
            c_order: "COL".into(),
            ldc: c_row_stride.into(),
            stride_c: 0.into(),
            beta: 1.0.into(),
            ..column_args.clone()
        },
        IList::ICons(a, IList::ICons(b, IList::ICons(c, matmul_tail))),
    );
    let batched_column_beta = &Ir::Op(
        CuBlasLtArgs {
            c_order: "COL".into(),
            ldc: c_row_stride.into(),
            stride_c: c_batch_stride.into(),
            beta: 1.0.into(),
            ..batched_column_args.clone()
        },
        IList::ICons(a, IList::ICons(b, IList::ICons(c, matmul_tail))),
    );
    let row_beta = &Ir::Op(
        CuBlasLtArgs {
            c_order: "ROW".into(),
            ldc: c_row_stride.into(),
            stride_c: 0.into(),
            beta: 1.0.into(),
            ..row_args.clone()
        },
        IList::ICons(a, IList::ICons(b, IList::ICons(c, matmul_tail))),
    );
    let batched_row_beta = &Ir::Op(
        CuBlasLtArgs {
            c_order: "ROW".into(),
            ldc: c_row_stride.into(),
            stride_c: c_batch_stride.into(),
            beta: 1.0.into(),
            ..batched_row_args.clone()
        },
        IList::ICons(a, IList::ICons(b, IList::ICons(c, matmul_tail))),
    );
    vec![
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        column_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                ne(epilogue, "RELU"),
                ne(epilogue, "RELU_BIAS"),
                ne(epilogue, "GELU"),
                ne(epilogue, "GELU_BIAS"),
                eq(
                    add,
                    Ir::Op(
                        OpKind::Add([n, m], matmul_add_strides, c_add_strides, add_out_strides),
                        [matmul, c],
                    ),
                ),
                eq(matmul_add_strides, [d_row_stride, d_col_stride]),
                eq(c_add_strides, [c_row_stride, c_col_stride]),
                eq(add_out_strides, [d_row_stride, d_col_stride]),
                eq(c_col_stride, DimExpr::MIter()),
                ne(c_row_stride, 0),
                eq(matmul_add_strides, add_out_strides),
                eq(c_dtype, c.dtype()),
            ),
            (
                column_beta,
                union(add, column_beta),
                set(column_beta.dtype(), d_dtype),
            ),
        )
        .label("cublaslt 2d matmul plus c beta"),
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        column_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                ne(epilogue, "RELU"),
                ne(epilogue, "RELU_BIAS"),
                ne(epilogue, "GELU"),
                ne(epilogue, "GELU_BIAS"),
                eq(
                    add,
                    Ir::Op(
                        OpKind::Add([n, m], c_add_strides, matmul_add_strides, add_out_strides),
                        [c, matmul],
                    ),
                ),
                eq(matmul_add_strides, [d_row_stride, d_col_stride]),
                eq(c_add_strides, [c_row_stride, c_col_stride]),
                eq(add_out_strides, [d_row_stride, d_col_stride]),
                eq(c_col_stride, DimExpr::MIter()),
                ne(c_row_stride, 0),
                eq(matmul_add_strides, add_out_strides),
                eq(c_dtype, c.dtype()),
            ),
            (
                column_beta,
                union(add, column_beta),
                set(column_beta.dtype(), d_dtype),
            ),
        )
        .label("cublaslt 2d c plus matmul beta"),
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        batched_column_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                ne(epilogue, "RELU"),
                ne(epilogue, "RELU_BIAS"),
                ne(epilogue, "GELU"),
                ne(epilogue, "GELU_BIAS"),
                eq(
                    add,
                    Ir::Op(
                        OpKind::Add(
                            [batch, n, m],
                            matmul_add_strides,
                            c_add_strides,
                            add_out_strides,
                        ),
                        [matmul, c],
                    ),
                ),
                eq(
                    matmul_add_strides,
                    [d_batch_stride, d_row_stride, d_col_stride],
                ),
                eq(c_add_strides, [c_batch_stride, c_row_stride, c_col_stride]),
                eq(
                    add_out_strides,
                    [d_batch_stride, d_row_stride, d_col_stride],
                ),
                eq(c_col_stride, DimExpr::MIter()),
                ne(c_row_stride, 0),
                eq(matmul_add_strides, add_out_strides),
                eq(c_dtype, c.dtype()),
            ),
            (
                batched_column_beta,
                union(add, batched_column_beta),
                set(batched_column_beta.dtype(), d_dtype),
            ),
        )
        .label("cublaslt batched matmul plus c beta"),
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        batched_column_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                ne(epilogue, "RELU"),
                ne(epilogue, "RELU_BIAS"),
                ne(epilogue, "GELU"),
                ne(epilogue, "GELU_BIAS"),
                eq(
                    add,
                    Ir::Op(
                        OpKind::Add(
                            [batch, n, m],
                            c_add_strides,
                            matmul_add_strides,
                            add_out_strides,
                        ),
                        [c, matmul],
                    ),
                ),
                eq(
                    matmul_add_strides,
                    [d_batch_stride, d_row_stride, d_col_stride],
                ),
                eq(c_add_strides, [c_batch_stride, c_row_stride, c_col_stride]),
                eq(
                    add_out_strides,
                    [d_batch_stride, d_row_stride, d_col_stride],
                ),
                eq(c_col_stride, DimExpr::MIter()),
                ne(c_row_stride, 0),
                eq(matmul_add_strides, add_out_strides),
                eq(c_dtype, c.dtype()),
            ),
            (
                batched_column_beta,
                union(add, batched_column_beta),
                set(batched_column_beta.dtype(), d_dtype),
            ),
        )
        .label("cublaslt batched c plus matmul beta"),
        // ROW-ordered D beta fusions. These pair with cublaslt_row_order_rewrite.rs,
        // where the cuBLASLt problem dimensions match Luminal's logical output [m,n].
        // A row-major C input with logical strides [row_stride, 1] maps directly to a
        // ROW-ordered cuBLASLt C[m,n] descriptor with ldc=row_stride.
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        row_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                ne(epilogue, "RELU"),
                ne(epilogue, "RELU_BIAS"),
                ne(epilogue, "GELU"),
                ne(epilogue, "GELU_BIAS"),
                eq(
                    add,
                    Ir::Op(
                        OpKind::Add([m, n], matmul_add_strides, c_add_strides, add_out_strides),
                        [matmul, c],
                    ),
                ),
                eq(matmul_add_strides, [d_row_stride, d_col_stride]),
                eq(c_add_strides, [c_row_stride, c_col_stride]),
                eq(add_out_strides, [d_row_stride, d_col_stride]),
                eq(c_col_stride, DimExpr::MIter()),
                ne(c_row_stride, 0),
                eq(matmul_add_strides, add_out_strides),
                eq(c_dtype, c.dtype()),
            ),
            (
                row_beta,
                union(add, row_beta),
                set(row_beta.dtype(), d_dtype),
            ),
        )
        .label("cublaslt row-order 2d matmul plus c beta"),
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        row_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                ne(epilogue, "RELU"),
                ne(epilogue, "RELU_BIAS"),
                ne(epilogue, "GELU"),
                ne(epilogue, "GELU_BIAS"),
                eq(
                    add,
                    Ir::Op(
                        OpKind::Add([m, n], c_add_strides, matmul_add_strides, add_out_strides),
                        [c, matmul],
                    ),
                ),
                eq(matmul_add_strides, [d_row_stride, d_col_stride]),
                eq(c_add_strides, [c_row_stride, c_col_stride]),
                eq(add_out_strides, [d_row_stride, d_col_stride]),
                eq(c_col_stride, DimExpr::MIter()),
                ne(c_row_stride, 0),
                eq(matmul_add_strides, add_out_strides),
                eq(c_dtype, c.dtype()),
            ),
            (
                row_beta,
                union(add, row_beta),
                set(row_beta.dtype(), d_dtype),
            ),
        )
        .label("cublaslt row-order 2d c plus matmul beta"),
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        batched_row_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                ne(epilogue, "RELU"),
                ne(epilogue, "RELU_BIAS"),
                ne(epilogue, "GELU"),
                ne(epilogue, "GELU_BIAS"),
                eq(
                    add,
                    Ir::Op(
                        OpKind::Add(
                            [batch, m, n],
                            matmul_add_strides,
                            c_add_strides,
                            add_out_strides,
                        ),
                        [matmul, c],
                    ),
                ),
                eq(
                    matmul_add_strides,
                    [d_batch_stride, d_row_stride, d_col_stride],
                ),
                eq(c_add_strides, [c_batch_stride, c_row_stride, c_col_stride]),
                eq(
                    add_out_strides,
                    [d_batch_stride, d_row_stride, d_col_stride],
                ),
                eq(c_col_stride, DimExpr::MIter()),
                ne(c_row_stride, 0),
                eq(matmul_add_strides, add_out_strides),
                eq(c_dtype, c.dtype()),
            ),
            (
                batched_row_beta,
                union(add, batched_row_beta),
                set(batched_row_beta.dtype(), d_dtype),
            ),
        )
        .label("cublaslt row-order batched matmul plus c beta"),
        rule(
            (
                eq(
                    matmul,
                    Ir::Op(
                        batched_row_args.clone(),
                        IList::ICons(a, IList::ICons(b, matmul_tail)),
                    ),
                ),
                ne(epilogue, "RELU"),
                ne(epilogue, "RELU_BIAS"),
                ne(epilogue, "GELU"),
                ne(epilogue, "GELU_BIAS"),
                eq(
                    add,
                    Ir::Op(
                        OpKind::Add(
                            [batch, m, n],
                            c_add_strides,
                            matmul_add_strides,
                            add_out_strides,
                        ),
                        [c, matmul],
                    ),
                ),
                eq(
                    matmul_add_strides,
                    [d_batch_stride, d_row_stride, d_col_stride],
                ),
                eq(c_add_strides, [c_batch_stride, c_row_stride, c_col_stride]),
                eq(
                    add_out_strides,
                    [d_batch_stride, d_row_stride, d_col_stride],
                ),
                eq(c_col_stride, DimExpr::MIter()),
                ne(c_row_stride, 0),
                eq(matmul_add_strides, add_out_strides),
                eq(c_dtype, c.dtype()),
            ),
            (
                batched_row_beta,
                union(add, batched_row_beta),
                set(batched_row_beta.dtype(), d_dtype),
            ),
        )
        .label("cublaslt row-order batched c plus matmul beta"),
    ]
}
