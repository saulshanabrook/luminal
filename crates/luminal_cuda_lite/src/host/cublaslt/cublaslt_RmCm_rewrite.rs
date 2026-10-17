// Row-major × Column-major matmul: C[m,n] = A[m,k] × B[k,n]
// A[m,k] row-major → expand to [m, n, k] with strides [k, 0, MIter]
// B[k,n] column-major → expand to [m, n, k] with strides [0, k, MIter]
//
// Row-major viewed as column-major (swap trick):
//   Row-major A[m,k] ≡ column-major A^T[k,m] with lda=k
//   Column-major B[k,n] is already column-major with ldb=k
//   Row-major C[m,n] ≡ column-major C^T[n,m] with ldc=n
//
// C^T[n,m] = (A × B)^T = B^T[n,k] × A^T[k,m]
// cuBLAS: cublasSgemm(OP_T, OP_N, n, m, k, α, B, k, A, k, β, C, n)

use super::rewrites::cublaslt_base_dtype;
use crate::typed_rules::*;
use egglog_experimental::typed::prelude::*;
use luminal::typed_reference::schema::*;

#[ruleset]
pub(crate) fn matmul_backend(
    a: &Ir,
    b: &Ir,
    sum: &Ir,
    m: &DimExpr,
    n: &DimExpr,
    a_m_stride: &DimExpr,
    a_n_stride: &DimExpr,
    a_k_stride: &DimExpr,
    b_m_stride: &DimExpr,
    b_n_stride: &DimExpr,
    b_k_stride: &DimExpr,
    dt: &DType,
    batch: &DimExpr,
    a_batch_stride: &DimExpr,
    b_batch_stride: &DimExpr,
) -> Vec<Rule> {
    let generic_args = GenericMatmulArgs::fresh();
    // For row-major A × column-major B with cuBLAS:
    // C^T = B^T × A^T → cublasSgemm(OP_T, OP_N, n, m, k, α, B, k, A, k, β, C, n)
    let matmul = &Ir::Op(
        cublaslt(
            n,               // cuBLAS m = our n (swapped)
            m,               // cuBLAS n = our m (swapped)
            &generic_args.k, // k unchanged
            "T",             // transa = Transpose (B is column-major, need B^T)
            "N",             // transb = No transpose
            "COL",
            "COL",
            "COL",
            "COL",      // A/B/C/D matrix orders
            b_n_stride, // lda = B's column stride (resolves to k after z→1)
            a_m_stride, // ldb = A's row stride (resolves to k after z→1)
            n,          // ldc = n (row-major C[m,n] viewed as col-major [n,m])
            n,          // ldd = ldc for current row-major output rewrites
            1,          // batch_count = 1
            0,          // stride_a = 0
            0,          // stride_b = 0
            0,          // stride_c = 0
            0,          // stride_d = 0
            dt,
            dt,
            dt,
            dt,
            "default",
            "default",
            1.0,
            0.0,
            "DEFAULT", // type tuple, alpha, beta
        ),
        [b, a],
    );
    // cuBLAS: cublas(OP_T, OP_N, n, m, k, B, lda=b_n_stride, A, ldb=a_m_stride, C, ldc=n)
    let batched_matmul = &Ir::Op(
        cublaslt(
            n,
            m,
            &generic_args.k,
            "T",
            "N",
            "COL",
            "COL",
            "COL",
            "COL",
            b_n_stride, // lda (cuBLAS A = our B, column stride)
            a_m_stride, // ldb (cuBLAS B = our A, row stride)
            n,          // ldc
            n,          // ldd
            batch,
            b_batch_stride, // stride_a (cuBLAS A = our B)
            a_batch_stride, // stride_b (cuBLAS B = our A)
            m * n,          // stride_c
            m * n,          // stride_d
            dt,
            dt,
            dt,
            dt,
            "default",
            "default",
            1.0,
            0.0,
            "DEFAULT",
        ),
        [b, a],
    );
    vec![
        rule(
            (
                // Match the generic matmul produced from Mul -> Sum.
                eq(sum, Ir::Op(generic_args.clone(), [a, b])),
                // Match exactly 2D output shape
                eq(&generic_args.out_shape, [m, n]),
                ne(m, 0),
                ne(n, 0),
                ne(&generic_args.k, 1),
                // Match exactly 3D strides [m, n, k]
                eq(
                    &generic_args.lhs_strides,
                    [a_m_stride, a_n_stride, a_k_stride],
                ),
                eq(
                    &generic_args.rhs_strides,
                    [b_m_stride, b_n_stride, b_k_stride],
                ),
                // Assert contiguous k stride on output (required for reduction)
                eq(&generic_args.sum_iter_stride, DimExpr::MIter()),
                eq(a_m_stride, DimExpr::MIter() * &generic_args.k),
                eq(a_n_stride, 0),
                // Assert A has strides [k*MIter, 0, MIter] (row-major A[m,k] broadcast to [m,n,k])
                eq(a_k_stride, DimExpr::MIter()),
                // Assert B has strides [0, k*MIter, MIter] (column-major B[k,n] broadcast to [m,n,k])
                eq(b_m_stride, 0),
                eq(b_n_stride, DimExpr::MIter() * &generic_args.k),
                eq(b_k_stride, DimExpr::MIter()),
                eq(dt, a.dtype()),
                eq(dt, b.dtype()),
                cublaslt_base_dtype(dt),
            ),
            (matmul, union(sum, matmul), set(matmul.dtype(), dt)),
        )
        .label("cublaslt row-major × column-major"),
        // Batched Row-major × Column-major: C[batch,m,n] = A[batch,m,k] × B[batch,k,n]
        // A row-major per batch: a_k_stride=MIter, a_n_stride=0
        // B column-major per batch: b_k_stride=MIter, b_m_stride=0
        rule(
            (
                eq(sum, Ir::Op(generic_args.clone(), [a, b])),
                eq(batch, generic_args.out_shape.nth_from_end(2_i64)),
                eq(m, generic_args.out_shape.nth_from_end(1_i64)),
                eq(n, generic_args.out_shape.nth_from_end(0_i64)),
                ne(m, 0),
                ne(n, 0),
                ne(&generic_args.k, 1),
                ne(batch, 0),
                // Uniform batch strides (contiguous per batch)
                eq(a_batch_stride, generic_args.lhs_strides.nth_from_end(3_i64)),
                eq(a_m_stride, generic_args.lhs_strides.nth_from_end(2_i64)),
                eq(a_n_stride, generic_args.lhs_strides.nth_from_end(1_i64)),
                eq(a_k_stride, generic_args.lhs_strides.nth_from_end(0_i64)),
                eq(b_batch_stride, generic_args.rhs_strides.nth_from_end(3_i64)),
                eq(b_m_stride, generic_args.rhs_strides.nth_from_end(2_i64)),
                eq(b_n_stride, generic_args.rhs_strides.nth_from_end(1_i64)),
                eq(b_k_stride, generic_args.rhs_strides.nth_from_end(0_i64)),
                eq(&generic_args.sum_iter_stride, DimExpr::MIter()),
                // A row-major: k=MIter, n=0, m_stride=k*MIter
                eq(a_k_stride, DimExpr::MIter()),
                eq(a_n_stride, 0),
                eq(a_m_stride, DimExpr::MIter() * &generic_args.k),
                // B column-major: k=MIter, m=0, n_stride=k*MIter
                eq(b_k_stride, DimExpr::MIter()),
                eq(b_m_stride, 0),
                eq(b_n_stride, DimExpr::MIter() * &generic_args.k),
                eq(a_batch_stride, m * a_m_stride),
                eq(b_batch_stride, n * b_n_stride),
                eq(dt, a.dtype()),
                eq(dt, b.dtype()),
                cublaslt_base_dtype(dt),
            ),
            (
                batched_matmul,
                union(sum, batched_matmul),
                set(batched_matmul.dtype(), dt),
            ),
        )
        .label("cublaslt batched row-major × column-major"),
    ]
}
