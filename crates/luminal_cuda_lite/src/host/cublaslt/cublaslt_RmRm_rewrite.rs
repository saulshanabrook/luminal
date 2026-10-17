// Row-major matmul: C[m,n] = A[m,k] × B[k,n]
// A[m,k] row-major → expand to [m, n, k] with strides [k, 0, MIter]
// B[k,n] row-major → permute to [n,k] then expand to [m, n, k] with strides [0, MIter, n]
//
// Row-major viewed as column-major (swap trick):
//   Row-major A[m,k] ≡ column-major [k,m] with lda=k
//   Row-major B[k,n] ≡ column-major [n,k] with ldb=n
//   Row-major C[m,n] ≡ column-major [n,m] with ldc=n
//
// cuBLAS computes: C_col[n,m] = B_col[n,k] × A_col[k,m]
// cublasSgemm(OP_N, OP_N, n, m, k, α, B, n, A, k, β, C, n)

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
    // For row-major C = A × B with cuBLAS (column-major):
    let matmul = &Ir::Op(
        cublaslt(
            n,               // cuBLAS m = our n (swapped)
            m,               // cuBLAS n = our m (swapped)
            &generic_args.k, // k unchanged
            "N",             // transa = No transpose
            "N",             // transb = No transpose
            "COL",
            "COL",
            "COL",
            "COL",      // A/B/C/D matrix orders
            b_k_stride, // lda = B's row stride (resolves to n after z→1)
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
    // cuBLAS swap: C^T[n,m] = B^T[n,k] × A^T[k,m] per batch
    // cublas(OP_N, OP_N, n, m, k, B, lda=b_k_stride, A, ldb=a_m_stride, C, ldc=n)
    let batched_matmul = &Ir::Op(
        cublaslt(
            n,
            m,
            &generic_args.k,
            "N",
            "N",
            "COL",
            "COL",
            "COL",
            "COL",
            b_k_stride,     // lda (cuBLAS A = our B, row stride)
            a_m_stride,     // ldb (cuBLAS B = our A, row stride)
            n,              // ldc (contiguous output per batch)
            n,              // ldd
            batch,          // batch_count
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
                eq(b_m_stride, 0),
                // Assert B has strides [0, MIter, n*MIter] (row-major B[k,n] permuted to [n,k] then broadcast to [m,n,k])
                eq(b_n_stride, DimExpr::MIter()),
                eq(b_k_stride, DimExpr::MIter() * n),
                eq(dt, a.dtype()),
                eq(dt, b.dtype()),
                cublaslt_base_dtype(dt),
            ),
            (matmul, union(sum, matmul), set(matmul.dtype(), dt)),
        )
        .label("cublaslt row-major x row-major"),
        // Batched Row-major × Row-major: C[batch,m,n] = A[batch,m,k] × B[batch,k,n]
        // In broadcast [batch, m, n, k] space:
        // A row-major per batch: a_k_stride=MIter, a_n_stride=0
        // B row-major per batch: b_n_stride=MIter, b_m_stride=0
        // Leading dimensions may differ from k/n when batch slices are non-contiguous.
        rule(
            (
                eq(sum, Ir::Op(generic_args.clone(), [a, b])),
                // Output shape: [batch, m, n]
                eq(batch, generic_args.out_shape.nth_from_end(2_i64)),
                eq(m, generic_args.out_shape.nth_from_end(1_i64)),
                eq(n, generic_args.out_shape.nth_from_end(0_i64)),
                ne(m, 0),
                ne(n, 0),
                ne(&generic_args.k, 1),
                ne(batch, 0),
                // Uniform batch strides (contiguous per batch, no GQA-style repetition)
                // A strides in [batch, m, n, k]
                eq(a_batch_stride, generic_args.lhs_strides.nth_from_end(3_i64)),
                eq(a_m_stride, generic_args.lhs_strides.nth_from_end(2_i64)),
                eq(a_n_stride, generic_args.lhs_strides.nth_from_end(1_i64)),
                eq(a_k_stride, generic_args.lhs_strides.nth_from_end(0_i64)),
                // B strides in [batch, m, n, k]
                eq(b_batch_stride, generic_args.rhs_strides.nth_from_end(3_i64)),
                eq(b_m_stride, generic_args.rhs_strides.nth_from_end(2_i64)),
                eq(b_n_stride, generic_args.rhs_strides.nth_from_end(1_i64)),
                eq(b_k_stride, generic_args.rhs_strides.nth_from_end(0_i64)),
                eq(&generic_args.sum_iter_stride, DimExpr::MIter()),
                // A row-major: k=MIter, n=0, m_stride=k*MIter
                eq(a_k_stride, DimExpr::MIter()),
                eq(a_n_stride, 0),
                eq(a_m_stride, DimExpr::MIter() * &generic_args.k),
                // B row-major: n=MIter, m=0, k_stride=n*MIter
                eq(b_n_stride, DimExpr::MIter()),
                eq(b_m_stride, 0),
                eq(b_k_stride, DimExpr::MIter() * n),
                eq(a_batch_stride, m * a_m_stride),
                eq(b_batch_stride, &generic_args.k * b_k_stride),
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
        .label("cublaslt batched row-major × row-major"),
    ]
}
