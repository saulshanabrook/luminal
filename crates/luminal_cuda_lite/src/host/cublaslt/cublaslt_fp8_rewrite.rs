// FP8 support is narrower than "any FP8 x any FP8". cuBLASLt's regular FP8
// matmul table supports these A/B descriptor pairs for F32 outputs:
//   E4M3 x E4M3
//   E4M3 x E5M2
//   E5M2 x E4M3
// and requires TN format on Ada/Hopper-class GPUs. These rules therefore match
// row-major x column-major Luminal matmuls, which the existing COL-order lowering
// describes as descriptor A = logical B, descriptor B = logical A, transa=T,
// transb=N.

use super::rewrites::cublaslt_fp8_f32_output_pair;
use crate::typed_rules::*;
use egglog_experimental::typed::builtins as egg;
use egglog_experimental::typed::prelude::*;
use luminal::typed_reference::schema::*;

#[relation]
pub(crate) fn cublaslt_fp8_scale_product(a0: Ir, a1: Ir, a2: Ir);

#[ruleset]
pub(crate) fn matmul_backend(
    activation_shape: &EList,
    raw_activation: &Ir,
    recip_input_scale: &Ir,
    out_strides9: &EList,
    input_scale: &Ir,
    size: &DimExpr,
    a_dtype: &DType,
    a: &Ir,
    b: &Ir,
    size1: &DimExpr,
    cast: &Ir,
    weight_scale: &Ir,
    scale_product: &Ir,
    cast_strides: &EList,
    scaled_out_strides: &EList,
    scaled: &Ir,
    m: &DimExpr,
    n: &DimExpr,
    a_m_stride: &DimExpr,
    a_n_stride: &DimExpr,
    a_k_stride: &DimExpr,
    b_m_stride: &DimExpr,
    b_n_stride: &DimExpr,
    b_k_stride: &DimExpr,
    b_dtype: &DType,
    out_strides15: &EList,
    size4: &DimExpr,
    size5: &DimExpr,
    batch: &DimExpr,
    a_batch_stride: &DimExpr,
    b_batch_stride: &DimExpr,
    size6: &DimExpr,
    size7: &DimExpr,
    size8: &DimExpr,
    size9: &DimExpr,
    size10: &DimExpr,
    size11: &DimExpr,
    inputs: &IList,
    matmul: &Ir,
    size12: &DimExpr,
    cid: &egg::I64,
    cx: &Ir,
    cw: &Ir,
    size13: &DimExpr,
    input_scale2: &Ir,
    size14: &DimExpr,
) -> Vec<Rule> {
    let generic_args = GenericMatmulArgs::fresh();
    let sgemm = &Ir::Op(
        cublaslt_scaled(
            n,
            m,
            &generic_args.k,
            "T",
            "N",
            "COL",
            "COL",
            "COL",
            "COL",
            b_n_stride,
            a_m_stride,
            n,
            n,
            1,
            0,
            0,
            0,
            0,
            b_dtype,
            a_dtype,
            DType::F32(),
            DType::F32(),
            "32F",
            "F32",
            1.0,
            0.0,
            "DEFAULT",
        ),
        [b, a, weight_scale, input_scale],
    );
    let batched_scaled_matmul = &Ir::Op(
        cublaslt_scaled(
            n,
            m,
            &generic_args.k,
            "T",
            "N",
            "COL",
            "COL",
            "COL",
            "COL",
            b_n_stride,
            a_m_stride,
            n,
            n,
            batch,
            b_batch_stride,
            a_batch_stride,
            m * n,
            m * n,
            b_dtype,
            a_dtype,
            DType::F32(),
            DType::F32(),
            "32F",
            "F32",
            1.0,
            0.0,
            "DEFAULT",
        ),
        [b, a, weight_scale, input_scale],
    );
    let e4m3_e4m3_matmul = &Ir::Op(
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
            b_n_stride,
            a_m_stride,
            n,
            n,
            1,
            0,
            0,
            0,
            0,
            DType::F8E4M3(),
            DType::F8E4M3(),
            DType::F32(),
            DType::F32(),
            "32F",
            "F32",
            1.0,
            0.0,
            "DEFAULT",
        ),
        [b, a],
    );
    let e5m2_e4m3_matmul = &Ir::Op(
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
            b_n_stride,
            a_m_stride,
            n,
            n,
            1,
            0,
            0,
            0,
            0,
            DType::F8E5M2(),
            DType::F8E4M3(),
            DType::F32(),
            DType::F32(),
            "32F",
            "F32",
            1.0,
            0.0,
            "DEFAULT",
        ),
        [b, a],
    );
    let e4m3_e5m2_matmul = &Ir::Op(
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
            b_n_stride,
            a_m_stride,
            n,
            n,
            1,
            0,
            0,
            0,
            0,
            DType::F8E4M3(),
            DType::F8E5M2(),
            DType::F32(),
            DType::F32(),
            "32F",
            "F32",
            1.0,
            0.0,
            "DEFAULT",
        ),
        [b, a],
    );
    let batched_e4m3_e4m3_matmul = &Ir::Op(
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
            b_n_stride,
            a_m_stride,
            n,
            n,
            batch,
            b_batch_stride,
            a_batch_stride,
            m * n,
            m * n,
            DType::F8E4M3(),
            DType::F8E4M3(),
            DType::F32(),
            DType::F32(),
            "32F",
            "F32",
            1.0,
            0.0,
            "DEFAULT",
        ),
        [b, a],
    );
    let batched_e5m2_e4m3_matmul = &Ir::Op(
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
            b_n_stride,
            a_m_stride,
            n,
            n,
            batch,
            b_batch_stride,
            a_batch_stride,
            m * n,
            m * n,
            DType::F8E5M2(),
            DType::F8E4M3(),
            DType::F32(),
            DType::F32(),
            "32F",
            "F32",
            1.0,
            0.0,
            "DEFAULT",
        ),
        [b, a],
    );
    let batched_e4m3_e5m2_matmul = &Ir::Op(
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
            b_n_stride,
            a_m_stride,
            n,
            n,
            batch,
            b_batch_stride,
            a_batch_stride,
            m * n,
            m * n,
            DType::F8E4M3(),
            DType::F8E5M2(),
            DType::F32(),
            DType::F32(),
            "32F",
            "F32",
            1.0,
            0.0,
            "DEFAULT",
        ),
        [b, a],
    );
    let scaled_activation = &Ir::Op(
        MulArgs {
            shape: activation_shape.into(),
            ..MulArgs::fresh()
        },
        [raw_activation, recip_input_scale],
    );
    let sum = &Ir::Op(generic_args.clone(), [a, b]);
    vec![
        // Match the scaled FP8 linear form directly before the unscaled FP8
        // matmul rewrite can hide the quantize/dequant scale structure.
        rule(
            (
                eq(
                    recip_input_scale,
                    Ir::Op(
                        OpKind::Recip(activation_shape, [0, 0], out_strides9),
                        [input_scale],
                    ),
                ),
                eq(a, Ir::Op(OpKind::Cast(size, a_dtype), [scaled_activation])),
                eq(cast, Ir::Op(OpKind::Cast(size1, DType::F32()), [sum])),
                eq(
                    scale_product,
                    Ir::Op(
                        OpKind::Mul(EList::ENil(), EList::ENil(), EList::ENil(), EList::ENil()),
                        [input_scale, weight_scale],
                    ),
                ),
                eq(
                    scaled,
                    Ir::Op(
                        OpKind::Mul(
                            &generic_args.out_shape,
                            cast_strides,
                            [0, 0],
                            scaled_out_strides,
                        ),
                        [cast, scale_product],
                    ),
                ),
                eq(cast_strides, scaled_out_strides),
                eq(&generic_args.out_shape, [m, n]),
                ne(m, 0),
                ne(n, 0),
                ne(&generic_args.k, 1),
                eq(
                    &generic_args.lhs_strides,
                    [a_m_stride, a_n_stride, a_k_stride],
                ),
                eq(
                    &generic_args.rhs_strides,
                    [b_m_stride, b_n_stride, b_k_stride],
                ),
                eq(&generic_args.sum_iter_stride, DimExpr::MIter()),
                eq(a_m_stride, DimExpr::MIter() * &generic_args.k),
                eq(a_n_stride, 0),
                eq(a_k_stride, DimExpr::MIter()),
                eq(b_m_stride, 0),
                eq(b_n_stride, DimExpr::MIter() * &generic_args.k),
                eq(b_k_stride, DimExpr::MIter()),
                eq(b_dtype, b.dtype()),
                cublaslt_fp8_f32_output_pair(a_dtype, b_dtype),
            ),
            (
                sgemm,
                union(scaled, sgemm),
                set(sgemm.dtype(), DType::F32()),
            ),
        )
        .label("cublaslt scaled fp8 row-major x column-major f32 output"),
        // Batched form of the scaled FP8 linear rewrite. The scale operands are
        // scalar tensors expanded across the last three output/activation axes.
        rule(
            (
                eq(
                    recip_input_scale,
                    Ir::Op(
                        OpKind::Recip(activation_shape, [0, 0, 0], out_strides15),
                        [input_scale],
                    ),
                ),
                eq(a, Ir::Op(OpKind::Cast(size4, a_dtype), [scaled_activation])),
                eq(cast, Ir::Op(OpKind::Cast(size5, DType::F32()), [sum])),
                eq(
                    scale_product,
                    Ir::Op(
                        OpKind::Mul(EList::ENil(), EList::ENil(), EList::ENil(), EList::ENil()),
                        [input_scale, weight_scale],
                    ),
                ),
                eq(
                    scaled,
                    Ir::Op(
                        OpKind::Mul(
                            &generic_args.out_shape,
                            cast_strides,
                            [0, 0, 0],
                            scaled_out_strides,
                        ),
                        [cast, scale_product],
                    ),
                ),
                eq(cast_strides, scaled_out_strides),
                eq(batch, generic_args.out_shape.nth_from_end(2_i64)),
                eq(m, generic_args.out_shape.nth_from_end(1_i64)),
                eq(n, generic_args.out_shape.nth_from_end(0_i64)),
                ne(m, 0),
                ne(n, 0),
                ne(&generic_args.k, 1),
                ne(batch, 0),
                eq(a_batch_stride, generic_args.lhs_strides.nth_from_end(3_i64)),
                eq(a_m_stride, generic_args.lhs_strides.nth_from_end(2_i64)),
                eq(a_n_stride, generic_args.lhs_strides.nth_from_end(1_i64)),
                eq(a_k_stride, generic_args.lhs_strides.nth_from_end(0_i64)),
                eq(b_batch_stride, generic_args.rhs_strides.nth_from_end(3_i64)),
                eq(b_m_stride, generic_args.rhs_strides.nth_from_end(2_i64)),
                eq(b_n_stride, generic_args.rhs_strides.nth_from_end(1_i64)),
                eq(b_k_stride, generic_args.rhs_strides.nth_from_end(0_i64)),
                eq(&generic_args.sum_iter_stride, DimExpr::MIter()),
                eq(a_k_stride, DimExpr::MIter()),
                eq(a_n_stride, 0),
                eq(a_m_stride, DimExpr::MIter() * &generic_args.k),
                eq(b_k_stride, DimExpr::MIter()),
                eq(b_m_stride, 0),
                eq(b_n_stride, DimExpr::MIter() * &generic_args.k),
                eq(a_batch_stride, m * a_m_stride),
                eq(b_batch_stride, n * b_n_stride),
                eq(b_dtype, b.dtype()),
                cublaslt_fp8_f32_output_pair(a_dtype, b_dtype),
            ),
            (
                batched_scaled_matmul,
                union(scaled, batched_scaled_matmul),
                set(batched_scaled_matmul.dtype(), DType::F32()),
            ),
        )
        .label("cublaslt scaled fp8 batched row-major x column-major f32 output"),
        rule(
            (
                eq(cast, Ir::Op(OpKind::Cast(size6, DType::F32()), [sum])),
                eq(&generic_args.out_shape, [m, n]),
                ne(m, 0),
                ne(n, 0),
                ne(&generic_args.k, 1),
                eq(
                    &generic_args.lhs_strides,
                    [a_m_stride, a_n_stride, a_k_stride],
                ),
                eq(
                    &generic_args.rhs_strides,
                    [b_m_stride, b_n_stride, b_k_stride],
                ),
                eq(&generic_args.sum_iter_stride, DimExpr::MIter()),
                eq(a_m_stride, DimExpr::MIter() * &generic_args.k),
                eq(a_n_stride, 0),
                eq(a_k_stride, DimExpr::MIter()),
                eq(b_m_stride, 0),
                eq(b_n_stride, DimExpr::MIter() * &generic_args.k),
                eq(b_k_stride, DimExpr::MIter()),
                eq(DType::F8E4M3(), a.dtype()),
                eq(DType::F8E4M3(), b.dtype()),
            ),
            (
                e4m3_e4m3_matmul,
                union(cast, e4m3_e4m3_matmul),
                set(e4m3_e4m3_matmul.dtype(), DType::F32()),
            ),
        )
        .label("cublaslt fp8 e4m3/e4m3 row-major x column-major f32 output"),
        rule(
            (
                eq(cast, Ir::Op(OpKind::Cast(size7, DType::F32()), [sum])),
                eq(&generic_args.out_shape, [m, n]),
                ne(m, 0),
                ne(n, 0),
                ne(&generic_args.k, 1),
                eq(
                    &generic_args.lhs_strides,
                    [a_m_stride, a_n_stride, a_k_stride],
                ),
                eq(
                    &generic_args.rhs_strides,
                    [b_m_stride, b_n_stride, b_k_stride],
                ),
                eq(&generic_args.sum_iter_stride, DimExpr::MIter()),
                eq(a_m_stride, DimExpr::MIter() * &generic_args.k),
                eq(a_n_stride, 0),
                eq(a_k_stride, DimExpr::MIter()),
                eq(b_m_stride, 0),
                eq(b_n_stride, DimExpr::MIter() * &generic_args.k),
                eq(b_k_stride, DimExpr::MIter()),
                eq(DType::F8E4M3(), a.dtype()),
                eq(DType::F8E5M2(), b.dtype()),
            ),
            (
                e5m2_e4m3_matmul,
                union(cast, e5m2_e4m3_matmul),
                set(e5m2_e4m3_matmul.dtype(), DType::F32()),
            ),
        )
        .label("cublaslt fp8 e5m2/e4m3 row-major x column-major f32 output"),
        rule(
            (
                eq(cast, Ir::Op(OpKind::Cast(size8, DType::F32()), [sum])),
                eq(&generic_args.out_shape, [m, n]),
                ne(m, 0),
                ne(n, 0),
                ne(&generic_args.k, 1),
                eq(
                    &generic_args.lhs_strides,
                    [a_m_stride, a_n_stride, a_k_stride],
                ),
                eq(
                    &generic_args.rhs_strides,
                    [b_m_stride, b_n_stride, b_k_stride],
                ),
                eq(&generic_args.sum_iter_stride, DimExpr::MIter()),
                eq(a_m_stride, DimExpr::MIter() * &generic_args.k),
                eq(a_n_stride, 0),
                eq(a_k_stride, DimExpr::MIter()),
                eq(b_m_stride, 0),
                eq(b_n_stride, DimExpr::MIter() * &generic_args.k),
                eq(b_k_stride, DimExpr::MIter()),
                eq(DType::F8E5M2(), a.dtype()),
                eq(DType::F8E4M3(), b.dtype()),
            ),
            (
                e4m3_e5m2_matmul,
                union(cast, e4m3_e5m2_matmul),
                set(e4m3_e5m2_matmul.dtype(), DType::F32()),
            ),
        )
        .label("cublaslt fp8 e4m3/e5m2 row-major x column-major f32 output"),
        rule(
            (
                eq(cast, Ir::Op(OpKind::Cast(size9, DType::F32()), [sum])),
                eq(batch, generic_args.out_shape.nth_from_end(2_i64)),
                eq(m, generic_args.out_shape.nth_from_end(1_i64)),
                eq(n, generic_args.out_shape.nth_from_end(0_i64)),
                ne(m, 0),
                ne(n, 0),
                ne(&generic_args.k, 1),
                ne(batch, 0),
                eq(a_batch_stride, generic_args.lhs_strides.nth_from_end(3_i64)),
                eq(a_m_stride, generic_args.lhs_strides.nth_from_end(2_i64)),
                eq(a_n_stride, generic_args.lhs_strides.nth_from_end(1_i64)),
                eq(a_k_stride, generic_args.lhs_strides.nth_from_end(0_i64)),
                eq(b_batch_stride, generic_args.rhs_strides.nth_from_end(3_i64)),
                eq(b_m_stride, generic_args.rhs_strides.nth_from_end(2_i64)),
                eq(b_n_stride, generic_args.rhs_strides.nth_from_end(1_i64)),
                eq(b_k_stride, generic_args.rhs_strides.nth_from_end(0_i64)),
                eq(&generic_args.sum_iter_stride, DimExpr::MIter()),
                eq(a_k_stride, DimExpr::MIter()),
                eq(a_n_stride, 0),
                eq(a_m_stride, DimExpr::MIter() * &generic_args.k),
                eq(b_k_stride, DimExpr::MIter()),
                eq(b_m_stride, 0),
                eq(b_n_stride, DimExpr::MIter() * &generic_args.k),
                eq(a_batch_stride, m * a_m_stride),
                eq(b_batch_stride, n * b_n_stride),
                eq(DType::F8E4M3(), a.dtype()),
                eq(DType::F8E4M3(), b.dtype()),
            ),
            (
                batched_e4m3_e4m3_matmul,
                union(cast, batched_e4m3_e4m3_matmul),
                set(batched_e4m3_e4m3_matmul.dtype(), DType::F32()),
            ),
        )
        .label("cublaslt fp8 e4m3/e4m3 batched row-major x column-major f32 output"),
        rule(
            (
                eq(cast, Ir::Op(OpKind::Cast(size10, DType::F32()), [sum])),
                eq(batch, generic_args.out_shape.nth_from_end(2_i64)),
                eq(m, generic_args.out_shape.nth_from_end(1_i64)),
                eq(n, generic_args.out_shape.nth_from_end(0_i64)),
                ne(m, 0),
                ne(n, 0),
                ne(&generic_args.k, 1),
                ne(batch, 0),
                eq(a_batch_stride, generic_args.lhs_strides.nth_from_end(3_i64)),
                eq(a_m_stride, generic_args.lhs_strides.nth_from_end(2_i64)),
                eq(a_n_stride, generic_args.lhs_strides.nth_from_end(1_i64)),
                eq(a_k_stride, generic_args.lhs_strides.nth_from_end(0_i64)),
                eq(b_batch_stride, generic_args.rhs_strides.nth_from_end(3_i64)),
                eq(b_m_stride, generic_args.rhs_strides.nth_from_end(2_i64)),
                eq(b_n_stride, generic_args.rhs_strides.nth_from_end(1_i64)),
                eq(b_k_stride, generic_args.rhs_strides.nth_from_end(0_i64)),
                eq(&generic_args.sum_iter_stride, DimExpr::MIter()),
                eq(a_k_stride, DimExpr::MIter()),
                eq(a_n_stride, 0),
                eq(a_m_stride, DimExpr::MIter() * &generic_args.k),
                eq(b_k_stride, DimExpr::MIter()),
                eq(b_m_stride, 0),
                eq(b_n_stride, DimExpr::MIter() * &generic_args.k),
                eq(a_batch_stride, m * a_m_stride),
                eq(b_batch_stride, n * b_n_stride),
                eq(DType::F8E4M3(), a.dtype()),
                eq(DType::F8E5M2(), b.dtype()),
            ),
            (
                batched_e5m2_e4m3_matmul,
                union(cast, batched_e5m2_e4m3_matmul),
                set(batched_e5m2_e4m3_matmul.dtype(), DType::F32()),
            ),
        )
        .label("cublaslt fp8 e5m2/e4m3 batched row-major x column-major f32 output"),
        rule(
            (
                eq(cast, Ir::Op(OpKind::Cast(size11, DType::F32()), [sum])),
                eq(batch, generic_args.out_shape.nth_from_end(2_i64)),
                eq(m, generic_args.out_shape.nth_from_end(1_i64)),
                eq(n, generic_args.out_shape.nth_from_end(0_i64)),
                ne(m, 0),
                ne(n, 0),
                ne(&generic_args.k, 1),
                ne(batch, 0),
                eq(a_batch_stride, generic_args.lhs_strides.nth_from_end(3_i64)),
                eq(a_m_stride, generic_args.lhs_strides.nth_from_end(2_i64)),
                eq(a_n_stride, generic_args.lhs_strides.nth_from_end(1_i64)),
                eq(a_k_stride, generic_args.lhs_strides.nth_from_end(0_i64)),
                eq(b_batch_stride, generic_args.rhs_strides.nth_from_end(3_i64)),
                eq(b_m_stride, generic_args.rhs_strides.nth_from_end(2_i64)),
                eq(b_n_stride, generic_args.rhs_strides.nth_from_end(1_i64)),
                eq(b_k_stride, generic_args.rhs_strides.nth_from_end(0_i64)),
                eq(&generic_args.sum_iter_stride, DimExpr::MIter()),
                eq(a_k_stride, DimExpr::MIter()),
                eq(a_n_stride, 0),
                eq(a_m_stride, DimExpr::MIter() * &generic_args.k),
                eq(b_k_stride, DimExpr::MIter()),
                eq(b_m_stride, 0),
                eq(b_n_stride, DimExpr::MIter() * &generic_args.k),
                eq(a_batch_stride, m * a_m_stride),
                eq(b_batch_stride, n * b_n_stride),
                eq(DType::F8E5M2(), a.dtype()),
                eq(DType::F8E4M3(), b.dtype()),
            ),
            (
                batched_e4m3_e5m2_matmul,
                union(cast, batched_e4m3_e5m2_matmul),
                set(batched_e4m3_e5m2_matmul.dtype(), DType::F32()),
            ),
        )
        .label("cublaslt fp8 e4m3/e5m2 batched row-major x column-major f32 output"),
        // Fold a trailing Cast(Bf16) into a scaled FP8 matmul's output dtype.
        // cuBLASLt supports BF16 C/D with FP8 A/B; beta is pinned to 0.0 so no C
        // operand exists whose physical dtype could disagree (same hazard as the
        // mixed-dtype fold guard).
        {
            let fp32_args = CuBlasLtScaledArgs {
                c_dtype: DType::F32(),
                d_dtype: DType::F32(),
                beta: 0.0.into(),
                ..CuBlasLtScaledArgs::fresh()
            };
            let bf16_matmul = &Ir::Op(
                CuBlasLtScaledArgs {
                    c_dtype: DType::Bf16(),
                    d_dtype: DType::Bf16(),
                    ..fp32_args.clone()
                },
                inputs,
            );
            rule(
                (
                    eq(matmul, Ir::Op(fp32_args.clone(), inputs)),
                    eq(cast, Ir::Op(OpKind::Cast(size12, DType::Bf16()), [matmul])),
                ),
                (
                    bf16_matmul,
                    union(cast, bf16_matmul),
                    set(bf16_matmul.dtype(), DType::Bf16()),
                ),
            )
            .label("cublaslt scaled fp8 bf16 output")
        },
        // Variant of the scaled FP8 rule for a pre-quantized activation produced by a
        // fused F8E4M3 custom op (rms-norm-quant custom (x, w, scale)). CONVENTION: f8-producing
        // custom ops put the quantization input scale as their LAST input; the rule
        // matches by arity and reads the scale from that slot. The dequant chain on
        // the output side is unchanged.
        rule(
            (
                eq(
                    a,
                    Ir::Op(
                        OpKind::CustomOpKind(cid, DType::F8E4M3()),
                        [cx, cw, input_scale],
                    ),
                ),
                eq(a_dtype, a.dtype()),
                eq(cast, Ir::Op(OpKind::Cast(size13, DType::F32()), [sum])),
                eq(
                    scale_product,
                    Ir::Op(
                        OpKind::Mul(EList::ENil(), EList::ENil(), EList::ENil(), EList::ENil()),
                        [input_scale2, weight_scale],
                    ),
                ),
                eq(input_scale, input_scale2),
                eq(
                    scaled,
                    Ir::Op(
                        OpKind::Mul(
                            &generic_args.out_shape,
                            cast_strides,
                            [0, 0],
                            scaled_out_strides,
                        ),
                        [cast, scale_product],
                    ),
                ),
                eq(cast_strides, scaled_out_strides),
                eq(&generic_args.out_shape, [m, n]),
                ne(m, 0),
                ne(n, 0),
                ne(&generic_args.k, 1),
                eq(
                    &generic_args.lhs_strides,
                    [a_m_stride, a_n_stride, a_k_stride],
                ),
                eq(
                    &generic_args.rhs_strides,
                    [b_m_stride, b_n_stride, b_k_stride],
                ),
                eq(&generic_args.sum_iter_stride, DimExpr::MIter()),
                eq(a_m_stride, DimExpr::MIter() * &generic_args.k),
                eq(a_n_stride, 0),
                eq(a_k_stride, DimExpr::MIter()),
                eq(b_m_stride, 0),
                eq(b_n_stride, DimExpr::MIter() * &generic_args.k),
                eq(b_k_stride, DimExpr::MIter()),
                eq(b_dtype, b.dtype()),
                cublaslt_fp8_f32_output_pair(a_dtype, b_dtype),
            ),
            (
                sgemm,
                union(scaled, sgemm),
                set(sgemm.dtype(), DType::F32()),
            ),
        )
        .label("cublaslt scaled fp8 RmCm f32 output, custom quantized activation arity 3"),
        // Variant of the scaled FP8 rule for a pre-quantized activation produced by a
        // fused F8E4M3 custom op (swiglu-quant custom (x, scale)). CONVENTION: f8-producing
        // custom ops put the quantization input scale as their LAST input; the rule
        // matches by arity and reads the scale from that slot. The dequant chain on
        // the output side is unchanged.
        rule(
            (
                eq(
                    a,
                    Ir::Op(
                        OpKind::CustomOpKind(cid, DType::F8E4M3()),
                        [cx, input_scale],
                    ),
                ),
                eq(a_dtype, a.dtype()),
                eq(cast, Ir::Op(OpKind::Cast(size14, DType::F32()), [sum])),
                eq(
                    scale_product,
                    Ir::Op(
                        OpKind::Mul(EList::ENil(), EList::ENil(), EList::ENil(), EList::ENil()),
                        [input_scale2, weight_scale],
                    ),
                ),
                eq(input_scale, input_scale2),
                eq(
                    scaled,
                    Ir::Op(
                        OpKind::Mul(
                            &generic_args.out_shape,
                            cast_strides,
                            [0, 0],
                            scaled_out_strides,
                        ),
                        [cast, scale_product],
                    ),
                ),
                eq(cast_strides, scaled_out_strides),
                eq(&generic_args.out_shape, [m, n]),
                ne(m, 0),
                ne(n, 0),
                ne(&generic_args.k, 1),
                eq(
                    &generic_args.lhs_strides,
                    [a_m_stride, a_n_stride, a_k_stride],
                ),
                eq(
                    &generic_args.rhs_strides,
                    [b_m_stride, b_n_stride, b_k_stride],
                ),
                eq(&generic_args.sum_iter_stride, DimExpr::MIter()),
                eq(a_m_stride, DimExpr::MIter() * &generic_args.k),
                eq(a_n_stride, 0),
                eq(a_k_stride, DimExpr::MIter()),
                eq(b_m_stride, 0),
                eq(b_n_stride, DimExpr::MIter() * &generic_args.k),
                eq(b_k_stride, DimExpr::MIter()),
                eq(b_dtype, b.dtype()),
                cublaslt_fp8_f32_output_pair(a_dtype, b_dtype),
            ),
            (
                sgemm,
                union(scaled, sgemm),
                set(sgemm.dtype(), DType::F32()),
            ),
        )
        .label("cublaslt scaled fp8 RmCm f32 output, custom quantized activation arity 2"),
    ]
}

#[ruleset]
pub(crate) fn cleanup(
    activation_shape: &EList,
    raw_activation: &Ir,
    out_strides12: &EList,
    input_scale: &Ir,
    size2: &DimExpr,
    a_dtype: &DType,
    a: &Ir,
    b: &Ir,
    size3: &DimExpr,
    cast: &Ir,
    weight_scale: &Ir,
    scale_product: &Ir,
    cast_strides: &EList,
    scaled_out_strides: &EList,
    scaled: &Ir,
    m: &DimExpr,
    n: &DimExpr,
    a_m_stride: &DimExpr,
    a_n_stride: &DimExpr,
    a_k_stride: &DimExpr,
    b_m_stride: &DimExpr,
    b_n_stride: &DimExpr,
    b_k_stride: &DimExpr,
    b_dtype: &DType,
    cm: &DimExpr,
    cn: &DimExpr,
    ck: &DimExpr,
    cta: &egg::String,
    ctb: &egg::String,
    cao: &egg::String,
    cbo: &egg::String,
    cco: &egg::String,
    cdo: &egg::String,
    clda: &DimExpr,
    cldb: &DimExpr,
    cldc: &DimExpr,
    cldd: &DimExpr,
    cbc: &DimExpr,
    csa: &DimExpr,
    csb: &DimExpr,
    csc: &DimExpr,
    csd: &DimExpr,
    cadt: &DType,
    cbdt: &DType,
    ccdt: &DType,
    cddt: &DType,
    ccompute: &egg::String,
    cscale: &egg::String,
    calpha: &egg::F64,
    cbeta: &egg::F64,
    cepilogue: &egg::String,
    raw_gemm: &Ir,
    fs_cast: &Ir,
    out_m: &DimExpr,
    out_n: &DimExpr,
    scale_strides: &EList,
    a_scale: &Ir,
    b_scale: &Ir,
    fs_scale: &Ir,
    fused_scale: &Ir,
) -> Vec<Rule> {
    let generic_args = GenericMatmulArgs::fresh();
    let raw_matmul = &Ir::Op(
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
            b_n_stride,
            a_m_stride,
            n,
            n,
            1,
            0,
            0,
            0,
            0,
            b_dtype,
            a_dtype,
            DType::F32(),
            DType::F32(),
            "32F",
            "F32",
            1.0,
            0.0,
            "DEFAULT",
        ),
        [b, a],
    );
    let scaled_matmul = &Ir::Op(
        OpKind::Mul(
            &generic_args.out_shape,
            cast_strides,
            [0, 0],
            scaled_out_strides,
        ),
        [cast, scale_product],
    );
    let raw_fused_matmul = &Ir::Op(
        cuda_binary_elementwise(
            "Mul",
            &generic_args.out_shape,
            cast_strides,
            scale_strides,
            scaled_out_strides,
            DType::F32(),
        ),
        [fs_cast, fs_scale],
    );
    let scaled_fused_matmul = &Ir::Op(
        cublaslt(
            cm, cn, ck, cta, ctb, cao, cbo, cco, cdo, clda, cldb, cldc, cldd, cbc, csa, csb, csc,
            csd, cadt, cbdt, ccdt, cddt, ccompute, cscale, calpha, cbeta, cepilogue,
        ),
        [a, b],
    );
    let recip_input_scale = &Ir::Op(
        OpKind::Recip(activation_shape, [0, 0], out_strides12),
        [input_scale],
    );
    let sum = &Ir::Op(generic_args.clone(), [a, b]);
    let fs_a_scale = &Ir::Op(
        fusion_start(EList::ENil(), EList::ENil(), DType::F32()),
        [a_scale],
    );
    let fs_b_scale = &Ir::Op(
        fusion_start(EList::ENil(), EList::ENil(), DType::F32()),
        [b_scale],
    );
    let sgemm = &Ir::Op(
        cublaslt_scaled(
            cm, cn, ck, cta, ctb, cao, cbo, cco, cdo, clda, cldb, cldc, cldd, cbc, csa, csb, csc,
            csd, cadt, cbdt, ccdt, cddt, ccompute, cscale, calpha, cbeta, cepilogue,
        ),
        [a, b, a_scale, b_scale],
    );
    let scaled_activation = &Ir::Op(
        MulArgs {
            shape: activation_shape.into(),
            ..MulArgs::fresh()
        },
        [raw_activation, recip_input_scale],
    );
    let scale_product_inner = &Ir::Op(
        cuda_binary_elementwise(
            "Mul",
            EList::ENil(),
            EList::ENil(),
            EList::ENil(),
            EList::ENil(),
            DType::F32(),
        ),
        [fs_a_scale, fs_b_scale],
    );
    vec![
        rule(
            (
                eq(a, Ir::Op(OpKind::Cast(size2, a_dtype), [scaled_activation])),
                eq(cast, Ir::Op(OpKind::Cast(size3, DType::F32()), [sum])),
                eq(
                    scale_product,
                    Ir::Op(
                        OpKind::Mul(EList::ENil(), EList::ENil(), EList::ENil(), EList::ENil()),
                        [input_scale, weight_scale],
                    ),
                ),
                eq(scaled, scaled_matmul),
                eq(cast_strides, scaled_out_strides),
                eq(&generic_args.out_shape, [m, n]),
                ne(m, 0),
                ne(n, 0),
                ne(&generic_args.k, 1),
                eq(
                    &generic_args.lhs_strides,
                    [a_m_stride, a_n_stride, a_k_stride],
                ),
                eq(
                    &generic_args.rhs_strides,
                    [b_m_stride, b_n_stride, b_k_stride],
                ),
                eq(&generic_args.sum_iter_stride, DimExpr::MIter()),
                eq(a_m_stride, DimExpr::MIter() * &generic_args.k),
                eq(a_n_stride, 0),
                eq(a_k_stride, DimExpr::MIter()),
                eq(b_m_stride, 0),
                eq(b_n_stride, DimExpr::MIter() * &generic_args.k),
                eq(b_k_stride, DimExpr::MIter()),
                eq(b_dtype, b.dtype()),
                cublaslt_fp8_f32_output_pair(a_dtype, b_dtype),
                eq(
                    scaled,
                    Ir::Op(
                        cublaslt_scaled(
                            n,
                            m,
                            &generic_args.k,
                            "T",
                            "N",
                            "COL",
                            "COL",
                            "COL",
                            "COL",
                            b_n_stride,
                            a_m_stride,
                            n,
                            n,
                            1,
                            0,
                            0,
                            0,
                            0,
                            b_dtype,
                            a_dtype,
                            DType::F32(),
                            DType::F32(),
                            "32F",
                            "F32",
                            1.0,
                            0.0,
                            "DEFAULT",
                        ),
                        [b, a, weight_scale, input_scale],
                    ),
                ),
                eq(cast, raw_matmul),
            ),
            (delete(scaled_matmul), delete(raw_matmul)),
        )
        .label("delete raw fp8 path when scaled cublaslt covers direct output scale"),
        rule(
            (
                eq(raw_gemm, scaled_fused_matmul),
                cublaslt_fp8_f32_output_pair(cadt, cbdt),
                eq(ccdt, DType::F32()),
                eq(cddt, DType::F32()),
                eq(cbeta, 0.0),
                eq(cepilogue, "DEFAULT"),
                eq(
                    fs_cast,
                    Ir::Op(
                        fusion_start(&generic_args.out_shape, cast_strides, DType::F32()),
                        [raw_gemm],
                    ),
                ),
                eq(&generic_args.out_shape, [out_m, out_n]),
                eq(scale_strides, [0, 0]),
                eq(
                    scale_product,
                    Ir::Op(
                        fusion_end(EList::ENil(), EList::ENil(), DType::F32()),
                        [scale_product_inner],
                    ),
                ),
                eq(
                    fs_scale,
                    Ir::Op(
                        fusion_start(&generic_args.out_shape, scale_strides, DType::F32()),
                        [scale_product],
                    ),
                ),
                eq(fused_scale, raw_fused_matmul),
                eq(cast_strides, scaled_out_strides),
                eq(
                    fused_scale,
                    Ir::Op(
                        fusion_start(&generic_args.out_shape, scaled_out_strides, DType::F32()),
                        [sgemm],
                    ),
                ),
            ),
            (delete(scaled_fused_matmul), delete(raw_fused_matmul)),
        )
        .label("delete raw fp8 path when scaled cublaslt covers fused output scale"),
    ]
}

#[ruleset]
pub(crate) fn fusion_grow(
    a_scale: &Ir,
    b_scale: &Ir,
    scale_product: &Ir,
    cm: &DimExpr,
    cn: &DimExpr,
    ck: &DimExpr,
    cta: &egg::String,
    ctb: &egg::String,
    cao: &egg::String,
    cbo: &egg::String,
    cco: &egg::String,
    cdo: &egg::String,
    clda: &DimExpr,
    cldb: &DimExpr,
    cldc: &DimExpr,
    cldd: &DimExpr,
    cbc: &DimExpr,
    csa: &DimExpr,
    csb: &DimExpr,
    csc: &DimExpr,
    csd: &DimExpr,
    cadt: &DType,
    cbdt: &DType,
    ccdt: &DType,
    cddt: &DType,
    ccompute: &egg::String,
    cscale: &egg::String,
    calpha: &egg::F64,
    cbeta: &egg::F64,
    cepilogue: &egg::String,
    a: &Ir,
    b: &Ir,
    out_shape: &EList,
    cast_strides: &EList,
    out_m: &DimExpr,
    out_n: &DimExpr,
    scale_strides: &EList,
    scaled_out_strides: &EList,
    fused_scale: &Ir,
) -> Vec<Rule> {
    let fused_scaled_matmul = &Ir::Op(
        cublaslt_scaled(
            cm, cn, ck, cta, ctb, cao, cbo, cco, cdo, clda, cldb, cldc, cldd, cbc, csa, csb, csc,
            csd, cadt, cbdt, ccdt, cddt, ccompute, cscale, calpha, cbeta, cepilogue,
        ),
        [a, b, a_scale, b_scale],
    );
    let fused_scaled_matmul_start = &Ir::Op(
        fusion_start(out_shape, scaled_out_strides, DType::F32()),
        [fused_scaled_matmul],
    );
    let fs_a_scale = &Ir::Op(
        fusion_start(EList::ENil(), EList::ENil(), DType::F32()),
        [a_scale],
    );
    let fs_b_scale = &Ir::Op(
        fusion_start(EList::ENil(), EList::ENil(), DType::F32()),
        [b_scale],
    );
    let raw_gemm = &Ir::Op(
        cublaslt(
            cm, cn, ck, cta, ctb, cao, cbo, cco, cdo, clda, cldb, cldc, cldd, cbc, csa, csb, csc,
            csd, cadt, cbdt, ccdt, cddt, ccompute, cscale, calpha, cbeta, cepilogue,
        ),
        [a, b],
    );
    let fs_scale = &Ir::Op(
        fusion_start(out_shape, scale_strides, DType::F32()),
        [scale_product],
    );
    let scale_product_inner = &Ir::Op(
        cuda_binary_elementwise(
            "Mul",
            EList::ENil(),
            EList::ENil(),
            EList::ENil(),
            EList::ENil(),
            DType::F32(),
        ),
        [fs_a_scale, fs_b_scale],
    );
    let fs_cast = &Ir::Op(
        fusion_start(out_shape, cast_strides, DType::F32()),
        [raw_gemm],
    );
    vec![
        // Stage the tensor-wide fp8 scale product (a_scale · b_scale), keyed
        // by its FusionEnd. ANCHORED at the scalar Mul: its two scalar
        // FusionStart operands are read from the Mul's input list, never
        // enumerated. Without this fact, the fused-output-scale rule below
        // matched ?fs_a_scale / ?fs_b_scale as free scalar-FusionStart atoms
        // and egglog enumerated their K² cross-product on graphs with many
        // scalar fusion boundaries (adaLN modulation + softmax) — a
        // super-linear compile-time blowup that OOMs the real FLUX compile.
        rule(
            (eq(
                scale_product,
                Ir::Op(
                    fusion_end(EList::ENil(), EList::ENil(), DType::F32()),
                    [scale_product_inner],
                ),
            ),),
            (cublaslt_fp8_scale_product(scale_product, a_scale, b_scale),),
        )
        .label("cublaslt fp8 scale-product fact"),
        // Fusion growth can make the live path consume a raw FP8 cuBLASLt
        // candidate through an internal CudaBinaryElementwise scale multiply,
        // instead of the original HLIR output-scale Mul. The scalar scale
        // product is tensor-wide, so the two scalar factors can be passed as
        // cuBLASLt A/B scale inputs and the internal multiply can be bypassed.
        rule(
            (
                cublaslt_fp8_f32_output_pair(cadt, cbdt),
                eq(ccdt, DType::F32()),
                eq(cddt, DType::F32()),
                eq(cbeta, 0.0),
                eq(cepilogue, "DEFAULT"),
                eq(out_shape, [out_m, out_n]),
                eq(scale_strides, [0, 0]),
                cublaslt_fp8_scale_product(scale_product, a_scale, b_scale),
                // ?fused_scale = Mul(fs_cast, fs_scale) is the rare anchor (one per
                // fp8 gemm). fs_scale wraps the scale-product FusionEnd; the staged
                // fact yields its two scalar operands without re-matching the scalar
                // FusionStart pair (see "cublaslt fp8 scale-product fact").
                eq(
                    fused_scale,
                    Ir::Op(
                        cuda_binary_elementwise(
                            "Mul",
                            out_shape,
                            cast_strides,
                            scale_strides,
                            scaled_out_strides,
                            DType::F32(),
                        ),
                        [fs_cast, fs_scale],
                    ),
                ),
                eq(cast_strides, scaled_out_strides),
            ),
            (
                fused_scaled_matmul,
                fused_scaled_matmul_start,
                union(fused_scale, fused_scaled_matmul_start),
                set(fused_scaled_matmul.dtype(), DType::F32()),
                set(fused_scaled_matmul_start.dtype(), DType::F32()),
            ),
        )
        .label("cublaslt scaled fp8 fused output-scale f32 output"),
    ]
}
