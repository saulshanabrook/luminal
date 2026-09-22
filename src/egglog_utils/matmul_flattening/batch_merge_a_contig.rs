// Batch-merge: collapse outermost two dims when A is contiguous, B is broadcast
// [d0, d1, ...] → [d0*d1, ...] via direct union
//
// Preconditions:
//   - A's outermost stride is contiguous: a_stride[0] = a_stride[1] * dim[1]
//   - B is broadcast on both leading dims: b_stride[0] = 0, b_stride[1] = 0
//   - Output strides are contiguous on the leading dims
//
// Direct union is safe because the total element count is preserved
// (d0*d1 = d0*d1) and the flat buffer layout is identical.

use crate::typed_reference::schema::*;
use egglog_experimental::typed::prelude::*;

#[ruleset]
pub(crate) fn matmul_merge_a_contiguous(
    va: &Ir,
    vb: &Ir,
    vsum: &Ir,
    vrest_sum_shape: &EList,
    vrest_mul_shape: &EList,
    vas0: &DimExpr,
    vrest_as: &EList,
    vrest_bs: &EList,
    vmos0: &DimExpr,
    vrest_mos: &EList,
    vsis0: &DimExpr,
    vrest_sis: &EList,
    vsos0: &DimExpr,
    vrest_sos: &EList,
    vdt: &DType,
    vsd0: &DimExpr,
    vsd1: &DimExpr,
    vd0: &DimExpr,
    vd1: &DimExpr,
    vas1: &DimExpr,
    vmos1: &DimExpr,
    vsis1: &DimExpr,
    vsos1: &DimExpr,
) -> Vec<Rule> {
    let product = MulArgs::fresh();
    let reduction = SumArgs::fresh();
    // Bind the complete constructor pattern as ordinary shared Rust syntax.
    // This is the source's ?mul equality, not a new egraph capture.
    let vmul = Ir::Op(product.clone(), [va, vb]);

    // Merged dimensions
    // Collapsed Mul: merged leading dim, A uses as1, B uses 0
    let vnew_mul = Ir::Op(
        OpKind::Mul(
            EList::ECons(vd0 * vd1, vrest_mul_shape),
            EList::ECons(vas1, vrest_as),
            EList::ECons(0i64, vrest_bs),
            EList::ECons(vmos1, vrest_mos),
        ),
        [va, vb],
    );

    // Collapsed Sum
    let vnew_sum = Ir::Op(
        OpKind::Sum(
            EList::ECons(vsd0 * vsd1, vrest_sum_shape),
            &reduction.iters,
            EList::ECons(vsis1, vrest_sis),
            &reduction.iter_stride,
            EList::ECons(vsos1, vrest_sos),
        ),
        [&vnew_mul],
    );
    vec![
        rule(
            (
                &vmul,
                eq(vsum, Ir::Op(reduction.clone(), [&vmul])),
                // Sum shape has >= 3 dims (result will have >= 2)
                eq(
                    &reduction.shape,
                    EList::ECons(vsd0, EList::ECons(vsd1, vrest_sum_shape)),
                ),
                ne(vrest_sum_shape, EList::ENil()),
                // Mul shape
                eq(
                    &product.shape,
                    EList::ECons(vd0, EList::ECons(vd1, vrest_mul_shape)),
                ),
                // A strides: a_s0 = a_s1 * d1 (contiguous)
                eq(
                    &product.a_strides,
                    EList::ECons(vas0, EList::ECons(vas1, vrest_as)),
                ),
                eq(vas0, vas1 * vd1),
                // B strides: broadcast on both leading dims
                eq(
                    &product.b_strides,
                    EList::ECons(0i64, EList::ECons(0i64, vrest_bs)),
                ),
                // Mul output strides: contiguous (os0 = os1 * d1)
                eq(
                    &product.out_strides,
                    EList::ECons(vmos0, EList::ECons(vmos1, vrest_mos)),
                ),
                eq(vmos0, vmos1 * vd1),
                // Sum input strides: contiguous leading
                eq(
                    &reduction.strides,
                    EList::ECons(vsis0, EList::ECons(vsis1, vrest_sis)),
                ),
                eq(vsis0, vsis1 * vsd1),
                // Sum output strides: contiguous leading
                eq(
                    &reduction.out_strides,
                    EList::ECons(vsos0, EList::ECons(vsos1, vrest_sos)),
                ),
                eq(vsos0, vsos1 * vsd1),
                eq(vdt, va.dtype()),
            ),
            (
                // Direct union
                union(vsum, &vnew_sum),
                set(vnew_mul.dtype(), vdt),
                set(vnew_sum.dtype(), vdt),
            ),
        )
        .label("batch-collapse merge A-contiguous B-broadcast"),
    ]
}
