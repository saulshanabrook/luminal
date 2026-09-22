// Squeeze: remove outermost dim=1 from Mul+Sum
// [1, d1, d2, ...] → [d1, d2, ...] via direct union
//
// When the outermost dimension of the Sum output is 1, it can always be
// removed regardless of strides (index 0 contributes 0*stride = 0 to
// every address). This enables downstream 2D matmul rules (cuBLAS) to match.
// The rule fires recursively: 4D → 3D → 2D.
//
// Direct union is safe because the total element count is preserved
// (1*N = N) and downstream ops use their own strides to index into the
// flat buffer, which has the same layout regardless of the leading dim=1.

use crate::typed_reference::schema::*;
use egglog_experimental::typed::{builtins as egg, prelude::*};

#[ruleset]
pub(crate) fn matmul_squeeze(
    va: &Ir,
    vb: &Ir,
    vsum: &Ir,
    vrest_sum_shape: &EList,
    vrest_sum_len: &egg::I64,
    vrest_mul_shape: &EList,
    vrest_as: &EList,
    vrest_bs: &EList,
    vrest_mos: &EList,
    vrest_sis: &EList,
    vrest_sos: &EList,
    vdt: &DType,
) -> Vec<Rule> {
    let product = MulArgs::fresh();
    let reduction = SumArgs::fresh();
    // Create collapsed Mul (drop outermost dim from all lists)
    let vnew_mul = Ir::Op(
        OpKind::Mul(vrest_mul_shape, vrest_as, vrest_bs, vrest_mos),
        [va, vb],
    );
    // Create collapsed Sum
    let vnew_sum = Ir::Op(
        OpKind::Sum(
            vrest_sum_shape,
            &reduction.iters,
            vrest_sis,
            &reduction.iter_stride,
            vrest_sos,
        ),
        [&vnew_mul],
    );
    let vmul = &Ir::Op(product.clone(), [va, vb]);
    vec![
        rule(
            (
                // Match the Mul + Sum pattern.
                eq(vsum, Ir::Op(reduction.clone(), [vmul])),
                // Sum output shape starts with 1
                eq(&reduction.shape, EList::ECons(1i64, vrest_sum_shape)),
                // Result must have >= 2 dims (don't collapse below 2D)
                ne(vrest_sum_shape, EList::ENil()),
                eq(vrest_sum_len, vrest_sum_shape.len()),
                vrest_sum_len.ge(2i64),
                // Mul shape also starts with 1
                eq(&product.shape, EList::ECons(1i64, vrest_mul_shape)),
                // Destructure all stride lists to drop first element
                eq(
                    &product.a_strides,
                    EConsArgs {
                        tail: vrest_as.clone(),
                        ..EConsArgs::fresh()
                    },
                ),
                eq(
                    &product.b_strides,
                    EConsArgs {
                        tail: vrest_bs.clone(),
                        ..EConsArgs::fresh()
                    },
                ),
                eq(
                    &product.out_strides,
                    EConsArgs {
                        tail: vrest_mos.clone(),
                        ..EConsArgs::fresh()
                    },
                ),
                eq(
                    &reduction.strides,
                    EConsArgs {
                        tail: vrest_sis.clone(),
                        ..EConsArgs::fresh()
                    },
                ),
                eq(
                    &reduction.out_strides,
                    EConsArgs {
                        tail: vrest_sos.clone(),
                        ..EConsArgs::fresh()
                    },
                ),
                // Get dtype
                eq(vdt, va.dtype()),
            ),
            (
                // Direct union — no wrapper needed
                union(vsum, &vnew_sum),
                // Propagate dtype
                set(vnew_mul.dtype(), vdt),
                set(vnew_sum.dtype(), vdt),
            ),
        )
        .label("batch-collapse squeeze dim=1"),
    ]
}
