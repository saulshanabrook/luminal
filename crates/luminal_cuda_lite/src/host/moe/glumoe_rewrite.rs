// GLUMoE: Match the expert computation subgraph of a gated MoE.
//
// One fused op supports two activation modes:
//   mode=0: Qwen-style SwiGLU (silu(gate) * up)
//   mode=1: Gemma-style GELU (gate * sigmoid(1.595769 * gate * (1 + 0.044715 * gate^2)))
//
// To keep matching fast, we stage through marker states:
//   1) Shared expert index/gather markers
//   2) Shared gate-up matmul marker
//   3) Activation marker (separate swiglu / gemma_gelu paths)
//   4) Down matmul marker (separate swiglu / gemma_gelu paths)
//   5) Final GLUMoE fusion (separate swiglu / gemma_gelu rules)

use crate::typed_rules::*;
use egglog_experimental::typed::prelude::*;
use luminal::typed_reference::schema::*;

#[sort]
pub(crate) struct GLUMoEExpertIndexState;

#[sort]
pub(crate) struct GLUMoEExpertGatherState;

#[sort]
pub(crate) struct GLUMoEGateUpState;

#[sort]
pub(crate) struct GLUMoESwiGLUState;

#[sort]
pub(crate) struct GLUMoEGemmaGELUState;

#[sort]
pub(crate) struct GLUMoESwiGLUDownState;

#[sort]
pub(crate) struct GLUMoEGemmaDownState;

#[constructor]
pub(crate) fn mk_glu_mo_e_expert_index_state(
    a0: DimExpr,
    a1: DimExpr,
    a2: Ir,
) -> GLUMoEExpertIndexState;

#[constructor]
pub(crate) fn mk_glu_mo_e_expert_gather_state(
    a0: DimExpr,
    a1: DimExpr,
    a2: Ir,
    a3: Ir,
) -> GLUMoEExpertGatherState;

#[constructor(args = GLUMoEGateUpStateArgs)]
pub(crate) fn mk_glu_mo_e_gate_up_state(
    a0: DimExpr,
    a1: DimExpr,
    a2: DimExpr,
    a3: Ir,
    a4: Ir,
    a5: Ir,
) -> GLUMoEGateUpState;

#[constructor]
pub(crate) fn mk_glu_mo_e_swi_glu_state(a0: GLUMoEGateUpState) -> GLUMoESwiGLUState;

#[constructor]
pub(crate) fn mk_glu_mo_e_gemma_gelu_state(a0: GLUMoEGateUpState) -> GLUMoEGemmaGELUState;

#[constructor]
pub(crate) fn mk_glu_mo_e_swi_glu_down_state(
    a0: DimExpr,
    a1: DimExpr,
    a2: DimExpr,
    a3: GLUMoESwiGLUState,
    a4: Ir,
    a5: Ir,
) -> GLUMoESwiGLUDownState;

#[constructor]
pub(crate) fn mk_glu_mo_e_gemma_down_state(
    a0: DimExpr,
    a1: DimExpr,
    a2: DimExpr,
    a3: GLUMoEGemmaGELUState,
    a4: Ir,
    a5: Ir,
) -> GLUMoEGemmaDownState;

#[function(merge = |_old: GLUMoEExpertIndexState, new: GLUMoEExpertIndexState| new)]
pub(crate) fn glumoe_expert_index(a0: Ir) -> GLUMoEExpertIndexState;

#[function(merge = |_old: GLUMoEExpertGatherState, new: GLUMoEExpertGatherState| new)]
pub(crate) fn glumoe_expert_gather(a0: Ir) -> GLUMoEExpertGatherState;

#[function(merge = |_old: GLUMoEGateUpState, new: GLUMoEGateUpState| new)]
pub(crate) fn glumoe_gate_up(a0: Ir) -> GLUMoEGateUpState;

#[function(merge = |_old: GLUMoESwiGLUState, new: GLUMoESwiGLUState| new)]
pub(crate) fn glumoe_swiglu(a0: Ir) -> GLUMoESwiGLUState;

#[function(merge = |_old: GLUMoEGemmaGELUState, new: GLUMoEGemmaGELUState| new)]
pub(crate) fn glumoe_gemma_gelu(a0: Ir) -> GLUMoEGemmaGELUState;

#[function(merge = |_old: GLUMoESwiGLUDownState, new: GLUMoESwiGLUDownState| new)]
pub(crate) fn glumoe_swiglu_down(a0: Ir) -> GLUMoESwiGLUDownState;

#[function(merge = |_old: GLUMoEGemmaDownState, new: GLUMoEGemmaDownState| new)]
pub(crate) fn glumoe_gemma_down(a0: Ir) -> GLUMoEGemmaDownState;

#[ruleset]
pub(crate) fn dtype_prop(inputs: &IList, e: &Ir) -> Rule {
    rule(
        (eq(e, Ir::Op(GLUMoEArgs::fresh(), inputs)),),
        (set(e.dtype(), DType::F32()),),
    )
}

#[allow(
    clippy::approx_constant,
    reason = "Exact pattern literals must match the authored graph."
)]
#[ruleset]
pub(crate) fn glumoe(
    io: &DimExpr,
    topk_idx: &Ir,
    iota_within_range: &DimExpr,
    add_idx: &Ir,
    idx: &Ir,
    within_range: &DimExpr,
    weights: &Ir,
    f: &Ir,
    gu_f: &Ir,
    gu_io: &DimExpr,
    gu_within_range: &DimExpr,
    gate_up_w: &Ir,
    x: &Ir,
    gu_matmul: &Ir,
    gate_up_state: &GLUMoEGateUpState,
    neg_gate: &Ir,
    activation: &Ir,
    swiglu_state: &GLUMoESwiGLUState,
    dn_f: &Ir,
    dn_io: &DimExpr,
    dn_within_range: &DimExpr,
    down_w: &Ir,
    dn_matmul: &Ir,
    gemma_state: &GLUMoEGemmaGELUState,
    routing_weights: &Ir,
    topk_vals: &Ir,
    weighted: &Ir,
    output: &Ir,
    per_expert_scale: &Ir,
) -> Vec<Rule> {
    let gate_up_reduction = SumArgs::fresh();
    let down_reduction = SumArgs::fresh();
    let output_reduction = SumArgs::fresh();
    let normalization_reduction = SumArgs {
        iters: output_reduction.iters.clone(),
        ..SumArgs::fresh()
    };
    let swiglu_replacement = &Ir::Op(
        glu_mo_e(
            gu_io,
            dn_io,
            &gate_up_reduction.iters,
            &down_reduction.iters,
            &output_reduction.iters,
            gu_within_range,
            dn_within_range,
            0,
        ),
        [x, topk_idx, topk_vals, gate_up_w, down_w, topk_vals],
    );
    let normalized_swiglu_replacement = &Ir::Op(
        glu_mo_e(
            gu_io,
            dn_io,
            &gate_up_reduction.iters,
            &down_reduction.iters,
            &output_reduction.iters,
            gu_within_range,
            dn_within_range,
            2,
        ),
        [x, topk_idx, topk_vals, gate_up_w, down_w, topk_vals],
    );
    let gemma_replacement = &Ir::Op(
        glu_mo_e(
            gu_io,
            dn_io,
            &gate_up_reduction.iters,
            &down_reduction.iters,
            &output_reduction.iters,
            gu_within_range,
            dn_within_range,
            1,
        ),
        [x, topk_idx, topk_vals, gate_up_w, down_w, per_expert_scale],
    );
    // Shared LHS fragments follow the index, activation, and routing chains.
    let iota_base = &Ir::Op(
        IotaArgs {
            expr: io.into(),
            ..IotaArgs::fresh()
        },
        IList::INil(),
    );
    let mul_base = &Ir::Op(MulArgs::fresh(), [topk_idx, iota_base]);
    let iota_within = &Ir::Op(
        OpKind::Iota(DimExpr::MIter(), iota_within_range),
        IList::INil(),
    );
    let gathered = &Ir::Op(GatherArgs::fresh(), [idx, weights]);
    let gu_matmul_mul = &Ir::Op(MulArgs::fresh(), [x, gu_f]);
    let up_iota = &Ir::Op(IotaArgs::fresh(), IList::INil());
    let up_slice = &Ir::Op(GatherArgs::fresh(), [up_iota, gu_matmul]);
    let neg = &Ir::Op(OpKind::Constant(-1.0), IList::INil());
    let log2e = &Ir::Op(OpKind::Constant(1.442695), IList::INil());
    let scaled = &Ir::Op(MulArgs::fresh(), [neg_gate, log2e]);
    let exp2_val = &Ir::Op(Exp2Args::fresh(), [scaled]);
    let one = &Ir::Op(OpKind::Constant(1.0), IList::INil());
    let plus = &Ir::Op(AddArgs::fresh(), [exp2_val, one]);
    let sigmoid = &Ir::Op(RecipArgs::fresh(), [plus]);
    let silu_out = &Ir::Op(MulArgs::fresh(), [gu_matmul, sigmoid]);
    let gelu_coeff_inner = &Ir::Op(OpKind::Constant(0.044715), IList::INil());
    let gelu_inner_scaled = &Ir::Op(MulArgs::fresh(), [gu_matmul, gelu_coeff_inner]);
    let gelu_inner_quad = &Ir::Op(MulArgs::fresh(), [gelu_inner_scaled, gu_matmul]);
    let gelu_poly = &Ir::Op(AddArgs::fresh(), [gelu_inner_quad, one]);
    let gelu_coeff_outer = &Ir::Op(OpKind::Constant(1.595769), IList::INil());
    let gelu_outer_scaled = &Ir::Op(MulArgs::fresh(), [gu_matmul, gelu_coeff_outer]);
    let gelu_scaled = &Ir::Op(MulArgs::fresh(), [gelu_outer_scaled, gelu_poly]);
    let gelu_out = &Ir::Op(MulArgs::fresh(), [gu_matmul, sigmoid]);
    let dn_matmul_mul = &Ir::Op(MulArgs::fresh(), [activation, dn_f]);
    let topk_row_offsets = &Ir::Op(IotaArgs::fresh(), IList::INil());
    let topk_flat_idx = &Ir::Op(AddArgs::fresh(), [topk_row_offsets, topk_idx]);
    let topk_norm = &Ir::Op(normalization_reduction.clone(), [topk_vals]);
    let topk_norm_factor = &Ir::Op(RecipArgs::fresh(), [topk_norm]);
    let normed_topk = &Ir::Op(MulArgs::fresh(), [topk_vals, topk_norm_factor]);
    let per_expert_vals = &Ir::Op(GatherArgs::fresh(), [topk_idx, per_expert_scale]);
    let expert_weights = &Ir::Op(MulArgs::fresh(), [normed_topk, per_expert_vals]);
    vec![
        rule(
            (eq(
                add_idx,
                Ir::Op(AddArgs::fresh(), [mul_base, iota_within]),
            ),),
            (set(
                glumoe_expert_index(add_idx),
                mk_glu_mo_e_expert_index_state(io, iota_within_range, topk_idx),
            ),),
        )
        .label("GLUMoE expert index marker"),
        rule(
            (
                eq(
                    glumoe_expert_index(idx),
                    mk_glu_mo_e_expert_index_state(io, within_range, topk_idx),
                ),
                eq(
                    f,
                    Ir::Op(
                        CastArgs {
                            dtype: DType::F32(),
                            ..CastArgs::fresh()
                        },
                        [gathered],
                    ),
                ),
            ),
            (set(
                glumoe_expert_gather(f),
                mk_glu_mo_e_expert_gather_state(io, within_range, topk_idx, weights),
            ),),
        )
        .label("GLUMoE expert gather marker"),
        rule(
            (
                eq(
                    glumoe_expert_gather(gu_f),
                    mk_glu_mo_e_expert_gather_state(gu_io, gu_within_range, topk_idx, gate_up_w),
                ),
                eq(
                    gu_matmul,
                    Ir::Op(gate_up_reduction.clone(), [gu_matmul_mul]),
                ),
            ),
            (set(
                glumoe_gate_up(gu_matmul),
                mk_glu_mo_e_gate_up_state(
                    gu_io,
                    &gate_up_reduction.iters,
                    gu_within_range,
                    x,
                    topk_idx,
                    gate_up_w,
                ),
            ),),
        )
        .label("GLUMoE gate-up matmul marker"),
        // ===== SwiGLU activation marker =====
        rule(
            (
                eq(gate_up_state, glumoe_gate_up(gu_matmul)),
                eq(gate_up_state, GLUMoEGateUpStateArgs::fresh()),
                eq(neg_gate, Ir::Op(MulArgs::fresh(), [gu_matmul, neg])),
                eq(activation, Ir::Op(MulArgs::fresh(), [silu_out, up_slice])),
            ),
            (set(
                glumoe_swiglu(activation),
                mk_glu_mo_e_swi_glu_state(gate_up_state),
            ),),
        )
        .label("GLUMoE swiglu marker"),
        // ===== Gemma GELU activation marker =====
        rule(
            (
                eq(gate_up_state, glumoe_gate_up(gu_matmul)),
                eq(gate_up_state, GLUMoEGateUpStateArgs::fresh()),
                eq(neg_gate, Ir::Op(MulArgs::fresh(), [gelu_scaled, neg])),
                eq(activation, Ir::Op(MulArgs::fresh(), [gelu_out, up_slice])),
            ),
            (set(
                glumoe_gemma_gelu(activation),
                mk_glu_mo_e_gemma_gelu_state(gate_up_state),
            ),),
        )
        .label("GLUMoE gemma gelu marker"),
        // ===== SwiGLU down marker =====
        rule(
            (
                eq(swiglu_state, glumoe_swiglu(activation)),
                eq(swiglu_state, mk_glu_mo_e_swi_glu_state(gate_up_state)),
                eq(
                    glumoe_expert_gather(dn_f),
                    mk_glu_mo_e_expert_gather_state(dn_io, dn_within_range, topk_idx, down_w),
                ),
                eq(dn_matmul, Ir::Op(down_reduction.clone(), [dn_matmul_mul])),
            ),
            (set(
                glumoe_swiglu_down(dn_matmul),
                mk_glu_mo_e_swi_glu_down_state(
                    dn_io,
                    &down_reduction.iters,
                    dn_within_range,
                    swiglu_state,
                    topk_idx,
                    down_w,
                ),
            ),),
        )
        .label("GLUMoE swiglu down marker"),
        // ===== Gemma GELU down marker =====
        rule(
            (
                eq(gemma_state, glumoe_gemma_gelu(activation)),
                eq(gemma_state, mk_glu_mo_e_gemma_gelu_state(gate_up_state)),
                eq(
                    glumoe_expert_gather(dn_f),
                    mk_glu_mo_e_expert_gather_state(dn_io, dn_within_range, topk_idx, down_w),
                ),
                eq(dn_matmul, Ir::Op(down_reduction.clone(), [dn_matmul_mul])),
            ),
            (set(
                glumoe_gemma_down(dn_matmul),
                mk_glu_mo_e_gemma_down_state(
                    dn_io,
                    &down_reduction.iters,
                    dn_within_range,
                    gemma_state,
                    topk_idx,
                    down_w,
                ),
            ),),
        )
        .label("GLUMoE gemma down marker"),
        // ===== Final fusion: mode 0 (SwiGLU) =====
        rule(
            (
                eq(
                    glumoe_swiglu_down(dn_matmul),
                    mk_glu_mo_e_swi_glu_down_state(
                        dn_io,
                        &down_reduction.iters,
                        dn_within_range,
                        swiglu_state,
                        topk_idx,
                        down_w,
                    ),
                ),
                eq(swiglu_state, mk_glu_mo_e_swi_glu_state(gate_up_state)),
                eq(
                    gate_up_state,
                    mk_glu_mo_e_gate_up_state(
                        gu_io,
                        &gate_up_reduction.iters,
                        gu_within_range,
                        x,
                        topk_idx,
                        gate_up_w,
                    ),
                ),
                eq(
                    topk_vals,
                    Ir::Op(GatherArgs::fresh(), [topk_flat_idx, routing_weights]),
                ),
                eq(weighted, Ir::Op(MulArgs::fresh(), [dn_matmul, topk_vals])),
                eq(output, Ir::Op(output_reduction.clone(), [weighted])),
            ),
            (swiglu_replacement, union(output, swiglu_replacement)),
        )
        .label("GLUMoE fused expert computation (swiglu)"),
        // ===== Final fusion: mode 2 (SwiGLU with row-normalized top-k weights) =====
        rule(
            (
                eq(
                    glumoe_swiglu_down(dn_matmul),
                    mk_glu_mo_e_swi_glu_down_state(
                        dn_io,
                        &down_reduction.iters,
                        dn_within_range,
                        swiglu_state,
                        topk_idx,
                        down_w,
                    ),
                ),
                eq(swiglu_state, mk_glu_mo_e_swi_glu_state(gate_up_state)),
                eq(
                    gate_up_state,
                    mk_glu_mo_e_gate_up_state(
                        gu_io,
                        &gate_up_reduction.iters,
                        gu_within_range,
                        x,
                        topk_idx,
                        gate_up_w,
                    ),
                ),
                eq(
                    topk_vals,
                    Ir::Op(GatherArgs::fresh(), [topk_flat_idx, routing_weights]),
                ),
                eq(weighted, Ir::Op(MulArgs::fresh(), [dn_matmul, normed_topk])),
                eq(output, Ir::Op(output_reduction.clone(), [weighted])),
            ),
            (
                normalized_swiglu_replacement,
                union(output, normalized_swiglu_replacement),
            ),
        )
        .label("GLUMoE fused expert computation (normalized swiglu)"),
        // ===== Final fusion: mode 1 (Gemma GELU) =====
        rule(
            (
                eq(
                    glumoe_gemma_down(dn_matmul),
                    mk_glu_mo_e_gemma_down_state(
                        dn_io,
                        &down_reduction.iters,
                        dn_within_range,
                        gemma_state,
                        topk_idx,
                        down_w,
                    ),
                ),
                eq(gemma_state, mk_glu_mo_e_gemma_gelu_state(gate_up_state)),
                eq(
                    gate_up_state,
                    mk_glu_mo_e_gate_up_state(
                        gu_io,
                        &gate_up_reduction.iters,
                        gu_within_range,
                        x,
                        topk_idx,
                        gate_up_w,
                    ),
                ),
                // Gemma expert weights: topk_weights = normed_topk * per_expert_scale.gather(topk_idx)
                eq(
                    topk_vals,
                    Ir::Op(GatherArgs::fresh(), [topk_flat_idx, routing_weights]),
                ),
                eq(
                    weighted,
                    Ir::Op(MulArgs::fresh(), [dn_matmul, expert_weights]),
                ),
                eq(output, Ir::Op(output_reduction.clone(), [weighted])),
            ),
            (gemma_replacement, union(output, gemma_replacement)),
        )
        .label("GLUMoE fused expert computation (gemma_gelu)"),
    ]
}
