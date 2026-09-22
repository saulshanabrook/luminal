// FlashInfer batch decode attention rewrite rule.
//
// Matches the paged attention pattern for ANY model with GQA:
//   Gather(K_cache) → GQA broadcast → Q*K^T matmul → scale → add mask → softmax → attn*V matmul
//   Gather(V_cache) → GQA broadcast ──────────────────────────────────────────→ attn*V matmul
//
// Structural anchors (prevent false matches on MLP/other ops):
//   - Gather ops from 2D cache pools (MLP never uses Gather)
//   - K and V cache gathers use the same flat gather index
//   - GQA broadcast via Mul(gathered, Constant(1.0)) with all-zero strides
//   - Scale Mul(QK, constant) connecting QK scores to mask Add
//   - Mask Add with zero-stride broadcast in first dim (nheads broadcast)
//   - Data flow: two sequential matmul+reduce pairs connected through softmax
//
// Rewrites retain the original flat-index and mask proof dependencies. The
// selected FlashInfer decoder recovers the compact index and keeps the four
// execution inputs (Q, K pool, V pool, compact gather_idx).
//
// Shape dimensions are egglog variables, not pinned constants.
// Dynamic dims "s" (batch/seq) and "c" (context) stay pinned as MVar.

use crate::typed_rules::*;
use egglog_experimental::typed::builtins as egg;
use egglog_experimental::typed::prelude::*;
use luminal::typed_reference::schema::*;

// K and V cache sources must come from the same cache-update provenance.
// This prevents a rolled loop-body K cache from pairing with the one remaining
// unrolled V cache outside the loop, which is shape-compatible but not the
// same attention instance.
#[relation]
pub(crate) fn flashinfer_cache_pair(a0: Ir, a1: Ir);

// Head dims the JIT-compiled FlashInfer kernels support (jit.rs asserts the
// same set). Rules must not capture islands for other dims (e.g. gemma-4's
// 512-dim full-attention layers stay on the HLIR chain).
#[relation]
pub(crate) fn flashinfer_head_dim_ok(a0: DimExpr);

// 512 is 16-bit only (the f32 kernel instantiation is gated out of the
// wrapper at HEAD_DIM > 256) — used by rules that pin a 16-bit dtype.
#[relation]
pub(crate) fn flashinfer_head_dim_ok_16bit(a0: DimExpr);

// Mask-chain facts staged for the attention island rules. Every island rule
// anchors at one of these tiny relations instead of inlining its mask atoms:
// the mask-fact joins are small and run once in kernel_fuse_late_pre, and the
// island joins stay narrow on rolled bodies with many distinct attention
// instances (inlined, the gemma body ground for 35+ minutes in fuse-late).
#[relation]
pub(crate) fn fi_triu_mask(a0: Ir, a1: Ir);

#[relation]
pub(crate) fn fi_window_term(a0: Ir, a1: Ir, a2: egg::F64);

#[relation]
pub(crate) fn fi_scaled_offset_mask(a0: Ir);

#[relation]
pub(crate) fn fi_direct_causal_mask(a0: Ir, a1: Ir);

// A constant-valued eclass regardless of dtype: either a plain HLIR Constant
// (f32 graphs) or a folded dtype-typed KernelConstant (the frontend emits
// `constant(v).cast(dt)` for non-f32 scalars, and the kernel_lower fold rule
// unions a KernelConstant into the Cast's eclass).
#[relation]
pub(crate) fn const_like(a0: Ir, a1: egg::F64);

// Dtypes with a tensor-core prefill runtime (16-bit only).
#[relation]
pub(crate) fn flashinfer_prefill_dt(a0: DType);

#[ruleset]
pub(crate) fn matmul_backend(
    v: &egg::F64,
    e: &Ir,
    cdt: &DType,
    k_dest: &Ir,
    scatter_idx: &Ir,
    k_src: &Ir,
    k_cache: &Ir,
    v_dest: &Ir,
    v_src: &Ir,
    v_cache: &Ir,
    loop_id: &egg::I64,
    k_loop_inputs: &IList,
    v_loop_inputs: &IList,
) -> Vec<Rule> {
    vec![
        rule(
            (eq(e, Ir::Op(OpKind::Constant(v), IList::INil())),),
            (const_like(e, v),),
        )
        .label("FlashInfer const-like hlir"),
        rule(
            (eq(e, Ir::Op(kernel_constant(v, cdt), IList::INil())),),
            (const_like(e, v),),
        )
        .label("FlashInfer const-like kernel"),
        rule(
            (
                eq(
                    k_cache,
                    Ir::Op(ScatterArgs::fresh(), [k_dest, scatter_idx, k_src]),
                ),
                eq(
                    v_cache,
                    Ir::Op(ScatterArgs::fresh(), [v_dest, scatter_idx, v_src]),
                ),
                eq(k_dest, InputArgs::fresh()),
                eq(v_dest, InputArgs::fresh()),
            ),
            (flashinfer_cache_pair(k_cache, v_cache),),
        )
        .label("FlashInfer raw cache pair"),
        rule(
            (
                eq(
                    k_cache,
                    Ir::Op(ScatterArgs::fresh(), [k_dest, scatter_idx, k_src]),
                ),
                eq(
                    v_cache,
                    Ir::Op(ScatterArgs::fresh(), [v_dest, scatter_idx, v_src]),
                ),
                eq(
                    k_dest,
                    Ir::Op(
                        LoopInputArgs {
                            loop_id: loop_id.into(),
                            ..LoopInputArgs::fresh()
                        },
                        k_loop_inputs,
                    ),
                ),
                eq(
                    v_dest,
                    Ir::Op(
                        LoopInputArgs {
                            loop_id: loop_id.into(),
                            ..LoopInputArgs::fresh()
                        },
                        v_loop_inputs,
                    ),
                ),
            ),
            (flashinfer_cache_pair(k_cache, v_cache),),
        )
        .label("FlashInfer rolled cache pair"),
    ]
}

#[allow(
    clippy::approx_constant,
    reason = "Exact pattern literals must match the authored graph."
)]
#[ruleset]
pub(crate) fn kernel_fuse_late2(
    masked: &Ir,
    nm_negone: &Ir,
    sc_log2e: &Ir,
    nheads: &DimExpr,
    hdim: &DimExpr,
    v_gqa: &Ir,
    mul: &Ir,
    nheads2: &DimExpr,
    hdim2: &DimExpr,
    output: &Ir,
    v_gqa_const: &Ir,
    nheads3: &DimExpr,
    hdim3: &DimExpr,
    kvdim: &DimExpr,
    num_slots_v: &DimExpr,
    k_idx: &Ir,
    v_cache: &Ir,
    nheads4: &DimExpr,
    hdim4: &DimExpr,
    q: &Ir,
    k_gqa: &Ir,
    mul2: &Ir,
    nheads5: &DimExpr,
    nheads6: &DimExpr,
    mask_rest_strides: &EList,
    mask: &Ir,
    sq_scale: &Ir,
    sq_scale_val: &egg::F64,
    k_gqa_const: &Ir,
    nheads7: &DimExpr,
    hdim5: &DimExpr,
    num_slots_k: &DimExpr,
    k_cache: &Ir,
    dt: &DType,
    s_lower: &egg::I64,
    s_upper: &egg::I64,
    q_pos: &Ir,
    cmask: &Ir,
    wterm: &Ir,
    q_pos2: &Ir,
    window_left: &egg::F64,
    v_data_width: &DimExpr,
    k_index_width: &DimExpr,
    k_data_width: &DimExpr,
) -> Vec<Rule> {
    // Shared query syntax: each rule still matches its own softmax chain.
    let smax = &Ir::Op(
        MaxArgs {
            iters: "c".into(),
            ..MaxArgs::fresh()
        },
        [masked],
    );
    let negmax = &Ir::Op(MulArgs::fresh(), [smax, nm_negone]);
    let shifted = &Ir::Op(AddArgs::fresh(), [masked, negmax]);
    let sc = &Ir::Op(MulArgs::fresh(), [shifted, sc_log2e]);
    let sexp = &Ir::Op(Exp2Args::fresh(), [sc]);
    let sexpsum = &Ir::Op(
        SumArgs {
            iters: "c".into(),
            ..SumArgs::fresh()
        },
        [sexp],
    );
    let srecip = &Ir::Op(RecipArgs::fresh(), [sexpsum]);
    let soft = &Ir::Op(MulArgs::fresh(), [sexp, srecip]);
    let attention = &Ir::Op(
        flash_infer_attention(nheads, kvdim / hdim, hdim, 1, "s", dt, sq_scale_val, -1.0),
        [q, k_cache, v_cache, k_idx, mask],
    );
    let scale_free_attention = &Ir::Op(
        flash_infer_attention(nheads2, kvdim / hdim, hdim, 1, "s", dt, 1.0, -1.0),
        [q, k_cache, v_cache, k_idx, mask],
    );
    let sliding_attention = &Ir::Op(
        flash_infer_attention(nheads2, kvdim / hdim, hdim, 1, "s", dt, 1.0, window_left),
        [q, k_cache, v_cache, k_idx, mask],
    );
    let v_gathered = &Ir::Op(
        GatherArgs {
            index_shape: [DimExpr::from("c"), kvdim.clone()].into(),
            data_shape: [num_slots_v, v_data_width].into(),
            ..GatherArgs::fresh()
        },
        [k_idx, v_cache],
    );
    let qk = &Ir::Op(
        SumArgs {
            shape: ([nheads5, &(DimExpr::MVar("s")), &(DimExpr::MVar("c"))]).into(),
            iter_stride: DimExpr::MIter(),
            ..SumArgs::fresh()
        },
        [mul2],
    );
    let k_gathered = &Ir::Op(
        GatherArgs {
            index_shape: [DimExpr::from("c"), k_index_width.clone()].into(),
            data_shape: [num_slots_k, k_data_width].into(),
            ..GatherArgs::fresh()
        },
        [k_idx, k_cache],
    );
    let scaled_qk = &Ir::Op(MulArgs::fresh(), [qk, sq_scale]);
    vec![
        rule(
            (
                // softmax internals connect the mask chain to the second matmul —
                // without this the two halves join as a cross-product across the
                // distinct layer instances of rolled multi-layer bodies.
                Fact::from(const_like(sc_log2e, 1.442695)),
                // ── Second matmul: Mul(softmax_out, V_gqa) ──
                // Shape: (nheads, s, hdim, c) — 4D
                eq(
                    mul,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads, &(DimExpr::MVar("s")), hdim, &(DimExpr::MVar("c"))])
                                .into(),
                            ..MulArgs::fresh()
                        },
                        [soft, v_gqa],
                    ),
                ),
                // ── Second matmul: Sum (reduction over c) → output ──
                // Shape: (nheads, s, hdim) — reduces c
                eq(
                    output,
                    Ir::Op(
                        SumArgs {
                            shape: ([nheads2, &(DimExpr::MVar("s")), hdim2]).into(),
                            iters: "c".into(),
                            iter_stride: DimExpr::MIter(),
                            ..SumArgs::fresh()
                        },
                        [mul],
                    ),
                ),
                Fact::from(const_like(v_gqa_const, 1.0)),
                // ── V GQA broadcast: Mul(V_gathered, 1.0) with zero-stride constant ──
                // Shape: (nheads, c, hdim) — 3D
                eq(
                    v_gqa,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads3, &(DimExpr::MVar("c")), hdim3]).into(),
                            b_strides: ([0, 0, 0]).into(),
                            ..MulArgs::fresh()
                        },
                        // ── V Gather: rows from V_cache (2D) ──
                        // Shape: (c, kvdim), Source: (num_slots, kvdim)
                        [v_gathered, v_gqa_const],
                    ),
                ),
                // ── First matmul: Mul(Q, K_gqa) ──
                // Shape: (nheads, s, c, hdim) — 4D
                eq(
                    mul2,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads4, &(DimExpr::MVar("s")), &(DimExpr::MVar("c")), hdim4])
                                .into(),
                            ..MulArgs::fresh()
                        },
                        [q, k_gqa],
                    ),
                ),
                // ── First matmul: Sum (reduction over hdim) → QK scores ──
                // Shape: (nheads, s, c) — reduces hdim
                // ── Mask Add: Add(scaled_QK, mask) ──
                // Shape: (nheads, s, c) — 3D
                // Mask is broadcast from (s, c) via zero-stride in first dim (nheads).
                eq(
                    masked,
                    Ir::Op(
                        AddArgs {
                            shape: ([nheads6, &(DimExpr::MVar("s")), &(DimExpr::MVar("c"))]).into(),
                            b_strides: EList::ECons(0, mask_rest_strides),
                            ..AddArgs::fresh()
                        },
                        [scaled_qk, mask],
                    ),
                ),
                // The mask-add's first operand is the QK score scaled by a broadcast
                // constant (1/sqrt(head_dim) — the frontend lowers `scores / s` to
                // `Mul(scores, const(1/s))`). This atom CONNECTS the Q/K/QK component
                // to the mask/softmax/output component; without it the premise graph
                // has two disconnected halves whose join cross-products across the
                // distinct layer instances of rolled multi-layer bodies (the gemma
                // 530GB egglog OOM), and Q could pair with another instance's mask.
                Fact::from(const_like(sq_scale, sq_scale_val)),
                // The mask proves a valid KV prefix; do not accept a precomputed mask Input.
                // Mask spelling staged as a fact — see "FlashInfer scaled-offset mask
                // fact".
                Fact::from(fi_scaled_offset_mask(mask)),
                Fact::from(const_like(k_gqa_const, 1.0)),
                // ── K GQA broadcast: Mul(K_gathered, 1.0) with zero-stride constant ──
                // Shape: (nheads, hdim, c) — 3D
                eq(
                    k_gqa,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads7, hdim5, &(DimExpr::MVar("c"))]).into(),
                            b_strides: ([0, 0, 0]).into(),
                            ..MulArgs::fresh()
                        },
                        // ── K Gather: rows from K_cache (2D) ──
                        // Shape: (c, kvdim), Source: (num_slots, kvdim)
                        [k_gathered, k_gqa_const],
                    ),
                ),
                // ── Dtype consistency ──
                eq(dt, q.dtype()),
                eq(dt, k_cache.dtype()),
                eq(dt, v_cache.dtype()),
                Fact::from(flashinfer_cache_pair(k_cache, v_cache)),
                Fact::from(flashinfer_head_dim_ok(hdim)),
            ),
            (
                attention,
                union(output, attention),
                set(attention.dtype(), dt),
            ),
        )
        .label("FlashInfer batch decode attention"),
        // Direct causal mask spelling:
        // (q_pos[:, None] < arange(c)[None, :]).cast(F32) * -1e10
        // This proves each row blocks a suffix and leaves a valid prefix.
        // This spelling currently lowers to the fp32 batch-decode runtime path,
        // not prefill. It is only valid when the active dynamic-dim bucket proves
        // s == 1.
        rule(
            [
                eq(s_lower, (DimExpr::MVar("s")).lower()),
                eq(s_upper, (DimExpr::MVar("s")).upper()),
                s_lower.gt(0_i64),
                s_upper.lt(2_i64),
                // softmax internals connect the mask chain to the second matmul —
                // without this the two halves join as a cross-product across the
                // distinct layer instances of rolled multi-layer bodies.
                Fact::from(const_like(sc_log2e, 1.442695)),
                eq(
                    mul,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads, &(DimExpr::MVar("s")), hdim, &(DimExpr::MVar("c"))])
                                .into(),
                            ..MulArgs::fresh()
                        },
                        [soft, v_gqa],
                    ),
                ),
                eq(
                    output,
                    Ir::Op(
                        SumArgs {
                            shape: ([nheads2, &(DimExpr::MVar("s")), hdim2]).into(),
                            iters: "c".into(),
                            iter_stride: DimExpr::MIter(),
                            ..SumArgs::fresh()
                        },
                        [mul],
                    ),
                ),
                Fact::from(const_like(v_gqa_const, 1.0)),
                eq(
                    v_gqa,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads3, &(DimExpr::MVar("c")), hdim3]).into(),
                            b_strides: ([0, 0, 0]).into(),
                            ..MulArgs::fresh()
                        },
                        [v_gathered, v_gqa_const],
                    ),
                ),
                eq(
                    mul2,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads4, &(DimExpr::MVar("s")), &(DimExpr::MVar("c")), hdim4])
                                .into(),
                            ..MulArgs::fresh()
                        },
                        [q, k_gqa],
                    ),
                ),
                eq(
                    masked,
                    Ir::Op(
                        AddArgs {
                            shape: ([nheads6, &(DimExpr::MVar("s")), &(DimExpr::MVar("c"))]).into(),
                            b_strides: EList::ECons(0, mask_rest_strides),
                            ..AddArgs::fresh()
                        },
                        [scaled_qk, mask],
                    ),
                ),
                // The mask-add's first operand is the QK score scaled by a broadcast
                // constant (1/sqrt(head_dim) — the frontend lowers `scores / s` to
                // `Mul(scores, const(1/s))`). This atom CONNECTS the Q/K/QK component
                // to the mask/softmax/output component; without it the premise graph
                // has two disconnected halves whose join cross-products across the
                // distinct layer instances of rolled multi-layer bodies (the gemma
                // 530GB egglog OOM), and Q could pair with another instance's mask.
                Fact::from(const_like(sq_scale, sq_scale_val)),
                // Mask spelling staged as a fact — see "FlashInfer direct causal mask
                // fact".
                Fact::from(fi_direct_causal_mask(mask, q_pos)),
                Fact::from(const_like(k_gqa_const, 1.0)),
                eq(
                    k_gqa,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads7, hdim5, &(DimExpr::MVar("c"))]).into(),
                            b_strides: ([0, 0, 0]).into(),
                            ..MulArgs::fresh()
                        },
                        [k_gathered, k_gqa_const],
                    ),
                ),
                eq(dt, q.dtype()),
                eq(dt, k_cache.dtype()),
                eq(dt, v_cache.dtype()),
                Fact::from(flashinfer_cache_pair(k_cache, v_cache)),
                Fact::from(flashinfer_head_dim_ok(hdim)),
            ],
            (
                attention,
                union(output, attention),
                set(attention.dtype(), dt),
            ),
        )
        .label("FlashInfer direct causal attention"),
        // Same attention skeleton as above, but accept a causal mask produced
        // by gathering rows from a square upper-triangular blocked mask:
        // triu(c, 1).cast(F32) * -1e10
        // gather(q_pos * c + arange(c))
        // This proves each query row exposes only a prefix of the gathered KV
        // sequence. The compact index supplies the final runtime dependency.
        // Like the direct causal spelling, this is decode-only until the fp32
        // prefill runtime exists. The active bucket must prove s == 1.
        rule(
            [
                eq(s_lower, (DimExpr::MVar("s")).lower()),
                eq(s_upper, (DimExpr::MVar("s")).upper()),
                s_lower.gt(0_i64),
                s_upper.lt(2_i64),
                // softmax internals connect the mask chain to the second matmul —
                // without this the two halves join as a cross-product across the
                // distinct layer instances of rolled multi-layer bodies.
                Fact::from(const_like(sc_log2e, 1.442695)),
                eq(
                    mul,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads, &(DimExpr::MVar("s")), hdim, &(DimExpr::MVar("c"))])
                                .into(),
                            ..MulArgs::fresh()
                        },
                        [soft, v_gqa],
                    ),
                ),
                eq(
                    output,
                    Ir::Op(
                        SumArgs {
                            shape: ([nheads2, &(DimExpr::MVar("s")), hdim2]).into(),
                            iters: "c".into(),
                            iter_stride: DimExpr::MIter(),
                            ..SumArgs::fresh()
                        },
                        [mul],
                    ),
                ),
                Fact::from(const_like(v_gqa_const, 1.0)),
                eq(
                    v_gqa,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads3, &(DimExpr::MVar("c")), hdim3]).into(),
                            b_strides: ([0, 0, 0]).into(),
                            ..MulArgs::fresh()
                        },
                        [v_gathered, v_gqa_const],
                    ),
                ),
                eq(
                    mul2,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads4, &(DimExpr::MVar("s")), &(DimExpr::MVar("c")), hdim4])
                                .into(),
                            ..MulArgs::fresh()
                        },
                        [q, k_gqa],
                    ),
                ),
                eq(
                    masked,
                    Ir::Op(
                        AddArgs {
                            shape: ([nheads6, &(DimExpr::MVar("s")), &(DimExpr::MVar("c"))]).into(),
                            b_strides: EList::ECons(0, mask_rest_strides),
                            ..AddArgs::fresh()
                        },
                        [scaled_qk, mask],
                    ),
                ),
                // The mask-add's first operand is the QK score scaled by a broadcast
                // constant (1/sqrt(head_dim) — the frontend lowers `scores / s` to
                // `Mul(scores, const(1/s))`). This atom CONNECTS the Q/K/QK component
                // to the mask/softmax/output component; without it the premise graph
                // has two disconnected halves whose join cross-products across the
                // distinct layer instances of rolled multi-layer bodies (the gemma
                // 530GB egglog OOM), and Q could pair with another instance's mask.
                Fact::from(const_like(sq_scale, sq_scale_val)),
                // Mask indices are row-aligned triangular-mask rows plus arange(c).
                // Triu-gather causal mask, staged as a fact in kernel_fuse_late_pre
                // (see "FlashInfer triu mask fact"). Anchoring this rule at the
                // tiny fact relation instead of inlining the ~20 mask atoms keeps
                // the join narrow on rolled bodies with many distinct attention
                // instances (gemma: 35+ min / OOM as a single inlined join).
                Fact::from(fi_triu_mask(mask, q_pos)),
                Fact::from(const_like(k_gqa_const, 1.0)),
                eq(
                    k_gqa,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads7, hdim5, &(DimExpr::MVar("c"))]).into(),
                            b_strides: ([0, 0, 0]).into(),
                            ..MulArgs::fresh()
                        },
                        [k_gathered, k_gqa_const],
                    ),
                ),
                eq(dt, q.dtype()),
                eq(dt, k_cache.dtype()),
                eq(dt, v_cache.dtype()),
                Fact::from(flashinfer_cache_pair(k_cache, v_cache)),
                Fact::from(flashinfer_head_dim_ok(hdim)),
            ],
            (
                attention,
                union(output, attention),
                set(attention.dtype(), dt),
            ),
        )
        .label("FlashInfer causal triu-gather attention"),
        // Same attention skeleton as above, but accept a causal mask produced
        // by gathering rows from a square upper-triangular blocked mask:
        // triu(c, 1).cast(F32) * -1e10
        // gather(q_pos * c + arange(c))
        // This proves each query row exposes only a prefix of the gathered KV
        // sequence. 16-bit variant: matches prefill buckets too, since the
        // tensor-core prefill runtime exists for f16/bf16.
        // No s==1 guard: 16-bit dtypes have a tensor-core prefill runtime, so
        // this matches every bucket. The runtime picks decode vs prefill from
        // total_q_tokens at execute time.
        rule(
            (
                Fact::from(flashinfer_prefill_dt(dt)),
                // softmax internals connect the mask chain to the second matmul —
                // without this the two halves join as a cross-product across the
                // distinct layer instances of rolled multi-layer bodies.
                Fact::from(const_like(sc_log2e, 1.442695)),
                eq(
                    mul,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads, &(DimExpr::MVar("s")), hdim, &(DimExpr::MVar("c"))])
                                .into(),
                            ..MulArgs::fresh()
                        },
                        [soft, v_gqa],
                    ),
                ),
                eq(
                    output,
                    Ir::Op(
                        SumArgs {
                            shape: ([nheads2, &(DimExpr::MVar("s")), hdim2]).into(),
                            iters: "c".into(),
                            iter_stride: DimExpr::MIter(),
                            ..SumArgs::fresh()
                        },
                        [mul],
                    ),
                ),
                Fact::from(const_like(v_gqa_const, 1.0)),
                eq(
                    v_gqa,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads3, &(DimExpr::MVar("c")), hdim3]).into(),
                            b_strides: ([0, 0, 0]).into(),
                            ..MulArgs::fresh()
                        },
                        [v_gathered, v_gqa_const],
                    ),
                ),
                eq(
                    mul2,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads4, &(DimExpr::MVar("s")), &(DimExpr::MVar("c")), hdim4])
                                .into(),
                            ..MulArgs::fresh()
                        },
                        [q, k_gqa],
                    ),
                ),
                eq(
                    masked,
                    Ir::Op(
                        AddArgs {
                            shape: ([nheads6, &(DimExpr::MVar("s")), &(DimExpr::MVar("c"))]).into(),
                            b_strides: EList::ECons(0, mask_rest_strides),
                            ..AddArgs::fresh()
                        },
                        [scaled_qk, mask],
                    ),
                ),
                // The mask-add's first operand is the QK score scaled by a broadcast
                // constant (1/sqrt(head_dim) — the frontend lowers `scores / s` to
                // `Mul(scores, const(1/s))`). This atom CONNECTS the Q/K/QK component
                // to the mask/softmax/output component; without it the premise graph
                // has two disconnected halves whose join cross-products across the
                // distinct layer instances of rolled multi-layer bodies (the gemma
                // 530GB egglog OOM), and Q could pair with another instance's mask.
                Fact::from(const_like(sq_scale, sq_scale_val)),
                // Mask indices are row-aligned triangular-mask rows plus arange(c).
                // Triu-gather causal mask, staged as a fact in kernel_fuse_late_pre
                // (see "FlashInfer triu mask fact"). Anchoring this rule at the
                // tiny fact relation instead of inlining the ~20 mask atoms keeps
                // the join narrow on rolled bodies with many distinct attention
                // instances (gemma: 35+ min / OOM as a single inlined join).
                Fact::from(fi_triu_mask(mask, q_pos)),
                Fact::from(const_like(k_gqa_const, 1.0)),
                eq(
                    k_gqa,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads7, hdim5, &(DimExpr::MVar("c"))]).into(),
                            b_strides: ([0, 0, 0]).into(),
                            ..MulArgs::fresh()
                        },
                        [k_gathered, k_gqa_const],
                    ),
                ),
                eq(dt, q.dtype()),
                eq(dt, k_cache.dtype()),
                eq(dt, v_cache.dtype()),
                Fact::from(flashinfer_cache_pair(k_cache, v_cache)),
                Fact::from(flashinfer_head_dim_ok(hdim)),
            ),
            (
                attention,
                union(output, attention),
                set(attention.dtype(), dt),
            ),
        )
        .label("FlashInfer causal triu-gather attention 16-bit prefill"),
        // Gemma-4 full attention: scale-free, causal triu-gather mask.
        // 16-bit only (covers head_dim 512); no s==1 guard — the tensor-core
        // prefill runtime handles s>1.
        rule(
            (
                eq(s_lower, (DimExpr::MVar("s")).lower()),
                eq(s_upper, (DimExpr::MVar("s")).upper()),
                s_lower.gt(0_i64),
                s_upper.lt(2_i64),
                Fact::from(flashinfer_prefill_dt(dt)),
                Fact::from(fi_triu_mask(mask, q_pos)),
                // SCALE-FREE: ?qk (the raw QK Sum) feeds the mask Add directly.
                eq(
                    masked,
                    Ir::Op(
                        AddArgs {
                            shape: ([nheads, &(DimExpr::MVar("s")), &(DimExpr::MVar("c"))]).into(),
                            b_strides: EList::ECons(0, mask_rest_strides),
                            ..AddArgs::fresh()
                        },
                        [qk, mask],
                    ),
                ),
                // mask Add → softmax internals → second matmul: one connected chain.
                Fact::from(const_like(sc_log2e, 1.442695)),
                eq(
                    mul,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads2, &(DimExpr::MVar("s")), hdim, &(DimExpr::MVar("c"))])
                                .into(),
                            ..MulArgs::fresh()
                        },
                        [soft, v_gqa],
                    ),
                ),
                eq(
                    output,
                    Ir::Op(
                        SumArgs {
                            shape: ([nheads3, &(DimExpr::MVar("s")), hdim2]).into(),
                            iters: "c".into(),
                            iter_stride: DimExpr::MIter(),
                            ..SumArgs::fresh()
                        },
                        [mul],
                    ),
                ),
                Fact::from(const_like(v_gqa_const, 1.0)),
                eq(
                    v_gqa,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads4, &(DimExpr::MVar("c")), hdim3]).into(),
                            b_strides: ([0, 0, 0]).into(),
                            ..MulArgs::fresh()
                        },
                        [v_gathered, v_gqa_const],
                    ),
                ),
                eq(
                    mul2,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads6, &(DimExpr::MVar("s")), &(DimExpr::MVar("c")), hdim4])
                                .into(),
                            ..MulArgs::fresh()
                        },
                        [q, k_gqa],
                    ),
                ),
                Fact::from(const_like(k_gqa_const, 1.0)),
                eq(
                    k_gqa,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads7, hdim5, &(DimExpr::MVar("c"))]).into(),
                            b_strides: ([0, 0, 0]).into(),
                            ..MulArgs::fresh()
                        },
                        [k_gathered, k_gqa_const],
                    ),
                ),
                eq(dt, q.dtype()),
                eq(dt, k_cache.dtype()),
                eq(dt, v_cache.dtype()),
                Fact::from(flashinfer_cache_pair(k_cache, v_cache)),
                Fact::from(flashinfer_head_dim_ok_16bit(hdim)),
            ),
            (
                scale_free_attention,
                union(output, scale_free_attention),
                set(scale_free_attention.dtype(), dt),
            ),
        )
        .label("FlashInfer gemma scale-free triu-gather attention"),
        // Gemma-4 sliding-window attention: scale-free, causal triu-gather
        // mask + window term; window_left carried onto the op.
        rule(
            [
                eq(s_lower, (DimExpr::MVar("s")).lower()),
                eq(s_upper, (DimExpr::MVar("s")).upper()),
                s_lower.gt(0_i64),
                s_upper.lt(2_i64),
                Fact::from(flashinfer_prefill_dt(dt)),
                eq(
                    mask,
                    Ir::Op(
                        AddArgs {
                            shape: ([&(DimExpr::MVar("s")), &(DimExpr::MVar("c"))]).into(),
                            ..AddArgs::fresh()
                        },
                        [cmask, wterm],
                    ),
                ),
                Fact::from(fi_triu_mask(cmask, q_pos)),
                Fact::from(fi_window_term(wterm, q_pos2, window_left)),
                eq(q_pos, q_pos2),
                // SCALE-FREE: ?qk (the raw QK Sum) feeds the mask Add directly.
                eq(
                    masked,
                    Ir::Op(
                        AddArgs {
                            shape: ([nheads, &(DimExpr::MVar("s")), &(DimExpr::MVar("c"))]).into(),
                            b_strides: EList::ECons(0, mask_rest_strides),
                            ..AddArgs::fresh()
                        },
                        [qk, mask],
                    ),
                ),
                // mask Add → softmax internals → second matmul: one connected chain.
                Fact::from(const_like(sc_log2e, 1.442695)),
                eq(
                    mul,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads2, &(DimExpr::MVar("s")), hdim, &(DimExpr::MVar("c"))])
                                .into(),
                            ..MulArgs::fresh()
                        },
                        [soft, v_gqa],
                    ),
                ),
                eq(
                    output,
                    Ir::Op(
                        SumArgs {
                            shape: ([nheads3, &(DimExpr::MVar("s")), hdim2]).into(),
                            iters: "c".into(),
                            iter_stride: DimExpr::MIter(),
                            ..SumArgs::fresh()
                        },
                        [mul],
                    ),
                ),
                Fact::from(const_like(v_gqa_const, 1.0)),
                eq(
                    v_gqa,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads4, &(DimExpr::MVar("c")), hdim3]).into(),
                            b_strides: ([0, 0, 0]).into(),
                            ..MulArgs::fresh()
                        },
                        [v_gathered, v_gqa_const],
                    ),
                ),
                eq(
                    mul2,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads6, &(DimExpr::MVar("s")), &(DimExpr::MVar("c")), hdim4])
                                .into(),
                            ..MulArgs::fresh()
                        },
                        [q, k_gqa],
                    ),
                ),
                Fact::from(const_like(k_gqa_const, 1.0)),
                eq(
                    k_gqa,
                    Ir::Op(
                        MulArgs {
                            shape: ([nheads7, hdim5, &(DimExpr::MVar("c"))]).into(),
                            b_strides: ([0, 0, 0]).into(),
                            ..MulArgs::fresh()
                        },
                        [k_gathered, k_gqa_const],
                    ),
                ),
                eq(dt, q.dtype()),
                eq(dt, k_cache.dtype()),
                eq(dt, v_cache.dtype()),
                Fact::from(flashinfer_cache_pair(k_cache, v_cache)),
                Fact::from(flashinfer_head_dim_ok_16bit(hdim)),
            ],
            (
                sliding_attention,
                union(output, sliding_attention),
                set(sliding_attention.dtype(), dt),
            ),
        )
        .label("FlashInfer gemma sliding triu-gather attention"),
    ]
}

#[ruleset]
pub(crate) fn kernel_fuse_late_pre(
    cmask: &Ir,
    col_offsets: &Ir,
    q_pos: &Ir,
    neg_mask_scale: &Ir,
    neg_mask_scale_val: &egg::F64,
    blocked_bool: &Ir,
    tri_row_strides: &EList,
    tri_col_strides: &EList,
    blocked_out_strides: &EList,
    size17: &DimExpr,
    tri_iota: &Ir,
    mask_offset: &Ir,
    mask: &Ir,
    mask_allowed: &Ir,
    mask_scale_const: &Ir,
    mask_scale_val: &egg::F64,
    q_pos_cmp_strides: &EList,
    col_cmp_strides: &EList,
    w_negbig: &Ir,
    wterm: &Ir,
    w_negbig_val: &egg::F64,
    w_lt_out_strides: &EList,
    size20: &DimExpr,
    size21: &DimExpr,
    w_qpos: &Ir,
    out_strides172: &EList,
    window_left: &egg::F64,
) -> Vec<Rule> {
    let columns = &Ir::Op(OpKind::Iota(DimExpr::MIter(), "c"), IList::INil());
    let ctx_scalar = &Ir::Op(OpKind::Iota("c", 1), IList::INil());
    let blocked_f32 = &Ir::Op(CastArgs::fresh(), [blocked_bool]);
    let tri_iota_f32 = &Ir::Op(OpKind::Cast(size17, DType::F32()), [tri_iota]);
    let zero_const = &Ir::Op(OpKind::Constant(0.0), IList::INil());
    let neg_one_const = &Ir::Op(OpKind::Constant(-1.0), IList::INil());
    let mask_scaled_allowed = &Ir::Op(MulArgs::fresh(), [mask_allowed, mask_scale_const]);
    let w_col_iota = &Ir::Op(OpKind::Iota(DimExpr::MIter(), "c"), IList::INil());
    let w_qpos_f = &Ir::Op(OpKind::Cast(size21, DType::F32()), [w_qpos]);
    let w_window = &Ir::Op(OpKind::Constant(window_left), IList::INil());
    let w_negone = &Ir::Op(OpKind::Constant(-1.0), IList::INil());
    let scaled_blocked = &Ir::Op(
        MulArgs {
            shape: ([&(DimExpr::MVar("c")), &(DimExpr::MVar("c"))]).into(),
            ..MulArgs::fresh()
        },
        [blocked_f32, neg_mask_scale],
    );
    let row_offsets = &Ir::Op(
        MulArgs {
            shape: ([&(DimExpr::MVar("s"))]).into(),
            ..MulArgs::fresh()
        },
        [q_pos, ctx_scalar],
    );
    let zero_offset = &Ir::Op(
        MulArgs {
            shape: ([&(DimExpr::MVar("c")), &(DimExpr::MVar("c"))]).into(),
            ..MulArgs::fresh()
        },
        [zero_const, neg_one_const],
    );
    let w_col_f = &Ir::Op(OpKind::Cast(size20, DType::F32()), [w_col_iota]);
    let w_neg = &Ir::Op(
        OpKind::Mul([&(DimExpr::MVar("s"))], [0], [0], out_strides172),
        [w_window, w_negone],
    );
    let mask_indices = &Ir::Op(
        AddArgs {
            shape: ([&(DimExpr::MVar("s")), &(DimExpr::MVar("c"))]).into(),
            ..AddArgs::fresh()
        },
        [row_offsets, col_offsets],
    );
    let tri_iota_minus_zero = &Ir::Op(
        AddArgs {
            shape: ([&(DimExpr::MVar("c")), &(DimExpr::MVar("c"))]).into(),
            ..AddArgs::fresh()
        },
        [tri_iota_f32, zero_offset],
    );
    let w_lo = &Ir::Op(
        AddArgs {
            shape: ([&(DimExpr::MVar("s"))]).into(),
            ..AddArgs::fresh()
        },
        [w_qpos_f, w_neg],
    );
    let w_lt = &Ir::Op(
        OpKind::LessThan(
            [&(DimExpr::MVar("s")), &(DimExpr::MVar("c"))],
            [&(DimExpr::MNum(0)), &(DimExpr::MIter())],
            [&(DimExpr::MIter()), &(DimExpr::MNum(0))],
            w_lt_out_strides,
        ),
        [w_col_f, w_lo],
    );
    let w_bool_cast = &Ir::Op(CastArgs::fresh(), [w_lt]);
    vec![
        // ── Gemma-4 variants ──
        // Scale-free scores (attention scaling 1.0) and, on sliding layers, a window
        // term added to the causal mask. The mask chains are staged into relation
        // facts (anchored at their -1e10 constants), and the attention skeleton
        // matches the softmax internals so the whole rule is ONE connected chain —
        // an unconnected ?masked/?soft pair cross-products across the distinct layer
        // instances of a rolled multi-layer body and never finishes joining.
        rule(
            (
                eq(
                    cmask,
                    Ir::Op(
                        GatherArgs {
                            index_shape: ([&(DimExpr::MVar("s")), &(DimExpr::MVar("c"))]).into(),
                            data_shape: ([&(DimExpr::MVar("c")), &(DimExpr::MVar("c"))]).into(),
                            ..GatherArgs::fresh()
                        },
                        [mask_indices, scaled_blocked],
                    ),
                ),
                eq(col_offsets, columns),
                Fact::from(const_like(neg_mask_scale, neg_mask_scale_val)),
                neg_mask_scale_val.lt(-9999999999.0),
                neg_mask_scale_val.gt(-10000000001.0),
                eq(
                    blocked_bool,
                    Ir::Op(
                        OpKind::LessThan(
                            [&(DimExpr::MVar("c")), &(DimExpr::MVar("c"))],
                            tri_row_strides,
                            tri_col_strides,
                            blocked_out_strides,
                        ),
                        [tri_iota_f32, tri_iota_minus_zero],
                    ),
                ),
                eq(tri_iota, columns),
            ),
            (fi_triu_mask(cmask, q_pos),),
        )
        .label("FlashInfer triu mask fact"),
        // Legacy decode mask spelling: Add(Mul(allowed, 1e10), offset). The
        // "batch decode attention" island rule anchors at this fact.
        rule(
            (
                eq(
                    mask,
                    Ir::Op(AddArgs::fresh(), [mask_scaled_allowed, mask_offset]),
                ),
                Fact::from(const_like(mask_scale_const, mask_scale_val)),
                mask_scale_val.gt(9999999999.0),
                mask_scale_val.lt(10000000001.0),
            ),
            (fi_scaled_offset_mask(mask),),
        )
        .label("FlashInfer scaled-offset mask fact"),
        // Direct causal mask spelling:
        // (q_pos[:, None] < arange(c)[None, :]).cast(F32) * -1e10
        // The "direct causal attention" island rule anchors at this fact.
        rule(
            (
                eq(
                    mask,
                    Ir::Op(
                        MulArgs {
                            shape: ([&(DimExpr::MVar("s")), &(DimExpr::MVar("c"))]).into(),
                            ..MulArgs::fresh()
                        },
                        [blocked_f32, neg_mask_scale],
                    ),
                ),
                Fact::from(const_like(neg_mask_scale, neg_mask_scale_val)),
                neg_mask_scale_val.lt(-9999999999.0),
                neg_mask_scale_val.gt(-10000000001.0),
                eq(
                    blocked_bool,
                    Ir::Op(
                        OpKind::LessThan(
                            [&(DimExpr::MVar("s")), &(DimExpr::MVar("c"))],
                            q_pos_cmp_strides,
                            col_cmp_strides,
                            blocked_out_strides,
                        ),
                        [q_pos, col_offsets],
                    ),
                ),
                eq(
                    col_offsets,
                    Ir::Op(OpKind::Iota(DimExpr::MIter(), "c"), IList::INil()),
                ),
            ),
            (fi_direct_causal_mask(mask, q_pos),),
        )
        .label("FlashInfer direct causal mask fact"),
        // window term: (col < q_pos - (W-1)) cast * -1e10; the constant W-1
        // is FlashInfer's window_left.
        rule(
            (
                eq(
                    wterm,
                    Ir::Op(
                        MulArgs {
                            shape: ([&(DimExpr::MVar("s")), &(DimExpr::MVar("c"))]).into(),
                            ..MulArgs::fresh()
                        },
                        [w_bool_cast, w_negbig],
                    ),
                ),
                Fact::from(const_like(w_negbig, w_negbig_val)),
                w_negbig_val.lt(-9999999999.0),
                w_negbig_val.gt(-10000000001.0),
            ),
            (fi_window_term(wterm, w_qpos, window_left),),
        )
        .label("FlashInfer window term fact"),
    ]
}

pub(crate) fn initial_actions() -> Vec<Action> {
    vec![
        Action::from(flashinfer_head_dim_ok(DimExpr::MNum(64))),
        Action::from(flashinfer_head_dim_ok(DimExpr::MNum(128))),
        Action::from(flashinfer_head_dim_ok(DimExpr::MNum(256))),
        Action::from(flashinfer_head_dim_ok_16bit(DimExpr::MNum(64))),
        Action::from(flashinfer_head_dim_ok_16bit(DimExpr::MNum(128))),
        Action::from(flashinfer_head_dim_ok_16bit(DimExpr::MNum(256))),
        Action::from(flashinfer_head_dim_ok_16bit(DimExpr::MNum(512))),
        Action::from(flashinfer_prefill_dt(DType::F16())),
        Action::from(flashinfer_prefill_dt(DType::Bf16())),
    ]
}
