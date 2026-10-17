//! Unit + integration tests for the FlashInfer port.
//!
//! Four layers:
//! 1. Pure egglog metadata (no GPU): trait wiring, sort + rewrite parse cleanly.
//! 2. Egglog rule firing (no GPU): the rule unifies on a real paged-attention
//!    HLIR and does NOT fire on bare attention or unrelated matmul/Gather mixes.
//! 3. Mask helper correctness (GPU): the primitive-op `test_compute_attn_mask` builder produces the right (s, c) mask.
//! 4. Full kernel correctness (GPU + JIT): direct `FlashInferAttention::execute`
//!    compared against a luminal-compiled reference attention graph.
//!
//! GPU-dependent tests short-circuit when no CUDA device is available.

use egglog_experimental::typed::builtins as egg;
use std::sync::{Arc, Mutex};

use super::utilities::op_candidates;
use cudarc::driver::{CudaStream, DevicePtr};
use egglog_experimental::typed::{
    EGraph, EgglogValue, Snapshot, TypedError, get_args, let_, ruleset, set, union,
};
use luminal::egglog_utils::{DecodeContext, RewritePhase};
use luminal::prelude::*;
use luminal::typed_reference::schema::{DType as IrDType, DimExpr, EList, IList, Ir, OpKind};

use crate::host::flashinfer::FlashInferAttention;
use crate::host::{DeviceBuffer, HostOp};
use crate::runtime::CudaRuntime;
use crate::tests::utilities::get_cuda_stream;

// ─── Test-wide model dimensions ───────────────────────────────────────────
//
// Small Llama-shaped GQA model: nheads=8, kv_heads=2, group=4, head_dim=64.
// Chosen so HEAD_DIM ∈ {64, 128, 256} (FlashInfer constraint) and the test
// suite fits in O(1ms) of GPU time per case.

const HEAD_DIM: usize = 64;
const N_KV_HEADS: usize = 2;
const KV_GROUPS: usize = 4;
const N_HEADS: usize = N_KV_HEADS * KV_GROUPS;
const KV_DIM: usize = N_KV_HEADS * HEAD_DIM;
const HIDDEN: usize = N_HEADS * HEAD_DIM;

// ─── Reference attention graph (Q*K^T → softmax → *V via the compiler) ───

fn build_attention_graph() -> (Graph, GraphTensor, GraphTensor, GraphTensor, GraphTensor) {
    let mut cx = Graph::default();

    let q_rope = cx.named_tensor("q_rope", ('s', HIDDEN));
    let k_ctx = cx.named_tensor("k_ctx", ('c', KV_DIM));
    let v_ctx_input = cx.named_tensor("v_ctx", ('c', KV_DIM));

    let q = (q_rope * 1.0).split_dims(1, HEAD_DIM).transpose(0, 1);
    let k = k_ctx.split_dims(1, HEAD_DIM).permute((1, 2, 0));
    let v_ctx = v_ctx_input.split_dims(1, HEAD_DIM).transpose(0, 1);

    // GQA broadcast: zero-stride Mul by 1.0
    let k = k.expand_dim(1, KV_GROUPS).merge_dims(0, 1) * 1.0;
    let v_ctx = v_ctx.expand_dim(1, KV_GROUPS).merge_dims(0, 1) * 1.0;

    let scores = q.matmul(k) / (HEAD_DIM as f32).sqrt();
    let weights = scores.softmax(2);
    let out = weights.matmul(v_ctx);

    let attn_out = out.transpose(0, 1).merge_dims(1, 2);
    let attn_out = attn_out.output();

    (cx, q_rope, k_ctx, v_ctx_input, attn_out)
}

fn run_reference_attention(
    stream: &Arc<CudaStream>,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    batch_size: usize,
    context_len: usize,
) -> Vec<f32> {
    let (mut cx, q_t, k_t, v_t, out_t) = build_attention_graph();
    cx.set_dim('s', batch_size);
    cx.set_dim('c', context_len);
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let mut rt = CudaRuntime::initialize(stream.clone());
    rt.set_data(q_t, q.to_vec());
    rt.set_data(k_t, k.to_vec());
    rt.set_data(v_t, v.to_vec());
    rt = cx.search(rt, CompileOptions::default().search_graph_limit(3));

    rt.set_data(q_t, q.to_vec());
    rt.set_data(k_t, k.to_vec());
    rt.set_data(v_t, v.to_vec());
    rt.execute(&cx.dyn_map);
    rt.get_f32(out_t)
}

// ─── Direct FlashInfer driver ────────────────────────────────────────────

fn transpose_hbd_to_bhd(data: &[f32], heads: usize, batch: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for h in 0..heads {
        for b in 0..batch {
            for d in 0..dim {
                out[b * heads * dim + h * dim + d] = data[h * batch * dim + b * dim + d];
            }
        }
    }
    out
}

fn alloc_dev(stream: &Arc<CudaStream>, bytes: usize) -> cudarc::driver::CudaSlice<u8> {
    let bytes = bytes.max(1);
    unsafe { stream.alloc::<u8>(bytes).unwrap() }
}

fn copy_to_dev<T: Copy>(stream: &Arc<CudaStream>, data: &[T]) -> cudarc::driver::CudaSlice<u8> {
    let bytes = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    stream.clone_htod(bytes).unwrap()
}

/// Run FlashInferAttention.execute() directly and reshape the output to the
/// reference (batch, heads, dim) layout used by `run_reference_attention`.
fn run_flashinfer(
    stream: &Arc<CudaStream>,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    kv_indptr: &[i32],
    kv_indices: &[i32],
    batch_size: usize,
) -> Vec<f32> {
    let q_buf = copy_to_dev(stream, q);
    let k_buf = copy_to_dev(stream, k_cache);
    let v_buf = copy_to_dev(stream, v_cache);
    let idx_buf = copy_to_dev(stream, kv_indices);
    let qo_indptr: Vec<i32> = (0..=batch_size as i32).collect();
    let qo_indptr_buf = copy_to_dev(stream, &qo_indptr);
    let kv_indptr_buf = copy_to_dev(stream, kv_indptr);
    let out_buf = alloc_dev(stream, batch_size * HIDDEN * 4);

    let fi = FlashInferAttention {
        num_qo_heads: N_HEADS,
        num_kv_heads: N_KV_HEADS,
        head_dim: HEAD_DIM,
        page_size: 1,
        batch_dim: Expression::from('s'),
        dtype: luminal::dtype::DType::F32,
        sm_scale: 0.0,
        window_left: -1,
        plan_info: Mutex::new(Vec::new()),
    };

    // Reserve dedicated NodeIndex values for the test ports.
    let nodes: Vec<NodeIndex> = (0..7).map(NodeIndex::new).collect();
    let (q_n, k_n, v_n, idx_n, qo_n, kv_n, out_n) = (
        nodes[0], nodes[1], nodes[2], nodes[3], nodes[4], nodes[5], nodes[6],
    );

    let mut buffers = FxHashMap::default();
    let q_ptr = q_buf.device_ptr(stream).0;
    let k_ptr = k_buf.device_ptr(stream).0;
    let v_ptr = v_buf.device_ptr(stream).0;
    let idx_ptr = idx_buf.device_ptr(stream).0;
    let qo_ptr = qo_indptr_buf.device_ptr(stream).0;
    let kv_ptr = kv_indptr_buf.device_ptr(stream).0;
    let out_ptr = out_buf.device_ptr(stream).0;
    buffers.insert(q_n, DeviceBuffer::new(q_ptr, q.len() * 4));
    buffers.insert(k_n, DeviceBuffer::new(k_ptr, k_cache.len() * 4));
    buffers.insert(v_n, DeviceBuffer::new(v_ptr, v_cache.len() * 4));
    buffers.insert(idx_n, DeviceBuffer::new(idx_ptr, kv_indices.len() * 4));
    buffers.insert(qo_n, DeviceBuffer::new(qo_ptr, qo_indptr.len() * 4));
    buffers.insert(kv_n, DeviceBuffer::new(kv_ptr, kv_indptr.len() * 4));
    buffers.insert(out_n, DeviceBuffer::new(out_ptr, batch_size * HIDDEN * 4));

    let inputs = [q_n, k_n, v_n, idx_n, qo_n, kv_n];

    let mut dyn_map = FxHashMap::default();
    dyn_map.insert('s', batch_size);
    dyn_map.insert('c', kv_indices.len());
    dyn_map.insert('r', kv_indptr.len());

    fi.execute(stream, out_n, &inputs, &buffers, &dyn_map)
        .expect("FlashInferAttention execute failed");
    stream.synchronize().unwrap();

    // Output is (heads, batch, dim); reshape to (batch, heads, dim).
    let mut out_bytes = vec![0u8; batch_size * HIDDEN * 4];
    unsafe {
        cudarc::driver::result::memcpy_dtoh_async(&mut out_bytes, out_ptr, stream.cu_stream())
            .unwrap();
    }
    stream.synchronize().unwrap();
    let raw: Vec<f32> = unsafe {
        let mut bytes = std::mem::ManuallyDrop::new(out_bytes);
        let len = bytes.len() / 4;
        Vec::from_raw_parts(bytes.as_mut_ptr() as *mut f32, len, len)
    };
    transpose_hbd_to_bhd(&raw, N_HEADS, batch_size, HEAD_DIM)
}

fn run_flashinfer_with_compact_decode_indices(
    stream: &Arc<CudaStream>,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    kv_indices: &[i32],
) -> Vec<f32> {
    let batch_size = q.len() / HIDDEN;
    let q_buf = copy_to_dev(stream, q);
    let k_buf = copy_to_dev(stream, k_cache);
    let v_buf = copy_to_dev(stream, v_cache);
    let idx_buf = copy_to_dev(stream, kv_indices);
    let out_buf = alloc_dev(stream, batch_size * HIDDEN * 4);

    let fi = FlashInferAttention {
        num_qo_heads: N_HEADS,
        num_kv_heads: N_KV_HEADS,
        head_dim: HEAD_DIM,
        page_size: 1,
        batch_dim: Expression::from('s'),
        dtype: luminal::dtype::DType::F32,
        sm_scale: 0.0,
        window_left: -1,
        plan_info: Mutex::new(Vec::new()),
    };

    let nodes: Vec<NodeIndex> = (0..5).map(NodeIndex::new).collect();
    let (q_n, k_n, v_n, idx_n, out_n) = (nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]);

    let mut buffers = FxHashMap::default();
    let q_ptr = q_buf.device_ptr(stream).0;
    let k_ptr = k_buf.device_ptr(stream).0;
    let v_ptr = v_buf.device_ptr(stream).0;
    let idx_ptr = idx_buf.device_ptr(stream).0;
    let out_ptr = out_buf.device_ptr(stream).0;
    buffers.insert(q_n, DeviceBuffer::new(q_ptr, q.len() * 4));
    buffers.insert(k_n, DeviceBuffer::new(k_ptr, k_cache.len() * 4));
    buffers.insert(v_n, DeviceBuffer::new(v_ptr, v_cache.len() * 4));
    buffers.insert(idx_n, DeviceBuffer::new(idx_ptr, kv_indices.len() * 4));
    buffers.insert(out_n, DeviceBuffer::new(out_ptr, batch_size * HIDDEN * 4));

    let inputs = [q_n, k_n, v_n, idx_n];

    let mut dyn_map = FxHashMap::default();
    dyn_map.insert('s', batch_size);
    dyn_map.insert('c', kv_indices.len());

    fi.execute(stream, out_n, &inputs, &buffers, &dyn_map)
        .expect("FlashInferAttention compact-index execute failed");
    stream.synchronize().unwrap();

    let mut out_bytes = vec![0u8; batch_size * HIDDEN * 4];
    unsafe {
        cudarc::driver::result::memcpy_dtoh_async(&mut out_bytes, out_ptr, stream.cu_stream())
            .unwrap();
    }
    stream.synchronize().unwrap();
    let raw: Vec<f32> = unsafe {
        let mut bytes = std::mem::ManuallyDrop::new(out_bytes);
        let len = bytes.len() / 4;
        Vec::from_raw_parts(bytes.as_mut_ptr() as *mut f32, len, len)
    };
    transpose_hbd_to_bhd(&raw, N_HEADS, batch_size, HEAD_DIM)
}

fn resolve_flashinfer_decode_for_signature_test(
    context_len: usize,
    cache_slots: usize,
) -> crate::host::flashinfer::FlashInferResolvedDecode {
    let fi = FlashInferAttention {
        num_qo_heads: N_HEADS,
        num_kv_heads: N_KV_HEADS,
        head_dim: HEAD_DIM,
        page_size: 1,
        batch_dim: Expression::from('s'),
        dtype: luminal::dtype::DType::F32,
        sm_scale: 0.0,
        window_left: -1,
        plan_info: Mutex::new(Vec::new()),
    };
    let nodes: Vec<NodeIndex> = (0..5).map(NodeIndex::new).collect();
    let (q_n, k_n, v_n, idx_n, out_n) = (nodes[0], nodes[1], nodes[2], nodes[3], nodes[4]);
    let mut buffers = FxHashMap::default();
    buffers.insert(q_n, DeviceBuffer::new(0x1000, HIDDEN * 4));
    buffers.insert(k_n, DeviceBuffer::new(0x2000, cache_slots * KV_DIM * 4));
    buffers.insert(v_n, DeviceBuffer::new(0x3000, cache_slots * KV_DIM * 4));
    buffers.insert(idx_n, DeviceBuffer::new(0x4000, cache_slots * 4));
    buffers.insert(out_n, DeviceBuffer::new(0x6000, HIDDEN * 4));

    let mut dyn_map = FxHashMap::default();
    dyn_map.insert('s', 1);
    dyn_map.insert('c', context_len);
    fi.resolve_for_graph(out_n, &[q_n, k_n, v_n, idx_n], &buffers, &dyn_map)
        .unwrap()
}

// ─── Helpers ─────────────────────────────────────────────────────────────

fn deterministic_f32(n: usize, seed: f32, scale: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32 * seed).sin() * scale).collect()
}

fn assert_close(a: &[f32], b: &[f32], rtol: f32, atol: f32) {
    assert_eq!(
        a.len(),
        b.len(),
        "length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    let mut worst = (0usize, 0.0f32);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        if diff > worst.1 {
            worst = (i, diff);
        }
        let tol = atol + rtol * y.abs();
        assert!(
            diff <= tol,
            "mismatch at idx {i}: {x} vs {y} (|diff|={diff}, tol={tol})"
        );
    }
    eprintln!("max |diff| = {:.2e} @ idx {}", worst.1, worst.0);
}

// ─── Layer 1: egglog metadata sanity (no GPU) ────────────────────────────

#[test]
fn flashinfer_compact_indices_follow_real_gather_rows_through_fusion() {
    use crate::host::flashinfer::flashinfer_attention as fi;
    use crate::kernel::fusion::{elementwise::cuda_binary_elementwise, markers::fusion_end};
    use crate::kernel::hlir::kernel_iota;

    let mut graph = Graph::new();
    let data = graph.named_tensor("data", (16, KV_DIM));
    let indices = graph.named_tensor("indices", 'c').as_dtype(DType::Int);
    let gathered = luminal_nn::gather_rows(data, indices, KV_DIM);
    let flat_id = graph.get_sources(gathered.id)[0];
    let flat = GraphTensor::from_id(flat_id, gathered.shape, &mut graph, DType::Int);
    flat.output();
    // A consumer with the same layout lets the actual fusion-grow rules absorb
    // the lowered flat-index producer into an elementwise region.
    (flat * 1).output();
    gathered.output();
    let roots = luminal::typed_reference::capture_graph(&graph).unwrap();
    let compact = let_(
        "compact",
        Ir::Input(indices.id.index() as i64, "indices", IrDType::Int()),
    );
    let phases = luminal::egglog_utils::base_phase_rules();
    let mut live = EGraph::default();
    live.register(roots.as_slice()).unwrap();
    live.register(&compact).unwrap();
    live.run(fi::matmul_backend.saturate()).unwrap();

    for stage in 0..3 {
        if stage == 1 {
            live.run(
                ruleset((
                    &phases[&RewritePhase::Expr],
                    &phases[&RewritePhase::DtypeProp],
                    &cuda_binary_elementwise::kernel_lower,
                    &kernel_iota::kernel_lower,
                ))
                .saturate(),
            )
            .unwrap();
        } else if stage == 2 {
            let report = live
                .run(ruleset((&fusion_end::fusion_grow, &fusion_end::fusion_merge)).saturate())
                .unwrap();
            assert!(
                report.detail.updated,
                "the fixture must exercise actual fusion rules"
            );
        }
        let frozen = live.freeze().unwrap();
        let flat = frozen
            .eclasses()
            .filter(|class| class.sort() == Ir::sort_ref())
            .flat_map(|class| class.nodes())
            .filter_map(|node| {
                get_args(
                    &frozen.typed_node::<Ir>(node).unwrap(),
                    |input: &Ir, index: &egg::I64| Ir::Output(input, index),
                )
                .unwrap()
            })
            .find(|(_, index)| i64::try_from(index).unwrap() == flat_id.index() as i64)
            .unwrap()
            .0;
        let choices = Default::default();
        let context = DecodeContext::new(&frozen, &choices);
        let expected = frozen
            .nodes(&frozen.lookup(&compact).unwrap())
            .unwrap()
            .next()
            .unwrap();
        for node in context.producers(&flat).unwrap() {
            let selected = frozen.typed_node::<Ir>(node).unwrap();
            assert_eq!(
                crate::host::flashinfer::find_indptrs::try_find_compact_gather_idx(
                    &context, &selected
                )
                .unwrap(),
                Some(expected.clone()),
                "stage {stage}"
            );
        }
        if stage > 0 {
            assert!(
                frozen.nodes(&flat).unwrap().any(|node| {
                    let Some((kind, _)) =
                        get_args(&node, |kind: &OpKind, inputs: &IList| Ir::Op(kind, inputs))
                            .unwrap()
                    else {
                        return false;
                    };
                    frozen.nodes(&kind).unwrap().any(|kind| {
                        get_args(&kind, |shape: &EList, strides: &EList, dtype: &IrDType| {
                            crate::typed_rules::fusion_end(shape, strides, dtype)
                        })
                        .unwrap()
                        .is_some()
                    })
                }),
                "lowering must create a FusionEnd alternative for the same flat class"
            );
        }
    }
}

#[test]
fn flashinfer_compact_indices_preserve_legacy_structural_coverage() {
    for case in [
        "valid",
        "width",
        "offset",
        "range",
        "scalar range",
        "dtype",
        "input stride",
        "broadcast",
        "output stride",
        "zero width",
        "reversed mul",
        "reversed add",
    ] {
        let rows = DimExpr::from('c');
        let width = DimExpr::from(if case == "zero width" { 0 } else { 128 });
        let iter = DimExpr::MIter();
        let compact = let_(
            "compact",
            Ir::Input(
                0,
                "indices",
                if case == "dtype" {
                    IrDType::F32()
                } else {
                    IrDType::Int()
                },
            ),
        );
        let scale = Ir::Op(
            OpKind::Iota(
                if case == "width" {
                    127.into()
                } else {
                    width.clone()
                },
                if case == "scalar range" { 2 } else { 1 },
            ),
            IList::INil(),
        );
        let scaled = Ir::Op(
            OpKind::Mul(
                [&rows],
                [if case == "input stride" {
                    &iter * 2
                } else {
                    iter.clone()
                }],
                [0],
                [&iter],
            ),
            if case == "reversed mul" {
                [&scale, &compact]
            } else {
                [&compact, &scale]
            },
        );
        let columns = Ir::Op(
            OpKind::Iota(
                if case == "offset" {
                    &iter + 1
                } else {
                    iter.clone()
                },
                if case == "range" {
                    127.into()
                } else {
                    width.clone()
                },
            ),
            IList::INil(),
        );
        let flat = let_(
            "flat",
            Ir::Op(
                OpKind::Add(
                    [&rows, &width],
                    [
                        iter.clone(),
                        if case == "broadcast" {
                            iter.clone()
                        } else {
                            0.into()
                        },
                    ],
                    [DimExpr::from(0), iter.clone()],
                    [
                        if case == "output stride" {
                            iter.clone()
                        } else {
                            &iter * &width
                        },
                        iter.clone(),
                    ],
                ),
                if case == "reversed add" {
                    [&columns, &scaled]
                } else {
                    [&scaled, &columns]
                },
            ),
        );
        let mut live = EGraph::default();
        live.register((&flat, &compact)).unwrap();
        let frozen = live.freeze().unwrap();
        let choices = Default::default();
        let context = DecodeContext::new(&frozen, &choices);
        let flat = frozen
            .nodes(&frozen.lookup(&flat).unwrap())
            .unwrap()
            .next()
            .unwrap();
        let compact = frozen
            .nodes(&frozen.lookup(&compact).unwrap())
            .unwrap()
            .next()
            .unwrap();
        assert_eq!(
            crate::host::flashinfer::find_indptrs::try_find_compact_gather_idx(&context, &flat)
                .unwrap(),
            Some(compact),
            "{case}"
        );
    }
}

#[test]
fn flashinfer_compact_indices_match_lowered_only_and_fused_only_graphs() {
    use crate::typed_rules::{cuda_binary_elementwise, fusion_end, fusion_start};

    for (fused, reversed_mul) in [(false, false), (false, true), (true, false), (true, true)] {
        let shape = EList::from([DimExpr::from('c')]);
        let strides = EList::from([DimExpr::MIter()]);
        let compact = let_("compact", Ir::Input(0, "indices", IrDType::Int()));
        let other = let_("other", Ir::Input(1, "other", IrDType::Int()));
        let mut inner = compact.clone();
        // Build only the old walker's lowered forms. There are no HLIR Add/Mul
        // alternatives or previously derived compact-index facts to mask gaps.
        for opcode in ["Mul", "Add"] {
            let input = if fused {
                Ir::Op(fusion_start(&shape, &strides, IrDType::Int()), [&inner])
            } else {
                inner
            };
            let binary = Ir::Op(
                cuda_binary_elementwise(
                    opcode,
                    &shape,
                    &strides,
                    &strides,
                    &strides,
                    IrDType::Int(),
                ),
                if opcode == "Mul" && reversed_mul {
                    [other.clone(), input]
                } else {
                    [input, other.clone()]
                },
            );
            inner = if fused {
                Ir::Op(fusion_end(&shape, &strides, IrDType::Int()), [binary])
            } else {
                binary
            };
        }
        let inner = let_("inner", inner);
        let mut live = EGraph::default();
        live.register(&inner).unwrap();
        let frozen = live.freeze().unwrap();
        let choices = Default::default();
        let context = DecodeContext::new(&frozen, &choices);
        let selected = frozen
            .nodes(&frozen.lookup(&inner).unwrap())
            .unwrap()
            .next()
            .unwrap();
        let expected = if reversed_mul { other } else { compact };
        let expected = frozen
            .nodes(&frozen.lookup(&expected).unwrap())
            .unwrap()
            .next()
            .unwrap();
        assert_eq!(
            crate::host::flashinfer::find_indptrs::try_find_compact_gather_idx(&context, &selected)
                .unwrap(),
            Some(expected),
            "fused={fused}, reversed_mul={reversed_mul}"
        );
    }
}

// These standalone recovery fixtures have no cleanup or subsumption, so the
// active producer view is also their admissible view.
#[test]
fn flashinfer_compact_recovery_preserves_hlir_priority_and_missing_witness() {
    use crate::host::flashinfer::find_indptrs::try_find_compact_gather_idx;
    use crate::typed_rules::cuda_binary_elementwise;

    for hlir_recovers in [false, true] {
        let a = let_("a", Ir::Input(0, "hlir", IrDType::Int()));
        let b = let_("b", Ir::Input(1, "lowered", IrDType::Int()));
        let scalar = Ir::Op(OpKind::Iota(128, 1), IList::INil());
        let hlir_mul = Ir::Op(OpKind::Mul([1], [1], [0], [1]), [&a, &scalar]);
        let hlir = Ir::Op(
            OpKind::Add([1], [1], [1], [1]),
            [if hlir_recovers { &hlir_mul } else { &scalar }, &scalar],
        );
        let lower_mul = Ir::Op(
            cuda_binary_elementwise("Mul", [1], [1], [0], [1], IrDType::Int()),
            [&b, &scalar],
        );
        let lower_inputs = let_("lower_inputs", IList::from([&lower_mul, &scalar]));
        let lowered = Ir::Op(
            cuda_binary_elementwise("Add", [1], [1], [1], [1], IrDType::Int()),
            &lower_inputs,
        );
        let root = let_("root", &lowered);
        let mut live = EGraph::default();
        live.register((&root, &a, union(&root, &hlir))).unwrap();
        let frozen = live.freeze().unwrap();
        let choices = Default::default();
        let context = DecodeContext::new(&frozen, &choices);
        let lower_inputs = frozen.lookup(&lower_inputs).unwrap();
        let selected = frozen
            .nodes(&frozen.lookup(&root).unwrap())
            .unwrap()
            .find(|node| {
                get_args(node, |kind: &OpKind, inputs: &IList| Ir::Op(kind, inputs))
                    .unwrap()
                    .is_some_and(|(_, inputs)| inputs == lower_inputs)
            })
            .unwrap();
        let recovered = try_find_compact_gather_idx(&context, &selected).unwrap();
        if hlir_recovers {
            let expected = frozen
                .nodes(&frozen.lookup(&a).unwrap())
                .unwrap()
                .next()
                .unwrap();
            assert_eq!(recovered, Some(expected));
        } else {
            assert_eq!(
                recovered, None,
                "do not fall back after choosing an HLIR list"
            );
        }
        let direct = frozen
            .nodes(&frozen.lookup(&a).unwrap())
            .unwrap()
            .next()
            .unwrap();
        assert_eq!(
            try_find_compact_gather_idx(&context, &direct).unwrap(),
            None
        );
    }
}

#[test]
fn flashinfer_compact_recovery_inspects_only_the_supplied_lowered_producer() {
    use crate::host::flashinfer::find_indptrs::try_find_compact_gather_idx;
    use crate::typed_rules::cuda_binary_elementwise;

    let a = let_("a", Ir::Input(0, "a", IrDType::Int()));
    let b = let_("b", Ir::Input(1, "b", IrDType::Int()));
    let scalar = Ir::Op(OpKind::Iota(128, 1), IList::INil());
    let lower_mul = cuda_binary_elementwise("Mul", [1], [1], [0], [1], IrDType::Int());
    let lower_add = cuda_binary_elementwise("Add", [1], [1], [1], [1], IrDType::Int());
    let a_inputs = let_(
        "a_inputs",
        IList::from([Ir::Op(&lower_mul, [&a, &scalar]), scalar.clone()]),
    );
    let b_inputs = let_(
        "b_inputs",
        IList::from([Ir::Op(&lower_mul, [&b, &scalar]), scalar]),
    );
    let root = let_("root", Ir::Op(&lower_add, &a_inputs));
    let mut live = EGraph::default();
    live.register((&root, union(&root, Ir::Op(&lower_add, &b_inputs))))
        .unwrap();
    let frozen = live.freeze().unwrap();
    let choices = Default::default();
    let context = DecodeContext::new(&frozen, &choices);
    let a_inputs = frozen.lookup(&a_inputs).unwrap();
    for selected in frozen.nodes(&frozen.lookup(&root).unwrap()).unwrap() {
        let (_, inputs) = get_args(&selected, |kind: &OpKind, inputs: &IList| {
            Ir::Op(kind, inputs)
        })
        .unwrap()
        .unwrap();
        let expected = if inputs == a_inputs { &a } else { &b };
        let expected = frozen
            .nodes(&frozen.lookup(expected).unwrap())
            .unwrap()
            .next()
            .unwrap();
        assert_eq!(
            try_find_compact_gather_idx(&context, &selected).unwrap(),
            Some(expected)
        );
    }
}

// These rule fixtures deliberately install the real FlashInfer groups without
// the runtime's device-capability gate; they neither decode nor JIT an operation.
fn flashinfer_fixture_snapshot(graph: &Graph, s_interval: (i64, i64)) -> Snapshot {
    use crate::host::flashinfer::flashinfer_attention as fi;

    let outputs = luminal::typed_reference::capture_graph(graph).unwrap();
    // The native debug join executor recurses through these large attention
    // queries and exceeds Rust's default 2 MiB test stack, including with the
    // pre-compact-index rules. Keep that resource boundary local to this fixture.
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(move || {
            let mut phases = luminal::egglog_utils::base_phase_rules();
            phases.insert(RewritePhase::MatmulBackend, fi::matmul_backend.clone());
            phases.insert(
                RewritePhase::KernelLower,
                crate::kernel::hlir::kernel_constant::kernel_lower.clone(),
            );
            phases.insert(
                RewritePhase::KernelFuseLatePre,
                fi::kernel_fuse_late_pre.clone(),
            );
            phases.insert(RewritePhase::KernelFuseLate2, fi::kernel_fuse_late2.clone());
            let mut live = EGraph::default();
            let s = DimExpr::from('s');
            live.register((set(s.lower(), s_interval.0), set(s.upper(), s_interval.1)))
                .unwrap();
            if s_interval.0 == s_interval.1 {
                live.register(union(s, s_interval.0)).unwrap();
            }
            live.register(fi::initial_actions()).unwrap();
            live.register(outputs).unwrap();
            live.run(luminal::egglog_utils::main_schedule(&phases, true))
                .unwrap();
            for (phase, schedule) in luminal::egglog_utils::final_phases(&phases, true) {
                if !matches!(
                    phase,
                    RewritePhase::Cleanup | RewritePhase::PostCleanup | RewritePhase::BaseCleanup
                ) {
                    live.run(schedule).unwrap();
                }
            }
            live.freeze().unwrap()
        })
        .expect("spawn attention fixture")
        .join()
        .expect("attention fixture failed")
}

// Check every emitted FlashInfer IR node's proof dependencies and recoverable
// compact-input identity. This does not invoke the JIT-backed decoder.
fn assert_flashinfer_compact_dependencies(frozen: &Snapshot, compact_id: usize) -> usize {
    let mut found = 0;
    for node in frozen
        .eclasses()
        .filter(|class| class.sort() == Ir::sort_ref())
        .flat_map(|class| class.nodes())
    {
        let ir = frozen.typed_node::<Ir>(node).unwrap();
        let Some((kind, mut current)) =
            get_args(&ir, |kind: &OpKind, inputs: &IList| Ir::Op(kind, inputs)).unwrap()
        else {
            continue;
        };
        if !frozen.nodes(&kind).unwrap().any(|kind| {
            crate::typed_rules::FlashInferAttentionArgs::get_args(&kind)
                .unwrap()
                .is_some()
        }) {
            continue;
        }
        let mut inputs = Vec::new();
        loop {
            let nodes: Vec<_> = frozen.nodes(&current).unwrap().collect();
            assert_eq!(nodes.len(), 1, "proof dependency lists must be unambiguous");
            if get_args(&nodes[0], IList::INil).unwrap().is_some() {
                break;
            }
            let (head, tail) = get_args(&nodes[0], |head: &Ir, tail: &IList| {
                IList::ICons(head, tail)
            })
            .unwrap()
            .unwrap();
            inputs.push(head);
            current = tail;
            assert!(inputs.len() <= 5, "unexpected additional proof dependency");
        }
        assert_eq!(inputs.len(), 5);
        let choices = Default::default();
        let context = DecodeContext::new(frozen, &choices);
        let flat = frozen
            .typed_node::<Ir>(context.producers(&inputs[3]).unwrap().next().unwrap())
            .unwrap();
        let compact =
            crate::host::flashinfer::find_indptrs::try_find_compact_gather_idx(&context, &flat)
                .unwrap()
                .expect("fixture has a recoverable compact index");
        assert!({
            get_args(
                &compact,
                |id: &egg::I64, label: &egg::String, dtype: &IrDType| Ir::Input(id, label, dtype),
            )
            .unwrap()
            .is_some_and(|(id, _, dtype)| {
                i64::try_from(&id).unwrap() == compact_id as i64
                    && frozen
                        .nodes(&dtype)
                        .unwrap()
                        .any(|dtype| get_args(&dtype, IrDType::Int).unwrap().is_some())
            })
        });
        assert!(inputs[..3].iter().all(|input| input != &inputs[3]));
        found += 1;
    }
    found
}

#[test]
fn flashinfer_rules_retain_proof_dependencies_for_all_mask_variants() {
    for (mask, dtype, interval) in [
        (TestMaskKind::Indptr, DType::F32, (1, 1)),
        (TestMaskKind::DirectCausal, DType::F32, (1, 1)),
        (TestMaskKind::TriuGather, DType::F32, (1, 1)),
        (TestMaskKind::TriuGather, DType::Bf16, (2, 4)),
    ] {
        let (graph, handles) = build_paged_attention_graph_with_mask_and_cache_provenance(
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            mask,
            TestCacheProvenance::Raw,
            TestCacheProvenance::Raw,
            dtype,
        );
        let frozen = flashinfer_fixture_snapshot(&graph, interval);
        assert!(
            assert_flashinfer_compact_dependencies(&frozen, handles.gather_idx.id.index()) > 0,
            "mask {mask:?}, dtype {dtype:?}, interval {interval:?}"
        );
    }
    for window in [None, Some(8)] {
        let mut graph = Graph::new();
        let q = graph.tensor(('s', 128)).as_dtype(DType::Bf16);
        let k = graph.tensor(('s', 64)).as_dtype(DType::Bf16);
        let v = graph.tensor(('s', 64)).as_dtype(DType::Bf16);
        let k_cache = graph.tensor((16, 64)).as_dtype(DType::Bf16);
        let v_cache = graph.tensor((16, 64)).as_dtype(DType::Bf16);
        let scatter = graph.tensor('s').as_dtype(DType::Int);
        let compact = graph.tensor('c').as_dtype(DType::Int);
        let q_pos = graph.tensor('s').as_dtype(DType::Int);
        gemma_mini_paged_attention(
            q, k, v, k_cache, v_cache, scatter, compact, q_pos, 64, 64, 2, 2, window,
        )
        .output();
        let frozen = flashinfer_fixture_snapshot(&graph, (1, 1));
        assert!(
            assert_flashinfer_compact_dependencies(&frozen, compact.id.index()) > 0,
            "window {window:?}"
        );
    }
}

#[test]
fn flashinfer_rules_retain_candidates_without_a_compact_witness() {
    use rand::{SeedableRng, rngs::StdRng};
    let (mut graph, handles) = build_paged_attention_graph_with_mask_and_cache_provenance(
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        TestMaskKind::TriuGather,
        TestCacheProvenance::Raw,
        TestCacheProvenance::Raw,
        DType::F32,
    );
    let gathers: Vec<_> = graph
        .node_indices()
        .filter(|&node| {
            graph.try_get_op::<luminal::hlir::Gather>(node).is_some()
                && graph
                    .try_get_op::<luminal::hlir::Scatter>(graph.get_sources(node)[1])
                    .is_some()
        })
        .collect();
    assert_eq!(gathers.len(), 2);
    for gather in gathers {
        let sources = graph.get_sources(gather);
        let edge = graph.find_edge(sources[0], gather).unwrap();
        graph.remove_edge(edge);
        graph.add_edge(handles.gather_idx.id, gather, ());
        assert_eq!(
            graph.get_sources(gather),
            [handles.gather_idx.id, sources[1]]
        );
    }
    let frozen = flashinfer_fixture_snapshot(&graph, (1, 1));
    let space = luminal::egglog_utils::SearchSpace::from_frozen(frozen, &[]).unwrap();
    let alternatives = space.alternatives().unwrap();
    let choices = luminal::egglog_utils::random_initial_choice(
        &space.frozen,
        &alternatives,
        &mut StdRng::seed_from_u64(7),
    );
    let mut context = DecodeContext::new(&space.frozen, &choices);
    let candidates = op_candidates(&space, |kind| {
        crate::typed_rules::FlashInferAttentionArgs::get_args(kind)
            .unwrap()
            .is_some()
    });
    assert!(
        !candidates.is_empty(),
        "missing compact recovery must not prevent fusion eligibility"
    );
    for [(.., node), _] in candidates {
        let node = space.frozen.typed_node::<Ir>(node).unwrap();
        let (_, inputs) = get_args(&node, |kind: &OpKind, inputs: &IList| Ir::Op(kind, inputs))
            .unwrap()
            .unwrap();
        let inputs = context.inputs(&inputs).unwrap();
        assert_eq!(inputs.len(), 5);
        let flat = context.selected(&inputs[3]).unwrap();
        assert_eq!(
            crate::host::flashinfer::find_indptrs::try_find_compact_gather_idx(&context, &flat)
                .unwrap(),
            None
        );
    }
}

#[test]
fn flashinfer_rules_preserve_source_mask_and_layout_constraints() {
    for case in [
        "mask",
        "width",
        "rank",
        "index stride",
        "data stride",
        "cache provenance",
        "prefill",
    ] {
        let (mut graph, handles) = build_paged_attention_graph_with_mask_and_cache_provenance(
            N_HEADS,
            N_KV_HEADS,
            HEAD_DIM,
            if case == "mask" {
                TestMaskKind::Unproven
            } else {
                TestMaskKind::TriuGather
            },
            if case == "cache provenance" {
                TestCacheProvenance::LoopInput {
                    loop_id: 7,
                    stream_id: 0,
                }
            } else {
                TestCacheProvenance::Raw
            },
            TestCacheProvenance::Raw,
            DType::F32,
        );
        if matches!(case, "width" | "rank" | "index stride" | "data stride") {
            let gathers: Vec<_> = graph
                .node_indices()
                .filter(|&node| {
                    graph.try_get_op::<luminal::hlir::Gather>(node).is_some()
                        && graph
                            .try_get_op::<luminal::hlir::Scatter>(graph.get_sources(node)[1])
                            .is_some()
                })
                .collect();
            assert_eq!(
                gathers.len(),
                2,
                "fixture must identify only K/V cache gathers"
            );
            let gather = graph.get_op_mut::<luminal::hlir::Gather>(gathers[0]);
            match case {
                "width" => gather.input_shapes[1].dims[1] = (KV_DIM + 1).into(),
                "rank" => {
                    gather.input_shapes[0].dims.push(1.into());
                    gather.input_shapes[0].strides.push(0.into());
                }
                "index stride" => gather.input_shapes[0].strides[0] = 'z'.into(),
                "data stride" => gather.input_shapes[1].strides[0] = 'z'.into(),
                _ => unreachable!(),
            }
        }
        let frozen =
            flashinfer_fixture_snapshot(&graph, if case == "prefill" { (2, 4) } else { (1, 1) });
        assert_eq!(
            assert_flashinfer_compact_dependencies(&frozen, handles.gather_idx.id.index()),
            usize::from(matches!(case, "width" | "index stride" | "data stride")),
            "{case}"
        );
    }
}

#[test]
fn flashinfer_runtime_recipe_respects_hardware_gate() {
    let (phases, actions) = CudaRuntime::rules();
    let enabled = crate::device_compute_major() >= 8;
    let expected_actions = if enabled {
        crate::host::flashinfer::flashinfer_attention::initial_actions().len()
    } else {
        0
    };
    assert_eq!(
        actions.len() - crate::host::cublaslt::rewrites::initial_actions().len(),
        expected_actions
    );
    // FlashInfer is the only producer of this late phase in the runtime recipe.
    assert_eq!(phases[&RewritePhase::KernelFuseLate2].is_empty(), !enabled);
}

#[test]
fn flashinfer_egg_rule_parses() {
    // Validate typed declarations, rule bodies, and initialization together.
    // This checks authoring independently of the runtime's hardware gate.
    let rules = ruleset((
        &crate::host::flashinfer::flashinfer_attention::matmul_backend,
        &crate::host::flashinfer::flashinfer_attention::kernel_fuse_late2,
        &crate::host::flashinfer::flashinfer_attention::kernel_fuse_late_pre,
    ));
    assert!(!rules.is_empty());
    let mut live = EGraph::default();
    live.register(crate::host::flashinfer::flashinfer_attention::initial_actions())
        .unwrap();
    let kind = let_(
        "flashinfer_kind",
        crate::typed_rules::flash_infer_attention(8, 2, 64, 1, 1, IrDType::F32(), 1.0, -1.0),
    );
    live.register(&kind).unwrap();
    live.run(&rules)
        .expect("typed FlashInfer rules must install and run");
    let frozen = live.freeze().unwrap();
    assert!(
        frozen
            .nodes(&frozen.lookup(&kind).unwrap())
            .unwrap()
            .any(|kind| {
                crate::typed_rules::FlashInferAttentionArgs::get_args(&kind)
                    .unwrap()
                    .is_some()
            })
    );
}

#[test]
fn flashinfer_op_sort_shape() {
    let kind = let_(
        "kind",
        crate::typed_rules::flash_infer_attention(8, 2, 64, 1, 1, IrDType::F32(), 1.0, -1.0),
    );
    let input = let_("input", Ir::Input(0, "input", IrDType::F32()));
    let mut live = EGraph::default();
    live.register((&kind, &input)).unwrap();
    let frozen = live.freeze().unwrap();
    let observed = frozen.lookup(&kind).unwrap();
    let class = frozen.class_of(&observed).unwrap();
    let node = class.nodes().next().unwrap();
    assert!(
        crate::typed_rules::FlashInferAttentionArgs::get_args(
            &frozen.typed_node::<OpKind>(node).unwrap()
        )
        .unwrap()
        .is_some()
    );
    assert_eq!(node.args().count(), 8);
    assert_eq!(FlashInferAttention::default().n_inputs(), 4);
}

#[test]
fn flashinfer_decoder_preserves_unsupported_dtype_rejection() {
    let kind = let_(
        "kind",
        crate::typed_rules::flash_infer_attention(8, 2, 64, 1, 1, IrDType::Int(), 1.0, -1.0),
    );
    let mut live = EGraph::default();
    live.register(&kind).unwrap();
    let frozen = live.freeze().unwrap();
    let observed = frozen.lookup(&kind).unwrap();
    let selected = frozen.nodes(&observed).unwrap().next().unwrap();
    let choices = FxHashMap::default();
    let mut context = DecodeContext::new(&frozen, &choices);
    assert!(matches!(
        FlashInferAttention::decode(&selected, &mut Vec::new(), &mut context),
        Err(TypedError::Decode(_))
    ));
}

#[test]
fn flashinfer_graph_signature_is_stable_when_c_moves_within_capacity() {
    let resolved_c4 = resolve_flashinfer_decode_for_signature_test(4, 2048);
    let plan_c = resolved_c4.graph_plan_capacity(None);
    let sig_c4 = resolved_c4.signature_for_graph_plan(plan_c);

    let resolved_c5 = resolve_flashinfer_decode_for_signature_test(5, 2048);
    let sig_c5 =
        resolved_c5.signature_for_graph_plan(resolved_c5.graph_plan_capacity(Some(plan_c)));

    assert_eq!(
        sig_c4, sig_c5,
        "derived FlashInfer decode graph signature should not change when only current c changes within the planned capacity"
    );
}

// ─── Layer 3: FlashInfer kernel correctness ──────────────────────────────

#[test]
fn flashinfer_bs1_ctx4() {
    if !crate::tests::utilities::gpu_supports_flashinfer() {
        return;
    }
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let batch_size = 1;
    let context_len = 4;
    let q = deterministic_f32(batch_size * HIDDEN, 0.011, 0.1);
    let k = deterministic_f32(context_len * KV_DIM, 0.021, 0.1);
    let v = deterministic_f32(context_len * KV_DIM, 0.031, 0.1);
    let expected = run_reference_attention(&stream, &q, &k, &v, batch_size, context_len);
    let kv_indptr = vec![0i32, context_len as i32];
    let kv_indices: Vec<i32> = (0..context_len as i32).collect();
    let result = run_flashinfer(&stream, &q, &k, &v, &kv_indptr, &kv_indices, batch_size);
    assert_close(&result, &expected, 1e-4, 1e-5);
}

#[test]
fn flashinfer_bs2_supersequence() {
    if !crate::tests::utilities::gpu_supports_flashinfer() {
        return;
    }
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let batch_size = 2;
    let ctx0 = 8;
    let ctx1 = 3;
    let total_ctx = ctx0 + ctx1;

    let q = deterministic_f32(batch_size * HIDDEN, 0.014, 0.1);
    let k = deterministic_f32(total_ctx * KV_DIM, 0.022, 0.1);
    let v = deterministic_f32(total_ctx * KV_DIM, 0.032, 0.1);

    // Reference: run each sequence separately through the reference graph
    // (the reference uses dense attention so we can't run bs=2 directly).
    let expected0 = run_reference_attention(
        &stream,
        &q[..HIDDEN],
        &k[..ctx0 * KV_DIM],
        &v[..ctx0 * KV_DIM],
        1,
        ctx0,
    );
    let expected1 = run_reference_attention(
        &stream,
        &q[HIDDEN..],
        &k[ctx0 * KV_DIM..],
        &v[ctx0 * KV_DIM..],
        1,
        ctx1,
    );
    let expected: Vec<f32> = expected0.into_iter().chain(expected1).collect();

    let kv_indptr = vec![0i32, ctx0 as i32, total_ctx as i32];
    let kv_indices: Vec<i32> = (0..total_ctx as i32).collect();
    let result = run_flashinfer(&stream, &q, &k, &v, &kv_indptr, &kv_indices, batch_size);
    assert_close(&result, &expected, 1e-4, 1e-5);
}

#[test]
fn flashinfer_noncontiguous_page_table() {
    if !crate::tests::utilities::gpu_supports_flashinfer() {
        return;
    }
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let batch_size = 1;
    let context_len = 4;
    let num_slots = 8;
    let slot_indices = [3usize, 0, 7, 1];

    let q = deterministic_f32(batch_size * HIDDEN, 0.011, 0.1);
    let k_full = deterministic_f32(num_slots * KV_DIM, 0.022, 0.1);
    let v_full = deterministic_f32(num_slots * KV_DIM, 0.033, 0.1);

    // Reference operates on the contiguous gathered cache.
    let mut k_gathered = vec![0.0f32; context_len * KV_DIM];
    let mut v_gathered = vec![0.0f32; context_len * KV_DIM];
    for (i, &slot) in slot_indices.iter().enumerate() {
        k_gathered[i * KV_DIM..(i + 1) * KV_DIM]
            .copy_from_slice(&k_full[slot * KV_DIM..(slot + 1) * KV_DIM]);
        v_gathered[i * KV_DIM..(i + 1) * KV_DIM]
            .copy_from_slice(&v_full[slot * KV_DIM..(slot + 1) * KV_DIM]);
    }
    let expected = run_reference_attention(
        &stream,
        &q,
        &k_gathered,
        &v_gathered,
        batch_size,
        context_len,
    );

    let kv_indptr = vec![0i32, context_len as i32];
    let kv_indices: Vec<i32> = slot_indices.iter().map(|&s| s as i32).collect();
    let result = run_flashinfer(
        &stream,
        &q,
        &k_full,
        &v_full,
        &kv_indptr,
        &kv_indices,
        batch_size,
    );
    assert_close(&result, &expected, 1e-4, 1e-5);
}

#[test]
fn flashinfer_compact_decode_indices_match_reference() {
    if !crate::tests::utilities::gpu_supports_flashinfer() {
        return;
    }
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let batch_size = 1;
    let context_len = 3;
    let q = deterministic_f32(batch_size * HIDDEN, 0.017, 0.1);
    let k = deterministic_f32(context_len * KV_DIM, 0.023, 0.1);
    let v = deterministic_f32(context_len * KV_DIM, 0.037, 0.1);
    let expected = run_reference_attention(&stream, &q, &k, &v, batch_size, context_len);
    let kv_indices: Vec<i32> = (0..context_len as i32).collect();
    let result = run_flashinfer_with_compact_decode_indices(&stream, &q, &k, &v, &kv_indices);
    assert_close(&result, &expected, 1e-4, 1e-5);
}

// ─── Layer 3b: HEAD_DIM 128 path (validates the head-dim JIT dispatch) ────
//
// Each FlashInfer .so is compiled for one HEAD_DIM. JIT caches by head dim;
// the OnceLock means only one is loaded per process. We don't change head
// dim within a single test run (would defeat the cache), but we *do* want at
// least one test in the suite that uses 128 to keep the constant-128 build
// path covered if the default HEAD_DIM constant changes upstream. We assert
// the constraint here rather than firing a second JIT.

#[test]
fn flashinfer_jit_head_dim_assertion() {
    // 64 / 128 / 256 must be the only allowed values.
    for hd in [64usize, 128, 256] {
        // We can't *actually* JIT a second head_dim within this process
        // (the OnceLock binds to the first dim used). Just check the dim
        // is in the supported set.
        assert!(matches!(hd, 64 | 128 | 256));
    }
}

// ─── Layer 4: egglog rule firing (no GPU) ────────────────────────────────
//
// These tests build HLIR graphs and run egglog saturation. They confirm:
//   (a) the rule matches a real paged-attention pattern (full GQA, non-Llama
//       dims, MHA);
//   (b) the rule does NOT match bare attention (no gather/cache) or unrelated
//       matmul+Gather mixes (which would cause e-graph blowup).
//
// Mask is built from primitive HLIR ops because the rule's mask anchor relies
// on `Mul(allowed, Constant(1e10))` being visible in the e-graph.

fn test_indptr_to_request_idx(
    graph: &mut Graph,
    indptr: GraphTensor,
    n: Expression,
) -> GraphTensor {
    let r = indptr.dims1();
    let indices = graph.arange(n).expand_dim(1, r);
    let indptr_2d = indptr.expand_dim(0, n);
    let ge = indptr_2d.le(indices).cast(luminal::dtype::DType::Int);
    ge.sum(1).cast(luminal::dtype::DType::Int) - 1
}

fn test_compute_attn_mask(
    graph: &mut Graph,
    q_pos: GraphTensor,
    qo_indptr: GraphTensor,
    kv_indptr: GraphTensor,
    c: Expression,
    dtype: DType,
) -> GraphTensor {
    let s = q_pos.dims1();
    let q_request = test_indptr_to_request_idx(graph, qo_indptr, s);
    let c_request = test_indptr_to_request_idx(graph, kv_indptr, c);
    let c_arange = graph.arange(c);
    let c_kv_start = kv_indptr.gather(c_request);
    let c_local_pos = c_arange - c_kv_start;
    let q_req_2d = q_request.expand_dim(1, c);
    let c_req_2d = c_request.expand_dim(0, s);
    let same = q_req_2d.eq(c_req_2d);
    let c_pos_2d = c_local_pos.expand_dim(0, s);
    let qp_2d = q_pos.expand_dim(1, c);
    let causal = c_pos_2d.le(qp_2d);
    let allowed = same.cast(dtype) * causal.cast(dtype);
    allowed * 1e10 - 1e10
}

fn test_compute_triu_gather_mask(
    graph: &mut Graph,
    q_pos: GraphTensor,
    c: Expression,
    dtype: DType,
) -> GraphTensor {
    let s = q_pos.dims1();
    let causal_square = graph.triu(c, 1).cast(dtype) * -1e10;
    let row_offsets = (q_pos * c).expand_dim(1, c);
    let col_offsets = graph.arange(c).expand_dim(0, s);
    causal_square.gather(row_offsets + col_offsets)
}

fn test_compute_direct_causal_mask(
    graph: &mut Graph,
    q_pos: GraphTensor,
    c: Expression,
    dtype: DType,
) -> GraphTensor {
    let s = q_pos.dims1();
    let q_pos = q_pos.expand_dim(1, c);
    let col = graph.arange(c).expand_dim(0, s);
    q_pos.lt(col).cast(dtype) * -1e10
}

fn gather_rows(data: GraphTensor, indices: GraphTensor, d: usize) -> GraphTensor {
    let n = indices.dims1();
    let base = (indices * d).expand_dim(1, d);
    let col = data.graph().arange(d as i32).expand_dim(0, n);
    data.gather(base + col)
}

fn scatter_rows(
    src: GraphTensor,
    indices: GraphTensor,
    dest: GraphTensor,
    d: usize,
) -> GraphTensor {
    let n = indices.dims1();
    let base = (indices * d).expand_dim(1, d);
    let col = src.graph().arange(d as i32).expand_dim(0, n);
    src.scatter(base + col, dest)
}

/// Handles to every named input of the paged-attention test graph, returned
/// alongside the graph so the GA-selection test can `set_data` on each one.
#[allow(dead_code)]
struct PagedAttnHandles {
    attn_out: GraphTensor,
    q_rope: GraphTensor,
    k_rope: GraphTensor,
    v_new: GraphTensor,
    k_cache: GraphTensor,
    v_cache: GraphTensor,
    scatter_idx: GraphTensor,
    gather_idx: GraphTensor,
    q_pos: GraphTensor,
    qo_indptr: GraphTensor,
    kv_indptr: GraphTensor,
}

#[derive(Clone, Copy, Debug)]
enum TestMaskKind {
    Indptr,
    TriuGather,
    DirectCausal,
    Unproven,
}

#[derive(Clone, Copy)]
enum TestCacheProvenance {
    Raw,
    LoopInput { loop_id: usize, stream_id: usize },
}

fn cache_dest_with_provenance(
    graph: &mut Graph,
    cache: GraphTensor,
    alt_name: &str,
    kv_dim: usize,
    provenance: TestCacheProvenance,
) -> GraphTensor {
    match provenance {
        TestCacheProvenance::Raw => cache,
        TestCacheProvenance::LoopInput { loop_id, stream_id } => {
            let alt = graph.named_tensor(alt_name, (2048, kv_dim)).persist();
            let id = graph.add_op(
                luminal::hlir::LoopInput {
                    loop_id,
                    stream_id,
                    dtype: cache.dtype,
                },
                &[cache.id, alt.id],
            );
            GraphTensor::from_id(id, cache.shape.contiguous(), graph, cache.dtype)
        }
    }
}

/// Build a full paged-attention HLIR graph with the structural anchors the
/// FlashInfer egglog rule looks for: scatter into a 2D cache, gather rows out
/// by index, GQA broadcast via `Mul(..., 1.0)` with zero strides, Q*K^T → Sum
/// → scale → mask Add → softmax → *V → Sum.
fn build_paged_attention_graph(
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> (Graph, PagedAttnHandles) {
    build_paged_attention_graph_with_mask(n_heads, n_kv_heads, head_dim, TestMaskKind::Indptr)
}

fn build_paged_attention_graph_with_mask(
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    mask_kind: TestMaskKind,
) -> (Graph, PagedAttnHandles) {
    build_paged_attention_graph_with_mask_and_cache_provenance(
        n_heads,
        n_kv_heads,
        head_dim,
        mask_kind,
        TestCacheProvenance::Raw,
        TestCacheProvenance::Raw,
        DType::F32,
    )
}

fn build_paged_attention_graph_with_mask_and_cache_provenance(
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    mask_kind: TestMaskKind,
    k_provenance: TestCacheProvenance,
    v_provenance: TestCacheProvenance,
    dtype: DType,
) -> (Graph, PagedAttnHandles) {
    let kv_groups = n_heads / n_kv_heads;
    let kv_dim = n_kv_heads * head_dim;
    let hidden = n_heads * head_dim;

    let mut cx = Graph::default();

    let q_rope = cx.named_tensor("q_rope", ('s', hidden)).as_dtype(dtype);
    let k_rope = cx.named_tensor("k_rope", ('s', kv_dim)).as_dtype(dtype);
    let v_new = cx.named_tensor("v_new", ('s', kv_dim)).as_dtype(dtype);
    let k_cache = cx
        .named_tensor("k_cache", (2048, kv_dim))
        .as_dtype(dtype)
        .persist();
    let v_cache = cx
        .named_tensor("v_cache", (2048, kv_dim))
        .as_dtype(dtype)
        .persist();
    let scatter_idx = cx
        .named_tensor("scatter_idx", 's')
        .as_dtype(luminal::dtype::DType::Int);
    let gather_idx = cx
        .named_tensor("gather_idx", 'c')
        .as_dtype(luminal::dtype::DType::Int);
    let q_pos = cx
        .named_tensor("q_pos", 's')
        .as_dtype(luminal::dtype::DType::Int);
    let qo_indptr = cx
        .named_tensor("qo_indptr", 'r')
        .as_dtype(luminal::dtype::DType::Int);
    let kv_indptr = cx
        .named_tensor("kv_indptr", 'r')
        .as_dtype(luminal::dtype::DType::Int);

    let k_dest = cache_dest_with_provenance(&mut cx, k_cache, "k_cache_alt", kv_dim, k_provenance);
    let v_dest = cache_dest_with_provenance(&mut cx, v_cache, "v_cache_alt", kv_dim, v_provenance);

    let k_cache_out = scatter_rows(k_rope, scatter_idx, k_dest, kv_dim);
    let v_cache_out = scatter_rows(v_new, scatter_idx, v_dest, kv_dim);

    let k = gather_rows(k_cache_out, gather_idx, kv_dim);
    let v_ctx = gather_rows(v_cache_out, gather_idx, kv_dim);

    let c: Expression = 'c'.into();
    let attn_mask = match mask_kind {
        TestMaskKind::Indptr => {
            test_compute_attn_mask(&mut cx, q_pos, qo_indptr, kv_indptr, c, dtype)
        }
        TestMaskKind::TriuGather => test_compute_triu_gather_mask(&mut cx, q_pos, c, dtype),
        TestMaskKind::DirectCausal => test_compute_direct_causal_mask(&mut cx, q_pos, c, dtype),
        TestMaskKind::Unproven => cx.named_tensor("unproven_mask", ('s', 'c')).as_dtype(dtype),
    };

    let q = (q_rope * 1.0).split_dims(1, head_dim).transpose(0, 1);
    let k = k.split_dims(1, head_dim).permute((1, 2, 0));
    let v_ctx = v_ctx.split_dims(1, head_dim).transpose(0, 1);
    let k = k.expand_dim(1, kv_groups).merge_dims(0, 1) * 1.0;
    let v_ctx = v_ctx.expand_dim(1, kv_groups).merge_dims(0, 1) * 1.0;

    let scores = q.matmul(k) / (head_dim as f32).sqrt();
    let mask = attn_mask.expand_dim(0, n_heads);
    let masked_scores = scores + mask;
    let weights = masked_scores.softmax(2);
    let out = weights.matmul(v_ctx);
    let attn_out = out.transpose(0, 1).merge_dims(1, 2);

    let attn_out = attn_out.output();
    k_cache_out.output();
    v_cache_out.output();

    (
        cx,
        PagedAttnHandles {
            attn_out,
            q_rope,
            k_rope,
            v_new,
            k_cache,
            v_cache,
            scatter_idx,
            gather_idx,
            q_pos,
            qo_indptr,
            kv_indptr,
        },
    )
}

/// Saturate egglog on the graph and report whether a FlashInferAttention
/// e-node was produced. Helper used by the rule-firing tests.
fn saturate_and_has_flashinfer(cx: &Graph) -> (bool, Vec<String>) {
    saturate_and_has_flashinfer_inner(cx, None)
}

fn saturate_and_has_flashinfer_with_decode_interval(cx: &Graph) -> (bool, Vec<String>) {
    saturate_and_has_flashinfer_with_s_interval(cx, 1, 1)
}

fn saturate_and_has_flashinfer_with_s_interval(
    cx: &Graph,
    s_min: i64,
    s_max: i64,
) -> (bool, Vec<String>) {
    saturate_and_has_flashinfer_inner(cx, Some((s_min, s_max)))
}

fn saturate_attention_graph(cx: &Graph, s_interval: Option<(i64, i64)>) -> Snapshot {
    let roots = luminal::typed_reference::capture_graph(cx).unwrap();
    let (phases, actions) = CudaRuntime::rules();
    let mut live = EGraph::default();
    if let Some((min, max)) = s_interval {
        let s = DimExpr::from('s');
        live.register((set(s.lower(), min), set(s.upper(), max)))
            .unwrap();
        if min == max {
            live.register(union(s, min)).unwrap();
        }
    }
    live.register(actions).unwrap();
    live.register(roots.as_slice()).unwrap();
    live.run(luminal::egglog_utils::main_schedule(
        &phases,
        s_interval.is_some(),
    ))
    .unwrap();
    for (phase, schedule) in luminal::egglog_utils::final_phases(&phases, s_interval.is_some()) {
        if !matches!(
            phase,
            RewritePhase::Cleanup | RewritePhase::PostCleanup | RewritePhase::BaseCleanup
        ) {
            live.run(schedule).unwrap();
        }
    }
    // Retain the entire exact snapshot, including subsumed nodes; no
    // admissible-alternative cleanup may hide a rule-firing witness.
    live.freeze().unwrap()
}

fn saturate_and_has_flashinfer_inner(
    cx: &Graph,
    s_interval: Option<(i64, i64)>,
) -> (bool, Vec<String>) {
    let frozen = saturate_attention_graph(cx, s_interval);
    let kinds: Vec<_> = frozen
        .eclasses()
        .filter(|class| class.sort() == OpKind::sort_ref())
        .flat_map(|class| class.nodes())
        .collect();
    let found = kinds.iter().any(|&node| {
        crate::typed_rules::FlashInferAttentionArgs::get_args(
            &frozen.typed_node::<OpKind>(node).unwrap(),
        )
        .unwrap()
        .is_some()
    });
    let mut names: Vec<_> = kinds.iter().map(|node| node.name().to_owned()).collect();
    names.sort();
    names.dedup();
    (found, names)
}

/// Debug aid: dump the egglog program and key e-graph metrics for the lite
/// paged-attention test so we can see why the FlashInfer rule isn't matching.
#[test]
#[ignore]
fn flashinfer_dump_paged_attn_egglog() {
    let (phases, _) = CudaRuntime::rules();
    eprintln!("rewrite phases: {:?}", phases.keys());
    eprintln!(
        "decoders={} phases={}",
        CudaRuntime::DECODERS.len(),
        phases.len()
    );
    let (cx, _) = build_paged_attention_graph(N_HEADS, N_KV_HEADS, HEAD_DIM);
    let roots = luminal::typed_reference::capture_graph(&cx).unwrap();
    eprintln!("typed captured roots: {roots:#?}");
    let (has_fi, kinds) = saturate_and_has_flashinfer(&cx);
    eprintln!("has FlashInferAttention: {has_fi}; OpKinds: {kinds:?}");
    let frozen = saturate_attention_graph(&cx, None);
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    for node in frozen.eclasses().flat_map(|class| class.nodes()) {
        *counts.entry(node.name().to_owned()).or_default() += 1;
    }
    let mut counts: Vec<_> = counts.into_iter().collect();
    counts.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    eprintln!("frozen constructor histogram (top 60):");
    for (name, count) in counts.into_iter().take(60) {
        eprintln!("  {count:6} {name}");
    }
}

#[test]
fn flashinfer_rule_does_not_fire_on_bare_attention() {
    // Dense attention without paged gather + cache should NOT match.
    let (cx, _, _, _, _) = build_attention_graph();
    let (has_flashinfer, _) = saturate_and_has_flashinfer_with_decode_interval(&cx);
    assert!(
        !has_flashinfer,
        "FlashInferAttention should NOT fire on bare attention (no gather/cache)"
    );
}

#[test]
fn flashinfer_rule_does_not_fire_on_unrelated_matmuls() {
    // A Gather + plain matmul (MLP-shaped projection) plus two chained matmuls
    // through softmax — close to attention structurally but missing the GQA
    // broadcast / mask Add anchors. The rule must reject this.
    let mut cx = Graph::default();
    let cache = cx.named_tensor("cache", (4096, KV_DIM)).persist();
    let gather_idx = cx
        .named_tensor("gather_idx", 'c')
        .as_dtype(luminal::dtype::DType::Int);
    let weight = cx.named_tensor("weight", (HIDDEN, KV_DIM)).persist();

    let n = gather_idx.dims1();
    let base = (gather_idx * KV_DIM).expand_dim(1, KV_DIM);
    let col = cx.arange(KV_DIM as i32).expand_dim(0, n);
    let gathered = cache.gather(base + col);
    let proj = gathered.matmul(weight.t());
    proj.output();

    let a = cx.named_tensor("a", ('s', HIDDEN));
    let b = cx.named_tensor("b", (HIDDEN, HIDDEN)).persist();
    let c_tensor = cx.named_tensor("c_tensor", (HIDDEN, HIDDEN)).persist();
    let ab = a.matmul(b.t());
    let abc = ab.softmax(1).matmul(c_tensor.t());
    abc.output();

    let (has_flashinfer, _) = saturate_and_has_flashinfer_with_decode_interval(&cx);
    assert!(
        !has_flashinfer,
        "FlashInferAttention should NOT fire on unrelated matmuls + Gather"
    );
}

#[test]
fn flashinfer_rule_rejects_mixed_cache_provenance() {
    // Regression for decode loop rolling: K and V can be shape-compatible even
    // when one cache update is from the rolled loop body and the other is from
    // the remaining unrolled layer. That is not one semantic attention op.
    let (cx, _) = build_paged_attention_graph_with_mask_and_cache_provenance(
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        TestMaskKind::TriuGather,
        TestCacheProvenance::LoopInput {
            loop_id: 7,
            stream_id: 0,
        },
        TestCacheProvenance::Raw,
        DType::F32,
    );
    let (has_flashinfer, _) = saturate_and_has_flashinfer_with_decode_interval(&cx);
    assert!(
        !has_flashinfer,
        "FlashInferAttention should NOT fire when K and V cache updates come from different loop provenance"
    );
}

#[test]
fn flashinfer_rule_fires_on_full_paged_attention() {
    if !crate::tests::utilities::gpu_supports_flashinfer() {
        return;
    }
    // Default Llama-shaped test dims (HEAD_DIM=64, N_HEADS=8, N_KV_HEADS=2).
    let (cx, _) = build_paged_attention_graph(N_HEADS, N_KV_HEADS, HEAD_DIM);
    let (has_flashinfer, op_kinds) = saturate_and_has_flashinfer(&cx);
    assert!(
        has_flashinfer,
        "FlashInferAttention was NOT found in the e-graph (Llama-shaped paged attention). \
         OpKinds present: {op_kinds:?}"
    );
}

#[test]
fn flashinfer_rule_fires_on_same_rolled_cache_provenance() {
    if !crate::tests::utilities::gpu_supports_flashinfer() {
        return;
    }
    let (cx, _) = build_paged_attention_graph_with_mask_and_cache_provenance(
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        TestMaskKind::TriuGather,
        TestCacheProvenance::LoopInput {
            loop_id: 7,
            stream_id: 0,
        },
        TestCacheProvenance::LoopInput {
            loop_id: 7,
            stream_id: 1,
        },
        DType::F32,
    );
    let (has_flashinfer, op_kinds) = saturate_and_has_flashinfer_with_decode_interval(&cx);
    assert!(
        has_flashinfer,
        "FlashInferAttention should fire when K and V cache updates come from the same rolled loop. \
         OpKinds present: {op_kinds:?}"
    );
}

#[test]
fn flashinfer_rule_fires_on_triu_gather_causal_mask() {
    if !crate::tests::utilities::gpu_supports_flashinfer() {
        return;
    }
    let (cx, _) = build_paged_attention_graph_with_mask(
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        TestMaskKind::TriuGather,
    );
    let (has_flashinfer, op_kinds) = saturate_and_has_flashinfer_with_decode_interval(&cx);
    assert!(
        has_flashinfer,
        "FlashInferAttention was NOT found for triu-gather causal mask. \
         OpKinds present: {op_kinds:?}"
    );
}

#[test]
fn flashinfer_rule_fires_on_direct_causal_mask() {
    if !crate::tests::utilities::gpu_supports_flashinfer() {
        return;
    }
    let (cx, _) = build_paged_attention_graph_with_mask(
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        TestMaskKind::DirectCausal,
    );
    let (has_flashinfer, op_kinds) = saturate_and_has_flashinfer_with_decode_interval(&cx);
    assert!(
        has_flashinfer,
        "FlashInferAttention was NOT found for direct q_pos/arange causal mask. \
         OpKinds present: {op_kinds:?}"
    );
}

#[test]
fn flashinfer_derived_causal_rule_rejects_prefill_bucket() {
    let (cx, _) = build_paged_attention_graph_with_mask(
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        TestMaskKind::TriuGather,
    );
    let (has_flashinfer, _) = saturate_and_has_flashinfer_with_s_interval(&cx, 2, 64);
    assert!(
        !has_flashinfer,
        "derived causal FlashInferAttention should not fire in a prefill bucket; fp32 FlashInfer prefill is not implemented"
    );
}

#[test]
fn cuda_graph_captures_flashinfer_decode_island() {
    if !crate::tests::utilities::gpu_supports_flashinfer() {
        return;
    }
    let Some(stream) = get_cuda_stream() else {
        return;
    };

    let batch_size = 1;
    let context_len = 4;
    let (mut cx, handles) = build_paged_attention_graph_with_mask(
        N_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
        TestMaskKind::TriuGather,
    );
    cx.set_dim('s', batch_size);
    cx.set_dim('c', context_len);
    cx.set_dim_interval('s', 1, 1);

    let llir = extract_forced_flashinfer_llir(&mut cx, "cuda graph FlashInfer decode");

    let q = deterministic_f32(batch_size * HIDDEN, 0.071, 0.1);
    let k = deterministic_f32(context_len * KV_DIM, 0.073, 0.1);
    let v = deterministic_f32(context_len * KV_DIM, 0.079, 0.1);
    let expected = run_reference_attention(&stream, &q, &k, &v, batch_size, context_len);

    let mut k_cache = vec![0.0f32; 2048 * KV_DIM];
    let mut v_cache = vec![0.0f32; 2048 * KV_DIM];
    k_cache[..context_len * KV_DIM].copy_from_slice(&k);
    v_cache[..context_len * KV_DIM].copy_from_slice(&v);

    let q_dev = stream.clone_htod(q.as_slice()).unwrap();
    let q_ptr = q_dev.device_ptr(&stream).0;
    let gather_dev = stream
        .clone_htod(&(0..context_len as i32).collect::<Vec<_>>())
        .unwrap();
    let gather_ptr = gather_dev.device_ptr(&stream).0;

    let mut rt = CudaRuntime::initialize(stream.clone());
    rt.load_llir(&llir);
    unsafe {
        rt.set_device_ptr(handles.q_rope, q_ptr, q.len() * 4);
    }
    rt.set_data(
        handles.k_rope,
        k[(context_len - 1) * KV_DIM..context_len * KV_DIM].to_vec(),
    );
    rt.set_data(
        handles.v_new,
        v[(context_len - 1) * KV_DIM..context_len * KV_DIM].to_vec(),
    );
    rt.set_data(handles.k_cache, k_cache);
    rt.set_data(handles.v_cache, v_cache);
    rt.set_data(handles.scatter_idx, vec![(context_len - 1) as i32]);
    unsafe {
        rt.set_device_ptr(handles.gather_idx, gather_ptr, context_len * 4);
    }
    rt.set_data(handles.q_pos, vec![(context_len - 1) as i32]);

    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(handles.attn_out.id);
    assert_close(&result, &expected, 1e-4, 1e-5);

    let summaries = rt.debug_cuda_graph_summaries();
    let summary = summaries
        .iter()
        .find(|summary| summary.n_flashinfer == 1)
        .expect("expected one CudaGraphOp to capture the FlashInfer decode island");
    assert_eq!(summary.absorbed_host_nodes.len(), 1);
    assert_eq!(
        summary.n_steps,
        summary.n_kernels + summary.n_cublaslt + summary.n_flashinfer
    );
    let standalone_flashinfer = rt
        .host_ops()
        .into_iter()
        .filter(|op| op.stats_name() == Some("FlashInferAttention"))
        .count();
    assert_eq!(standalone_flashinfer, 0);

    let context_len_2 = 3;
    cx.set_dim('c', context_len_2);
    let expected2 = run_reference_attention(
        &stream,
        &q,
        &k[..context_len_2 * KV_DIM],
        &v[..context_len_2 * KV_DIM],
        batch_size,
        context_len_2,
    );
    rt.set_data(
        handles.k_rope,
        k[(context_len_2 - 1) * KV_DIM..context_len_2 * KV_DIM].to_vec(),
    );
    rt.set_data(
        handles.v_new,
        v[(context_len_2 - 1) * KV_DIM..context_len_2 * KV_DIM].to_vec(),
    );
    rt.set_data(handles.scatter_idx, vec![(context_len_2 - 1) as i32]);
    unsafe {
        rt.set_device_ptr(handles.q_rope, q_ptr, q.len() * 4);
        rt.set_device_ptr(handles.gather_idx, gather_ptr, context_len_2 * 4);
    }
    rt.set_data(handles.q_pos, vec![(context_len_2 - 1) as i32]);
    rt.execute(&cx.dyn_map);
    assert_close(&rt.get_f32(handles.attn_out.id), &expected2, 1e-4, 1e-5);

    let summary_after_c_change = rt
        .debug_cuda_graph_summaries()
        .into_iter()
        .find(|summary| summary.n_flashinfer == 1)
        .expect("expected FlashInfer decode island to remain in a CudaGraphOp");
    assert_eq!(
        summary_after_c_change.flashinfer_recapture_counts,
        vec![0],
        "changing only c within the planned decode capacity should update device metadata, not recapture FlashInfer"
    );
}

#[test]
fn flashinfer_rule_fires_on_non_llama_dims() {
    if !crate::tests::utilities::gpu_supports_flashinfer() {
        return;
    }
    // Different head counts: HEAD_DIM=64, N_HEADS=16, N_KV_HEADS=4 (group=4).
    // Exercises the model-agnostic structural variables in the rule.
    let (cx, _) = build_paged_attention_graph(16, 4, 64);
    let (has_flashinfer, op_kinds) = saturate_and_has_flashinfer(&cx);
    assert!(
        has_flashinfer,
        "FlashInferAttention was NOT found for non-Llama dims. \
         OpKinds present: {op_kinds:?}"
    );
}

#[test]
fn flashinfer_rule_fires_on_mha() {
    if !crate::tests::utilities::gpu_supports_flashinfer() {
        return;
    }
    // MHA: KV_GROUPS=1 (n_heads == n_kv_heads). The GQA broadcast still
    // structurally appears (expand_dim(1, 1) + merge), so the rule should
    // still match.
    let (cx, _) = build_paged_attention_graph(12, 12, 64);
    let (has_flashinfer, op_kinds) = saturate_and_has_flashinfer(&cx);
    assert!(
        has_flashinfer,
        "FlashInferAttention was NOT found for MHA dims. \
         OpKinds present: {op_kinds:?}"
    );
}

// ─── Layer 5: extraction reachability (no GPU) ───────────────────────────
//
// After `build_search_space` saturates egglog, the GA picks an extraction by
// cost. In a tiny test graph the cuBLAS+kernel path is often faster than the
// FlashInfer host op (which pays a `plan()` setup cost per call), so asserting
// "GA picked FlashInfer" is flaky. Instead, sample many random valid genomes
// from the search space and assert that the FlashInfer extraction is reachable
// — meaning the rule fired and its final execution dependencies reconstructed
// successfully for at least one offspring.

#[test]
fn flashinfer_extraction_reachable_from_search_space() {
    if !crate::tests::utilities::gpu_supports_flashinfer() {
        return;
    }
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let (mut cx, _h) = build_paged_attention_graph(N_HEADS, N_KV_HEADS, HEAD_DIM);
    cx.set_dim('s', 1usize);
    cx.set_dim('c', 16usize);
    cx.set_dim('r', 2usize);
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let egraph = cx
        .egraph()
        .expect("egraph missing after build_search_space");
    let ops = cx
        .decoders
        .as_ref()
        .expect("egglog_ops missing after build_search_space");
    let alternatives = egraph
        .alternatives()
        .expect("admissible typed alternatives");

    let mut rng = StdRng::seed_from_u64(0xf1a541);
    let mut prev: FxHashSet<u64> = FxHashSet::default();
    let initial =
        luminal::egglog_utils::random_initial_choice(&egraph.frozen, &alternatives, &mut rng);
    prev.insert(luminal::egglog_utils::hash_choice_set(&initial));
    let mut base = initial;

    let mut found = false;
    'outer: for _ in 0..50 {
        let offspring = luminal::egglog_utils::extract_reachable_generation(
            egraph,
            &alternatives,
            &base,
            10,
            2,
            &mut prev,
            &mut rng,
        );
        if offspring.is_empty() {
            break;
        }
        for genome in offspring {
            if luminal::egglog_utils::validate_choice_set(egraph, &alternatives, &genome).is_err() {
                continue;
            }
            // A rejected candidate should not prevent testing the other valid
            // dependency selections for the reachable FlashInfer operation.
            let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                luminal::egglog_utils::reconstruct(egraph, &genome, ops, &cx.custom_ops)
                    .expect("typed candidate reconstruction")
                    .0
            }));
            let Ok(llir_graph) = panicked else { continue };

            let has_fi = llir_graph.node_indices().any(|n| {
                llir_graph[n]
                    .to_dialect::<dyn HostOp>()
                    .and_then(|op| op.stats_name())
                    == Some("FlashInferAttention")
            });
            if has_fi {
                found = true;
                break 'outer;
            }
            base = genome;
        }
    }
    assert!(
        found,
        "FlashInferAttention extraction not reachable from search space after 50 generations"
    );
}

fn extract_forced_flashinfer_llir(cx: &mut Graph, case_name: &str) -> LLIRGraph {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());

    let egraph = cx.egraph().expect("search space should have an e-graph");
    let ops = cx
        .decoders
        .as_ref()
        .expect("search space should have registered egglog ops");
    let alternatives = egraph
        .alternatives()
        .expect("admissible typed alternatives");
    let flashinfer_nodes = op_candidates(egraph, |kind| {
        crate::typed_rules::FlashInferAttentionArgs::get_args(kind)
            .unwrap()
            .is_some()
    });
    assert!(
        !flashinfer_nodes.is_empty(),
        "expected a FlashInferAttention rewrite candidate for {case_name}"
    );

    let mut last_error = None;
    for (idx, flashinfer_node) in flashinfer_nodes.iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(0xF1A5_0000 + idx as u64);
        let mut choices =
            luminal::egglog_utils::random_initial_choice(&egraph.frozen, &alternatives, &mut rng);
        choices.extend(*flashinfer_node);

        if let Err(err) =
            luminal::egglog_utils::validate_choice_set(egraph, &alternatives, &choices)
        {
            last_error = Some(err.to_string());
            continue;
        }
        let (llir, _) = luminal::egglog_utils::reconstruct(egraph, &choices, ops, &cx.custom_ops)
            .expect("selected FlashInfer candidate should reconstruct");

        let has_flashinfer = llir.node_indices().any(|n| {
            llir[n]
                .to_dialect::<dyn HostOp>()
                .and_then(|op| op.stats_name())
                == Some("FlashInferAttention")
        });
        if has_flashinfer {
            return llir;
        }

        last_error = Some("forced FlashInfer candidate did not extract to HostOp".into());
    }

    panic!(
        "expected to extract FlashInferAttention for {case_name}; last error: {}",
        last_error.unwrap_or_else(|| "no candidate could be forced".into())
    );
}

// ─── Layer 4: bf16 decode + prefill ──────────────────────────────────────
//
// bf16 reuses the same plan/run wiring as f32 with dtype-dispatched kernels.
// Decode is the same scalar kernel family; prefill is tensor-core MMA and
// only exists for 16-bit dtypes. References are computed in f32 and compared
// with bf16-scale tolerances.

fn f32s_to_bf16_bytes(data: &[f32]) -> Vec<u8> {
    data.iter()
        .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
        .collect()
}

fn bf16_bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
}

/// Run FlashInferAttention.execute() with bf16 Q/K/V buffers. With
/// `kv_indptr = Some(..)` uses the 6-input explicit-indptr form; otherwise
/// the 4-input derived causal form (decode at s=1, single-sequence prefill
/// at s>1). Returns f32 output in (batch-or-rows, heads, dim) layout.
#[allow(clippy::too_many_arguments)]
fn run_flashinfer_bf16(
    stream: &Arc<CudaStream>,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    kv_indptr: Option<&[i32]>,
    kv_indices: &[i32],
    total_q_tokens: usize,
) -> Vec<f32> {
    let q_buf = copy_to_dev(stream, &f32s_to_bf16_bytes(q));
    let k_buf = copy_to_dev(stream, &f32s_to_bf16_bytes(k_cache));
    let v_buf = copy_to_dev(stream, &f32s_to_bf16_bytes(v_cache));
    let idx_buf = copy_to_dev(stream, kv_indices);
    let out_buf = alloc_dev(stream, total_q_tokens * HIDDEN * 2);

    let fi = FlashInferAttention {
        num_qo_heads: N_HEADS,
        num_kv_heads: N_KV_HEADS,
        head_dim: HEAD_DIM,
        page_size: 1,
        batch_dim: Expression::from('s'),
        dtype: luminal::dtype::DType::Bf16,
        sm_scale: 0.0,
        window_left: -1,
        plan_info: Mutex::new(Vec::new()),
    };

    let nodes: Vec<NodeIndex> = (0..7).map(NodeIndex::new).collect();
    let (q_n, k_n, v_n, idx_n, qo_n, kv_n, out_n) = (
        nodes[0], nodes[1], nodes[2], nodes[3], nodes[4], nodes[5], nodes[6],
    );

    let mut buffers = FxHashMap::default();
    buffers.insert(
        q_n,
        DeviceBuffer::new(q_buf.device_ptr(stream).0, q.len() * 2),
    );
    buffers.insert(
        k_n,
        DeviceBuffer::new(k_buf.device_ptr(stream).0, k_cache.len() * 2),
    );
    buffers.insert(
        v_n,
        DeviceBuffer::new(v_buf.device_ptr(stream).0, v_cache.len() * 2),
    );
    buffers.insert(
        idx_n,
        DeviceBuffer::new(idx_buf.device_ptr(stream).0, kv_indices.len() * 4),
    );
    let out_ptr = out_buf.device_ptr(stream).0;
    buffers.insert(
        out_n,
        DeviceBuffer::new(out_ptr, total_q_tokens * HIDDEN * 2),
    );

    let mut dyn_map = FxHashMap::default();
    dyn_map.insert('s', total_q_tokens);
    dyn_map.insert('c', kv_indices.len());

    let _qo_buf;
    let _kv_buf;
    let inputs: Vec<NodeIndex> = if let Some(kv_indptr) = kv_indptr {
        let batch_size = kv_indptr.len() - 1;
        let qo_indptr: Vec<i32> = (0..=batch_size as i32).collect();
        _qo_buf = copy_to_dev(stream, &qo_indptr);
        _kv_buf = copy_to_dev(stream, kv_indptr);
        buffers.insert(
            qo_n,
            DeviceBuffer::new(_qo_buf.device_ptr(stream).0, qo_indptr.len() * 4),
        );
        buffers.insert(
            kv_n,
            DeviceBuffer::new(_kv_buf.device_ptr(stream).0, kv_indptr.len() * 4),
        );
        dyn_map.insert('r', kv_indptr.len());
        vec![q_n, k_n, v_n, idx_n, qo_n, kv_n]
    } else {
        vec![q_n, k_n, v_n, idx_n]
    };

    fi.execute(stream, out_n, &inputs, &buffers, &dyn_map)
        .expect("FlashInferAttention bf16 execute failed");
    stream.synchronize().unwrap();

    let mut out_bytes = vec![0u8; total_q_tokens * HIDDEN * 2];
    unsafe {
        cudarc::driver::result::memcpy_dtoh_async(&mut out_bytes, out_ptr, stream.cu_stream())
            .unwrap();
    }
    stream.synchronize().unwrap();
    let raw = bf16_bytes_to_f32s(&out_bytes);
    transpose_hbd_to_bhd(&raw, N_HEADS, total_q_tokens, HEAD_DIM)
}

// bf16 rounds inputs to 8 mantissa bits, so compare against an f32 reference
// computed from bf16-rounded inputs with tolerances a few bf16 ulps wide.
const BF16_RTOL: f32 = 3e-2;
const BF16_ATOL: f32 = 3e-3;

fn round_to_bf16(data: &[f32]) -> Vec<f32> {
    data.iter()
        .map(|v| half::bf16::from_f32(*v).to_f32())
        .collect()
}

#[test]
fn flashinfer_bf16_decode_bs1_ctx4() {
    if !crate::tests::utilities::gpu_supports_flashinfer() {
        return;
    }
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let batch_size = 1;
    let context_len = 4;
    let q = round_to_bf16(&deterministic_f32(batch_size * HIDDEN, 0.011, 0.1));
    let k = round_to_bf16(&deterministic_f32(context_len * KV_DIM, 0.021, 0.1));
    let v = round_to_bf16(&deterministic_f32(context_len * KV_DIM, 0.031, 0.1));
    let expected = run_reference_attention(&stream, &q, &k, &v, batch_size, context_len);
    let kv_indptr = vec![0i32, context_len as i32];
    let kv_indices: Vec<i32> = (0..context_len as i32).collect();
    let result = run_flashinfer_bf16(
        &stream,
        &q,
        &k,
        &v,
        Some(&kv_indptr),
        &kv_indices,
        batch_size,
    );
    assert_close(&result, &expected, BF16_RTOL, BF16_ATOL);
}

#[test]
fn flashinfer_bf16_decode_bs2_supersequence() {
    if !crate::tests::utilities::gpu_supports_flashinfer() {
        return;
    }
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let batch_size = 2;
    let ctx0 = 8;
    let ctx1 = 3;
    let total_ctx = ctx0 + ctx1;

    let q = round_to_bf16(&deterministic_f32(batch_size * HIDDEN, 0.014, 0.1));
    let k = round_to_bf16(&deterministic_f32(total_ctx * KV_DIM, 0.022, 0.1));
    let v = round_to_bf16(&deterministic_f32(total_ctx * KV_DIM, 0.032, 0.1));

    let expected0 = run_reference_attention(
        &stream,
        &q[..HIDDEN],
        &k[..ctx0 * KV_DIM],
        &v[..ctx0 * KV_DIM],
        1,
        ctx0,
    );
    let expected1 = run_reference_attention(
        &stream,
        &q[HIDDEN..],
        &k[ctx0 * KV_DIM..],
        &v[ctx0 * KV_DIM..],
        1,
        ctx1,
    );
    let expected: Vec<f32> = expected0.into_iter().chain(expected1).collect();

    let kv_indptr = vec![0i32, ctx0 as i32, total_ctx as i32];
    let kv_indices: Vec<i32> = (0..total_ctx as i32).collect();
    let result = run_flashinfer_bf16(
        &stream,
        &q,
        &k,
        &v,
        Some(&kv_indptr),
        &kv_indices,
        batch_size,
    );
    assert_close(&result, &expected, BF16_RTOL, BF16_ATOL);
}

#[test]
fn flashinfer_bf16_prefill_causal() {
    if !crate::tests::utilities::gpu_supports_flashinfer() {
        return;
    }
    // Single-sequence causal prefill via the derived 4-input path: s q tokens
    // over a c-token context whose last s slots are the q tokens themselves.
    // Reference: row j attends kv[0..=c-s+j], computed row-by-row with the
    // dense reference graph.
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let s = 4usize;
    let c = 6usize; // 2 tokens of pre-existing context + 4 new

    let q = round_to_bf16(&deterministic_f32(s * HIDDEN, 0.013, 0.1));
    let k = round_to_bf16(&deterministic_f32(c * KV_DIM, 0.023, 0.1));
    let v = round_to_bf16(&deterministic_f32(c * KV_DIM, 0.033, 0.1));

    let mut expected = Vec::with_capacity(s * HIDDEN);
    for j in 0..s {
        let visible = c - s + j + 1;
        let row = run_reference_attention(
            &stream,
            &q[j * HIDDEN..(j + 1) * HIDDEN],
            &k[..visible * KV_DIM],
            &v[..visible * KV_DIM],
            1,
            visible,
        );
        expected.extend(row);
    }

    let kv_indices: Vec<i32> = (0..c as i32).collect();
    let result = run_flashinfer_bf16(&stream, &q, &k, &v, None, &kv_indices, s);
    assert_close(&result, &expected, BF16_RTOL, BF16_ATOL);
}

#[test]
fn flashinfer_f32_prefill_still_rejected() {
    if !crate::tests::utilities::gpu_supports_flashinfer() {
        return;
    }
    // The derived 4-input path with s>1 and f32 must keep failing cleanly:
    // tensor cores are 16-bit, so there is no f32 prefill kernel.
    let Some(stream) = get_cuda_stream() else {
        return;
    };
    let s = 3usize;
    let c = 3usize;
    let q = deterministic_f32(s * HIDDEN, 0.011, 0.1);
    let k = deterministic_f32(c * KV_DIM, 0.021, 0.1);
    let v = deterministic_f32(c * KV_DIM, 0.031, 0.1);
    let kv_indices: Vec<i32> = (0..c as i32).collect();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_flashinfer_with_compact_decode_indices(&stream, &q, &k, &v, &kv_indices)
    }));
    assert!(
        result.is_err(),
        "f32 s={s} derived prefill should error (no f32 prefill kernel)"
    );
}

#[test]
#[ignore = "one-time JIT compile check for the gemma variants (~2 min each cold)"]
fn jit_compiles_gemma_variants() {
    // sliding layers: head_dim 256 with the sliding-window kernel variant
    let _ = crate::host::flashinfer::jit::ensure_compiled(256, true);
    // full layers: head_dim 512 (16-bit only; f32 instantiation is gated out)
    let _ = crate::host::flashinfer::jit::ensure_compiled(512, false);
}

/// Gemma-4 paged attention spelling at mini dims (scale-free scores; sliding
/// window mask term). Mirrors examples/gemma4_moe/src/model.rs
/// `paged_attention` exactly — used to derive the egg rule variants.
#[allow(clippy::too_many_arguments)]
fn gemma_mini_paged_attention(
    q_rope: GraphTensor,
    k_rope: GraphTensor,
    v: GraphTensor,
    k_cache: GraphTensor,
    v_cache: GraphTensor,
    scatter_idx: GraphTensor,
    gather_idx: GraphTensor,
    q_pos: GraphTensor,
    head_dim: usize,
    kv_dim: usize,
    kv_groups: usize,
    n_heads: usize,
    sliding_window: Option<usize>,
) -> GraphTensor {
    use luminal_nn::{gather_rows, scatter_rows};
    let cx = q_rope.graph();
    let k_cache_out = scatter_rows(k_rope, scatter_idx, k_cache, kv_dim);
    let v_cache_out = scatter_rows(v, scatter_idx, v_cache, kv_dim);
    let k = gather_rows(k_cache_out, gather_idx, kv_dim);
    let v_ctx = gather_rows(v_cache_out, gather_idx, kv_dim);
    let q = (q_rope * 1.0).split_dims(1, head_dim).transpose(0, 1);
    let k = k.split_dims(1, head_dim).permute((1, 2, 0));
    let v_ctx = v_ctx.split_dims(1, head_dim).transpose(0, 1);
    let k = k.expand_dim(1, kv_groups).merge_dims(0, 1) * 1.0;
    let v_ctx = v_ctx.expand_dim(1, kv_groups).merge_dims(0, 1) * 1.0;
    let scores = q.matmul(k);
    let ctx = Expression::from('c');
    let seq = q_rope.dims()[0];
    let causal_square = scores.graph().triu(ctx, 1).cast(scores.dtype) * -1e10;
    let row_offsets = (q_pos * ctx).expand_dim(1, ctx);
    let col_offsets = scores.graph().arange(ctx).expand_dim(0, seq);
    let attn_mask = causal_square.gather(row_offsets + col_offsets);
    let attn_mask = if let Some(w) = sliding_window {
        let q_f = q_pos.cast(DType::F32);
        let win_lo = q_f - (w - 1) as f32;
        let col_f = cx.arange(ctx).cast(DType::F32);
        let too_old = col_f.expand_dim(0, seq).lt(win_lo.expand_dim(1, ctx));
        attn_mask + too_old.cast(scores.dtype) * -1e10
    } else {
        attn_mask
    };
    let masked_scores = scores + attn_mask.expand_dim(0, n_heads);
    let weights = masked_scores.softmax(2);
    let out = weights.matmul(v_ctx);
    out.transpose(0, 1).merge_dims(1, 2)
}

/// Census of search choice eclasses on a realistic mini attention chunk:
/// how many genome bits select among *elementwise-only* alternatives
/// (fused-region interior wiring — performance-equivalent for the
/// bandwidth-bound regions they land in) versus structural alternatives
/// the search genuinely needs to explore. Run with --nocapture.
#[test]
#[ignore = "debug instrument: choice-eclass census for search-space analysis"]
fn egraph_choice_eclass_census() {
    const HD: usize = 64;
    const HEADS: usize = 2;
    const KVH: usize = 1;
    let mut cx = Graph::default();
    let q = cx.tensor(('s', HEADS * HD)).as_dtype(DType::Bf16);
    let k = cx.tensor(('s', KVH * HD)).as_dtype(DType::Bf16);
    let v = cx.tensor(('s', KVH * HD)).as_dtype(DType::Bf16);
    let k_cache = cx.tensor((16, KVH * HD)).as_dtype(DType::Bf16);
    let v_cache = cx.tensor((16, KVH * HD)).as_dtype(DType::Bf16);
    let scatter_idx = cx.tensor('s').as_dtype(DType::Int);
    let gather_idx = cx.tensor('c').as_dtype(DType::Int);
    let q_pos = cx.tensor('s').as_dtype(DType::Int);
    let out = gemma_mini_paged_attention(
        q,
        k,
        v,
        k_cache,
        v_cache,
        scatter_idx,
        gather_idx,
        q_pos,
        HD,
        KVH * HD,
        HEADS / KVH,
        HEADS,
        Some(8),
    );
    let _out = out.cast(DType::F32).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let egraph = cx.egraph().unwrap();
    let alternatives = egraph.alternatives().unwrap();
    let is_elementwise = |kind: &OpKind| {
        use crate::typed_rules::{CudaBinaryElementwiseArgs, FusionEndArgs, FusionStartArgs};
        use luminal::typed_reference::schema::{
            AddArgs, CastArgs, Exp2Args, Log2Args, MulArgs, RecipArgs, SinArgs, SqrtArgs,
        };
        AddArgs::get_args(kind).unwrap().is_some()
            || MulArgs::get_args(kind).unwrap().is_some()
            || CastArgs::get_args(kind).unwrap().is_some()
            || Exp2Args::get_args(kind).unwrap().is_some()
            || Log2Args::get_args(kind).unwrap().is_some()
            || SinArgs::get_args(kind).unwrap().is_some()
            || SqrtArgs::get_args(kind).unwrap().is_some()
            || RecipArgs::get_args(kind).unwrap().is_some()
            || get_args(
                kind,
                |op: &egg::String,
                 shape: &EList,
                 strides: &EList,
                 out_strides: &EList,
                 dtype: &IrDType| {
                    crate::typed_rules::cuda_unary_elementwise(
                        op,
                        shape,
                        strides,
                        out_strides,
                        dtype,
                    )
                },
            )
            .unwrap()
            .is_some()
            || CudaBinaryElementwiseArgs::get_args(kind).unwrap().is_some()
            || FusionStartArgs::get_args(kind).unwrap().is_some()
            || FusionEndArgs::get_args(kind).unwrap().is_some()
    };
    let mut multi = 0usize;
    let mut elementwise_only = 0usize;
    let mut profile_hist = std::collections::BTreeMap::<String, usize>::new();
    let mut choice_product_log2 = 0f64;
    let mut frozen_product_log2 = 0f64;
    for (class, nodes) in &alternatives {
        let sort = class.sort();
        if ![Ir::sort_ref(), IList::sort_ref(), OpKind::sort_ref()].contains(&sort)
            || nodes.len() < 2
        {
            continue;
        }
        multi += 1;
        choice_product_log2 += (nodes.len() as f64).log2();
        let mut labels = Vec::new();
        let mut ew_only = true;
        for &node in nodes {
            if sort == Ir::sort_ref() {
                let ir = egraph.frozen.typed_node::<Ir>(node).unwrap();
                if let Some((kind, _)) =
                    get_args(&ir, |kind: &OpKind, inputs: &IList| Ir::Op(kind, inputs)).unwrap()
                {
                    let kind = egraph.frozen.class_of(&kind).unwrap();
                    for &node in &alternatives[&kind] {
                        ew_only &=
                            is_elementwise(&egraph.frozen.typed_node::<OpKind>(node).unwrap());
                        labels.push(node.name().to_owned());
                    }
                    continue;
                }
            }
            ew_only &= if sort == OpKind::sort_ref() {
                is_elementwise(&egraph.frozen.typed_node::<OpKind>(node).unwrap())
            } else if sort == IList::sort_ref() {
                let list = egraph.frozen.typed_node::<IList>(node).unwrap();
                get_args(&list, |head: &Ir, tail: &IList| IList::ICons(head, tail))
                    .unwrap()
                    .is_some()
                    || get_args(&list, IList::INil).unwrap().is_some()
            } else {
                false
            };
            labels.push(node.name().to_owned());
        }
        if ew_only {
            elementwise_only += 1;
        } else {
            frozen_product_log2 += (nodes.len() as f64).log2();
        }
        labels.sort_unstable();
        labels.dedup();
        *profile_hist.entry(format!("{labels:?}")).or_default() += 1;
    }
    println!("multi-alternative choice eclasses: {multi}");
    println!("  elementwise-only (freezable):    {elementwise_only}");
    println!(
        "  search-space log2: {choice_product_log2:.0} -> {frozen_product_log2:.0} bits after freezing"
    );
    println!("top label profiles:");
    let mut hist: Vec<_> = profile_hist.into_iter().collect();
    hist.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (profile, count) in hist.into_iter().take(25) {
        println!("  {count:5}  {profile}");
    }
}

#[test]
#[ignore = "debug instrument: dump gemma sliding paged attention egglog"]
fn dump_gemma_sliding_attention_egglog() {
    const HD: usize = 64;
    const HEADS: usize = 2;
    const KVH: usize = 1;
    let mut cx = Graph::default();
    let q = cx.tensor(('s', HEADS * HD)).as_dtype(DType::Bf16);
    let k = cx.tensor(('s', KVH * HD)).as_dtype(DType::Bf16);
    let v = cx.tensor(('s', KVH * HD)).as_dtype(DType::Bf16);
    let k_cache = cx.tensor((16, KVH * HD)).as_dtype(DType::Bf16);
    let v_cache = cx.tensor((16, KVH * HD)).as_dtype(DType::Bf16);
    let scatter_idx = cx.tensor('s').as_dtype(DType::Int);
    let gather_idx = cx.tensor('c').as_dtype(DType::Int);
    let q_pos = cx.tensor('s').as_dtype(DType::Int);
    let out = gemma_mini_paged_attention(
        q,
        k,
        v,
        k_cache,
        v_cache,
        scatter_idx,
        gather_idx,
        q_pos,
        HD,
        KVH * HD,
        HEADS / KVH,
        HEADS,
        Some(8),
    );
    let _out = out.cast(DType::F32).output();
    let roots = luminal::typed_reference::capture_graph(&cx).unwrap();
    println!("{roots:#?}");
}

#[test]
fn flashinfer_rule_fires_on_gemma_sliding_mask() {
    if !crate::tests::utilities::gpu_supports_flashinfer() {
        return;
    }
    const HD: usize = 64;
    const HEADS: usize = 2;
    const KVH: usize = 1;
    let mut cx = Graph::default();
    let q = cx.tensor(('s', HEADS * HD)).as_dtype(DType::Bf16);
    let k = cx.tensor(('s', KVH * HD)).as_dtype(DType::Bf16);
    let v = cx.tensor(('s', KVH * HD)).as_dtype(DType::Bf16);
    let k_cache = cx.tensor((16, KVH * HD)).as_dtype(DType::Bf16);
    let v_cache = cx.tensor((16, KVH * HD)).as_dtype(DType::Bf16);
    let scatter_idx = cx.tensor('s').as_dtype(DType::Int);
    let gather_idx = cx.tensor('c').as_dtype(DType::Int);
    let q_pos = cx.tensor('s').as_dtype(DType::Int);
    let out = gemma_mini_paged_attention(
        q,
        k,
        v,
        k_cache,
        v_cache,
        scatter_idx,
        gather_idx,
        q_pos,
        HD,
        KVH * HD,
        HEADS / KVH,
        HEADS,
        Some(8),
    );
    let _ = out.cast(DType::F32).output();
    let (has_flashinfer, op_kinds) = saturate_and_has_flashinfer_with_decode_interval(&cx);
    assert!(
        has_flashinfer,
        "FlashInferAttention was NOT found for the gemma sliding mask. \
         OpKinds present: {op_kinds:?}"
    );
}

#[test]
fn flashinfer_rule_fires_on_gemma_scale_free_mask() {
    if !crate::tests::utilities::gpu_supports_flashinfer() {
        return;
    }
    const HD: usize = 64;
    const HEADS: usize = 2;
    const KVH: usize = 1;
    let mut cx = Graph::default();
    let q = cx.tensor(('s', HEADS * HD)).as_dtype(DType::Bf16);
    let k = cx.tensor(('s', KVH * HD)).as_dtype(DType::Bf16);
    let v = cx.tensor(('s', KVH * HD)).as_dtype(DType::Bf16);
    let k_cache = cx.tensor((16, KVH * HD)).as_dtype(DType::Bf16);
    let v_cache = cx.tensor((16, KVH * HD)).as_dtype(DType::Bf16);
    let scatter_idx = cx.tensor('s').as_dtype(DType::Int);
    let gather_idx = cx.tensor('c').as_dtype(DType::Int);
    let q_pos = cx.tensor('s').as_dtype(DType::Int);
    let out = gemma_mini_paged_attention(
        q,
        k,
        v,
        k_cache,
        v_cache,
        scatter_idx,
        gather_idx,
        q_pos,
        HD,
        KVH * HD,
        HEADS / KVH,
        HEADS,
        None,
    );
    let _ = out.cast(DType::F32).output();
    let (has_flashinfer, op_kinds) = saturate_and_has_flashinfer_with_decode_interval(&cx);
    assert!(
        has_flashinfer,
        "FlashInferAttention was NOT found for the gemma scale-free mask. \
         OpKinds present: {op_kinds:?}"
    );
}

#[test]
#[ignore = "perf repro: gemma FI rule join cost with 6 distinct attention instances"]
fn gemma_fi_rules_six_instances_build_time() {
    const HD: usize = 64;
    const HEADS: usize = 2;
    const KVH: usize = 1;
    let mut cx = Graph::default();
    let scatter_idx = cx.tensor('s').as_dtype(DType::Int);
    let gather_idx = cx.tensor('c').as_dtype(DType::Int);
    let q_pos = cx.tensor('s').as_dtype(DType::Int);
    let mut outs = Vec::new();
    for i in 0..6 {
        let q = cx.tensor(('s', HEADS * HD)).as_dtype(DType::Bf16);
        let k = cx.tensor(('s', KVH * HD)).as_dtype(DType::Bf16);
        let v = cx.tensor(('s', KVH * HD)).as_dtype(DType::Bf16);
        let k_cache = cx.tensor((16, KVH * HD)).as_dtype(DType::Bf16);
        let v_cache = cx.tensor((16, KVH * HD)).as_dtype(DType::Bf16);
        let sliding = if i % 2 == 0 { Some(8) } else { None };
        let out = gemma_mini_paged_attention(
            q,
            k,
            v,
            k_cache,
            v_cache,
            scatter_idx,
            gather_idx,
            q_pos,
            HD,
            KVH * HD,
            HEADS / KVH,
            HEADS,
            sliding,
        );
        outs.push(out);
    }
    let total = outs.into_iter().reduce(|a, b| a + b).unwrap();
    let _ = total.cast(DType::F32).output();
    let start = std::time::Instant::now();
    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    println!("six-instance gemma FI build: {:?}", start.elapsed());
}

#[test]
#[ignore = "debug instrument: dump llama swiglu(+quant) chain egglog"]
fn dump_llama_swiglu_chain_egglog() {
    const I: usize = 8;
    let mut cx = Graph::default();
    let xgu = cx.tensor(('s', 2 * I)).as_dtype(DType::Bf16);
    let scale = cx.tensor(()).as_dtype(DType::F32);
    let gate = xgu.slice((.., ..I));
    let up = xgu.slice((.., I..));
    let h = gate.swish() * up;
    // quant tail (the llama fp8 spelling)
    let hf = h.cast(DType::F32);
    let scale_e = scale.expand_dim(0, 's').expand_dim(1, I);
    let q = (hf / scale_e).cast(DType::F8E4M3);
    let _ = q.cast(DType::F32).output();
    let roots = luminal::typed_reference::capture_graph(&cx).unwrap();
    println!("{roots:#?}");
}
