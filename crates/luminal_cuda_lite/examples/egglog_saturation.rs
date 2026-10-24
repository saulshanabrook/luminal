use std::{collections::BTreeMap, time::Instant};

use egglog_experimental::typed::{
    EGraph, Rule, Schedule, Snapshot, builtins as egg, delete, rule, ruleset, sequence,
};
use itertools::Itertools;
use luminal::typed_reference::schema::{DType as EggDType, DimExpr, EList, IList, Ir, OpKind};
use luminal::{
    dtype::DType, egglog_utils::RewritePhase, op::Runtime, prelude::*, shape::Expression,
    typed_reference::capture_graph,
};
use luminal_cuda_lite::runtime::CudaRuntime;

const DEFAULT_PASSES: usize = 256;
const MOE_SEQ: usize = 2;
const MOE_HIDDEN: usize = 16;
const MOE_NUM_EXPERTS: usize = 8;
const MOE_TOP_K: usize = 2;
const MOE_INTERMEDIATE: usize = 6;
const GEMMA_RMS_NORM_EPS: f32 = 1e-6;

#[derive(Debug, Clone, Copy)]
enum Backend {
    Reference,
    Cuda,
}

#[derive(Debug, Clone, Copy)]
enum Mode {
    Current,
    Steps,
    FullDefault,
    FullCycle,
}

#[derive(Debug, Clone, Copy)]
enum Case {
    Mul,
    UnaryChain(usize),
    Gelu,
    Softmax,
    LayerNorm,
    Matmul,
    Attention,
    QwenMoe,
    GemmaMoe,
    /// N stacked Flux-schnell-style DiT blocks: adaLN modulation (shift/scale/gate
    /// from a conditioning Linear, broadcast over seq) wrapping multi-head
    /// attention + a gelu MLP, with gated residuals. Reproduces the cuBLASLt fp8
    /// rules' compile-time blowup: the modulation broadcasts + softmax are a rich
    /// scalar-`FusionStart` source, and the `cublaslt scaled fp8 fused
    /// output-scale` rule matches two scalar FusionStarts as free atoms — a K²
    /// cross-product that explodes super-linearly with block count (0 matches),
    /// which is why the real 57-block FLUX.1-schnell compile OOMs.
    FluxBlock(usize),
}

#[derive(Debug)]
struct Args {
    backend: Backend,
    mode: Mode,
    case: Case,
    passes: usize,
    cleanup: bool,
    skip_roll: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        backend: Backend::Cuda,
        mode: Mode::Current,
        case: Case::Gelu,
        passes: DEFAULT_PASSES,
        cleanup: true,
        skip_roll: false,
    };

    let mut iter = std::env::args().skip(1);
    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "--backend" => {
                args.backend = match iter.next().as_deref() {
                    Some("reference") => Backend::Reference,
                    Some("cuda") => Backend::Cuda,
                    other => panic!("invalid --backend {other:?}; use reference|cuda"),
                };
            }
            "--mode" => {
                args.mode = match iter.next().as_deref() {
                    Some("current") => Mode::Current,
                    Some("steps") => Mode::Steps,
                    Some("full-default") => Mode::FullDefault,
                    Some("full-cycle") => Mode::FullCycle,
                    other => panic!(
                        "invalid --mode {other:?}; use current|steps|full-default|full-cycle"
                    ),
                };
            }
            "--case" => {
                args.case = parse_case(&iter.next().expect("missing --case value"));
            }
            "--passes" => {
                args.passes = iter
                    .next()
                    .expect("missing --passes value")
                    .parse()
                    .expect("invalid --passes value");
            }
            "--no-cleanup" => args.cleanup = false,
            "--skip-roll" => args.skip_roll = true,
            "--help" | "-h" => {
                println!(
                    "Usage: egglog_saturation [OPTIONS]\n\
                     \n\
                     Options:\n\
                       --backend reference|cuda       default: cuda\n\
                       --mode current|steps|full-default|full-cycle\n\
                       --case mul|unary-chain:N|gelu|softmax|layer-norm|matmul|attention|qwen-moe|gemma-moe|flux-block:N\n\
                       --passes N                    default: 256\n\
                       --no-cleanup                  omit backend/HLIR cleanup rules\n\
                       --skip-roll                   skip auto loop rolling prepass"
                );
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}; use --help"),
        }
    }

    args
}

fn parse_case(s: &str) -> Case {
    if let Some(n) = s.strip_prefix("unary-chain:") {
        return Case::UnaryChain(n.parse().expect("invalid unary-chain length"));
    }
    if let Some(n) = s.strip_prefix("flux-block:") {
        return Case::FluxBlock(n.parse().expect("invalid flux-block count"));
    }
    match s {
        "mul" => Case::Mul,
        "gelu" => Case::Gelu,
        "softmax" => Case::Softmax,
        "layer-norm" | "layer_norm" => Case::LayerNorm,
        "matmul" => Case::Matmul,
        "attention" => Case::Attention,
        "qwen-moe" | "qwen_moe" => Case::QwenMoe,
        "gemma-moe" | "gemma_moe" => Case::GemmaMoe,
        other => panic!("unknown case {other}"),
    }
}

fn build_case(case: Case) -> Graph {
    let mut cx = Graph::new();
    let out = match case {
        Case::Mul => {
            let x = cx.tensor((64, 64));
            x * x
        }
        Case::UnaryChain(n) => {
            let mut x = cx.tensor((64, 64));
            for i in 0..n {
                x = match i % 6 {
                    0 => x.sin(),
                    1 => x.sqrt(),
                    2 => x.reciprocal(),
                    3 => x.exp2(),
                    4 => x.log2(),
                    _ => x * 1.125,
                };
            }
            x
        }
        Case::Gelu => cx.tensor((64, 64)).gelu(),
        Case::Softmax => cx.tensor((128, 128)).softmax(1),
        Case::LayerNorm => cx.tensor((128, 128)).layer_norm(1, 1e-5),
        Case::Matmul => {
            let a = cx.tensor((32, 64));
            let b = cx.tensor((64, 32));
            a.matmul(b)
        }
        Case::Attention => {
            let q = cx.tensor((64, 32));
            let k = cx.tensor((64, 32));
            let v = cx.tensor((64, 32));
            let scores = q.matmul(k.permute((1, 0))) * (1.0 / 32.0_f32.sqrt());
            scores.softmax(1).matmul(v)
        }
        Case::QwenMoe => build_qwen_moe(&mut cx),
        Case::GemmaMoe => build_gemma_moe(&mut cx),
        Case::FluxBlock(blocks) => build_flux_block(&mut cx, blocks),
    };
    let _ = out.output();
    cx
}

/// N stacked Flux-schnell-style DiT blocks (adaLN modulation + MHA + gelu MLP).
/// See `Case::FluxBlock` for why this reproduces the fp8 rule compile-time blowup.
fn build_flux_block(cx: &mut Graph, blocks: usize) -> GraphTensor {
    let seq = 16usize;
    let dim = 64usize;
    let heads = 4usize;
    let hd = dim / heads;
    let mlp = dim * 2;
    let attn_scale = 1.0f32 / (hd as f32).sqrt();
    let mut x = cx.tensor((seq, dim)).as_dtype(DType::F16);
    for _ in 0..blocks {
        // adaLN modulation: a Linear from a conditioning vector produces 6 (dim,)
        // chunks (shift/scale/gate ×2), each broadcast over seq.
        let temb = cx.tensor((1, dim)).as_dtype(DType::F16);
        let mod_lin = cx.tensor((dim, dim * 6)).as_dtype(DType::F16);
        let m = temb.matmul(mod_lin); // (1, 6*dim)
        let chunk = |mm: GraphTensor, i: usize| {
            mm.slice((.., i * dim..(i + 1) * dim))
                .squeeze(0)
                .expand_lhs((seq,))
        };
        let (shift1, scale1, gate1) = (chunk(m, 0), chunk(m, 1), chunk(m, 2));
        let (shift2, scale2, gate2) = (chunk(m, 3), chunk(m, 4), chunk(m, 5));
        // ── attention sublayer ──
        let h = x.layer_norm(1, 1e-5) * (scale1 + 1.0) + shift1;
        let qw = cx.tensor((dim, dim)).as_dtype(DType::F16);
        let kw = cx.tensor((dim, dim)).as_dtype(DType::F16);
        let vw = cx.tensor((dim, dim)).as_dtype(DType::F16);
        let ow = cx.tensor((dim, dim)).as_dtype(DType::F16);
        let q = h.matmul(qw).split_dims(1, hd).permute((1, 0, 2));
        let k = h.matmul(kw).split_dims(1, hd).permute((1, 0, 2));
        let v = h.matmul(vw).split_dims(1, hd).permute((1, 0, 2));
        let scores = (q.matmul(k.permute((0, 2, 1))) * attn_scale).softmax(2);
        let attn = scores
            .matmul(v)
            .permute((1, 0, 2))
            .merge_dims(1, 2)
            .matmul(ow);
        x += gate1 * attn;
        // ── gelu MLP sublayer ──
        let h2 = x.layer_norm(1, 1e-5) * (scale2 + 1.0) + shift2;
        let up = cx.tensor((dim, mlp)).as_dtype(DType::F16);
        let down = cx.tensor((mlp, dim)).as_dtype(DType::F16);
        x += gate2 * h2.matmul(up).gelu().matmul(down);
    }
    x
}

fn build_qwen_moe(cx: &mut Graph) -> GraphTensor {
    cx.set_dim('s', MOE_SEQ);
    let x = cx.tensor(('s', MOE_HIDDEN));
    let router = cx.tensor((MOE_NUM_EXPERTS, MOE_HIDDEN));
    let gate_up_weights = cx
        .tensor((MOE_NUM_EXPERTS, MOE_INTERMEDIATE * 2, MOE_HIDDEN))
        .as_dtype(DType::Bf16);
    let down_weights = cx
        .tensor((MOE_NUM_EXPERTS, MOE_HIDDEN, MOE_INTERMEDIATE))
        .as_dtype(DType::Bf16);

    let n = x.dims().len();
    let e_dim = *router.dims().first().unwrap();
    let k_expr = Expression::from(MOE_TOP_K);

    let routing_weights = x.matmul(router.t()).softmax(n - 1);
    let top_k_indices = routing_weights.topk_indexes(MOE_TOP_K, n - 1);
    let row_offsets = x
        .graph()
        .iota(Expression::from('z') / k_expr * e_dim, top_k_indices.dims());
    let routing_flat_idx = row_offsets + top_k_indices;
    let top_k_values = routing_weights.gather(routing_flat_idx);

    let gate_up_gathered = gather_experts(x, top_k_indices, gate_up_weights).cast(DType::F32);
    let x_exp = x.expand_dim(n - 1, MOE_TOP_K).unsqueeze(n);
    let gate_up_out = x_exp.matmul(gate_up_gathered.transpose(2, 3)).squeeze(n);
    let gate = gate_up_out.slice((.., .., ..MOE_INTERMEDIATE));
    let up = gate_up_out.slice((.., .., MOE_INTERMEDIATE..));
    let hidden = gate.silu() * up;

    let down_gathered = gather_experts(x, top_k_indices, down_weights).cast(DType::F32);
    let down_out = hidden
        .unsqueeze(2)
        .matmul(down_gathered.transpose(2, 3))
        .squeeze(2);
    let mut weights_exp = top_k_values.unsqueeze(top_k_values.dims().len());
    weights_exp.shape.expand(down_out.dims());
    (down_out * weights_exp).sum(n - 1)
}

fn build_gemma_moe(cx: &mut Graph) -> GraphTensor {
    cx.set_dim('s', MOE_SEQ);
    let router_input = cx.tensor(('s', MOE_HIDDEN));
    let expert_input = cx.tensor(('s', MOE_HIDDEN));
    let router_scale = cx.tensor(MOE_HIDDEN);
    let router_proj = cx.tensor((MOE_NUM_EXPERTS, MOE_HIDDEN));
    let per_expert_scale = cx.tensor(MOE_NUM_EXPERTS);
    let gate_up_weights = cx
        .tensor((MOE_NUM_EXPERTS, MOE_INTERMEDIATE * 2, MOE_HIDDEN))
        .as_dtype(DType::Bf16);
    let down_weights = cx
        .tensor((MOE_NUM_EXPERTS, MOE_HIDDEN, MOE_INTERMEDIATE))
        .as_dtype(DType::Bf16);

    let n = router_input.dims().len();
    let e_dim = *router_proj.dims().first().unwrap();
    let k_expr = Expression::from(MOE_TOP_K);

    let router_hidden = router_input.std_norm(n - 1, GEMMA_RMS_NORM_EPS)
        * router_scale.expand_lhs(&router_input.dims()[..n - 1])
        * (MOE_HIDDEN as f32).sqrt().recip();
    let routing_weights = router_hidden.matmul(router_proj.t()).softmax(n - 1);
    let top_k_indices = routing_weights.topk_indexes(MOE_TOP_K, n - 1);
    let row_offsets = router_input
        .graph()
        .iota(Expression::from('z') / k_expr * e_dim, top_k_indices.dims());
    let routing_flat_idx = row_offsets + top_k_indices;
    let top_k_values = routing_weights.gather(routing_flat_idx);
    let top_k_norm = top_k_values.sum(n - 1).expand_dim(n - 1, MOE_TOP_K);
    let top_k_weights = (top_k_values / top_k_norm) * per_expert_scale.gather(top_k_indices);

    let gate_up_gathered =
        gather_experts(expert_input, top_k_indices, gate_up_weights).cast(DType::F32);
    let x_exp = expert_input.expand_dim(n - 1, MOE_TOP_K).unsqueeze(n);
    let gate_up_out = x_exp.matmul(gate_up_gathered.transpose(2, 3)).squeeze(n);
    let gate = gate_up_out.slice((.., .., ..MOE_INTERMEDIATE));
    let up = gate_up_out.slice((.., .., MOE_INTERMEDIATE..));
    let hidden = gemma_gelu(gate) * up;

    let down_gathered = gather_experts(expert_input, top_k_indices, down_weights).cast(DType::F32);
    let down_out = hidden
        .unsqueeze(2)
        .matmul(down_gathered.transpose(2, 3))
        .squeeze(2);
    let mut weights_exp = top_k_weights.unsqueeze(top_k_weights.dims().len());
    weights_exp.shape.expand(down_out.dims());
    (down_out * weights_exp).sum(n - 1)
}

fn gather_experts(
    graph_source: GraphTensor,
    top_k_indices: GraphTensor,
    weights: GraphTensor,
) -> GraphTensor {
    let (_, d1, d2) = weights.dims3();
    let io = d1 * d2;
    let base = top_k_indices * io;
    let within = graph_source.graph().iota(Expression::from('z'), (d1, d2));
    let n_base = base.dims().len();
    let exp_base = base.expand_dim(n_base, d1).expand_dim(n_base + 1, d2);
    let mut exp_within = within;
    for (axis, dim) in base.dims().iter().enumerate() {
        exp_within = exp_within.expand_dim(axis, *dim);
    }
    weights.gather(exp_base + exp_within)
}

#[allow(clippy::excessive_precision)]
fn gemma_gelu(x: GraphTensor) -> GraphTensor {
    let scaled = 1.5957691216 * x * (1. + 0.044715 * x * x);
    x * scaled.sigmoid()
}

fn split_cycle(phases: &luminal::egglog_utils::PhaseRules) -> [(&'static str, Schedule); 2] {
    use RewritePhase::*;
    [
        (
            "producers",
            sequence((
                phases[&Expr].saturate(),
                phases[&DtypeProp].saturate(),
                &phases[&MatmulFlatten],
                &phases[&KernelLower],
                &phases[&DirectKernel],
                &phases[&KernelSpecialize],
                &phases[&BufferReuse],
                &phases[&MatmulBackend],
                &phases[&Glumoe],
                &phases[&FusionPair],
            ))
            .saturate(),
        ),
        (
            "fusion",
            sequence((
                phases[&Expr].saturate(),
                phases[&DtypeProp].saturate(),
                &phases[&FusionGrow],
                &phases[&FusionMerge],
            ))
            .saturate(),
        ),
    ]
}

fn phase(egraph: &mut EGraph, name: &str, schedule: impl Into<Schedule>) -> bool {
    let before = egraph.num_tuples().expect("table census");
    let start = Instant::now();
    let (report, diagnostic) = egraph
        .run_with_diagnostics(schedule)
        .unwrap_or_else(|error| panic!("failed phase {name}: {error}"));
    let elapsed = start.elapsed();
    let after = egraph.num_tuples().expect("table census");
    println!(
        "phase {name:<18} {elapsed_ms:>8.2} ms | tuples {before} -> {after} ({delta:+}) | updated={updated}",
        elapsed_ms = elapsed.as_secs_f64() * 1000.0,
        delta = after as isize - before as isize,
        updated = report.updated,
    );
    // This diagnostic experiment prints the backend document, including its
    // rule metrics and available plans, without a second report model.
    println!(
        "  diagnostic {}\n{}",
        diagnostic.format,
        std::str::from_utf8(&diagnostic.content).expect("backend JSON diagnostic is UTF-8")
    );
    report.updated
}

fn frozen_summary(frozen: &Snapshot, roots: usize) {
    let mut top_ops = BTreeMap::new();
    let mut nodes = 0;
    let mut classes = 0;
    for class in frozen.eclasses() {
        let mut has_active_node = false;
        for node in class.nodes().filter(|node| !node.is_subsumed()) {
            has_active_node = true;
            nodes += 1;
            *top_ops.entry(node.name().to_string()).or_insert(0_usize) += 1;
        }
        classes += usize::from(has_active_node);
    }
    let top_ops = top_ops
        .into_iter()
        .sorted_by_key(|(_, count)| std::cmp::Reverse(*count))
        .take(12)
        .map(|(op, count)| format!("{op}={count}"))
        .join(", ");
    println!(
        "freeze equality_nodes={nodes} equality_classes={classes} output_roots={roots} top_ops={top_ops}"
    );
}

fn run(args: Args) {
    use RewritePhase::*;
    let mut graph = build_case(args.case);
    let rolled = if args.skip_roll {
        0
    } else {
        graph.auto_roll_loops_prepass()
    };
    // The former inline datatype* / dtype declarations now have one canonical
    // definition in luminal::typed_reference::schema (Ir, OpKind, IList, DType).
    // Backend constructors live in luminal_cuda_lite::typed_rules. Captured roots
    // preserve output order directly, replacing the artificial OutputJoin node.
    let outputs = capture_graph(&graph).expect("lower HLIR");
    let (phases, actions) = match args.backend {
        Backend::Reference => ReferenceRuntime::rules(),
        Backend::Cuda => CudaRuntime::rules(),
    };
    let intervals = false; // This harness supplies no dynamic interval facts.
    let cleanup = args.cleanup && matches!(args.backend, Backend::Cuda);
    // The original standalone harness deleted every cleanup-eligible HLIR
    // constructor unconditionally. Keep that experiment distinct from production
    // conditional cleanup. These are the sixteen HEAD cleanup() == true kinds.
    let hlir_cleanup = ruleset(
        |inputs: &IList,
         value: &egg::F64,
         a: &DimExpr,
         b: &DimExpr,
         dtype: &EggDType,
         shape: &EList,
         strides: &EList,
         other: &EList,
         output: &EList,
         extra: &EList| {
            vec![
                OpKind::Constant(value),
                OpKind::Iota(a, b),
                OpKind::Cast(a, dtype),
                OpKind::Log2(shape, strides, output),
                OpKind::Exp2(shape, strides, output),
                OpKind::Sin(shape, strides, output),
                OpKind::Recip(shape, strides, output),
                OpKind::Sqrt(shape, strides, output),
                OpKind::Add(shape, strides, other, output),
                OpKind::Mul(shape, strides, other, output),
                OpKind::Mod(shape, strides, other, output),
                OpKind::LessThan(shape, strides, other, output),
                OpKind::Gather(shape, strides, other, output),
                OpKind::Scatter(shape, strides, other, output, extra),
                OpKind::Sum(shape, a, strides, b, output),
                OpKind::Max(shape, a, strides, b, output),
            ]
            .into_iter()
            .map(|kind| {
                let op = Ir::Op(kind, inputs);
                rule((&op,), (delete(&op),))
            })
            .collect::<Vec<_>>()
        },
    );
    if cleanup {
        phases.insert(Cleanup, ruleset((&phases[&Cleanup], &hlir_cleanup)));
    }
    // All rules registered by the original catalog explicitly selected a named
    // phase; its unnamed default group was empty, not the union of main phases.
    let default_rules = ruleset(Vec::<Rule>::new());
    println!(
        "case={:?} backend={:?} mode={:?} passes={} cleanup={} rolled={} hlir_nodes={} roots={}",
        args.case,
        args.backend,
        args.mode,
        args.passes,
        cleanup,
        rolled,
        graph.graph.node_count(),
        outputs.len(),
    );
    let mut egraph = EGraph::default();
    let before = egraph.num_tuples().expect("table census");
    let start = Instant::now();
    egraph.register(actions).expect("backend initialization");
    // Install every rule before ending setup, without applying any of them.
    egraph
        .run(ruleset(phases.values().cloned().collect::<Vec<_>>()).repeat(0))
        .expect("rule installation");
    egraph.register(outputs.as_slice()).expect("captured roots");
    let after = egraph.num_tuples().expect("table census");
    println!(
        "setup {:>8.2} ms | tuples {before} -> {after} ({:+})",
        start.elapsed().as_secs_f64() * 1000.0,
        after as isize - before as isize,
    );

    match args.mode {
        Mode::Current | Mode::Steps => {
            for pass in 1..=args.passes {
                let mut updated = false;
                for (name, schedule) in split_cycle(&phases) {
                    updated |= phase(&mut egraph, &format!("{pass:03} {name}"), schedule);
                }
                if matches!(args.mode, Mode::Current) && !updated {
                    break;
                }
            }
        }
        Mode::FullDefault => {
            phase(&mut egraph, "expr", phases[&Expr].saturate());
            phase(&mut egraph, "dtype", phases[&DtypeProp].saturate());
            phase(&mut egraph, "default-full", default_rules.saturate());
        }
        Mode::FullCycle => {
            phase(
                &mut egraph,
                "cycle-full",
                luminal::egglog_utils::main_schedule(&phases, intervals),
            );
        }
    }
    phase(&mut egraph, "final expr", &phases[&Expr].saturate());
    if cleanup {
        phase(&mut egraph, "cleanup", &phases[&Cleanup].saturate());
    }
    phase(
        &mut egraph,
        "base cleanup",
        &phases[&BaseCleanup].saturate(),
    );
    let frozen = egraph.freeze().expect("exact snapshot");
    for output in &outputs {
        frozen.lookup(output).expect("retained output capture");
    }
    frozen_summary(&frozen, outputs.len());
}

fn main() {
    run(parse_args());
}
