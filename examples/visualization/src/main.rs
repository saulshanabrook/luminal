// glibc malloc can livelock inside nvrtcCompileProgram after heavy search
// heap churn. This allocator also interposes allocations inside CUDA libraries.
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use luminal::{
    egglog_utils::{
        log_channel_enabled, random_initial_choice, reconstruct, run_schedule, trace_frozen_shape,
    },
    prelude::*,
    typed_reference::{capture_graph, interval_actions},
    visualization::{ToDot, ToHtml},
};
use luminal_cuda_lite::runtime::CudaRuntime;
use std::fs;

fn main() {
    let mut graph = Graph::new();
    let (m, n, k) = (4096, 14336, 9192);
    let a = graph.tensor((m, k));
    let b = graph.tensor((k, n));
    a.matmul(b).output();
    fs::write("HLIR.dot", graph.graph.to_dot().unwrap()).unwrap();

    // Use the production setup and schedule, including late backend phases.
    graph.auto_roll_loops_prepass();
    let roots = capture_graph(&graph).expect("typed graph capture failed");
    let (phases, actions) = CudaRuntime::rules();
    let log = log_channel_enabled(false, "EGGLOG_LOG");
    let full_start = std::time::Instant::now();
    let mut egraph = typed::EGraph::new(typed::EGraphOptions::default());
    run_schedule(
        &mut egraph,
        full_start,
        &phases,
        !graph.dim_intervals.is_empty(),
        log,
        |egraph| {
            egraph.register(interval_actions(&graph.dim_intervals, []))?;
            egraph.register(&actions)?;
            egraph.register(&roots)
        },
    )
    .expect("typed schedule failed");
    let mut space = SearchSpace::from_frozen(egraph.freeze().unwrap(), &roots)
        .expect("typed search-space construction failed");
    trace_frozen_shape(&space, log);
    space.clean_hlir = true;
    space.cleanup_ignored = CudaRuntime::cleanup_ignored;
    // Serialization is only for display; reconstruction uses the frozen snapshot.
    let egraph = egraph.into_raw();
    fs::write("egraph.dot", egraph.to_dot().unwrap()).unwrap();
    fs::write("egraph.html", egraph.to_html().unwrap()).unwrap();
    let decoders: Vec<_> = CudaRuntime::DECODERS
        .iter()
        .chain(luminal::hlir::DECODERS)
        .copied()
        .collect();
    let alternatives = space.alternatives().expect("valid alternatives");
    let choices = random_initial_choice(&space.frozen, &alternatives, &mut rand::rng());
    let (llir, _) = reconstruct(&space, &choices, &decoders, &[]).expect("valid chosen forest");
    fs::write("LLIR.dot", llir.to_dot().unwrap()).unwrap();
}
