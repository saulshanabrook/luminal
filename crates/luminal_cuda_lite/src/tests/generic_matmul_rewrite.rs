use super::utilities::op_candidates;
use luminal::egglog_utils::{random_initial_choice, reconstruct, validate_choice_set};
use luminal::prelude::*;
use rand::{SeedableRng, rngs::StdRng};

use crate::{kernel::KernelOp, runtime::CudaRuntime};

use super::utilities::{assert_close, get_cuda_stream};

#[test]
fn generic_matmul_covers_noncontiguous_merged_head_projection() {
    let mut cx = Graph::default();
    let heads = 3;
    let seq = 4;
    let head_dim = 5;
    let hidden = heads * head_dim;
    let out_dim = 7;

    let attn = cx.tensor((heads, seq, head_dim));
    let weight = cx.tensor((out_dim, hidden));
    let merged = attn.transpose(0, 1).merge_dims(1, 2);
    merged.matmul(weight.t()).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let llir = extract_forced_kernel_llir(&mut cx, "GenericMatmul");
    let names = llir_kernel_names(&llir);

    assert!(
        names.contains(&"GenericMatmul"),
        "expected generic matmul fallback, kernels: {names:?}"
    );
    assert!(
        !names.contains(&"Mul") && !names.contains(&"SumReduce"),
        "generic matmul should prune the broadcast multiply/sum fallback, kernels: {names:?}"
    );
}

#[test]
fn generic_matmul_executes_noncontiguous_merged_head_projection() {
    let mut cx = Graph::default();
    let heads = 3;
    let seq = 4;
    let head_dim = 5;
    let hidden = heads * head_dim;
    let out_dim = 7;

    let attn = cx.tensor((heads, seq, head_dim));
    let weight = cx.tensor((out_dim, hidden));
    let merged = attn.transpose(0, 1).merge_dims(1, 2);
    let output = merged.matmul(weight.t()).output();

    cx.build_search_space::<CudaRuntime>(CompileOptions::default());
    let stream = get_cuda_stream().expect("CUDA device required for GenericMatmul execution test");
    let mut rt = CudaRuntime::initialize(stream);

    let attn_data = seeded_data(heads * seq * head_dim, 0.19, -0.09);
    let weight_data = seeded_data(out_dim * hidden, 0.14, -0.06);
    rt.set_data(attn, attn_data.as_slice());
    rt.set_data(weight, weight_data.as_slice());

    rt = cx.search(rt, CompileOptions::default().search_graph_limit(1));
    assert!(
        rt.kernel_names().contains(&"GenericMatmul"),
        "expected GenericMatmul to be selected, kernels: {:?}",
        rt.kernel_names()
    );

    rt.execute(&cx.dyn_map);
    let result = rt.get_f32(output.id);

    let mut expected = vec![0.0; seq * out_dim];
    for token in 0..seq {
        for out_col in 0..out_dim {
            let mut sum = 0.0;
            for inner in 0..hidden {
                let head = inner / head_dim;
                let dim = inner % head_dim;
                let attn_idx = head * seq * head_dim + token * head_dim + dim;
                sum += attn_data[attn_idx] * weight_data[out_col * hidden + inner];
            }
            expected[token * out_dim + out_col] = sum;
        }
    }

    assert_close(&result, &expected, 1e-5, 1e-5);
}

fn seeded_data(len: usize, scale: f32, bias: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = ((i * 37 + 11) % 97) as f32 / 97.0;
            x * scale + bias
        })
        .collect()
}

fn extract_forced_kernel_llir(cx: &mut Graph, kernel_name: &str) -> LLIRGraph {
    let egraph = cx.egraph().expect("search space should have an e-graph");
    let ops = cx
        .decoders
        .as_ref()
        .expect("search space should have registered egglog ops");
    let alternatives = egraph
        .alternatives()
        .expect("admissible typed alternatives");
    let kernel_nodes = op_candidates(egraph, |kind| {
        crate::typed_rules::GenericMatmulArgs::get_args(kind)
            .unwrap()
            .is_some()
    });
    assert!(
        !kernel_nodes.is_empty(),
        "expected at least one {kernel_name} candidate"
    );

    for (idx, kernel_node) in kernel_nodes.iter().enumerate() {
        let mut rng = StdRng::seed_from_u64(0x9EEE_0000 + idx as u64);
        let mut choices = random_initial_choice(&egraph.frozen, &alternatives, &mut rng);
        choices.extend(*kernel_node);

        if validate_choice_set(egraph, &alternatives, &choices).is_err() {
            continue;
        }
        let (llir, _) = reconstruct(egraph, &choices, ops, &cx.custom_ops)
            .expect("selected CUDA candidate should reconstruct");
        if llir_kernel_names(&llir).contains(&kernel_name) {
            return llir;
        }
    }

    panic!("could not extract a valid {kernel_name} candidate");
}

fn llir_kernel_names(llir: &LLIRGraph) -> Vec<&'static str> {
    llir.node_indices()
        .filter_map(|node| {
            llir[node]
                .to_dialect::<dyn KernelOp>()
                .map(|kernel| kernel.kernel_name())
        })
        .collect()
}
