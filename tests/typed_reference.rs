use egglog_experimental::typed::{
    EGraph, EGraphOptions, builtins as egg, get_args, let_, set, union,
};
use luminal::{
    egglog_utils::SearchSpace,
    graph::{Graph, LLIRGraph, unroll_loops_in_llir},
    hlir::{LoopEnd, LoopStart, Output, ReferenceRuntime},
    op::Runtime,
    prelude::{CompileOptions, NodeIndex},
    typed_reference::{
        self,
        schema::{DType, DimExpr, Ir, OpKind},
    },
};
use petgraph::Direction;
use rand::{SeedableRng, rngs::StdRng};
use rustc_hash::FxHashMap;

#[test]
fn singleton_loop_trip_count_is_numeric_in_both_producer_orders() {
    for numeric_first in [true, false] {
        let numeric = DimExpr::MNum(3);
        let symbolic = DimExpr::MVar("s");
        let trips = let_("trips", if numeric_first { &numeric } else { &symbolic });
        let initial = Ir::Input(0, "initial", DType::F32());
        let increment = Ir::Input(1, "increment", DType::F32());
        let start = Ir::LoopStart(initial, 0, 0, &trips, DType::F32());
        let body = Ir::Op(
            OpKind::Add(
                [1],
                [DimExpr::MIter()],
                [DimExpr::MIter()],
                [DimExpr::MIter()],
            ),
            [start, increment],
        );
        let end = Ir::LoopEnd(body, 0, 0, DType::F32());
        let output = let_("output", Ir::Output(end, 2));
        let mut live = EGraph::default();
        live.register(&trips).unwrap();
        live.register((
            set(symbolic.lower(), 3),
            set(symbolic.upper(), 3),
            union(&symbolic, &numeric),
            &output,
        ))
        .unwrap();
        let space = SearchSpace::from_frozen(live.freeze().unwrap(), &[output]).unwrap();
        let observed = space.frozen.lookup(&trips).unwrap();
        let producers: Vec<_> = space.frozen.nodes(&observed).unwrap().collect();
        assert_eq!(producers.len(), 2);
        for (producer, is_numeric) in producers.iter().zip([numeric_first, !numeric_first]) {
            if is_numeric {
                let (value,) = get_args(producer, |n: &egg::I64| DimExpr::MNum(n))
                    .unwrap()
                    .expect("expected the existing MNum producer");
                assert_eq!(i64::try_from(value).unwrap(), 3);
            } else {
                let (name,) = get_args(producer, |name: &egg::String| DimExpr::MVar(name))
                    .unwrap()
                    .expect("expected the existing MVar producer");
                assert_eq!(String::try_from(name).unwrap(), "s");
            }
        }
        let mut forest = space
            .reconstruct_reference(&mut StdRng::seed_from_u64(7), &[])
            .unwrap();
        let decoded = forest
            .graph
            .node_weights()
            .find_map(|op| op.to_op::<LoopStart>())
            .unwrap()
            .iters;
        assert_eq!(decoded.to_usize(), Some(3), "numeric_first={numeric_first}");
        unroll_loops_in_llir(&mut forest.graph);
        assert!(
            forest
                .graph
                .node_weights()
                .all(|op| { op.to_op::<LoopStart>().is_none() && op.to_op::<LoopEnd>().is_none() })
        );
        let mut runtime = ReferenceRuntime::default();
        runtime.load_llir(&forest.graph);
        runtime.set_data(NodeIndex::new(0), vec![10.0f32]);
        runtime.set_data(NodeIndex::new(1), vec![2.0f32]);
        runtime.execute(&[('s', 3)].into_iter().collect());
        assert_eq!(
            runtime.get_f32(NodeIndex::new(2)),
            &vec![16.0],
            "numeric_first={numeric_first}",
        );
    }
}

fn execute_fixture(
    graph: &LLIRGraph,
    inputs: [NodeIndex; 4],
    outputs: [NodeIndex; 3],
) -> Vec<Vec<f32>> {
    let mut runtime = ReferenceRuntime::default();
    runtime.load_llir(graph);
    for (input, data) in inputs.into_iter().zip([
        vec![1.0f32, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
        vec![2.0, 2.0, 3.0],
    ]) {
        runtime.set_data(input, data);
    }
    runtime.execute(&FxHashMap::default());
    outputs
        .into_iter()
        .map(|output| runtime.get_f32(output).clone())
        .collect()
}

#[test]
fn retained_outputs_share_one_frozen_forest() {
    let mut graph = Graph::new();
    let b = graph.tensor(3);
    let c = graph.tensor(3);
    let g = graph.tensor(3);
    let e = graph.tensor(3);
    let p = b * c;
    let a = (p + g).output();
    let d = (p / e).sin().output();
    let p_out = p.output();
    let input_ids = [b.id, c.id, g.id, e.id];
    let output_ids = [a.id, d.id, p_out.id];

    let captures = typed_reference::capture_graph(&graph).unwrap();
    let mut sink_nodes: Vec<_> = graph.externals(Direction::Outgoing).collect();
    sink_nodes.sort_unstable_by_key(|node| node.index());
    let by_id: FxHashMap<_, _> = sink_nodes
        .into_iter()
        .zip(captures)
        .map(|(node, capture)| (NodeIndex::new(graph.get_op::<Output>(node).node), capture))
        .collect();
    let outputs = [
        by_id[&a.id].clone(),
        by_id[&d.id].clone(),
        by_id[&p_out.id].clone(),
        by_id[&p_out.id].clone(),
    ];
    let mut egraph = EGraph::new(EGraphOptions::default());
    egraph.register(&outputs).unwrap();
    typed_reference::optimize(&mut egraph, &FxHashMap::default(), false).unwrap();
    let count = egraph.num_tuples().unwrap();
    let frozen = egraph.freeze().unwrap();
    assert_eq!(
        egraph.num_tuples().unwrap(),
        count,
        "freeze must not evaluate"
    );
    assert_eq!(
        frozen.lookup(&outputs[2]).unwrap(),
        frozen.lookup(&outputs[3]).unwrap()
    );

    // The snapshot remains usable after mutation, scope restoration, and drop.
    egraph.push().unwrap();
    let later = let_("later", Ir::Input(999i64, "later", DType::F32()));
    egraph.register(&later).unwrap();
    assert!(frozen.lookup(&later).is_err());
    egraph.pop().unwrap();
    drop(egraph);

    let space = SearchSpace::from_frozen(frozen.clone(), &outputs).unwrap();
    assert_eq!(space.roots.len(), 4);
    assert_eq!(space.roots[2], space.roots[3]);
    let forest = space
        .reconstruct_reference(&mut StdRng::seed_from_u64(7), &[])
        .unwrap();
    assert_eq!(forest.outputs.len(), 4);
    assert_eq!(forest.outputs[2], forest.outputs[3]);
    let producer = forest
        .graph
        .neighbors_directed(forest.outputs[2], Direction::Incoming)
        .next()
        .unwrap();
    assert_eq!(
        forest
            .graph
            .neighbors_directed(producer, Direction::Outgoing)
            .count(),
        3,
        "p is shared by a, d, and p_out"
    );

    let actual = execute_fixture(&forest.graph, input_ids, output_ids);
    let expected = [
        vec![11.0f32, 18.0, 27.0],
        vec![2.0f32.sin(), 5.0f32.sin(), 6.0f32.sin()],
        vec![4.0, 10.0, 18.0],
    ];
    for (actual, expected) in actual.iter().zip(&expected) {
        assert_eq!(actual.len(), expected.len());
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() < 1e-5),
            "{actual:?} != {expected:?}"
        );
    }

    // A different ordered list from exactly the same rootless snapshot.
    let reordered = [outputs[2].clone(), outputs[0].clone(), outputs[2].clone()];
    let reordered_space = SearchSpace::from_frozen(frozen.clone(), &reordered).unwrap();
    assert_eq!(
        reordered_space.roots,
        [
            space.roots[2].clone(),
            space.roots[0].clone(),
            space.roots[2].clone()
        ]
    );
    let alternatives = space.alternatives().unwrap();
    for (class, nodes) in reordered_space.alternatives().unwrap() {
        assert_eq!(
            nodes, alternatives[&class],
            "constructor identities must not depend on selected root order"
        );
    }
}

#[test]
fn graph_reference_route_keeps_multiple_roots() {
    let mut graph = Graph::new();
    let a = graph.tensor(3);
    let b = graph.tensor(3);
    let product = a * b;
    product.output();
    (product + a).output();
    graph.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let space = graph.egraph().unwrap();
    assert_eq!(space.roots.len(), 2);
    assert!(
        space
            .roots
            .iter()
            .all(|root| space.frozen.nodes(root).unwrap().count() > 0)
    );
}

#[test]
fn frozen_root_errors_retain_the_callers_index() {
    use egglog_experimental::typed::TypedError;

    let known = let_("known", Ir::Input(0i64, "known", DType::F32()));
    let missing = let_("missing", Ir::Input(1i64, "missing", DType::F32()));
    let mut egraph = EGraph::new(EGraphOptions::default());
    egraph.register(&known).unwrap();
    let frozen = egraph.freeze().unwrap();
    assert!(matches!(
        SearchSpace::from_frozen(frozen.clone(), &[known.clone(), known, missing]),
        Err(TypedError::UnresolvedRoot { index: Some(2), .. })
    ));
    assert!(
        SearchSpace::from_frozen(frozen, &[])
            .unwrap()
            .roots
            .is_empty()
    );
}

#[test]
fn repeated_optimization_reuses_rules_and_activates_intervals() {
    use egglog_experimental::typed::eq;
    use luminal::{shape::DimInterval, typed_reference::schema::DimExpr};
    use std::collections::HashSet;

    let input = let_("first", Ir::Input(0i64, "first", DType::F32()));
    let dimension = let_("dimension", DimExpr::from("s") + 1i64);
    let mut egraph = EGraph::new(EGraphOptions::default());
    egraph.register((&input, &dimension)).unwrap();
    typed_reference::optimize(&mut egraph, &FxHashMap::default(), false).unwrap();
    assert!(egraph.check(eq(input.dtype(), DType::F32())).unwrap());
    let rules: HashSet<_> = egraph
        .stats()
        .unwrap()
        .search_and_apply_time_per_rule
        .into_keys()
        .collect();
    assert!(
        !rules.is_empty(),
        "native reports must expose executed rule identities"
    );

    // No interval rules have run yet. A second call must reuse all fixed rules,
    // including the selected catalog from the parameterized unroll generator.
    let count = egraph.num_tuples().unwrap();
    typed_reference::optimize(&mut egraph, &FxHashMap::default(), false).unwrap();
    assert_eq!(egraph.num_tuples().unwrap(), count);
    assert_eq!(
        egraph
            .stats()
            .unwrap()
            .search_and_apply_time_per_rule
            .into_keys()
            .collect::<HashSet<_>>(),
        rules
    );

    // Existing cursors still process newly added inputs, and an initially
    // inactive group is installed when interval facts first become available.
    let later = let_("later", Ir::Input(1i64, "later", DType::F64()));
    egraph.register(&later).unwrap();
    let intervals = [('s', DimInterval::new(2, 5))].into_iter().collect();
    typed_reference::optimize(&mut egraph, &intervals, false).unwrap();
    assert!(
        egraph
            .check((
                eq(later.dtype(), DType::F64()),
                eq(dimension.lower(), 3i64),
                eq(dimension.upper(), 6i64),
            ))
            .unwrap()
    );
    let interval_rules: HashSet<_> = egraph
        .stats()
        .unwrap()
        .search_and_apply_time_per_rule
        .into_keys()
        .collect();
    assert!(interval_rules.is_superset(&rules));
    assert!(interval_rules.len() > rules.len());
    typed_reference::optimize(&mut egraph, &intervals, false).unwrap();
    assert_eq!(
        egraph
            .stats()
            .unwrap()
            .search_and_apply_time_per_rule
            .into_keys()
            .collect::<HashSet<_>>(),
        interval_rules
    );
}
