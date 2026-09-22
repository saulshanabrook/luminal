use super::*;
use crate::egglog_utils::{DecodeContext, SearchSpace, random_initial_choice, reconstruct};
use egglog_experimental::typed::FrozenValueView;
use rand::{SeedableRng, rngs::StdRng};
use rustc_hash::FxHashMap;
use schema::{DType, EList, IList};

#[test]
fn dtype_native_conversion_requires_a_concrete_constructor() -> Result<(), TypedError> {
    use crate::dtype::DType as Host;
    for host in [
        Host::F32,
        Host::F64,
        Host::F16,
        Host::Bf16,
        Host::TF32,
        Host::Int,
        Host::I64,
        Host::I4,
        Host::U4,
        Host::I8,
        Host::U8,
        Host::I16,
        Host::U16,
        Host::Bool,
        Host::F8UE8M0,
        Host::F8E4M3,
        Host::F8E5M2,
        Host::F6E2M3,
        Host::F6E3M2,
        Host::F4E2M1,
    ] {
        let authored = DType::from(host);
        assert_eq!(Host::try_from(&authored)?, host);
        assert_eq!(Host::try_from(authored.clone())?, host);
        let capture = let_("dtype", authored);
        let mut live = EGraph::new(EGraphOptions::default());
        live.register(&capture)?;
        let frozen = live.freeze()?;
        let unresolved = frozen.lookup(&capture)?;
        let tuples = live.num_tuples()?;
        assert!(Host::try_from(&unresolved).is_err());
        let selected = frozen.nodes(&unresolved)?.next().unwrap();
        assert_eq!(Host::try_from(&selected)?, host);
        assert_eq!(live.num_tuples()?, tuples);
        drop(frozen);
        drop(live);
        assert_eq!(Host::try_from(selected)?, host);
    }
    assert!(Host::try_from(var::<DType>("unknown")).is_err());
    let root = let_("alternatives", DType::F32());
    let mut live = EGraph::new(EGraphOptions::default());
    live.register((&root, union(&root, DType::Int64())))?;
    let frozen = live.freeze()?;
    let values = frozen
        .nodes(&frozen.lookup(&root)?)?
        .map(Host::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    assert!(values.contains(&Host::F32));
    assert!(values.contains(&Host::I64));
    Ok(())
}

#[test]
fn frozen_fields_are_exact_values_not_display_strings() {
    let label = "Input \"quoted\" \\backslash\nline";
    let capture = let_("input", Ir::Input(i64::MAX, label, DType::F32()));
    let mut live = EGraph::new(EGraphOptions::default());
    live.register(&capture).unwrap();
    let frozen = live.freeze().unwrap();
    let root = frozen.lookup(&capture).unwrap();
    assert!(live.freeze().unwrap().as_view(&root).is_err());
    let space = SearchSpace::from_frozen(frozen, &[capture.clone(), capture]).unwrap();
    drop(live);
    assert_eq!(space.roots[0], space.roots[1]);
    let forest = space
        .reconstruct_reference(&mut StdRng::seed_from_u64(7), &[])
        .unwrap();
    assert_eq!(forest.outputs[0], forest.outputs[1]);
    let input = forest.graph[forest.outputs[0]]
        .to_op::<crate::hlir::Input>()
        .unwrap();
    assert_eq!(input.node, i64::MAX as usize);
    assert_eq!(input.label, label);
}

#[test]
fn frozen_cycles_and_ordered_roots_keep_snapshot_identity() {
    let root = let_("root", Ir::Input(7i64, "root", DType::F32()));
    let other = let_("other", Ir::Input(8i64, "other", DType::F32()));
    let mut live = EGraph::new(EGraphOptions::default());
    live.register((&root, &other, union(&root, Ir::Output(&root, 0i64))))
        .unwrap();
    let frozen = live.freeze().unwrap();
    let a = SearchSpace::from_frozen(frozen.clone(), &[root.clone(), other.clone()]).unwrap();
    let b = SearchSpace::from_frozen(frozen, &[other, root.clone(), root]).unwrap();
    assert_eq!(
        b.roots,
        [a.roots[1].clone(), a.roots[0].clone(), a.roots[0].clone()]
    );
    let FrozenValueView::EClass(class) = a.frozen.as_view(&a.roots[0]).unwrap().view() else {
        unreachable!()
    };
    assert_eq!(class.nodes().len(), 2);
    assert!(
        class
            .nodes()
            .any(|node| node.eclass_children().any(|child| child == class))
    );
    let alternatives = a.alternatives().unwrap();
    let mut choices =
        random_initial_choice(&a.frozen, &alternatives, &mut StdRng::seed_from_u64(7));
    choices.insert(
        class,
        class
            .nodes()
            .find(|node| node.eclass_children().any(|child| child == class))
            .unwrap(),
    );
    let (graph, roots) = reconstruct(&a, &choices, crate::hlir::DECODERS, &[]).unwrap();
    assert_eq!(roots.len(), 2);
    assert!(petgraph::algo::is_cyclic_directed(&graph));
}

#[test]
fn reference_constants_decode_stored_float_bits() {
    let expected = [-0.0f64, f64::INFINITY, f64::NEG_INFINITY];
    let captures: Vec<_> = expected
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let_(
                format!("float/{index}"),
                Ir::Op(OpKind::Constant(value), IList::INil()),
            )
        })
        .collect();
    let mut live = EGraph::new(EGraphOptions::default());
    live.register(&captures).unwrap();
    let space = SearchSpace::from_frozen(live.freeze().unwrap(), &captures).unwrap();
    let forest = space
        .reconstruct_reference(&mut StdRng::seed_from_u64(7), &[])
        .unwrap();
    for (&expected, &output) in expected.iter().zip(&forest.outputs) {
        let op = forest.graph[output]
            .to_dialect::<dyn crate::hlir::ReferenceOp>()
            .unwrap();
        let crate::hlir::ReferenceData::F32(actual) = op.execute(Vec::new(), &FxHashMap::default())
        else {
            unreachable!()
        };
        assert_eq!(actual[0].to_bits(), (expected as f32).to_bits());
    }
}

#[test]
fn selection_pruning_does_not_erase_producer_free_frozen_classes() {
    let missing = let_("missing", Ir::Input(1i64, "missing", DType::F32()));
    let root = let_("root", Ir::Input(2i64, "root", DType::F32()));
    let mut live = EGraph::new(EGraphOptions::default());
    live.register((
        &missing,
        &root,
        union(&root, Ir::Output(&missing, 0i64)),
        delete(Ir::Input(1i64, "missing", DType::F32())),
    ))
    .unwrap();
    let frozen = live.freeze().unwrap();
    let observed_root = frozen.lookup(&root).unwrap();
    assert_eq!(
        frozen
            .nodes(&frozen.lookup(&missing).unwrap())
            .unwrap()
            .len(),
        0
    );
    assert_eq!(frozen.nodes(&observed_root).unwrap().len(), 2);
    let space = SearchSpace::from_frozen(frozen.clone(), std::slice::from_ref(&root)).unwrap();
    let alternatives = space.alternatives().unwrap();
    let FrozenValueView::EClass(class) = frozen.as_view(&observed_root).unwrap().view() else {
        unreachable!()
    };
    assert_eq!(alternatives[&class].len(), 1);
    let invalid = SearchSpace::from_frozen(frozen.clone(), &[root, missing]).unwrap();
    assert!(matches!(
        invalid.alternatives(),
        Err(TypedError::UnresolvedRoot { index: Some(1), .. })
    ));
    assert_eq!(frozen.nodes(&observed_root).unwrap().len(), 2);
}

#[test]
fn dimension_selection_handles_cycles_and_repeated_children() {
    let root = let_("dim", DimExpr::from(3i64));
    let list = let_("list", EList::from([root.clone(), root.clone()]));
    let mut live = EGraph::new(EGraphOptions::default());
    live.register((&root, &list, union(&root, &root + 0i64)))
        .unwrap();
    let frozen = live.freeze().unwrap();
    let choices = Default::default();
    let mut context = DecodeContext::new(&frozen, &choices);
    assert_eq!(
        context
            .expression(&frozen.lookup(&root).unwrap())
            .unwrap()
            .to_usize(),
        Some(3)
    );
    let decoded = context.expressions(&frozen.lookup(&list).unwrap()).unwrap();
    assert_eq!(
        decoded
            .iter()
            .map(|value| value.to_usize())
            .collect::<Vec<_>>(),
        [Some(3), Some(3)]
    );
}

#[test]
fn singleton_interval_prefers_the_numeric_representative() {
    let intervals = [('s', DimInterval::new(1, 1))].into_iter().collect();
    let result = simplify_dimension(Expression::from('s').lt(2), Some(&intervals)).unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result.exec(&[('s', 1)].into_iter().collect()), Some(1));
    assert_eq!(result.to_usize(), Some(1));
}
#[test]
fn cleanup_expression_facts_delete_every_existing_row() {
    use schema::EList;

    let x = DimExpr::MVar("x");
    let two = DimExpr::MNum(2);
    let list = EList::ECons(&x, EList::ENil());
    let replace = DimExpr::MReplace(&x, &x, &two);
    let replace_list = EList::MReplaceList(&list, &x, &two);
    let replace_nth = EList::ReplaceNthFromEnd(&list, &two, 0);
    let remove_nth = EList::RemoveNthFromEnd(&list, 0);
    let row_major = EList::RowMajor(&list);
    let queries = [
        Fact::from(&replace),
        Fact::from(&replace_list),
        Fact::from(&replace_nth),
        Fact::from(&remove_nth),
        Fact::from(&row_major),
        Fact::from(list.len()),
        Fact::from(list.nth_from_end(0)),
        Fact::from(list.n_elements()),
    ];
    let mut egraph = EGraph::new(EGraphOptions::default());
    egraph
        .register((
            &replace,
            &replace_list,
            &replace_nth,
            &remove_nth,
            &row_major,
            set(list.len(), 1),
            set(list.nth_from_end(0), &x),
            set(list.n_elements(), &x),
        ))
        .unwrap();
    assert!(egraph.check(&queries).unwrap());
    egraph.run(static_rules::cleanup.saturate()).unwrap();
    for query in &queries {
        assert!(!egraph.check(query).unwrap(), "cleanup left {query:?}");
    }
    let count = egraph.num_tuples().unwrap();
    assert!(!egraph.run(&static_rules::cleanup).unwrap().updated);
    assert_eq!(egraph.num_tuples().unwrap(), count);
}
