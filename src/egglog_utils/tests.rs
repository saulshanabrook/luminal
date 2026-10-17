use super::*;
use crate::typed_reference::schema::{self, DType, DimExpr, EList, IList, Ir, OpKind};
use egglog_experimental::typed::builtins as egg;
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn ordered_decoders_fall_through_and_stop_at_the_first_success_or_error() -> Result<(), TypedError>
{
    fn fail(
        kind: &OpKind,
        _: &mut Vec<Ir>,
        _: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        if schema::ConstantArgs::get_args(kind)?.is_none() {
            return Ok(None);
        }
        Err(TypedError::Decode("backend rejected constant".into()))
    }
    let backend: DecodeFn = |kind, inputs, _| {
        if schema::ConstantArgs::get_args(kind)?.is_none() {
            return Ok(None);
        }
        assert_eq!(
            inputs.len(),
            1,
            "nonmatching decoders must leave inputs intact"
        );
        Ok(Some(LLIROp::new::<dyn crate::hlir::ReferenceOp>(Box::new(
            crate::hlir::Constant(42.0),
        ))))
    };
    let root = let_(
        "constant",
        Ir::Op(OpKind::Constant(1.0), [Ir::Input(0, "input", DType::F32())]),
    );
    let mut live = EGraph::default();
    live.register(&root)?;
    let space = SearchSpace::from_frozen(live.freeze()?, &[root])?;
    let alternatives = space.alternatives()?;
    let choices =
        random_initial_choice(&space.frozen, &alternatives, &mut StdRng::seed_from_u64(7));
    validate_choice_set(&space, &alternatives, &choices)?;
    let mut decoders = vec![crate::hlir::Exp2::decode as DecodeFn, backend, fail];
    decoders.extend_from_slice(crate::hlir::DECODERS);
    let (graph, roots) = reconstruct(&space, &choices, &decoders, &[])?;
    let op = graph[roots[0]]
        .to_dialect::<dyn crate::hlir::ReferenceOp>()
        .unwrap();
    assert_eq!(
        op.as_ref()
            .as_ref()
            .as_any()
            .downcast_ref::<crate::hlir::Constant>()
            .unwrap()
            .0,
        42.0
    );
    assert!(crate::hlir::cleanup(&OpKind::Constant(1.0))?);
    assert!(matches!(
        reconstruct(&space, &choices, &[fail, backend], &[]),
        Err(TypedError::Decode(message)) if message == "backend rejected constant"
    ));
    assert!(matches!(
        reconstruct(&space, &choices, &[crate::hlir::Exp2::decode], &[]),
        Err(TypedError::Decode(message)) if message.contains("no decoder accepted")
    ));
    // Without a backend match the CPU retains its old acceptance of extra inputs.
    let (graph, roots) = reconstruct(&space, &choices, crate::hlir::DECODERS, &[])?;
    let op = graph[roots[0]]
        .to_dialect::<dyn crate::hlir::ReferenceOp>()
        .unwrap();
    assert_eq!(
        op.as_ref()
            .as_ref()
            .as_any()
            .downcast_ref::<crate::hlir::Constant>()
            .unwrap()
            .0,
        1.0
    );
    assert_eq!(graph.edge_count(), 1);
    Ok(())
}

#[test]
fn nonmatching_core_decoders_leave_inputs_unchanged() -> Result<(), TypedError> {
    let mut graph = EGraph::default();
    let frozen = graph.freeze()?;
    let choices = Default::default();
    let mut context = DecodeContext::new(&frozen, &choices);
    let original = vec![Ir::Input(0, "untouched", DType::F32())];
    for decoder in crate::hlir::DECODERS {
        let mut inputs = original.clone();
        assert!(
            decoder(
                &OpKind::CustomOpKind(0, DType::F32()),
                &mut inputs,
                &mut context
            )?
            .is_none()
        );
        assert_eq!(inputs, original);
    }
    Ok(())
}

#[test]
fn dimension_metadata_prefers_mnum_on_equal_cost() -> Result<(), TypedError> {
    for other in [DimExpr::MVar("s"), DimExpr::MFloat(1.0)] {
        for constant_first in [false, true] {
            let constant = DimExpr::MNum(1);
            let (first, second) = if constant_first {
                (&constant, &other)
            } else {
                (&other, &constant)
            };
            let root = let_("dimension", first);
            let mut live = EGraph::default();
            live.register((&root, union(&root, second)))?;
            let frozen = live.freeze()?;
            let observed = frozen.lookup(&root)?;
            let class = frozen.class_of(&observed)?;
            let first = class.nodes().find(|node| !node.is_subsumed()).unwrap();
            let first = frozen.typed_node::<DimExpr>(first)?;
            assert_eq!(
                get_args(&first, |value: &egg::I64| DimExpr::MNum(value))?.is_some(),
                constant_first,
            );
            let choices = Default::default();
            let mut context = DecodeContext::new(&frozen, &choices);
            let (number,) = get_args(&context.selected(&observed)?, |value: &egg::I64| {
                DimExpr::MNum(value)
            })?
            .expect("an equally short MNum must win");
            assert_eq!(i64::try_from(&number)?, 1);
            assert_eq!(context.expression(&observed)?, Expression::from(1));
        }
    }
    Ok(())
}

#[test]
fn dimension_metadata_keeps_the_first_nonnumeric_tie() -> Result<(), TypedError> {
    for other in [DimExpr::MVar("t"), DimExpr::MFloat(1.0)] {
        for variable_first in [false, true] {
            let variable = DimExpr::MVar("s");
            let (first, second) = if variable_first {
                (&variable, &other)
            } else {
                (&other, &variable)
            };
            let root = let_("dimension", first);
            let mut live = EGraph::default();
            live.register((&root, union(&root, second)))?;
            let frozen = live.freeze()?;
            let observed = frozen.lookup(&root)?;
            let expected = frozen.nodes(&observed)?.next().unwrap();
            let choices = Default::default();
            let mut context = DecodeContext::new(&frozen, &choices);
            assert_eq!(context.selected(&observed)?, expected);
            if let Some((name,)) = get_args(&expected, |name: &egg::String| DimExpr::MVar(name))? {
                let name = String::try_from(&name)?;
                assert_eq!(
                    context.expression(&observed)?,
                    Expression::from(name.chars().next().unwrap()),
                );
            } else {
                assert!(context.expression(&observed).is_err());
            }
        }
    }
    Ok(())
}

#[test]
fn dimension_metadata_keeps_node_count_primary() -> Result<(), TypedError> {
    for constant_first in [false, true] {
        let constant = DimExpr::MNum(1);
        let iter = DimExpr::MIter();
        let (first, second) = if constant_first {
            (&constant, &iter)
        } else {
            (&iter, &constant)
        };
        let root = let_("dimension", first);
        let mut live = EGraph::default();
        live.register((&root, union(&root, second)))?;
        let frozen = live.freeze()?;
        let observed = frozen.lookup(&root)?;
        let choices = Default::default();
        let mut context = DecodeContext::new(&frozen, &choices);
        assert!(get_args(&context.selected(&observed)?, DimExpr::MIter)?.is_some());
        assert_eq!(context.expression(&observed)?, Expression::from('z'));
    }
    Ok(())
}

#[test]
fn cleanup_is_conditional_and_does_not_change_the_exact_snapshot() {
    let constant = let_("constant", Ir::Op(OpKind::Constant(1.0), IList::INil()));
    let lowered = Ir::Op(OpKind::LoopInput(7, 9, DType::F32()), IList::INil());
    let mut live = EGraph::default();
    live.register(&constant).unwrap();
    let mut only =
        SearchSpace::from_frozen(live.freeze().unwrap(), std::slice::from_ref(&constant)).unwrap();
    only.clean_hlir = true;
    assert!(
        only.alternatives().is_ok(),
        "cleanup must keep the last producer"
    );
    live.register(union(&constant, lowered)).unwrap();
    let mut space = SearchSpace::from_frozen(live.freeze().unwrap(), &[constant]).unwrap();
    space.clean_hlir = true;
    let class = space.frozen.class_of(&space.roots[0]).unwrap();
    assert_eq!(class.nodes().len(), 2);
    let alternatives = space.alternatives().unwrap();
    assert_eq!(alternatives[&class].len(), 1);
    let selected = space
        .frozen
        .typed_node::<Ir>(alternatives[&class][0])
        .unwrap();
    let (kind, _) = get_args(&selected, |kind: &OpKind, inputs: &IList| {
        Ir::Op(kind, inputs)
    })
    .unwrap()
    .unwrap();
    assert!(
        space
            .frozen
            .nodes(&kind)
            .unwrap()
            .any(|node| { schema::LoopInputArgs::get_args(&node).unwrap().is_some() })
    );
    assert_eq!(class.nodes().len(), 2);
    // When this kind is classified as a formerly direct-IR constructor,
    // it must not become new survivor evidence merely due to normalization.
    space.cleanup_ignored = |kind| Ok(schema::LoopInputArgs::get_args(kind)?.is_some());
    assert_eq!(space.alternatives().unwrap()[&class].len(), 2);
}

#[test]
fn cleanup_classification_requires_every_active_kind_producer() -> Result<(), TypedError> {
    let core = let_("core", OpKind::Constant(1.0));
    let other = let_("other", OpKind::LoopInput(7, 9, DType::F32()));
    let root = let_("root", Ir::Op(&core, IList::INil()));
    let mut live = EGraph::default();
    live.register((&core, &other, &root))?;
    live.register((
        union(&core, OpKind::Iota(DimExpr::MIter(), 3)),
        union(&other, OpKind::LoopOutput(7, 9, DType::F32())),
        union(&root, Ir::Op(&other, IList::INil())),
    ))?;
    let mut space = SearchSpace::from_frozen(live.freeze()?, std::slice::from_ref(&root))?;
    space.clean_hlir = true;
    space.cleanup_ignored = |kind| Ok(schema::LoopInputArgs::get_args(kind)?.is_some());
    assert_eq!(space.frozen.nodes(&space.frozen.lookup(&core)?)?.len(), 2);
    assert_eq!(space.frozen.nodes(&space.frozen.lookup(&other)?)?.len(), 2);
    let class = space.frozen.class_of(&space.roots[0])?;
    // Both core kinds are cleanable; the mixed ignored/nonignored class survives.
    assert_eq!(space.alternatives()?[&class].len(), 1);
    space.cleanup_ignored = |kind| {
        Ok(schema::LoopInputArgs::get_args(kind)?.is_some()
            || schema::LoopOutputArgs::get_args(kind)?.is_some())
    };
    assert_eq!(space.alternatives()?[&class].len(), 2);

    live.register(union(&core, OpKind::LoopInput(11, 13, DType::F32())))?;
    let mut space = SearchSpace::from_frozen(live.freeze()?, &[root])?;
    space.clean_hlir = true;
    assert_eq!(space.frozen.nodes(&space.frozen.lookup(&core)?)?.len(), 3);
    let class = space.frozen.class_of(&space.roots[0])?;
    // One noncleanable producer prevents removing the mixed core alternative.
    assert_eq!(space.alternatives()?[&class].len(), 2);
    Ok(())
}

#[test]
fn metadata_lists_and_dtypes_use_the_first_active_producer() -> Result<(), TypedError> {
    let list = let_("list", EList::from([2, 3]));
    let dtype = let_("dtype", DType::F64());
    let mut live = EGraph::default();
    live.register((
        &list,
        &dtype,
        union(&list, EList::from([4])),
        union(&dtype, DType::F32()),
    ))?;
    let frozen = live.freeze()?;
    let choices = Default::default();
    let mut context = DecodeContext::new(&frozen, &choices);
    let list = frozen.lookup(&list)?;
    let dtype = frozen.lookup(&dtype)?;
    assert_eq!(
        context.selected(&list)?,
        frozen.nodes(&list)?.next().unwrap()
    );
    assert_eq!(
        context.selected(&dtype)?,
        frozen.nodes(&dtype)?.next().unwrap()
    );
    let selected = context.selected(&list)?;
    let (head, _) = get_args(&selected, |head: &DimExpr, tail: &EList| {
        EList::ECons(head, tail)
    })?
    .unwrap();
    let expected_len = if context.expression(&head)?.to_usize() == Some(4) {
        1
    } else {
        2
    };
    assert_eq!(context.expressions(&list)?.len(), expected_len);
    assert_eq!(
        context.dtype(&dtype)?,
        crate::dtype::DType::try_from(context.selected(&dtype)?)?
    );
    Ok(())
}

#[test]
fn metadata_compatible_opkind_is_preferred_and_substituted_before_exact_decoding()
-> Result<(), TypedError> {
    let inconsistent = OpKind::Exp2([2, 3], [1], [1]);
    let compatible = OpKind::Log2([2], [1], [1]);
    let kind = let_("kind", &inconsistent);
    let root = let_("root", Ir::Op(&kind, [Ir::Input(0, "input", DType::F32())]));
    let mut live = EGraph::default();
    live.register((&root, union(&kind, &compatible)))?;
    let space = SearchSpace::from_frozen(live.freeze()?, &[root])?;
    let alternatives = space.alternatives()?;
    let observed = space.frozen.lookup(&kind)?;
    let class = space.frozen.class_of(&observed)?;
    let nodes = &alternatives[&class];
    let consistent = *nodes
        .iter()
        .find(|&&node| {
            schema::Log2Args::get_args(&space.frozen.typed_node::<OpKind>(node).unwrap())
                .unwrap()
                .is_some()
        })
        .unwrap();
    let inconsistent = *nodes
        .iter()
        .find(|&&node| {
            schema::Exp2Args::get_args(&space.frozen.typed_node::<OpKind>(node).unwrap())
                .unwrap()
                .is_some()
        })
        .unwrap();
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..32 {
        assert_eq!(
            random_initial_choice(&space.frozen, &alternatives, &mut rng)[&class],
            consistent
        );
    }
    let mut choices = random_initial_choice(&space.frozen, &alternatives, &mut rng);
    choices.insert(class, inconsistent);
    validate_choice_set(&space, &alternatives, &choices)?;
    // Selection inspection itself remains exact; substitution is Luminal policy.
    assert_eq!(
        DecodeContext::new(&space.frozen, &choices).selected(&observed)?,
        space.frozen.typed_node::<OpKind>(inconsistent)?
    );
    let (graph, roots) = reconstruct(&space, &choices, crate::hlir::DECODERS, &[])?;
    let op = graph[roots[0]]
        .to_dialect::<dyn crate::hlir::ReferenceOp>()
        .unwrap();
    assert!(op.as_ref().as_ref().as_any().is::<crate::hlir::Log2>());
    assert_eq!(choices[&class], inconsistent);
    Ok(())
}

#[test]
fn reconstructed_metadata_skips_producers_removed_by_missing_child_pruning()
-> Result<(), TypedError> {
    let shape = let_("shape", EList::from([2]));
    let root = let_(
        "root",
        Ir::Op(
            OpKind::Sin(&shape, [1], [1]),
            [Ir::Input(0, "input", DType::F32())],
        ),
    );
    let mut live = EGraph::default();
    live.register((&root, union(&shape, EList::from([3]))))?;
    let before = live.freeze()?;
    let first = before.nodes(&before.lookup(&shape)?)?.next().unwrap();
    let (head, _) = get_args(&first, |head: &DimExpr, tail: &EList| {
        EList::ECons(head, tail)
    })?
    .unwrap();
    let first_head = before.nodes(&head)?.next().unwrap();
    let (number,) = get_args(&first_head, |number: &egg::I64| DimExpr::MNum(number))?.unwrap();
    let removed = i64::try_from(&number)?;
    live.register(subsume(DimExpr::MNum(removed)))?;
    let space = SearchSpace::from_frozen(live.freeze()?, &[root])?;
    let alternatives = space.alternatives()?;
    let observed_shape = space.frozen.lookup(&shape)?;
    let class = space.frozen.class_of(&observed_shape)?;
    assert_eq!(class.nodes().len(), 2);
    assert_eq!(alternatives[&class].len(), 1);
    let forest = space.reconstruct_reference(&mut StdRng::seed_from_u64(7), &[])?;
    let op = forest.graph[forest.outputs[0]]
        .to_dialect::<dyn crate::hlir::ReferenceOp>()
        .unwrap();
    let sin = op
        .as_ref()
        .as_ref()
        .as_any()
        .downcast_ref::<crate::hlir::Sin>()
        .unwrap();
    assert_eq!(
        sin.shape[0].to_usize(),
        Some(if removed == 2 { 3 } else { 2 })
    );
    Ok(())
}

#[test]
fn coarse_hash_uses_all_admissible_ir_classes_and_ignores_metadata_values() -> Result<(), TypedError>
{
    let mut hashes = Vec::new();
    for (index, label, dtype) in [(0, "first", DType::F32()), (99, "second", DType::F64())] {
        let root = let_("root", Ir::Input(index, label, dtype));
        let mut live = EGraph::default();
        live.register(&root)?;
        let space = SearchSpace::from_frozen(live.freeze()?, std::slice::from_ref(&root))?;
        hashes.push(hash_search_space(&space)?);
        live.register(Ir::Input(100, "unreachable", DType::Int()))?;
        let with_unreachable =
            SearchSpace::from_frozen(live.freeze()?, std::slice::from_ref(&root))?;
        assert_ne!(
            hashes.last().copied().unwrap(),
            hash_search_space(&with_unreachable)?
        );
        let alternatives = with_unreachable.alternatives()?;
        assert_eq!(
            random_initial_choice(
                &with_unreachable.frozen,
                &alternatives,
                &mut StdRng::seed_from_u64(7)
            )
            .len(),
            2
        );
        live.register(subsume(Ir::Input(100, "unreachable", DType::Int())))?;
        let pruned = SearchSpace::from_frozen(live.freeze()?, &[root])?;
        assert_eq!(hashes.last().copied().unwrap(), hash_search_space(&pruned)?);
    }
    assert_eq!(hashes[0], hashes[1]);
    Ok(())
}

#[test]
fn schedule_reports_preserve_outer_cycles_and_optional_late_phase() -> Result<(), TypedError> {
    let root = let_("dimension", DimExpr::MVar("x"));
    let mut phases = base_phase_rules();
    phases.insert(
        RewritePhase::KernelLower,
        ruleset(rewrite(DimExpr::MVar("x"), 1i64)),
    );
    let full_start = std::time::Instant::now() - std::time::Duration::from_secs(1);
    let mut live = EGraph::default();
    let report = run_schedule(&mut live, full_start, &phases, false, false, |live| {
        live.register(&root)
    })?;
    assert!(report.total_time >= std::time::Duration::from_secs(1));
    assert_eq!(
        report
            .phases
            .iter()
            .map(|phase| phase.name.as_str())
            .collect::<Vec<_>>(),
        [
            "cycle 001 main",
            "cycle 002 main",
            "fuse late",
            "final expr",
            "cleanup",
            "post cleanup",
            "base cleanup"
        ]
    );
    assert!(report.phases[0].updated);
    assert!(!report.phases[1].updated);
    assert!(!report.full.num_matches_per_rule.is_empty());
    phases.insert(
        RewritePhase::LateAnalysis,
        ruleset(rewrite(DimExpr::MVar("late"), 2i64)),
    );
    let report = run_schedule(
        &mut live,
        std::time::Instant::now(),
        &phases,
        false,
        false,
        |_| Ok(()),
    )?;
    assert_eq!(report.phases.last().unwrap().name, "late pass 01");
    Ok(())
}

#[test]
fn choices_count_only_search_sorts_and_are_checked_against_their_class() {
    let a = let_("a", Ir::Input(0i64, "a", DType::F32()));
    let b = let_("b", Ir::Input(1i64, "b", DType::F32()));
    let mut live = EGraph::default();
    live.register((
        &a,
        &b,
        union(&a, Ir::Output(&b, 0i64)),
        union(DType::F32(), DType::F64()),
    ))
    .unwrap();
    let space = SearchSpace::from_frozen(live.freeze().unwrap(), &[a, b]).unwrap();
    let alternatives = space.alternatives().unwrap();
    assert_eq!(count_choice_sets_up_to(&alternatives, usize::MAX), 2);
    assert_eq!(count_choice_sets_up_to(&alternatives, 1), 1);
    assert_eq!(count_choice_sets_up_to(&alternatives, 0), 0);
    let mut choices =
        random_initial_choice(&space.frozen, &alternatives, &mut StdRng::seed_from_u64(7));
    validate_choice_set(&space, &alternatives, &choices).unwrap();
    let a = space.frozen.class_of(&space.roots[0]).unwrap();
    let b = space.frozen.class_of(&space.roots[1]).unwrap();
    choices.insert(a, choices[&b]);
    assert!(validate_choice_set(&space, &alternatives, &choices).is_err());
    assert!(
        DecodeContext::new(&space.frozen, &choices)
            .selected(&space.roots[0])
            .is_err()
    );
    choices.remove(&a);
    assert!(validate_choice_set(&space, &alternatives, &choices).is_err());
}

#[test]
fn choice_validation_still_rejects_unsupported_structural_ir() -> Result<(), TypedError> {
    let root = let_(
        "root",
        schema::consumed_buffer(Ir::Input(0, "input", DType::F32())),
    );
    let mut live = EGraph::default();
    live.register(&root)?;
    let space = SearchSpace::from_frozen(live.freeze()?, &[root])?;
    let alternatives = space.alternatives()?;
    let choices =
        random_initial_choice(&space.frozen, &alternatives, &mut StdRng::seed_from_u64(7));
    assert!(matches!(
        validate_choice_set(&space, &alternatives, &choices),
        Err(TypedError::Invalid(message)) if message.contains("No extractor for IR")
    ));
    Ok(())
}

#[test]
fn reconstruction_uses_the_selected_input_list() {
    let a = let_("a", Ir::Input(0i64, "a", DType::F32()));
    let b = let_("b", Ir::Input(1i64, "b", DType::F32()));
    let inputs = let_("inputs", IList::ICons(&a, IList::INil()));
    let shape = EList::from([1i64]);
    let root = let_(
        "root",
        Ir::Op(OpKind::Exp2(&shape, &shape, &shape), &inputs),
    );
    let mut live = EGraph::default();
    live.register((
        &a,
        &b,
        &inputs,
        &root,
        union(&inputs, IList::ICons(&b, IList::INil())),
    ))
    .unwrap();
    let space = SearchSpace::from_frozen(live.freeze().unwrap(), &[root]).unwrap();
    let alternatives = space.alternatives().unwrap();
    let mut choices =
        random_initial_choice(&space.frozen, &alternatives, &mut StdRng::seed_from_u64(7));
    let observed_inputs = space.frozen.lookup(&inputs).unwrap();
    let observed_b = space.frozen.lookup(&b).unwrap();
    let class = space.frozen.class_of(&observed_inputs).unwrap();
    let selected = class
        .nodes()
        .find(|&node| {
            let typed = space.frozen.typed_node::<IList>(node).unwrap();
            get_args(&typed, |head: &Ir, tail: &IList| IList::ICons(head, tail))
                .unwrap()
                .is_some_and(|(head, _)| head == observed_b)
        })
        .unwrap();
    choices.insert(class, selected);
    let (graph, _) = reconstruct(&space, &choices, crate::hlir::DECODERS, &[]).unwrap();
    let inputs: Vec<_> = graph
        .node_weights()
        .filter_map(|op| op.to_op::<crate::hlir::Input>())
        .collect();
    assert_eq!(inputs.len(), 1);
    assert_eq!(inputs[0].node, 1);
}

#[test]
fn decoder_replacements_require_an_originally_reachable_exact_producer() -> Result<(), TypedError> {
    fn recover_nine(
        kind: &OpKind,
        inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        if schema::Exp2Args::get_args(kind)?.is_none() {
            return Ok(None);
        }
        let mut replacement = None;
        for class in context
            .frozen
            .eclasses()
            .filter(|class| class.sort() == Ir::sort_ref())
        {
            for producer in class.nodes() {
                let node = context.frozen.typed_node::<Ir>(producer)?;
                if get_args(
                    &node,
                    |id: &egg::I64, label: &egg::String, dtype: &DType| Ir::Input(id, label, dtype),
                )?
                .is_some_and(|(id, _, _)| i64::try_from(&id).unwrap() == 9)
                {
                    replacement = Some(node);
                    break;
                }
            }
        }
        inputs[0] = replacement.expect("fixture has Input 9");
        crate::hlir::Exp2::decode(kind, inputs, context)
    }

    for case in ["reachable", "unreachable", "unselected same-class"] {
        let input = let_(
            "input",
            Ir::Input(
                if case == "reachable" { 9 } else { 0 },
                "input",
                DType::F32(),
            ),
        );
        let other = let_("other", Ir::Input(9, "other", DType::F32()));
        let original = Ir::Op(OpKind::Log2([1], [1], [1]), [&input]);
        let root = let_("root", Ir::Op(OpKind::Exp2([1], [1], [1]), [original]));
        let mut live = EGraph::default();
        live.register(&root)?;
        if case != "reachable" {
            live.register(&other)?;
        }
        if case == "unselected same-class" {
            live.register(union(&input, &other))?;
        }
        let space = SearchSpace::from_frozen(live.freeze()?, &[root])?;
        let alternatives = space.alternatives()?;
        let mut choices =
            random_initial_choice(&space.frozen, &alternatives, &mut StdRng::seed_from_u64(7));
        if case == "unselected same-class" {
            let class = space.frozen.class_of(&space.frozen.lookup(&input)?)?;
            let selected = class
                .nodes()
                .find(|&node| {
                    let node = space.frozen.typed_node::<Ir>(node).unwrap();
                    get_args(
                        &node,
                        |id: &egg::I64, label: &egg::String, dtype: &DType| {
                            Ir::Input(id, label, dtype)
                        },
                    )
                    .unwrap()
                    .is_some_and(|(id, _, _)| i64::try_from(&id).unwrap() == 0)
                })
                .unwrap();
            choices.insert(class, selected);
        }
        let mut decoders = vec![recover_nine as DecodeFn];
        decoders.extend_from_slice(crate::hlir::DECODERS);
        let result = reconstruct(&space, &choices, &decoders, &[]);
        if case == "reachable" {
            let (graph, roots) = result?;
            assert_eq!(
                graph.node_count(),
                3,
                "the unused original Log2 must remain"
            );
            let source = graph
                .neighbors_directed(roots[0], petgraph::Direction::Incoming)
                .next()
                .unwrap();
            assert_eq!(graph[source].to_op::<crate::hlir::Input>().unwrap().node, 9);
        } else {
            assert!(
                matches!(result, Err(TypedError::Decode(message)) if message.contains("not reachable")),
                "{case}"
            );
        }
    }
    Ok(())
}

#[test]
fn metadata_without_a_finite_tree_returns_an_error() {
    let value = let_("value", DimExpr::MNum(1i64));
    let mut live = EGraph::default();
    live.register((
        &value,
        union(&value, &value + &value),
        delete(DimExpr::MNum(1i64)),
    ))
    .unwrap();
    let frozen = live.freeze().unwrap();
    let choices = Default::default();
    assert!(
        DecodeContext::new(&frozen, &choices)
            .expression(&frozen.lookup(&value).unwrap())
            .is_err()
    );
}

#[test]
fn final_schedule_preserves_cleanup_order() {
    let input = let_("input", Ir::Input(0i64, "input", DType::F32()));
    let output = let_("output", Ir::Output(&input, 0i64));
    let mut phases = base_phase_rules();
    phases.insert(
        RewritePhase::PostCleanup,
        ruleset(|input: &Ir, index: &egg::I64| rewrite(Ir::Output(input, index), input)),
    );
    let mut live = EGraph::default();
    live.register(&output).unwrap();
    run_schedule(
        &mut live,
        std::time::Instant::now(),
        &phases,
        false,
        false,
        |_| Ok(()),
    )
    .unwrap();
    assert_eq!(
        final_phases(&phases, false).map(|(phase, _)| phase),
        [
            RewritePhase::KernelFuseLate,
            RewritePhase::Expr,
            RewritePhase::Cleanup,
            RewritePhase::PostCleanup,
            RewritePhase::BaseCleanup,
        ]
    );
    let frozen = live.freeze().unwrap();
    assert!(
        frozen
            .nodes(&frozen.lookup(&output).unwrap())
            .unwrap()
            .any(|node| {
                get_args(
                    &node,
                    |index: &egg::I64, label: &egg::String, dtype: &DType| {
                        Ir::Input(index, label, dtype)
                    },
                )
                .unwrap()
                .is_some()
            })
    );
}
