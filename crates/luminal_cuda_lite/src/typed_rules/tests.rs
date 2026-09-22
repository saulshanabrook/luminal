use super::*;

#[test]
fn unrelated_kind_is_rejected_by_every_decoder_without_changing_inputs() -> Result<(), TypedError> {
    use egglog_experimental::typed::FrozenValueView;
    use luminal::egglog_utils::DecodeContext;
    use luminal::op::Runtime;

    let kind = let_("kind", OpKind::Constant(2.0));
    let input = let_("input", Ir::Input(7, "sentinel", DType::F32()));
    let mut graph = EGraph::default();
    graph.register((&kind, &input))?;
    let frozen = graph.freeze()?;
    let kind = frozen.nodes(&frozen.lookup(&kind)?)?.next().unwrap();
    let input = frozen.lookup(&input)?;
    let FrozenValueView::EClass(expected) = frozen.as_view(&input)?.view() else {
        unreachable!()
    };
    let choices = Default::default();
    let mut context = DecodeContext::new(&frozen, &choices);
    for decode in crate::runtime::CudaRuntime::DECODERS {
        let mut inputs = vec![input.clone()];
        assert!(decode(&kind, &mut inputs, &mut context)?.is_none());
        assert_eq!(inputs.len(), 1);
        let FrozenValueView::EClass(actual) = frozen.as_view(&inputs[0])?.view() else {
            unreachable!()
        };
        assert_eq!(actual, expected);
    }
    Ok(())
}

#[test]
fn complete_rulesets_install_without_a_cuda_device() -> Result<(), TypedError> {
    use crate::host::flashinfer::flashinfer_attention as fi;
    use luminal::op::Runtime;

    // Validate the complete default recipe without CUDA execution. Device
    // detection is fallible; FlashInfer is also installed explicitly below so
    // both sides of its hardware gate validate every authored rule.
    let (phases, actions) = crate::runtime::CudaRuntime::rules();
    let mut graph = EGraph::default();
    graph.register((actions, fi::initial_actions()))?;
    for rules in phases.values() {
        graph.run(rules.repeat(0))?;
    }
    graph.run(
        ruleset((
            &fi::matmul_backend,
            &fi::kernel_fuse_late2,
            &fi::kernel_fuse_late_pre,
        ))
        .repeat(0),
    )?;
    Ok(())
}

#[test]
fn constant_and_iota_lowering_preserve_arbitrary_input_lists() -> Result<(), TypedError> {
    let rules = ruleset((
        &crate::kernel::hlir::kernel_constant::kernel_lower,
        &crate::kernel::hlir::kernel_iota::kernel_lower,
    ));
    for (value, range, inputs) in [
        (2.5, 3, IList::INil()),
        (3.5, 4, IList::from([Ir::Input(0, "input", DType::F32())])),
    ] {
        let mut graph = EGraph::default();
        let constant = let_("constant", Ir::Op(OpKind::Constant(value), &inputs));
        let iota = let_(
            "iota",
            Ir::Op(
                OpKind::Iota(DimExpr::MIter(), DimExpr::MNum(range)),
                &inputs,
            ),
        );
        let lowered_constant = Ir::Op(kernel_constant(value, DType::F32()), &inputs);
        let lowered_iota = Ir::Op(kernel_iota(DimExpr::MIter(), DimExpr::MNum(range)), &inputs);
        graph.register((&constant, &iota))?;
        graph.run(&rules)?;
        assert!(graph.check((
            eq(&constant, &lowered_constant),
            eq(&iota, &lowered_iota),
            eq(lowered_constant.dtype(), DType::F32()),
            eq(lowered_iota.dtype(), DType::Int())
        ))?);
        if inputs != IList::INil() {
            assert!(!graph.check(eq(
                &constant,
                Ir::Op(kernel_constant(value, DType::F32()), IList::INil())
            ))?);
        }
        let frozen = graph.freeze()?;
        for source in [&constant, &iota] {
            let value = frozen.lookup(source)?;
            assert!(
                frozen
                    .nodes(&value)?
                    .all(|node| !frozen.is_subsumed(&node).unwrap())
            );
        }
    }
    Ok(())
}

#[test]
fn gather_lowering_retains_dtype_guard_and_ordered_inputs() -> Result<(), TypedError> {
    let mut graph = EGraph::default();
    let indexes = Ir::Input(0, "indexes", DType::Int());
    let data = Ir::Input(1, "data", DType::Bf16());
    let shape = EList::from([DimExpr::MNum(2)]);
    let strides = EList::from([DimExpr::MIter()]);
    let data_shape = EList::from([DimExpr::MNum(8)]);
    let inputs = IList::from([indexes, data.clone()]);
    let original = Ir::Op(
        OpKind::Gather(&shape, &strides, &data_shape, &strides),
        &inputs,
    );
    let lowered = Ir::Op(
        kernel_gather(
            &shape,
            &strides,
            &data_shape,
            &strides,
            EList::RowMajor(&shape),
            DType::Bf16(),
        ),
        &inputs,
    );
    graph.register(&original)?;
    let rules = &crate::kernel::hlir::kernel_gather::kernel_lower;
    graph.run(rules)?;
    assert!(!graph.check(eq(&original, &lowered))?);
    graph.register(set(data.dtype(), DType::Bf16()))?;
    graph.run(rules)?;
    assert!(graph.check(eq(original, lowered))?);
    Ok(())
}
