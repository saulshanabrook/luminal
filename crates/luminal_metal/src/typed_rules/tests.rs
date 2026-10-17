use super::*;
use luminal::op::DecodeFn;

#[test]
fn unary_and_binary_lowering_preserve_dtype_guards_and_input_order() -> Result<(), TypedError> {
    let shape = EList::from([DimExpr::MNum(3)]);
    let strides = EList::from([DimExpr::MIter()]);
    let out = EList::from([DimExpr::MNum(7)]);
    let a = Ir::Input(0, "a", DType::F16());
    let b = Ir::Input(1, "b", DType::F16());
    let unary = let_("unary", Ir::Op(OpKind::Sin(&shape, &strides, &out), [&a]));
    let binary = let_(
        "binary",
        Ir::Op(OpKind::Mul(&shape, &strides, &out, &strides), [&a, &b]),
    );
    let lowered_unary = Ir::Op(metal_sin(&shape, &strides, &out), [&a]);
    let lowered_binary = Ir::Op(metal_mul(&shape, &strides, &out, &strides), [&a, &b]);
    let rules = ruleset((
        &crate::kernel::metal_sin::kernel_lower,
        &crate::kernel::metal_mul::kernel_lower,
    ));
    let mut graph = EGraph::default();
    graph.register((&unary, &binary))?;
    graph.run(&rules)?;
    assert!(!graph.check(eq(&unary, &lowered_unary))?);
    assert!(!graph.check(eq(&binary, &lowered_binary))?);

    graph.register(set(a.dtype(), DType::F16()))?;
    graph.run(&rules)?;
    assert!(graph.check((
        eq(&unary, &lowered_unary),
        eq(&binary, &lowered_binary),
        eq(lowered_unary.dtype(), DType::F16()),
        eq(lowered_binary.dtype(), DType::F16())
    ))?);
    assert!(!graph.check(eq(
        &binary,
        Ir::Op(metal_mul(&shape, &strides, &out, &strides), [&b, &a])
    ))?);
    let frozen = graph.freeze()?;
    for source in [&unary, &binary] {
        let value = frozen.lookup(source)?;
        assert!(
            frozen
                .class_of(&value)?
                .nodes()
                .any(|node| node.is_subsumed())
        );
    }
    Ok(())
}

#[test]
fn constant_and_gather_preserve_inputs_dtype_and_subsumption() -> Result<(), TypedError> {
    let mut graph = EGraph::default();
    let constant = let_("constant", Ir::Op(OpKind::Constant(2.5), IList::INil()));
    graph.register(&constant)?;
    graph.run(&crate::kernel::metal_constant::kernel_lower)?;
    let lowered = Ir::Op(metal_constant(2.5), IList::INil());
    assert!(graph.check((eq(&constant, &lowered), eq(lowered.dtype(), DType::F32())))?);
    let frozen = graph.freeze()?;
    let value = frozen.lookup(&constant)?;
    assert!(frozen.class_of(&value)?.nodes().any(|node| {
        let typed = frozen.typed_node::<Ir>(node).unwrap();
        let (kind, _) = get_args(&typed, |kind: &OpKind, inputs: &IList| Ir::Op(kind, inputs))
            .unwrap()
            .unwrap();
        frozen.nodes(&kind).unwrap().into_iter().any(|kind| {
            get_args(&kind, |value: &egg::F64| OpKind::Constant(value))
                .unwrap()
                .is_some()
        }) && node.is_subsumed()
    }));

    let indexes = Ir::Input(0, "indexes", DType::Int());
    let data = Ir::Input(1, "data", DType::F16());
    let shape = EList::from([DimExpr::MNum(3)]);
    let stride = EList::from([DimExpr::MIter()]);
    let data_shape = EList::from([DimExpr::MNum(8)]);
    let gather = Ir::Op(
        OpKind::Gather(&shape, &stride, &data_shape, &stride),
        IList::from([indexes.clone(), data.clone()]),
    );
    graph.register((&gather, set(data.dtype(), DType::F16())))?;
    graph.run(&crate::kernel::metal_gather::kernel_lower)?;
    let lowered = Ir::Op(
        metal_gather(
            &shape,
            &stride,
            &data_shape,
            &stride,
            EList::RowMajor(&shape),
        ),
        IList::from([indexes, data]),
    );
    assert!(graph.check((eq(gather, &lowered), eq(lowered.dtype(), DType::F16())))?);
    Ok(())
}

#[test]
fn matmul_preserves_operand_order_layout_and_dtype() -> Result<(), TypedError> {
    let mut graph = EGraph::default();
    let lhs = Ir::Input(0, "lhs", DType::F32());
    let rhs = Ir::Input(1, "rhs", DType::F32());
    let m = DimExpr::MNum(2);
    let n = DimExpr::MNum(3);
    let k = DimExpr::MNum(4);
    let iter = DimExpr::MIter();
    let out_shape = EList::from([m.clone(), n.clone()]);
    let out_strides = EList::from([&n * &iter, iter.clone()]);
    let mul = Ir::Op(
        metal_mul(
            EList::from([m.clone(), n.clone(), k.clone()]),
            EList::from([&k * &iter, DimExpr::MNum(0), iter.clone()]),
            EList::from([DimExpr::MNum(0), iter.clone(), &n * &iter]),
            EList::ENil(),
        ),
        IList::from([lhs.clone(), rhs.clone()]),
    );
    let sum = Ir::Op(
        metal_sum(&out_shape, &k, EList::ENil(), &iter, &out_strides),
        IList::from([mul]),
    );
    graph.register((&sum, set(sum.dtype(), DType::F32())))?;
    let rules = &crate::kernel::m_p_s_matmul::kernel_lower;
    graph.run(rules)?;
    let lowered = Ir::Op(
        mps_matmul(&m, &n, &k, &k * &iter, &n * &iter, &n * &iter, 0, 0),
        IList::from([lhs, rhs]),
    );
    assert!(graph.check((eq(sum, &lowered), eq(lowered.dtype(), DType::F32())))?);
    Ok(())
}

#[test]
fn unary_decoder_keeps_layout_validation_at_compile_time() -> Result<(), TypedError> {
    let mut graph = EGraph::default();
    let kind = let_(
        "kind",
        metal_exp2(
            EList::from([DimExpr::MNum(3)]),
            EList::ENil(),
            EList::ENil(),
        ),
    );
    graph.register(&kind)?;
    let frozen = graph.freeze()?;
    let observed = frozen.lookup(&kind)?;
    let selected = frozen.nodes(&observed)?.next().unwrap();
    let choices = Default::default();
    let mut context = luminal::egglog_utils::DecodeContext::new(&frozen, &choices);
    let decoded = crate::kernel::prepare_exp2(
        &selected,
        &mut vec![Ir::Input(0, "input", DType::F32())],
        &mut context,
    )?
    .unwrap();
    let kernel = decoded
        .to_dialect::<dyn crate::kernel::MetalKernelOp>()
        .unwrap();
    assert_eq!(kernel.output_size().to_usize(), Some(3));
    let debug = format!("{kernel:?}");
    assert!(debug.contains("input_strides: []"), "{debug}");
    assert!(debug.contains("output_strides: []"), "{debug}");
    Ok(())
}

#[test]
fn unary_args_follow_the_selected_producer_and_frozen_metadata() -> Result<(), TypedError> {
    let shape = EList::from([DimExpr::MVar("s")]);
    let strides = EList::from([DimExpr::MIter()]);
    let kind = metal_sin(&shape, &strides, &strides);
    let selector =
        |shape: &EList, strides: &EList, output: &EList| metal_sin(shape, strides, output);
    assert_eq!(
        get_args(&kind, selector)?.unwrap(),
        (shape.clone(), strides.clone(), strides.clone())
    );

    let kind = let_("kind", kind);
    let shape = let_("shape", shape);
    let mut graph = EGraph::default();
    graph.register((
        &shape,
        &kind,
        union(&kind, OpKind::Sin(&shape, &strides, &strides)),
        union(DimExpr::MVar("s"), DimExpr::MNum(4)),
    ))?;
    let frozen = graph.freeze()?;
    let observed = frozen.lookup(&kind)?;
    let class = frozen.class_of(&observed)?;
    assert_eq!(class.nodes().len(), 2);
    for node in class.nodes() {
        let choices = [(class, node)].into_iter().collect();
        let mut context = luminal::egglog_utils::DecodeContext::new(&frozen, &choices);
        let selected = context.selected(&observed)?;
        let original_inputs = vec![Ir::Input(0, "input", DType::F32())];
        let mut inputs = original_inputs.clone();
        let decoded = crate::kernel::prepare_sin(&selected, &mut inputs, &mut context)?;
        assert_eq!(inputs, original_inputs);
        if let Some((shape_arg, input_arg, output_arg)) = get_args(&selected, selector)? {
            assert_eq!(
                frozen.class_of(&shape_arg)?,
                frozen.class_of(&frozen.lookup(&shape)?)?
            );
            assert_eq!(input_arg, output_arg);
            // The preserved tie policy need not select the literal producer.
            let dims = [('s', 4)].into_iter().collect();
            assert_eq!(
                context
                    .expressions(&shape_arg)?
                    .iter()
                    .map(|e| e.exec(&dims))
                    .collect::<Vec<_>>(),
                vec![Some(4)]
            );
            let prepared = decoded.expect("selected MetalSin must decode");
            let prepared = prepared
                .to_dialect::<dyn crate::kernel::MetalKernelOp>()
                .unwrap();
            assert_eq!(prepared.output_size().exec(&dims), Some(4));
        } else {
            assert!(get_args(&selected, selector)?.is_none());
            assert!(
                decoded.is_none(),
                "decoder must not choose a different producer"
            );
        }
    }
    Ok(())
}

#[test]
fn unary_decoder_preserves_inputs_and_exact_dimension_decoding() -> Result<(), TypedError> {
    let mut graph = EGraph::default();
    let strides = EList::from([DimExpr::MIter()]);
    let integral = let_(
        "integral",
        metal_sin(EList::from([DimExpr::MNum(3)]), &strides, &strides),
    );
    let fractional = let_(
        "fractional",
        metal_sin(EList::from([DimExpr::MFloat(3.5)]), &strides, &strides),
    );
    graph.register((&integral, &fractional))?;
    let frozen = graph.freeze()?;
    let choices = Default::default();
    let mut context = luminal::egglog_utils::DecodeContext::new(&frozen, &choices);
    let input = Ir::Input(0, "input", DType::F32());
    let selected = frozen.nodes(&frozen.lookup(&integral)?)?.next().unwrap();
    for mut inputs in [vec![], vec![input.clone(), input.clone()]] {
        assert!(crate::kernel::prepare_sin(&selected, &mut inputs, &mut context)?.is_some());
    }
    let selected = frozen.nodes(&frozen.lookup(&fractional)?)?.next().unwrap();
    assert!(matches!(
        crate::kernel::prepare_sin(&selected, &mut vec![input], &mut context),
        Err(TypedError::Decode(_))
    ));
    Ok(())
}

#[test]
fn unary_decoder_owns_shape_and_preserves_dynamic_metrics() -> Result<(), TypedError> {
    for dimensions in [vec![], vec![DimExpr::MNum(0)], vec![DimExpr::MVar("s")]] {
        let dynamic = dimensions == vec![DimExpr::MVar("s")];
        let prepared = {
            let mut graph = EGraph::default();
            let strides = EList::from(vec![DimExpr::MIter(); dimensions.len()]);
            let kind = let_(
                "kind",
                metal_sin(EList::from(dimensions), &strides, &strides),
            );
            graph.register(&kind)?;
            let frozen = graph.freeze()?;
            let selected = frozen.nodes(&frozen.lookup(&kind)?)?.next().unwrap();
            let choices = Default::default();
            let mut context = luminal::egglog_utils::DecodeContext::new(&frozen, &choices);
            crate::kernel::prepare_sin(
                &selected,
                &mut vec![Ir::Input(0, "input", DType::F32())],
                &mut context,
            )?
            .unwrap()
        };
        // Neither frozen observations nor the decode context survive preparation.
        let prepared = prepared
            .to_dialect::<dyn crate::kernel::MetalKernelOp>()
            .unwrap();
        for size in [0, 1, 7] {
            let dyn_map = [('s', size)].into_iter().collect();
            let expected = if dynamic { size.max(1) } else { 1 };
            assert_eq!(prepared.output_size().exec(&dyn_map), Some(expected));
            assert_eq!(prepared.bytes_loaded(&dyn_map), expected * 4);
            assert_eq!(prepared.bytes_stored(&dyn_map), expected * 4);
            assert_eq!(prepared.flops(&dyn_map), expected);
        }
    }
    Ok(())
}

#[test]
fn unary_preparation_keeps_stable_operation_identity() -> Result<(), TypedError> {
    let shape = EList::from([DimExpr::MNum(4)]);
    let strides = EList::from([DimExpr::MIter()]);
    for (kind, decoder, name) in [
        (
            metal_sin(&shape, &strides, &strides),
            crate::kernel::prepare_sin as DecodeFn,
            "MetalSin",
        ),
        (
            metal_exp2(&shape, &strides, &strides),
            crate::kernel::prepare_exp2 as DecodeFn,
            "MetalExp2",
        ),
    ] {
        let prepared = {
            let mut graph = EGraph::default();
            let kind = let_("kind", kind);
            graph.register(&kind)?;
            let frozen = graph.freeze()?;
            let selected = frozen.nodes(&frozen.lookup(&kind)?)?.next().unwrap();
            let choices = Default::default();
            let mut context = luminal::egglog_utils::DecodeContext::new(&frozen, &choices);
            decoder(
                &selected,
                &mut vec![Ir::Input(0, "input", DType::F32())],
                &mut context,
            )?
            .unwrap()
        };
        let display = prepared.to_string();
        assert!(display.contains(name), "{display}");
        assert!(!display.contains("scalar_expr"), "{display}");
        assert!(!display.contains("0x"), "{display}");
    }
    Ok(())
}

#[test]
fn binary_decoder_owns_shape_and_preserves_dynamic_metrics() -> Result<(), TypedError> {
    use crate::kernel::{MetalAdd, MetalLessThan, MetalMod, MetalMul};
    for dimensions in [vec![], vec![DimExpr::MVar("s")]] {
        let scalar = dimensions.is_empty();
        let strides = EList::from(vec![DimExpr::MIter(); dimensions.len()]);
        let shape = EList::from(dimensions);
        for (kind, decoder, flops_per_element) in [
            (
                metal_add(&shape, &strides, &strides, &strides),
                MetalAdd::decode as DecodeFn,
                1,
            ),
            (
                metal_mul(&shape, &strides, &strides, &strides),
                MetalMul::decode as DecodeFn,
                1,
            ),
            (
                metal_mod(&shape, &strides, &strides, &strides),
                MetalMod::decode as DecodeFn,
                10,
            ),
            (
                metal_less_than(&shape, &strides, &strides, &strides),
                MetalLessThan::decode as DecodeFn,
                1,
            ),
        ] {
            let prepared = {
                let kind = let_("kind", kind);
                let mut live = EGraph::default();
                live.register(&kind)?;
                let frozen = live.freeze()?;
                let selected = frozen.nodes(&frozen.lookup(&kind)?)?.next().unwrap();
                let choices = Default::default();
                let mut context = luminal::egglog_utils::DecodeContext::new(&frozen, &choices);
                decoder(
                    &selected,
                    &mut vec![
                        Ir::Input(0, "a", DType::F32()),
                        Ir::Input(1, "b", DType::F32()),
                    ],
                    &mut context,
                )?
                .unwrap()
            };
            let kernel = prepared
                .to_dialect::<dyn crate::kernel::MetalKernelOp>()
                .unwrap();
            if scalar {
                let debug = format!("{kernel:?}");
                for field in [
                    "shape: []",
                    "a_strides: []",
                    "b_strides: []",
                    "output_strides: []",
                ] {
                    assert!(debug.contains(field), "{debug}");
                }
            }
            for size in [0, 1, 7] {
                let dyn_map = [('s', size)].into_iter().collect();
                let expected = if scalar { 1 } else { size.max(1) };
                assert_eq!(kernel.output_size().exec(&dyn_map), Some(expected));
                assert_eq!(kernel.bytes_loaded(&dyn_map), expected * 8);
                assert_eq!(kernel.bytes_stored(&dyn_map), expected * 4);
                assert_eq!(kernel.flops(&dyn_map), expected * flops_per_element);
            }
        }
    }
    Ok(())
}

#[test]
fn native_kernel_defaults_and_introspection_remain_available() {
    use crate::kernel::*;
    let kernels: Vec<Box<dyn MetalKernelOp>> = vec![
        Box::<MetalExp2>::default(),
        Box::<MetalLog2>::default(),
        Box::<MetalSin>::default(),
        Box::<MetalSqrt>::default(),
        Box::<MetalRecip>::default(),
        Box::<MetalAdd>::default(),
        Box::<MetalMul>::default(),
        Box::<MetalMod>::default(),
        Box::<MetalLessThan>::default(),
    ];
    for kernel in kernels {
        assert_eq!(kernel.output_size().to_usize(), Some(1));
        assert!(format!("{kernel:?}").contains("shape: []"));
    }
    let mul = MetalMul::default().mul_info().unwrap();
    assert!(mul.shape.is_empty());
    assert!(mul.a_strides.is_empty());
    assert!(mul.b_strides.is_empty());
    assert!(mul.output_strides.is_empty());
    let sum = MetalSumReduce::default().sum_reduce_info().unwrap();
    assert!(sum.shape.is_empty());
    assert!(sum.strides.is_empty());
}
