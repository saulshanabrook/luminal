//! The original analysis-only late pass, expressed as typed relations and rules.
//! Byte rules cover the original OpKind operations, including dormant backend kinds.
use crate::typed_rules::*;
use egglog_experimental::typed::prelude::*;
use luminal::typed_reference::schema::*;

#[relation]
fn metal_output_bytes(kind: OpKind, bytes: DimExpr);

#[relation]
fn metal_local_memory(node: Ir, bytes: DimExpr);

pub(crate) fn rules() -> Ruleset {
    ruleset(
        |node: &Ir, kind: &OpKind, shape: &EList, elements: &DimExpr| {
            let mut rules = vec![
                rule(
                    eq(node, Ir::from(InputArgs::fresh())),
                    metal_local_memory(node, 0),
                )
                .label("metal-memory-input"),
                rule(
                    eq(node, Ir::from(OutputArgs::fresh())),
                    metal_local_memory(node, 0),
                )
                .label("metal-memory-output"),
                // OutputJoin was replaced by the requested ordered root forest.
                rule(
                    (
                        eq(
                            node,
                            Ir::from(OpArgs {
                                kind: kind.clone(),
                                ..OpArgs::fresh()
                            }),
                        ),
                        metal_output_bytes(kind, elements),
                    ),
                    metal_local_memory(node, elements),
                )
                .label("metal-memory-op-local"),
            ];

            // Keep the original name-sorted order. Loop boundaries have no shape premise.
            for (label, pattern, shape_bytes) in [
                (
                    "metal-memory-Add-f32-shape",
                    OpKind::from(AddArgs {
                        shape: shape.clone(),
                        ..AddArgs::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-Exp2-f32-shape",
                    OpKind::from(Exp2Args {
                        shape: shape.clone(),
                        ..Exp2Args::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-LessThan-f32-shape",
                    OpKind::from(LessThanArgs {
                        shape: shape.clone(),
                        ..LessThanArgs::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-Log2-f32-shape",
                    OpKind::from(Log2Args {
                        shape: shape.clone(),
                        ..Log2Args::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-LoopInput-zero",
                    OpKind::from(LoopInputArgs::fresh()),
                    None,
                ),
                (
                    "metal-memory-LoopInputStatic-zero",
                    OpKind::from(LoopInputStaticArgs::fresh()),
                    None,
                ),
                (
                    "metal-memory-LoopOutput-zero",
                    OpKind::from(LoopOutputArgs::fresh()),
                    None,
                ),
                (
                    "metal-memory-LoopOutputSelect-zero",
                    OpKind::from(LoopOutputSelectArgs::fresh()),
                    None,
                ),
                (
                    "metal-memory-Max-f32-shape",
                    OpKind::from(MaxArgs {
                        shape: shape.clone(),
                        ..MaxArgs::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-MetalAdd-f32-shape",
                    OpKind::from(MetalAddArgs {
                        shape: shape.clone(),
                        ..MetalAddArgs::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-MetalExp2-f32-shape",
                    OpKind::from(MetalExp2Args {
                        shape: shape.clone(),
                        ..MetalExp2Args::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-MetalLessThan-bool-shape",
                    OpKind::from(MetalLessThanArgs {
                        shape: shape.clone(),
                        ..MetalLessThanArgs::fresh()
                    }),
                    Some(elements.clone()),
                ),
                (
                    "metal-memory-MetalLog2-f32-shape",
                    OpKind::from(MetalLog2Args {
                        shape: shape.clone(),
                        ..MetalLog2Args::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-MetalMax-f32-shape",
                    OpKind::from(MetalMaxArgs {
                        shape: shape.clone(),
                        ..MetalMaxArgs::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-MetalMod-f32-shape",
                    OpKind::from(MetalModArgs {
                        shape: shape.clone(),
                        ..MetalModArgs::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-MetalMul-f32-shape",
                    OpKind::from(MetalMulArgs {
                        shape: shape.clone(),
                        ..MetalMulArgs::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-MetalRecip-f32-shape",
                    OpKind::from(MetalRecipArgs {
                        shape: shape.clone(),
                        ..MetalRecipArgs::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-MetalScatterNoCopy-f32-shape",
                    OpKind::from(MetalScatterNoCopyArgs {
                        dest_shape: shape.clone(),
                        ..MetalScatterNoCopyArgs::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-MetalSin-f32-shape",
                    OpKind::from(MetalSinArgs {
                        shape: shape.clone(),
                        ..MetalSinArgs::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-MetalSqrt-f32-shape",
                    OpKind::from(MetalSqrtArgs {
                        shape: shape.clone(),
                        ..MetalSqrtArgs::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-MetalSum-f32-shape",
                    OpKind::from(MetalSumArgs {
                        shape: shape.clone(),
                        ..MetalSumArgs::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-Mod-f32-shape",
                    OpKind::from(ModArgs {
                        shape: shape.clone(),
                        ..ModArgs::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-Mul-f32-shape",
                    OpKind::from(MulArgs {
                        shape: shape.clone(),
                        ..MulArgs::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-Recip-f32-shape",
                    OpKind::from(RecipArgs {
                        shape: shape.clone(),
                        ..RecipArgs::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-Scatter-f32-shape",
                    OpKind::from(ScatterArgs {
                        dest_shape: shape.clone(),
                        ..ScatterArgs::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-Sin-f32-shape",
                    OpKind::from(SinArgs {
                        shape: shape.clone(),
                        ..SinArgs::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-Sqrt-f32-shape",
                    OpKind::from(SqrtArgs {
                        shape: shape.clone(),
                        ..SqrtArgs::fresh()
                    }),
                    Some(elements * 4),
                ),
                (
                    "metal-memory-Sum-f32-shape",
                    OpKind::from(SumArgs {
                        shape: shape.clone(),
                        ..SumArgs::fresh()
                    }),
                    Some(elements * 4),
                ),
            ] {
                let rule = if let Some(bytes) = shape_bytes {
                    rule(
                        (eq(kind, pattern), eq(elements, shape.n_elements())),
                        metal_output_bytes(kind, bytes),
                    )
                } else {
                    rule(eq(kind, pattern), metal_output_bytes(kind, 0))
                };
                rules.push(rule.label(label));
            }
            rules
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use luminal::op::Runtime;

    #[test]
    fn omitting_sin_lowering_leaves_dormant_support_rules_inert() -> Result<(), TypedError> {
        use luminal::egglog_utils::{
            RewritePhase::{KernelLower, LateAnalysis},
            main_schedule,
        };
        let mut graph = luminal::graph::Graph::default();
        graph.tensor((2, 3)).sin().output();
        let roots = luminal::typed_reference::capture_graph(&graph)?;
        for enabled in [false, true] {
            let (mut phases, actions) = crate::MetalRuntime::rules();
            // Sin is the only operation in this fixture. Leave all analysis
            // installed while removing the phase that can produce MetalSin.
            if !enabled {
                phases.insert(KernelLower, ruleset(()));
            }
            let mut live = EGraph::default();
            live.register(&actions)?;
            live.register(&roots)?;
            live.run(main_schedule(&phases, false))?;
            // Inspect analysis before final cleanup deletes n_elements metadata.
            live.run(phases[&LateAnalysis].saturate())?;
            let frozen = live.freeze()?;
            assert_eq!(
                !frozen
                    .table(|shape: &EList, strides: &EList, out: &EList| {
                        metal_sin(shape, strides, out)
                    })?
                    .rows
                    .is_empty(),
                enabled,
            );
            assert_eq!(
                frozen
                    .table(|kind: &OpKind, bytes: &DimExpr| metal_output_bytes(kind, bytes))?
                    .rows
                    .iter()
                    .any(|row| {
                        frozen
                            .nodes(&row.args.0)
                            .unwrap()
                            .any(|kind| MetalSinArgs::get_args(&kind).unwrap().is_some())
                    }),
                enabled,
            );
        }
        Ok(())
    }

    #[test]
    fn installed_late_catalog_preserves_old_opkind_filter_and_byte_facts() -> Result<(), TypedError>
    {
        let theory = rules();
        assert_eq!(theory.len(), 31); // Original 32, without the removed OutputJoin.
        let shape = EList::from([2, 3]);
        let input = Ir::Input(7, "input", DType::F32());
        let unary = metal_exp2(&shape, [3, 1], [3, 1]);
        let boolean = metal_less_than(&shape, [3, 1], [0, 1], [3, 1]);
        let core_boolean = OpKind::LessThan(&shape, [3, 1], [0, 1], [3, 1]);
        let loop_kind = OpKind::LoopInput(17, 19, DType::F32());
        let op = Ir::Op(&unary, [&input]);
        let output = Ir::Output(&op, 11);
        let mut live = EGraph::new(EGraphOptions::default());
        live.register((
            &output,
            &boolean,
            &core_boolean,
            &loop_kind,
            set(shape.n_elements(), 6),
        ))?;
        live.run(theory.saturate())?;
        assert!(live.check((
            metal_local_memory(&input, 0),
            metal_local_memory(&output, 0),
            metal_local_memory(&op, DimExpr::from(6) * 4),
            metal_output_bytes(&unary, DimExpr::from(6) * 4),
            metal_output_bytes(&boolean, 6),
            metal_output_bytes(&core_boolean, DimExpr::from(6) * 4),
            metal_output_bytes(&loop_kind, 0),
        ))?);
        // Formerly direct-IR operations remain outside this analysis and do
        // not count as new survivor evidence for conditional core cleanup.
        let nil = EList::ENil();
        for (index, kind) in [
            metal_constant(1.0),
            metal_iota(DimExpr::MIter(), 3),
            metal_cast(1, DType::F32()),
            metal_gather(&nil, &nil, &nil, &nil, &nil),
            metal_scatter(&nil, &nil, &nil, &nil, &nil, &nil),
            mps_matmul(1, 1, 1, 1, 1, 1, 0, 0),
            mps_batched_matmul(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0),
            generic_matmul(&nil, &nil, 1, &nil, &nil, &nil, 1, &nil),
        ]
        .into_iter()
        .enumerate()
        {
            assert!(crate::MetalRuntime::cleanup_ignored(&kind)?);
            let kind = let_(format!("ignored_{index}"), kind);
            live.register(&kind)?;
            live.run(theory.saturate())?;
            let frozen = live.freeze()?;
            let kind = frozen.lookup(&kind)?;
            assert!(
                frozen
                    .table(|kind: &OpKind, bytes: &DimExpr| metal_output_bytes(kind, bytes))?
                    .rows
                    .iter()
                    .all(|row| row.args.0 != kind)
            );
        }
        for kind in [unary, boolean, core_boolean, loop_kind] {
            assert!(!crate::MetalRuntime::cleanup_ignored(&kind)?);
        }
        Ok(())
    }
}
