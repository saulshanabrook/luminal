//! CPU rule generators and the fixed dtype propagation theory.

use super::schema::*;
use egglog_experimental::typed::{builtins as egg, prelude::*};

fn inherited_dtype(
    value: &Ir,
    source: &Ir,
    tail: &IList,
    dt: &DType,
    kind: OpKind,
    source_index: usize,
) -> Rule {
    let mut inputs = IList::ICons(source, tail);
    for index in 0..source_index {
        inputs = IList::ICons(
            var::<Ir>(format!("inherited_dtype::ignored_{index}")),
            inputs,
        );
    }
    rule(
        (eq(value, Ir::Op(kind, inputs)), eq(dt, source.dtype())),
        (set(value.dtype(), dt),),
    )
}

#[ruleset]
pub(crate) fn dtypes(value: &Ir, source: &Ir, tail: &IList, dt: &DType) -> Vec<Rule> {
    let mut rules = vec![
        rule(
            eq(
                value,
                InputArgs {
                    dtype: dt.clone(),
                    ..InputArgs::fresh()
                },
            ),
            set(value.dtype(), dt),
        ),
        rule(
            eq(
                value,
                LoopStartArgs {
                    dtype: dt.clone(),
                    ..LoopStartArgs::fresh()
                },
            ),
            set(value.dtype(), dt),
        ),
        rule(
            eq(
                value,
                LoopEndArgs {
                    dtype: dt.clone(),
                    ..LoopEndArgs::fresh()
                },
            ),
            set(value.dtype(), dt),
        ),
        rule(
            (
                eq(
                    value,
                    OutputArgs {
                        inp: source.clone(),
                        ..OutputArgs::fresh()
                    },
                ),
                eq(dt, source.dtype()),
            ),
            set(value.dtype(), dt),
        ),
    ];
    for kind in [
        CustomOpKindArgs {
            dtype: dt.clone(),
            ..CustomOpKindArgs::fresh()
        }
        .into(),
        LoopInputArgs {
            dtype: dt.clone(),
            ..LoopInputArgs::fresh()
        }
        .into(),
        LoopInputStaticArgs {
            dtype: dt.clone(),
            ..LoopInputStaticArgs::fresh()
        }
        .into(),
        LoopOutputArgs {
            dtype: dt.clone(),
            ..LoopOutputArgs::fresh()
        }
        .into(),
        LoopOutputSelectArgs {
            dtype: dt.clone(),
            ..LoopOutputSelectArgs::fresh()
        }
        .into(),
        CastArgs {
            dtype: dt.clone(),
            ..CastArgs::fresh()
        }
        .into(),
    ] {
        rules.push(rule(
            eq(
                value,
                OpArgs {
                    kind,
                    ..OpArgs::fresh()
                },
            ),
            set(value.dtype(), dt),
        ));
    }
    for (kind, dtype) in [
        (ConstantArgs::fresh().into(), DType::F32()),
        (IotaArgs::fresh().into(), DType::Int()),
        (LessThanArgs::fresh().into(), DType::Bool()),
    ] {
        rules.push(rule(
            eq(
                value,
                OpArgs {
                    kind,
                    ..OpArgs::fresh()
                },
            ),
            set(value.dtype(), dtype),
        ));
    }
    rules.push(inherited_dtype(
        value,
        source,
        tail,
        dt,
        GatherArgs::fresh().into(),
        1,
    ));
    rules.push(inherited_dtype(
        value,
        source,
        tail,
        dt,
        ScatterArgs::fresh().into(),
        2,
    ));
    for kind in [
        Exp2Args::fresh().into(),
        Log2Args::fresh().into(),
        SinArgs::fresh().into(),
        RecipArgs::fresh().into(),
        SqrtArgs::fresh().into(),
        AddArgs::fresh().into(),
        MulArgs::fresh().into(),
        ModArgs::fresh().into(),
        SumArgs::fresh().into(),
        MaxArgs::fresh().into(),
    ] {
        rules.push(inherited_dtype(value, source, tail, dt, kind, 0));
    }
    rules
}

/// Both state positions and every admitted trip count retain the same rules
/// as `hlir::binary_op_unroll_rules`. Shared Rust bindings replace rule-local
/// source lets; evaluation and unions still execute through egglog commands.
pub fn binary_unroll_rules(
    kind: impl Fn(&EList, &EList, &EList, &EList) -> OpKind,
    max_trips: usize,
) -> Result<Ruleset, TypedError> {
    i64::try_from(max_trips)
        .map_err(|_| TypedError::Invalid("binary unroll trip count must fit i64".into()))?;
    let mut sources = Vec::new();
    sources
        .try_reserve_exact(max_trips)
        .map_err(|error| TypedError::Invalid(error.to_string()))?;
    sources.extend((0..max_trips).map(|index| var::<Ir>(format!("s{index}"))));
    Ok(ruleset(
        |initial: &Ir,
         loop_id: &egg::I64,
         slot: &egg::I64,
         stream: &egg::I64,
         dt: &DType,
         shape: &EList,
         a_strides: &EList,
         b_strides: &EList,
         out_strides: &EList,
         ls: &Ir,
         li: &Ir,
         body: &Ir,
         end: &Ir| {
            let mut rules = Vec::new();
            for trips in 2..=max_trips {
                for state_position in 0..2 {
                    let op_kind = kind(shape, a_strides, b_strides, out_strides);
                    let label = format!(
                        "unroll {} body trips={trips} state={state_position}",
                        op_kind.call_name().expect("binary constructor")
                    );
                    let ordered_inputs = |state: &Ir, per_iter: &Ir| {
                        let (a, b) = if state_position == 0 {
                            (state, per_iter)
                        } else {
                            (per_iter, state)
                        };
                        IList::from([a, b])
                    };
                    let mut unrolled = initial.clone();
                    for source in &sources[..trips] {
                        unrolled = Ir::Op(&op_kind, ordered_inputs(&unrolled, source));
                    }
                    rules.push(
                        rule(
                            (
                                eq(ls, Ir::LoopStart(initial, loop_id, slot, trips as i64, dt)),
                                eq(
                                    li,
                                    Ir::Op(
                                        OpKind::LoopInput(loop_id, stream, dt),
                                        &sources[..trips],
                                    ),
                                ),
                                eq(body, Ir::Op(op_kind, ordered_inputs(ls, li))),
                                eq(end, Ir::LoopEnd(body, loop_id, slot, dt)),
                            ),
                            (union(end, unrolled),),
                        )
                        .label(label),
                    );
                }
            }
            rules
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_patterns_keep_omitted_fields_independent() {
        let f32 = DType::F32();
        let f64 = DType::F64();
        let int = DType::Int();
        let dest = Ir::Input(10, "destination", &f32);
        let index = Ir::Input(20, "indices", &int);
        let source = Ir::Input(30, "source", &f64);

        // The ignored identifiers differ within each constructor pattern.
        let start = Ir::LoopStart(&dest, 41, 42, 3, &f64);
        let end = Ir::LoopEnd(&start, 51, 52, &f32);
        let selected = Ir::Op(OpKind::LoopOutputSelect(61, 62, 63, &int), [&source]);

        let fields = [1, 2, 3, 4, 5].map(|n| EList::from([n]));
        let inputs = [&dest, &index, &source];
        // Scatter's two ignored prefix inputs and five ignored fields all differ.
        let scatter = Ir::Op(
            OpKind::Scatter(&fields[0], &fields[1], &fields[2], &fields[3], &fields[4]),
            inputs,
        );

        let mut egraph = EGraph::new(EGraphOptions::default());
        egraph
            .register((&start, &end, &selected, &scatter))
            .unwrap();
        egraph.run(dtypes.saturate()).unwrap();
        assert!(
            egraph
                .check((
                    eq(start.dtype(), &f64),
                    eq(end.dtype(), &f32),
                    eq(selected.dtype(), &int),
                    eq(scatter.dtype(), &f64),
                ))
                .unwrap()
        );
    }

    #[test]
    fn unroll_groups_keep_factory_calls_and_catalog_bounds() {
        use std::cell::Cell;

        for max_trips in 0usize..=6 {
            let calls = Cell::new(0);
            let group = binary_unroll_rules(
                |shape, a, b, out| {
                    calls.set(calls.get() + 1);
                    OpKind::Add(shape, a, b, out)
                },
                max_trips,
            )
            .unwrap();
            assert_eq!(group.len(), 2 * max_trips.saturating_sub(1));
            assert_eq!(calls.get(), group.len());
        }

        let calls = Cell::new(0);
        let result = binary_unroll_rules(
            |shape, a, b, out| {
                calls.set(calls.get() + 1);
                OpKind::Add(shape, a, b, out)
            },
            usize::MAX,
        );
        assert!(matches!(result, Err(TypedError::Invalid(_))));
        assert_eq!(calls.get(), 0);
    }

    #[test]
    fn generated_binary_unroll_rules_cover_both_positions_and_all_trip_counts() {
        type Kind = fn(&EList, &EList, &EList, &EList) -> OpKind;
        let kinds: [Kind; 4] = [
            |s, a, b, o| OpKind::Add(s, a, b, o),
            |s, a, b, o| OpKind::Mul(s, a, b, o),
            |s, a, b, o| OpKind::Mod(s, a, b, o),
            |s, a, b, o| OpKind::LessThan(s, a, b, o),
        ];
        for kind in kinds {
            let rules = binary_unroll_rules(kind, 6).unwrap();
            for trips in 2..=6 {
                for state_position in 0..2 {
                    let dt = DType::F32();
                    let initial = Ir::Input(0i64, "initial", &dt);
                    let sources: Vec<_> = (0..trips)
                        .map(|index| Ir::Input(100i64 + index as i64, "stream", &dt))
                        .collect();
                    let inputs = IList::from(&sources);
                    let nil = EList::ENil();
                    let kind = kind(&nil, &nil, &nil, &nil);
                    let apply = |state: Ir, source: Ir| {
                        let (a, b) = if state_position == 0 {
                            (state, source)
                        } else {
                            (source, state)
                        };
                        Ir::Op(&kind, [a, b])
                    };
                    let start = Ir::LoopStart(&initial, 7i64, 0i64, trips as i64, &dt);
                    let per_iter = Ir::Op(OpKind::LoopInput(7i64, 0i64, &dt), inputs);
                    let end = let_("end", Ir::LoopEnd(apply(start, per_iter), 7i64, 0i64, dt));
                    let unrolled = sources.into_iter().fold(initial, apply);
                    let mut egraph = EGraph::new(EGraphOptions::default());
                    egraph.register(&end).unwrap();
                    egraph.run(&rules).unwrap();
                    assert!(
                        egraph.check(eq(end, unrolled)).unwrap(),
                        "trips={trips}, state_position={state_position}"
                    );
                }
            }
        }
    }
}
