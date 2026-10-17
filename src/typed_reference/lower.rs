//! Convert host HLIR metadata and edges into a shared typed expression DAG.
//! This is serialization only: no graph matching, fusion, or op selection.

use super::schema::*;
use crate::{
    graph::Graph,
    shape::{Expression, Term},
};
use egglog_experimental::typed::{TypedError, let_};
use petgraph::{Direction, graph::NodeIndex};
use rustc_hash::FxHashMap;

pub(crate) fn dimension(value: &Expression, iteration_variable: bool) -> DimExpr {
    let mut stack = Vec::new();
    for term in value.terms.read().iter() {
        let next = match *term {
            Term::Num(n) => n.into(),
            Term::Var('z') if iteration_variable => DimExpr::MIter(),
            Term::Var(c) => c.into(),
            op => {
                // Preserve the host expression's reverse-Polish operand order.
                let a = stack.pop().expect("valid host dimension expression");
                let b = stack.pop().expect("valid host dimension expression");
                match op {
                    Term::Add => a + b,
                    Term::Sub => a - b,
                    Term::Mul => a * b,
                    Term::Div => a / b,
                    Term::CeilDiv => DimExpr::MCeilDiv(a, b),
                    Term::Mod => a % b,
                    Term::Min => DimExpr::MMin(a, b),
                    Term::Max => DimExpr::MMax(a, b),
                    Term::And | Term::Or => panic!("egglog does not implement {op:?}"),
                    Term::Gte => DimExpr::MGte(a, b),
                    Term::Lt => DimExpr::MLt(a, b),
                    Term::Num(_) | Term::Var(_) => unreachable!(),
                }
            }
        };
        stack.push(next);
    }
    assert_eq!(stack.len(), 1, "valid host dimension expression");
    stack.pop().unwrap()
}

pub(crate) fn dimensions(values: &[Expression]) -> EList {
    values.iter().map(|value| dimension(value, true)).collect()
}

/// Lower each host DAG node once into ordinary shared Rust expressions, then
/// capture only sinks in ascending NodeIndex order, without OutputJoin.
pub fn capture_graph(graph: &Graph) -> Result<Vec<Ir>, TypedError> {
    use std::{cmp::Reverse, collections::BinaryHeap};
    let mut indegrees: FxHashMap<_, _> = graph
        .node_indices()
        .map(|node| {
            (
                node,
                graph.neighbors_directed(node, Direction::Incoming).count(),
            )
        })
        .collect();
    let mut ready = BinaryHeap::new();
    for (&node, &degree) in &indegrees {
        if degree == 0 {
            ready.push((Reverse(node.index()), node));
        }
    }
    let mut values: FxHashMap<NodeIndex, Ir> = FxHashMap::default();
    while let Some((_, node)) = ready.pop() {
        let inputs: Vec<_> = graph
            .get_sources(node)
            .iter()
            .map(|source| &values[source])
            .collect();
        let expression = graph[node].to_ir(&inputs);
        values.insert(node, expression);
        for target in graph.neighbors_directed(node, Direction::Outgoing) {
            let degree = indegrees.get_mut(&target).unwrap();
            *degree -= 1;
            if *degree == 0 {
                ready.push((Reverse(target.index()), target));
            }
        }
    }
    let mut roots: Vec<_> = graph.externals(Direction::Outgoing).collect();
    roots.sort_unstable_by_key(|node| node.index());
    Ok(roots
        .into_iter()
        .enumerate()
        .map(|(index, node)| let_(format!("luminal::output/{index}"), values[&node].clone()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        egglog_utils::DecodeContext,
        hlir,
        op::{DecodeFn, HLIROp},
        shape::ShapeTracker,
    };
    use egglog_experimental::typed::{EGraph, builtins as egg, eq, get_args};

    #[test]
    fn host_decoders_preserve_metadata_rank_acceptance() -> Result<(), TypedError> {
        let cases: Vec<(DecodeFn, OpKind, usize)> = vec![
            (hlir::Log2::decode, OpKind::Log2([2, 3], [1], [3, 1]), 1),
            (hlir::Exp2::decode, OpKind::Exp2([2, 3], [3, 1], [1]), 1),
            (hlir::Sin::decode, OpKind::Sin([2, 3], [1], [3, 1]), 1),
            (hlir::Recip::decode, OpKind::Recip([2, 3], [1], [3, 1]), 1),
            (hlir::Sqrt::decode, OpKind::Sqrt([2, 3], [1], [3, 1]), 1),
            (
                hlir::Add::decode,
                OpKind::Add([2, 3], [1], [3, 1], [3, 1]),
                2,
            ),
            (
                hlir::Mul::decode,
                OpKind::Mul([2, 3], [3, 1], [1], [3, 1]),
                2,
            ),
            (
                hlir::Mod::decode,
                OpKind::Mod([2, 3], [3, 1], [3, 1], [1]),
                2,
            ),
            (
                hlir::LessThan::decode,
                OpKind::LessThan([2, 3], [1], [3, 1], [3, 1]),
                2,
            ),
            (
                hlir::SumReduce::decode,
                OpKind::Sum([3], 2, [3, 1], 1, [1]),
                1,
            ),
            (
                hlir::MaxReduce::decode,
                OpKind::Max([3], 2, [1], 1, [3, 1]),
                1,
            ),
            (
                hlir::Gather::decode,
                OpKind::Gather([2], [2, 1], [3, 4], [4, 1]),
                2,
            ),
            (
                hlir::Gather::decode,
                OpKind::Gather([2], [1], [3, 4], [1]),
                2,
            ),
            (
                hlir::Scatter::decode,
                OpKind::Scatter([2, 3], [1], [4], [1], [1]),
                3,
            ),
            (
                hlir::Scatter::decode,
                OpKind::Scatter([2, 3], [3, 1], [4], [4, 1], [1]),
                3,
            ),
            (
                hlir::Scatter::decode,
                OpKind::Scatter([2, 3], [3, 1], [4], [1], [4, 1]),
                3,
            ),
            // Valid independent ranks and scalar reductions share the same
            // acceptance contract as the unchecked metadata cases above.
            (
                hlir::Gather::decode,
                OpKind::Gather([2], [1], [3, 4], [4, 1]),
                2,
            ),
            (
                hlir::Scatter::decode,
                OpKind::Scatter([2, 3], [3, 1], [4], [1], [1]),
                3,
            ),
            (
                hlir::SumReduce::decode,
                OpKind::Sum(EList::ENil(), 3, EList::ENil(), 1, EList::ENil()),
                1,
            ),
            (
                hlir::MaxReduce::decode,
                OpKind::Max(EList::ENil(), 3, EList::ENil(), 1, EList::ENil()),
                1,
            ),
        ];
        for (decoder, kind, arity) in cases {
            let kind = let_("kind", kind);
            let input = let_("input", Ir::Input(0, "input", DType::F32()));
            let mut live = EGraph::default();
            live.register((&kind, &input))?;
            let frozen = live.freeze()?;
            let observed = frozen.lookup(&kind)?;
            let class = frozen.class_of(&observed)?;
            let selected = frozen.typed_node(class.nodes().next().unwrap())?;
            let choices = FxHashMap::default();
            let mut context = DecodeContext::new(&frozen, &choices);
            let inputs = vec![frozen.lookup(&input)?; arity];
            assert!(
                decoder(&selected, &mut inputs.to_vec(), &mut context)?.is_some(),
                "decoder {decoder:?} rejected metadata accepted by the old extractor",
            );
        }
        Ok(())
    }

    #[test]
    fn host_dimension_conversion_retains_iteration_context_and_order() {
        let iteration = Expression::from('z');
        assert_eq!(dimension(&iteration, false), DimExpr::from('z'));
        assert_eq!(dimension(&iteration, true), DimExpr::MIter());
        let x = Expression::from('x');
        let y = Expression::from('y');
        assert_eq!(
            dimension(&(x - y), false),
            DimExpr::from('x') - DimExpr::from('y')
        );
        assert_eq!(
            dimension(&(x / y), false),
            DimExpr::from('x') / DimExpr::from('y')
        );
        assert_eq!(
            dimension(&(x % y), false),
            DimExpr::from('x') % DimExpr::from('y')
        );
        let values = [Expression::from(2), iteration, Expression::from('x')];
        assert_eq!(
            dimensions(&values),
            EList::from([DimExpr::from(2), DimExpr::MIter(), DimExpr::from('x')])
        );
    }

    #[test]
    fn host_constants_keep_legacy_six_decimal_rounding() {
        for (host, expected) in [
            (0.12345678f32, 0.123457f64),
            (-0.12345678, -0.123457),
            (0.000000123, 0.0),
        ] {
            let value = let_("constant", hlir::Constant(host).to_ir(&[]));
            let mut egraph = EGraph::default();
            egraph.register(&value).unwrap();
            assert!(
                egraph
                    .check(eq(value, Ir::Op(OpKind::Constant(expected), IList::INil())))
                    .unwrap()
            );
        }
    }

    #[test]
    fn numeric_constant_conversion_matches_the_legacy_bit_pattern() {
        let mut bits = vec![
            0,
            1,
            0x007f_ffff,
            0x0080_0000,
            0x7f7f_ffff,
            0x8000_0000,
            0x8000_0001,
            0x807f_ffff,
            0x8080_0000,
            0xff7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
            0x7f80_0001,
            0xffc0_1234,
        ];
        for value in [1.0f32 / 128.0, 3.0 / 128.0, -1.0 / 128.0, -3.0 / 128.0] {
            bits.extend([value.to_bits() - 1, value.to_bits(), value.to_bits() + 1]);
        }
        for sign in [0, 1 << 31] {
            for exponent in 0..=255 {
                for mantissa in [0, 1, 2, 0x003f_ffff, 0x0040_0000, 0x007f_fffe, 0x007f_ffff] {
                    bits.push(sign | exponent << 23 | mantissa);
                }
            }
        }
        let mut state = 0x513c_2f19u32;
        for _ in 0..65_536 {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            bits.push(state);
        }
        for bits in bits {
            let host = f32::from_bits(bits);
            let expected = format!("{host:.6}").parse::<f64>().unwrap();
            let authored = hlir::Constant(host).to_ir(&[]);
            let (kind, _) = get_args(&authored, |kind: &OpKind, inputs: &IList| {
                Ir::Op(kind, inputs)
            })
            .unwrap()
            .unwrap();
            let (value,) = get_args(&kind, |value: &egg::F64| OpKind::Constant(value))
                .unwrap()
                .unwrap();
            assert_eq!(
                f64::try_from(&value).unwrap().to_bits(),
                expected.to_bits(),
                "f32 bits: {bits:#010x}"
            );
        }
    }

    #[test]
    fn host_authoring_preserves_input_and_metadata_field_order() {
        let a = Ir::Input(1, "left", crate::dtype::DType::F32);
        let b = Ir::Input(2, "right", crate::dtype::DType::F32);
        let mut left = ShapeTracker::new([2, 3]);
        left.strides[0] = 11.into();
        left.strides[1] = 7.into();
        let mut right = ShapeTracker::new([2, 3]);
        right.strides[0] = 13.into();
        right.strides[1] = 5.into();
        let binary = hlir::Mod {
            input_shapes: vec![left, right],
            ..Default::default()
        };
        assert_eq!(
            binary.to_ir(&[&a, &b]),
            Ir::Op(
                OpKind::Mod(
                    [2, 3],
                    [11, 7],
                    [13, 5],
                    [DimExpr::MIter() * 3, DimExpr::MIter()],
                ),
                [&a, &b],
            )
        );
        let gather = hlir::Gather {
            input_shapes: vec![left, right],
            ..Default::default()
        };
        assert_eq!(
            gather.to_ir(&[&b, &a]),
            Ir::Op(OpKind::Gather([2, 3], [11, 7], [2, 3], [13, 5]), [&b, &a])
        );
        let reduction = hlir::SumReduce {
            input_shape: left,
            dim: 0,
            ..Default::default()
        };
        assert_eq!(
            reduction.to_ir(&[&a]),
            Ir::Op(OpKind::Sum([3], 2, [7], 11, [DimExpr::MIter()]), [&a])
        );
        let scatter = hlir::Scatter {
            dest_shape: vec![2.into(), 3.into()],
            dest_strides: vec![11.into(), 7.into()],
            index_shape: vec![5.into()],
            index_strides: vec![13.into()],
            src_strides: vec![17.into()],
        };
        assert_eq!(
            scatter.to_ir(&[&a, &b, &a]),
            Ir::Op(
                OpKind::Scatter([2, 3], [11, 7], [5], [13], [17]),
                [&a, &b, &a]
            )
        );
    }
}
