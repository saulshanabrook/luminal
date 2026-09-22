//! Shared typed declarations, host lowering, and dimension reasoning.
pub(crate) mod dynamic_rules;
pub(crate) mod lower;
pub mod schema;
pub(crate) mod static_rules;
#[cfg(test)]
mod tests;

use crate::{
    egglog_utils::{DecodeContext, base_phase_rules, run_schedule},
    shape::{DimInterval, DynDimIntervals, Expression},
};
use egglog_experimental::typed::prelude::*;
pub use lower::capture_graph;
pub use schema::Ir;
use schema::{DimExpr, OpKind};

/// One reconstructed forest; duplicate roots reference the same producer.
pub struct ReferenceForest {
    pub graph: crate::graph::LLIRGraph,
    pub outputs: Vec<petgraph::graph::NodeIndex>,
}

impl crate::egglog_utils::SearchSpace {
    pub fn reconstruct_reference(
        &self,
        rng: &mut impl rand::Rng,
        custom_ops: &[Box<dyn crate::op::CustomOp>],
    ) -> Result<ReferenceForest, TypedError> {
        let alternatives = self.alternatives()?;
        let choices = crate::egglog_utils::random_initial_choice(&self.frozen, &alternatives, rng);
        crate::egglog_utils::validate_choice_set(self, &alternatives, &choices)?;
        let (graph, outputs) =
            crate::egglog_utils::reconstruct(self, &choices, crate::hlir::DECODERS, custom_ops)?;
        Ok(ReferenceForest { graph, outputs })
    }
}

// These exact instances are shared by every optimization call. The generator
// remains parameterized, while the CPU's selected catalog is a fixed theory.
#[ruleset]
pub(crate) fn expressions() -> Ruleset {
    let unroll = [
        dynamic_rules::binary_unroll_rules(|shape, a, b, out| OpKind::Add(shape, a, b, out), 4),
        dynamic_rules::binary_unroll_rules(|shape, a, b, out| OpKind::Mul(shape, a, b, out), 4),
        dynamic_rules::binary_unroll_rules(|shape, a, b, out| OpKind::Mod(shape, a, b, out), 4),
        dynamic_rules::binary_unroll_rules(
            |shape, a, b, out| OpKind::LessThan(shape, a, b, out),
            4,
        ),
    ]
    .map(|rules| rules.expect("the fixed CPU unroll catalog's maximum of four is supported"));
    ruleset((&static_rules::expression, &static_rules::loops, unroll))
}

/// Build bound-setting and singleton-union actions; extra variables default to unbounded.
pub fn interval_actions(
    intervals: &DynDimIntervals,
    vars: impl IntoIterator<Item = char>,
) -> Vec<Action> {
    let mut vars: Vec<_> = intervals.keys().copied().chain(vars).collect();
    vars.sort_unstable();
    vars.dedup();
    let mut actions = Vec::new();
    for var in vars {
        let interval = intervals
            .get(&var)
            .copied()
            .unwrap_or_else(DimInterval::unbounded);
        let expr = DimExpr::from(var);
        actions.push(set(expr.lower(), interval.min));
        actions.push(set(expr.upper(), interval.max));
        if interval.min == interval.max {
            actions.push(union(expr, interval.min));
        }
    }
    actions
}

/// Execute the fixed CPU theory using the same phase driver as every backend.
pub fn optimize(
    egraph: &mut EGraph,
    intervals: &DynDimIntervals,
    log: bool,
) -> Result<(), TypedError> {
    run_schedule(
        egraph,
        std::time::Instant::now(),
        &base_phase_rules(),
        !intervals.is_empty(),
        log,
        |egraph| egraph.register(interval_actions(intervals, [])),
    )?;
    Ok(())
}

pub(crate) fn simplify_dimension(
    value: Expression,
    intervals: Option<&DynDimIntervals>,
) -> Result<Expression, TypedError> {
    let expr = let_("luminal::dimension", lower::dimension(&value, false));
    let mut egraph = EGraph::new(EGraphOptions::default());
    let schedule = if let Some(intervals) = intervals {
        egraph.register(interval_actions(intervals, value.dyn_vars()))?;
        sequence((&static_rules::expression, &static_rules::intervals)).repeat(5)
    } else {
        static_rules::expression.repeat(5)
    };
    egraph.register(&expr)?;
    egraph.run(schedule)?;
    egraph.run(static_rules::cleanup.saturate())?;
    let frozen = egraph.freeze()?;
    let root = frozen.lookup(&expr)?;
    let choices = Default::default();
    DecodeContext::new(&frozen, &choices).expression(&root)
}

pub(crate) fn dimensions_equal(lhs: Expression, rhs: Expression) -> Result<bool, TypedError> {
    let lhs = let_("luminal::lhs", lower::dimension(&lhs, false));
    let rhs = let_("luminal::rhs", lower::dimension(&rhs, false));
    let mut egraph = EGraph::new(EGraphOptions::default());
    egraph.register((&lhs, &rhs))?;
    egraph.run(static_rules::expression.saturate())?;
    egraph.run(static_rules::cleanup.saturate())?;
    egraph.check(eq(lhs, rhs))
}
