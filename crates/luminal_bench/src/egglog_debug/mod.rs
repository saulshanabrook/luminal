//! Egglog debugging and analysis utilities.
//!
//! This module provides tools for diagnosing egglog lowering issues,
//! particularly when HLIR operations fail to convert to backend implementations.

mod analysis;
mod report;
mod trace;

pub use analysis::*;
pub use report::*;
pub use trace::*;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Summarize HLIR node types in a graph.
pub fn summarize_hlir_ops(cx: &luminal::prelude::Graph) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for node in cx.graph.node_indices() {
        let name = cx.graph[node].type_name().to_string();
        *counts.entry(name).or_insert(0) += 1;
    }
    counts
}

/// Summarize authored root calls without parsing their display strings.
pub fn summarize_egglog_ops(program: &DependencyGraph) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for node in program.nodes.values() {
        *counts.entry(node.op_type.clone()).or_insert(0) += 1;
    }
    counts
}

/// Result of dtype analysis for a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DTypeStatus {
    /// dtype was successfully resolved
    Resolved(String),
    /// dtype lookup failed
    Missing(String),
}

impl DTypeStatus {
    pub fn is_missing(&self) -> bool {
        matches!(self, DTypeStatus::Missing(_))
    }

    pub fn is_resolved(&self) -> bool {
        matches!(self, DTypeStatus::Resolved(_))
    }
}

impl std::fmt::Display for DTypeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DTypeStatus::Resolved(s) => write!(f, "{}", s),
            DTypeStatus::Missing(err) => write!(f, "<missing:{}>", err),
        }
    }
}

/// Result of evaluating an arbitrary egglog function for a node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FactStatus {
    /// function value was successfully resolved
    Resolved(String),
    /// function lookup failed
    Missing(String),
}

impl FactStatus {
    pub fn is_missing(&self) -> bool {
        matches!(self, FactStatus::Missing(_))
    }
}

impl std::fmt::Display for FactStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FactStatus::Resolved(s) => write!(f, "{}", s),
            FactStatus::Missing(err) => write!(f, "<missing:{}>", err),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::{
        egglog_utils::SearchSpace,
        prelude::*,
        typed_reference::schema::{DType, DimExpr, EList, OpKind},
    };

    #[test]
    fn source_aliases_preserve_order_sharing_and_inspection() {
        let mut cx = Graph::new();
        let x = cx.tensor(4);
        (x + x).output();
        let program = DependencyGraph::from_hlir(&cx).unwrap();
        assert_eq!(program.order, ["t0", "t1", "t2"]);
        assert_eq!(program.nodes["t1"].inputs, ["t0", "t0"]);
        assert_eq!(program.roots, ["t2"]);
        assert_eq!(program.trace_back("t2", 20).len(), 3);
        let mut egraph = EGraph::default();
        for name in &program.order {
            egraph.register(&program.captures[name]).unwrap();
        }
        let frozen = SearchSpace::from_frozen(egraph.freeze().unwrap(), &[]).unwrap();
        let report = inspect_var(&frozen, &program, "t0", "HLIR");
        assert!(report.eval_error.is_none());
        assert_eq!(report.class_labels, ["Input"]);
        assert!(
            report.enodes[0]
                .children
                .iter()
                .any(|child| child.class_labels == ["0"])
        );
        assert!(
            inspect_var(&frozen, &program, "missing", "HLIR")
                .eval_error
                .is_some()
        );
    }

    #[test]
    fn reports_keep_global_coverage_missing_facts_and_json_shape() {
        #[relation(name = "seen")]
        fn seen(node: luminal::typed_reference::schema::Ir);

        let mut cx = Graph::new();
        cx.tensor(4).output();
        let program = DependencyGraph::from_hlir(&cx).unwrap();
        let mut egraph = EGraph::default();
        for name in &program.order {
            egraph.register(&program.captures[name]).unwrap();
        }
        egraph
            .register((
                set(program.captures["t0"].dtype(), DType::F32()),
                OpKind::Constant(2.0),
                seen(&program.captures["t0"]),
            ))
            .unwrap();
        let frozen = SearchSpace::from_frozen(egraph.freeze().unwrap(), &[]).unwrap();
        let report = analyze(
            &frozen,
            &program,
            "HLIR",
            &[
                FactQuery {
                    fn_name: "dtype".into(),
                    vars: vec!["t0".into(), "missing".into()],
                },
                FactQuery {
                    fn_name: "seen".into(),
                    vars: vec!["t0".into(), "t1".into()],
                },
            ],
            &[("Constant".into(), "MetalConstant".into())],
        );
        assert_eq!(report.op_reports[0].total_classes, 1);
        assert_eq!(report.op_reports[0].missing.len(), 1);
        assert!(
            matches!(&report.facts["dtype"]["t0"], FactStatus::Resolved(value) if value == "F32")
        );
        assert!(report.facts["dtype"]["missing"].is_missing());
        assert!(
            matches!(&report.facts["seen"]["t0"], FactStatus::Resolved(value) if value == "()")
        );
        assert!(report.facts["seen"]["t1"].is_missing());
        let json = serde_json::to_value(DebugReport {
            case_name: "probe".into(),
            size: 4,
            hlir_counts: Default::default(),
            egglog_counts: Default::default(),
            hlir_analysis: Some(report),
            backend_analysis: None,
            var_inspections: Vec::new(),
            function_traces: Vec::new(),
            build_succeeded: false,
        })
        .unwrap();
        assert!(json.get("build_succeeded").is_some());
        assert!(json["hlir_analysis"].get("op_reports").is_some());
        assert_eq!(
            frozen
                .frozen
                .eclasses()
                .flat_map(|class| class.nodes())
                .filter(|node| node.name() == "Input")
                .count(),
            1
        );
    }

    #[test]
    fn coverage_excludes_subsumed_and_missing_producer_cascades() {
        let program = DependencyGraph::default();
        let mut egraph = EGraph::default();
        let constant = OpKind::Constant(2.0);
        let backend = OpKind::Iota(1, 4);
        let head = DimExpr::MVar("dead");
        let shape = EList::from([&head]);
        egraph
            .register((
                union(constant, &backend),
                OpKind::Exp2(&shape, &shape, &shape),
            ))
            .unwrap();
        egraph.register((subsume(backend), subsume(head))).unwrap();
        let space = SearchSpace::from_frozen(egraph.freeze().unwrap(), &[]).unwrap();
        let report = analyze(
            &space,
            &program,
            "HLIR",
            &[],
            &[
                ("Constant".into(), "Iota".into()),
                ("Exp2".into(), "MetalExp2".into()),
            ],
        );
        assert_eq!(report.op_reports[0].missing.len(), 1);
        assert_eq!(report.op_reports[1].total_classes, 0);
        assert!(
            space
                .frozen
                .eclasses()
                .flat_map(|class| class.nodes())
                .any(|node| node.name() == "Exp2")
        );
        let unknown = analyze_dtype_chain(&space, &program, "unknown");
        assert!(unknown.all_resolved); // Preserve HEAD's unannotated unknown-target behavior.
        assert!(unknown.chain[0].dtype.is_none());
    }
}
