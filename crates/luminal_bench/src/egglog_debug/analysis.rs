//! Diagnostic queries over typed captures and immutable observations.
use super::{
    DTypeChainAnalysis, DTypeStatus, DependencyGraph, FactStatus, FunctionChainAnalysis,
    FunctionTraceEntry,
};
use egglog_experimental::typed::{ClassRef, Snapshot, TableRowRef, ValueRef, prelude::*};
use luminal::{
    egglog_utils::{Alternatives, SearchSpace, final_phases, main_schedule},
    op::Runtime,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Analysis result for lowering.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct LoweringAnalysis {
    pub label: String,
    pub root_labels: Vec<String>,
    pub output_input_labels: Vec<String>,
    /// Optional op-coverage reports (only filled when explicitly requested).
    pub op_reports: Vec<OpLoweringReport>,
    /// Optional evaluated facts, keyed by function name then variable name.
    pub facts: BTreeMap<String, BTreeMap<String, FactStatus>>,
}

/// Query for evaluating a function on a set of variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactQuery {
    pub fn_name: String,
    pub vars: Vec<String>,
}

/// Missing backend equivalent for a specific HLIR op instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpMissing {
    pub class_id: String,
    pub op: String,
    pub children: Vec<ChildInspection>,
}

/// Report for op lowering coverage.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OpLoweringReport {
    pub label: String,
    pub hlir_op: String,
    pub backend_op: String,
    pub total_classes: usize,
    pub missing: Vec<OpMissing>,
}

/// Inspection of a specific variable's eclass and dtype facts.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VarInspection {
    pub label: String,
    pub var: String,
    pub let_line: Option<String>,
    pub eval_error: Option<String>,
    pub class_id: Option<String>,
    pub class_type: Option<String>,
    pub class_labels: Vec<String>,
    pub dtype: Option<String>,
    pub enodes: Vec<EnodeInspection>,
}

/// Inspection of an enode within an eclass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnodeInspection {
    pub label: String,
    pub children: Vec<ChildInspection>,
}

/// Inspection of a child eclass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChildInspection {
    pub class_id: String,
    pub class_type: String,
    pub class_labels: Vec<String>,
    pub dtype: Option<String>,
}

/// Run the same recipe with either HLIR-only or backend declarations.
/// Source aliases are captures only in this diagnostic destination.
pub fn run_egraph<R: Runtime>(program: &DependencyGraph) -> Result<SearchSpace, TypedError> {
    let (phases, actions) = R::rules();
    let mut egraph = EGraph::default();
    egraph.register(actions)?;
    for name in &program.order {
        egraph.register(&program.captures[name])?;
    }
    // The original diagnostic full_egglog recipe runs one saturated main
    // schedule plus final phases, without the production driver's outer loop.
    egraph.run(main_schedule(&phases, false))?;
    for (_, schedule) in final_phases(&phases, false) {
        egraph.run(schedule)?;
    }
    SearchSpace::from_frozen(egraph.freeze()?, &[])
}

fn class_labels<'f>(class: ClassRef<'f>, alternatives: &Alternatives<'f>) -> Vec<String> {
    let mut labels: Vec<_> = alternatives
        .get(&class)
        .into_iter()
        .flatten()
        .map(|node| node.name().to_owned())
        .collect();
    labels.sort();
    labels.dedup();
    labels
}

/// Read an existing unary table row. No primitive evaluation, insertion or extraction.
fn function_row<'f>(
    frozen: &'f Snapshot,
    name: &str,
    matches: impl Fn(ValueRef<'f>) -> bool,
) -> Option<TableRowRef<'f>> {
    frozen
        .rows()
        .filter(|row| row.name() == name && row.args().len() == 1)
        .find(|row| matches(row.args().next().unwrap()))
}

fn describe<'f>(value: ValueRef<'f>, alternatives: &Alternatives<'f>) -> String {
    match value.class() {
        Some(class) => class_labels(class, alternatives).join("|"),
        None => luminal::visualization::display_frozen_value(value),
    }
}

fn eval_function<'f>(
    frozen: &'f Snapshot,
    alternatives: &Alternatives<'f>,
    program: &DependencyGraph,
    name: &str,
    var: &str,
) -> FactStatus {
    let result = (|| {
        let capture = program
            .captures
            .get(var)
            .ok_or_else(|| format!("unknown variable {var}"))?;
        let observed = frozen.lookup(capture).map_err(|error| error.to_string())?;
        let input = frozen
            .class_of(&observed)
            .map_err(|error| error.to_string())?;
        function_row(frozen, name, |value| value.class() == Some(input))
            .map(|row| {
                row.value()
                    .map_or_else(|| "()".into(), |value| describe(value, alternatives))
            })
            .ok_or_else(|| format!("no {name} row for {var}"))
    })();
    match result {
        Ok(value) => FactStatus::Resolved(value),
        Err(error) => FactStatus::Missing(error),
    }
}

fn inspect_child<'f>(
    frozen: &'f Snapshot,
    alternatives: &Alternatives<'f>,
    value: ValueRef<'f>,
) -> ChildInspection {
    let (class_id, class_labels) = match value.class() {
        Some(class) => (
            format!(
                "class {}",
                frozen
                    .eclasses()
                    .position(|candidate| candidate == class)
                    .unwrap()
            ),
            class_labels(class, alternatives),
        ),
        None => (
            describe(value, alternatives),
            vec![describe(value, alternatives)],
        ),
    };
    ChildInspection {
        class_id,
        class_type: value.sort().to_string(),
        class_labels,
        dtype: function_row(frozen, "dtype", |input| input == value)
            .and_then(|row| row.value())
            .map(|value| describe(value, alternatives)),
    }
}

/// Inspect one retained source alias without evaluating its initializer.
pub fn inspect_var(
    space: &SearchSpace,
    program: &DependencyGraph,
    var: &str,
    label: &str,
) -> VarInspection {
    let frozen = &space.frozen;
    let alternatives = space
        .alternatives()
        .expect("rootless diagnostic observation");
    let mut result = VarInspection {
        label: label.to_owned(),
        var: var.to_owned(),
        let_line: program.lines.get(var).cloned(),
        ..Default::default()
    };
    let observed = program
        .captures
        .get(var)
        .ok_or_else(|| format!("unknown variable {var}"))
        .and_then(|capture| frozen.lookup(capture).map_err(|error| error.to_string()));
    let observed = match observed {
        Ok(value) => value,
        Err(error) => {
            result.eval_error = Some(error);
            return result;
        }
    };
    let class = match frozen.class_of(&observed) {
        Ok(class) => class,
        Err(error) => {
            result.eval_error = Some(error.to_string());
            return result;
        }
    };
    result.class_id = Some(format!(
        "class {}",
        frozen
            .eclasses()
            .position(|candidate| candidate == class)
            .unwrap()
    ));
    result.class_type = Some(class.sort().to_string());
    result.class_labels = class_labels(class, &alternatives);
    result.dtype = function_row(frozen, "dtype", |value| value.class() == Some(class))
        .and_then(|row| row.value())
        .map(|value| describe(value, &alternatives));
    result.enodes = alternatives
        .get(&class)
        .into_iter()
        .flatten()
        .map(|node| EnodeInspection {
            label: node.name().to_owned(),
            children: node
                .args()
                .map(|arg| inspect_child(frozen, &alternatives, arg))
                .collect(),
        })
        .collect();
    result
}

/// Preserve report structure and the original global op-coverage scope.
pub fn analyze(
    space: &SearchSpace,
    program: &DependencyGraph,
    label: &str,
    fact_queries: &[FactQuery],
    op_mappings: &[(String, String)],
) -> LoweringAnalysis {
    let frozen = &space.frozen;
    let alternatives = space
        .alternatives()
        .expect("rootless diagnostic observation");
    let mut analysis = LoweringAnalysis {
        label: label.to_owned(),
        ..Default::default()
    };
    for root in &program.roots {
        let inspection = inspect_var(space, program, root, label);
        analysis.root_labels.extend(inspection.class_labels);
    }
    analysis.root_labels.sort();
    analysis.root_labels.dedup();
    for class in frozen.eclasses() {
        if let Some(output) = alternatives
            .get(&class)
            .into_iter()
            .flatten()
            .find(|node| node.name() == "Output")
        {
            if let Some(input) = output.args().next() {
                analysis.output_input_labels =
                    inspect_child(frozen, &alternatives, input).class_labels;
            }
            break;
        }
    }
    for (source, backend) in op_mappings {
        let mut report = OpLoweringReport {
            label: label.to_owned(),
            hlir_op: source.clone(),
            backend_op: backend.clone(),
            ..Default::default()
        };
        for (index, class) in frozen.eclasses().enumerate() {
            let nodes: Vec<_> = alternatives.get(&class).into_iter().flatten().collect();
            let source_nodes: Vec<_> = nodes.iter().filter(|node| node.name() == source).collect();
            if source_nodes.is_empty() {
                continue;
            }
            report.total_classes += 1;
            if nodes.iter().any(|node| node.name() == backend) {
                continue;
            }
            report.missing.push(OpMissing {
                class_id: format!("class {index}"),
                op: source.clone(),
                children: source_nodes[0]
                    .args()
                    .map(|value| inspect_child(frozen, &alternatives, value))
                    .collect(),
            });
        }
        analysis.op_reports.push(report);
    }
    for query in fact_queries {
        analysis.facts.insert(
            query.fn_name.clone(),
            query
                .vars
                .iter()
                .map(|var| {
                    (
                        var.clone(),
                        eval_function(frozen, &alternatives, program, &query.fn_name, var),
                    )
                })
                .collect(),
        );
    }
    analysis
}

pub fn analyze_dtype_chain(
    space: &SearchSpace,
    program: &DependencyGraph,
    target: &str,
) -> DTypeChainAnalysis {
    let frozen = &space.frozen;
    let alternatives = space
        .alternatives()
        .expect("rootless diagnostic observation");
    let mut chain = program.trace_back(target, 20);
    for entry in &mut chain {
        if !program.nodes.contains_key(&entry.var) {
            continue;
        }
        entry.dtype = Some(
            match eval_function(frozen, &alternatives, program, "dtype", &entry.var) {
                FactStatus::Resolved(value) => DTypeStatus::Resolved(value),
                FactStatus::Missing(error) => DTypeStatus::Missing(error),
            },
        );
    }
    let first_missing = chain
        .iter()
        .find(|entry| matches!(&entry.dtype, Some(DTypeStatus::Missing(_))))
        .map(|entry| entry.var.clone());
    let all_resolved = first_missing.is_none();
    DTypeChainAnalysis {
        target: target.to_owned(),
        chain,
        first_missing,
        all_resolved,
    }
}

pub fn analyze_function_chain(
    space: &SearchSpace,
    program: &DependencyGraph,
    name: &str,
    target: &str,
) -> FunctionChainAnalysis {
    let frozen = &space.frozen;
    let alternatives = space
        .alternatives()
        .expect("rootless diagnostic observation");
    let chain: Vec<_> = program
        .trace_back(target, 20)
        .into_iter()
        .map(|entry| FunctionTraceEntry {
            depth: entry.depth,
            var: entry.var.clone(),
            op_type: entry.op_type,
            status: eval_function(frozen, &alternatives, program, name, &entry.var),
        })
        .collect();
    let first_missing = chain
        .iter()
        .find(|entry| entry.status.is_missing())
        .map(|entry| entry.var.clone());
    let all_resolved = first_missing.is_none();
    FunctionChainAnalysis {
        target: target.to_owned(),
        fn_name: name.to_owned(),
        chain,
        first_missing,
        all_resolved,
    }
}
