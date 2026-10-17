//! Dependency chain tracing for dtype propagation analysis.

use super::{DTypeStatus, FactStatus};
use egglog_experimental::typed::{EgglogValue, TypedError, let_};
use luminal::{prelude::*, typed_reference::schema::Ir};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::{
    cmp::Reverse,
    collections::{BTreeMap, BinaryHeap},
};

/// A node in the dependency graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepNode {
    pub var: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub dtype: Option<DTypeStatus>,
}

/// Source dependencies and typed captures used only by the diagnostic tool.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct DependencyGraph {
    pub nodes: HashMap<String, DepNode>,
    pub roots: Vec<String>,
    #[serde(skip)]
    pub(crate) captures: BTreeMap<String, Ir>,
    #[serde(skip)]
    pub(crate) lines: BTreeMap<String, String>,
    #[serde(skip)]
    pub order: Vec<String>,
}

impl DependencyGraph {
    /// Keep the original topological tN aliases without executable source text.
    /// Per-node captures belong to this diagnostic graph, not production lowering.
    pub fn from_hlir(cx: &Graph) -> Result<Self, TypedError> {
        use petgraph::Direction;
        let mut graph = Self::default();
        let mut indegrees: HashMap<_, _> = cx
            .graph
            .node_indices()
            .map(|node| {
                (
                    node,
                    cx.graph
                        .neighbors_directed(node, Direction::Incoming)
                        .count(),
                )
            })
            .collect();
        let mut ready = BinaryHeap::new();
        for (&node, &degree) in &indegrees {
            if degree == 0 {
                ready.push((Reverse(node.index()), node));
            }
        }
        let mut names: HashMap<NodeIndex, String> = HashMap::new();
        while let Some((_, node)) = ready.pop() {
            let inputs: Vec<String> = cx
                .get_sources(node)
                .iter()
                .map(|source| names[source].clone())
                .collect();
            let values: Vec<_> = inputs.iter().map(|name| &graph.captures[name]).collect();
            let expression = cx.graph[node].to_ir(&values);
            let name = format!("t{}", graph.order.len());
            let op_type = expression
                .call_name()
                .ok_or_else(|| {
                    TypedError::Invalid("HLIR diagnostic lowering must produce a call".into())
                })?
                .to_owned();
            graph
                .lines
                .insert(name.clone(), format!("let {name} = {expression:?};"));
            graph
                .captures
                .insert(name.clone(), let_(name.clone(), expression));
            graph.nodes.insert(
                name.clone(),
                DepNode {
                    var: name.clone(),
                    op_type,
                    inputs,
                    dtype: None,
                },
            );
            graph.order.push(name.clone());
            names.insert(node, name);
            for target in cx.graph.neighbors_directed(node, Direction::Outgoing) {
                let degree = indegrees.get_mut(&target).unwrap();
                *degree -= 1;
                if *degree == 0 {
                    ready.push((Reverse(target.index()), target));
                }
            }
        }
        if names.len() != indegrees.len() {
            return Err(TypedError::Invalid(
                "diagnostic HLIR contains a cycle".into(),
            ));
        }
        graph.roots = cx
            .graph
            .externals(Direction::Outgoing)
            .filter_map(|node| names.remove(&node))
            .collect();
        Ok(graph)
    }

    /// A display-only typed listing. It is never parsed or submitted to Egglog.
    pub fn display_program(&self) -> String {
        self.order
            .iter()
            .map(|name| self.lines[name].as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Trace the dependency chain from a target variable back to inputs.
    pub fn trace_back(&self, target: &str, max_depth: usize) -> Vec<TraceEntry> {
        let mut result = Vec::new();
        self.trace_back_recursive(target, 0, max_depth, &mut result, &mut HashSet::new());
        result
    }

    fn trace_back_recursive(
        &self,
        var: &str,
        depth: usize,
        max_depth: usize,
        result: &mut Vec<TraceEntry>,
        visited: &mut HashSet<String>,
    ) {
        if depth > max_depth || visited.contains(var) {
            return;
        }
        visited.insert(var.to_string());

        let node = match self.nodes.get(var) {
            Some(n) => n,
            None => {
                result.push(TraceEntry {
                    depth,
                    var: var.to_string(),
                    op_type: "<unknown>".to_string(),
                    dtype: None,
                });
                return;
            }
        };

        result.push(TraceEntry {
            depth,
            var: node.var.clone(),
            op_type: node.op_type.clone(),
            dtype: node.dtype.clone(),
        });

        for input in &node.inputs {
            self.trace_back_recursive(input, depth + 1, max_depth, result, visited);
        }
    }

    /// Find the first node in a chain that has missing dtype.
    pub fn find_dtype_break(&self, target: &str) -> Option<String> {
        let trace = self.trace_back(target, 20);
        for entry in trace {
            if let Some(DTypeStatus::Missing(_)) = entry.dtype {
                return Some(entry.var);
            }
        }
        None
    }
}

/// Entry in a trace result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEntry {
    pub depth: usize,
    pub var: String,
    pub op_type: String,
    pub dtype: Option<DTypeStatus>,
}

impl TraceEntry {
    /// Format as indented tree line.
    pub fn format_tree(&self) -> String {
        let indent = "  ".repeat(self.depth);
        let prefix = if self.depth == 0 { "" } else { "├── " };
        let dtype_str = match &self.dtype {
            Some(d) => format!(" dtype={}", d),
            None => String::new(),
        };
        format!(
            "{}{}{} ({}){}",
            indent, prefix, self.var, self.op_type, dtype_str
        )
    }
}

/// Entry in a function trace result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionTraceEntry {
    pub depth: usize,
    pub var: String,
    pub op_type: String,
    pub status: FactStatus,
}

impl FunctionTraceEntry {
    pub fn format_tree(&self) -> String {
        let indent = "  ".repeat(self.depth);
        let prefix = if self.depth == 0 { "" } else { "├── " };
        format!(
            "{}{}{} ({}) {}",
            indent, prefix, self.var, self.op_type, self.status
        )
    }
}

/// Result of dtype chain analysis.
#[derive(Debug, Serialize, Deserialize)]
pub struct DTypeChainAnalysis {
    pub target: String,
    pub chain: Vec<TraceEntry>,
    pub first_missing: Option<String>,
    pub all_resolved: bool,
}

impl DTypeChainAnalysis {
    /// Create from dependency graph and target variable.
    pub fn analyze(graph: &DependencyGraph, target: &str) -> Self {
        let chain = graph.trace_back(target, 20);
        let first_missing = chain
            .iter()
            .find(|e| matches!(&e.dtype, Some(DTypeStatus::Missing(_))))
            .map(|e| e.var.clone());
        let all_resolved = chain
            .iter()
            .all(|e| matches!(&e.dtype, Some(DTypeStatus::Resolved(_)) | None));

        DTypeChainAnalysis {
            target: target.to_string(),
            chain,
            first_missing,
            all_resolved,
        }
    }
}

/// Result of function chain analysis.
#[derive(Debug, Serialize, Deserialize)]
pub struct FunctionChainAnalysis {
    pub target: String,
    pub fn_name: String,
    pub chain: Vec<FunctionTraceEntry>,
    pub first_missing: Option<String>,
    pub all_resolved: bool,
}
