use anyhow::Result;
use egglog_experimental::typed::{FrozenScalar, FrozenValue, FrozenValueView};
use egglog_experimental::{EGraph, SerializeConfig};
use std::fmt::{Debug, Display};
use std::string::String;

use crate::prelude::petgraph::dot::{Config, Dot};
use crate::prelude::*;
use petgraph::Directed;
use petgraph::prelude::StableGraph;
use std::io::Write;

pub trait ToHtml {
    fn to_html(&self) -> Result<String>;
}

pub trait ToDot {
    fn to_dot(&self) -> Result<String>;
}

const EGRAPH_VIS_TEMPLATE: &str = r#"
<div id="egraph-visualizer"></div>
<link rel="stylesheet" href="https://esm.sh/egraph-visualizer/dist/style.css" />
<script type="module">
    import { mount } from "https://esm.sh/egraph-visualizer";
    const egraph = {{JSON_TEMPLATE}};
    const mounted = mount(document.getElementById("egraph-visualizer"));
    mounted.render([JSON.stringify(egraph)]);
</script>"#;

/// A display label, not a value codec or an executable expression.
pub fn display_frozen_value(value: FrozenValue<'_>) -> String {
    match value.view() {
        FrozenValueView::Scalar(value) => match value {
            FrozenScalar::Unit => "()".into(),
            FrozenScalar::Bool(value) => value.to_string(),
            FrozenScalar::I64(value) => value.to_string(),
            FrozenScalar::F64(bits) => format!("{:?}", f64::from_bits(*bits)),
            FrozenScalar::String(value) => format!("{value:?}"),
            FrozenScalar::BigInt(value) => value.to_string(),
            FrozenScalar::BigRat(value) => value.to_string(),
            FrozenScalar::Rational(value) => value.to_string(),
        },
        _ => value.sort().name().to_owned(),
    }
}

impl ToHtml for EGraph {
    fn to_html(&self) -> Result<String> {
        let egraph_as_json =
            serde_json::to_string_pretty(&self.serialize(SerializeConfig::default()).egraph)?
                .replace('<', "\\u003c");
        Ok(EGRAPH_VIS_TEMPLATE.replace("{{JSON_TEMPLATE}}", &egraph_as_json))
    }
}

impl ToDot for EGraph {
    fn to_dot(&self) -> Result<String> {
        Ok(self.serialize(SerializeConfig::default()).egraph.to_dot())
    }
}

/// Implements `ToDot` for [`LLIRGraph`] and [`HLIRGraph`]
/// TODO: This simple extraction can be improved in the future with support for edge labels
impl<N, E> ToDot for petgraph::stable_graph::StableGraph<N, E>
where
    N: std::fmt::Debug,
    E: std::fmt::Debug,
{
    fn to_dot(&self) -> Result<String> {
        Ok(format!(
            "{:?}",
            Dot::with_config(self, &[Config::EdgeNoLabel])
        ))
    }
}

/// View a debug graph in the browser using Luminal Visualizer
pub fn display_graph<E>(graph: &StableGraph<impl Display + Debug, E, Directed, u32>) {
    let dot_source = graph_to_dot(graph, None);

    // Open in Luminal Visualizer (served from graph_viewer dev server)
    let url = format!(
        "http://viz.luminal.com/?dot={}",
        urlencoding::encode(&dot_source)
    );
    let _ = open::that(&url);
}

/// View a debug graph in the browser using Luminal Visualizer
pub fn display_graph_marked_nodes<E>(
    graph: &StableGraph<impl Display + Debug, E, Directed, u32>,
    mark_nodes: Option<Vec<NodeIndex>>,
) {
    let dot_source = graph_to_dot(graph, mark_nodes);

    // Open in Luminal Visualizer (served from graph_viewer dev server)
    let url = format!(
        "http://localhost:5173/?dot={}",
        urlencoding::encode(&dot_source)
    );
    let _ = open::that(&url);
}

/// View a debug graph in the browser using Luminal Visualizer
pub fn display_graph_to_file<E>(
    graph: &StableGraph<impl Display + Debug, E, Directed, u32>,
    mark_nodes: Option<Vec<NodeIndex>>,
    file_name: &str,
) {
    let dot_source = graph_to_dot(graph, mark_nodes);

    // Write the DOT source to the file
    let mut file = std::fs::File::create(file_name).unwrap();
    file.write_all(dot_source.as_bytes()).unwrap();
}

fn escape_dot_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn graph_to_dot<E>(
    graph: &StableGraph<impl Display + Debug, E, Directed, u32>,
    mark_nodes: Option<Vec<NodeIndex>>,
) -> String {
    let mut dot = String::from("digraph {\n");
    let mut map = std::collections::HashMap::new();

    for (next_id, node) in graph.node_indices().enumerate() {
        let weight = graph.node_weight(node).unwrap();
        let label = escape_dot_string(&format!("{}", weight));
        let tooltip = escape_dot_string(&format!("{:?}", weight));
        let id = next_id;
        map.insert(node, id);

        let is_marked = mark_nodes
            .as_ref()
            .map(|m| m.contains(&node))
            .unwrap_or(false);

        if is_marked {
            dot.push_str(&format!(
                "    n{id} [label=\"{label}\", tooltip=\"{tooltip}\", style=filled, fillcolor=\"#ffeb3b\", fontcolor=\"black\", color=\"#ffd600\"];\n"
            ));
        } else {
            dot.push_str(&format!(
                "    n{id} [label=\"{label}\", tooltip=\"{tooltip}\"];\n"
            ));
        }
    }

    for edge in graph.edge_indices() {
        let (src, dest) = graph.edge_endpoints(edge).unwrap();
        if let (Some(src_id), Some(dest_id)) = (map.get(&src), map.get(&dest)) {
            dot.push_str(&format!("    n{src_id} -> n{dest_id};\n"));
        }
    }

    dot.push_str("}\n");
    dot
}

#[cfg(test)]
mod native_tests {
    use super::*;
    use crate::typed_reference::schema::DimExpr;
    use egglog_experimental::typed::{EGraph as TypedEGraph, builtins as egg, prelude::*};

    #[sort]
    struct DisplayFixture;

    #[declarations]
    impl DisplayFixture {
        fn values(items: egg::Vec<DimExpr>) -> Self;
        #[constructor(cost = 0)]
        fn free(value: DisplayFixture) -> Self;
        #[constructor(cost = 37)]
        fn expensive(value: DisplayFixture) -> Self;
    }

    #[test]
    fn rendering_preserves_declared_costs_and_subsumption_across_cycles() {
        let mut egraph = TypedEGraph::new(EGraphOptions::default());
        let base = DisplayFixture::values([DimExpr::from(9)]);
        let free = DisplayFixture::free(&base);
        let expensive = DisplayFixture::expensive(&base);
        egraph
            .register((
                union(&base, &free),
                union(&base, &expensive),
                subsume(&free),
            ))
            .unwrap();
        let egraph = egraph.into_raw();
        let data = egraph.serialize(SerializeConfig::default()).egraph;
        let nodes = &data.nodes;
        for (expression, cost) in [(&base, 1.0), (&free, 0.0), (&expensive, 37.0)] {
            let name = expression.call_name().unwrap();
            let node = nodes.values().find(|node| node.op == name).unwrap();
            assert_eq!(node.cost.into_inner(), cost);
            assert_eq!(node.subsumed, expression == &free);
            assert_eq!(
                data.class_data[&node.eclass].typ.as_deref(),
                Some(DisplayFixture::sort_ref().name())
            );
            if expression != &base {
                let child = &node.children[0];
                assert_eq!(nodes[child].eclass, node.eclass);
            }
        }
        for node in nodes.values() {
            if node.op != free.call_name().unwrap() && node.op != expensive.call_name().unwrap() {
                assert_eq!(node.cost.into_inner(), 1.0);
            }
        }
        let html = egraph.to_html().unwrap();
        assert!(html.contains("\"cost\": 0.0"));
        assert!(html.contains("\"cost\": 37.0"));
        assert!(
            egraph
                .to_dot()
                .unwrap()
                .contains(expensive.call_name().unwrap())
        );
    }

    #[test]
    fn rendering_preserves_cycles_literals_containers_and_escaping() {
        let mut egraph = TypedEGraph::new(EGraphOptions::default());
        let x = DimExpr::MVar("</script>\"\\\n");
        let sum = x.clone() + 1;
        egraph
            .register((union(&x, &sum), DisplayFixture::values([x.clone(), x])))
            .unwrap();
        let egraph = egraph.into_raw();
        let data = egraph.serialize(SerializeConfig::default()).egraph;
        let nodes = &data.nodes;
        assert!(nodes.values().any(|node| node.op == "MAdd"));
        assert!(nodes.values().any(|node| node.op == "1"));
        assert!(nodes.values().any(|node| {
            node.children.len() == 2
                && nodes[&node.children[0]].eclass == nodes[&node.children[1]].eclass
        }));
        for node in nodes.values() {
            for child in &node.children {
                assert!(nodes.contains_key(child));
            }
        }
        let html = egraph.to_html().unwrap();
        assert_eq!(html.matches("</script>").count(), 1);
        assert!(html.contains("\\u003c/script>"));
        let dot = egraph.to_dot().unwrap();
        assert!(dot.contains("MVar"));
        assert!(dot.contains("MAdd"));
    }
}
