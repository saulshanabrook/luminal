use anyhow::Result;
use egglog_experimental::typed::{Snapshot, ValueRef, builtins as egg};
use std::collections::{BTreeMap, HashMap, VecDeque};
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
pub fn display_frozen_value(value: ValueRef<'_>) -> String {
    if value.class().is_some() {
        value.sort().to_string()
    } else {
        value.to_string()
    }
}

/// Presentation data for the egraph visualizer. IDs are local to this rendering;
/// class references are never unfolded or selected for extraction.
#[derive(serde::Serialize)]
struct Visualization {
    nodes: BTreeMap<String, VisualizationNode>,
    root_eclasses: Vec<String>,
    class_data: BTreeMap<String, VisualizationClass>,
}

#[derive(serde::Serialize)]
struct VisualizationNode {
    op: String,
    children: Vec<String>,
    eclass: String,
    cost: f64,
    subsumed: bool,
}

#[derive(serde::Serialize)]
struct VisualizationClass {
    #[serde(rename = "type")]
    typ: Option<String>,
}

fn visualization(snapshot: &Snapshot) -> Result<Visualization> {
    let mut data = Visualization {
        nodes: BTreeMap::new(),
        root_eclasses: Vec::new(),
        class_data: BTreeMap::new(),
    };
    let mut producers = Vec::new();
    let mut class_ids = HashMap::new();
    let mut representatives = HashMap::new();
    for (index, class) in snapshot.eclasses().enumerate() {
        let class_id = format!("c{index}");
        class_ids.insert(class, class_id.clone());
        data.class_data.insert(
            class_id.clone(),
            VisualizationClass {
                typ: Some(class.sort().to_string()),
            },
        );
        for node in class.nodes() {
            let node_id = format!("n{}", producers.len());
            representatives
                .entry(class)
                .or_insert_with(|| node_id.clone());
            producers.push((class_id.clone(), node_id, node));
        }
    }

    // Intern exact primitive/container values before adding edges. Equality
    // fields stop at their existing class, so cycles never require recursion.
    let mut values = Vec::new();
    let mut value_ids = HashMap::new();
    let rows: Vec<_> = snapshot.rows().collect();
    let mut pending: VecDeque<_> = producers
        .iter()
        .flat_map(|(_, _, node)| node.args())
        .chain(rows.iter().flat_map(|row| row.args().chain(row.value())))
        .collect();
    while let Some(value) = pending.pop_front() {
        if value.class().is_some() || value_ids.contains_key(&value) {
            continue;
        }
        value_ids.insert(value, values.len());
        values.push(value);
        pending.extend(value.children());
    }

    let child_id = |value: ValueRef<'_>| match value.class() {
        Some(class) => representatives.get(&class).cloned().ok_or_else(|| {
            anyhow::anyhow!("cannot render a reference to an equality class without producers")
        }),
        None => Ok(format!("v{}", value_ids[&value])),
    };
    for (class_id, node_id, node) in producers {
        let children = node.args().map(&child_id).collect::<Result<Vec<_>>>()?;
        // As in the visualizer's existing model, an absent declared cost and
        // primitive/container values use the display cost 1.
        let cost = node
            .declared_cost()
            .map(|value| i64::try_from(&value.try_typed::<egg::I64>()?))
            .transpose()?
            .unwrap_or(1) as f64;
        data.nodes.insert(
            node_id,
            VisualizationNode {
                op: node.name().to_owned(),
                children,
                eclass: class_id,
                cost,
                subsumed: node.is_subsumed(),
            },
        );
    }
    for (index, value) in values.into_iter().enumerate() {
        let id = format!("v{index}");
        let children = value
            .children()
            .map(&child_id)
            .collect::<Result<Vec<_>>>()?;
        data.class_data.insert(
            id.clone(),
            VisualizationClass {
                typ: Some(value.sort().to_string()),
            },
        );
        data.nodes.insert(
            id.clone(),
            VisualizationNode {
                op: display_frozen_value(value),
                children,
                eclass: id,
                cost: 1.0,
                subsumed: false,
            },
        );
    }
    for (index, row) in rows.into_iter().enumerate() {
        let id = format!("r{index}");
        let children = row.args().map(&child_id).collect::<Result<Vec<_>>>()?;
        let eclass = match row.value() {
            Some(value) => match value.class() {
                Some(class) => class_ids[&class].clone(),
                None => format!("v{}", value_ids[&value]),
            },
            // Relations have no output value. This is only a display group for
            // the stored fact, not a fabricated equality class or Unit value.
            None => {
                data.class_data
                    .insert(id.clone(), VisualizationClass { typ: None });
                id.clone()
            }
        };
        data.nodes.insert(
            id,
            VisualizationNode {
                op: row.name().to_owned(),
                children,
                eclass,
                cost: 1.0,
                subsumed: false,
            },
        );
    }
    Ok(data)
}

impl ToHtml for Snapshot {
    fn to_html(&self) -> Result<String> {
        let egraph_as_json =
            serde_json::to_string_pretty(&visualization(self)?)?.replace('<', "\\u003c");
        Ok(EGRAPH_VIS_TEMPLATE.replace("{{JSON_TEMPLATE}}", &egraph_as_json))
    }
}

impl ToDot for Snapshot {
    fn to_dot(&self) -> Result<String> {
        let data = visualization(self)?;
        let mut classes: BTreeMap<_, Vec<_>> = BTreeMap::new();
        for (node_id, node) in &data.nodes {
            classes
                .entry(node.eclass.as_str())
                .or_default()
                .push((node_id, node));
        }
        let mut dot = String::from("digraph egraph {\n    compound=true;\n");
        for (class_id, class) in &data.class_data {
            dot.push_str(&format!(
                "    subgraph cluster_{class_id} {{\n        label=\"{}\";\n",
                escape_dot_string(class.typ.as_deref().unwrap_or_default())
            ));
            for (node_id, node) in classes.get(class_id.as_str()).into_iter().flatten() {
                dot.push_str(&format!(
                    "        {node_id} [label=\"{}\", tooltip=\"cost {}\", style=\"{}\"];\n",
                    escape_dot_string(&node.op),
                    node.cost,
                    if node.subsumed { "dashed" } else { "solid" },
                ));
            }
            dot.push_str("    }\n");
        }
        for (node_id, node) in &data.nodes {
            for (index, child) in node.children.iter().enumerate() {
                dot.push_str(&format!("    {node_id} -> {child} [label=\"{index}\"];\n"));
            }
        }
        dot.push_str("}\n");
        Ok(dot)
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
mod snapshot_tests {
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
        #[function(no_merge)]
        fn length(&self) -> egg::I64;
        #[function(no_merge)]
        fn mirror(&self) -> Self;
        #[relation]
        fn present(&self);
    }

    const VALUES: &str = concat!(module_path!(), "::DisplayFixture::values");
    const FREE: &str = concat!(module_path!(), "::DisplayFixture::free");
    const EXPENSIVE: &str = concat!(module_path!(), "::DisplayFixture::expensive");
    const LENGTH: &str = concat!(module_path!(), "::DisplayFixture::length");
    const MIRROR: &str = concat!(module_path!(), "::DisplayFixture::mirror");
    const PRESENT: &str = concat!(module_path!(), "::DisplayFixture::present");

    #[test]
    fn rendering_preserves_function_and_relation_rows_without_constructor_producers() {
        let root = let_(
            "table_root",
            DisplayFixture::values(egg::Vec::of([DimExpr::from(9)])),
        );
        let mut egraph = TypedEGraph::new().unwrap();
        egraph
            .register((
                &root,
                set(root.length(), 3_i64),
                set(root.mirror(), &root),
                root.present(),
            ))
            .unwrap();
        let snapshot = egraph.freeze().unwrap();
        let class = snapshot.class_of(&snapshot.lookup(&root).unwrap()).unwrap();
        assert!(
            class
                .nodes()
                .all(|node| ![LENGTH, MIRROR, PRESENT].contains(&node.name()))
        );
        assert_eq!(
            snapshot
                .rows()
                .filter(|row| [LENGTH, MIRROR, PRESENT].contains(&row.name()))
                .count(),
            3
        );
        let data = visualization(&snapshot).unwrap();
        let nodes = &data.nodes;
        let base = nodes.values().find(|node| node.op == VALUES).unwrap();
        let length = nodes.values().find(|node| node.op == LENGTH).unwrap();
        let scalar = nodes.values().find(|node| node.op == "3").unwrap();
        assert_eq!(length.eclass, scalar.eclass);
        let mirror = nodes.values().find(|node| node.op == MIRROR).unwrap();
        assert_eq!(mirror.eclass, base.eclass);
        let present = nodes.values().find(|node| node.op == PRESENT).unwrap();
        assert!(data.class_data[&present.eclass].typ.is_none());
        for row in [length, mirror, present] {
            assert_eq!(row.children.len(), 1);
            assert_eq!(nodes[&row.children[0]].eclass, base.eclass);
            assert_eq!(row.cost, 1.0);
            assert!(!row.subsumed);
        }
        let html = snapshot.to_html().unwrap();
        let dot = snapshot.to_dot().unwrap();
        for name in [LENGTH, MIRROR, PRESENT] {
            assert!(html.contains(name));
            assert!(dot.contains(name));
        }
    }

    #[test]
    fn rendering_rejects_direct_references_to_empty_classes() {
        let base = DisplayFixture::values(egg::Vec::of([DimExpr::from(9)]));
        let empty = let_("empty_direct", &base);
        let mut egraph = TypedEGraph::new().unwrap();
        egraph
            .register((&empty, DisplayFixture::free(&empty)))
            .unwrap();
        egraph.register(delete(&base)).unwrap();
        let snapshot = egraph.freeze().unwrap();
        assert_eq!(
            snapshot
                .class_of(&snapshot.lookup(&empty).unwrap())
                .unwrap()
                .nodes()
                .len(),
            0
        );
        for rendered in [snapshot.to_html(), snapshot.to_dot()] {
            assert!(
                rendered
                    .unwrap_err()
                    .to_string()
                    .contains("cannot render a reference to an equality class without producers")
            );
        }
    }

    #[test]
    fn rendering_rejects_container_references_to_empty_classes() {
        let number = DimExpr::from(9);
        let empty = let_("empty_container", &number);
        let mut egraph = TypedEGraph::new().unwrap();
        egraph
            .register((
                &empty,
                DisplayFixture::values(egg::Vec::of([empty.clone()])),
            ))
            .unwrap();
        egraph.register(delete(&number)).unwrap();
        let snapshot = egraph.freeze().unwrap();
        assert_eq!(
            snapshot
                .class_of(&snapshot.lookup(&empty).unwrap())
                .unwrap()
                .nodes()
                .len(),
            0
        );
        for rendered in [snapshot.to_html(), snapshot.to_dot()] {
            assert!(
                rendered
                    .unwrap_err()
                    .to_string()
                    .contains("cannot render a reference to an equality class without producers")
            );
        }
    }

    #[test]
    fn rendering_preserves_declared_costs_and_subsumption_across_cycles() {
        let mut egraph = TypedEGraph::new().unwrap();
        let base = DisplayFixture::values(egg::Vec::of([DimExpr::from(9)]));
        let free = DisplayFixture::free(&base);
        let expensive = DisplayFixture::expensive(&base);
        egraph
            .register((
                union(&base, &free),
                union(&base, &expensive),
                subsume(&free),
            ))
            .unwrap();
        let egraph = egraph.freeze().unwrap();
        let data = visualization(&egraph).unwrap();
        let nodes = &data.nodes;
        for (expression, name, cost) in [
            (&base, VALUES, 1.0),
            (&free, FREE, 0.0),
            (&expensive, EXPENSIVE, 37.0),
        ] {
            let node = nodes.values().find(|node| node.op == name).unwrap();
            assert_eq!(node.cost, cost);
            assert_eq!(node.subsumed, expression == &free);
            assert_eq!(
                data.class_data[&node.eclass].typ.as_deref(),
                Some(DisplayFixture::sort_ref().to_string().as_str())
            );
            if expression != &base {
                let child = &node.children[0];
                assert_eq!(nodes[child].eclass, node.eclass);
            }
        }
        for node in nodes.values() {
            if node.op != FREE && node.op != EXPENSIVE {
                assert_eq!(node.cost, 1.0);
            }
        }
        let html = egraph.to_html().unwrap();
        assert!(html.contains("\"cost\": 0.0"));
        assert!(html.contains("\"cost\": 37.0"));
        assert!(egraph.to_dot().unwrap().contains(EXPENSIVE));
    }

    #[test]
    fn rendering_preserves_cycles_literals_containers_and_escaping() {
        let mut egraph = TypedEGraph::new().unwrap();
        let x = DimExpr::MVar("</script>\"\\\n");
        let sum = x.clone() + 1;
        egraph
            .register((
                union(&x, &sum),
                DisplayFixture::values(egg::Vec::of([x.clone(), x])),
            ))
            .unwrap();
        let egraph = egraph.freeze().unwrap();
        let data = visualization(&egraph).unwrap();
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
