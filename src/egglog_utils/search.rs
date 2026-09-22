use super::*;
use crate::hlir::{Input, LoopEnd, LoopStart, Output};
use crate::typed_reference::schema::{self, DType as IrDType, DimExpr, EList, IList, Ir, OpKind};
use egglog_experimental::typed::{FrozenEClass, FrozenENode, FrozenValueView, builtins as egg};
use petgraph::graph::NodeIndex;
use rand::Rng;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A snapshot and its ordered observed roots. Selection never changes either.
#[derive(Clone)]
pub struct SearchSpace {
    pub frozen: FrozenEGraph,
    pub roots: Vec<Ir>,
    /// Apply conditional HLIR cleanup to alternatives, without changing the snapshot.
    pub clean_hlir: bool,
    /// Kinds excluded from survivor evidence during conditional HLIR cleanup.
    pub cleanup_ignored: fn(&OpKind) -> Result<bool, TypedError>,
}

impl std::fmt::Debug for SearchSpace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SearchSpace")
            .field("frozen", &self.frozen)
            .field("roots", &self.roots)
            .field("clean_hlir", &self.clean_hlir)
            .finish()
    }
}

/// Successful native metadata values only. Keys own their snapshot provenance;
/// candidate-dependent IR, input lists, and producer choices never enter here.
#[derive(Default)]
struct Metadata {
    dimensions: FxHashMap<DimExpr, Expression>,
    lists: FxHashMap<EList, Vec<Expression>>,
}

pub type EGraphChoiceSet<'f> = FxHashMap<FrozenEClass<'f>, FrozenENode<'f>>;
pub type Alternatives<'f> = FxHashMap<FrozenEClass<'f>, Vec<FrozenENode<'f>>>;

impl SearchSpace {
    pub fn from_frozen(frozen: FrozenEGraph, captures: &[Ir]) -> Result<Self, TypedError> {
        let roots = captures
            .iter()
            .enumerate()
            .map(|(index, capture)| {
                frozen
                    .lookup(capture)
                    .map_err(|error| TypedError::UnresolvedRoot {
                        index: Some(index),
                        reason: error.to_string(),
                    })
            })
            .collect::<Result<_, _>>()?;
        Ok(Self {
            frozen,
            roots,
            clean_hlir: false,
            cleanup_ignored: |_| Ok(false),
        })
    }

    /// Derive admissible alternatives without copying values, fields, or edges.
    /// Missing-producer pruning is a selection policy; cyclic alternatives stay
    /// in the exact snapshot and in this view when their producers still exist.
    pub fn alternatives(&self) -> Result<Alternatives<'_>, TypedError> {
        let mut alternatives: Alternatives<'_> = self
            .frozen
            .eclasses()
            .map(|class| {
                (
                    class,
                    class.nodes().filter(|node| !node.is_subsumed()).collect(),
                )
            })
            .collect();
        if self.clean_hlir {
            let mut remove = FxHashSet::default();
            for (class, nodes) in &alternatives {
                if class.sort() != Ir::sort_ref() {
                    continue;
                }
                let mut cleanable = Vec::new();
                let mut survivor = false;
                for &node in nodes {
                    let typed = self.frozen.typed_node::<Ir>(node)?;
                    if let Some((kind, _)) =
                        get_args(&typed, |kind: &OpKind, inputs: &IList| Ir::Op(kind, inputs))?
                    {
                        let FrozenValueView::EClass(kind) = self.frozen.as_view(&kind)?.view()
                        else {
                            unreachable!()
                        };
                        // The old conditional cleanup inspected only Op nodes,
                        // not direct-IR backend constructors. Preserve that
                        // distinction after those constructors become OpKinds.
                        let mut all_cleanable = false;
                        if let Some(nodes) = alternatives.get(&kind) {
                            let kinds = nodes
                                .iter()
                                .map(|&node| self.frozen.typed_node::<OpKind>(node))
                                .collect::<Result<Vec<_>, _>>()?;
                            if kinds.iter().try_fold(true, |all, kind| {
                                Ok::<_, TypedError>(all && (self.cleanup_ignored)(kind)?)
                            })? {
                                continue;
                            }
                            all_cleanable = kinds.iter().try_fold(true, |all, kind| {
                                Ok::<_, TypedError>(all && crate::hlir::cleanup(kind)?)
                            })?;
                        }
                        if all_cleanable {
                            cleanable.push(node);
                        } else {
                            survivor = true;
                        }
                    }
                }
                if survivor {
                    remove.extend(cleanable);
                }
            }
            for nodes in alternatives.values_mut() {
                nodes.retain(|node| !remove.contains(node));
            }
        }
        prune_missing_producers(&mut alternatives);
        for (index, root) in self.roots.iter().enumerate() {
            let FrozenValueView::EClass(class) = self.frozen.as_view(root)?.view() else {
                unreachable!()
            };
            if !alternatives.contains_key(&class) {
                return Err(TypedError::UnresolvedRoot {
                    index: Some(index),
                    reason: "root has no non-subsumed producer".into(),
                });
            }
        }
        Ok(alternatives)
    }
}

fn prune_missing_producers(alternatives: &mut Alternatives<'_>) {
    loop {
        let empty: FxHashSet<_> = alternatives
            .iter()
            .filter_map(|(class, nodes)| nodes.is_empty().then_some(*class))
            .collect();
        if empty.is_empty() {
            break;
        }
        alternatives.retain(|class, _| !empty.contains(class));
        for nodes in alternatives.values_mut() {
            nodes.retain(|node| node.eclass_children().all(|child| !empty.contains(&child)));
        }
    }
}

fn searchable(class: FrozenEClass<'_>) -> bool {
    let sort = class.sort();
    sort == Ir::sort_ref() || sort == IList::sort_ref() || sort == OpKind::sort_ref()
}

pub fn count_choice_sets_up_to(alternatives: &Alternatives<'_>, limit: usize) -> usize {
    alternatives
        .iter()
        .filter(|(class, _)| searchable(**class))
        .fold(usize::from(limit != 0), |count, (_, nodes)| {
            count.saturating_mul(nodes.len()).min(limit)
        })
}

/// Coarse diagnostic fingerprint of admissible IR/IList classes only.
/// Like the old serialized hash, this compares sort/producer counts and the
/// constructor-name/arity multiset. It does not identify graph topology, values,
/// or OpKind metadata, and its numeric value is not stable across versions.
pub fn hash_search_space(space: &SearchSpace) -> Result<u64, TypedError> {
    let alternatives = space.alternatives()?;
    let mut classes = Vec::new();
    let mut nodes = Vec::new();
    for (class, producers) in alternatives {
        let sort = if class.sort() == Ir::sort_ref() {
            "IR"
        } else if class.sort() == IList::sort_ref() {
            "IList"
        } else {
            continue;
        };
        classes.push((sort, producers.len()));
        nodes.extend(
            producers
                .into_iter()
                .map(|node| (node.name().to_owned(), node.args().len())),
        );
    }
    classes.sort_unstable();
    nodes.sort_unstable();
    let mut hash = DefaultHasher::new();
    classes.len().hash(&mut hash);
    classes.hash(&mut hash);
    nodes.hash(&mut hash);
    Ok(hash.finish())
}

// Preserve the old metadata-compatibility preference as visible search policy.
// This deliberately does not redefine what a typed decoder's selected node means.
fn extractor_list_len<'f>(
    frozen: &FrozenEGraph,
    mut class: FrozenEClass<'f>,
    alternatives: &Alternatives<'f>,
) -> Option<usize> {
    let mut len = 0;
    let mut seen = FxHashSet::default();
    loop {
        if !seen.insert(class) {
            return None;
        }
        if class.sort() != EList::sort_ref() && class.sort() != IList::sort_ref() {
            return Some(len);
        }
        let node = alternatives.get(&class)?.first()?;
        let is_cons = if class.sort() == EList::sort_ref() {
            get_args(
                &frozen
                    .typed_node::<EList>(*node)
                    .expect("snapshot list producer"),
                |head: &DimExpr, tail: &EList| EList::ECons(head, tail),
            )
            .expect("typed ECons selector")
            .is_some()
        } else {
            get_args(
                &frozen
                    .typed_node::<IList>(*node)
                    .expect("snapshot input-list producer"),
                |head: &Ir, tail: &IList| IList::ICons(head, tail),
            )
            .expect("typed ICons selector")
            .is_some()
        };
        if !is_cons {
            return Some(len);
        }
        len += 1;
        let Some(tail) = node.eclass_children().nth(1) else {
            return Some(len);
        };
        class = tail;
    }
}

fn opkind_metadata_consistent<'f>(
    frozen: &FrozenEGraph,
    node: FrozenENode<'f>,
    alternatives: &Alternatives<'f>,
) -> bool {
    let lengths: Vec<_> = node
        .args()
        .filter_map(|field| {
            let FrozenValueView::EClass(class) = field.view() else {
                return None;
            };
            if class.sort() == EList::sort_ref() || class.sort() == IList::sort_ref() {
                extractor_list_len(frozen, class, alternatives)
            } else {
                None
            }
        })
        .collect();
    lengths.is_empty() || lengths.iter().all(|length| *length == lengths[0])
}

pub fn random_initial_choice<'f>(
    frozen: &FrozenEGraph,
    alternatives: &Alternatives<'f>,
    rng: &mut impl Rng,
) -> EGraphChoiceSet<'f> {
    alternatives
        .iter()
        .filter(|(class, _)| searchable(**class))
        .map(|(&class, nodes)| {
            let consistent: Vec<_> = if class.sort() == OpKind::sort_ref() {
                nodes
                    .iter()
                    .copied()
                    .filter(|node| opkind_metadata_consistent(frozen, *node, alternatives))
                    .collect()
            } else {
                Vec::new()
            };
            let candidates = if consistent.is_empty() {
                nodes.as_slice()
            } else {
                &consistent
            };
            (class, candidates[rng.random_range(0..candidates.len())])
        })
        .collect()
}

/// Validate choice coverage, class membership and reachable structural IR.
/// Decoder acceptance is checked once, during reconstruction.
pub fn validate_choice_set(
    space: &SearchSpace,
    alternatives: &Alternatives<'_>,
    choices: &EGraphChoiceSet<'_>,
) -> Result<(), TypedError> {
    for (&class, nodes) in alternatives {
        if searchable(class) && choices.get(&class).is_none_or(|node| !nodes.contains(node)) {
            return Err(TypedError::Invalid(
                "choice is missing or does not belong to its admissible class".into(),
            ));
        }
    }
    for root in &space.roots {
        let FrozenValueView::EClass(class) = space.frozen.as_view(root)?.view() else {
            unreachable!()
        };
        if !choices.contains_key(&class) {
            return Err(TypedError::Invalid("root has no selected producer".into()));
        }
    }
    let mut pending = Vec::new();
    for root in &space.roots {
        let FrozenValueView::EClass(class) = space.frozen.as_view(root)?.view() else {
            unreachable!()
        };
        pending.push(class);
    }
    let mut seen = FxHashSet::default();
    while let Some(class) = pending.pop() {
        if !seen.insert(class) {
            continue;
        }
        let node = choices
            .get(&class)
            .ok_or_else(|| TypedError::Invalid("reachable class has no choice".into()))?;
        if class.sort() == Ir::sort_ref() {
            let typed = space.frozen.typed_node::<Ir>(*node)?;
            if schema::InputArgs::get_args(&typed)?.is_none()
                && schema::OutputArgs::get_args(&typed)?.is_none()
                && schema::LoopStartArgs::get_args(&typed)?.is_none()
                && schema::LoopEndArgs::get_args(&typed)?.is_none()
                && schema::OpArgs::get_args(&typed)?.is_none()
            {
                return Err(TypedError::Invalid(format!(
                    "No extractor for IR {:?}",
                    node.name()
                )));
            }
        }
        pending.extend(node.eclass_children().filter(|class| searchable(*class)));
    }
    Ok(())
}

fn choice_hash(class: FrozenEClass<'_>, node: FrozenENode<'_>) -> u64 {
    let mut hash = DefaultHasher::new();
    (class, node).hash(&mut hash);
    hash.finish()
}

pub fn hash_choice_set(choices: &EGraphChoiceSet<'_>) -> u64 {
    choices
        .iter()
        .fold(0, |hash, (&class, &node)| hash ^ choice_hash(class, node))
}

pub fn extract_reachable_generation<'f>(
    space: &'f SearchSpace,
    alternatives: &Alternatives<'f>,
    base: &EGraphChoiceSet<'f>,
    generation_size: usize,
    mutations_per_generation: usize,
    prev_selected: &mut FxHashSet<u64>,
    rng: &mut impl Rng,
) -> Vec<EGraphChoiceSet<'f>> {
    let mut pending = Vec::new();
    for root in &space.roots {
        let FrozenValueView::EClass(class) =
            space.frozen.as_view(root).expect("observed root").view()
        else {
            unreachable!()
        };
        pending.push(class);
    }
    let mut visited = FxHashSet::default();
    let mut mutable = Vec::new();
    while let Some(class) = pending.pop() {
        if !visited.insert(class) {
            continue;
        }
        if let Some(node) = base.get(&class) {
            if alternatives[&class].len() > 1 {
                mutable.push(class);
            }
            pending.extend(node.eclass_children().filter(|class| searchable(*class)));
        }
    }
    extract_generation_from_classes(
        &space.frozen,
        alternatives,
        base,
        &mutable,
        generation_size,
        mutations_per_generation,
        prev_selected,
        rng,
    )
}

/// Mutate all searchable classes, retaining the separate fuzz-test policy.
pub fn extract_generation<'f>(
    frozen: &FrozenEGraph,
    alternatives: &Alternatives<'f>,
    base: &EGraphChoiceSet<'f>,
    generation_size: usize,
    mutations_per_generation: usize,
    prev_selected: &mut FxHashSet<u64>,
    rng: &mut impl Rng,
) -> Vec<EGraphChoiceSet<'f>> {
    let mutable: Vec<_> = alternatives
        .iter()
        .filter(|(class, nodes)| searchable(**class) && nodes.len() > 1)
        .map(|(class, _)| *class)
        .collect();
    extract_generation_from_classes(
        frozen,
        alternatives,
        base,
        &mutable,
        generation_size,
        mutations_per_generation,
        prev_selected,
        rng,
    )
}

#[allow(clippy::too_many_arguments)] // Shared reachable/all-class mutation policy and its snapshot.
fn extract_generation_from_classes<'f>(
    frozen: &FrozenEGraph,
    alternatives: &Alternatives<'f>,
    base: &EGraphChoiceSet<'f>,
    mutable: &[FrozenEClass<'f>],
    generation_size: usize,
    mutations_per_generation: usize,
    prev_selected: &mut FxHashSet<u64>,
    rng: &mut impl Rng,
) -> Vec<EGraphChoiceSet<'f>> {
    let base_hash = hash_choice_set(base);
    if mutable.is_empty() {
        return if prev_selected.insert(base_hash) {
            vec![base.clone()]
        } else {
            Vec::new()
        };
    }
    let mut offspring = Vec::new();
    for _ in 0..generation_size * 100 {
        if offspring.len() >= generation_size {
            break;
        }
        let mut choices = base.clone();
        let mut hash = base_hash;
        for _ in 0..rng.random_range(1..=mutations_per_generation) {
            let class = mutable[rng.random_range(0..mutable.len())];
            let nodes = &alternatives[&class];
            let consistent: Vec<_> = if class.sort() == OpKind::sort_ref() {
                nodes
                    .iter()
                    .copied()
                    .filter(|node| opkind_metadata_consistent(frozen, *node, alternatives))
                    .collect()
            } else {
                Vec::new()
            };
            let candidates = if consistent.is_empty() {
                nodes.as_slice()
            } else {
                &consistent
            };
            let node = candidates[rng.random_range(0..candidates.len())];
            if let Some(old) = choices.insert(class, node) {
                hash ^= choice_hash(class, old);
            }
            hash ^= choice_hash(class, node);
        }
        if prev_selected.insert(hash) {
            offspring.push(choices);
        }
    }
    offspring
}

// The old serialized extractor counted constructor and scalar leaf nodes.
// Keep its path-sensitive traversal/memoization, rather than assigning one
// globally optimal producer to every metadata class.
// On equal cost, prefer an existing MNum so singleton dimensions decode without
// a variable environment; all remaining ties retain their first producer.
fn shortest_dimension<'f>(
    frozen: &FrozenEGraph,
    class: FrozenEClass<'f>,
    alternatives: Option<&Alternatives<'f>>,
    seen: &mut FxHashSet<FrozenENode<'f>>,
    cache: &mut FxHashMap<FrozenENode<'f>, Option<(usize, Vec<FrozenENode<'f>>)>>,
) -> Option<(usize, Vec<FrozenENode<'f>>)> {
    class
        .nodes()
        .filter(|node| {
            !node.is_subsumed()
                && alternatives
                    .is_none_or(|view| view.get(&class).is_some_and(|nodes| nodes.contains(node)))
        })
        .filter_map(|node| {
            if seen.contains(&node) {
                return None;
            }
            if let Some(cached) = cache.get(&node) {
                return cached.clone();
            }
            seen.insert(node);
            let mut cost = 1;
            let mut trajectory = vec![node];
            let complete = node.args().try_for_each(|field| {
                if let FrozenValueView::EClass(child) = field.view() {
                    let (child_cost, children) =
                        shortest_dimension(frozen, child, alternatives, seen, cache)?;
                    cost += child_cost;
                    trajectory.extend(children);
                } else {
                    cost += 1;
                }
                Some(())
            });
            seen.remove(&node);
            let result = complete.map(|()| (cost, trajectory));
            cache.insert(node, result.clone());
            result
        })
        .min_by_key(|(cost, nodes)| {
            let numeric = class.sort() == DimExpr::sort_ref()
                && get_args(
                    &frozen
                        .typed_node::<DimExpr>(nodes[0])
                        .expect("snapshot dimension producer"),
                    |value: &egg::I64| DimExpr::MNum(value),
                )
                .expect("typed MNum selector")
                .is_some();
            (*cost, !numeric)
        })
}

/// Candidate-local native metadata decoding and exact producer inspection.
pub struct DecodeContext<'f, 'c> {
    pub frozen: &'f FrozenEGraph,
    pub choices: &'c EGraphChoiceSet<'f>,
    metadata: Metadata,
    alternatives: Option<&'c Alternatives<'f>>,
}

impl<'f, 'c> DecodeContext<'f, 'c> {
    pub fn new(frozen: &'f FrozenEGraph, choices: &'c EGraphChoiceSet<'f>) -> Self {
        Self {
            frozen,
            choices,
            metadata: Metadata::default(),
            alternatives: None,
        }
    }

    /// Inspect active producers in native order within this reconstruction's
    /// admissible view. Reject foreign snapshots and preserve cleanup pruning.
    /// Standalone contexts have no cleanup view and expose all active producers.
    pub fn producers<S: EqualitySort>(
        &self,
        value: &S,
    ) -> Result<impl Iterator<Item = FrozenENode<'f>> + '_, TypedError> {
        let FrozenValueView::EClass(class) = self.frozen.as_view(value)?.view() else {
            return Err(TypedError::Decode(
                "expected an observed equality class".into(),
            ));
        };
        Ok(class.nodes().filter(move |node| {
            !node.is_subsumed()
                && self
                    .alternatives
                    .is_none_or(|view| view.get(&class).is_some_and(|nodes| nodes.contains(node)))
        }))
    }

    /// Resolve candidate choices; use the first producer for lists/dtypes and
    /// shortest expressions for dimensions, preferring MNum on equal cost.
    pub fn selected<S: EqualitySort>(&mut self, value: &S) -> Result<S, TypedError> {
        let FrozenValueView::EClass(class) = self.frozen.as_view(value)?.view() else {
            return Err(TypedError::Decode(
                "expected an observed equality class".into(),
            ));
        };
        let node = if searchable(class) {
            *self.choices.get(&class).ok_or_else(|| {
                TypedError::Decode("candidate is missing a producer choice".into())
            })?
        } else if class.sort() == DimExpr::sort_ref() {
            shortest_dimension(
                self.frozen,
                class,
                self.alternatives,
                &mut FxHashSet::default(),
                &mut FxHashMap::default(),
            )
            .and_then(|(_, nodes)| nodes.first().copied())
            .ok_or_else(|| TypedError::Decode("metadata has no finite producer".into()))?
        } else {
            class
                .nodes()
                .find(|node| {
                    !node.is_subsumed()
                        && self.alternatives.is_none_or(|view| {
                            view.get(&class).is_some_and(|nodes| nodes.contains(node))
                        })
                })
                .ok_or_else(|| TypedError::Decode("metadata has no producer".into()))?
        };
        let selected = self.frozen.typed_node(node)?;
        if self.frozen.as_view(&selected)? != class.value() {
            return Err(TypedError::Invalid(
                "selected producer belongs to another equality class".into(),
            ));
        }
        Ok(selected)
    }

    pub fn expression(&mut self, value: &DimExpr) -> Result<Expression, TypedError> {
        let FrozenValueView::EClass(class) = self.frozen.as_view(value)?.view() else {
            return Err(TypedError::Decode("expected an expression class".into()));
        };
        if let Some(expression) = self.metadata.dimensions.get(value) {
            return Ok(*expression);
        }
        let (_, trajectory) = shortest_dimension(
            self.frozen,
            class,
            self.alternatives,
            &mut FxHashSet::default(),
            &mut FxHashMap::default(),
        )
        .ok_or_else(|| TypedError::Decode("metadata has no finite producer".into()))?;
        fn decode<'f>(
            frozen: &'f FrozenEGraph,
            trajectory: &mut impl Iterator<Item = FrozenENode<'f>>,
        ) -> Result<Expression, TypedError> {
            let node = frozen.typed_node::<DimExpr>(
                trajectory.next().expect("complete expression trajectory"),
            )?;
            if let Some((number,)) = get_args(&node, |n: &egg::I64| DimExpr::MNum(n))? {
                return Ok(Expression::from(i64::try_from(number)?));
            }
            if let Some((name,)) = get_args(&node, |name: &egg::String| DimExpr::MVar(name))? {
                let name = String::try_from(name)?;
                return name
                    .chars()
                    .next()
                    .map(Expression::from)
                    .ok_or_else(|| TypedError::Decode("empty dimension name".into()));
            }
            if get_args(&node, DimExpr::MIter)?.is_some() {
                return Ok(Expression::from('z'));
            }
            macro_rules! binary {
                ($selector:expr, $convert:expr) => {
                    if get_args(&node, $selector)?.is_some() {
                        let left = decode(frozen, trajectory)?;
                        let right = decode(frozen, trajectory)?;
                        return Ok(($convert)(left, right));
                    }
                };
            }
            binary!(
                |a: &DimExpr, b: &DimExpr| a + b,
                |a: Expression, b: Expression| a + b
            );
            binary!(
                |a: &DimExpr, b: &DimExpr| a - b,
                |a: Expression, b: Expression| a - b
            );
            binary!(
                |a: &DimExpr, b: &DimExpr| a * b,
                |a: Expression, b: Expression| a * b
            );
            binary!(
                |a: &DimExpr, b: &DimExpr| a / b,
                |a: Expression, b: Expression| a / b
            );
            binary!(
                |a: &DimExpr, b: &DimExpr| a % b,
                |a: Expression, b: Expression| a % b
            );
            binary!(
                |a: &DimExpr, b: &DimExpr| DimExpr::MMin(a, b),
                |a: Expression, b: Expression| a.min(b)
            );
            binary!(
                |a: &DimExpr, b: &DimExpr| DimExpr::MMax(a, b),
                |a: Expression, b: Expression| a.max(b)
            );
            binary!(
                |a: &DimExpr, b: &DimExpr| DimExpr::MAnd(a, b),
                |a: Expression, b: Expression| a & b
            );
            binary!(
                |a: &DimExpr, b: &DimExpr| DimExpr::MOr(a, b),
                |a: Expression, b: Expression| a | b
            );
            binary!(
                |a: &DimExpr, b: &DimExpr| DimExpr::MGte(a, b),
                |a: Expression, b: Expression| a.gte(b)
            );
            binary!(
                |a: &DimExpr, b: &DimExpr| DimExpr::MLt(a, b),
                |a: Expression, b: Expression| a.lt(b)
            );
            binary!(
                |a: &DimExpr, b: &DimExpr| DimExpr::MCeilDiv(a, b),
                |a: Expression, b: Expression| a.ceil_div(b)
            );
            binary!(
                |a: &DimExpr, b: &DimExpr| DimExpr::MFloorTo(a, b),
                |a: Expression, b: Expression| a / b * b
            );
            Err(TypedError::Decode(
                "unsupported finite dimension constructor".into(),
            ))
        }
        let result = decode(self.frozen, &mut trajectory.into_iter())?;
        self.metadata.dimensions.insert(value.clone(), result);
        Ok(result)
    }

    pub fn expressions(&mut self, value: &EList) -> Result<Vec<Expression>, TypedError> {
        self.frozen.as_view(value)?;
        if let Some(values) = self.metadata.lists.get(value) {
            return Ok(values.clone());
        }
        let mut current = value.clone();
        let mut output = Vec::new();
        loop {
            let node = self.selected(&current)?;
            if get_args(&node, EList::ENil)?.is_some() {
                break;
            }
            let Some((head, tail)) = get_args(&node, |head: &DimExpr, tail: &EList| {
                EList::ECons(head, tail)
            })?
            else {
                return Err(TypedError::Decode(
                    "unresolved expression list transformation".into(),
                ));
            };
            output.push(self.expression(&head)?);
            current = tail;
        }
        self.metadata.lists.insert(value.clone(), output.clone());
        Ok(output)
    }

    pub fn dtype(&mut self, value: &IrDType) -> Result<crate::dtype::DType, TypedError> {
        crate::dtype::DType::try_from(self.selected(value)?)
    }

    pub fn inputs(&mut self, value: &IList) -> Result<Vec<Ir>, TypedError> {
        let mut current = value.clone();
        let mut output = Vec::new();
        loop {
            let node = self.selected(&current)?;
            if get_args(&node, IList::INil)?.is_some() {
                return Ok(output);
            }
            let Some((head, tail)) =
                get_args(&node, |head: &Ir, tail: &IList| IList::ICons(head, tail))?
            else {
                return Err(TypedError::Decode("invalid input list constructor".into()));
            };
            output.push(head);
            current = tail;
        }
    }
}

/// Reconstruct the original candidate forest, then connect its decoded edges.
pub fn reconstruct(
    space: &SearchSpace,
    choices: &EGraphChoiceSet<'_>,
    ops: &[DecodeFn],
    custom_ops: &[Box<dyn CustomOp>],
) -> Result<(LLIRGraph, Vec<NodeIndex>), TypedError> {
    let alternatives = space.alternatives()?;
    let mut context = DecodeContext::new(&space.frozen, choices);
    context.alternatives = Some(&alternatives);
    let mut graph = LLIRGraph::default();
    let mut producers = FxHashMap::default();
    let mut edges = Vec::new();
    // Discover reachability before decoders replace dependencies. Keep unused
    // original producers and never materialize an otherwise-unreachable input.
    let mut pending = Vec::new();
    for root in &space.roots {
        let FrozenValueView::EClass(class) = space.frozen.as_view(root)?.view() else {
            unreachable!()
        };
        pending.push(class);
    }
    let mut reachable = FxHashSet::default();
    let mut ir_nodes = Vec::new();
    while let Some(class) = pending.pop() {
        if !reachable.insert(class) {
            continue;
        }
        let node = *choices
            .get(&class)
            .ok_or_else(|| TypedError::Decode("reachable class has no selected producer".into()))?;
        pending.extend(node.eclass_children().filter(|class| searchable(*class)));
        if class.sort() == Ir::sort_ref() {
            ir_nodes.push(space.frozen.typed_node::<Ir>(node)?);
        }
    }
    for node in ir_nodes {
        let (op, inputs) = if let Some((index, label, dtype)) = get_args(
            &node,
            |index: &egg::I64, label: &egg::String, dtype: &IrDType| Ir::Input(index, label, dtype),
        )? {
            let index = usize::try_from(i64::try_from(&index)?)
                .map_err(|error| TypedError::Decode(error.to_string()))?;
            (
                LLIROp::new::<Input>(Box::new(Input {
                    node: index,
                    label: String::try_from(&label)?,
                    dtype: context.dtype(&dtype)?,
                })),
                Vec::new(),
            )
        } else if let Some((input, index)) = get_args(&node, |input: &Ir, index: &egg::I64| {
            Ir::Output(input, index)
        })? {
            let index = usize::try_from(i64::try_from(&index)?)
                .map_err(|error| TypedError::Decode(error.to_string()))?;
            (
                LLIROp::new::<Output>(Box::new(Output { node: index })),
                vec![context.selected(&input)?],
            )
        } else if let Some((input, loop_id, slot_idx, iters, dtype)) = get_args(
            &node,
            |input: &Ir,
             loop_id: &egg::I64,
             slot_idx: &egg::I64,
             iters: &DimExpr,
             dtype: &IrDType| Ir::LoopStart(input, loop_id, slot_idx, iters, dtype),
        )? {
            let loop_id = usize::try_from(i64::try_from(&loop_id)?)
                .map_err(|error| TypedError::Decode(error.to_string()))?;
            let slot_idx = usize::try_from(i64::try_from(&slot_idx)?)
                .map_err(|error| TypedError::Decode(error.to_string()))?;
            (
                LLIROp::new::<LoopStart>(Box::new(LoopStart {
                    loop_id,
                    slot_idx,
                    iters: context.expression(&iters)?,
                    dtype: context.dtype(&dtype)?,
                })),
                vec![context.selected(&input)?],
            )
        } else if let Some((input, loop_id, slot_idx, dtype)) = get_args(
            &node,
            |input: &Ir, loop_id: &egg::I64, slot_idx: &egg::I64, dtype: &IrDType| {
                Ir::LoopEnd(input, loop_id, slot_idx, dtype)
            },
        )? {
            let loop_id = usize::try_from(i64::try_from(&loop_id)?)
                .map_err(|error| TypedError::Decode(error.to_string()))?;
            let slot_idx = usize::try_from(i64::try_from(&slot_idx)?)
                .map_err(|error| TypedError::Decode(error.to_string()))?;
            (
                LLIROp::new::<LoopEnd>(Box::new(LoopEnd {
                    loop_id,
                    slot_idx,
                    dtype: context.dtype(&dtype)?,
                })),
                vec![context.selected(&input)?],
            )
        } else if let Some((kind, inputs)) =
            get_args(&node, |kind: &OpKind, inputs: &IList| Ir::Op(kind, inputs))?
        {
            let FrozenValueView::EClass(kind_class) = context.frozen.as_view(&kind)?.view() else {
                unreachable!()
            };
            // Prefer a metadata-compatible OpKind. The decoder inspects exactly
            // this chosen node.
            let nodes = &alternatives[&kind_class];
            let chosen = choices
                .get(&kind_class)
                .copied()
                .filter(|node| opkind_metadata_consistent(context.frozen, *node, &alternatives))
                .or_else(|| {
                    nodes.iter().copied().find(|node| {
                        opkind_metadata_consistent(context.frozen, *node, &alternatives)
                    })
                })
                .or_else(|| nodes.first().copied())
                .ok_or_else(|| TypedError::Decode("OpKind has no producer".into()))?;
            let kind = context.frozen.typed_node::<OpKind>(chosen)?;
            let mut inputs = context
                .inputs(&inputs)?
                .iter()
                .map(|input| context.selected(input))
                .collect::<Result<Vec<_>, _>>()?;
            if let Some((index, _)) = get_args(&kind, |index: &egg::I64, dtype: &IrDType| {
                OpKind::CustomOpKind(index, dtype)
            })? {
                let index = usize::try_from(i64::try_from(&index)?)
                    .map_err(|error| TypedError::Decode(error.to_string()))?;
                (
                    custom_ops
                        .get(index)
                        .ok_or_else(|| {
                            TypedError::Decode("custom op index is out of range".into())
                        })?
                        .to_llir_op(),
                    inputs,
                )
            } else {
                let mut decoded = None;
                for decoder in ops {
                    if let Some(op) = decoder(&kind, &mut inputs, &mut context)? {
                        decoded = Some(op);
                        break;
                    }
                }
                let op = decoded.ok_or_else(|| {
                    TypedError::Decode(format!("no decoder accepted {:?}", chosen.name()))
                })?;
                (op, inputs)
            }
        } else {
            return Err(TypedError::Decode(
                "unsupported structural IR constructor".into(),
            ));
        };
        let destination = graph.add_node(op);
        producers.insert(node, destination);
        for input in inputs {
            edges.push((input, destination));
        }
    }
    for (input, destination) in edges {
        let source = *producers.get(&input).ok_or_else(|| {
            TypedError::Decode(
                "decoded dependency producer was not reachable in the original candidate".into(),
            )
        })?;
        graph.add_edge(source, destination, ());
    }
    let roots = space
        .roots
        .iter()
        .map(|root| Ok(producers[&context.selected(root)?]))
        .collect::<Result<_, TypedError>>()?;
    Ok((graph, roots))
}

#[test]
fn decode_producer_inspection_preserves_cleanup_and_snapshot_ownership() -> Result<(), TypedError> {
    let root = let_("root", Ir::Op(OpKind::Constant(1.0), IList::INil()));
    let mut live = EGraph::default();
    live.register(union(
        &root,
        Ir::Op(OpKind::LoopInput(7, 9, IrDType::F32()), IList::INil()),
    ))?;
    let foreign = live.freeze()?.lookup(&root)?;
    let mut space = SearchSpace::from_frozen(live.freeze()?, &[root])?;
    space.clean_hlir = true;
    let alternatives = space.alternatives()?;
    let choices = Default::default();
    let mut context = DecodeContext::new(&space.frozen, &choices);
    assert_eq!(context.producers(&space.roots[0])?.count(), 2);
    context.alternatives = Some(&alternatives);
    assert_eq!(context.producers(&space.roots[0])?.count(), 1);
    assert!(context.producers(&foreign).is_err());
    Ok(())
}
