//! Run in a separate integration-test process: resetting the global host
//! expression interner must not race other tests that use Expression handles.
use egglog_experimental::typed::prelude::*;
use luminal::{egglog_utils::SearchSpace, shape::Expression, typed_reference::schema::*};
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn frozen_metadata_can_be_reconstructed_after_host_interner_reset() -> Result<(), TypedError> {
    let input = Ir::Input(0, "input", DType::F32());
    let root = let_(
        "output",
        Ir::Op(
            OpKind::Sin([3], [DimExpr::MIter()], [DimExpr::MIter()]),
            [input],
        ),
    );
    let mut live = EGraph::new(EGraphOptions::default());
    live.register(&root)?;
    let space = SearchSpace::from_frozen(live.freeze()?, &[root])?;
    let mut rng = StdRng::seed_from_u64(1);
    let first = space.reconstruct_reference(&mut rng, &[])?;
    assert_eq!(first.graph.node_count(), 2);
    drop(first);
    Expression::clear_interner();
    // Debugging a snapshot must not touch its invalidated hidden host handles.
    assert!(format!("{space:?}").contains("SearchSpace"));
    let rebuilt = space.reconstruct_reference(&mut rng, &[])?;
    assert_eq!(rebuilt.graph.node_count(), 2);
    // Formatting the prepared kernel reads its reconstructed host expressions.
    let prepared = rebuilt
        .graph
        .node_weights()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    assert!(
        prepared
            .iter()
            .any(|text| text.contains("Sin") && text.contains('3'))
    );
    Ok(())
}
