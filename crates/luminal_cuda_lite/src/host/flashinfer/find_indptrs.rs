//! Recover the compact row-index dependency of an already-selected FlashInfer op.

use egglog_experimental::typed::{TypedError, builtins as egg, get_args};
use luminal::egglog_utils::DecodeContext;
use luminal::typed_reference::schema::{
    AddArgs, DType, EList, IList, InputArgs, Ir, MulArgs, OpArgs, OpKind,
};

/// Walk Add/Mul inputs and return the first recoverable compact index.
/// `flat` is the candidate's selected producer; internal producer ties use the
/// admissible snapshot's native order, without searching additional branches.
pub(crate) fn try_find_compact_gather_idx(
    context: &DecodeContext<'_, '_>,
    flat: &Ir,
) -> Result<Option<Ir>, TypedError> {
    let Some(add_inputs) = logical_binary_inputs(
        context,
        flat,
        |kind| Ok(AddArgs::get_args(kind)?.is_some()),
        "Add",
    )?
    else {
        return Ok(None);
    };
    for input in add_inputs {
        let Some(mul_inputs) = logical_binary_inputs(
            context,
            &input,
            |kind| Ok(MulArgs::get_args(kind)?.is_some()),
            "Mul",
        )?
        else {
            continue;
        };
        for input in mul_inputs {
            for node in context.producers(&input)? {
                let node = context.frozen.typed_node::<Ir>(node)?;
                if get_args(
                    &node,
                    |id: &egg::I64, label: &egg::String, dtype: &DType| Ir::Input(id, label, dtype),
                )?
                .is_some()
                {
                    return Ok(Some(node));
                }
            }
        }
    }
    Ok(None)
}

fn walk_ilist(context: &DecodeContext<'_, '_>, inputs: &IList) -> Result<Vec<Ir>, TypedError> {
    let mut result = Vec::new();
    let mut current = inputs.clone();
    loop {
        let node = context
            .producers(&current)?
            .next()
            .expect("admissible input list producer");
        let node = context.frozen.typed_node::<IList>(node)?;
        let Some((head, tail)) =
            get_args(&node, |head: &Ir, tail: &IList| IList::ICons(head, tail))?
        else {
            // The old walk also stopped at non-ICons list transformations.
            return Ok(result);
        };
        let mut nodes = context.producers(&head)?;
        let first = nodes.next().expect("admissible IR producer");
        let mut selected = context.frozen.typed_node::<Ir>(first)?;
        for node in std::iter::once(first).chain(nodes) {
            let node = context.frozen.typed_node::<Ir>(node)?;
            if OpArgs::get_args(&node)?.is_some() || InputArgs::get_args(&node)?.is_some() {
                selected = node;
                break;
            }
        }
        result.push(selected);
        current = tail;
    }
}

fn logical_binary_inputs(
    context: &DecodeContext<'_, '_>,
    node: &Ir,
    matches_hlir: impl Fn(&OpKind) -> Result<bool, TypedError>,
    op_name: &str,
) -> Result<Option<Vec<Ir>>, TypedError> {
    for candidate in context.producers(node)? {
        let candidate = context.frozen.typed_node::<Ir>(candidate)?;
        let Some((kind, inputs)) = get_args(&candidate, |kind: &OpKind, inputs: &IList| {
            Ir::Op(kind, inputs)
        })?
        else {
            continue;
        };
        let kind = context
            .producers(&kind)?
            .next()
            .expect("admissible OpKind producer");
        if matches_hlir(&context.frozen.typed_node::<OpKind>(kind)?)? {
            // Selecting an HLIR list commits to this branch, even if it later
            // contains no recoverable compact input.
            return Ok(Some(walk_ilist(context, &inputs)?));
        }
    }

    // No HLIR match: inspect only the supplied producer, then the first fused
    // element if it is a FusionEnd. Do not search other lowered alternatives.
    let Some((kind, inputs)) =
        get_args(node, |kind: &OpKind, inputs: &IList| Ir::Op(kind, inputs))?
    else {
        return Ok(None);
    };
    let kind = context.frozen.typed_node::<OpKind>(
        context
            .producers(&kind)?
            .next()
            .expect("admissible OpKind producer"),
    )?;
    let (kind, inputs) = if get_args(&kind, |shape: &EList, strides: &EList, dtype: &DType| {
        crate::typed_rules::fusion_end(shape, strides, dtype)
    })?
    .is_some()
    {
        let Some(element) = walk_ilist(context, &inputs)?.into_iter().next() else {
            return Ok(None);
        };
        let Some((kind, inputs)) = get_args(&element, |kind: &OpKind, inputs: &IList| {
            Ir::Op(kind, inputs)
        })?
        else {
            return Ok(None);
        };
        let kind = context.frozen.typed_node::<OpKind>(
            context
                .producers(&kind)?
                .next()
                .expect("admissible OpKind producer"),
        )?;
        (kind, inputs)
    } else {
        (kind, inputs)
    };
    let Some((opcode, _, _, _, _, _)) = get_args(
        &kind,
        |opcode: &egg::String, shape: &EList, a: &EList, b: &EList, out: &EList, dtype: &DType| {
            crate::typed_rules::cuda_binary_elementwise(opcode, shape, a, b, out, dtype)
        },
    )?
    else {
        return Ok(None);
    };
    if String::try_from(&opcode)? != op_name {
        return Ok(None);
    }
    walk_ilist(context, &inputs)?
        .into_iter()
        .map(|input| unwrap_fusion_start(context, input))
        .collect::<Result<Vec<_>, _>>()
        .map(Some)
}

fn unwrap_fusion_start(context: &DecodeContext<'_, '_>, node: Ir) -> Result<Ir, TypedError> {
    let Some((kind, inputs)) =
        get_args(&node, |kind: &OpKind, inputs: &IList| Ir::Op(kind, inputs))?
    else {
        return Ok(node);
    };
    let kind = context.frozen.typed_node::<OpKind>(
        context
            .producers(&kind)?
            .next()
            .expect("admissible OpKind producer"),
    )?;
    if get_args(&kind, |shape: &EList, strides: &EList, dtype: &DType| {
        crate::typed_rules::fusion_start(shape, strides, dtype)
    })?
    .is_none()
    {
        return Ok(node);
    }
    Ok(walk_ilist(context, &inputs)?
        .into_iter()
        .next()
        .unwrap_or(node))
}
