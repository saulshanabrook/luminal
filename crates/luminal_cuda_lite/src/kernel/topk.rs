//! Fused stable-argsort ranks: one kernel for the O(E²) pairwise rank
//! computation that `stable_argsort(descending)` spells as ~8 HLIR kernels
//! (two broadcast value views, three LessThans, the index tie-break chain,
//! and the comparison sum).
//!
//! `rank[s, j] = Σ_i (x[s,i] > x[s,j]) || (x[s,i] == x[s,j] && i < j)`
//!
//! The kernel reproduces the chain's exact count-based semantics (including
//! the index tie-break) and the rank→index scatter in one launch: element j
//! lands at sorted position rank(j) — bit-identical to the decomposed
//! spelling. The rewrite matches all the way through the scatter tail and
//! unions into the sorted-indices eclass emitted by
//! `stable_argsort(descending)` / `topk_indexes`.

use egglog_experimental::typed::{self as typed, TypedError, builtins as egg};
use luminal::egglog_utils::DecodeContext;
use luminal::typed_reference::schema::{DimExpr, EList, IList, Ir, OpKind};
use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device,
    kernel::{KernelOp, hlir::generate_dyn_dims_defines},
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{op::*, prelude::*};

#[derive(Default, Debug, Clone)]
pub struct KernelStableSortIdx {
    /// Output shape `(rows, E)`; rows may be dynamic.
    out_shape: Vec<Expression>,
}

pub(crate) mod kernel_stable_sort_idx {
    use crate::typed_rules::*;
    use egglog_experimental::typed::builtins as egg;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    // Stable descending rank = sum(primary + equal_value * earlier_index).
    // Three stages (pre -> late -> late) walk strongly anchored chains. One
    // monolithic ~25-atom join multiplies loosely paired binary atoms by the
    // number of layer instances; later stages instead use relation-bound values.
    #[relation]
    pub(crate) fn stable_ranks_part(a0: Ir, a1: Ir, a2: Ir, a3: Ir, a4: Ir, a5: EList);

    #[relation]
    pub(crate) fn stable_ranks(a0: Ir, a1: Ir, a2: EList);

    #[ruleset]
    pub(crate) fn kernel_fuse_late_pre(
        av_row: &DimExpr,
        x: &Ir,
        one_a: &Ir,
        a_val: &Ir,
        bv_row: &DimExpr,
        x2: &Ir,
        one_b: &Ir,
        b_val: &Ir,
        p_shape: &EList,
        p_a: &EList,
        p_b: &EList,
        p_o: &EList,
        pf_size: &DimExpr,
        eqidx: &Ir,
        out_shape: &EList,
        r_size: &DimExpr,
        ranks: &Ir,
    ) -> Vec<Rule> {
        let source = &Ir::Op(OpKind::Constant(1.0), IList::INil());
        let prim = &Ir::Op(OpKind::LessThan(p_shape, p_a, p_b, p_o), [b_val, a_val]);
        let prim_f = &Ir::Op(OpKind::Cast(pf_size, DType::F32()), [prim]);
        let cmp = &Ir::Op(AddArgs::fresh(), [prim_f, eqidx]);
        let sum = &Ir::Op(
            SumArgs {
                shape: out_shape.into(),
                ..SumArgs::fresh()
            },
            [cmp],
        );
        vec![
            rule(
                (
                    eq(
                        a_val,
                        Ir::Op(
                            MulArgs {
                                a_strides: ([av_row, &(DimExpr::MIter()), &(DimExpr::MNum(0))])
                                    .into(),
                                ..MulArgs::fresh()
                            },
                            [x, one_a],
                        ),
                    ),
                    eq(one_a, source),
                    eq(
                        b_val,
                        Ir::Op(
                            MulArgs {
                                a_strides: ([bv_row, &(DimExpr::MNum(0)), &(DimExpr::MIter())])
                                    .into(),
                                ..MulArgs::fresh()
                            },
                            [x2, one_b],
                        ),
                    ),
                    eq(x, x2),
                    eq(one_b, source),
                    eq(ranks, Ir::Op(OpKind::Cast(r_size, DType::Int()), [sum])),
                    eq(DType::F32(), x.dtype()),
                ),
                (stable_ranks_part(ranks, eqidx, a_val, b_val, x, out_shape),),
            )
            .label("stable ranks part"),
        ]
    }

    #[ruleset]
    pub(crate) fn kernel_fuse_late(
        ranks: &Ir,
        a_val: &Ir,
        b_val: &Ir,
        x: &Ir,
        out_shape: &EList,
        if_size: &DimExpr,
        ic_shape: &EList,
        ic_o: &EList,
        iota_a: &Ir,
        iota_b: &Ir,
        nl_one: &Ir,
        nl_negone: &Ir,
        ltf_size: &DimExpr,
        lt_shape: &EList,
        lt_a: &EList,
        lt_b: &EList,
        lt_o: &EList,
        a_val2: &Ir,
        b_val2: &Ir,
        ng_one: &Ir,
        ng_negone: &Ir,
        gtf_size: &DimExpr,
        gt_shape: &EList,
        gt_a: &EList,
        gt_b: &EList,
        gt_o: &EList,
        b_val3: &Ir,
        a_val3: &Ir,
        os_range: &DimExpr,
        e2: &DimExpr,
        b_range: &DimExpr,
        sorted: &Ir,
        z_range: &DimExpr,
        v_range: &DimExpr,
        rows: &DimExpr,
        e: &DimExpr,
    ) -> Vec<Rule> {
        let sorted_indices = &Ir::Op(kernel_stable_sort_idx(out_shape), [x]);
        let idx_cmp = &Ir::Op(
            OpKind::LessThan(
                ic_shape,
                [
                    &(DimExpr::MNum(0)),
                    &(DimExpr::MIter()),
                    &(DimExpr::MNum(0)),
                ],
                [
                    &(DimExpr::MNum(0)),
                    &(DimExpr::MNum(0)),
                    &(DimExpr::MIter()),
                ],
                ic_o,
            ),
            [iota_a, iota_b],
        );
        let lt = &Ir::Op(
            OpKind::LessThan(lt_shape, lt_a, lt_b, lt_o),
            [a_val2, b_val2],
        );
        let gt = &Ir::Op(
            OpKind::LessThan(gt_shape, gt_a, gt_b, gt_o),
            [b_val3, a_val3],
        );
        let one_s = &Ir::Op(OpKind::Iota(1, os_range), IList::INil());
        let base = &Ir::Op(OpKind::Iota(DimExpr::MIter() * e2, b_range), IList::INil());
        let zeros = &Ir::Op(OpKind::Iota(0, z_range), IList::INil());
        let vals = &Ir::Op(OpKind::Iota(DimExpr::MIter(), v_range), IList::INil());
        let idx_f = &Ir::Op(OpKind::Cast(if_size, DType::F32()), [idx_cmp]);
        let lt_f = &Ir::Op(OpKind::Cast(ltf_size, DType::F32()), [lt]);
        let gt_f = &Ir::Op(OpKind::Cast(gtf_size, DType::F32()), [gt]);
        let rsc = &Ir::Op(MulArgs::fresh(), [ranks, one_s]);
        let nl_neg = &Ir::Op(MulArgs::fresh(), [lt_f, nl_negone]);
        let ng_neg = &Ir::Op(MulArgs::fresh(), [gt_f, ng_negone]);
        let adj = &Ir::Op(AddArgs::fresh(), [base, rsc]);
        let not_lt = &Ir::Op(AddArgs::fresh(), [nl_neg, nl_one]);
        let not_gt = &Ir::Op(AddArgs::fresh(), [ng_neg, ng_one]);
        let val_eq = &Ir::Op(MulArgs::fresh(), [not_lt, not_gt]);
        let eqidx = &Ir::Op(MulArgs::fresh(), [val_eq, idx_f]);
        vec![
            rule(
                (
                    stable_ranks_part(ranks, eqidx, a_val, b_val, x, out_shape),
                    eq(a_val, a_val2),
                    eq(b_val, b_val2),
                    eq(a_val, a_val3),
                    eq(b_val, b_val3),
                ),
                (stable_ranks(ranks, x, out_shape),),
            )
            .label("stable ranks tiebreak"),
            rule(
                (
                    stable_ranks(ranks, x, out_shape),
                    eq(sorted, Ir::Op(ScatterArgs::fresh(), [zeros, adj, vals])),
                    eq(out_shape, [rows, e]),
                    eq(e, e2),
                ),
                (
                    sorted_indices,
                    union(sorted, sorted_indices),
                    set(sorted_indices.dtype(), DType::Int()),
                ),
            )
            .label("kernel stable ranks descending"),
        ]
    }
}

impl KernelStableSortIdx {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_out_shape,)) = typed::get_args(kind, |out_shape: &EList| {
            crate::typed_rules::kernel_stable_sort_idx(out_shape)
        })?
        else {
            return Ok(None);
        };
        let decoded_out_shape = context.expressions(&field_out_shape)?;
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            out_shape: decoded_out_shape,
        })
            as Box<dyn KernelOp>)))
    }
}

impl KernelOp for KernelStableSortIdx {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        let rows = self.out_shape[0];
        let e = self.out_shape[1].to_usize().expect("ranks E is static");
        assert!(e <= 1024, "stable ranks kernel supports E <= 1024");

        let vars: FxHashSet<char> = rows.dyn_vars().into_iter().collect();
        let (dyn_defines, _sorted) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let kernel = format!(
            "{dyn_defines}
extern \"C\" {{
    __global__ void stable_sort_idx_k(int *out, const float *x{dyn_dims_param}) {{
        __shared__ float row[{e}];
        long long r = blockIdx.x;
        int j = threadIdx.x;
        const float* xr = x + r * {e};
        if (j < {e}) {{
            row[j] = xr[j];
            // Prefill with 0 (the decomposed chain's scatter dest is a
            // materialized zeros iota). For real inputs the ranks are a
            // permutation and every slot is overwritten; with NaN/garbage
            // inputs (search-time dummy buffers) ranks collide and unwritten
            // slots would otherwise hold stale ints that downstream
            // consumers use as gather/expert indices — an OOB crash.
            out[r * {e} + j] = 0;
        }}
        __syncthreads();
        if (j >= {e}) return;
        float vj = row[j];
        int count = 0;
        for (int i = 0; i < {e}; i++) {{
            float vi = row[i];
            // descending rank with index tie-break (lower index first)
            count += (vi > vj) || (vi == vj && i < j);
        }}
        // ranks are a permutation: element j lands at sorted position count
        out[r * {e} + count] = j;
    }}
}}"
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("stable_sort_idx_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        let tpb = e.next_multiple_of(32).min(1024);
        (
            func,
            module,
            kernel,
            (rows, Expression::from(1usize), Expression::from(1usize)),
            (
                Expression::from(tpb),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            Expression::from(0usize),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn output_dtype(&self) -> DType {
        DType::Int
    }

    fn all_dyn_vars(&self) -> FxHashSet<char> {
        self.out_shape[0].dyn_vars().into_iter().collect()
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_size() * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        let e = self.out_shape[1];
        self.out_shape[0] * e * e * 2
    }

    fn kernel_name(&self) -> &'static str {
        "StableSortIdx"
    }
}
