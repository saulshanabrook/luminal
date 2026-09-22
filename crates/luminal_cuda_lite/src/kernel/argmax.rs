//! Fused argmax over the last axis.
//!
//! Matches the frontend `argmax(axis)` decomposition (src/frontend/unary.rs):
//!
//! ```text
//! mx      = Max(x, axis)                       broadcast back over axis
//! ne      = Cast(F32)(LessThan(x, mx)) + Cast(F32)(LessThan(mx, x))
//! eq      = Cast(Bool)(ne * -1 + 1)
//! one_hot = Cast(Int)(eq)
//! out     = Max(one_hot * Iota(cols), axis)
//! ```
//!
//! and unions a single kernel into the final Max's eclass. Output dtype is
//! Int, exactly like the decomposed chain. Tie rule preserved: the highest
//! index among equal maxima wins (the decomposition takes a max over
//! `index * one_hot`). Used for GPU-side greedy sampling: the host reads one
//! i32 per row instead of a vocab-sized logit row.

use egglog_experimental::typed::{self as typed, TypedError, builtins as egg};
use luminal::egglog_utils::DecodeContext;
use luminal::typed_reference::schema::{DimExpr, EList, IList, Ir, OpKind};
use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device, cuda_dtype,
    kernel::{KernelOp, hlir::dtype_includes},
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{op::*, prelude::*};

// Wide blocks: one block scans a whole vocab row (≈128K elements), so
// maximize per-block parallelism.
const TPB: usize = 1024;

#[derive(Default, Debug, Clone)]
pub struct KernelArgmax {
    out_shape: Vec<Expression>,
    cols: Expression,
    dtype: DType,
}

pub(crate) mod kernel_argmax {
    use crate::typed_rules::*;
    use egglog_experimental::typed::builtins as egg;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_specialize(
        out_shape: &EList,
        cols: &DimExpr,
        out: &Ir,
        cols2: &DimExpr,
        oh_size: &DimExpr,
        eq_size: &DimExpr,
        lt1_size: &DimExpr,
        lt2_size: &DimExpr,
        lt_shape: &EList,
        x_strides1: &EList,
        mx_strides1: &EList,
        lt1_out_strides: &EList,
        x: &Ir,
        lt_shape2: &EList,
        mx_strides2: &EList,
        x_strides2: &EList,
        lt2_out_strides: &EList,
        dt: &DType,
    ) -> Vec<Rule> {
        let am = &Ir::Op(kernel_argmax(out_shape, cols, dt), [x]);
        let iota = &Ir::Op(OpKind::Iota(DimExpr::MIter(), cols2), IList::INil());
        let one = &Ir::Op(kernel_constant(1.0, DType::F32()), IList::INil());
        let negone = &Ir::Op(kernel_constant(-1.0, DType::F32()), IList::INil());
        let mx = &Ir::Op(
            MaxArgs {
                iter_stride: DimExpr::MIter(),
                ..MaxArgs::fresh()
            },
            [x],
        );
        let lt1 = &Ir::Op(
            OpKind::LessThan(lt_shape, x_strides1, mx_strides1, lt1_out_strides),
            [x, mx],
        );
        let lt2 = &Ir::Op(
            OpKind::LessThan(lt_shape2, mx_strides2, x_strides2, lt2_out_strides),
            [mx, x],
        );
        let lt1f = &Ir::Op(OpKind::Cast(lt1_size, DType::F32()), [lt1]);
        let lt2f = &Ir::Op(OpKind::Cast(lt2_size, DType::F32()), [lt2]);
        let ne = &Ir::Op(AddArgs::fresh(), [lt1f, lt2f]);
        let neg = &Ir::Op(MulArgs::fresh(), [ne, negone]);
        let plus1 = &Ir::Op(AddArgs::fresh(), [neg, one]);
        let equal_mask = &Ir::Op(OpKind::Cast(eq_size, DType::Bool()), [plus1]);
        let one_hot = &Ir::Op(OpKind::Cast(oh_size, DType::Int()), [equal_mask]);
        let prod = &Ir::Op(MulArgs::fresh(), [one_hot, iota]);
        vec![
            rule(
                (
                    eq(
                        out,
                        Ir::Op(
                            MaxArgs {
                                shape: out_shape.into(),
                                iters: cols.into(),
                                iter_stride: DimExpr::MIter(),
                                ..MaxArgs::fresh()
                            },
                            [prod],
                        ),
                    ),
                    eq(dt, x.dtype()),
                ),
                (am, union(out, am), set(am.dtype(), DType::Int())),
            )
            .label("kernel argmax last axis"),
        ]
    }
}

impl KernelArgmax {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_shape, field_cols, field_dtype)) = typed::get_args(
            kind,
            |shape: &EList, cols: &DimExpr, dtype: &luminal::typed_reference::schema::DType| {
                crate::typed_rules::kernel_argmax(shape, cols, dtype)
            },
        )?
        else {
            return Ok(None);
        };
        let decoded_shape = context.expressions(&field_shape)?;
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            out_shape: decoded_shape,
            cols: context.expression(&field_cols)?,
            dtype: context.dtype(&field_dtype)?,
        })
            as Box<dyn KernelOp>)))
    }
}

impl KernelOp for KernelArgmax {
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
        let vars = self
            .out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.cols.dyn_vars())
            .collect::<FxHashSet<_>>();
        let ty = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = crate::kernel::hlir::generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let cols = self.cols.to_kernel();
        let n_rows: Expression = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(1);

        // Tie rule: highest index wins, matching both the decomposed chain
        // (max over index*one_hot) and the CPU sampler's `max_by` (last max).
        let kernel = format!(
            "{includes}
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define NEG_INF_F __int_as_float(0xff800000)
{dyn_defines}
extern \"C\" {{
    __global__ void argmax_k(int *out, const {ty} *in{dyn_dims_param}) {{
        __shared__ float warp_vals[{TPB} / WARP_SIZE];
        __shared__ int warp_idxs[{TPB} / WARP_SIZE];
        long long const_z = blockIdx.x;
        long long cols = {cols};
        int tid = threadIdx.x;
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        const {ty} *row = in + const_z * cols;
        float best = NEG_INF_F;
        int best_idx = 0;
        for (long long i = tid; i < cols; i += {TPB}) {{
            float v = (float)row[i];
            if (v > best || (v == best && (int)i > best_idx)) {{
                best = v;
                best_idx = (int)i;
            }}
        }}

        #pragma unroll
        for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
            float ov = __shfl_down_sync(FULL_MASK, best, s);
            int oi = __shfl_down_sync(FULL_MASK, best_idx, s);
            if (ov > best || (ov == best && oi > best_idx)) {{
                best = ov;
                best_idx = oi;
            }}
        }}
        if (lane_id == 0) {{
            warp_vals[warp_id] = best;
            warp_idxs[warp_id] = best_idx;
        }}
        __syncthreads();

        if (warp_id == 0) {{
            int cnt = {TPB} / WARP_SIZE;
            best = tid < cnt ? warp_vals[tid] : NEG_INF_F;
            best_idx = tid < cnt ? warp_idxs[tid] : 0;
            #pragma unroll
            for (int s = cnt / 2; s > 0; s /= 2) {{
                float ov = __shfl_down_sync(FULL_MASK, best, s);
                int oi = __shfl_down_sync(FULL_MASK, best_idx, s);
                if (ov > best || (ov == best && oi > best_idx)) {{
                    best = ov;
                    best_idx = oi;
                }}
            }}
            if (tid == 0) {{
                out[const_z] = best_idx;
            }}
        }}
    }}
}}"
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("argmax_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_rows, 1.into(), 1.into()),
            (TPB.into(), 1.into(), 1.into()),
            64.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(1)
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn output_dtype(&self) -> DType {
        DType::Int
    }

    fn bytes_loaded(&self) -> Expression {
        (self.output_size() * self.cols * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.output_size() * self.cols
    }

    fn kernel_name(&self) -> &'static str {
        "Argmax"
    }
}
