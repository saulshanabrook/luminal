//! Grouped MoE expert GEMV: per-(token, top-k slot) matrix-vector products
//! against the selected expert's weights, read directly from the full expert
//! weight tensor via on-device routing indices.
//!
//! `out[s, k, o] = Σ_d x[s, (k,) d] · W[topk[s,k], o, d]`
//!
//! The egglog rewrite matches the pure-HLIR gather-experts spelling the MoE
//! examples use (and that the GLUMoE host-op rules also match):
//!
//! ```text
//! flat_idx = topk·(O·D) + iota(z, (O, D))          ; expert offset + within
//! gathered = Gather(flat_idx, W[E,O,D])  [s,k,O,D] ; BF16
//! f32      = Cast(F32)(gathered)
//! mm       = Sum_d(Mul(x, f32^T))        [s,k,1,O] ; batched mat-vec
//! ```
//!
//! and unions this kernel into the matmul's eclass. Compared to extracting
//! that chain directly, the kernel streams each selected expert's BF16 rows
//! exactly once (no [s,k,O,D] gathered intermediate, no BF16→F32 weight-cast
//! round-trip), reads the routing indices on-device, and captures into the
//! decode CUDA graph — unlike the GLUMoE host op, which copies the routing
//! tensors to the host and drives a per-expert cuBLASLt loop with a stream
//! sync per layer.
//!
//! Both x layouts are handled by one rule: the x-operand strides over the
//! mul shape `[s,k,1,O,D]` are carried into the op, so a shared `(s, D)`
//! input (gate_up; k-stride 0) and a per-slot `(s, k, D)` input (down)
//! differ only in the embedded stride constants.

use egglog_experimental::typed::{self as typed, TypedError, builtins as egg};
use luminal::egglog_utils::DecodeContext;
use luminal::typed_reference::schema::{DimExpr, EList, IList, Ir, OpKind};
use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device,
    kernel::{KernelOp, hlir::dtype_includes, hlir::generate_dyn_dims_defines},
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{op::*, prelude::*};

const TPB: usize = 256;
const WARPS_PER_BLOCK: usize = TPB / 32;

#[derive(Default, Debug, Clone)]
pub struct KernelMoEGemv {
    /// Matmul output shape `[s, k, 1, O]` (s may be dynamic).
    out_shape: Vec<Expression>,
    /// Reduce dimension (= D, the expert row length).
    k_dim: Expression,
    /// Expert weight tensor shape `[E, O, D]`.
    w_shape: Vec<Expression>,
    /// x-operand strides over the mul shape `[s, k, 1, O, D]`.
    x_strides: Vec<Expression>,
    /// topk-operand strides over `(s, k)`. NOT necessarily contiguous: the
    /// routing indices are typically a `..k` slice VIEW of the full `(s, E)`
    /// argsort output, so the row stride is E, not k. Reading the buffer as
    /// contiguous (s, k) silently picks bottom-ranked experts for rows ≥ 1.
    topk_strides: Vec<Expression>,
}

pub(crate) mod kernel_mo_e_gemv {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    // Expert flat index = topk * (O*D) + within-expert iota. The topk strides
    // describe a sliced (s,E) argsort view. Gather BF16 weights, widen to F32,
    // then reduce D in (E,O,D); the transposed view reads O with z*D and D
    // with z, while the activation broadcasts over O.

    #[ruleset]
    pub(crate) fn matmul_backend(
        mb_a: &EList,
        topk_idx: &Ir,
        ts_s: &DimExpr,
        ts_k: &DimExpr,
        iota_within_range: &DimExpr,
        g_data_shape: &EList,
        weights: &Ir,
        f32_size: &DimExpr,
        x_strides: &EList,
        w_strides: &EList,
        x: &Ir,
        out_shape: &EList,
        k_dim: &DimExpr,
        mm: &Ir,
        e: &DimExpr,
        o: &DimExpr,
        d: &DimExpr,
        s: &DimExpr,
        k: &DimExpr,
        o2: &DimExpr,
        ws_s: &DimExpr,
        ws_k: &DimExpr,
        ws_1: &DimExpr,
        d3: &DimExpr,
        xs_s: &DimExpr,
        xs_k: &DimExpr,
        xs_1: &DimExpr,
    ) -> Vec<Rule> {
        let kmoe = &Ir::Op(
            kernel_mo_e_gemv(out_shape, k_dim, g_data_shape, x_strides, mb_a),
            [x, topk_idx, weights],
        );
        let iota_base = &Ir::Op(IotaArgs::fresh(), IList::INil());
        let iota_within = &Ir::Op(
            OpKind::Iota(DimExpr::MIter(), iota_within_range),
            IList::INil(),
        );
        let mul_base = &Ir::Op(
            MulArgs {
                a_strides: mb_a.into(),
                ..MulArgs::fresh()
            },
            [topk_idx, iota_base],
        );
        let add_idx = &Ir::Op(AddArgs::fresh(), [mul_base, iota_within]);
        let gathered = &Ir::Op(
            GatherArgs {
                data_shape: g_data_shape.into(),
                ..GatherArgs::fresh()
            },
            [add_idx, weights],
        );
        let f32 = &Ir::Op(OpKind::Cast(f32_size, DType::F32()), [gathered]);
        let mm_mul = &Ir::Op(
            MulArgs {
                a_strides: x_strides.into(),
                b_strides: w_strides.into(),
                ..MulArgs::fresh()
            },
            [x, f32],
        );
        vec![
            rule(
                (
                    eq(mb_a, [ts_s, ts_k]),
                    eq(DType::Bf16(), weights.dtype()),
                    eq(DType::F32(), x.dtype()),
                    eq(
                        mm,
                        Ir::Op(
                            SumArgs {
                                shape: out_shape.into(),
                                iters: k_dim.into(),
                                ..SumArgs::fresh()
                            },
                            [mm_mul],
                        ),
                    ),
                    eq(g_data_shape, [e, o, d]),
                    eq(d, k_dim),
                    eq(out_shape, [s, k, &(DimExpr::MNum(1)), o2]),
                    eq(o, o2),
                    eq(
                        w_strides,
                        [
                            ws_s,
                            ws_k,
                            ws_1,
                            &(DimExpr::MIter() * d3),
                            &(DimExpr::MIter()),
                        ],
                    ),
                    eq(d3, d),
                    eq(
                        x_strides,
                        [xs_s, xs_k, xs_1, &(DimExpr::MNum(0)), &(DimExpr::MIter())],
                    ),
                ),
                (kmoe, union(mm, kmoe), set(kmoe.dtype(), DType::F32())),
            )
            .label("kernel moe gemv from expert gather"),
        ]
    }
}

impl KernelMoEGemv {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((
            field_out_shape,
            field_k_dim,
            field_w_shape,
            field_x_strides,
            field_topk_strides,
        )) = typed::get_args(
            kind,
            |out_shape: &EList,
             k_dim: &DimExpr,
             w_shape: &EList,
             x_strides: &EList,
             topk_strides: &EList| {
                crate::typed_rules::kernel_mo_e_gemv(
                    out_shape,
                    k_dim,
                    w_shape,
                    x_strides,
                    topk_strides,
                )
            },
        )?
        else {
            return Ok(None);
        };
        let decoded_out_shape = context.expressions(&field_out_shape)?;
        let decoded_w_shape = context.expressions(&field_w_shape)?;
        let decoded_x_strides = context.expressions(&field_x_strides)?;
        let decoded_topk_strides = context.expressions(&field_topk_strides)?;
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            out_shape: decoded_out_shape,
            k_dim: context.expression(&field_k_dim)?,
            w_shape: decoded_w_shape,
            x_strides: decoded_x_strides,
            topk_strides: decoded_topk_strides,
        })
            as Box<dyn KernelOp>)))
    }
}

impl KernelMoEGemv {
    /// (s, top_k, d_out, d_in, n_experts, x s-stride, x k-stride,
    /// topk s-stride, topk k-stride)
    #[allow(clippy::type_complexity)]
    fn dims(
        &self,
    ) -> (
        Expression,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    ) {
        let s = self.out_shape[0];
        let top_k = self.out_shape[1].to_usize().expect("MoE top_k is static");
        let d_out = self.out_shape[3].to_usize().expect("MoE d_out is static");
        let d_in = self.k_dim.to_usize().expect("MoE d_in is static");
        let n_experts = self.w_shape[0].to_usize().expect("MoE E is static");
        let resolve = |e: Expression| -> usize {
            e.substitute('z', Expression::from(1usize))
                .simplify()
                .to_usize()
                .expect("MoE x stride is static")
        };
        let xs_s = resolve(self.x_strides[0]);
        let xs_k = resolve(self.x_strides[1]);
        let ts_s = resolve(self.topk_strides[0]);
        let ts_k = resolve(self.topk_strides[1]);
        (s, top_k, d_out, d_in, n_experts, xs_s, xs_k, ts_s, ts_k)
    }
}

impl KernelOp for KernelMoEGemv {
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
        let (s, k, o, d, e, xs_s, xs_k, ts_s, ts_k) = self.dims();
        let includes = dtype_includes(&[DType::Bf16]);

        let vars: FxHashSet<char> = s.dyn_vars().into_iter().collect();
        let (dyn_defines, _sorted) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let s_expr = s.to_kernel();

        // 16-byte lane loads (8 bf16 + 2×float4) when D is 8-aligned.
        let body = if d % 8 == 0 {
            format!(
                r#"
        float acc = 0.0f;
        for (int i = lane * 8; i < {d}; i += 32 * 8) {{
            uint4 wv = __ldcs((const uint4*)(wr + i));
            const __nv_bfloat16* we = (const __nv_bfloat16*)&wv;
            float4 xv0 = *(const float4*)(xr + i);
            float4 xv1 = *(const float4*)(xr + i + 4);
            acc += (float)we[0] * xv0.x + (float)we[1] * xv0.y
                 + (float)we[2] * xv0.z + (float)we[3] * xv0.w
                 + (float)we[4] * xv1.x + (float)we[5] * xv1.y
                 + (float)we[6] * xv1.z + (float)we[7] * xv1.w;
        }}"#
            )
        } else {
            format!(
                r#"
        float acc = 0.0f;
        for (int i = lane; i < {d}; i += 32) {{
            acc += (float)wr[i] * xr[i];
        }}"#
            )
        };

        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void moe_gemv_k(float *out, const float *x, const int *topk, const __nv_bfloat16 *w{dyn_dims_param}) {{
        const long long total = (long long)({s_expr}) * {k} * {o};
        long long gw = (long long)blockIdx.x * {WARPS_PER_BLOCK} + (threadIdx.x >> 5);
        if (gw >= total) return;
        int lane = threadIdx.x & 31;
        int o_idx = (int)(gw % {o});
        long long sk = gw / {o};
        long long s_idx = sk / {k};
        long long kk = sk % {k};

        int expert = topk[s_idx * {ts_s} + kk * {ts_k}];
        if (expert < 0 || expert >= {e}) {{
            if (lane == 0) out[gw] = 0.0f;
            return;
        }}
        const __nv_bfloat16* wr = w + ((long long)expert * {o} + o_idx) * {d};
        const float* xr = x + s_idx * {xs_s} + kk * {xs_k};
{body}

        #pragma unroll
        for (int sft = 16; sft > 0; sft >>= 1) {{
            acc += __shfl_down_sync(0xffffffff, acc, sft);
        }}
        if (lane == 0) {{
            out[gw] = acc;
        }}
    }}
}}"
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("moe_gemv_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        let (s, k, o, ..) = self.dims();
        (
            func,
            module,
            kernel,
            (
                (s * k * o).ceil_div(WARPS_PER_BLOCK),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            (
                Expression::from(TPB),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            Expression::from(0usize),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        let (s, k, o, ..) = self.dims();
        s * k * o
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn output_dtype(&self) -> DType {
        DType::F32
    }

    fn all_dyn_vars(&self) -> FxHashSet<char> {
        self.out_shape[0].dyn_vars().into_iter().collect()
    }

    fn bytes_loaded(&self) -> Expression {
        // Each (s, k) slot streams one expert's O×D BF16 rows plus its input
        // row and routing index.
        let (s, k, o, d, ..) = self.dims();
        s * k * (o * d * 2 + d * 4 + 4)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        let (s, k, o, d, ..) = self.dims();
        s * k * o * d * 2
    }

    fn kernel_name(&self) -> &'static str {
        "MoEGemv"
    }
}
