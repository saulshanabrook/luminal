//! Fused RMSNorm: `x * rsqrt(mean(x²) + eps) * w` in one kernel.
//!
//! Replaces the decomposed norm sandwich (cast → square-mul → mean-reduce →
//! +eps → sqrt → recip → mul → weight-mul → cast: ~6-8 graph nodes) that the
//! 16-bit pipeline spells as `norm_in_f32`. Per the dtype contract the norm
//! computes in F32: the kernel loads `dtype` rows, accumulates the mean of
//! squares in F32, and rounds once at the store — the same semantics the
//! explicit-cast spelling expresses, minus the per-op intermediate roundings
//! (the decomposed path computes entirely in F32 between the casts too).
//!
//! Layout: x `(rows, cols)` contiguous in `dtype` with dynamic `rows`;
//! w `(cols,)` F32. One block per row; F32 warp + block reduction.

use egglog_experimental::typed::{self as typed, TypedError, builtins as egg};
use luminal::egglog_utils::DecodeContext;
use luminal::typed_reference::schema::{DimExpr, EList, IList, Ir, OpKind};
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    dtype::DType, op::CustomOp, op::LLIROp, prelude::FxHashMap, prelude::GraphTensor,
    shape::Expression,
};

use crate::compile_module_image_for_current_device;
use crate::kernel::KernelOp;

const TPB: usize = 1024;

#[derive(Debug, Clone)]
pub struct RMSNormKernel {
    pub rows: Expression,
    pub cols: usize,
    pub eps: f32,
    pub dtype: DType,
}

impl KernelOp for RMSNormKernel {
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
        let cols = self.cols;
        let eps = self.eps;
        let ty = crate::cuda_dtype(self.dtype);
        let includes = crate::kernel::hlir::dtype_includes(&[self.dtype]);
        let kernel = format!(
            r#"{includes}
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
extern "C" __global__ void rms_norm_k(
    {ty}* __restrict__ out,
    const {ty}* __restrict__ x,
    const float* __restrict__ w
) {{
    const int COLS = {cols};
    __shared__ float warp_sums[{TPB} / WARP_SIZE];
    long long row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    const {ty}* xr = x + row * COLS;
    {ty}* yr = out + row * COLS;

    float partial = 0.0f;
#if {cols} % 8 == 0
    {{
        const uint4* xv = (const uint4*)xr;
        for (int c = tid; c < COLS / 8; c += {TPB}) {{
            uint4 chunk = xv[c];
            const {ty}* xe = (const {ty}*)&chunk;
            #pragma unroll
            for (int e = 0; e < 8; e++) {{
                float v = (float)xe[e];
                partial += v * v;
            }}
        }}
    }}
#else
    for (int i = tid; i < COLS; i += {TPB}) {{
        float v = (float)xr[i];
        partial += v * v;
    }}
#endif

    #pragma unroll
    for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
        partial += __shfl_down_sync(FULL_MASK, partial, s);
    }}
    if (lane_id == 0) {{
        warp_sums[warp_id] = partial;
    }}
    __syncthreads();

    if (warp_id == 0) {{
        int cnt = {TPB} / WARP_SIZE;
        float block_sum = tid < cnt ? warp_sums[tid] : 0.0f;
        #pragma unroll
        for (int s = cnt / 2; s > 0; s /= 2) {{
            block_sum += __shfl_down_sync(FULL_MASK, block_sum, s);
        }}
        if (tid == 0) {{
            warp_sums[0] = rsqrtf(block_sum / (float)COLS + {eps:.10}f);
        }}
    }}
    __syncthreads();
    float rinv = warp_sums[0];

    for (int i = tid; i < COLS; i += {TPB}) {{
        yr[i] = ({ty})((float)xr[i] * rinv * w[i]);
    }}
}}
"#
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("rms_norm_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            "rms_norm_k".to_string(),
            (
                self.rows,
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
        self.rows * self.cols
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn bytes_loaded(&self) -> Expression {
        // Two passes over x plus the weight row.
        (self.rows * self.cols * self.dtype.bits() * 2).ceil_div(8) + self.cols * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.rows * self.cols * 4
    }

    fn kernel_name(&self) -> &'static str {
        "RMSNorm"
    }
}

#[derive(Debug, Clone)]
pub struct RMSNormCustom(pub RMSNormKernel);

impl CustomOp for RMSNormCustom {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(self.0.clone()) as Box<dyn KernelOp>)
    }
}

/// Fused `x * rsqrt(mean(x², last axis) + eps) * w`.
///
/// `x` is `(rows, cols)` in any float dtype (F32 accumulation inside),
/// `w` is `(cols,)` F32. Returns `(rows, cols)` in `x`'s dtype.
pub fn fused_rms_norm(x: GraphTensor, w: GraphTensor, eps: f32) -> GraphTensor {
    assert_eq!(w.dtype, DType::F32, "RMSNorm weight must be F32");
    let x_dims = x.dims();
    assert_eq!(x_dims.len(), 2, "RMSNorm x must be 2-D (rows, cols)");
    let rows = x_dims[0];
    let cols = x_dims[1].to_usize().expect("RMSNorm cols must be static");
    assert_eq!(
        w.dims()[0].to_usize().expect("RMSNorm weight dim"),
        cols,
        "RMSNorm weight length mismatch"
    );

    let kern = RMSNormKernel {
        rows,
        cols,
        eps,
        dtype: x.dtype,
    };
    let cx = unsafe { &mut *x.graph_ref };
    cx.custom_op(RMSNormCustom(kern), vec![x, w], (rows, cols), x.dtype)
}

// ═══════════════════════════════════════════════════════════
// Fused RMSNorm + static-scale FP8 quantization
//
// `q = (f8)(x · rsqrt(mean(x²) + eps) · w / in_scale)` in one kernel — the
// norm output's only consumer in the fp8 pipeline is the per-linear
// quantization, so the rounding to bf16 between them is pure overhead.
//
// CONVENTION (load-bearing for the e-graph): F8E4M3-producing custom ops put
// the quantization input scale as their LAST input. The scaled-fp8 GEMM
// rewrites (cuBLASLt + tensor-core GEMV) match `(CustomOpKind ?id (F8E4M3))`
// by arity and read the scale from that slot.
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct RMSNormQuantKernel {
    pub rows: Expression,
    pub cols: usize,
    pub eps: f32,
    /// Input dtype (16-bit); output is always F8E4M3.
    pub dtype: DType,
}

impl KernelOp for RMSNormQuantKernel {
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
        let cols = self.cols;
        let eps = self.eps;
        let ty = crate::cuda_dtype(self.dtype);
        let includes = crate::kernel::hlir::dtype_includes(&[self.dtype, DType::F8E4M3]);
        let kernel = format!(
            r#"{includes}
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
extern "C" __global__ void rms_norm_quant_k(
    __nv_fp8_e4m3* __restrict__ out,
    const {ty}* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ in_scale
) {{
    const int COLS = {cols};
    __shared__ float warp_sums[{TPB} / WARP_SIZE];
    long long row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    const {ty}* xr = x + row * COLS;
    __nv_fp8_e4m3* yr = out + row * COLS;

    float partial = 0.0f;
#if {cols} % 8 == 0
    {{
        const uint4* xv = (const uint4*)xr;
        for (int c = tid; c < COLS / 8; c += {TPB}) {{
            uint4 chunk = xv[c];
            const {ty}* xe = (const {ty}*)&chunk;
            #pragma unroll
            for (int e = 0; e < 8; e++) {{
                float v = (float)xe[e];
                partial += v * v;
            }}
        }}
    }}
#else
    for (int i = tid; i < COLS; i += {TPB}) {{
        float v = (float)xr[i];
        partial += v * v;
    }}
#endif

    #pragma unroll
    for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
        partial += __shfl_down_sync(FULL_MASK, partial, s);
    }}
    if (lane_id == 0) {{
        warp_sums[warp_id] = partial;
    }}
    __syncthreads();

    if (warp_id == 0) {{
        int cnt = {TPB} / WARP_SIZE;
        float block_sum = tid < cnt ? warp_sums[tid] : 0.0f;
        #pragma unroll
        for (int s = cnt / 2; s > 0; s /= 2) {{
            block_sum += __shfl_down_sync(FULL_MASK, block_sum, s);
        }}
        if (tid == 0) {{
            warp_sums[0] = rsqrtf(block_sum / (float)COLS + {eps:.10}f);
        }}
    }}
    __syncthreads();
    float rinv = warp_sums[0] * (1.0f / in_scale[0]);

    for (int i = tid; i < COLS; i += {TPB}) {{
        yr[i] = (__nv_fp8_e4m3)((float)xr[i] * rinv * w[i]);
    }}
}}
"#
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("rms_norm_quant_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            "rms_norm_quant_k".to_string(),
            (
                self.rows,
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
        self.rows * self.cols
    }

    fn output_bytes(&self) -> Expression {
        self.output_size()
    }

    fn output_dtype(&self) -> DType {
        DType::F8E4M3
    }

    fn bytes_loaded(&self) -> Expression {
        (self.rows * self.cols * self.dtype.bits() * 2).ceil_div(8) + self.cols * 4 + 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.rows * self.cols * 5
    }

    fn kernel_name(&self) -> &'static str {
        "RMSNormQuant"
    }
}

#[derive(Debug, Clone)]
pub struct RMSNormQuantCustom(pub RMSNormQuantKernel);

impl CustomOp for RMSNormQuantCustom {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(self.0.clone()) as Box<dyn KernelOp>)
    }
}

/// Fused `(f8)(rms_norm(x, w, eps) / in_scale)`.
///
/// Inputs `(x, w, in_scale)` — the scale is LAST per the f8 custom-op
/// convention the scaled-GEMM rewrites rely on. `x` is `(rows, cols)`
/// 16-bit, `w` `(cols,)` F32, `in_scale` a scalar F32 tensor. Returns
/// `(rows, cols)` F8E4M3.
pub fn fused_rms_norm_quant(
    x: GraphTensor,
    w: GraphTensor,
    eps: f32,
    in_scale: GraphTensor,
) -> GraphTensor {
    assert_eq!(w.dtype, DType::F32, "RMSNorm weight must be F32");
    assert_eq!(in_scale.dtype, DType::F32, "quant scale must be F32");
    let x_dims = x.dims();
    assert_eq!(x_dims.len(), 2, "RMSNorm x must be 2-D (rows, cols)");
    let rows = x_dims[0];
    let cols = x_dims[1].to_usize().expect("RMSNorm cols must be static");

    let kern = RMSNormQuantKernel {
        rows,
        cols,
        eps,
        dtype: x.dtype,
    };
    let cx = unsafe { &mut *x.graph_ref };
    cx.custom_op(
        RMSNormQuantCustom(kern),
        vec![x, w, in_scale],
        (rows, cols),
        DType::F8E4M3,
    )
}

// ═══════════════════════════════════════════════════════════
// Egglog-matched RMSNorm: unions the fused kernel into the decomposed HLIR
// chain the models spell (per the pure-HLIR rule):
//
//   Cast(F32)(x_bf16) → Mul(x,x) → Sum(last) → ×Recip(Iota(cols)) → +eps
//   → Sqrt → Recip → ×x → ×w → Cast(Bf16)
//
// Two variants: 2-D (s, cols) plain norms and 3-D (s, h, d) per-head QK
// norms. The 3-D input is a split view of a contiguous (s, h·d) buffer, so
// the same row-per-block kernel serves both with rows = product of the
// non-reduced dims.
// ═══════════════════════════════════════════════════════════

#[derive(Default, Debug, Clone)]
pub struct KernelRMSNorm {
    out_shape: Vec<Expression>,
    eps: f64,
}

pub(crate) mod kernel_r_m_s_norm {
    use crate::typed_rules::*;
    use egglog_experimental::typed::builtins as egg;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    // Stage the Sqrt/Recip/epsilon core before the weight-mul tail. The latter
    // then joins with rin/xf bound, avoiding a monolithic cross-product of
    // distinct layer instances in rolled bodies.
    #[relation]
    pub(crate) fn rms_rinv(a0: Ir, a1: Ir, a2: Ir, a3: egg::F64, a4: DimExpr);

    #[ruleset]
    pub(crate) fn kernel_fuse_late_pre(
        xf_size: &DimExpr,
        xb: &Ir,
        xf: &Ir,
        xf2: &Ir,
        cols: &DimExpr,
        nc_size: &DimExpr,
        cols3: &DimExpr,
        nc_range: &DimExpr,
        eps: &egg::F64,
        sq2_shape: &EList,
        sq2_in: &EList,
        sq2_out: &EList,
        rin: &Ir,
    ) -> Vec<Rule> {
        let sq = &Ir::Op(MulArgs::fresh(), [xf, xf2]);
        let ncst = &Ir::Op(OpKind::Iota(cols3, nc_range), IList::INil());
        let epsc = &Ir::Op(OpKind::Constant(eps), IList::INil());
        let sum = &Ir::Op(
            SumArgs {
                iters: cols.into(),
                iter_stride: DimExpr::MIter(),
                ..SumArgs::fresh()
            },
            [sq],
        );
        let ncast = &Ir::Op(OpKind::Cast(nc_size, DType::F32()), [ncst]);
        let rcpn = &Ir::Op(RecipArgs::fresh(), [ncast]);
        let mean = &Ir::Op(MulArgs::fresh(), [sum, rcpn]);
        let pe = &Ir::Op(AddArgs::fresh(), [mean, epsc]);
        let sqr = &Ir::Op(OpKind::Sqrt(sq2_shape, sq2_in, sq2_out), [pe]);
        vec![
            rule(
                (
                    eq(xf, Ir::Op(OpKind::Cast(xf_size, DType::F32()), [xb])),
                    eq(DType::Bf16(), xb.dtype()),
                    eq(xf, xf2),
                    eq(cols, cols3),
                    eq(rin, Ir::Op(RecipArgs::fresh(), [sqr])),
                ),
                (rms_rinv(rin, xf, xb, eps, cols),),
            )
            .label("rms rinv core"),
        ]
    }

    #[ruleset]
    pub(crate) fn kernel_fuse_late(
        rin: &Ir,
        xf: &Ir,
        xb: &Ir,
        eps: &egg::F64,
        cols: &DimExpr,
        xf3: &Ir,
        wg_shape: &EList,
        wg_b: &EList,
        w: &Ir,
        o_size: &DimExpr,
        out: &Ir,
        rows: &DimExpr,
        cols2: &DimExpr,
        d0: &DimExpr,
        d1: &DimExpr,
    ) -> Vec<Rule> {
        let krms = &Ir::Op(kernel_rms_norm(wg_shape, eps), [xb, w]);
        let nrm = &Ir::Op(MulArgs::fresh(), [rin, xf3]);
        let wgt = &Ir::Op(
            MulArgs {
                shape: wg_shape.into(),
                b_strides: wg_b.into(),
                ..MulArgs::fresh()
            },
            [nrm, w],
        );
        vec![
            rule(
                (
                    rms_rinv(rin, xf, xb, eps, cols),
                    eq(xf, xf3),
                    eq(DType::F32(), w.dtype()),
                    eq(out, Ir::Op(OpKind::Cast(o_size, DType::Bf16()), [wgt])),
                    eq(wg_shape, [rows, cols2]),
                    eq(cols, cols2),
                    eq(wg_b, [&(DimExpr::MNum(0)), &(DimExpr::MIter())]),
                ),
                (krms, union(out, krms), set(krms.dtype(), DType::Bf16())),
            )
            .label("kernel rms norm bf16 2d"),
            rule(
                (
                    rms_rinv(rin, xf, xb, eps, cols),
                    eq(xf, xf3),
                    eq(DType::F32(), w.dtype()),
                    eq(out, Ir::Op(OpKind::Cast(o_size, DType::Bf16()), [wgt])),
                    eq(wg_shape, [d0, d1, cols2]),
                    eq(cols, cols2),
                    eq(
                        wg_b,
                        [
                            &(DimExpr::MNum(0)),
                            &(DimExpr::MNum(0)),
                            &(DimExpr::MIter()),
                        ],
                    ),
                ),
                (krms, union(out, krms), set(krms.dtype(), DType::Bf16())),
            )
            .label("kernel rms norm bf16 3d"),
        ]
    }
}

impl KernelRMSNorm {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_out_shape, field_eps)) =
            typed::get_args(kind, |out_shape: &EList, eps: &egg::F64| {
                crate::typed_rules::kernel_rms_norm(out_shape, eps)
            })?
        else {
            return Ok(None);
        };
        let decoded_out_shape = context.expressions(&field_out_shape)?;
        let out_shape = decoded_out_shape;
        let eps: f64 = f64::try_from(&field_eps)?;
        let cols = out_shape
            .last()
            .and_then(|c| c.to_usize())
            .ok_or_else(|| TypedError::Decode("RMSNorm cols must be static".into()))?;
        let rows = out_shape[..out_shape.len() - 1]
            .iter()
            .copied()
            .product::<Expression>();
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(RMSNormKernel {
            rows,
            cols,
            eps: eps as f32,
            dtype: DType::Bf16,
        })
            as Box<dyn KernelOp>)))
    }
}

// ═══════════════════════════════════════════════════════════
// Egglog-matched fused RMSNorm + fp8 quant: the rms_rinv staged core plus
// the weight-mul tail and the per-linear quant spelling
// `cast_f8(norm_bf16_f32 · recip(scale))`, re-fused into the existing
// RMSNormQuantKernel (norm + quantize in one pass).
// ═══════════════════════════════════════════════════════════

#[derive(Default, Debug, Clone)]
pub struct KernelRMSNormQuant {
    out_shape: Vec<Expression>,
    eps: f64,
}

pub(crate) mod kernel_r_m_s_norm_quant {
    use super::kernel_r_m_s_norm::rms_rinv;
    use crate::typed_rules::*;
    use egglog_experimental::typed::builtins as egg;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_fuse_late(
        rin: &Ir,
        xf: &Ir,
        xb: &Ir,
        eps: &egg::F64,
        cols: &DimExpr,
        xf3: &Ir,
        wg_shape: &EList,
        wg_b: &EList,
        w: &Ir,
        o_size: &DimExpr,
        rows: &DimExpr,
        cols2: &DimExpr,
        qf_size: &DimExpr,
        scale: &Ir,
        q_size: &DimExpr,
        q: &Ir,
    ) -> Vec<Rule> {
        let krq = &Ir::Op(kernel_rms_norm_quant(wg_shape, eps), [xb, w, scale]);
        let nrm = &Ir::Op(MulArgs::fresh(), [rin, xf3]);
        let qrecip = &Ir::Op(RecipArgs::fresh(), [scale]);
        let wgt = &Ir::Op(
            MulArgs {
                shape: wg_shape.into(),
                b_strides: wg_b.into(),
                ..MulArgs::fresh()
            },
            [nrm, w],
        );
        let out = &Ir::Op(OpKind::Cast(o_size, DType::Bf16()), [wgt]);
        let qf = &Ir::Op(OpKind::Cast(qf_size, DType::F32()), [out]);
        let qmul = &Ir::Op(MulArgs::fresh(), [qf, qrecip]);
        vec![
            rule(
                (
                    rms_rinv(rin, xf, xb, eps, cols),
                    eq(xf, xf3),
                    eq(DType::F32(), w.dtype()),
                    eq(wg_shape, [rows, cols2]),
                    eq(cols, cols2),
                    eq(wg_b, [&(DimExpr::MNum(0)), &(DimExpr::MIter())]),
                    eq(q, Ir::Op(OpKind::Cast(q_size, DType::F8E4M3()), [qmul])),
                ),
                (krq, union(q, krq), set(krq.dtype(), DType::F8E4M3())),
            )
            .label("kernel rms norm quant f8"),
        ]
    }
}

impl KernelRMSNormQuant {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_out_shape, field_eps)) =
            typed::get_args(kind, |out_shape: &EList, eps: &egg::F64| {
                crate::typed_rules::kernel_rms_norm_quant(out_shape, eps)
            })?
        else {
            return Ok(None);
        };
        let decoded_out_shape = context.expressions(&field_out_shape)?;
        let out_shape = decoded_out_shape;
        let eps: f64 = f64::try_from(&field_eps)?;
        let kern = RMSNormQuantKernel {
            rows: out_shape[0],
            cols: out_shape[1]
                .to_usize()
                .ok_or_else(|| TypedError::Decode("rms cols must be static".into()))?,
            eps: eps as f32,
            dtype: DType::Bf16,
        };
        Ok(Some(LLIROp::new::<dyn KernelOp>(
            Box::new(kern) as Box<dyn KernelOp>
        )))
    }
}
