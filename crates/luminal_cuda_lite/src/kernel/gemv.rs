//! Warp-per-row GEMV for M=1 decode matmuls.
//!
//! Matches the same row-major × column-major `GenericMatmul` pattern as the
//! cuBLASLt RmCm rewrite, restricted to m == 1 and 16-bit dtypes, and unions
//! a lean kernel into the same eclass — the GA search then picks cuBLASLt or
//! this per shape. Rationale (measured on Llama 3 8B decode): the nvjet GEMV
//! kernels carry ~5µs of fixed launch/tail cost per call, which dominates the
//! small projections (o-proj: 13.2µs vs an 8.3µs traffic floor). One warp per
//! output row with vectorized 16-byte loads and F32 accumulation has ~1µs of
//! fixed cost and no splitK reduction pass.

use egglog_experimental::typed::{self as typed, TypedError, builtins as egg};
use luminal::egglog_utils::DecodeContext;
use luminal::typed_reference::schema::{DimExpr, EList, IList, Ir, OpKind};
use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device, cuda_dtype,
    kernel::{KernelOp, hlir::dtype_includes, hlir::generate_dyn_dims_defines},
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
#[allow(unused_imports)]
use luminal::dtype::DType;
use luminal::{op::*, prelude::*};

const TPB: usize = 256;
const WARPS_PER_BLOCK: usize = TPB / 32;

#[derive(Default, Debug, Clone)]
pub struct KernelGemv {
    n: Expression,
    k: Expression,
    dtype: DType,
}

pub(crate) mod kernel_gemv {
    use crate::typed_rules::*;
    use egglog_experimental::typed::builtins as egg;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    // Row-major activation [1,k], column-major [k,n] weights stored as [n,k].
    // Each 16-bit dtype has literal-m=1 and interval-pinned dynamic variants:
    // lower/upper have no rows for literals.

    #[ruleset]
    pub(crate) fn matmul_backend(
        a: &Ir,
        b: &Ir,
        sum: &Ir,
        n: &DimExpr,
        a_m_stride: &DimExpr,
        a_n_stride: &DimExpr,
        a_k_stride: &DimExpr,
        b_m_stride: &DimExpr,
        b_n_stride: &DimExpr,
        b_k_stride: &DimExpr,
        dt: &DType,
        m: &DimExpr,
        m_lower: &egg::I64,
        m_upper: &egg::I64,
    ) -> Vec<Rule> {
        let generic_args = GenericMatmulArgs::fresh();
        let variants = [
            (
                "static",
                vec![eq(&generic_args.out_shape, [&DimExpr::MNum(1), n])],
            ),
            (
                "dyn",
                vec![
                    eq(&generic_args.out_shape, [m, n]),
                    eq(m_lower, m.lower()),
                    eq(m_upper, m.upper()),
                    m_lower.gt(0_i64),
                    m_upper.lt(2_i64),
                ],
            ),
        ];
        let mut rules = Vec::new();
        for (variant, shape_conditions) in variants {
            for (name, dtype) in [("Bf16", DType::Bf16()), ("F16", DType::F16())] {
                let gemv = &Ir::Op(kernel_gemv(n, &generic_args.k, &dtype), [a, b]);
                rules.push(
                    rule(
                        (
                            eq(sum, Ir::Op(generic_args.clone(), [a, b])),
                            &shape_conditions,
                            ne(n, 0),
                            ne(&generic_args.k, 1),
                            eq(
                                &generic_args.lhs_strides,
                                [a_m_stride, a_n_stride, a_k_stride],
                            ),
                            eq(
                                &generic_args.rhs_strides,
                                [b_m_stride, b_n_stride, b_k_stride],
                            ),
                            eq(&generic_args.sum_iter_stride, DimExpr::MIter()),
                            eq(a_m_stride, DimExpr::MIter() * &generic_args.k),
                            eq(a_n_stride, 0),
                            eq(a_k_stride, DimExpr::MIter()),
                            eq(b_m_stride, 0),
                            eq(b_n_stride, DimExpr::MIter() * &generic_args.k),
                            eq(b_k_stride, DimExpr::MIter()),
                            eq(dt, a.dtype()),
                            eq(dt, b.dtype()),
                            eq(dt, &dtype),
                        ),
                        (gemv, union(sum, gemv), set(gemv.dtype(), &dtype)),
                    )
                    .label(format!("kernel gemv m1 {name} {variant}")),
                );
            }
        }
        rules
    }
}

impl KernelGemv {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_n, field_k, field_dtype)) = typed::get_args(
            kind,
            |n: &DimExpr, k: &DimExpr, dtype: &luminal::typed_reference::schema::DType| {
                crate::typed_rules::kernel_gemv(n, k, dtype)
            },
        )?
        else {
            return Ok(None);
        };
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            n: context.expression(&field_n)?,
            k: context.expression(&field_k)?,
            dtype: context.dtype(&field_dtype)?,
        })
            as Box<dyn KernelOp>)))
    }
}

impl KernelOp for KernelGemv {
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
            .n
            .dyn_vars()
            .into_iter()
            .chain(self.k.dyn_vars())
            .collect::<FxHashSet<_>>();
        let ty = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n = self.n.to_kernel();
        let k = self.k.to_kernel();

        // 16-byte vectorized lane loads (8 × 16-bit) when K is statically a
        // multiple of 8; scalar loop otherwise. F32 accumulation throughout.
        let vectorized = self.k.to_usize().map(|k| k % 8 == 0).unwrap_or(false);
        let body = if vectorized {
            format!(
                r#"
        const uint4* xr = (const uint4*)x;
        const uint4* wr = (const uint4*)(w + row * ({k}));
        long long chunks = ({k}) / 8;
        float acc = 0.0f;
        for (long long c = lane; c < chunks; c += 32) {{
            uint4 xv = xr[c];
            uint4 wv = wr[c];
            const {ty}* xe = (const {ty}*)&xv;
            const {ty}* we = (const {ty}*)&wv;
            #pragma unroll
            for (int e = 0; e < 8; e++) {{
                acc += (float)xe[e] * (float)we[e];
            }}
        }}"#
            )
        } else {
            format!(
                r#"
        const {ty}* wr = w + row * ({k});
        float acc = 0.0f;
        for (long long i = lane; i < ({k}); i += 32) {{
            acc += (float)x[i] * (float)wr[i];
        }}"#
            )
        };

        let kernel = format!(
            "{includes}
#define FULL_MASK 0xffffffff
{dyn_defines}
extern \"C\" {{
    __global__ void gemv_k({ty} *out, const {ty} *x, const {ty} *w{dyn_dims_param}) {{
        long long row = (long long)blockIdx.x * {WARPS_PER_BLOCK} + (threadIdx.x >> 5);
        if (row >= ({n})) return;
        int lane = threadIdx.x & 31;
{body}

        #pragma unroll
        for (int s = 16; s > 0; s /= 2) {{
            acc += __shfl_down_sync(FULL_MASK, acc, s);
        }}
        if (lane == 0) {{
            out[row] = ({ty})acc;
        }}
    }}
}}"
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("gemv_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (self.n.ceil_div(WARPS_PER_BLOCK), 1.into(), 1.into()),
            (TPB.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.n
    }

    fn output_bytes(&self) -> Expression {
        (self.n * self.dtype.bits()).ceil_div(8)
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn bytes_loaded(&self) -> Expression {
        ((self.n * self.k + self.k) * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.n * self.k * 2
    }

    fn kernel_name(&self) -> &'static str {
        "Gemv"
    }
}

// ═══════════════════════════════════════════════════════════
// Tensor-core scaled FP8 GEMV:
//   out_bf16[n] = (Σ_k x_f8[k]·w_f8[n,k]) · in_scale·w_scale
//
// Matches the scaled-fp8 linear chain (KernelQuantF8 activation →
// GenericMatmul → Cast(F32) → ×(in_scale·w_scale) → Cast(Bf16)) at m == 1
// and fuses GEMM + dequant + output cast into one mma.sync kernel.
//
// Why tensor cores: a CUDA-core fp8 GEMV is cvt-ALU-bound (fp8→f16→f32 per
// element) at ~2.2× the traffic floor; mma.sync.m16n8k32.e4m3 contracts
// 32 fp8 per instruction with f32 accumulators, leaving the kernel purely
// bandwidth-bound (75-83% of floor measured vs nvjet's 51-67%).
//
// Layout (the M=1 k-permutation trick): the contraction only requires the A
// and B fragments to carry the SAME physical k in matching slots, so each
// lane feeds the mma from plain contiguous 16-byte loads — no ldmatrix, no
// shared-memory staging. Lane l of a warp covers row n = l/4 and k-slot
// group lg = l%4: per 128-k step it loads two uint4 at
// w[n][k0 + 32·lg (+16)], x replicated across A rows (D is row-replicated;
// lanes 0-3 hold the result row). 8 warps per block split K over the same
// 8 rows (grid = N/8 keeps all SMs busy at small N), reduced through smem.
// ═══════════════════════════════════════════════════════════

#[derive(Default, Debug, Clone)]
pub struct KernelGemvF8 {
    n: Expression,
    k: Expression,
}

pub(crate) mod kernel_gemv_f8 {
    use crate::typed_rules::*;
    use egglog_experimental::typed::builtins as egg;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    // F8 custom ops place their quantization input scale LAST. The standalone
    // quant and fused norm/SiLU quant patterns recover it by producer arity.

    #[ruleset]
    pub(crate) fn matmul_backend(
        q_size: &DimExpr,
        x: &Ir,
        in_scale: &Ir,
        a: &Ir,
        b: &Ir,
        sum: &Ir,
        n: &DimExpr,
        a_m_stride: &DimExpr,
        a_n_stride: &DimExpr,
        a_k_stride: &DimExpr,
        b_m_stride: &DimExpr,
        b_n_stride: &DimExpr,
        b_k_stride: &DimExpr,
        c_size: &DimExpr,
        cast: &Ir,
        in_scale2: &Ir,
        w_scale: &Ir,
        scale_product: &Ir,
        scaled: &Ir,
        o_size: &DimExpr,
        out_bf16: &Ir,
        cid: &egg::I64,
        cx: &Ir,
        cw: &Ir,
        m: &DimExpr,
        m_lower: &egg::I64,
        m_upper: &egg::I64,
    ) -> Vec<Rule> {
        let generic_args = GenericMatmulArgs::fresh();
        let gemv = &Ir::Op(
            kernel_gemv_f8(n, &generic_args.k),
            [a, b, in_scale, w_scale],
        );
        let variants = [
            (
                "static",
                vec![eq(&generic_args.out_shape, [&DimExpr::MNum(1), n])],
            ),
            (
                "dyn",
                vec![
                    eq(&generic_args.out_shape, [m, n]),
                    eq(m_lower, m.lower()),
                    eq(m_upper, m.upper()),
                    m_lower.gt(0_i64),
                    m_upper.lt(2_i64),
                ],
            ),
        ];
        let mut rules = Vec::new();
        for (variant, shape_conditions) in variants {
            for (producer, source) in [
                ("quant", Ir::Op(kernel_quant_f8(q_size), [x, in_scale])),
                (
                    "custom3",
                    Ir::Op(
                        OpKind::CustomOpKind(cid, DType::F8E4M3()),
                        [cx, cw, in_scale],
                    ),
                ),
                (
                    "custom2",
                    Ir::Op(OpKind::CustomOpKind(cid, DType::F8E4M3()), [cx, in_scale]),
                ),
            ] {
                rules.push(
                    rule(
                        (
                            eq(a, source),
                            eq(sum, Ir::Op(generic_args.clone(), [a, b])),
                            &shape_conditions,
                            ne(n, 0),
                            ne(&generic_args.k, 1),
                            eq(
                                &generic_args.lhs_strides,
                                [a_m_stride, a_n_stride, a_k_stride],
                            ),
                            eq(
                                &generic_args.rhs_strides,
                                [b_m_stride, b_n_stride, b_k_stride],
                            ),
                            eq(&generic_args.sum_iter_stride, DimExpr::MIter()),
                            eq(a_m_stride, DimExpr::MIter() * &generic_args.k),
                            eq(a_n_stride, 0),
                            eq(a_k_stride, DimExpr::MIter()),
                            eq(b_m_stride, 0),
                            eq(b_n_stride, DimExpr::MIter() * &generic_args.k),
                            eq(b_k_stride, DimExpr::MIter()),
                            eq(DType::F8E4M3(), b.dtype()),
                            eq(cast, Ir::Op(OpKind::Cast(c_size, DType::F32()), [sum])),
                            eq(
                                scale_product,
                                Ir::Op(
                                    OpKind::Mul(
                                        EList::ENil(),
                                        EList::ENil(),
                                        EList::ENil(),
                                        EList::ENil(),
                                    ),
                                    [in_scale2, w_scale],
                                ),
                            ),
                            eq(in_scale, in_scale2),
                            eq(
                                scaled,
                                Ir::Op(
                                    MulArgs {
                                        b_strides: ([0, 0]).into(),
                                        ..MulArgs::fresh()
                                    },
                                    [cast, scale_product],
                                ),
                            ),
                            eq(
                                out_bf16,
                                Ir::Op(OpKind::Cast(o_size, DType::Bf16()), [scaled]),
                            ),
                        ),
                        (
                            gemv,
                            union(out_bf16, gemv),
                            set(gemv.dtype(), DType::Bf16()),
                        ),
                    )
                    .label(format!(
                        "kernel gemv f8 tensor core m1 {variant} {producer}"
                    )),
                );
            }
        }
        rules
    }
}

impl KernelGemvF8 {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_n, field_k)) = typed::get_args(kind, |n: &DimExpr, k: &DimExpr| {
            crate::typed_rules::kernel_gemv_f8(n, k)
        })?
        else {
            return Ok(None);
        };
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            n: context.expression(&field_n)?,
            k: context.expression(&field_k)?,
        })
            as Box<dyn KernelOp>)))
    }
}

const GEMV_F8_NWARPS: usize = 8;

impl KernelOp for KernelGemvF8 {
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
            .n
            .dyn_vars()
            .into_iter()
            .chain(self.k.dyn_vars())
            .collect::<FxHashSet<_>>();
        let includes = dtype_includes(&[DType::Bf16, DType::F8E4M3]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n = self.n.to_kernel();
        let k = self.k.to_kernel();

        // Aligned shapes (k % 128 == 0, n % 8 == 0 — every llama projection)
        // take the branch-free loop; otherwise loads are bounds-padded with
        // zero bytes (fp8 0x00 == 0.0 contributes nothing) and stores are
        // predicated.
        let aligned = self.k.to_usize().map(|k| k % 128 == 0).unwrap_or(false)
            && self.n.to_usize().map(|n| n % 8 == 0).unwrap_or(false);

        let load_w = if aligned {
            "uint4 wv = __ldcs((const uint4*)(wrow + k0 + 16 * v));"
        } else {
            "uint4 wv = make_uint4(0, 0, 0, 0);
                if (row_valid) {
                    long long rem = ({k}) - (k0 + 16 * v);
                    if (rem >= 16) {
                        wv = __ldcs((const uint4*)(wrow + k0 + 16 * v));
                    } else if (rem > 0) {
                        unsigned char* wb = (unsigned char*)&wv;
                        for (int t = 0; t < rem; t++) wb[t] = ((const unsigned char*)wrow)[k0 + 16 * v + t];
                    }
                }"
        };
        let load_x = if aligned {
            "uint4 xv = __ldg((const uint4*)(x + k0 + 16 * v));"
        } else {
            "uint4 xv = make_uint4(0, 0, 0, 0);
                {
                    long long rem = ({k}) - (k0 + 16 * v);
                    if (rem >= 16) {
                        xv = __ldg((const uint4*)(x + k0 + 16 * v));
                    } else if (rem > 0) {
                        unsigned char* xb = (unsigned char*)&xv;
                        for (int t = 0; t < rem; t++) xb[t] = ((const unsigned char*)x)[k0 + 16 * v + t];
                    }
                }"
        };
        let store_guard = if aligned {
            ""
        } else {
            "if (row0 + lane >= ({n})) return;"
        };

        let load_w = if aligned {
            load_w.to_string()
        } else {
            load_w.replace("{k}", &k)
        };
        let load_x = if aligned {
            load_x.to_string()
        } else {
            load_x.replace("{k}", &k)
        };
        let store_guard = store_guard.replace("{n}", &n);

        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void gemv_f8_tc_k(__nv_bfloat16 *out, const __nv_fp8_e4m3 *x, const __nv_fp8_e4m3 *w, const float *in_scale, const float *w_scale{dyn_dims_param}) {{
        __shared__ float partial[{GEMV_F8_NWARPS}][8];
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int lg = lane & 3;
        const int n_in_tile = lane >> 2;
        const long long row0 = (long long)blockIdx.x * 8;
        const long long row = row0 + n_in_tile;
        const bool row_valid = row < ({n});
        const __nv_fp8_e4m3* wrow = w + (row_valid ? row : 0) * (long long)({k});

        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
        const long long kstep = 128LL * {GEMV_F8_NWARPS};
        for (long long kb = warp * 128LL; kb < ({k}); kb += kstep) {{
            const long long k0 = kb + 32 * lg;
            #pragma unroll
            for (int v = 0; v < 2; v++) {{
                {load_w}
                {load_x}
                asm volatile(
                    \"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \"
                    \"{{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};\\n\"
                    : \"+f\"(d0), \"+f\"(d1), \"+f\"(d2), \"+f\"(d3)
                    : \"r\"(xv.x), \"r\"(xv.x), \"r\"(xv.y), \"r\"(xv.y), \"r\"(wv.x), \"r\"(wv.y));
                asm volatile(
                    \"mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 \"
                    \"{{%0,%1,%2,%3}}, {{%4,%5,%6,%7}}, {{%8,%9}}, {{%0,%1,%2,%3}};\\n\"
                    : \"+f\"(d0), \"+f\"(d1), \"+f\"(d2), \"+f\"(d3)
                    : \"r\"(xv.z), \"r\"(xv.z), \"r\"(xv.w), \"r\"(xv.w), \"r\"(wv.z), \"r\"(wv.w));
            }}
        }}
        // D is row-replicated; lanes 0-3 hold result row cols {{2lg, 2lg+1}}.
        if (lane < 4) {{
            partial[warp][2 * lg] = d0;
            partial[warp][2 * lg + 1] = d1;
        }}
        __syncthreads();
        if (warp == 0 && lane < 8) {{
            {store_guard}
            float acc = 0.f;
            #pragma unroll
            for (int wp = 0; wp < {GEMV_F8_NWARPS}; wp++) acc += partial[wp][lane];
            out[row0 + lane] = (__nv_bfloat16)(acc * in_scale[0] * w_scale[0]);
        }}
    }}
}}"
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("gemv_f8_tc_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (self.n.ceil_div(8), 1.into(), 1.into()),
            ((GEMV_F8_NWARPS * 32).into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.n
    }

    fn output_bytes(&self) -> Expression {
        self.n * 2
    }

    fn output_dtype(&self) -> DType {
        DType::Bf16
    }

    fn bytes_loaded(&self) -> Expression {
        self.n * self.k + self.k + 8
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.n * self.k * 2
    }

    fn kernel_name(&self) -> &'static str {
        "GemvF8"
    }
}
