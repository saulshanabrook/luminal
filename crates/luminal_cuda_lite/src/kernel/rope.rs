//! Fused RoPE (rotary position embedding) — interleaved-pair convention.
//!
//! Replaces flux2's 6-op RoPE chain (split / slice / squeeze / neg / concat /
//! merge_dims / 4× cast / mul / add) with a single kernel launch per call.
//! ~120 RoPE calls per forward pass at full DiT depth.
//!
//! Convention: `repeat_interleave_real=True` (Flux 2 / diffusers), so adjacent
//! dim pairs rotate together. For an input `[a0, b0, a1, b1, ...]` and per-
//! position `(cos, sin)`, the output is
//!   `out[2j]   = x[2j]   * cos[2j]   - x[2j+1] * sin[2j]`
//!   `out[2j+1] = x[2j+1] * cos[2j+1] + x[2j]   * sin[2j+1]`
//!
//! Layout: x `(S, H, D)`, cos/sin `(S, D)` (broadcast across H).

use egglog_experimental::typed::{self as typed, TypedError, builtins as egg};
use luminal::egglog_utils::DecodeContext;
use luminal::typed_reference::schema::{DimExpr, EList, IList, Ir, OpKind};
use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    dtype::DType, op::CustomOp, op::LLIROp, prelude::FxHashMap, prelude::FxHashSet,
    prelude::GraphTensor, shape::Expression,
};

use crate::compile_module_image_for_current_device;
use crate::kernel::KernelOp;

#[derive(Debug, Clone)]
pub struct RoPEKernel {
    pub s: usize,
    pub h: usize,
    pub d: usize,
}

const TPB: usize = 64;

impl KernelOp for RoPEKernel {
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
        let s = self.s;
        let h = self.h;
        let d = self.d;
        assert!(d.is_multiple_of(2), "RoPE head_dim must be even");
        let kernel = format!(
            r#"
extern "C" __global__ void rope_kernel(
    float* __restrict__ out,
    const float* __restrict__ x,
    const float* __restrict__ cos_,
    const float* __restrict__ sin_
) {{
    const int S = {s};
    const int H = {h};
    const int D = {d};
    int sh = blockIdx.x;       // 0..S*H
    int s_idx = sh / H;
    int tid = threadIdx.x;

    const float* xr   = x    + sh    * D;
    const float* cosr = cos_ + s_idx * D;
    const float* sinr = sin_ + s_idx * D;
    float* yr = out + sh * D;

    for (int i = tid; i < D; i += {TPB}) {{
        float xi = xr[i];
        float xpair;
        if ((i & 1) == 0) {{
            // even: paired with i+1, rotated value is -x[i+1]
            xpair = -xr[i + 1];
        }} else {{
            // odd: paired with i-1, rotated value is +x[i-1]
            xpair = xr[i - 1];
        }}
        yr[i] = xi * cosr[i] + xpair * sinr[i];
    }}
}}
"#
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("rope_kernel").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            "rope_kernel".to_string(),
            (
                Expression::from(s * h),
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
        Expression::from(self.s * self.h * self.d)
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn output_dtype(&self) -> DType {
        DType::F32
    }

    fn bytes_loaded(&self) -> Expression {
        // x: full (S,H,D); cos/sin: (S,D) read H times each but cached.
        Expression::from(self.s * self.h * self.d * 4 + self.s * self.d * 4 * 2)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        // 4 per output element (mul, neg/load, mul, add).
        Expression::from(self.s * self.h * self.d * 4)
    }

    fn kernel_name(&self) -> &'static str {
        "RoPE"
    }
}

#[derive(Debug, Clone)]
pub struct RoPECustom(pub RoPEKernel);

impl CustomOp for RoPECustom {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(self.0.clone()) as Box<dyn KernelOp>)
    }
}

/// Apply RoPE: `x` shape `(S, H, D)` F32, `cos`/`sin` shape `(S, D)` F32.
/// Returns `(S, H, D)` F32.
pub fn apply_rope(x: GraphTensor, cos: GraphTensor, sin: GraphTensor) -> GraphTensor {
    assert_eq!(x.dtype, DType::F32, "RoPE x must be F32");
    let cos = if cos.dtype == DType::F32 {
        cos
    } else {
        cos.cast(DType::F32)
    };
    let sin = if sin.dtype == DType::F32 {
        sin
    } else {
        sin.cast(DType::F32)
    };
    let x_dims = x.dims();
    assert_eq!(x_dims.len(), 3, "RoPE x must be 3-D (S, H, D)");
    let s = x_dims[0].to_usize().expect("RoPE: S must be static");
    let h = x_dims[1].to_usize().expect("RoPE: H must be static");
    let d = x_dims[2].to_usize().expect("RoPE: D must be static");
    let cos_dims = cos.dims();
    let sin_dims = sin.dims();
    assert_eq!(cos_dims.len(), 2, "RoPE cos must be 2-D (S, D)");
    assert_eq!(sin_dims.len(), 2, "RoPE sin must be 2-D (S, D)");
    assert_eq!(cos_dims[0].to_usize().unwrap(), s, "RoPE cos S mismatch");
    assert_eq!(cos_dims[1].to_usize().unwrap(), d, "RoPE cos D mismatch");
    assert_eq!(sin_dims[0].to_usize().unwrap(), s, "RoPE sin S mismatch");
    assert_eq!(sin_dims[1].to_usize().unwrap(), d, "RoPE sin D mismatch");

    let kern = RoPEKernel { s, h, d };
    let cx = unsafe { &mut *x.graph_ref };
    cx.custom_op(RoPECustom(kern), vec![x, cos, sin], (s, h, d), DType::F32)
}

// ═══════════════════════════════════════════════════════════
// Half-rotation RoPE (Llama 3 convention), dtype-aware, dynamic S.
//
// Rotates the two halves of each head: for j in 0..D/2
//   out[j]       = x[j]       * cos[j] - x[j + D/2] * sin[j]
//   out[j + D/2] = x[j + D/2] * cos[j] + x[j]       * sin[j]
//
// The input is read from a row of a (possibly wider) projection output via
// `pitch` (row stride in elements) and `offset` (column offset), so q and k
// can be roped straight out of a fused QKV GEMM without materializing
// slices. cos/sin are (S, D/2) F32; x/out are `dtype` (F32 or 16-bit, math
// in F32). S is dynamic: the kernel derives everything from blockIdx, so
// only the grid expression carries the dyn dim.
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct RoPEHalfKernel {
    pub s: Expression,
    pub h: usize,
    pub d: usize,
    /// Input row stride in elements (e.g. 6144 for a fused QKV row).
    pub pitch: usize,
    /// Column offset of this head group within the input row.
    pub offset: usize,
    pub dtype: DType,
}

impl KernelOp for RoPEHalfKernel {
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
        let h = self.h;
        let d = self.d;
        let pitch = self.pitch;
        let offset = self.offset;
        assert!(d.is_multiple_of(2), "RoPE head_dim must be even");
        let half = d / 2;
        let ty = crate::cuda_dtype(self.dtype);
        let includes = crate::kernel::hlir::dtype_includes(&[self.dtype]);
        let kernel = format!(
            r#"{includes}
extern "C" __global__ void rope_half_kernel(
    {ty}* __restrict__ out,
    const {ty}* __restrict__ x,
    const float* __restrict__ cos_,
    const float* __restrict__ sin_
) {{
    const int H = {h};
    const int D = {d};
    const int HALF = {half};
    int sh = blockIdx.x;       // 0..S*H
    int s_idx = sh / H;
    int h_idx = sh - s_idx * H;
    int tid = threadIdx.x;

    const {ty}* xr   = x    + (long long)s_idx * {pitch} + {offset} + h_idx * D;
    const float* cosr = cos_ + (long long)s_idx * HALF;
    const float* sinr = sin_ + (long long)s_idx * HALF;
    {ty}* yr = out + (long long)sh * D;

    for (int j = tid; j < HALF; j += {TPB}) {{
        float x0 = (float)xr[j];
        float x1 = (float)xr[j + HALF];
        float c = cosr[j];
        float s = sinr[j];
        yr[j] = ({ty})(x0 * c - x1 * s);
        yr[j + HALF] = ({ty})(x1 * c + x0 * s);
    }}
}}
"#
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("rope_half_kernel").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            "rope_half_kernel".to_string(),
            (
                self.s * self.h,
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
        self.s * self.h * self.d
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn bytes_loaded(&self) -> Expression {
        (self.s * self.h * self.d * self.dtype.bits()).ceil_div(8) + self.s * self.d * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.s * self.h * self.d * 4
    }

    fn kernel_name(&self) -> &'static str {
        "RoPEHalf"
    }
}

#[derive(Debug, Clone)]
pub struct RoPEHalfCustom(pub RoPEHalfKernel);

impl CustomOp for RoPEHalfCustom {
    fn to_llir_op(&self) -> LLIROp {
        LLIROp::new::<dyn KernelOp>(Box::new(self.0.clone()) as Box<dyn KernelOp>)
    }
}

/// Half-rotation RoPE over a head group inside a projection output.
///
/// `x` is `(S, pitch)` (e.g. a fused QKV output), `offset`/`h`/`d` select the
/// head group, `cos`/`sin` are `(S, d/2)` F32. Returns contiguous `(S, h*d)`
/// in `x`'s dtype. One kernel replaces the ~10-op rope chain per projection.
pub fn apply_rope_half(
    x: GraphTensor,
    offset: usize,
    h: usize,
    d: usize,
    cos: GraphTensor,
    sin: GraphTensor,
) -> GraphTensor {
    assert_eq!(cos.dtype, DType::F32, "RoPE cos must be F32");
    assert_eq!(sin.dtype, DType::F32, "RoPE sin must be F32");
    let x_dims = x.dims();
    assert_eq!(x_dims.len(), 2, "RoPE x must be 2-D (S, pitch)");
    let s = x_dims[0];
    let pitch = x_dims[1].to_usize().expect("RoPE: pitch must be static");
    assert!(offset + h * d <= pitch, "RoPE head group exceeds row pitch");

    let kern = RoPEHalfKernel {
        s,
        h,
        d,
        pitch,
        offset,
        dtype: x.dtype,
    };
    let cx = unsafe { &mut *x.graph_ref };

    cx.custom_op(RoPEHalfCustom(kern), vec![x, cos, sin], (s, h * d), x.dtype)
}

// ═══════════════════════════════════════════════════════════
// Fused RoPE + KV-cache scatter
//
// The K head group's rope output is consumed by exactly one op: the in-place
// scatter into the cache pool. Fusing them writes the rotated values straight
// to their cache slots, removing one kernel launch and one (s, kv_dim)
// intermediate buffer per layer. Created by `fuse_rope_scatter`, an LLIR
// peephole that runs on every loaded graph (search candidates included), so
// it only fires when the search actually selected the in-place scatter.
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct RoPEScatterKernel {
    pub rope: RoPEHalfKernel,
    /// Total element count of the scatter destination (the cache pool).
    dest_size: Expression,
    /// Flattened scatter-index expression over `z`, where `z` is the element
    /// position in the rope output's contiguous (s, h·d) layout.
    idx_flat: Expression,
}

impl KernelOp for RoPEScatterKernel {
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
        let h = self.rope.h;
        let d = self.rope.d;
        let pitch = self.rope.pitch;
        let offset = self.rope.offset;
        let half = d / 2;
        let ty = crate::cuda_dtype(self.rope.dtype);
        let includes = crate::kernel::hlir::dtype_includes(&[self.rope.dtype]);

        let vars: FxHashSet<char> = self
            .idx_flat
            .dyn_vars()
            .into_iter()
            .chain(self.dest_size.dyn_vars())
            .collect();
        let (dyn_defines, _sorted) = crate::kernel::hlir::generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let idx_expr = self.idx_flat.to_kernel();
        let dest_n = self.dest_size.to_kernel();

        let kernel = format!(
            r#"{includes}
{dyn_defines}
extern "C" __global__ void rope_scatter_kernel(
    {ty}* __restrict__ dest,
    const int* __restrict__ indexes,
    const {ty}* __restrict__ x,
    const float* __restrict__ cos_,
    const float* __restrict__ sin_{dyn_dims_param}
) {{
    const int H = {h};
    const int D = {d};
    const int HALF = {half};
    const long long KVD = (long long)H * D;
    int sh = blockIdx.x;       // 0..S*H
    int s_idx = sh / H;
    int h_idx = sh - s_idx * H;
    int tid = threadIdx.x;

    const {ty}* xr   = x    + (long long)s_idx * {pitch} + {offset} + h_idx * D;
    const float* cosr = cos_ + (long long)s_idx * HALF;
    const float* sinr = sin_ + (long long)s_idx * HALF;

    for (int j = tid; j < HALF; j += {TPB}) {{
        float x0 = (float)xr[j];
        float x1 = (float)xr[j + HALF];
        float c = cosr[j];
        float s = sinr[j];
        {{
            long long const_z = (long long)s_idx * KVD + (long long)h_idx * D + j;
            int idx = indexes[{idx_expr}];
            if (idx >= 0 && idx < ({dest_n})) {{
                dest[idx] = ({ty})(x0 * c - x1 * s);
            }}
        }}
        {{
            long long const_z = (long long)s_idx * KVD + (long long)h_idx * D + j + HALF;
            int idx = indexes[{idx_expr}];
            if (idx >= 0 && idx < ({dest_n})) {{
                dest[idx] = ({ty})(x1 * c + x0 * s);
            }}
        }}
    }}
}}
"#
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("rope_scatter_kernel").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (
                self.rope.s * self.rope.h,
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

    fn build_params(
        &self,
        _stream: &Arc<CudaStream>,
        _output_ptr: u64,
        input_ptrs: &[u64],
        _internal_bufs: &[cudarc::driver::CudaSlice<u8>],
        dyn_dims_ptr: u64,
    ) -> Vec<u64> {
        // rope_scatter_kernel: (dest, indexes, x, cos, sin [, dyn_dims]).
        // Writes in place through dest (input 0), not through output_ptr.
        let mut params = vec![
            input_ptrs[0],
            input_ptrs[1],
            input_ptrs[2],
            input_ptrs[3],
            input_ptrs[4],
        ];
        if dyn_dims_ptr != 0 {
            params.push(dyn_dims_ptr);
        }
        params
    }

    fn output_aliases_input(&self) -> Option<usize> {
        Some(0)
    }

    fn all_dyn_vars(&self) -> FxHashSet<char> {
        self.idx_flat
            .dyn_vars()
            .into_iter()
            .chain(self.dest_size.dyn_vars())
            .chain(self.rope.s.dyn_vars())
            .collect()
    }

    fn output_size(&self) -> Expression {
        self.dest_size
    }

    fn output_bytes(&self) -> Expression {
        (self.dest_size * self.rope.dtype.bits()).ceil_div(8)
    }

    fn output_dtype(&self) -> DType {
        self.rope.dtype
    }

    fn bytes_loaded(&self) -> Expression {
        let rotated = self.rope.s * self.rope.h * self.rope.d;
        (rotated * self.rope.dtype.bits()).ceil_div(8) + rotated * 4 + self.rope.s * self.rope.d * 4
    }

    fn bytes_stored(&self) -> Expression {
        (self.rope.s * self.rope.h * self.rope.d * self.rope.dtype.bits()).ceil_div(8)
    }

    fn flops(&self) -> Expression {
        self.rope.s * self.rope.h * self.rope.d * 4
    }

    fn kernel_name(&self) -> &'static str {
        "RoPEScatter"
    }
}

/// LLIR peephole: fuse `RoPEHalfKernel → KernelScatterNoCopy` pairs into one
/// [`RoPEScatterKernel`] that rotates and writes straight into the cache pool.
///
/// Fires only when (all checked structurally, otherwise the pair is left
/// untouched):
/// - the scatter is the rope output's only consumer,
/// - the scatter reads its source contiguously (identity layout over the
///   rope output),
/// - the scatter index grid matches the rope output shape `(s, h·d)`,
/// - dtypes agree.
///
/// Returns `None` when nothing fused (callers keep the original graph).
/// Count of rope→scatter pairs fused so far (test/debug introspection).
pub static ROPE_SCATTER_FUSIONS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

pub(crate) fn fuse_rope_scatter(
    llir: &luminal::prelude::LLIRGraph,
) -> Option<luminal::prelude::LLIRGraph> {
    use crate::kernel::other_ops::KernelScatterNoCopy;
    use luminal::prelude::petgraph::{Direction, visit::EdgeRef};
    use luminal::shape::flatten_strides;

    let sorted_inputs = |graph: &luminal::prelude::LLIRGraph,
                         node: luminal::prelude::NodeIndex|
     -> Vec<luminal::prelude::NodeIndex> {
        let mut edges: Vec<_> = graph
            .edges_directed(node, Direction::Incoming)
            .map(|e| (e.id(), e.source()))
            .collect();
        edges.sort_by_key(|(id, _)| *id);
        edges.into_iter().map(|(_, source)| source).collect()
    };

    let mut fusions: FxHashMap<luminal::prelude::NodeIndex, (luminal::prelude::NodeIndex, LLIROp)> =
        FxHashMap::default();
    let mut fused_ropes: FxHashSet<luminal::prelude::NodeIndex> = FxHashSet::default();

    for node in llir.node_indices() {
        let Some(kernel) = llir[node].to_dialect::<dyn KernelOp>() else {
            continue;
        };
        let Some(scatter) = kernel
            .as_ref()
            .as_ref()
            .as_any()
            .downcast_ref::<KernelScatterNoCopy>()
        else {
            continue;
        };
        let inputs = sorted_inputs(llir, node);
        if inputs.len() != 3 {
            continue;
        }
        let src = inputs[2];
        if fused_ropes.contains(&src) {
            continue;
        }
        let Some(src_kernel) = llir[src].to_dialect::<dyn KernelOp>() else {
            continue;
        };
        let Some(rope) = src_kernel
            .as_ref()
            .as_ref()
            .as_any()
            .downcast_ref::<RoPEHalfKernel>()
        else {
            continue;
        };
        // The scatter must be the rope's only consumer.
        if llir.edges_directed(src, Direction::Outgoing).count() != 1 {
            continue;
        }
        if scatter.dtype != rope.dtype {
            continue;
        }
        // Scatter grid must be the rope output shape (s, h·d)…
        if scatter.index_shape.len() != 2
            || scatter.index_shape[1].to_usize() != Some(rope.h * rope.d)
            || scatter.index_shape[0].simplify() != rope.s.simplify()
        {
            continue;
        }
        // …read contiguously (identity layout over the rope output).
        let z = Expression::from('z');
        let contig: Vec<Expression> = (0..scatter.index_shape.len())
            .map(|i| {
                scatter.index_shape[i + 1..]
                    .iter()
                    .copied()
                    .product::<Expression>()
                    * z
            })
            .collect();
        let src_flat = flatten_strides(&scatter.index_shape, &scatter.src_strides).simplify();
        let contig_flat = flatten_strides(&scatter.index_shape, &contig).simplify();
        if src_flat != contig_flat {
            continue;
        }

        let fused = RoPEScatterKernel {
            rope: rope.clone(),
            dest_size: scatter.dest_shape.iter().copied().product(),
            idx_flat: flatten_strides(&scatter.index_shape, &scatter.index_strides),
        };
        fusions.insert(
            node,
            (
                src,
                LLIROp::new::<dyn KernelOp>(Box::new(fused) as Box<dyn KernelOp>),
            ),
        );
        fused_ropes.insert(src);
    }

    // Second arm: KernelRoPE (egglog-fused rotary, (heads, seq, hd) buffer)
    // whose sole consumer is the cache scatter reading it through the
    // (s, heads*hd) deinterleave view.
    for node in llir.node_indices() {
        if fusions.contains_key(&node) {
            continue;
        }
        let Some(kernel) = llir[node].to_dialect::<dyn KernelOp>() else {
            continue;
        };
        let Some(scatter) = kernel
            .as_ref()
            .as_ref()
            .as_any()
            .downcast_ref::<KernelScatterNoCopy>()
        else {
            continue;
        };
        let inputs = sorted_inputs(llir, node);
        if inputs.len() != 3 {
            continue;
        }
        let src = inputs[2];
        if fused_ropes.contains(&src) {
            continue;
        }
        let Some(src_kernel) = llir[src].to_dialect::<dyn KernelOp>() else {
            continue;
        };
        let Some(rope) = src_kernel
            .as_ref()
            .as_ref()
            .as_any()
            .downcast_ref::<KernelRoPE>()
        else {
            continue;
        };
        if llir.edges_directed(src, Direction::Outgoing).count() != 1 {
            continue;
        }
        if scatter.dtype != DType::Bf16 {
            continue;
        }
        let heads = rope.out_shape[0];
        let seq = rope.out_shape[1];
        let hd = rope.out_shape[2];
        let kvd = (heads * hd).simplify();
        // Index grid must be the (s, heads*hd) deinterleaved layout…
        if scatter.index_shape.len() != 2
            || scatter.index_shape[1].simplify() != kvd
            || scatter.index_shape[0].simplify() != seq.simplify()
        {
            continue;
        }
        // …and the source must be read through exactly the deinterleave view
        // of the (heads, seq, hd) rope buffer: offset(s_i, c) =
        // (c/hd)·hd·seq + s_i·hd + c%hd.
        let z = Expression::from('z');
        // NB: association order must match the emission exactly —
        // `simplify()` does not canonicalize Mul associativity, so
        // ((z/hd)*hd)*seq matches but (z/hd)*(hd*seq) does not.
        let expected = vec![z * hd, ((z / hd) * hd) * seq + z % hd];
        let src_flat = flatten_strides(&scatter.index_shape, &scatter.src_strides).simplify();
        let expected_flat = flatten_strides(&scatter.index_shape, &expected).simplify();
        if src_flat != expected_flat {
            continue;
        }

        let fused = KernelRoPEScatterFused {
            rope: rope.clone(),
            dest_size: scatter.dest_shape.iter().copied().product(),
            idx_flat: flatten_strides(&scatter.index_shape, &scatter.index_strides),
        };
        fusions.insert(
            node,
            (
                src,
                LLIROp::new::<dyn KernelOp>(Box::new(fused) as Box<dyn KernelOp>),
            ),
        );
        fused_ropes.insert(src);
    }

    if fusions.is_empty() {
        return None;
    }
    ROPE_SCATTER_FUSIONS.fetch_add(fusions.len(), std::sync::atomic::Ordering::Relaxed);

    // Rebuild the graph so per-node input-edge id order stays deterministic
    // (StableGraph reuses freed edge ids, so in-place edge surgery would
    // scramble the sorted-by-edge-id input convention).
    let mut new = luminal::prelude::LLIRGraph::default();
    let mut map: FxHashMap<luminal::prelude::NodeIndex, luminal::prelude::NodeIndex> =
        FxHashMap::default();
    for node in llir.node_indices() {
        if fused_ropes.contains(&node) {
            continue;
        }
        let weight = fusions
            .get(&node)
            .map(|(_, op)| op.clone())
            .unwrap_or_else(|| llir[node].clone());
        map.insert(node, new.add_node(weight));
    }
    for node in llir.node_indices() {
        if fused_ropes.contains(&node) {
            continue;
        }
        let mut inputs = sorted_inputs(llir, node);
        if let Some((rope_node, _)) = fusions.get(&node) {
            let rope_inputs = sorted_inputs(llir, *rope_node);
            // (dest, idx, rope) → (dest, idx, <rope inputs…>):
            // RoPEHalfKernel contributes (x, cos, sin); KernelRoPE (x, pos).
            inputs = [&inputs[..2], &rope_inputs[..]].concat();
        }
        for input in inputs {
            new.add_edge(map[&input], map[&node], ());
        }
    }
    Some(new)
}

// ═══════════════════════════════════════════════════════════
// KernelRoPE — egglog-matched fused rotary (half convention, bf16).
//
// Matches the full HLIR rotary chain the qwen/gemma models spell:
// inv-freq (iota·2 → cast → ×(1/hd) → ×ln(theta) → ×log2(e) → exp2 → recip),
// angles (cast(pos) × inv_freq → sum), sin / cos-as-sin(π/2−x), bf16 casts,
// the x0 strided view + x1 offset-slice gather, the rotation arithmetic, and
// the concat (2 clamped gathers + 2 mask iotas). The rule roots at the
// concat Add eclass — the last materialized tensor of the chain; the
// trailing transpose+merge is a view applied by consumers, so the fused
// kernel writes the same (heads, seq, hd) buffer the concat produces and
// every consumer works unchanged. One kernel replaces ~13 launches per rope
// call; the angle chain (shared by the q and k calls within a layer)
// becomes dead.
//
// The kernel mirrors the decomposed numerics exactly: angle math in F32 with
// the same op order/spellings (exp2f, 1.0f/x, sinf, sin(−x+π/2)), and a bf16
// rounding at every decomposed op boundary (cos/sin casts, each mul, the −1
// mul). The concat's ×{0,1} masks and +0 are value-preserving and elided.
// ═══════════════════════════════════════════════════════════

#[derive(Default, Debug, Clone)]
pub struct KernelRoPE {
    /// `(heads, seq, head_dim)` — seq may be dynamic.
    out_shape: Vec<Expression>,
    /// Row stride of the x input in elements (the projection width).
    width: usize,
    ln_theta: f64,
    inv_hd: f64,
}

pub(crate) mod kernel_ro_p_e {
    use crate::typed_rules::*;
    use egglog_experimental::typed::builtins as egg;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    // A monolithic ~45-atom join took 4m56s on a 279-node mini layer.
    // Stage the angle chain at its Constant anchors, then bind cos/sin from
    // rope_angles before matching rotation so each Mul remains selective.
    #[relation]
    pub(crate) fn rope_invf(a0: Ir, a1: egg::F64, a2: egg::F64);

    #[relation]
    pub(crate) fn rope_angles(a0: Ir, a1: Ir, a2: Ir, a3: egg::F64, a4: egg::F64);

    #[relation]
    #[allow(clippy::too_many_arguments)] // Positional fields of the native kernel constructor.
    pub(crate) fn rope_rotated(
        a0: Ir,
        a1: Ir,
        a2: Ir,
        a3: Ir,
        a4: DimExpr,
        a5: DimExpr,
        a6: DimExpr,
        a7: DimExpr,
        a8: egg::F64,
        a9: egg::F64,
    );

    #[allow(
        clippy::approx_constant,
        reason = "Exact pattern literals must match the authored graph."
    )]
    #[ruleset]
    pub(crate) fn kernel_fuse_late_pre(
        if_range: &DimExpr,
        ic_size: &DimExpr,
        inv_hd: &egg::F64,
        ln_theta: &egg::F64,
        invf: &Ir,
    ) -> Rule {
        let iot2 = &Ir::Op(OpKind::Iota(DimExpr::MIter() * 2, if_range), IList::INil());
        let invhd_c = &Ir::Op(OpKind::Constant(inv_hd), IList::INil());
        let lnt_c = &Ir::Op(OpKind::Constant(ln_theta), IList::INil());
        let l2e = &Ir::Op(OpKind::Constant(1.442695), IList::INil());
        let iotf = &Ir::Op(OpKind::Cast(ic_size, DType::F32()), [iot2]);
        let fr1 = &Ir::Op(MulArgs::fresh(), [iotf, invhd_c]);
        let fr2 = &Ir::Op(MulArgs::fresh(), [fr1, lnt_c]);
        let fr3 = &Ir::Op(MulArgs::fresh(), [fr2, l2e]);
        let ex = &Ir::Op(Exp2Args::fresh(), [fr3]);
        rule(
            (eq(invf, Ir::Op(RecipArgs::fresh(), [ex])),),
            (rope_invf(invf, ln_theta, inv_hd),),
        )
        .label("kernel rope invf stage")
    }

    #[allow(
        clippy::approx_constant,
        reason = "Exact pattern literals must match the authored graph."
    )]
    #[ruleset]
    pub(crate) fn kernel_fuse_late(
        invf: &Ir,
        ln_theta: &egg::F64,
        inv_hd: &egg::F64,
        pc_size: &DimExpr,
        pos: &Ir,
        emb: &Ir,
        s1_sh: &EList,
        s1_in: &EList,
        s1_out: &EList,
        sb_size: &DimExpr,
        sinb: &Ir,
        emb2: &Ir,
        s2_sh: &EList,
        s2_in: &EList,
        s2_out: &EList,
        cb_size: &DimExpr,
        cosb: &Ir,
        e_hd: &DimExpr,
        e_w: &DimExpr,
        e_hd2: &DimExpr,
        x: &Ir,
        a1_sh: &EList,
        x0out: &Ir,
        nb_size: &DimExpr,
        x1out: &Ir,
        x3: &Ir,
        x4: &Ir,
        e_seq: &DimExpr,
        e_ch: &DimExpr,
        e_hs: &DimExpr,
        x1_range: &DimExpr,
        cat: &Ir,
        heads: &DimExpr,
        seqd: &DimExpr,
        hdd: &DimExpr,
        e_hdm: &DimExpr,
        c0_range: &DimExpr,
        k0_size: &DimExpr,
        mk0_range: &DimExpr,
        c1_range: &DimExpr,
        k1_size: &DimExpr,
        mk1_range: &DimExpr,
    ) -> Vec<Rule> {
        let rotated = &Ir::Op(kernel_ro_pe(a1_sh, e_w, ln_theta, inv_hd), [x, pos]);
        let posf = &Ir::Op(OpKind::Cast(pc_size, DType::F32()), [pos]);
        let sinv = &Ir::Op(OpKind::Sin(s1_sh, s1_in, s1_out), [emb]);
        let m1c = &Ir::Op(OpKind::Constant(-1.0), IList::INil());
        let hpi = &Ir::Op(OpKind::Constant(1.570796), IList::INil());
        let x0c = &Ir::Op(
            MulArgs {
                a_strides: ([
                    &(DimExpr::MIter() * e_hd),
                    &(DimExpr::MIter() * e_w),
                    &(DimExpr::MIter()),
                ])
                .into(),
                b_strides: ([
                    &(DimExpr::MNum(0)),
                    &(DimExpr::MIter() * e_hd2),
                    &(DimExpr::MIter()),
                ])
                .into(),
                ..MulArgs::fresh()
            },
            [x, cosb],
        );
        let x0s = &Ir::Op(
            MulArgs {
                a_strides: ([
                    &(DimExpr::MIter() * e_hd),
                    &(DimExpr::MIter() * e_w),
                    &(DimExpr::MIter()),
                ])
                .into(),
                ..MulArgs::fresh()
            },
            [x3, sinb],
        );
        let x1idx = &Ir::Op(
            OpKind::Iota(
                (((DimExpr::MIter() % e_hd2) + e_hd2)
                    + (((DimExpr::MIter() / e_hd2) % e_seq) * e_hd))
                    + ((DimExpr::MIter() / e_ch) * e_hs),
                x1_range,
            ),
            IList::INil(),
        );
        let c0idx = &Ir::Op(
            OpKind::Iota(
                (DimExpr::MMin(DimExpr::MIter() % e_hd, e_hdm)
                    + (((DimExpr::MIter() / e_hd) % e_seq) * e_hd2))
                    + ((DimExpr::MIter() / e_hs) * e_ch),
                c0_range,
            ),
            IList::INil(),
        );
        let mk = &Ir::Op(
            OpKind::Iota(DimExpr::MLt(DimExpr::MIter() % e_hd, e_hd2), mk0_range),
            IList::INil(),
        );
        let c1idx = &Ir::Op(
            OpKind::Iota(
                (DimExpr::MMax((DimExpr::MIter() % e_hd) - e_hd2, 0)
                    + (((DimExpr::MIter() / e_hd) % e_seq) * e_hd2))
                    + ((DimExpr::MIter() / e_hs) * e_ch),
                c1_range,
            ),
            IList::INil(),
        );
        let mk2 = &Ir::Op(
            OpKind::Iota(DimExpr::MGte(DimExpr::MIter() % e_hd, e_hd2), mk1_range),
            IList::INil(),
        );
        let embm = &Ir::Op(MulArgs::fresh(), [posf, invf]);
        let neg = &Ir::Op(MulArgs::fresh(), [emb2, m1c]);
        let negb = &Ir::Op(OpKind::Cast(nb_size, DType::Bf16()), [m1c]);
        let x2 = &Ir::Op(
            GatherArgs {
                data_strides: ([
                    &(DimExpr::MIter() * e_hd),
                    &(DimExpr::MIter() * e_w),
                    &(DimExpr::MIter()),
                ])
                .into(),
                ..GatherArgs::fresh()
            },
            [x1idx, x4],
        );
        let x0g = &Ir::Op(GatherArgs::fresh(), [c0idx, x0out]);
        let mk0b = &Ir::Op(OpKind::Cast(k0_size, DType::Bf16()), [mk]);
        let x1g = &Ir::Op(GatherArgs::fresh(), [c1idx, x1out]);
        let mk1b = &Ir::Op(OpKind::Cast(k1_size, DType::Bf16()), [mk2]);
        let shift = &Ir::Op(AddArgs::fresh(), [neg, hpi]);
        let x1s = &Ir::Op(MulArgs::fresh(), [x2, sinb]);
        let x1c = &Ir::Op(MulArgs::fresh(), [x2, cosb]);
        let x0m = &Ir::Op(MulArgs::fresh(), [x0g, mk0b]);
        let x1m = &Ir::Op(MulArgs::fresh(), [x1g, mk1b]);
        let cosv = &Ir::Op(OpKind::Sin(s2_sh, s2_in, s2_out), [shift]);
        let x1sn = &Ir::Op(MulArgs::fresh(), [x1s, negb]);
        vec![
            rule(
                (
                    rope_invf(invf, ln_theta, inv_hd),
                    eq(DType::Int(), pos.dtype()),
                    eq(emb, Ir::Op(SumArgs::fresh(), [embm])),
                    eq(sinb, Ir::Op(OpKind::Cast(sb_size, DType::Bf16()), [sinv])),
                    eq(emb, emb2),
                    eq(cosb, Ir::Op(OpKind::Cast(cb_size, DType::Bf16()), [cosv])),
                ),
                (rope_angles(cosb, sinb, pos, ln_theta, inv_hd),),
            )
            .label("kernel rope angles stage"),
            rule(
                (
                    rope_angles(cosb, sinb, pos, ln_theta, inv_hd),
                    eq(x0out, Ir::Op(AddArgs::fresh(), [x0c, x1sn])),
                    eq(x1out, Ir::Op(AddArgs::fresh(), [x1c, x0s])),
                    eq(x, x3),
                    eq(x, x4),
                    eq(DType::Bf16(), x.dtype()),
                ),
                (rope_rotated(
                    x0out, x1out, x, pos, e_hd, e_hd2, e_w, e_seq, ln_theta, inv_hd,
                ),),
            )
            .label("kernel rope rotation stage"),
            rule(
                (
                    rope_rotated(
                        x0out, x1out, x, pos, e_hd, e_hd2, e_w, e_seq, ln_theta, inv_hd,
                    ),
                    eq(
                        cat,
                        Ir::Op(
                            AddArgs {
                                shape: a1_sh.into(),
                                ..AddArgs::fresh()
                            },
                            [x0m, x1m],
                        ),
                    ),
                    eq(a1_sh, [heads, seqd, hdd]),
                    eq(hdd, e_hd),
                    eq(seqd, e_seq),
                ),
                (
                    rotated,
                    union(cat, rotated),
                    set(rotated.dtype(), DType::Bf16()),
                ),
            )
            .label("kernel rope half bf16"),
        ]
    }
}

impl KernelRoPE {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_out_shape, field_width, field_ln_theta, field_inv_hd)) = typed::get_args(
            kind,
            |out_shape: &EList, width: &DimExpr, ln_theta: &egg::F64, inv_hd: &egg::F64| {
                crate::typed_rules::kernel_ro_pe(out_shape, width, ln_theta, inv_hd)
            },
        )?
        else {
            return Ok(None);
        };
        let decoded_out_shape = context.expressions(&field_out_shape)?;
        let out_shape = decoded_out_shape;
        let width = context
            .expression(&field_width)?
            .to_usize()
            .ok_or_else(|| TypedError::Decode("RoPE width must be static".into()))?;
        let ln_theta: f64 = f64::try_from(&field_ln_theta)?;
        let inv_hd: f64 = f64::try_from(&field_inv_hd)?;
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            out_shape,
            width,
            ln_theta,
            inv_hd,
        })
            as Box<dyn KernelOp>)))
    }
}

impl KernelOp for KernelRoPE {
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
        let heads = self.out_shape[0].to_usize().expect("RoPE heads is static");
        let seq = self.out_shape[1];
        let hd = self.out_shape[2]
            .to_usize()
            .expect("RoPE head_dim is static");
        let w = self.width;
        let half = hd / 2;
        let lnt = self.ln_theta as f32;
        let inv_hd = self.inv_hd as f32;

        let vars: FxHashSet<char> = seq.dyn_vars().into_iter().collect();
        let (dyn_defines, _sorted) = crate::kernel::hlir::generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let seq_expr = seq.to_kernel();

        let kernel = format!(
            "#include <cuda_bf16.h>
{dyn_defines}
extern \"C\" {{
    __global__ void rope_k(__nv_bfloat16 *out, const __nv_bfloat16 *x, const int *pos{dyn_dims_param}) {{
        long long s = blockIdx.y;
        int h = blockIdx.z;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= {hd}) return;
        int fi = (i < {half}) ? i : (i - {half});
        // angle chain in F32 with the decomposed op spellings
        float v = (float)(2 * fi);
        float freq = ((v * {inv_hd:.10e}f) * {lnt:.10e}f) * 1.442695f;
        float invf = 1.0f / exp2f(freq);
        float angle = (float)pos[s] * invf;
        // bf16 rounding at each decomposed op boundary
        float sb = __bfloat162float(__float2bfloat16(sinf(angle)));
        float cb = __bfloat162float(__float2bfloat16(sinf(angle * -1.0f + 1.570796f)));
        long long xbase = s * {w} + (long long)h * {hd};
        float out_v;
        if (i < {half}) {{
            float x0 = __bfloat162float(x[xbase + i]);
            float x1 = __bfloat162float(x[xbase + i + {half}]);
            float x0c = __bfloat162float(__float2bfloat16(x0 * cb));
            float x1s = __bfloat162float(__float2bfloat16(x1 * sb));
            float x1sn = __bfloat162float(__float2bfloat16(x1s * -1.0f));
            out_v = x0c + x1sn;
        }} else {{
            float x1 = __bfloat162float(x[xbase + i]);
            float x0 = __bfloat162float(x[xbase + i - {half}]);
            float x1c = __bfloat162float(__float2bfloat16(x1 * cb));
            float x0s = __bfloat162float(__float2bfloat16(x0 * sb));
            out_v = x1c + x0s;
        }}
        out[((long long)h * ({seq_expr}) + s) * {hd} + i] = __float2bfloat16(out_v);
    }}
}}"
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("rope_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        let tpb = hd.min(256);
        (
            func,
            module,
            kernel,
            (
                Expression::from(hd.div_ceil(tpb)),
                seq,
                Expression::from(heads),
            ),
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
        self.output_size() * 2
    }

    fn output_dtype(&self) -> DType {
        DType::Bf16
    }

    fn all_dyn_vars(&self) -> FxHashSet<char> {
        self.out_shape[1].dyn_vars().into_iter().collect()
    }

    fn bytes_loaded(&self) -> Expression {
        self.output_size() * 2 + self.out_shape[1] * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.output_size() * 16
    }

    fn kernel_name(&self) -> &'static str {
        "RoPEFused"
    }
}

// ═══════════════════════════════════════════════════════════
// KernelRoPEScatterFused — LLIR peephole product: KernelRoPE whose sole
// consumer is the KV-cache KernelScatterNoCopy reading the rope buffer
// through the (s, heads·hd) deinterleave view. Rotates and writes straight
// into the cache pool rows; output aliases the dest (input 0).
// ═══════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct KernelRoPEScatterFused {
    rope: KernelRoPE,
    /// Total element count of the scatter destination (the cache pool).
    dest_size: Expression,
    /// Flattened scatter-index expression over `z`, where `z` is the element
    /// position in the (s, heads·hd) deinterleaved index grid.
    idx_flat: Expression,
}

impl KernelOp for KernelRoPEScatterFused {
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
        let heads = self.rope.out_shape[0]
            .to_usize()
            .expect("RoPE heads is static");
        let seq = self.rope.out_shape[1];
        let hd = self.rope.out_shape[2]
            .to_usize()
            .expect("RoPE head_dim is static");
        let w = self.rope.width;
        let half = hd / 2;
        let lnt = self.rope.ln_theta as f32;
        let inv_hd = self.rope.inv_hd as f32;
        let kvd = heads * hd;

        let vars: FxHashSet<char> = self.all_dyn_vars();
        let (dyn_defines, _sorted) = crate::kernel::hlir::generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let idx_expr = self.idx_flat.to_kernel();
        let dest_n = self.dest_size.to_kernel();

        let kernel = format!(
            "#include <cuda_bf16.h>
{dyn_defines}
extern \"C\" {{
    __global__ void rope_scatter_fused_k(__nv_bfloat16 *dest, const int *indexes, const __nv_bfloat16 *x, const int *pos{dyn_dims_param}) {{
        long long s = blockIdx.y;
        int h = blockIdx.z;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= {hd}) return;
        int fi = (i < {half}) ? i : (i - {half});
        float v = (float)(2 * fi);
        float freq = ((v * {inv_hd:.10e}f) * {lnt:.10e}f) * 1.442695f;
        float invf = 1.0f / exp2f(freq);
        float angle = (float)pos[s] * invf;
        float sb = __bfloat162float(__float2bfloat16(sinf(angle)));
        float cb = __bfloat162float(__float2bfloat16(sinf(angle * -1.0f + 1.570796f)));
        long long xbase = s * {w} + (long long)h * {hd};
        float out_v;
        if (i < {half}) {{
            float x0 = __bfloat162float(x[xbase + i]);
            float x1 = __bfloat162float(x[xbase + i + {half}]);
            float x0c = __bfloat162float(__float2bfloat16(x0 * cb));
            float x1s = __bfloat162float(__float2bfloat16(x1 * sb));
            float x1sn = __bfloat162float(__float2bfloat16(x1s * -1.0f));
            out_v = x0c + x1sn;
        }} else {{
            float x1 = __bfloat162float(x[xbase + i]);
            float x0 = __bfloat162float(x[xbase + i - {half}]);
            float x1c = __bfloat162float(__float2bfloat16(x1 * cb));
            float x0s = __bfloat162float(__float2bfloat16(x0 * sb));
            out_v = x1c + x0s;
        }}
        long long const_z = s * {kvd} + (long long)h * {hd} + i;
        int idx = indexes[{idx_expr}];
        if (idx >= 0 && idx < ({dest_n})) {{
            dest[idx] = __float2bfloat16(out_v);
        }}
    }}
}}"
        );

        let (module, func) = if let Some((m, f)) = compile_cache.get(&kernel) {
            (m.clone(), f.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("rope_scatter_fused_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        let tpb = hd.min(256);
        (
            func,
            module,
            kernel,
            (
                Expression::from(hd.div_ceil(tpb)),
                seq,
                Expression::from(heads),
            ),
            (
                Expression::from(tpb),
                Expression::from(1usize),
                Expression::from(1usize),
            ),
            Expression::from(0usize),
            FxHashMap::default(),
        )
    }

    fn build_params(
        &self,
        _stream: &Arc<CudaStream>,
        _output_ptr: u64,
        input_ptrs: &[u64],
        _internal_bufs: &[cudarc::driver::CudaSlice<u8>],
        dyn_dims_ptr: u64,
    ) -> Vec<u64> {
        // rope_scatter_fused_k: (dest, indexes, x, pos [, dyn_dims]).
        // Writes in place through dest (input 0), not through output_ptr.
        let mut params = vec![input_ptrs[0], input_ptrs[1], input_ptrs[2], input_ptrs[3]];
        if dyn_dims_ptr != 0 {
            params.push(dyn_dims_ptr);
        }
        params
    }

    fn output_aliases_input(&self) -> Option<usize> {
        Some(0)
    }

    fn all_dyn_vars(&self) -> FxHashSet<char> {
        self.idx_flat
            .dyn_vars()
            .into_iter()
            .chain(self.dest_size.dyn_vars())
            .chain(self.rope.out_shape[1].dyn_vars())
            .collect()
    }

    fn output_size(&self) -> Expression {
        self.dest_size
    }

    fn output_bytes(&self) -> Expression {
        self.dest_size * 2
    }

    fn output_dtype(&self) -> DType {
        DType::Bf16
    }

    fn bytes_loaded(&self) -> Expression {
        let rotated: Expression = self.rope.out_shape.iter().copied().product();
        rotated * 2 + rotated * 4 + self.rope.out_shape[1] * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.rope.out_shape.iter().copied().product::<Expression>() * 2
    }

    fn flops(&self) -> Expression {
        self.rope.out_shape.iter().copied().product::<Expression>() * 16
    }

    fn kernel_name(&self) -> &'static str {
        "RoPEScatterFused"
    }
}
