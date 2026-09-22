use egglog_experimental::typed::{self as typed, TypedError, builtins as egg};
use luminal::egglog_utils::DecodeContext;
use luminal::typed_reference::schema::{DimExpr, EList, IList, Ir, OpKind};
use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device, cuda_dtype,
    kernel::{CudaFunctionExt, KernelOp},
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use itertools::Itertools;
use luminal::{
    hlir::{LessThan, MaxReduce, Mod, Scatter, SumReduce},
    op::*,
    prelude::*,
};

/// Generates CUDA include directives based on the dtypes used in a kernel
pub fn dtype_includes(dtypes: &[DType]) -> String {
    let needs_fp16 = dtypes.iter().any(|d| matches!(d, DType::F16));
    let needs_bf16 = dtypes.iter().any(|d| matches!(d, DType::Bf16));
    let needs_fp8 = dtypes
        .iter()
        .any(|d| matches!(d, DType::F8E4M3 | DType::F8E5M2 | DType::F8UE8M0));
    let needs_fp6 = dtypes
        .iter()
        .any(|d| matches!(d, DType::F6E2M3 | DType::F6E3M2));
    let needs_fp4 = dtypes.iter().any(|d| matches!(d, DType::F4E2M1));
    let mut s = String::new();
    if needs_fp16 {
        s.push_str("#include <cuda_fp16.h>\n");
    }
    if needs_bf16 {
        s.push_str("#include <cuda_bf16.h>\n");
    }
    if needs_fp8 {
        s.push_str("#include <cuda_fp8.h>\n");
    }
    if needs_fp6 {
        s.push_str("#include <cuda_fp6.h>\n");
    }
    if needs_fp4 {
        s.push_str("#include <cuda_fp4.h>\n");
    }
    s
}

#[derive(Default, Debug, Clone)]

pub struct KernelMaxReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
    dtype: DType,
}
pub(crate) mod kernel_max {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(
        shape: &EList,
        iters: &DimExpr,
        strides: &EList,
        iter_stride: &DimExpr,
        out_strides: &EList,
        inputs: &IList,
        dtype: &DType,
    ) -> Rule {
        let source = &Ir::Op(
            OpKind::Max(shape, iters, strides, iter_stride, out_strides),
            inputs,
        );
        rule(
            (source, eq(dtype, source.dtype())),
            (union(
                source,
                Ir::Op(
                    kernel_max(shape, iters, strides, iter_stride, out_strides, dtype),
                    inputs,
                ),
            ),),
        )
    }
}

impl KernelMaxReduce {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((
            field_shape,
            field_iters,
            field_strides,
            field_iter_stride,
            field_out_strides,
            field_dtype,
        )) = typed::get_args(
            kind,
            |shape: &EList,
             iters: &DimExpr,
             strides: &EList,
             iter_stride: &DimExpr,
             out_strides: &EList,
             dtype: &luminal::typed_reference::schema::DType| {
                crate::typed_rules::kernel_max(
                    shape,
                    iters,
                    strides,
                    iter_stride,
                    out_strides,
                    dtype,
                )
            },
        )?
        else {
            return Ok(None);
        };
        let decoded_shape = context.expressions(&field_shape)?;
        let decoded_strides = context.expressions(&field_strides)?;
        let decoded_out_strides = context.expressions(&field_out_strides)?;
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            out_shape: decoded_shape,
            iters: context.expression(&field_iters)?,
            in_stride: decoded_strides,
            iter_stride: context.expression(&field_iter_stride)?,
            out_stride: decoded_out_strides,
            dtype: context.dtype(&field_dtype)?,
        })
            as Box<dyn KernelOp>)))
    }
}

impl KernelOp for KernelMaxReduce {
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
            .chain(self.in_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.iters.dyn_vars())
            .chain(self.iter_stride.dyn_vars())
            .collect::<FxHashSet<_>>();

        let dtype = cuda_dtype(self.dtype);
        // Sub-32-bit float storage reduces through a float accumulator.
        // Exact for max: selection never rounds, and the final store writes a
        // value already representable in the storage dtype.
        let low_precision_storage = matches!(
            self.dtype,
            DType::F8E4M3 | DType::F8E5M2 | DType::F8UE8M0 | DType::F16 | DType::Bf16
        );
        let accum_dtype = if low_precision_storage {
            "float"
        } else {
            dtype
        };
        let includes = dtype_includes(&[self.dtype]);
        let n_outputs: Expression = self.out_shape.iter().copied().product();
        let threads_per_block = 256; // 8 warps per block
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let iter_stride_of_i = self.iter_stride.to_kernel().replace("const_z", "i");
        let load_value = if low_precision_storage {
            format!("static_cast<float>(in[in_start + {iter_stride_of_i}])")
        } else {
            format!("in[in_start + {iter_stride_of_i}]")
        };

        let kernel = format!(
            "{includes}
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define FULL_MASK 0xffffffff
#define NEG_INF_F __int_as_float(0xff800000)
{dyn_defines}
extern \"C\" {{
    __global__ void reduce_max_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        __shared__ {accum_dtype} warp_sums[THREADS_PER_BLOCK / WARP_SIZE];
        long long const_z = blockIdx.x;

        int tid = threadIdx.x;
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        long long in_start = {in_index};
        long long iters = {iters};

        {accum_dtype} max_value = ({accum_dtype})NEG_INF_F;
        for (long long i = tid; i < iters; i += THREADS_PER_BLOCK) {{
            max_value = fmaxf(max_value, {load_value});
        }}

        #pragma unroll
        for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
            max_value = fmaxf(max_value, __shfl_down_sync(FULL_MASK, max_value, s));
        }}

        if (lane_id == 0) {{
            warp_sums[warp_id] = max_value;
        }}
        __syncthreads();

        if (warp_id == 0) {{
            int cnt = THREADS_PER_BLOCK / WARP_SIZE;
            {accum_dtype} block_max = tid < cnt ? warp_sums[tid] : ({accum_dtype})NEG_INF_F;

            #pragma unroll
            for (int s = cnt / 2; s > 0; s /= 2) {{
                block_max = fmaxf(block_max, __shfl_down_sync(FULL_MASK, block_max, s));
            }}

            if (tid == 0) {{
                out[{out_index}] = ({dtype})block_max;
            }}
        }}
    }}
}}",
            dtype = dtype,
            accum_dtype = accum_dtype,
            in_index = flatten_strides(&self.out_shape, &self.in_stride).to_kernel(),
            out_index = flatten_strides(&self.out_shape, &self.out_stride).to_kernel(),
            iters = self.iters.to_kernel(),
            load_value = load_value,
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("reduce_max_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_outputs, 1.into(), 1.into()),                // grid
            (threads_per_block.into(), 1.into(), 1.into()), // blocks
            32.into(),                                      // shmem size
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        (self.out_shape.iter().copied().product::<Expression>() * self.iters * self.dtype.bits())
            .ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product::<Expression>() * self.iters
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "MaxReduce"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelSumReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
    dtype: DType,
}
pub(crate) mod kernel_sum {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(
        shape: &EList,
        iters: &DimExpr,
        strides: &EList,
        iter_stride: &DimExpr,
        out_strides: &EList,
        inputs: &IList,
        dtype: &DType,
    ) -> Rule {
        let source = &Ir::Op(
            OpKind::Sum(shape, iters, strides, iter_stride, out_strides),
            inputs,
        );
        rule(
            (source, eq(dtype, source.dtype())),
            (union(
                source,
                Ir::Op(
                    kernel_sum(shape, iters, strides, iter_stride, out_strides, dtype),
                    inputs,
                ),
            ),),
        )
    }
}

impl KernelSumReduce {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((
            field_shape,
            field_iters,
            field_strides,
            field_iter_stride,
            field_out_strides,
            field_dtype,
        )) = typed::get_args(
            kind,
            |shape: &EList,
             iters: &DimExpr,
             strides: &EList,
             iter_stride: &DimExpr,
             out_strides: &EList,
             dtype: &luminal::typed_reference::schema::DType| {
                crate::typed_rules::kernel_sum(
                    shape,
                    iters,
                    strides,
                    iter_stride,
                    out_strides,
                    dtype,
                )
            },
        )?
        else {
            return Ok(None);
        };
        let decoded_shape = context.expressions(&field_shape)?;
        let decoded_strides = context.expressions(&field_strides)?;
        let decoded_out_strides = context.expressions(&field_out_strides)?;
        Ok(Some({
            let out_shape = decoded_shape;
            let iters = context.expression(&field_iters)?;
            let in_stride = decoded_strides;
            let iter_stride = context.expression(&field_iter_stride)?;
            let out_stride = decoded_out_strides;
            let dtype = context.dtype(&field_dtype)?;
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                out_shape,
                iters,
                in_stride,
                iter_stride,
                out_stride,
                dtype,
            }) as Box<dyn KernelOp>)
        }))
    }
}

impl KernelOp for KernelSumReduce {
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
            .chain(self.in_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.iters.dyn_vars())
            .chain(self.iter_stride.dyn_vars())
            .collect::<FxHashSet<_>>();

        let dtype = cuda_dtype(self.dtype);
        // Sub-32-bit float storage accumulates in float — the reduction
        // analogue of cuBLASLt's COMPUTE_32F_FAST_16BF policy for 16-bit
        // GEMMs (16-bit IO, F32 accumulation, one rounding at the store).
        let uses_fp8_storage = matches!(
            self.dtype,
            DType::F8E4M3 | DType::F8E5M2 | DType::F8UE8M0 | DType::F16 | DType::Bf16
        );
        let accum_dtype = if uses_fp8_storage { "float" } else { dtype };
        let includes = dtype_includes(&[self.dtype]);
        let n_outputs: Expression = self.out_shape.iter().copied().product();
        let threads_per_block = 256; // 8 warps per block
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let iter_stride_of_i = self.iter_stride.to_kernel().replace("const_z", "i");
        let load_value = if uses_fp8_storage {
            format!("static_cast<float>(in_data[in_start + {iter_stride_of_i}])")
        } else {
            format!("in_data[in_start + {iter_stride_of_i}]")
        };
        let zero = if uses_fp8_storage {
            "0.0f".to_string()
        } else {
            format!("({dtype})0")
        };

        let kernel = format!(
            "{includes}
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define FULL_MASK 0xffffffff
{dyn_defines}
extern \"C\" {{
    __global__ void reduce_sum_k({dtype} *out, const {dtype} *in_data{dyn_dims_param}) {{
        __shared__ {accum_dtype} warp_sums[THREADS_PER_BLOCK / WARP_SIZE];
        long long const_z = blockIdx.x;

        int tid = threadIdx.x;
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        long long in_start = {in_index};
        long long iters = {iters};

        {accum_dtype} partial = {zero};
        {accum_dtype} comp = {zero};   // Kahan compensation
        for (long long i = tid; i < iters; i += THREADS_PER_BLOCK) {{
            {accum_dtype} y = {load_value} - comp;
            {accum_dtype} t = partial + y;
            comp = (t - partial) - y;
            partial = t;
        }}

        #pragma unroll
        for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
            partial += __shfl_down_sync(FULL_MASK, partial, s);
        }}

        if (lane_id == 0) {{
            warp_sums[warp_id] = partial;
        }}
        __syncthreads();

        if (warp_id == 0) {{
            int cnt = THREADS_PER_BLOCK / WARP_SIZE;
            {accum_dtype} block_sum = tid < cnt ? warp_sums[tid] : {zero};

            #pragma unroll
            for (int s = cnt / 2; s > 0; s /= 2) {{
                block_sum += __shfl_down_sync(FULL_MASK, block_sum, s);
            }}

            if (tid == 0) {{
                out[{out_index}] = ({dtype})block_sum;
            }}
        }}
    }}
}}",
            dtype = dtype,
            accum_dtype = accum_dtype,
            in_index = flatten_strides(&self.out_shape, &self.in_stride).to_kernel(),
            out_index = flatten_strides(&self.out_shape, &self.out_stride).to_kernel(),
            iters = self.iters.to_kernel(),
            load_value = load_value,
            zero = zero,
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("reduce_sum_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_outputs, 1.into(), 1.into()),                // grid
            (threads_per_block.into(), 1.into(), 1.into()), // blocks (warp-parallel)
            32.into(),                                      // shmem for warp_sums
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        (self.out_shape.iter().copied().product::<Expression>() * self.iters * self.dtype.bits())
            .ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product::<Expression>() * self.iters
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "SumReduce"
    }
}

/// Fused `Cast(F32) → SumReduce → Cast(16-bit)` with an F32 accumulator.
///
/// The dtype contract requires accumulation precision to be expressed as
/// explicit casts in HLIR: a 16-bit tensor summed with F32 accumulation is
/// written `x.cast(F32).sum(..).cast(dt)`. Lowered naively that is three
/// kernels and two F32-sized intermediate buffers. This op matches the
/// explicit pattern and unions a single kernel (16-bit loads, F32 Kahan
/// accumulation, 16-bit store) into the outer Cast's eclass — same dtype,
/// same semantics, one kernel.
#[derive(Default, Debug, Clone)]
pub struct KernelCastSumReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
    dtype: DType,
}

pub(crate) mod kernel_cast_sum {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_specialize(
        cast_size: &DimExpr,
        x: &Ir,
        shape: &EList,
        iters: &DimExpr,
        strides: &EList,
        iter_stride: &DimExpr,
        out_strides: &EList,
        out_size: &DimExpr,
        out_cast: &Ir,
    ) -> Vec<Rule> {
        let f16_sum = &Ir::Op(
            kernel_cast_sum(
                shape,
                iters,
                strides,
                iter_stride,
                out_strides,
                DType::F16(),
            ),
            [x],
        );
        let bf16_sum = &Ir::Op(
            kernel_cast_sum(
                shape,
                iters,
                strides,
                iter_stride,
                out_strides,
                DType::Bf16(),
            ),
            [x],
        );
        let x_cast = &Ir::Op(OpKind::Cast(cast_size, DType::F32()), [x]);
        let sum = &Ir::Op(
            OpKind::Sum(shape, iters, strides, iter_stride, out_strides),
            [x_cast],
        );
        vec![
            rule(
                (
                    eq(DType::F16(), x.dtype()),
                    eq(
                        out_cast,
                        Ir::Op(OpKind::Cast(out_size, DType::F16()), [sum]),
                    ),
                ),
                (
                    f16_sum,
                    union(out_cast, f16_sum),
                    set(f16_sum.dtype(), DType::F16()),
                ),
            )
            .label("kernel-cast-sum-F16"),
            rule(
                (
                    eq(DType::Bf16(), x.dtype()),
                    eq(
                        out_cast,
                        Ir::Op(OpKind::Cast(out_size, DType::Bf16()), [sum]),
                    ),
                ),
                (
                    bf16_sum,
                    union(out_cast, bf16_sum),
                    set(bf16_sum.dtype(), DType::Bf16()),
                ),
            )
            .label("kernel-cast-sum-Bf16"),
        ]
    }
}

impl KernelCastSumReduce {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((
            field_shape,
            field_iters,
            field_strides,
            field_iter_stride,
            field_out_strides,
            field_dtype,
        )) = typed::get_args(
            kind,
            |shape: &EList,
             iters: &DimExpr,
             strides: &EList,
             iter_stride: &DimExpr,
             out_strides: &EList,
             dtype: &luminal::typed_reference::schema::DType| {
                crate::typed_rules::kernel_cast_sum(
                    shape,
                    iters,
                    strides,
                    iter_stride,
                    out_strides,
                    dtype,
                )
            },
        )?
        else {
            return Ok(None);
        };
        let decoded_shape = context.expressions(&field_shape)?;
        let decoded_strides = context.expressions(&field_strides)?;
        let decoded_out_strides = context.expressions(&field_out_strides)?;
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            out_shape: decoded_shape,
            iters: context.expression(&field_iters)?,
            in_stride: decoded_strides,
            iter_stride: context.expression(&field_iter_stride)?,
            out_stride: decoded_out_strides,
            dtype: context.dtype(&field_dtype)?,
        })
            as Box<dyn KernelOp>)))
    }
}

impl KernelOp for KernelCastSumReduce {
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
            .chain(self.in_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.iters.dyn_vars())
            .chain(self.iter_stride.dyn_vars())
            .collect::<FxHashSet<_>>();

        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let n_outputs: Expression = self.out_shape.iter().copied().product();
        let threads_per_block = 256; // 8 warps per block
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let iter_stride_of_i = self.iter_stride.to_kernel().replace("const_z", "i");

        let kernel = format!(
            "{includes}
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define FULL_MASK 0xffffffff
{dyn_defines}
extern \"C\" {{
    __global__ void cast_reduce_sum_k({dtype} *out, const {dtype} *in_data{dyn_dims_param}) {{
        __shared__ float warp_sums[THREADS_PER_BLOCK / WARP_SIZE];
        long long const_z = blockIdx.x;

        int tid = threadIdx.x;
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        long long in_start = {in_index};
        long long iters = {iters};

        float partial = 0.0f;
        float comp = 0.0f;   // Kahan compensation
        for (long long i = tid; i < iters; i += THREADS_PER_BLOCK) {{
            float y = static_cast<float>(in_data[in_start + {iter_stride_of_i}]) - comp;
            float t = partial + y;
            comp = (t - partial) - y;
            partial = t;
        }}

        #pragma unroll
        for (int s = WARP_SIZE / 2; s > 0; s /= 2) {{
            partial += __shfl_down_sync(FULL_MASK, partial, s);
        }}

        if (lane_id == 0) {{
            warp_sums[warp_id] = partial;
        }}
        __syncthreads();

        if (warp_id == 0) {{
            int cnt = THREADS_PER_BLOCK / WARP_SIZE;
            float block_sum = tid < cnt ? warp_sums[tid] : 0.0f;

            #pragma unroll
            for (int s = cnt / 2; s > 0; s /= 2) {{
                block_sum += __shfl_down_sync(FULL_MASK, block_sum, s);
            }}

            if (tid == 0) {{
                out[{out_index}] = ({dtype})block_sum;
            }}
        }}
    }}
}}",
            dtype = dtype,
            in_index = flatten_strides(&self.out_shape, &self.in_stride).to_kernel(),
            out_index = flatten_strides(&self.out_shape, &self.out_stride).to_kernel(),
            iters = self.iters.to_kernel(),
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("cast_reduce_sum_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_outputs, 1.into(), 1.into()),                // grid
            (threads_per_block.into(), 1.into(), 1.into()), // blocks (warp-parallel)
            32.into(),                                      // shmem for warp_sums
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        (self.out_shape.iter().copied().product::<Expression>() * self.iters * self.dtype.bits())
            .ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product::<Expression>() * self.iters
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "CastSumReduce"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelGather {
    out_shape: Vec<Expression>,
    index_stride: Vec<Expression>,
    data_shape: Vec<Expression>,
    data_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

pub(crate) mod kernel_gather {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    // Match an open IList tail, as Gather's dtype rule does. A strict INil
    // misses deep-model Gathers whose tail eclass has merged with another
    // chain (for example YOLO's stacked make_contiguous chains).

    #[ruleset]
    pub(crate) fn kernel_lower(
        index_shape: &EList,
        index_strides: &EList,
        data_shape: &EList,
        data_strides: &EList,
        indexes: &Ir,
        data: &Ir,
        tail: &IList,
        dt: &DType,
    ) -> Rule {
        let source = &Ir::Op(
            OpKind::Gather(index_shape, index_strides, data_shape, data_strides),
            IList::ICons(indexes, IList::ICons(data, tail)),
        );
        rule(
            (source, eq(dt, data.dtype())),
            (union(
                source,
                Ir::Op(
                    kernel_gather(
                        index_shape,
                        index_strides,
                        data_shape,
                        data_strides,
                        EList::RowMajor(index_shape),
                        dt,
                    ),
                    [indexes, data],
                ),
            ),),
        )
    }
}

impl KernelGather {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((
            field_out_shape,
            field_index_strides,
            field_data_shape,
            field_data_strides,
            field_out_strides,
            field_dtype,
        )) = typed::get_args(
            kind,
            |out_shape: &EList,
             index_strides: &EList,
             data_shape: &EList,
             data_strides: &EList,
             out_strides: &EList,
             dtype: &luminal::typed_reference::schema::DType| {
                crate::typed_rules::kernel_gather(
                    out_shape,
                    index_strides,
                    data_shape,
                    data_strides,
                    out_strides,
                    dtype,
                )
            },
        )?
        else {
            return Ok(None);
        };
        let decoded_out_shape = context.expressions(&field_out_shape)?;
        let decoded_index_strides = context.expressions(&field_index_strides)?;
        let decoded_data_shape = context.expressions(&field_data_shape)?;
        let decoded_data_strides = context.expressions(&field_data_strides)?;
        let decoded_out_strides = context.expressions(&field_out_strides)?;
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            out_shape: decoded_out_shape,
            index_stride: decoded_index_strides,
            data_shape: decoded_data_shape,
            data_stride: decoded_data_strides,
            out_stride: decoded_out_strides,
            dtype: context.dtype(&field_dtype)?,
        }))))
    }
}

impl KernelOp for KernelGather {
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
            .chain(self.index_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.data_shape.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.data_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let idx_idx = flatten_strides(&self.out_shape, &self.index_stride).to_kernel();
        let data_idx = flatten_strides(&self.data_shape, &self.data_stride).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void gather({dtype} *C, const int *indexes, const {dtype} *data{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        {dtype}* out = C + {out_idx};
        const_z = indexes[{idx_idx}];
        *out = data[{data_idx}];
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("gather").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.out_shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(256), 1.into(), 1.into()),
            (out_size.min(256), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn all_dyn_vars(&self) -> FxHashSet<char> {
        self.out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.index_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.data_shape.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.data_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .collect()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        // Data + indices (indices are always int32)
        (self.output_size() * self.dtype.bits()).ceil_div(8) + self.output_size() * 4
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "Gather"
    }
}

// KernelScatter: inverse of gather - out = copy(dest); out[indexes[i]] = src[i]
// Two-phase: memcpy graph node copies dest→output, then scatter kernel runs in same CUDA graph.
#[derive(Debug, Clone)]
pub struct KernelScatter {
    dest_shape: Vec<Expression>,
    dest_strides: Vec<Expression>,
    index_shape: Vec<Expression>,
    index_strides: Vec<Expression>,
    src_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    dtype: DType,
}

impl Default for KernelScatter {
    fn default() -> Self {
        Self {
            dest_shape: Vec::new(),
            dest_strides: Vec::new(),
            index_shape: Vec::new(),
            index_strides: Vec::new(),
            src_strides: Vec::new(),
            out_strides: Vec::new(),
            dtype: DType::F32,
        }
    }
}

pub(crate) mod kernel_scatter {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(
        dest_shape: &EList,
        dest_strides: &EList,
        index_shape: &EList,
        index_strides: &EList,
        src_strides: &EList,
        dest: &Ir,
        indexes: &Ir,
        src: &Ir,
        dt: &DType,
    ) -> Rule {
        let source = &Ir::Op(
            OpKind::Scatter(
                dest_shape,
                dest_strides,
                index_shape,
                index_strides,
                src_strides,
            ),
            [dest, indexes, src],
        );
        rule(
            (source, eq(dt, src.dtype())),
            (union(
                source,
                Ir::Op(
                    kernel_scatter(
                        dest_shape,
                        dest_strides,
                        index_shape,
                        index_strides,
                        src_strides,
                        EList::RowMajor(dest_shape),
                        dt,
                    ),
                    [dest, indexes, src],
                ),
            ),),
        )
    }
}

impl KernelScatter {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((
            field_dest_shape,
            field_dest_strides,
            field_index_shape,
            field_index_strides,
            field_src_strides,
            field_out_strides,
            field_dtype,
        )) = typed::get_args(
            kind,
            |dest_shape: &EList,
             dest_strides: &EList,
             index_shape: &EList,
             index_strides: &EList,
             src_strides: &EList,
             out_strides: &EList,
             dtype: &luminal::typed_reference::schema::DType| {
                crate::typed_rules::kernel_scatter(
                    dest_shape,
                    dest_strides,
                    index_shape,
                    index_strides,
                    src_strides,
                    out_strides,
                    dtype,
                )
            },
        )?
        else {
            return Ok(None);
        };
        let decoded_dest_shape = context.expressions(&field_dest_shape)?;
        let decoded_dest_strides = context.expressions(&field_dest_strides)?;
        let decoded_index_shape = context.expressions(&field_index_shape)?;
        let decoded_index_strides = context.expressions(&field_index_strides)?;
        let decoded_src_strides = context.expressions(&field_src_strides)?;
        let decoded_out_strides = context.expressions(&field_out_strides)?;
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            dest_shape: decoded_dest_shape,
            dest_strides: decoded_dest_strides,
            index_shape: decoded_index_shape,
            index_strides: decoded_index_strides,
            src_strides: decoded_src_strides,
            out_strides: decoded_out_strides,
            dtype: context.dtype(&field_dtype)?,
        }))))
    }
}

impl KernelOp for KernelScatter {
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
        let all_vars: FxHashSet<char> = self
            .dest_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.dest_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.index_shape.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.index_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.src_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_strides.iter().flat_map(|e| e.dyn_vars()))
            .collect();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&all_vars);
        let dyn_dims_param = if all_vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        // Single-kernel scatter: copy dest→output then scatter src→output[indexes]
        // Launched as 1 block of 1024 threads with __syncthreads() barrier.
        let n_src_elements = self
            .index_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let n_dest_elements = self
            .dest_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let copy_dest_idx = flatten_strides(&self.dest_shape, &self.dest_strides).to_kernel();
        let copy_out_idx = flatten_strides(&self.dest_shape, &self.out_strides).to_kernel();
        let scatter_idx_idx = flatten_strides(&self.index_shape, &self.index_strides).to_kernel();
        let scatter_src_idx = flatten_strides(&self.index_shape, &self.src_strides).to_kernel();
        let scatter_kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void scatter(
        {dtype} *out, const {dtype} *dest, const int *indexes, const {dtype} *src{dyn_dims_param}
    ) {{
        int tid = threadIdx.x;
        long long n_dest = {n_dest_elements};
        long long n_src = {n_src_elements};
        // Phase 1: materialize dest into the contiguous output layout.
        // dest may be a strided or broadcast view, so copying dest[i] would read
        // past the physical source buffer for expanded tensors.
        for (long long const_z = tid; const_z < n_dest; const_z += blockDim.x) {{
            out[{copy_out_idx}] = dest[{copy_dest_idx}];
        }}
        __syncthreads();
        // Phase 2: scatter src → output[indexes[i]]
        for (long long const_z = tid; const_z < n_src; const_z += blockDim.x) {{
            int idx = indexes[{scatter_idx_idx}];
            if (idx >= 0 && idx < n_dest) {{
                out[idx] = src[{scatter_src_idx}];
            }}
        }}
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&scatter_kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx =
                compile_module_image_for_current_device(stream.context(), &scatter_kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("scatter").unwrap();
            compile_cache.insert(scatter_kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        (
            func,
            module,
            scatter_kernel,
            (1.into(), 1.into(), 1.into()),    // grid: 1 block
            (1024.into(), 1.into(), 1.into()), // block: 1024 threads
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.dest_shape.iter().copied().product()
    }

    fn all_dyn_vars(&self) -> FxHashSet<char> {
        self.dest_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.dest_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.index_shape.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.index_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.src_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_strides.iter().flat_map(|e| e.dyn_vars()))
            .collect()
    }

    fn output_bytes(&self) -> Expression {
        let elem_size: Expression = match self.dtype {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::Int => 4,
            DType::F16 | DType::Bf16 | DType::I16 | DType::U16 => 2,
            DType::Bool
            | DType::I8
            | DType::U8
            | DType::F8UE8M0
            | DType::F8E4M3
            | DType::F8E5M2 => 1,
            other => panic!("Unsupported dtype for scatter output_bytes: {other:?}"),
        }
        .into();
        self.output_size() * elem_size
    }

    fn build_params(
        &self,
        _stream: &Arc<CudaStream>,
        output_ptr: u64,
        input_ptrs: &[u64],
        _internal_bufs: &[CudaSlice<u8>],
        dyn_dims_ptr: u64,
    ) -> Vec<u64> {
        // params: (out, dest, indexes, src [, dyn_dims])
        // input_ptrs: [dest, indexes, src]
        let mut params = vec![output_ptr, input_ptrs[0], input_ptrs[1], input_ptrs[2]];
        if dyn_dims_ptr != 0 {
            params.push(dyn_dims_ptr);
        }
        params
    }

    fn bytes_loaded(&self) -> Expression {
        let data_elem_size: Expression = match self.dtype {
            DType::F64 | DType::I64 => 8,
            DType::F32 | DType::Int => 4,
            DType::F16 | DType::Bf16 | DType::I16 | DType::U16 => 2,
            DType::Bool
            | DType::I8
            | DType::U8
            | DType::F8UE8M0
            | DType::F8E4M3
            | DType::F8E5M2 => 1,
            other => panic!("Unsupported dtype for scatter bytes_loaded: {other:?}"),
        }
        .into();
        let n_src: Expression = self.index_shape.iter().copied().product();
        // dest (copy) + indices + src
        self.output_size() * data_elem_size + n_src * 4 + n_src * data_elem_size
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn output_data_input(&self) -> Option<usize> {
        Some(0) // output is derived from dest (input 0): copy dest→output then scatter
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "Scatter"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelIota {
    expr: Expression,
    range: Expression,
}

pub(crate) mod kernel_iota {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(expr: &DimExpr, range: &DimExpr, inputs: &IList) -> Rule {
        let source = &Ir::Op(OpKind::Iota(expr, range), inputs);
        let replacement = &Ir::Op(kernel_iota(expr, range), inputs);
        rule(
            source,
            (
                union(source, replacement),
                set(replacement.dtype(), DType::Int()),
            ),
        )
    }
}

impl KernelIota {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_expr, field_range)) =
            typed::get_args(kind, |expr: &DimExpr, range: &DimExpr| {
                crate::typed_rules::kernel_iota(expr, range)
            })?
        else {
            return Ok(None);
        };
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            expr: context.expression(&field_expr)?,
            range: context.expression(&field_range)?,
        }))))
    }
}

impl KernelOp for KernelIota {
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
        let mut vars = self.expr.dyn_vars().into_iter().collect::<FxHashSet<_>>();
        vars.extend(self.range.dyn_vars());
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let range = self.range.to_kernel();
        let kernel = format!(
            "
{dyn_defines}
extern \"C\" {{
    __global__ void iota_k(int *C{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {range}) return;
        C[const_z] = {};
    }}
}}",
            self.expr.to_kernel(),
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("iota_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        (
            func,
            module,
            kernel,
            (self.range.ceil_div(256), 1.into(), 1.into()),
            (256.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.range
    }

    fn output_bytes(&self) -> Expression {
        // Iota always outputs int32 (4 bytes)
        self.output_size() * 4
    }

    fn bytes_loaded(&self) -> Expression {
        0.into()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn output_dtype(&self) -> DType {
        DType::Int
    }

    fn kernel_name(&self) -> &'static str {
        "Iota"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelMod {
    out_shape: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

pub(crate) mod kernel_mod {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(
        shape: &EList,
        a_strides: &EList,
        b_strides: &EList,
        out_strides: &EList,
        inputs: &IList,
        dtype: &DType,
    ) -> Rule {
        let source = &Ir::Op(
            OpKind::Mod(shape, a_strides, b_strides, out_strides),
            inputs,
        );
        rule(
            (source, eq(dtype, source.dtype())),
            (union(
                source,
                Ir::Op(
                    kernel_mod(shape, a_strides, b_strides, out_strides, dtype),
                    inputs,
                ),
            ),),
        )
    }
}

impl KernelMod {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_shape, field_a_strides, field_b_strides, field_out_strides, field_dtype)) =
            typed::get_args(
                kind,
                |shape: &EList,
                 a_strides: &EList,
                 b_strides: &EList,
                 out_strides: &EList,
                 dtype: &luminal::typed_reference::schema::DType| {
                    crate::typed_rules::kernel_mod(shape, a_strides, b_strides, out_strides, dtype)
                },
            )?
        else {
            return Ok(None);
        };
        let decoded_shape = context.expressions(&field_shape)?;
        let decoded_a_strides = context.expressions(&field_a_strides)?;
        let decoded_b_strides = context.expressions(&field_b_strides)?;
        let decoded_out_strides = context.expressions(&field_out_strides)?;
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            out_shape: decoded_shape,
            a_stride: decoded_a_strides,
            b_stride: decoded_b_strides,
            out_stride: decoded_out_strides,
            dtype: context.dtype(&field_dtype)?,
        }))))
    }
}

impl KernelOp for KernelMod {
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
            .chain(self.a_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.b_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let a_idx = flatten_strides(&self.out_shape, &self.a_stride).to_kernel();
        let b_idx = flatten_strides(&self.out_shape, &self.b_stride).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void mod_k({dtype} *C, const {dtype} *A, const {dtype} *B{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        C[{out_idx}] = fmodf(A[{a_idx}], B[{b_idx}]);
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("mod_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.out_shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(256), 1.into(), 1.into()),
            (out_size.min(256), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        // Both inputs have same dtype
        self.output_bytes() * 2
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "Mod"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelLessThan {
    out_shape: Vec<Expression>,
    a_stride: Vec<Expression>,
    b_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
    dtype: DType,
}

pub(crate) mod kernel_less_than {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(
        shape: &EList,
        a_strides: &EList,
        b_strides: &EList,
        out_strides: &EList,
        inp_a: &Ir,
        inp_b: &Ir,
        dt: &DType,
    ) -> Rule {
        let source = &Ir::Op(
            OpKind::LessThan(shape, a_strides, b_strides, out_strides),
            [inp_a, inp_b],
        );
        rule(
            (source, eq(dt, inp_a.dtype())),
            (union(
                source,
                Ir::Op(
                    kernel_less_than(shape, a_strides, b_strides, out_strides, dt),
                    [inp_a, inp_b],
                ),
            ),),
        )
    }
}

impl KernelLessThan {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_shape, field_a_strides, field_b_strides, field_out_strides, field_dtype)) =
            typed::get_args(
                kind,
                |shape: &EList,
                 a_strides: &EList,
                 b_strides: &EList,
                 out_strides: &EList,
                 dtype: &luminal::typed_reference::schema::DType| {
                    crate::typed_rules::kernel_less_than(
                        shape,
                        a_strides,
                        b_strides,
                        out_strides,
                        dtype,
                    )
                },
            )?
        else {
            return Ok(None);
        };
        let decoded_shape = context.expressions(&field_shape)?;
        let decoded_a_strides = context.expressions(&field_a_strides)?;
        let decoded_b_strides = context.expressions(&field_b_strides)?;
        let decoded_out_strides = context.expressions(&field_out_strides)?;
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            out_shape: decoded_shape,
            a_stride: decoded_a_strides,
            b_stride: decoded_b_strides,
            out_stride: decoded_out_strides,
            dtype: context.dtype(&field_dtype)?,
        }))))
    }
}

impl KernelOp for KernelLessThan {
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
            .chain(self.a_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.b_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();
        let dtype = cuda_dtype(self.dtype);

        let includes = dtype_includes(&[self.dtype, self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let n_elements = self
            .out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .to_kernel();
        let out_idx = flatten_strides(&self.out_shape, &self.out_stride).to_kernel();
        let a_idx = flatten_strides(&self.out_shape, &self.a_stride).to_kernel();
        let b_idx = flatten_strides(&self.out_shape, &self.b_stride).to_kernel();
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void less_than_k(unsigned char *C, const {dtype} *A, const {dtype} *B{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_elements}) return;
        C[{out_idx}] = A[{a_idx}] < B[{b_idx}] ? 1 : 0;
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("less_than_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let out_size = self.out_shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(256), 1.into(), 1.into()),
            (out_size.min(256), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_bytes(&self) -> Expression {
        // LessThan outputs Bool (unsigned char, 1 byte per element)
        self.output_size()
    }

    fn bytes_loaded(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
            + (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn output_dtype(&self) -> DType {
        DType::Bool
    }

    fn kernel_name(&self) -> &'static str {
        "LessThan"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelConstant {
    value: f32,
    dtype: DType,
}

pub(crate) mod kernel_constant {
    use crate::typed_rules::*;
    use egglog_experimental::typed::builtins as egg;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(
        value: &egg::F64,
        inputs: &IList,
        val: &egg::F64,
        size: &DimExpr,
        cast: &Ir,
    ) -> Vec<Rule> {
        let source = &Ir::Op(kernel_constant(value, DType::F32()), inputs);
        let replacement = &Ir::Op(OpKind::Constant(value), inputs);
        let f16_constant = &Ir::Op(kernel_constant(val, DType::F16()), IList::INil());
        let bf16_constant = &Ir::Op(kernel_constant(val, DType::Bf16()), IList::INil());
        let f32_constant = &Ir::Op(kernel_constant(val, DType::F32()), IList::INil());
        let c = &Ir::Op(OpKind::Constant(val), IList::INil());
        vec![
            rule(
                replacement,
                (
                    union(replacement, source),
                    set(source.dtype(), DType::F32()),
                ),
            ),
            rule(
                (eq(cast, Ir::Op(OpKind::Cast(size, DType::F16()), [c])),),
                (
                    f16_constant,
                    union(cast, f16_constant),
                    set(f16_constant.dtype(), DType::F16()),
                ),
            )
            .label("kernel-constant-cast-F16"),
            rule(
                (eq(cast, Ir::Op(OpKind::Cast(size, DType::Bf16()), [c])),),
                (
                    bf16_constant,
                    union(cast, bf16_constant),
                    set(bf16_constant.dtype(), DType::Bf16()),
                ),
            )
            .label("kernel-constant-cast-Bf16"),
            rule(
                (eq(cast, Ir::Op(OpKind::Cast(size, DType::F32()), [c])),),
                (
                    f32_constant,
                    union(cast, f32_constant),
                    set(f32_constant.dtype(), DType::F32()),
                ),
            )
            .label("kernel-constant-cast-F32"),
        ]
    }
}

impl KernelConstant {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_value, field_dtype)) = typed::get_args(
            kind,
            |value: &egg::F64, dtype: &luminal::typed_reference::schema::DType| {
                crate::typed_rules::kernel_constant(value, dtype)
            },
        )?
        else {
            return Ok(None);
        };
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            value: f64::try_from(&field_value)? as f32,
            dtype: context.dtype(&field_dtype)?,
        }))))
    }
}

impl KernelOp for KernelConstant {
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
        let value_str = if self.value.is_nan() {
            "__int_as_float(0x7fc00000)".to_string()
        } else if self.value.is_infinite() {
            if self.value > 0.0 {
                "__int_as_float(0x7f800000)".to_string()
            } else {
                "__int_as_float(0xff800000)".to_string()
            }
        } else {
            format!("{:.10}f", self.value)
        };
        let cuda_ty = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let kernel = format!(
            "{includes}
extern \"C\" {{
    __global__ void constant_k({cuda_ty} *out) {{
        out[0] = ({cuda_ty})({value_str});
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("constant_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        (
            func,
            module,
            kernel,
            (1.into(), 1.into(), 1.into()),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        1.into()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        0.into()
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "Constant"
    }
}

#[derive(Default, Debug, Clone)]
pub struct KernelCast {
    size: Expression,
    in_dtype: DType,
    out_dtype: DType,
}

pub(crate) mod kernel_cast {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(size: &DimExpr, dtype: &DType, inp: &Ir, in_dt: &DType) -> Rule {
        let source = &Ir::Op(OpKind::Cast(size, dtype), [inp]);
        rule(
            (source, eq(in_dt, inp.dtype())),
            (union(
                source,
                Ir::Op(kernel_cast(size, in_dt, dtype), [inp]),
            ),),
        )
    }
}

impl KernelCast {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_size, field_dtype, field_src_dtype)) = typed::get_args(
            kind,
            |size: &DimExpr,
             dtype: &luminal::typed_reference::schema::DType,
             src_dtype: &luminal::typed_reference::schema::DType| {
                crate::typed_rules::kernel_cast(size, dtype, src_dtype)
            },
        )?
        else {
            return Ok(None);
        };
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            size: context.expression(&field_size)?,
            in_dtype: context.dtype(&field_dtype)?,
            out_dtype: context.dtype(&field_src_dtype)?,
        }))))
    }
}

impl KernelOp for KernelCast {
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
        let out_dtype = cuda_dtype(self.out_dtype);
        let includes = dtype_includes(&[self.in_dtype, self.out_dtype]);
        let vars = self.size.dyn_vars().into_iter().collect::<FxHashSet<_>>();
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let size = self.size.to_kernel();

        let kernel = if self.in_dtype.bits() < 8 {
            // Sub-byte packed types: multiple values packed per byte.
            // Extract the correct bits using bit-level addressing.
            let bits = self.in_dtype.bits();
            let in_cuda_type = cuda_dtype(self.in_dtype);
            let mask = (1u32 << bits) - 1;
            format!(
                "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void cast_k({out_dtype} *out, const unsigned char *in_raw{dyn_dims_param}) {{
        long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= {size}) return;
        long long bit_offset = idx * {bits};
        long long byte_idx = bit_offset >> 3;
        int bit_pos = (int)(bit_offset & 7);
        unsigned short raw = (unsigned short)in_raw[byte_idx];
        if (bit_pos + {bits} > 8) raw |= ((unsigned short)in_raw[byte_idx + 1]) << 8;
        {in_cuda_type} val;
        val.__x = (unsigned char)((raw >> bit_pos) & {mask}u);
        out[idx] = ({out_dtype})val;
    }}
}}"
            )
        } else {
            let in_dtype = cuda_dtype(self.in_dtype);
            format!(
                "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void cast_k({out_dtype} *out, const {in_dtype} *in{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {size}) return;
        out[const_z] = ({out_dtype})in[const_z];
    }}
}}"
            )
        };
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("cast_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        (
            func,
            module,
            kernel,
            (self.size.ceil_div(256), 1.into(), 1.into()),
            (256.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.size
    }

    fn output_bytes(&self) -> Expression {
        (self.size * self.out_dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        (self.size * self.in_dtype.bits()).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn output_dtype(&self) -> DType {
        self.out_dtype
    }

    fn kernel_name(&self) -> &'static str {
        "Cast"
    }
}

/// Thread-local global dim ordering override. When set, `generate_dyn_dims_defines`
/// uses this ordering for buffer indices instead of the kernel's local ordering.
/// This ensures all kernels in a CudaGraphOp use consistent indices into the shared
/// dyn_dims buffer.
thread_local! {
    static GLOBAL_DYN_DIMS: std::cell::RefCell<Option<Vec<char>>> = const { std::cell::RefCell::new(None) };
}

/// Set the global dyn dims ordering for subsequent kernel compilations.
pub fn set_global_dyn_dims(dims: Vec<char>) {
    GLOBAL_DYN_DIMS.with(|g| *g.borrow_mut() = Some(dims));
}

/// Clear the global dyn dims ordering.
pub fn clear_global_dyn_dims() {
    GLOBAL_DYN_DIMS.with(|g| *g.borrow_mut() = None);
}

/// Get the current global dyn dims ordering.
pub fn get_global_dyn_dims() -> Option<Vec<char>> {
    GLOBAL_DYN_DIMS.with(|g| g.borrow().clone())
}

/// Generate #define macros for dynamic dimensions that read from a shared dyn_dims buffer.
/// The buffer layout is alphabetically sorted by dim char for consistency.
/// Returns (defines_string, sorted_dims) where sorted_dims gives the order of dims in the buffer.
///
/// When a global dyn dims ordering is set (via `set_global_dyn_dims`), indices are based
/// on the global ordering to ensure consistency across kernels sharing a dyn_dims buffer.
pub fn generate_dyn_dims_defines(vars: &FxHashSet<char>) -> (String, Vec<char>) {
    if vars.is_empty() {
        return (String::new(), Vec::new());
    }
    // Check for global ordering override
    let global = GLOBAL_DYN_DIMS.with(|g| g.borrow().clone());
    if let Some(mut global_order) = global {
        // Use global ordering for indices - each dim gets its position in the global list
        // Dynamically extend the ordering if a kernel uses a dim not in the pre-scan
        let mut extended = false;
        for dim in vars.iter().sorted() {
            if !global_order.contains(dim) {
                global_order.push(*dim);
                extended = true;
            }
        }
        if extended {
            global_order.sort();
            // Update the thread-local so subsequent kernels see the extended ordering
            set_global_dyn_dims(global_order.clone());
        }
        let defines = vars
            .iter()
            .sorted()
            .map(|dim| {
                let idx = global_order
                    .iter()
                    .position(|d| d == dim)
                    .expect("Dim must be in global ordering after extension");
                format!("#define const_{dim} dyn_dims[{idx}]")
            })
            .collect::<Vec<_>>()
            .join("\n");
        return (defines, global_order);
    }
    // Default: local ordering
    let mut sorted_dims: Vec<char> = vars.iter().copied().collect();
    sorted_dims.sort();
    let defines = sorted_dims
        .iter()
        .enumerate()
        .map(|(idx, dim)| format!("#define const_{dim} dyn_dims[{idx}]"))
        .collect::<Vec<_>>()
        .join("\n");
    (defines, sorted_dims)
}

/// Get the offset for a dynamic dimension in the shared dyn_dims buffer.
/// Returns None if the dim is not in the set.
pub fn get_dyn_dim_offset(dim: char, sorted_dims: &[char]) -> Option<usize> {
    sorted_dims.iter().position(|&d| d == dim)
}

#[derive(Default, Debug, Clone)]
pub struct KernelEmbed {
    batch_shape: Vec<Expression>,  // batch dimensions (e.g., [seq_len])
    token_stride: Vec<Expression>, // stride for token_ids input
    out_stride: Vec<Expression>,   // stride for output
    embed_dim: Expression,         // embedding dimension
    dtype: DType,                  // embedding table / output dtype
}

pub(crate) mod kernel_embed {
    use crate::typed_rules::*;
    use egglog_experimental::typed::builtins as egg;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_specialize(
        idx_shape: &EList,
        embed_shape: &EList,
        indices: &Ir,
        embed_table: &Ir,
        gather: &Ir,
        add_out_stride: &EList,
        mul_result: &Ir,
        iota_result: &Ir,
        token_cast_stride: &EList,
        token_ids_cast: &Ir,
        mul_const: &Ir,
        token_ids: &Ir,
        embed_dim: &DimExpr,
        embed_dt: &DType,
        token_stride: &EList,
    ) -> Vec<Rule> {
        let batch_shape = &EList::RemoveNthFromEnd(idx_shape, 0_i64);
        let out_stride_batch = &EList::RemoveNthFromEnd(add_out_stride, 0_i64);
        let cast_embed = &Ir::Op(
            kernel_embed(
                batch_shape,
                token_cast_stride,
                out_stride_batch,
                embed_dim,
                embed_dt,
            ),
            [token_ids_cast, embed_table],
        );
        let embed = &Ir::Op(
            kernel_embed(
                batch_shape,
                token_stride,
                out_stride_batch,
                embed_dim,
                embed_dt,
            ),
            [token_ids, embed_table],
        );
        vec![
            rule(
                (
                    eq(
                        gather,
                        Ir::Op(
                            GatherArgs {
                                index_shape: idx_shape.into(),
                                data_shape: embed_shape.into(),
                                ..GatherArgs::fresh()
                            },
                            [indices, embed_table],
                        ),
                    ),
                    eq(idx_shape.len(), 2_i64),
                    eq(
                        indices,
                        Ir::Op(
                            AddArgs {
                                out_strides: add_out_stride.into(),
                                ..AddArgs::fresh()
                            },
                            [mul_result, iota_result],
                        ),
                    ),
                    eq(
                        mul_result,
                        Ir::Op(
                            MulArgs {
                                a_strides: token_cast_stride.into(),
                                ..MulArgs::fresh()
                            },
                            [token_ids_cast, mul_const],
                        ),
                    ),
                    // A genuine 1D-token embedding has one token stride,
                    // matching RemoveNthFromEnd(idx_shape). A 2D token view
                    // would produce mismatched ranks in flatten_strides.
                    eq(token_cast_stride.len(), 1_i64),
                    eq(token_ids_cast, Ir::Op(CastArgs::fresh(), [token_ids])),
                    eq(embed_dim, embed_shape.nth_from_end(0_i64)),
                    eq(embed_dt, embed_table.dtype()),
                ),
                (
                    batch_shape,
                    out_stride_batch,
                    cast_embed,
                    union(gather, cast_embed),
                    set(cast_embed.dtype(), embed_dt),
                ),
            )
            .label("kernel embed with cast mul"),
            rule(
                (
                    eq(
                        gather,
                        Ir::Op(
                            GatherArgs {
                                index_shape: idx_shape.into(),
                                data_shape: embed_shape.into(),
                                ..GatherArgs::fresh()
                            },
                            [indices, embed_table],
                        ),
                    ),
                    eq(idx_shape.len(), 2_i64),
                    eq(
                        indices,
                        Ir::Op(
                            AddArgs {
                                out_strides: add_out_stride.into(),
                                ..AddArgs::fresh()
                            },
                            [iota_result, mul_result],
                        ),
                    ),
                    eq(
                        mul_result,
                        Ir::Op(
                            MulArgs {
                                a_strides: token_cast_stride.into(),
                                ..MulArgs::fresh()
                            },
                            [token_ids_cast, mul_const],
                        ),
                    ),
                    // Keep the same 1D-token stride guard for reversed Add.
                    eq(token_cast_stride.len(), 1_i64),
                    eq(token_ids_cast, Ir::Op(CastArgs::fresh(), [token_ids])),
                    eq(embed_dim, embed_shape.nth_from_end(0_i64)),
                    eq(embed_dt, embed_table.dtype()),
                ),
                (
                    batch_shape,
                    out_stride_batch,
                    cast_embed,
                    union(gather, cast_embed),
                    set(cast_embed.dtype(), embed_dt),
                ),
            )
            .label("kernel embed with cast mul reversed"),
            rule(
                (
                    eq(
                        gather,
                        Ir::Op(
                            GatherArgs {
                                index_shape: idx_shape.into(),
                                data_shape: embed_shape.into(),
                                ..GatherArgs::fresh()
                            },
                            [indices, embed_table],
                        ),
                    ),
                    eq(idx_shape.len(), 2_i64),
                    eq(
                        indices,
                        Ir::Op(
                            AddArgs {
                                out_strides: add_out_stride.into(),
                                ..AddArgs::fresh()
                            },
                            [mul_result, iota_result],
                        ),
                    ),
                    eq(
                        mul_result,
                        Ir::Op(
                            MulArgs {
                                a_strides: token_stride.into(),
                                ..MulArgs::fresh()
                            },
                            [token_ids, mul_const],
                        ),
                    ),
                    eq(token_stride.len(), 1_i64),
                    eq(embed_dim, embed_shape.nth_from_end(0_i64)),
                    eq(embed_dt, embed_table.dtype()),
                ),
                (
                    batch_shape,
                    out_stride_batch,
                    embed,
                    union(gather, embed),
                    set(embed.dtype(), embed_dt),
                ),
            )
            .label("kernel embed with mul"),
            rule(
                (
                    eq(
                        gather,
                        Ir::Op(
                            GatherArgs {
                                index_shape: idx_shape.into(),
                                data_shape: embed_shape.into(),
                                ..GatherArgs::fresh()
                            },
                            [indices, embed_table],
                        ),
                    ),
                    eq(idx_shape.len(), 2_i64),
                    eq(
                        indices,
                        Ir::Op(
                            AddArgs {
                                out_strides: add_out_stride.into(),
                                ..AddArgs::fresh()
                            },
                            [iota_result, mul_result],
                        ),
                    ),
                    eq(
                        mul_result,
                        Ir::Op(
                            MulArgs {
                                a_strides: token_stride.into(),
                                ..MulArgs::fresh()
                            },
                            [token_ids, mul_const],
                        ),
                    ),
                    eq(token_stride.len(), 1_i64),
                    eq(embed_dim, embed_shape.nth_from_end(0_i64)),
                    eq(embed_dt, embed_table.dtype()),
                ),
                (
                    batch_shape,
                    out_stride_batch,
                    embed,
                    union(gather, embed),
                    set(embed.dtype(), embed_dt),
                ),
            )
            .label("kernel embed with mul reversed"),
        ]
    }
}

impl KernelEmbed {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((
            field_batch_shape,
            field_token_stride,
            field_out_stride,
            field_embed_dim,
            field_dtype,
        )) = typed::get_args(
            kind,
            |batch_shape: &EList,
             token_stride: &EList,
             out_stride: &EList,
             embed_dim: &DimExpr,
             dtype: &luminal::typed_reference::schema::DType| {
                crate::typed_rules::kernel_embed(
                    batch_shape,
                    token_stride,
                    out_stride,
                    embed_dim,
                    dtype,
                )
            },
        )?
        else {
            return Ok(None);
        };
        let decoded_batch_shape = context.expressions(&field_batch_shape)?;
        let decoded_token_stride = context.expressions(&field_token_stride)?;
        let decoded_out_stride = context.expressions(&field_out_stride)?;
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            batch_shape: decoded_batch_shape,
            token_stride: decoded_token_stride,
            out_stride: decoded_out_stride,
            embed_dim: context.expression(&field_embed_dim)?,
            dtype: context.dtype(&field_dtype)?,
        }))))
    }
}

impl KernelOp for KernelEmbed {
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
        let batch_size = self
            .batch_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(1);
        let vars = self
            .batch_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.token_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.embed_dim.dyn_vars())
            .collect::<FxHashSet<_>>();
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let token_offset_expr = flatten_strides(&self.batch_shape, &self.token_stride).to_kernel();
        let out_offset_expr = flatten_strides(&self.batch_shape, &self.out_stride).to_kernel();
        let embed_dim_expr = self.embed_dim.to_kernel();
        let total_threads = batch_size * self.embed_dim;
        let n_elements = total_threads.to_kernel();
        let cuda_ty = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void embed({cuda_ty} *out, const int *token_ids, const {cuda_ty} *embed_table{dyn_dims_param}) {{
        long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= {n_elements}) return;
        long long embed_dim = {embed_dim_expr};
        long long batch_idx = idx / embed_dim;
        long long embed_idx = idx % embed_dim;
        long long const_z = batch_idx;
        long long token_offset = {token_offset_expr};
        long long out_offset = {out_offset_expr};
        int token_id = token_ids[token_offset];
        out[out_offset + embed_idx] = embed_table[(long long)token_id * embed_dim + embed_idx];
    }}
}}"
        );
        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("embed").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        // Return empty constants map - we now use shared dyn_dims buffer
        let constants = FxHashMap::default();
        (
            func,
            module,
            kernel,
            (total_threads.ceil_div(256), 1.into(), 1.into()),
            (256.into(), 1.into(), 1.into()),
            0.into(),
            constants,
        )
    }

    fn output_size(&self) -> Expression {
        self.batch_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(1)
            * self.embed_dim
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        let batch_size = self
            .batch_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(1);
        // Load: 1 token ID (4 bytes) per batch + 1 embedding row per batch
        batch_size * ((self.embed_dim * self.dtype.bits()).ceil_div(8) + 4)
    }

    fn bytes_stored(&self) -> Expression {
        // Store: 1 embedding row per batch element
        self.output_bytes()
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn flops(&self) -> Expression {
        // No FLOPs - just memory copy
        0.into()
    }

    fn kernel_name(&self) -> &'static str {
        "Embed"
    }
}
