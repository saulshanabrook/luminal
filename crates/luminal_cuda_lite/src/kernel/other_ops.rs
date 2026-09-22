use egglog_experimental::typed::{self as typed, TypedError, builtins as egg};
use luminal::egglog_utils::DecodeContext;
use luminal::typed_reference::schema::{DimExpr, EList, IList, Ir, OpKind};
use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device, cuda_dtype,
    kernel::KernelOp,
    kernel::hlir::{dtype_includes, generate_dyn_dims_defines},
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use itertools::Itertools;
use luminal::{op::*, prelude::*};

#[derive(Default, Debug, Clone)]

// Mean fusion remains disabled: its union can expose genomes with accumulated
// floating-point error across layers. The unfused Sum + Mul(Recip(Cast(Iota)))
// path remains available.
pub struct KernelMeanReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
    dtype: DType,
}
impl KernelMeanReduce {
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
                crate::typed_rules::kernel_mean(
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

impl KernelOp for KernelMeanReduce {
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
        let threads_per_block: usize = 256; // 8 warps per block
        let n_warps = threads_per_block / 32;
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void reduce_mean_k({dtype} *out, const {dtype} *in{dyn_dims_param}) {{
        long long const_z = blockIdx.x;
        long long n_elements = {n_outputs};
        if (const_z >= n_elements) return;

        long long in_start = {in_index};
        long long iters = {iters};
        long long iter_stride = {iter_stride};

        float thread_sum = 0.0f;
        for (long long i = threadIdx.x; i < iters; i += {threads_per_block})
            thread_sum += (float)in[in_start + i * iter_stride];

        for (int offset = 16; offset > 0; offset >>= 1)
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);

        __shared__ float warp_sums[{n_warps}];
        int lane = threadIdx.x & 31;
        int warp = threadIdx.x >> 5;
        if (lane == 0) warp_sums[warp] = thread_sum;
        __syncthreads();

        if (threadIdx.x == 0) {{
            float sum = 0.0f;
            for (int w = 0; w < {n_warps}; w++) sum += warp_sums[w];
            out[{out_index}] = ({dtype})(sum / (float)iters);
        }}
    }}
}}",
            dtype = dtype,
            in_index = flatten_strides(&self.out_shape, &self.in_stride).to_kernel(),
            out_index = flatten_strides(&self.out_shape, &self.out_stride).to_kernel(),
            n_outputs = n_outputs.to_kernel(),
            iters = self.iters.to_kernel(),
            iter_stride = self
                .iter_stride
                .substitute('z', Expression::from(1))
                .simplify()
                .to_kernel(),
            threads_per_block = threads_per_block,
            n_warps = n_warps,
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("reduce_mean_k").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_outputs, 1.into(), 1.into()),                // grid
            (threads_per_block.into(), 1.into(), 1.into()), // block
            0.into(),                                       // shmem size
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
        let n_outputs: Expression = self.out_shape.iter().copied().product();
        n_outputs * self.iters + n_outputs
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "MeanReduce"
    }
}

// =============================================================================
// KernelScatterNoCopy: In-place scatter that writes directly to dest buffer
// without copying. The output buffer aliases the dest buffer.
// =============================================================================

#[derive(Debug, Clone)]
pub struct KernelScatterNoCopy {
    pub(crate) dest_shape: Vec<Expression>,
    pub(crate) dest_strides: Vec<Expression>,
    pub(crate) index_shape: Vec<Expression>,
    pub(crate) index_strides: Vec<Expression>,
    pub(crate) src_strides: Vec<Expression>,
    pub(crate) out_strides: Vec<Expression>,
    pub(crate) dtype: DType,
}

impl Default for KernelScatterNoCopy {
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

pub(crate) mod kernel_scatter_no_copy {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    // ConsumedBuffer permits an in-place scatter only when the destination
    // also represents the output layout; strided/broadcast destinations need
    // the regular Scatter's contiguous copy. Cleanup deletes shared claims;
    // base_cleanup resolves surviving claims to the source and deletes them.
    // Deleted claims cascade through IList/Op, leaving the copying alternative.
    #[relation]
    pub(crate) fn consumed_buffer_ilist_contains(a0: IList, a1: Ir);

    #[ruleset]
    pub(crate) fn cleanup(
        head: &Ir,
        tail: &IList,
        list: &IList,
        item: &Ir,
        a: &Ir,
        cb: &Ir,
        k1: &OpKind,
        ilist1: &IList,
        k2: &OpKind,
        ilist2: &IList,
    ) -> Vec<Rule> {
        let op1 = &Ir::Op(k1, ilist1);
        let op2 = &Ir::Op(k2, ilist2);
        vec![
            rule(
                (eq(list, IList::ICons(head, tail)),),
                (consumed_buffer_ilist_contains(list, head),),
            )
            .label("consumed-buffer-ilist-contains-head"),
            rule(
                (
                    eq(list, IList::ICons(head, tail)),
                    Fact::from(consumed_buffer_ilist_contains(tail, item)),
                ),
                (consumed_buffer_ilist_contains(list, item),),
            )
            .label("consumed-buffer-ilist-contains-tail"),
            rule(
                (
                    eq(cb, consumed_buffer(a)),
                    Fact::from(consumed_buffer_ilist_contains(ilist1, cb)),
                    ne(op1, op2),
                    Fact::from(consumed_buffer_ilist_contains(ilist2, a)),
                ),
                (delete(consumed_buffer(a)),),
            )
            .label("consumed-buffer-cleanup-shared-op-use"),
        ]
    }

    #[ruleset]
    pub(crate) fn buffer_reuse(
        ds: &EList,
        dst: &EList,
        is: &EList,
        istr: &EList,
        ss: &EList,
        os: &EList,
        dt: &DType,
        dest: &Ir,
        indexes: &Ir,
        src: &Ir,
        scatter: &Ir,
        dty: &DType,
    ) -> Rule {
        let consumed = &consumed_buffer(dest);
        let nocopy = &Ir::Op(
            kernel_scatter_no_copy(ds, dst, is, istr, ss, os, dt),
            [consumed, indexes, src],
        );
        rule(
            (
                eq(
                    scatter,
                    Ir::Op(
                        kernel_scatter(ds, dst, is, istr, ss, os, dt),
                        [dest, indexes, src],
                    ),
                ),
                eq(dst, os),
                eq(dty, src.dtype()),
            ),
            (
                consumed,
                nocopy,
                union(scatter, nocopy),
                set(nocopy.dtype(), dty),
            ),
        )
        .label("scatter to scatter-no-copy")
    }

    #[ruleset]
    pub(crate) fn dtype_prop(a: &Ir, cb: &Ir, dt: &DType) -> Rule {
        rule(
            (eq(cb, consumed_buffer(a)), eq(dt, a.dtype())),
            (set(cb.dtype(), dt),),
        )
        .label("consumed-buffer-dtype")
    }

    #[ruleset]
    // A surviving no-copy scatter dominates the copying one. Run this before
    // base_cleanup resolves ConsumedBuffer back to the destination.
    pub(crate) fn post_cleanup(
        dest: &Ir,
        cb: &Ir,
        ds: &EList,
        dst: &EList,
        is: &EList,
        istr: &EList,
        ss: &EList,
        os: &EList,
        dt: &DType,
        indexes: &Ir,
        src: &Ir,
        scatter: &Ir,
    ) -> Rule {
        let source = &Ir::Op(
            kernel_scatter(ds, dst, is, istr, ss, os, dt),
            [dest, indexes, src],
        );
        rule(
            (
                eq(cb, consumed_buffer(dest)),
                eq(scatter, source),
                Ir::Op(
                    kernel_scatter_no_copy(ds, dst, is, istr, ss, os, dt),
                    [cb, indexes, src],
                ),
            ),
            (delete(source),),
        )
        .label("scatter-no-copy-dominates-valid-consumed-buffer")
    }

    #[ruleset]
    pub(crate) fn base_cleanup(a: &Ir, cb: &Ir) -> Rule {
        rule(
            (eq(cb, consumed_buffer(a)),),
            (union(cb, a), delete(consumed_buffer(a))),
        )
        .label("consumed-buffer-resolve")
    }
}

impl KernelScatterNoCopy {
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
                crate::typed_rules::kernel_scatter_no_copy(
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

impl KernelOp for KernelScatterNoCopy {
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
        let scatter_idx_idx = flatten_strides(&self.index_shape, &self.index_strides).to_kernel();
        let scatter_src_idx = flatten_strides(&self.index_shape, &self.src_strides).to_kernel();
        let scatter_kernel = format!(
            "{includes}
{dyn_defines}
extern \"C\" {{
    __global__ void scatter_nocopy({dtype} *dest, const int *indexes, const {dtype} *src{dyn_dims_param}) {{
        long long const_z = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        if (const_z >= {n_src_elements}) return;
        int idx = indexes[{scatter_idx_idx}];
        if (idx >= 0 && idx < {n_dest_elements}) {{
            dest[idx] = src[{scatter_src_idx}];
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
            let func = module.load_function("scatter_nocopy").unwrap();
            compile_cache.insert(scatter_kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };
        let n_src: Expression = self.index_shape.iter().copied().product();
        (
            func,
            module,
            scatter_kernel,
            (n_src.ceil_div(256), 1.into(), 1.into()),
            (256.into(), 1.into(), 1.into()),
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
        _output_ptr: u64,
        input_ptrs: &[u64],
        _internal_bufs: &[CudaSlice<u8>],
        dyn_dims_ptr: u64,
    ) -> Vec<u64> {
        // scatter_nocopy kernel: (dest, indexes, src [, dyn_dims])
        // Write directly to dest buffer (input_ptrs[0]), NOT to output_ptr
        let mut params = vec![input_ptrs[0], input_ptrs[1], input_ptrs[2]];
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
        // Only load indices + src (no dest copy!)
        n_src * 4 + n_src * data_elem_size
    }

    fn bytes_stored(&self) -> Expression {
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
            other => panic!("Unsupported dtype for scatter bytes_stored: {other:?}"),
        }
        .into();
        let n_src: Expression = self.index_shape.iter().copied().product();
        // Only store the scattered elements
        n_src * data_elem_size
    }

    fn flops(&self) -> Expression {
        0.into()
    }

    fn output_aliases_input(&self) -> Option<usize> {
        Some(0) // output aliases dest (input 0)
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "ScatterNoCopy"
    }
}
