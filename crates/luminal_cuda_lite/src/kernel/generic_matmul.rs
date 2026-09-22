use egglog_experimental::typed::{self as typed, TypedError, builtins as egg};
use luminal::egglog_utils::DecodeContext;
use luminal::typed_reference::schema::{DimExpr, EList, IList, Ir, OpKind};
use std::sync::Arc;

use crate::{
    compile_module_image_for_current_device, cuda_dtype,
    kernel::{
        KernelOp,
        hlir::{dtype_includes, generate_dyn_dims_defines},
    },
};
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{op::*, prelude::*, shape::flatten_strides};

#[derive(Default, Debug, Clone)]
pub struct GenericMatmul {
    out_shape: Vec<Expression>,
    mul_shape: Vec<Expression>,
    k: Expression,
    lhs_strides: Vec<Expression>,
    rhs_strides: Vec<Expression>,
    sum_input_strides: Vec<Expression>,
    sum_iter_stride: Expression,
    out_strides: Vec<Expression>,
    dtype: DType,
}

pub(crate) mod rewrites {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn matmul_backend(lhs: &Ir, rhs: &Ir, sum: &Ir, dt: &DType) -> Rule {
        let product = MulArgs::fresh();
        let reduction = SumArgs::fresh();
        let generic = &Ir::Op(
            generic_matmul(
                &reduction.shape,
                &product.shape,
                &reduction.iters,
                &product.a_strides,
                &product.b_strides,
                &reduction.strides,
                &reduction.iter_stride,
                &reduction.out_strides,
                dt,
            ),
            [lhs, rhs],
        );
        let mul = &Ir::Op(product.clone(), [lhs, rhs]);
        rule(
            (
                eq(sum, Ir::Op(reduction.clone(), [mul])),
                eq(dt, sum.dtype()),
            ),
            (generic, union(sum, generic), set(generic.dtype(), dt)),
        )
        .label("generic-matmul-cuda-mul-sum")
    }

    #[ruleset]
    pub(crate) fn cleanup(
        lhs: &Ir,
        rhs: &Ir,
        mul: &Ir,
        out_shape: &EList,
        k: &DimExpr,
        sum_input_strides: &EList,
        sum_iter_stride: &DimExpr,
        out_strides: &EList,
        sum: &Ir,
        generic_inputs: &IList,
        dt: &DType,
        sum_inputs: &IList,
        sum_node: &Ir,
    ) -> Vec<Rule> {
        let sum_source = &Ir::Op(
            OpKind::Sum(
                out_shape,
                k,
                sum_input_strides,
                sum_iter_stride,
                out_strides,
            ),
            [mul],
        );
        let kernel_sum_source = &Ir::Op(
            kernel_sum(
                out_shape,
                k,
                sum_input_strides,
                sum_iter_stride,
                out_strides,
                dt,
            ),
            sum_inputs,
        );
        vec![
            rule(
                (
                    eq(mul, Ir::Op(MulArgs::fresh(), [lhs, rhs])),
                    eq(sum, sum_source),
                    eq(sum, Ir::Op(GenericMatmulArgs::fresh(), generic_inputs)),
                ),
                (delete(sum_source),),
            )
            .label("delete-sum-when-generic-matmul-exists"),
            rule(
                (
                    eq(sum_node, kernel_sum_source),
                    eq(sum_node, Ir::Op(GenericMatmulArgs::fresh(), generic_inputs)),
                ),
                (delete(kernel_sum_source),),
            )
            .label("delete-kernel-sum-when-generic-matmul-exists"),
        ]
    }
}

impl GenericMatmul {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((
            field_out_shape,
            field_mul_shape,
            field_k,
            field_lhs_strides,
            field_rhs_strides,
            field_sum_input_strides,
            field_sum_iter_stride,
            field_out_strides,
            field_dtype,
        )) = typed::get_args(
            kind,
            |out_shape: &EList,
             mul_shape: &EList,
             k: &DimExpr,
             lhs_strides: &EList,
             rhs_strides: &EList,
             sum_input_strides: &EList,
             sum_iter_stride: &DimExpr,
             out_strides: &EList,
             dtype: &luminal::typed_reference::schema::DType| {
                crate::typed_rules::generic_matmul(
                    out_shape,
                    mul_shape,
                    k,
                    lhs_strides,
                    rhs_strides,
                    sum_input_strides,
                    sum_iter_stride,
                    out_strides,
                    dtype,
                )
            },
        )?
        else {
            return Ok(None);
        };
        let decoded_out_shape = context.expressions(&field_out_shape)?;
        let decoded_mul_shape = context.expressions(&field_mul_shape)?;
        let decoded_lhs_strides = context.expressions(&field_lhs_strides)?;
        let decoded_rhs_strides = context.expressions(&field_rhs_strides)?;
        let decoded_sum_input_strides = context.expressions(&field_sum_input_strides)?;
        let decoded_out_strides = context.expressions(&field_out_strides)?;
        Ok(Some(LLIROp::new::<dyn KernelOp>(Box::new(Self {
            out_shape: decoded_out_shape,
            mul_shape: decoded_mul_shape,
            k: context.expression(&field_k)?,
            lhs_strides: decoded_lhs_strides,
            rhs_strides: decoded_rhs_strides,
            sum_input_strides: decoded_sum_input_strides,
            sum_iter_stride: context.expression(&field_sum_iter_stride)?,
            out_strides: decoded_out_strides,
            dtype: context.dtype(&field_dtype)?,
        }))))
    }
}

impl KernelOp for GenericMatmul {
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
        let vars = self.all_dyn_vars();
        let dtype = cuda_dtype(self.dtype);
        let includes = dtype_includes(&[self.dtype]);
        let (dyn_defines, _sorted_dims) = generate_dyn_dims_defines(&vars);
        let dyn_dims_param = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };

        let n_outputs = self.output_size();
        let sum_base_idx = flatten_strides(&self.out_shape, &self.sum_input_strides).to_kernel();
        let iter_offset = self.sum_iter_stride.to_kernel().replace("const_z", "i");
        let lhs_idx = flatten_strides(&self.mul_shape, &self.lhs_strides)
            .to_kernel()
            .replace("const_z", "mul_idx");
        let rhs_idx = flatten_strides(&self.mul_shape, &self.rhs_strides)
            .to_kernel()
            .replace("const_z", "mul_idx");
        let out_idx = flatten_strides(&self.out_shape, &self.out_strides).to_kernel();
        let k = self.k.to_kernel();

        let kernel = format!(
            "{includes}
#define WARP_SIZE 32
#define THREADS_PER_BLOCK 256
#define FULL_MASK 0xffffffff
{dyn_defines}
extern \"C\" {{
    __global__ void generic_matmul({dtype} *out, const {dtype} *lhs, const {dtype} *rhs{dyn_dims_param}) {{
        __shared__ float warp_sums[THREADS_PER_BLOCK / WARP_SIZE];
        long long const_z = blockIdx.x;
        if (const_z >= {n_outputs}) return;

        int tid = threadIdx.x;
        int lane_id = tid % WARP_SIZE;
        int warp_id = tid / WARP_SIZE;

        long long base_idx = {sum_base_idx};
        long long iters = {k};

        float partial = 0.0f;
        for (long long i = tid; i < iters; i += THREADS_PER_BLOCK) {{
            long long mul_idx = base_idx + {iter_offset};
            partial += static_cast<float>(lhs[{lhs_idx}]) * static_cast<float>(rhs[{rhs_idx}]);
        }}

        #pragma unroll
        for (int s = WARP_SIZE / 2; s > 0; s >>= 1) {{
            partial += __shfl_down_sync(FULL_MASK, partial, s);
        }}

        if (lane_id == 0) {{
            warp_sums[warp_id] = partial;
        }}
        __syncthreads();

        if (warp_id == 0) {{
            float block_sum = tid < (THREADS_PER_BLOCK / WARP_SIZE) ? warp_sums[tid] : 0.0f;

            #pragma unroll
            for (int s = (THREADS_PER_BLOCK / WARP_SIZE) / 2; s > 0; s >>= 1) {{
                block_sum += __shfl_down_sync(FULL_MASK, block_sum, s);
            }}

            if (tid == 0) {{
                out[{out_idx}] = ({dtype})block_sum;
            }}
        }}
    }}
}}",
            n_outputs = n_outputs.to_kernel(),
        );

        let (module, func) = if let Some((module, func)) = compile_cache.get(&kernel) {
            (module.clone(), func.clone())
        } else {
            let ptx = compile_module_image_for_current_device(stream.context(), &kernel).unwrap();
            let module = stream.context().load_module(ptx).unwrap();
            let func = module.load_function("generic_matmul").unwrap();
            compile_cache.insert(kernel.clone(), (module.clone(), func.clone()));
            (module, func)
        };

        (
            func,
            module,
            kernel,
            (n_outputs, 1.into(), 1.into()),
            (256.into(), 1.into(), 1.into()),
            32.into(),
            FxHashMap::default(),
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn all_dyn_vars(&self) -> FxHashSet<char> {
        self.out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.mul_shape.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.k.dyn_vars())
            .chain(self.lhs_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.rhs_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.sum_input_strides.iter().flat_map(|e| e.dyn_vars()))
            .chain(self.sum_iter_stride.dyn_vars())
            .chain(self.out_strides.iter().flat_map(|e| e.dyn_vars()))
            .collect()
    }

    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.dtype.bits()).ceil_div(8)
    }

    fn bytes_loaded(&self) -> Expression {
        (self.output_size() * self.k * self.dtype.bits() * 2).ceil_div(8)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }

    fn flops(&self) -> Expression {
        self.output_size() * self.k * 2
    }

    fn output_dtype(&self) -> DType {
        self.dtype
    }

    fn kernel_name(&self) -> &'static str {
        "GenericMatmul"
    }
}
