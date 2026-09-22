use super::{MetalEncodeContext, MetalKernelOp, MetalMulInfo, MetalSumReduceInfo};
use egglog_experimental::typed::{self as typed, TypedError, builtins as egg};
use luminal::egglog_utils::DecodeContext;
use luminal::typed_reference::schema::{DimExpr, EList, Ir, OpKind};
use luminal::{op::*, prelude::*, shape::flatten_strides};
use metal::{
    Buffer, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLLanguageVersion, MTLSize,
    foreign_types::{ForeignType, ForeignTypeRef},
};
use objc::runtime::Object;
use objc::{class, msg_send, sel, sel_impl};

fn compile_shader(device: &Device, source: &str, function_name: &str) -> ComputePipelineState {
    let options = metal::CompileOptions::new();
    options.set_language_version(MTLLanguageVersion::V2_4);
    let library = device
        .new_library_with_source(source, &options)
        .unwrap_or_else(|err| {
            panic!("Failed to compile Metal shader {function_name}: {err:?}\n{source}")
        });
    let function = library
        .get_function(function_name, None)
        .expect("Failed to get function from library");
    device
        .new_compute_pipeline_state_with_function(&function)
        .unwrap_or_else(|err| {
            panic!("Failed to create Metal compute pipeline state for {function_name}: {err:?}\n{source}")
        })
}

fn lower_dynamic_consts(mut code: String) -> String {
    for c in b'a'..=b'y' {
        let symbol = c as char;
        code = code.replace(
            &format!("const_{symbol}"),
            &format!("dyn[{}]", (c - b'a') as usize),
        );
    }
    code
}

pub(crate) fn lower_expression_for_metal(expr: &Expression, index_var: &str) -> String {
    lower_dynamic_consts(expr.to_kernel().replace("const_z", index_var))
}

fn metal_buffer_type(dtype: DType) -> &'static str {
    match dtype {
        DType::F32 => "float",
        DType::F16 => "half",
        DType::Int => "int",
        _ => panic!("Metal dtype {dtype:?} is not supported yet"),
    }
}

fn metal_numeric_read(dtype: DType, buffer: &str, index: &str) -> String {
    match dtype {
        DType::F32 => format!("{buffer}[{index}]"),
        DType::F16 => format!("float({buffer}[{index}])"),
        DType::Int => format!("float({buffer}[{index}])"),
        _ => panic!("Metal dtype {dtype:?} is not supported yet"),
    }
}

fn metal_numeric_write(dtype: DType, expr: &str) -> String {
    match dtype {
        DType::F32 => expr.to_string(),
        DType::F16 => format!("half({expr})"),
        DType::Int => format!("int({expr})"),
        _ => panic!("Metal dtype {dtype:?} is not supported yet"),
    }
}

fn metal_copy_value(dtype: DType, buffer: &str, index: &str) -> String {
    match dtype {
        DType::F32 | DType::F16 | DType::Int => format!("{buffer}[{index}]"),
        _ => panic!("Metal dtype {dtype:?} is not supported yet"),
    }
}

fn metal_binary_op_values(
    output_dtype: DType,
    a_dtype: DType,
    b_dtype: DType,
    a_idx: &str,
    b_idx: &str,
) -> (String, String) {
    let read: fn(DType, &str, &str) -> String = if output_dtype == DType::Int {
        metal_copy_value
    } else {
        metal_numeric_read
    };
    (read(a_dtype, "a", a_idx), read(b_dtype, "b", b_idx))
}

macro_rules! impl_unary_metrics {
    ($self:ident, $dyn_map:ident) => {
        fn bytes_loaded(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            let n = $self.output_size().exec($dyn_map).unwrap_or(0);
            n * std::mem::size_of::<f32>()
        }

        fn bytes_stored(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            let n = $self.output_size().exec($dyn_map).unwrap_or(0);
            n * std::mem::size_of::<f32>()
        }

        fn flops(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            $self.output_size().exec($dyn_map).unwrap_or(0)
        }
    };
}

/// Generate metrics methods for binary ops: 2 inputs read, 1 output write, flops_per_elem per element
macro_rules! impl_binary_metrics {
    ($self:ident, $dyn_map:ident, $flops_per_elem:expr) => {
        fn bytes_loaded(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            let n = $self.output_size().exec($dyn_map).unwrap_or(0);
            n * 2 * std::mem::size_of::<f32>()
        }

        fn bytes_stored(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            let n = $self.output_size().exec($dyn_map).unwrap_or(0);
            n * std::mem::size_of::<f32>()
        }

        fn flops(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            $self.output_size().exec($dyn_map).unwrap_or(0) * $flops_per_elem
        }
    };
}

/// Generate metrics methods for reduce ops
macro_rules! impl_reduce_metrics {
    ($self:ident, $dyn_map:ident) => {
        fn bytes_loaded(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            let n_outputs = $self.output_size().exec($dyn_map).unwrap_or(0);
            let iters = $self.iters.exec($dyn_map).unwrap_or(0);
            n_outputs * iters * std::mem::size_of::<f32>()
        }

        fn bytes_stored(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            let n = $self.output_size().exec($dyn_map).unwrap_or(0);
            n * std::mem::size_of::<f32>()
        }

        fn flops(&$self, $dyn_map: &FxHashMap<char, usize>) -> usize {
            let n_outputs = $self.output_size().exec($dyn_map).unwrap_or(0);
            let iters = $self.iters.exec($dyn_map).unwrap_or(0);
            n_outputs * iters
        }
    };
}

macro_rules! metal_unary_op {
    ($name:ident, $prepare:ident, $constructor:ident, $expr_builder:expr) => {
        #[derive(Debug, Default, Clone)]
        pub struct $name {
            shape: Vec<Expression>,
            input_strides: Vec<Expression>,
            output_strides: Vec<Expression>,
        }

        pub fn $prepare(
            kind: &OpKind,
            _inputs: &mut Vec<Ir>,
            context: &mut DecodeContext<'_, '_>,
        ) -> Result<Option<LLIROp>, TypedError> {
            let Some((shape, strides, out_strides)) = typed::get_args(
                kind,
                |shape: &EList, strides: &EList, out_strides: &EList| {
                    crate::typed_rules::$constructor(shape, strides, out_strides)
                },
            )?
            else {
                return Ok(None);
            };
            Ok(Some(LLIROp::new::<dyn MetalKernelOp>(Box::new($name {
                shape: context.expressions(&shape)?,
                input_strides: context.expressions(&strides)?,
                output_strides: context.expressions(&out_strides)?,
            }))))
        }

        impl MetalKernelOp for $name {
            fn compile(
                &self,
                device: &Device,
                input_dtypes: &[DType],
                output_dtype: DType,
            ) -> Option<ComputePipelineState> {
                let input_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
                let input_ty = metal_buffer_type(input_dtype);
                let output_ty = metal_buffer_type(output_dtype);
                // Generate strided index expressions
                let inp_index = flatten_strides(&self.shape, &self.input_strides);
                let out_index = flatten_strides(&self.shape, &self.output_strides);

                // Convert expressions to Metal code
                let inp_idx = lower_expression_for_metal(&inp_index, "idx");
                let out_idx = lower_expression_for_metal(&out_index, "idx");
                let input_expr = metal_numeric_read(input_dtype, "inp", &inp_idx);
                let body_expr = ($expr_builder)(&input_expr);
                let write_expr = metal_numeric_write(output_dtype, &body_expr);

                let source = format!(
                    r#"
                    #include <metal_stdlib>
                    using namespace metal;

                    kernel void mkernel(
                        device {input_ty} *inp [[buffer(0)]],
                        device {output_ty} *out [[buffer(1)]],
                        constant int *dyn [[buffer({dyn_buffer_index})]],
                        constant uint &n_elements [[buffer({n_elements_index})]],
                        uint idx [[thread_position_in_grid]]
                    ) {{
                        if (idx < n_elements) {{
                            out[{out_idx}] = {write_expr};
                        }}
                    }}
                    "#,
                    out_idx = out_idx,
                    input_ty = input_ty,
                    output_ty = output_ty,
                    write_expr = write_expr,
                    dyn_buffer_index = 2u64,
                    n_elements_index = 3u64,
                );
                Some(compile_shader(device, &source, "mkernel"))
            }

            fn output_size(&self) -> Expression {
                self.shape
                    .iter()
                    .cloned()
                    .product::<Expression>()
                    .max(Expression::from(1))
            }

            fn encode_compute(
                &self,
                encoder: &ComputeCommandEncoderRef,
                pipeline: &ComputePipelineState,
                inputs: &[&Buffer],
                output: &Buffer,
                dyn_map: &FxHashMap<char, usize>,
            ) {
                let n_elements = self.output_size().exec(dyn_map).unwrap() as u32;

                encoder.set_compute_pipeline_state(pipeline);
                encoder.set_buffer(0, Some(inputs[0]), 0);
                encoder.set_buffer(1, Some(output), 0);
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<u32>() as u64,
                    &n_elements as *const u32 as *const _,
                );

                let thread_group_size = MTLSize::new(256, 1, 1);
                let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
                encoder.dispatch_thread_groups(thread_groups, thread_group_size);
            }

            // Performance metrics for MBU/MFU (unary: 1 read, 1 write, 1 flop per element)
            impl_unary_metrics!(self, dyn_map);
        }
    };
}

pub(crate) mod metal_exp2 {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(
        shape: &EList,
        strides: &EList,
        out_strides: &EList,
        inp: &Ir,
        dt: &DType,
    ) -> Rule {
        let source = &Ir::Op(OpKind::Exp2(shape, strides, out_strides), [inp]);
        let replacement = &Ir::Op(metal_exp2(shape, strides, out_strides), [inp]);
        rule(
            (source, eq(dt, inp.dtype())),
            (
                union(source, replacement),
                subsume(source),
                set(replacement.dtype(), dt),
            ),
        )
    }
}

pub(crate) mod metal_log2 {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(
        shape: &EList,
        strides: &EList,
        out_strides: &EList,
        inp: &Ir,
        dt: &DType,
    ) -> Rule {
        let source = &Ir::Op(OpKind::Log2(shape, strides, out_strides), [inp]);
        let replacement = &Ir::Op(metal_log2(shape, strides, out_strides), [inp]);
        rule(
            (source, eq(dt, inp.dtype())),
            (
                union(source, replacement),
                subsume(source),
                set(replacement.dtype(), dt),
            ),
        )
    }
}

pub(crate) mod metal_sin {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(
        shape: &EList,
        strides: &EList,
        out_strides: &EList,
        inp: &Ir,
        dt: &DType,
    ) -> Rule {
        let source = &Ir::Op(OpKind::Sin(shape, strides, out_strides), [inp]);
        let replacement = &Ir::Op(metal_sin(shape, strides, out_strides), [inp]);
        rule(
            (source, eq(dt, inp.dtype())),
            (
                union(source, replacement),
                subsume(source),
                set(replacement.dtype(), dt),
            ),
        )
    }
}

pub(crate) mod metal_sqrt {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(
        shape: &EList,
        strides: &EList,
        out_strides: &EList,
        inp: &Ir,
        dt: &DType,
    ) -> Rule {
        let source = &Ir::Op(OpKind::Sqrt(shape, strides, out_strides), [inp]);
        let replacement = &Ir::Op(metal_sqrt(shape, strides, out_strides), [inp]);
        rule(
            (source, eq(dt, inp.dtype())),
            (
                union(source, replacement),
                subsume(source),
                set(replacement.dtype(), dt),
            ),
        )
    }
}

pub(crate) mod metal_recip {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(
        shape: &EList,
        strides: &EList,
        out_strides: &EList,
        inp: &Ir,
        dt: &DType,
    ) -> Rule {
        let source = &Ir::Op(OpKind::Recip(shape, strides, out_strides), [inp]);
        let replacement = &Ir::Op(metal_recip(shape, strides, out_strides), [inp]);
        rule(
            (source, eq(dt, inp.dtype())),
            (
                union(source, replacement),
                subsume(source),
                set(replacement.dtype(), dt),
            ),
        )
    }
}

metal_unary_op!(MetalExp2, prepare_exp2, metal_exp2, |x: &str| format!(
    "exp2({x})"
));
metal_unary_op!(MetalLog2, prepare_log2, metal_log2, |x: &str| format!(
    "log2({x})"
));
metal_unary_op!(MetalSin, prepare_sin, metal_sin, |x: &str| format!(
    "sin({x})"
));
metal_unary_op!(MetalSqrt, prepare_sqrt, metal_sqrt, |x: &str| format!(
    "sqrt({x})"
));
metal_unary_op!(MetalRecip, prepare_recip, metal_recip, |x: &str| format!(
    "1.0f / ({x})"
));

#[derive(Debug, Default, Clone)]
pub struct MetalAdd {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
}

pub(crate) mod metal_add {
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
    ) -> Vec<Rule> {
        let source = &Ir::Op(
            OpKind::Add(shape, a_strides, b_strides, out_strides),
            [inp_a, inp_b],
        );
        let replacement = &Ir::Op(
            metal_add(shape, a_strides, b_strides, out_strides),
            [inp_a, inp_b],
        );
        vec![
            rule(
                (source, eq(dt, inp_a.dtype())),
                (
                    union(source, replacement),
                    subsume(source),
                    set(replacement.dtype(), dt),
                ),
            ),
            rule(
                source,
                (
                    union(source, replacement),
                    subsume(source),
                    set(replacement.dtype(), DType::F32()),
                ),
            ),
        ]
    }
}

impl MetalAdd {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_shape, field_a_strides, field_b_strides, field_out_strides)) =
            typed::get_args(
                kind,
                |shape: &EList, a_strides: &EList, b_strides: &EList, out_strides: &EList| {
                    crate::typed_rules::metal_add(shape, a_strides, b_strides, out_strides)
                },
            )?
        else {
            return Ok(None);
        };
        let decoded_shape = context.expressions(&field_shape)?;
        let decoded_a_strides = context.expressions(&field_a_strides)?;
        let decoded_b_strides = context.expressions(&field_b_strides)?;
        let decoded_out_strides = context.expressions(&field_out_strides)?;
        Ok(Some(LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
            shape: decoded_shape,
            a_strides: decoded_a_strides,
            b_strides: decoded_b_strides,
            output_strides: decoded_out_strides,
        }))))
    }
}

impl MetalKernelOp for MetalAdd {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        let a_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let b_dtype = input_dtypes.get(1).copied().unwrap_or(a_dtype);
        let a_ty = metal_buffer_type(a_dtype);
        let b_ty = metal_buffer_type(b_dtype);
        let out_ty = metal_buffer_type(output_dtype);
        // Generate strided index expressions using 'z' = thread index
        let a_index = flatten_strides(&self.shape, &self.a_strides);
        let b_index = flatten_strides(&self.shape, &self.b_strides);
        let out_index = flatten_strides(&self.shape, &self.output_strides);

        // Convert expressions to Metal code, replacing 'const_z' with 'idx'
        let a_idx = lower_expression_for_metal(&a_index, "idx");
        let b_idx = lower_expression_for_metal(&b_index, "idx");
        let out_idx = lower_expression_for_metal(&out_index, "idx");
        let (a_val, b_val) = metal_binary_op_values(output_dtype, a_dtype, b_dtype, &a_idx, &b_idx);
        let out_val = metal_numeric_write(output_dtype, &format!("({a_val}) + ({b_val})"));

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device {a_ty} *a [[buffer(0)]],
                device {b_ty} *b [[buffer(1)]],
                device {out_ty} *out [[buffer(2)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                constant uint &n_elements [[buffer({n_elements_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_idx}] = {out_val};
                }}
            }}
            "#,
            a_ty = a_ty,
            b_ty = b_ty,
            out_ty = out_ty,
            out_val = out_val,
            dyn_buffer_index = 3u64,
            n_elements_index = 4u64,
        );
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self.output_size().exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }

    // Performance metrics for MBU/MFU (binary: 2 reads, 1 write, 1 flop per element)
    impl_binary_metrics!(self, dyn_map, 1);
}

#[derive(Debug, Default, Clone)]
pub struct MetalMul {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
}

pub(crate) mod metal_mul {
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
            OpKind::Mul(shape, a_strides, b_strides, out_strides),
            [inp_a, inp_b],
        );
        let replacement = &Ir::Op(
            metal_mul(shape, a_strides, b_strides, out_strides),
            [inp_a, inp_b],
        );
        rule(
            (source, eq(dt, inp_a.dtype())),
            (
                union(source, replacement),
                subsume(source),
                set(replacement.dtype(), dt),
            ),
        )
    }
}

impl MetalMul {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_shape, field_a_strides, field_b_strides, field_out_strides)) =
            typed::get_args(
                kind,
                |shape: &EList, a_strides: &EList, b_strides: &EList, out_strides: &EList| {
                    crate::typed_rules::metal_mul(shape, a_strides, b_strides, out_strides)
                },
            )?
        else {
            return Ok(None);
        };
        let decoded_shape = context.expressions(&field_shape)?;
        let decoded_a_strides = context.expressions(&field_a_strides)?;
        let decoded_b_strides = context.expressions(&field_b_strides)?;
        let decoded_out_strides = context.expressions(&field_out_strides)?;
        Ok(Some(LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
            shape: decoded_shape,
            a_strides: decoded_a_strides,
            b_strides: decoded_b_strides,
            output_strides: decoded_out_strides,
        }))))
    }
}

impl MetalKernelOp for MetalMul {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        let a_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let b_dtype = input_dtypes.get(1).copied().unwrap_or(a_dtype);
        let a_ty = metal_buffer_type(a_dtype);
        let b_ty = metal_buffer_type(b_dtype);
        let out_ty = metal_buffer_type(output_dtype);
        let a_index = flatten_strides(&self.shape, &self.a_strides);
        let b_index = flatten_strides(&self.shape, &self.b_strides);
        let out_index = flatten_strides(&self.shape, &self.output_strides);

        let a_idx = lower_expression_for_metal(&a_index, "idx");
        let b_idx = lower_expression_for_metal(&b_index, "idx");
        let out_idx = lower_expression_for_metal(&out_index, "idx");
        let (a_val, b_val) = metal_binary_op_values(output_dtype, a_dtype, b_dtype, &a_idx, &b_idx);
        let out_val = metal_numeric_write(output_dtype, &format!("({a_val}) * ({b_val})"));

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device {a_ty} *a [[buffer(0)]],
                device {b_ty} *b [[buffer(1)]],
                device {out_ty} *out [[buffer(2)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                constant uint &n_elements [[buffer({n_elements_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_idx}] = {out_val};
                }}
            }}
            "#,
            a_ty = a_ty,
            b_ty = b_ty,
            out_ty = out_ty,
            out_val = out_val,
            dyn_buffer_index = 3u64,
            n_elements_index = 4u64,
        );
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self.output_size().exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }

    // Performance metrics (binary: 2 reads, 1 write, 1 flop per element)
    impl_binary_metrics!(self, dyn_map, 1);

    fn mul_info(&self) -> Option<MetalMulInfo> {
        Some(MetalMulInfo {
            shape: self.shape.clone(),
            a_strides: self.a_strides.clone(),
            b_strides: self.b_strides.clone(),
            output_strides: self.output_strides.clone(),
        })
    }
}

// MetalMod: a % b using fmod
#[derive(Debug, Default, Clone)]
pub struct MetalMod {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
}

pub(crate) mod metal_mod {
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
            OpKind::Mod(shape, a_strides, b_strides, out_strides),
            [inp_a, inp_b],
        );
        let replacement = &Ir::Op(
            metal_mod(shape, a_strides, b_strides, out_strides),
            [inp_a, inp_b],
        );
        rule(
            (source, eq(dt, inp_a.dtype())),
            (
                union(source, replacement),
                subsume(source),
                set(replacement.dtype(), dt),
            ),
        )
    }
}

impl MetalMod {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_shape, field_a_strides, field_b_strides, field_out_strides)) =
            typed::get_args(
                kind,
                |shape: &EList, a_strides: &EList, b_strides: &EList, out_strides: &EList| {
                    crate::typed_rules::metal_mod(shape, a_strides, b_strides, out_strides)
                },
            )?
        else {
            return Ok(None);
        };
        let decoded_shape = context.expressions(&field_shape)?;
        let decoded_a_strides = context.expressions(&field_a_strides)?;
        let decoded_b_strides = context.expressions(&field_b_strides)?;
        let decoded_out_strides = context.expressions(&field_out_strides)?;
        Ok(Some(LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
            shape: decoded_shape,
            a_strides: decoded_a_strides,
            b_strides: decoded_b_strides,
            output_strides: decoded_out_strides,
        }))))
    }
}

impl MetalKernelOp for MetalMod {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        let a_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let b_dtype = input_dtypes.get(1).copied().unwrap_or(a_dtype);
        let a_ty = metal_buffer_type(a_dtype);
        let b_ty = metal_buffer_type(b_dtype);
        let out_ty = metal_buffer_type(output_dtype);
        let a_index = flatten_strides(&self.shape, &self.a_strides);
        let b_index = flatten_strides(&self.shape, &self.b_strides);
        let out_index = flatten_strides(&self.shape, &self.output_strides);

        let a_idx = lower_expression_for_metal(&a_index, "idx");
        let b_idx = lower_expression_for_metal(&b_index, "idx");
        let out_idx = lower_expression_for_metal(&out_index, "idx");
        let (a_val, b_val) = metal_binary_op_values(output_dtype, a_dtype, b_dtype, &a_idx, &b_idx);
        let out_expr = if output_dtype == DType::Int {
            format!("({a_val}) % ({b_val})")
        } else {
            format!("fmod({a_val}, {b_val})")
        };
        let out_val = metal_numeric_write(output_dtype, &out_expr);

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device {a_ty} *a [[buffer(0)]],
                device {b_ty} *b [[buffer(1)]],
                device {out_ty} *out [[buffer(2)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                constant uint &n_elements [[buffer({n_elements_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_idx}] = {out_val};
                }}
            }}
            "#,
            a_ty = a_ty,
            b_ty = b_ty,
            out_ty = out_ty,
            out_val = out_val,
            dyn_buffer_index = 3u64,
            n_elements_index = 4u64,
        );
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self.output_size().exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }

    // Performance metrics (binary: 2 reads, 1 write, ~10 flops for fmod)
    impl_binary_metrics!(self, dyn_map, 10);
}

// MetalLessThan: a < b ? 1.0 : 0.0
#[derive(Debug, Default, Clone)]
pub struct MetalLessThan {
    shape: Vec<Expression>,
    a_strides: Vec<Expression>,
    b_strides: Vec<Expression>,
    output_strides: Vec<Expression>,
}

pub(crate) mod metal_less_than {
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
        let replacement = &Ir::Op(
            metal_less_than(shape, a_strides, b_strides, out_strides),
            [inp_a, inp_b],
        );
        rule(
            (source, eq(dt, inp_a.dtype())),
            (
                union(source, replacement),
                subsume(source),
                set(replacement.dtype(), dt),
            ),
        )
    }
}

impl MetalLessThan {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_shape, field_a_strides, field_b_strides, field_out_strides)) =
            typed::get_args(
                kind,
                |shape: &EList, a_strides: &EList, b_strides: &EList, out_strides: &EList| {
                    crate::typed_rules::metal_less_than(shape, a_strides, b_strides, out_strides)
                },
            )?
        else {
            return Ok(None);
        };
        let decoded_shape = context.expressions(&field_shape)?;
        let decoded_a_strides = context.expressions(&field_a_strides)?;
        let decoded_b_strides = context.expressions(&field_b_strides)?;
        let decoded_out_strides = context.expressions(&field_out_strides)?;
        Ok(Some(LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
            shape: decoded_shape,
            a_strides: decoded_a_strides,
            b_strides: decoded_b_strides,
            output_strides: decoded_out_strides,
        }))))
    }
}

impl MetalKernelOp for MetalLessThan {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        let a_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let b_dtype = input_dtypes.get(1).copied().unwrap_or(a_dtype);
        let a_ty = metal_buffer_type(a_dtype);
        let b_ty = metal_buffer_type(b_dtype);
        let out_ty = metal_buffer_type(output_dtype);
        let a_index = flatten_strides(&self.shape, &self.a_strides);
        let b_index = flatten_strides(&self.shape, &self.b_strides);
        let out_index = flatten_strides(&self.shape, &self.output_strides);

        let a_idx = lower_expression_for_metal(&a_index, "idx");
        let b_idx = lower_expression_for_metal(&b_index, "idx");
        let out_idx = lower_expression_for_metal(&out_index, "idx");
        let a_val = metal_numeric_read(a_dtype, "a", &a_idx);
        let b_val = metal_numeric_read(b_dtype, "b", &b_idx);
        let out_val = metal_numeric_write(
            output_dtype,
            &format!("(({a_val}) < ({b_val})) ? 1.0f : 0.0f"),
        );

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device {a_ty} *a [[buffer(0)]],
                device {b_ty} *b [[buffer(1)]],
                device {out_ty} *out [[buffer(2)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                constant uint &n_elements [[buffer({n_elements_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_idx}] = {out_val};
                }}
            }}
            "#,
            a_ty = a_ty,
            b_ty = b_ty,
            out_ty = out_ty,
            out_val = out_val,
            dyn_buffer_index = 3u64,
            n_elements_index = 4u64,
        );
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        // Metal currently materializes comparisons as numeric 0/1 values.
        DType::F32
    }

    fn output_size(&self) -> Expression {
        self.shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self.output_size().exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }

    // Performance metrics (binary: 2 reads, 1 write, 1 comparison per element)
    impl_binary_metrics!(self, dyn_map, 1);
}

// ============================================================================
// Reduce Operations
// ============================================================================

#[derive(Debug, Default, Clone)]
pub struct MetalSumReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
}

pub(crate) mod metal_sum {
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
        inp: &Ir,
        dt: &DType,
    ) -> Rule {
        let source = &Ir::Op(
            OpKind::Sum(shape, iters, strides, iter_stride, out_strides),
            [inp],
        );
        let replacement = &Ir::Op(
            metal_sum(shape, iters, strides, iter_stride, out_strides),
            [inp],
        );
        rule(
            (source, eq(dt, inp.dtype())),
            (
                union(source, replacement),
                subsume(source),
                set(replacement.dtype(), dt),
            ),
        )
    }
}

impl MetalSumReduce {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_shape, field_iters, field_strides, field_iter_stride, field_out_strides)) =
            typed::get_args(
                kind,
                |shape: &EList,
                 iters: &DimExpr,
                 strides: &EList,
                 iter_stride: &DimExpr,
                 out_strides: &EList| {
                    crate::typed_rules::metal_sum(shape, iters, strides, iter_stride, out_strides)
                },
            )?
        else {
            return Ok(None);
        };
        let decoded_shape = context.expressions(&field_shape)?;
        let decoded_strides = context.expressions(&field_strides)?;
        let decoded_out_strides = context.expressions(&field_out_strides)?;
        Ok(Some(LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
            out_shape: decoded_shape,
            iters: context.expression(&field_iters)?,
            in_stride: decoded_strides,
            iter_stride: context.expression(&field_iter_stride)?,
            out_stride: decoded_out_strides,
        }))))
    }
}

impl MetalKernelOp for MetalSumReduce {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        let input_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let input_ty = metal_buffer_type(input_dtype);
        let output_ty = metal_buffer_type(output_dtype);
        let in_index = flatten_strides(&self.out_shape, &self.in_stride);
        let out_index = flatten_strides(&self.out_shape, &self.out_stride);

        let in_idx = lower_expression_for_metal(&in_index, "gid");
        let out_idx = lower_expression_for_metal(&out_index, "gid");
        let iters = lower_expression_for_metal(&self.iters, "gid");
        // iter_stride is an offset expression over the reduction-loop variable, not a scalar stride.
        let iter_offset = lower_expression_for_metal(&self.iter_stride, "i");
        let in_val = metal_numeric_read(input_dtype, "in", &format!("in_start + {iter_offset}"));
        let out_val = metal_numeric_write(output_dtype, "block_sum");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            #define THREADS_PER_GROUP 256

            kernel void mkernel(
                const device {input_ty} *in [[buffer(0)]],
                device {output_ty} *out [[buffer(1)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                constant uint &n_outputs [[buffer({n_outputs_index})]],
                uint gid [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]]
            ) {{
                if (gid >= n_outputs) return;

                threadgroup float partials[THREADS_PER_GROUP];

                int in_start = {in_idx};
                int iters = {iters};
                (void)dyn;

                float sum = 0.0f;
                for (int i = tid; i < iters; i += THREADS_PER_GROUP) {{
                    sum += {in_val};
                }}

                partials[tid] = sum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {{
                    if (tid < stride) {{
                        partials[tid] += partials[tid + stride];
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }}

                if (tid == 0) {{
                    float block_sum = partials[0];
                    out[{out_idx}] = {out_val};
                }}
            }}
            "#,
            input_ty = input_ty,
            output_ty = output_ty,
            in_val = in_val,
            out_val = out_val,
            dyn_buffer_index = 2u64,
            n_outputs_index = 3u64,
        );
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_outputs = self.output_size().exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &n_outputs as *const u32 as *const _,
        );

        // One threadgroup per output element
        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new(n_outputs as u64, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }

    // Performance metrics for reduce ops
    impl_reduce_metrics!(self, dyn_map);

    fn sum_reduce_info(&self) -> Option<MetalSumReduceInfo> {
        Some(MetalSumReduceInfo {
            shape: self.out_shape.clone(),
            strides: self.out_stride.clone(),
            iters: self.iters,
            iter_stride: self.iter_stride,
        })
    }
}

#[derive(Debug, Default, Clone)]
pub struct MetalMaxReduce {
    out_shape: Vec<Expression>,
    iters: Expression,
    in_stride: Vec<Expression>,
    iter_stride: Expression,
    out_stride: Vec<Expression>,
}

pub(crate) mod metal_max {
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
        inp: &Ir,
        dt: &DType,
    ) -> Rule {
        let source = &Ir::Op(
            OpKind::Max(shape, iters, strides, iter_stride, out_strides),
            [inp],
        );
        let replacement = &Ir::Op(
            metal_max(shape, iters, strides, iter_stride, out_strides),
            [inp],
        );
        rule(
            (source, eq(dt, inp.dtype())),
            (
                union(source, replacement),
                subsume(source),
                set(replacement.dtype(), dt),
            ),
        )
    }
}

impl MetalMaxReduce {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_shape, field_iters, field_strides, field_iter_stride, field_out_strides)) =
            typed::get_args(
                kind,
                |shape: &EList,
                 iters: &DimExpr,
                 strides: &EList,
                 iter_stride: &DimExpr,
                 out_strides: &EList| {
                    crate::typed_rules::metal_max(shape, iters, strides, iter_stride, out_strides)
                },
            )?
        else {
            return Ok(None);
        };
        let decoded_shape = context.expressions(&field_shape)?;
        let decoded_strides = context.expressions(&field_strides)?;
        let decoded_out_strides = context.expressions(&field_out_strides)?;
        Ok(Some(LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
            out_shape: decoded_shape,
            iters: context.expression(&field_iters)?,
            in_stride: decoded_strides,
            iter_stride: context.expression(&field_iter_stride)?,
            out_stride: decoded_out_strides,
        }))))
    }
}

impl MetalKernelOp for MetalMaxReduce {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        let input_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let input_ty = metal_buffer_type(input_dtype);
        let output_ty = metal_buffer_type(output_dtype);
        let in_index = flatten_strides(&self.out_shape, &self.in_stride);
        let out_index = flatten_strides(&self.out_shape, &self.out_stride);

        let in_idx = lower_expression_for_metal(&in_index, "gid");
        let out_idx = lower_expression_for_metal(&out_index, "gid");
        let iters = lower_expression_for_metal(&self.iters, "gid");
        // iter_stride is an offset expression over the reduction-loop variable, not a scalar stride.
        let iter_offset = lower_expression_for_metal(&self.iter_stride, "i");
        let in_val = metal_numeric_read(input_dtype, "in", &format!("in_start + {iter_offset}"));
        let out_val = metal_numeric_write(output_dtype, "block_max");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            #define THREADS_PER_GROUP 256
            #define NEG_INF_F (-INFINITY)

            kernel void mkernel(
                const device {input_ty} *in [[buffer(0)]],
                device {output_ty} *out [[buffer(1)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                constant uint &n_outputs [[buffer({n_outputs_index})]],
                uint gid [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]]
            ) {{
                if (gid >= n_outputs) return;

                threadgroup float partials[THREADS_PER_GROUP];

                int in_start = {in_idx};
                int iters = {iters};
                (void)dyn;

                float max_val = NEG_INF_F;
                for (int i = tid; i < iters; i += THREADS_PER_GROUP) {{
                    max_val = fmax(max_val, {in_val});
                }}

                partials[tid] = max_val;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {{
                    if (tid < stride) {{
                        partials[tid] = fmax(partials[tid], partials[tid + stride]);
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }}

                if (tid == 0) {{
                    float block_max = partials[0];
                    out[{out_idx}] = {out_val};
                }}
            }}
            "#,
            input_ty = input_ty,
            output_ty = output_ty,
            in_val = in_val,
            out_val = out_val,
            dyn_buffer_index = 2u64,
            n_outputs_index = 3u64,
        );
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_outputs = self.output_size().exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &n_outputs as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new(n_outputs as u64, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }

    // Performance metrics for reduce ops
    impl_reduce_metrics!(self, dyn_map);
}

#[derive(Debug, Default, Clone)]
pub struct MPSMatmul {
    pub m: Expression,
    pub n: Expression,
    pub k: Expression,
    pub lhs_row_stride: Expression,
    pub rhs_row_stride: Expression,
    pub out_row_stride: Expression,
    pub transpose_lhs: bool,
    pub transpose_rhs: bool,
}

pub(crate) mod m_p_s_matmul {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(
        m: &DimExpr,
        n: &DimExpr,
        k: &DimExpr,
        sum_in_strides: &EList,
        mul_output_strides: &EList,
        lhs: &Ir,
        rhs: &Ir,
        dt: &DType,
    ) -> Vec<Rule> {
        let row_row = &Ir::Op(
            metal_sum(
                [m, n],
                k,
                sum_in_strides,
                DimExpr::MIter(),
                [&(n * DimExpr::MIter()), &(DimExpr::MIter())],
            ),
            [&(Ir::Op(
                metal_mul(
                    [m, n, k],
                    [
                        &(k * DimExpr::MIter()),
                        &(DimExpr::MNum(0)),
                        &(DimExpr::MIter()),
                    ],
                    [
                        &(DimExpr::MNum(0)),
                        &(DimExpr::MIter()),
                        &(n * DimExpr::MIter()),
                    ],
                    mul_output_strides,
                ),
                IList::ICons(lhs, IList::ICons(rhs, IList::INil())),
            ))],
        );
        let mps_row_row = &Ir::Op(
            mps_matmul(
                m,
                n,
                k,
                k * DimExpr::MIter(),
                n * DimExpr::MIter(),
                n * DimExpr::MIter(),
                0_i64,
                0_i64,
            ),
            [lhs, rhs],
        );
        let row_col = &Ir::Op(
            metal_sum(
                [m, n],
                k,
                sum_in_strides,
                DimExpr::MIter(),
                [&(n * DimExpr::MIter()), &(DimExpr::MIter())],
            ),
            [&(Ir::Op(
                metal_mul(
                    [m, n, k],
                    [
                        &(k * DimExpr::MIter()),
                        &(DimExpr::MNum(0)),
                        &(DimExpr::MIter()),
                    ],
                    [
                        &(DimExpr::MNum(0)),
                        &(k * DimExpr::MIter()),
                        &(DimExpr::MIter()),
                    ],
                    mul_output_strides,
                ),
                IList::ICons(lhs, IList::ICons(rhs, IList::INil())),
            ))],
        );
        let mps_row_col = &Ir::Op(
            mps_matmul(
                m,
                n,
                k,
                k * DimExpr::MIter(),
                k * DimExpr::MIter(),
                n * DimExpr::MIter(),
                0_i64,
                1_i64,
            ),
            [lhs, rhs],
        );
        let col_row = &Ir::Op(
            metal_sum(
                [m, n],
                k,
                sum_in_strides,
                DimExpr::MIter(),
                [&(n * DimExpr::MIter()), &(DimExpr::MIter())],
            ),
            [&(Ir::Op(
                metal_mul(
                    [m, n, k],
                    [
                        &(DimExpr::MIter()),
                        &(DimExpr::MNum(0)),
                        &(m * DimExpr::MIter()),
                    ],
                    [
                        &(DimExpr::MNum(0)),
                        &(DimExpr::MIter()),
                        &(n * DimExpr::MIter()),
                    ],
                    mul_output_strides,
                ),
                IList::ICons(lhs, IList::ICons(rhs, IList::INil())),
            ))],
        );
        let mps_col_row = &Ir::Op(
            mps_matmul(
                m,
                n,
                k,
                m * DimExpr::MIter(),
                n * DimExpr::MIter(),
                n * DimExpr::MIter(),
                1_i64,
                0_i64,
            ),
            [lhs, rhs],
        );
        let col_col = &Ir::Op(
            metal_sum(
                [m, n],
                k,
                sum_in_strides,
                DimExpr::MIter(),
                [&(n * DimExpr::MIter()), &(DimExpr::MIter())],
            ),
            [&(Ir::Op(
                metal_mul(
                    [m, n, k],
                    [
                        &(DimExpr::MIter()),
                        &(DimExpr::MNum(0)),
                        &(m * DimExpr::MIter()),
                    ],
                    [
                        &(DimExpr::MNum(0)),
                        &(k * DimExpr::MIter()),
                        &(DimExpr::MIter()),
                    ],
                    mul_output_strides,
                ),
                IList::ICons(lhs, IList::ICons(rhs, IList::INil())),
            ))],
        );
        let mps_col_col = &Ir::Op(
            mps_matmul(
                m,
                n,
                k,
                m * DimExpr::MIter(),
                k * DimExpr::MIter(),
                n * DimExpr::MIter(),
                1_i64,
                1_i64,
            ),
            [lhs, rhs],
        );
        vec![
            rule(
                (row_row, eq(dt, row_row.dtype())),
                (union(row_row, mps_row_row), set(mps_row_row.dtype(), dt)),
            )
            .label("mps-matmul-row-row"),
            rule(
                (row_col, eq(dt, row_col.dtype())),
                (union(row_col, mps_row_col), set(mps_row_col.dtype(), dt)),
            )
            .label("mps-matmul-row-transposed-rhs"),
            rule(
                (col_row, eq(dt, col_row.dtype())),
                (union(col_row, mps_col_row), set(mps_col_row.dtype(), dt)),
            )
            .label("mps-matmul-transposed-lhs-row"),
            rule(
                (col_col, eq(dt, col_col.dtype())),
                (union(col_col, mps_col_col), set(mps_col_col.dtype(), dt)),
            )
            .label("mps-matmul-transposed-lhs-transposed-rhs"),
        ]
    }

    #[ruleset]
    pub(crate) fn cleanup(inputs: &IList, mul: &Ir, sum: &Ir, lhs: &Ir, rhs: &Ir) -> Rule {
        let source = &Ir::Op(MetalMulArgs::fresh(), inputs);
        let reduction = &Ir::Op(MetalSumArgs::fresh(), [mul]);
        rule(
            (
                eq(mul, source),
                eq(sum, reduction),
                eq(sum, Ir::Op(MPSMatmulArgs::fresh(), [lhs, rhs])),
            ),
            (delete(reduction), delete(source)),
        )
        .label("delete-broadcast-mul-sum-when-mps-matmul-exists")
    }
}

impl MPSMatmul {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((
            field_m,
            field_n,
            field_k,
            field_lhs_row_stride,
            field_rhs_row_stride,
            field_out_row_stride,
            field_transpose_lhs,
            field_transpose_rhs,
        )) = typed::get_args(
            kind,
            |m: &DimExpr,
             n: &DimExpr,
             k: &DimExpr,
             lhs_row_stride: &DimExpr,
             rhs_row_stride: &DimExpr,
             out_row_stride: &DimExpr,
             transpose_lhs: &egg::I64,
             transpose_rhs: &egg::I64| {
                crate::typed_rules::mps_matmul(
                    m,
                    n,
                    k,
                    lhs_row_stride,
                    rhs_row_stride,
                    out_row_stride,
                    transpose_lhs,
                    transpose_rhs,
                )
            },
        )?
        else {
            return Ok(None);
        };
        Ok(Some(LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
            m: context.expression(&field_m)?,
            n: context.expression(&field_n)?,
            k: context.expression(&field_k)?,
            lhs_row_stride: context.expression(&field_lhs_row_stride)?,
            rhs_row_stride: context.expression(&field_rhs_row_stride)?,
            out_row_stride: context.expression(&field_out_row_stride)?,
            transpose_lhs: match i64::try_from(&field_transpose_lhs)? {
                0 => false,
                1 => true,
                value => {
                    return Err(TypedError::Decode(format!(
                        "invalid transpose flag {value}"
                    )));
                }
            },
            transpose_rhs: match i64::try_from(&field_transpose_rhs)? {
                0 => false,
                1 => true,
                value => {
                    return Err(TypedError::Decode(format!(
                        "invalid transpose flag {value}"
                    )));
                }
            },
        }))))
    }
}

impl MPSMatmul {
    fn row_bytes(row_stride: Expression, dtype: DType, dyn_map: &FxHashMap<char, usize>) -> u64 {
        let elems = row_stride
            .substitute('z', Expression::from(1))
            .exec(dyn_map)
            .unwrap();
        (elems * dtype.bits().div_ceil(8)) as u64
    }

    fn matrix(buffer: &Buffer, descriptor: *mut Object) -> *mut Object {
        unsafe {
            let matrix: *mut Object = msg_send![class!(MPSMatrix), alloc];
            msg_send![matrix, initWithBuffer: buffer.as_ptr() descriptor: descriptor]
        }
    }

    fn matrix_with_offset(
        buffer: &Buffer,
        offset_bytes: u64,
        descriptor: *mut Object,
    ) -> *mut Object {
        unsafe {
            let matrix: *mut Object = msg_send![class!(MPSMatrix), alloc];
            msg_send![
                matrix,
                initWithBuffer: buffer.as_ptr()
                offset: offset_bytes as usize
                descriptor: descriptor
            ]
        }
    }
}

impl MetalKernelOp for MPSMatmul {
    fn compile(
        &self,
        _device: &Device,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        None
    }

    fn infer_output_dtype(&self, input_dtypes: &[DType]) -> DType {
        input_dtypes.first().copied().unwrap_or(DType::F32)
    }

    fn output_size(&self) -> Expression {
        self.m * self.n
    }

    fn encode_compute(
        &self,
        _encoder: &ComputeCommandEncoderRef,
        _pipeline: &ComputePipelineState,
        _inputs: &[&Buffer],
        _output: &Buffer,
        _dyn_map: &FxHashMap<char, usize>,
    ) {
        panic!("MPSMatmul encodes directly onto the command buffer")
    }

    fn encode(
        &self,
        context: &mut MetalEncodeContext<'_>,
        _pipeline: Option<&ComputePipelineState>,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) {
        assert_eq!(inputs.len(), 2, "MPSMatmul expects lhs and rhs inputs");
        let lhs_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let rhs_dtype = input_dtypes.get(1).copied().unwrap_or(lhs_dtype);
        let m = self.m.exec(dyn_map).unwrap();
        let n = self.n.exec(dyn_map).unwrap();
        let k = self.k.exec(dyn_map).unwrap();

        let lhs_rows = if self.transpose_lhs { k } else { m };
        let lhs_cols = if self.transpose_lhs { m } else { k };
        let rhs_rows = if self.transpose_rhs { n } else { k };
        let rhs_cols = if self.transpose_rhs { k } else { n };

        let (lhs_desc, rhs_desc, out_desc, kernel) = {
            let mut cache = context.mps_cache.borrow_mut();
            (
                cache.matrix_descriptor(
                    lhs_rows,
                    lhs_cols,
                    Self::row_bytes(self.lhs_row_stride, lhs_dtype, dyn_map),
                    lhs_dtype,
                ),
                cache.matrix_descriptor(
                    rhs_rows,
                    rhs_cols,
                    Self::row_bytes(self.rhs_row_stride, rhs_dtype, dyn_map),
                    rhs_dtype,
                ),
                cache.matrix_descriptor(
                    m,
                    n,
                    Self::row_bytes(self.out_row_stride, output_dtype, dyn_map),
                    output_dtype,
                ),
                cache.matrix_multiplication(
                    context.command_buffer,
                    self.transpose_lhs,
                    self.transpose_rhs,
                    m,
                    n,
                    k,
                    1.0,
                    0.0,
                ),
            )
        };

        let lhs = Self::matrix(inputs[0], lhs_desc);
        let rhs = Self::matrix(inputs[1], rhs_desc);
        let out = Self::matrix(output, out_desc);

        unsafe {
            let _: () = msg_send![
                kernel,
                encodeToCommandBuffer: context.command_buffer.as_ptr()
                leftMatrix: lhs
                rightMatrix: rhs
                resultMatrix: out
            ];
            let _: () = msg_send![lhs, release];
            let _: () = msg_send![rhs, release];
            let _: () = msg_send![out, release];
        }
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let m = self.m.exec(dyn_map).unwrap_or(0);
        let n = self.n.exec(dyn_map).unwrap_or(0);
        let k = self.k.exec(dyn_map).unwrap_or(0);
        2 * m * n * k * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let m = self.m.exec(dyn_map).unwrap_or(0);
        let n = self.n.exec(dyn_map).unwrap_or(0);
        m * n * std::mem::size_of::<f32>()
    }

    fn flops(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let m = self.m.exec(dyn_map).unwrap_or(0);
        let n = self.n.exec(dyn_map).unwrap_or(0);
        let k = self.k.exec(dyn_map).unwrap_or(0);
        2 * m * n * k
    }

    fn is_matmul(&self) -> bool {
        true
    }
}

#[derive(Debug, Default, Clone)]
pub struct MPSBatchedMatmul {
    pub batch: Expression,
    pub m: Expression,
    pub n: Expression,
    pub k: Expression,
    pub lhs_batch_stride: Expression,
    pub lhs_row_stride: Expression,
    pub rhs_batch_stride: Expression,
    pub rhs_row_stride: Expression,
    pub out_batch_stride: Expression,
    pub out_row_stride: Expression,
    pub transpose_lhs: bool,
    pub transpose_rhs: bool,
}

pub(crate) mod m_p_s_batched_matmul {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(
        batch: &DimExpr,
        m: &DimExpr,
        n: &DimExpr,
        k: &DimExpr,
        sum_in_strides: &EList,
        out_batch_stride: &DimExpr,
        out_row_stride: &DimExpr,
        lhs_batch_stride: &DimExpr,
        lhs_row_stride: &DimExpr,
        rhs_batch_stride: &DimExpr,
        rhs_row_stride: &DimExpr,
        mul_output_strides: &EList,
        lhs: &Ir,
        rhs: &Ir,
        dt: &DType,
    ) -> Vec<Rule> {
        let row_source = &Ir::Op(
            metal_sum(
                [batch, m, n],
                k,
                sum_in_strides,
                DimExpr::MIter(),
                [out_batch_stride, out_row_stride, &(DimExpr::MIter())],
            ),
            [&(Ir::Op(
                metal_mul(
                    [batch, m, n, k],
                    [
                        lhs_batch_stride,
                        lhs_row_stride,
                        &(DimExpr::MNum(0)),
                        &(DimExpr::MIter()),
                    ],
                    [
                        rhs_batch_stride,
                        &(DimExpr::MNum(0)),
                        &(DimExpr::MIter()),
                        rhs_row_stride,
                    ],
                    mul_output_strides,
                ),
                IList::ICons(lhs, IList::ICons(rhs, IList::INil())),
            ))],
        );
        let row_matmul = &Ir::Op(
            mps_batched_matmul(
                batch,
                m,
                n,
                k,
                lhs_batch_stride,
                lhs_row_stride,
                rhs_batch_stride,
                rhs_row_stride,
                out_batch_stride,
                out_row_stride,
                0_i64,
                0_i64,
            ),
            [lhs, rhs],
        );
        let transposed_source = &Ir::Op(
            metal_sum(
                [batch, m, n],
                k,
                sum_in_strides,
                DimExpr::MIter(),
                [out_batch_stride, out_row_stride, &(DimExpr::MIter())],
            ),
            [&(Ir::Op(
                metal_mul(
                    [batch, m, n, k],
                    [
                        lhs_batch_stride,
                        lhs_row_stride,
                        &(DimExpr::MNum(0)),
                        &(DimExpr::MIter()),
                    ],
                    [
                        rhs_batch_stride,
                        &(DimExpr::MNum(0)),
                        rhs_row_stride,
                        &(DimExpr::MIter()),
                    ],
                    mul_output_strides,
                ),
                IList::ICons(lhs, IList::ICons(rhs, IList::INil())),
            ))],
        );
        let transposed_matmul = &Ir::Op(
            mps_batched_matmul(
                batch,
                m,
                n,
                k,
                lhs_batch_stride,
                lhs_row_stride,
                rhs_batch_stride,
                rhs_row_stride,
                out_batch_stride,
                out_row_stride,
                0_i64,
                1_i64,
            ),
            [lhs, rhs],
        );
        let wrapped_row_source = &Ir::Op(
            metal_sum(
                [batch, m, n],
                k,
                sum_in_strides,
                DimExpr::MIter(),
                [out_batch_stride, out_row_stride, &(DimExpr::MIter())],
            ),
            [&(Ir::Op(
                metal_mul(
                    [batch, m, n, k],
                    [
                        lhs_batch_stride,
                        lhs_row_stride,
                        &(DimExpr::MNum(0)),
                        &((((DimExpr::MIter() / k) * k) * m) + (DimExpr::MIter() % k)),
                    ],
                    [
                        rhs_batch_stride,
                        &(DimExpr::MNum(0)),
                        &(DimExpr::MIter()),
                        rhs_row_stride,
                    ],
                    mul_output_strides,
                ),
                IList::ICons(lhs, IList::ICons(rhs, IList::INil())),
            ))],
        );
        let wrapped_transposed_source = &Ir::Op(
            metal_sum(
                [batch, m, n],
                k,
                sum_in_strides,
                DimExpr::MIter(),
                [out_batch_stride, out_row_stride, &(DimExpr::MIter())],
            ),
            [&(Ir::Op(
                metal_mul(
                    [batch, m, n, k],
                    [
                        lhs_batch_stride,
                        lhs_row_stride,
                        &(DimExpr::MNum(0)),
                        &((((DimExpr::MIter() / k) * k) * m) + (DimExpr::MIter() % k)),
                    ],
                    [
                        rhs_batch_stride,
                        &(DimExpr::MNum(0)),
                        rhs_row_stride,
                        &(DimExpr::MIter()),
                    ],
                    mul_output_strides,
                ),
                IList::ICons(lhs, IList::ICons(rhs, IList::INil())),
            ))],
        );
        vec![
            rule(
                (
                    row_source,
                    eq(dt, row_source.dtype()),
                    eq(lhs_row_stride, k * DimExpr::MIter()),
                    eq(rhs_row_stride, n * DimExpr::MIter()),
                    eq(out_row_stride, n * DimExpr::MIter()),
                ),
                (union(row_source, row_matmul), set(row_matmul.dtype(), dt)),
            )
            .label("mps-batched-matmul-row-row"),
            rule(
                (
                    transposed_source,
                    eq(dt, transposed_source.dtype()),
                    eq(lhs_row_stride, k * DimExpr::MIter()),
                    eq(rhs_row_stride, k * DimExpr::MIter()),
                    eq(out_row_stride, n * DimExpr::MIter()),
                ),
                (
                    union(transposed_source, transposed_matmul),
                    set(transposed_matmul.dtype(), dt),
                ),
            )
            .label("mps-batched-matmul-row-transposed-rhs"),
            rule(
                (
                    wrapped_row_source,
                    eq(dt, wrapped_row_source.dtype()),
                    eq(lhs_row_stride, k * DimExpr::MIter()),
                    eq(rhs_row_stride, n * DimExpr::MIter()),
                    eq(out_row_stride, n * DimExpr::MIter()),
                ),
                (
                    union(wrapped_row_source, row_matmul),
                    set(row_matmul.dtype(), dt),
                ),
            )
            .label("mps-batched-matmul-row-row-wrapped-inner"),
            rule(
                (
                    wrapped_transposed_source,
                    eq(dt, wrapped_transposed_source.dtype()),
                    eq(lhs_row_stride, k * DimExpr::MIter()),
                    eq(rhs_row_stride, k * DimExpr::MIter()),
                    eq(out_row_stride, n * DimExpr::MIter()),
                ),
                (
                    union(wrapped_transposed_source, transposed_matmul),
                    set(transposed_matmul.dtype(), dt),
                ),
            )
            .label("mps-batched-matmul-row-transposed-rhs-wrapped-inner"),
        ]
    }

    #[ruleset]
    pub(crate) fn cleanup(inputs: &IList, mul: &Ir, sum: &Ir, lhs: &Ir, rhs: &Ir) -> Rule {
        let source = &Ir::Op(MetalMulArgs::fresh(), inputs);
        let reduction = &Ir::Op(MetalSumArgs::fresh(), [mul]);
        rule(
            (
                eq(mul, source),
                eq(sum, reduction),
                eq(sum, Ir::Op(MPSBatchedMatmulArgs::fresh(), [lhs, rhs])),
            ),
            (delete(reduction), delete(source)),
        )
        .label("delete-broadcast-mul-sum-when-mps-batched-matmul-exists")
    }
}

impl MPSBatchedMatmul {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((
            field_batch,
            field_m,
            field_n,
            field_k,
            field_lhs_batch_stride,
            field_lhs_row_stride,
            field_rhs_batch_stride,
            field_rhs_row_stride,
            field_out_batch_stride,
            field_out_row_stride,
            field_transpose_lhs,
            field_transpose_rhs,
        )) = typed::get_args(
            kind,
            |batch: &DimExpr,
             m: &DimExpr,
             n: &DimExpr,
             k: &DimExpr,
             lhs_batch_stride: &DimExpr,
             lhs_row_stride: &DimExpr,
             rhs_batch_stride: &DimExpr,
             rhs_row_stride: &DimExpr,
             out_batch_stride: &DimExpr,
             out_row_stride: &DimExpr,
             transpose_lhs: &egg::I64,
             transpose_rhs: &egg::I64| {
                crate::typed_rules::mps_batched_matmul(
                    batch,
                    m,
                    n,
                    k,
                    lhs_batch_stride,
                    lhs_row_stride,
                    rhs_batch_stride,
                    rhs_row_stride,
                    out_batch_stride,
                    out_row_stride,
                    transpose_lhs,
                    transpose_rhs,
                )
            },
        )?
        else {
            return Ok(None);
        };
        Ok(Some(LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
            batch: context.expression(&field_batch)?,
            m: context.expression(&field_m)?,
            n: context.expression(&field_n)?,
            k: context.expression(&field_k)?,
            lhs_batch_stride: context.expression(&field_lhs_batch_stride)?,
            lhs_row_stride: context.expression(&field_lhs_row_stride)?,
            rhs_batch_stride: context.expression(&field_rhs_batch_stride)?,
            rhs_row_stride: context.expression(&field_rhs_row_stride)?,
            out_batch_stride: context.expression(&field_out_batch_stride)?,
            out_row_stride: context.expression(&field_out_row_stride)?,
            transpose_lhs: match i64::try_from(&field_transpose_lhs)? {
                0 => false,
                1 => true,
                value => {
                    return Err(TypedError::Decode(format!(
                        "invalid transpose flag {value}"
                    )));
                }
            },
            transpose_rhs: match i64::try_from(&field_transpose_rhs)? {
                0 => false,
                1 => true,
                value => {
                    return Err(TypedError::Decode(format!(
                        "invalid transpose flag {value}"
                    )));
                }
            },
        }))))
    }
}

impl MetalKernelOp for MPSBatchedMatmul {
    fn compile(
        &self,
        _device: &Device,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        None
    }

    fn infer_output_dtype(&self, input_dtypes: &[DType]) -> DType {
        input_dtypes.first().copied().unwrap_or(DType::F32)
    }

    fn output_size(&self) -> Expression {
        self.batch * self.m * self.n
    }

    fn encode_compute(
        &self,
        _encoder: &ComputeCommandEncoderRef,
        _pipeline: &ComputePipelineState,
        _inputs: &[&Buffer],
        _output: &Buffer,
        _dyn_map: &FxHashMap<char, usize>,
    ) {
        panic!("MPSBatchedMatmul encodes directly onto the command buffer")
    }

    fn encode(
        &self,
        context: &mut MetalEncodeContext<'_>,
        _pipeline: Option<&ComputePipelineState>,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) {
        assert_eq!(
            inputs.len(),
            2,
            "MPSBatchedMatmul expects lhs and rhs inputs"
        );
        let lhs_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let rhs_dtype = input_dtypes.get(1).copied().unwrap_or(lhs_dtype);
        let batch = self.batch.exec(dyn_map).unwrap();
        let m = self.m.exec(dyn_map).unwrap();
        let n = self.n.exec(dyn_map).unwrap();
        let k = self.k.exec(dyn_map).unwrap();

        let lhs_rows = if self.transpose_lhs { k } else { m };
        let lhs_cols = if self.transpose_lhs { m } else { k };
        let rhs_rows = if self.transpose_rhs { n } else { k };
        let rhs_cols = if self.transpose_rhs { k } else { n };

        let lhs_row_bytes = MPSMatmul::row_bytes(self.lhs_row_stride, lhs_dtype, dyn_map);
        let rhs_row_bytes = MPSMatmul::row_bytes(self.rhs_row_stride, rhs_dtype, dyn_map);
        let out_row_bytes = MPSMatmul::row_bytes(self.out_row_stride, output_dtype, dyn_map);
        let (lhs_desc, rhs_desc, out_desc, kernel) = {
            let mut cache = context.mps_cache.borrow_mut();
            (
                cache.matrix_descriptor(lhs_rows, lhs_cols, lhs_row_bytes, lhs_dtype),
                cache.matrix_descriptor(rhs_rows, rhs_cols, rhs_row_bytes, rhs_dtype),
                cache.matrix_descriptor(m, n, out_row_bytes, output_dtype),
                cache.matrix_multiplication(
                    context.command_buffer,
                    self.transpose_lhs,
                    self.transpose_rhs,
                    m,
                    n,
                    k,
                    1.0,
                    0.0,
                ),
            )
        };

        unsafe {
            for batch_idx in 0..batch {
                let batch_expr = Expression::from(batch_idx as i64);
                let lhs_offset = self
                    .lhs_batch_stride
                    .substitute('z', batch_expr)
                    .exec(dyn_map)
                    .unwrap()
                    * lhs_dtype.bits().div_ceil(8);
                let rhs_offset = self
                    .rhs_batch_stride
                    .substitute('z', batch_expr)
                    .exec(dyn_map)
                    .unwrap()
                    * rhs_dtype.bits().div_ceil(8);
                let out_offset = self
                    .out_batch_stride
                    .substitute('z', batch_expr)
                    .exec(dyn_map)
                    .unwrap()
                    * output_dtype.bits().div_ceil(8);

                let lhs = MPSMatmul::matrix_with_offset(inputs[0], lhs_offset as u64, lhs_desc);
                let rhs = MPSMatmul::matrix_with_offset(inputs[1], rhs_offset as u64, rhs_desc);
                let out = MPSMatmul::matrix_with_offset(output, out_offset as u64, out_desc);
                let _: () = msg_send![
                    kernel,
                    encodeToCommandBuffer: context.command_buffer.as_ptr()
                    leftMatrix: lhs
                    rightMatrix: rhs
                    resultMatrix: out
                ];
                let _: () = msg_send![lhs, release];
                let _: () = msg_send![rhs, release];
                let _: () = msg_send![out, release];
            }
        }
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let batch = self.batch.exec(dyn_map).unwrap_or(0);
        let m = self.m.exec(dyn_map).unwrap_or(0);
        let n = self.n.exec(dyn_map).unwrap_or(0);
        let k = self.k.exec(dyn_map).unwrap_or(0);
        2 * batch * m * n * k * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let batch = self.batch.exec(dyn_map).unwrap_or(0);
        let m = self.m.exec(dyn_map).unwrap_or(0);
        let n = self.n.exec(dyn_map).unwrap_or(0);
        batch * m * n * std::mem::size_of::<f32>()
    }

    fn flops(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let batch = self.batch.exec(dyn_map).unwrap_or(0);
        let m = self.m.exec(dyn_map).unwrap_or(0);
        let n = self.n.exec(dyn_map).unwrap_or(0);
        let k = self.k.exec(dyn_map).unwrap_or(0);
        2 * batch * m * n * k
    }

    fn is_matmul(&self) -> bool {
        true
    }
}

#[derive(Debug, Default, Clone)]
pub struct GenericMatmul {
    pub out_shape: Vec<Expression>,
    pub mul_shape: Vec<Expression>,
    pub k: Expression,
    pub lhs_strides: Vec<Expression>,
    pub rhs_strides: Vec<Expression>,
    pub sum_input_strides: Vec<Expression>,
    pub sum_iter_stride: Expression,
    pub out_strides: Vec<Expression>,
}

pub(crate) mod generic_matmul {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn matmul_backend(
        out_shape: &EList,
        k: &DimExpr,
        sum_iter_stride: &DimExpr,
        out_strides: &EList,
        mul_shape: &EList,
        lhs_strides: &EList,
        rhs_strides: &EList,
        lhs: &Ir,
        rhs: &Ir,
        dt: &DType,
        sum_input_strides: &EList,
        mul_output_strides: &EList,
    ) -> Rule {
        let source = &Ir::Op(
            metal_sum(
                out_shape,
                k,
                sum_input_strides,
                sum_iter_stride,
                out_strides,
            ),
            [&(Ir::Op(
                metal_mul(mul_shape, lhs_strides, rhs_strides, mul_output_strides),
                IList::ICons(lhs, IList::ICons(rhs, IList::INil())),
            ))],
        );
        let replacement = &Ir::Op(
            generic_matmul(
                out_shape,
                mul_shape,
                k,
                lhs_strides,
                rhs_strides,
                sum_input_strides,
                sum_iter_stride,
                out_strides,
            ),
            [lhs, rhs],
        );
        rule(
            (source, eq(dt, source.dtype())),
            (union(source, replacement), set(replacement.dtype(), dt)),
        )
        .label("generic-matmul-metal-mul-sum")
    }

    #[ruleset]
    pub(crate) fn cleanup(inputs: &IList, mul: &Ir, sum: &Ir, lhs: &Ir, rhs: &Ir) -> Rule {
        let source = &Ir::Op(MetalMulArgs::fresh(), inputs);
        let reduction = &Ir::Op(MetalSumArgs::fresh(), [mul]);
        rule(
            (
                eq(mul, source),
                eq(sum, reduction),
                eq(sum, Ir::Op(GenericMatmulArgs::fresh(), [lhs, rhs])),
            ),
            (delete(reduction), delete(source)),
        )
        .label("delete-broadcast-mul-sum-when-generic-matmul-exists")
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
        )) = typed::get_args(
            kind,
            |out_shape: &EList,
             mul_shape: &EList,
             k: &DimExpr,
             lhs_strides: &EList,
             rhs_strides: &EList,
             sum_input_strides: &EList,
             sum_iter_stride: &DimExpr,
             out_strides: &EList| {
                crate::typed_rules::generic_matmul(
                    out_shape,
                    mul_shape,
                    k,
                    lhs_strides,
                    rhs_strides,
                    sum_input_strides,
                    sum_iter_stride,
                    out_strides,
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
        Ok(Some(LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
            out_shape: decoded_out_shape,
            mul_shape: decoded_mul_shape,
            k: context.expression(&field_k)?,
            lhs_strides: decoded_lhs_strides,
            rhs_strides: decoded_rhs_strides,
            sum_input_strides: decoded_sum_input_strides,
            sum_iter_stride: context.expression(&field_sum_iter_stride)?,
            out_strides: decoded_out_strides,
        }))))
    }
}

impl MetalKernelOp for GenericMatmul {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        let lhs_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let rhs_dtype = input_dtypes.get(1).copied().unwrap_or(lhs_dtype);
        let lhs_ty = metal_buffer_type(lhs_dtype);
        let rhs_ty = metal_buffer_type(rhs_dtype);
        let out_ty = metal_buffer_type(output_dtype);

        let sum_base = flatten_strides(&self.out_shape, &self.sum_input_strides);
        let sum_base_idx = lower_expression_for_metal(&sum_base, "gid");
        let iter_offset = lower_expression_for_metal(&self.sum_iter_stride, "i");
        let lhs_index = flatten_strides(&self.mul_shape, &self.lhs_strides);
        let rhs_index = flatten_strides(&self.mul_shape, &self.rhs_strides);
        let out_index = flatten_strides(&self.out_shape, &self.out_strides);
        let lhs_idx = lower_expression_for_metal(&lhs_index, "mul_idx");
        let rhs_idx = lower_expression_for_metal(&rhs_index, "mul_idx");
        let out_idx = lower_expression_for_metal(&out_index, "gid");
        let iters = lower_expression_for_metal(&self.k, "gid");
        let lhs_val = metal_numeric_read(lhs_dtype, "lhs", &lhs_idx);
        let rhs_val = metal_numeric_read(rhs_dtype, "rhs", &rhs_idx);
        let out_val = metal_numeric_write(output_dtype, "block_sum");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            #define THREADS_PER_GROUP 256

            kernel void mkernel(
                const device {lhs_ty} *lhs [[buffer(0)]],
                const device {rhs_ty} *rhs [[buffer(1)]],
                device {out_ty} *out [[buffer(2)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                constant uint &n_outputs [[buffer({n_outputs_index})]],
                uint gid [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]]
            ) {{
                if (gid >= n_outputs) return;

                threadgroup float partials[THREADS_PER_GROUP];
                int base_idx = {sum_base_idx};
                int iters = {iters};
                (void)dyn;

                float sum = 0.0f;
                for (int i = tid; i < iters; i += THREADS_PER_GROUP) {{
                    int mul_idx = base_idx + {iter_offset};
                    sum += ({lhs_val}) * ({rhs_val});
                }}

                partials[tid] = sum;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint stride = THREADS_PER_GROUP / 2; stride > 0; stride >>= 1) {{
                    if (tid < stride) {{
                        partials[tid] += partials[tid + stride];
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }}

                if (tid == 0) {{
                    float block_sum = partials[0];
                    out[{out_idx}] = {out_val};
                }}
            }}
            "#,
            lhs_ty = lhs_ty,
            rhs_ty = rhs_ty,
            out_ty = out_ty,
            sum_base_idx = sum_base_idx,
            iters = iters,
            iter_offset = iter_offset,
            lhs_val = lhs_val,
            rhs_val = rhs_val,
            out_idx = out_idx,
            out_val = out_val,
            dyn_buffer_index = 3u64,
            n_outputs_index = 4u64,
        );
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn infer_output_dtype(&self, input_dtypes: &[DType]) -> DType {
        input_dtypes.first().copied().unwrap_or(DType::F32)
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_outputs = self.output_size().exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &n_outputs as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new(n_outputs as u64, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n_outputs = self.output_size().exec(dyn_map).unwrap_or(0);
        let k = self.k.exec(dyn_map).unwrap_or(0);
        2 * n_outputs * k * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        self.output_size().exec(dyn_map).unwrap_or(0) * std::mem::size_of::<f32>()
    }

    fn flops(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n_outputs = self.output_size().exec(dyn_map).unwrap_or(0);
        let k = self.k.exec(dyn_map).unwrap_or(0);
        2 * n_outputs * k
    }

    fn is_matmul(&self) -> bool {
        true
    }
}

// ============================================================================
// Data Operations
// ============================================================================

// MetalConstant: produces a single float value
#[derive(Debug, Default, Clone)]
pub struct MetalConstant {
    value: f32,
}

pub(crate) mod metal_constant {
    use crate::typed_rules::*;
    use egglog_experimental::typed::builtins as egg;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(value: &egg::F64) -> Rule {
        let source = &Ir::Op(OpKind::Constant(value), IList::INil());
        let replacement = &Ir::Op(metal_constant(value), IList::INil());
        rule(
            source,
            (
                union(source, replacement),
                subsume(source),
                set(replacement.dtype(), DType::F32()),
            ),
        )
    }
}

impl MetalConstant {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        _context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_value,)) = typed::get_args(kind, |value: &egg::F64| {
            crate::typed_rules::metal_constant(value)
        })?
        else {
            return Ok(None);
        };
        Ok(Some(LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
            value: f64::try_from(&field_value)? as f32,
        }))))
    }
}

impl MetalKernelOp for MetalConstant {
    fn compile(
        &self,
        device: &Device,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        // Ensure value is formatted with decimal point for Metal (e.g., -1.0f not -1f)
        let value_str = if self.value.fract() == 0.0 {
            format!("{:.1}", self.value)
        } else {
            format!("{}", self.value)
        };

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device float *out [[buffer(0)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx == 0) {{
                    out[0] = {value}f;
                }}
            }}
            "#,
            value = value_str,
            dyn_buffer_index = 1u64,
        );
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn output_size(&self) -> Expression {
        Expression::from(1)
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        DType::F32
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        _inputs: &[&Buffer],
        output: &Buffer,
        _dyn_map: &FxHashMap<char, usize>,
    ) {
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(output), 0);

        let thread_group_size = MTLSize::new(1, 1, 1);
        let thread_groups = MTLSize::new(1, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }
}

// MetalIota: generates sequence [expr(0), expr(1), ..., expr(range-1)]
#[derive(Debug, Default, Clone)]
pub struct MetalIota {
    expr: Expression,
    range: Expression,
}

pub(crate) mod metal_iota {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(expr: &DimExpr, range: &DimExpr) -> Rule {
        let source = &Ir::Op(OpKind::Iota(expr, range), IList::INil());
        let replacement = &Ir::Op(metal_iota(expr, range), IList::INil());
        rule(
            source,
            (
                union(source, replacement),
                subsume(source),
                set(replacement.dtype(), DType::Int()),
            ),
        )
    }
}

impl MetalIota {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_expr, field_range)) =
            typed::get_args(kind, |expr: &DimExpr, range: &DimExpr| {
                crate::typed_rules::metal_iota(expr, range)
            })?
        else {
            return Ok(None);
        };
        Ok(Some(LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
            expr: context.expression(&field_expr)?,
            range: context.expression(&field_range)?,
        }))))
    }
}

impl MetalKernelOp for MetalIota {
    fn compile(
        &self,
        device: &Device,
        _input_dtypes: &[DType],
        _output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        // Generate the expression as Metal code
        let expr_code = lower_expression_for_metal(&self.expr, "idx");

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device int *out [[buffer(0)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                constant uint &n_elements [[buffer({n_elements_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[idx] = (int)({expr});
                }}
            }}
            "#,
            expr = expr_code,
            dyn_buffer_index = 1u64,
            n_elements_index = 2u64,
        );
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn output_size(&self) -> Expression {
        self.range
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        DType::Int
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        _inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self.range.exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(output), 0);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }
}

// MetalGather: indexed lookup - out[i] = data[indexes[i]]
#[derive(Debug, Default, Clone)]
pub struct MetalGather {
    out_shape: Vec<Expression>,
    index_stride: Vec<Expression>,
    data_shape: Vec<Expression>,
    data_stride: Vec<Expression>,
    out_stride: Vec<Expression>,
}

pub(crate) mod metal_gather {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(
        index_shape: &EList,
        index_strides: &EList,
        data_shape: &EList,
        data_strides: &EList,
        indexes: &Ir,
        data: &Ir,
        dt: &DType,
    ) -> Rule {
        let source = &Ir::Op(
            metal_gather(
                index_shape,
                index_strides,
                data_shape,
                data_strides,
                EList::RowMajor(index_shape),
            ),
            [indexes, data],
        );
        let replacement = &Ir::Op(
            OpKind::Gather(index_shape, index_strides, data_shape, data_strides),
            [indexes, data],
        );
        rule(
            (replacement, eq(dt, data.dtype())),
            (
                union(replacement, source),
                subsume(replacement),
                set(source.dtype(), dt),
            ),
        )
    }
}

impl MetalGather {
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
        )) = typed::get_args(
            kind,
            |out_shape: &EList,
             index_strides: &EList,
             data_shape: &EList,
             data_strides: &EList,
             out_strides: &EList| {
                crate::typed_rules::metal_gather(
                    out_shape,
                    index_strides,
                    data_shape,
                    data_strides,
                    out_strides,
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
        Ok(Some(LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
            out_shape: decoded_out_shape,
            index_stride: decoded_index_strides,
            data_shape: decoded_data_shape,
            data_stride: decoded_data_strides,
            out_stride: decoded_out_strides,
        }))))
    }
}

impl MetalKernelOp for MetalGather {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        let data_dtype = input_dtypes.get(1).copied().unwrap_or(DType::F32);
        let data_ty = metal_buffer_type(data_dtype);
        let out_ty = metal_buffer_type(output_dtype);
        let out_idx =
            lower_expression_for_metal(&flatten_strides(&self.out_shape, &self.out_stride), "idx");
        let index_idx = lower_expression_for_metal(
            &flatten_strides(&self.out_shape, &self.index_stride),
            "idx",
        );
        let data_idx = lower_expression_for_metal(
            &flatten_strides(&self.data_shape, &self.data_stride),
            "gathered_index",
        );
        let gathered_val = metal_copy_value(data_dtype, "data", &data_idx);

        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                const device int *indexes [[buffer(0)]],
                const device {data_ty} *data [[buffer(1)]],
                device {out_ty} *out [[buffer(2)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                constant uint &n_elements [[buffer({n_elements_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    int gathered_index = indexes[{index_idx}];
                    out[{out_idx}] = {gathered_val};
                }}
            }}
            "#,
            data_ty = data_ty,
            out_ty = out_ty,
            gathered_val = gathered_val,
            dyn_buffer_index = 3u64,
            n_elements_index = 4u64,
        );
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn output_size(&self) -> Expression {
        self.out_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn infer_output_dtype(&self, input_dtypes: &[DType]) -> DType {
        input_dtypes.get(1).copied().unwrap_or(DType::F32)
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self.output_size().exec(dyn_map).unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0); // indexes
        encoder.set_buffer(1, Some(inputs[1]), 0); // data
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }

    // Gather metrics: read indices + read gathered data + write output
    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self.output_size().exec(dyn_map).unwrap_or(0);
        // Read n indices (i32 = 4 bytes) + n data elements (f32 = 4 bytes)
        n * std::mem::size_of::<i32>() + n * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self.output_size().exec(dyn_map).unwrap_or(0);
        // Write n output elements (f32)
        n * std::mem::size_of::<f32>()
    }

    fn flops(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        // Gather is memory-bound, no significant FLOPs
        0
    }
}

// MetalScatter: inverse of gather - out = copy(dest); out[indexes[i]] = src[i]
// Uses two sequential dispatches: copy then scatter. Metal guarantees order within an encoder.
#[derive(Debug, Clone)]
pub struct MetalScatter {
    dest_shape: Vec<Expression>,
    dest_strides: Vec<Expression>,
    index_shape: Vec<Expression>,
    index_strides: Vec<Expression>,
    src_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
    copy_pipeline: std::sync::OnceLock<ComputePipelineState>,
}

impl Default for MetalScatter {
    fn default() -> Self {
        Self {
            dest_shape: Vec::new(),
            dest_strides: Vec::new(),
            index_shape: Vec::new(),
            index_strides: Vec::new(),
            src_strides: Vec::new(),
            out_strides: Vec::new(),
            copy_pipeline: std::sync::OnceLock::new(),
        }
    }
}

pub(crate) mod metal_scatter {
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
            metal_scatter(
                dest_shape,
                dest_strides,
                index_shape,
                index_strides,
                src_strides,
                EList::RowMajor(dest_shape),
            ),
            [dest, indexes, src],
        );
        let replacement = &Ir::Op(
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
            (replacement, eq(dt, src.dtype())),
            (
                union(replacement, source),
                subsume(replacement),
                set(source.dtype(), dt),
            ),
        )
    }
}

impl MetalScatter {
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
        )) = typed::get_args(
            kind,
            |dest_shape: &EList,
             dest_strides: &EList,
             index_shape: &EList,
             index_strides: &EList,
             src_strides: &EList,
             out_strides: &EList| {
                crate::typed_rules::metal_scatter(
                    dest_shape,
                    dest_strides,
                    index_shape,
                    index_strides,
                    src_strides,
                    out_strides,
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
        Ok(Some(LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
            dest_shape: decoded_dest_shape,
            dest_strides: decoded_dest_strides,
            index_shape: decoded_index_shape,
            index_strides: decoded_index_strides,
            src_strides: decoded_src_strides,
            out_strides: decoded_out_strides,
            copy_pipeline: std::sync::OnceLock::new(),
        }))))
    }
}

impl MetalKernelOp for MetalScatter {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        let dest_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let src_dtype = input_dtypes.get(2).copied().unwrap_or(output_dtype);
        let dest_ty = metal_buffer_type(dest_dtype);
        let src_ty = metal_buffer_type(src_dtype);
        let out_ty = metal_buffer_type(output_dtype);
        // Compile the copy kernel and store it
        let dest_idx = lower_expression_for_metal(
            &flatten_strides(&self.dest_shape, &self.dest_strides),
            "idx",
        );
        let out_copy_idx = lower_expression_for_metal(
            &flatten_strides(&self.dest_shape, &self.out_strides),
            "idx",
        );
        let copy_source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void copy_kernel(
                device {out_ty} *out [[buffer(0)]],
                const device {dest_ty} *dest [[buffer(1)]],
                constant uint &n_elements [[buffer(2)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[{out_copy_idx}] = dest[{dest_idx}];
                }}
            }}
            "#,
            out_ty = out_ty,
            dest_ty = dest_ty,
            dyn_buffer_index = 4u64
        );
        let _ = self
            .copy_pipeline
            .set(compile_shader(device, &copy_source, "copy_kernel"));

        // Compile the scatter kernel (returned as the main pipeline)
        let index_idx = lower_expression_for_metal(
            &flatten_strides(&self.index_shape, &self.index_strides),
            "idx",
        );
        let src_idx = lower_expression_for_metal(
            &flatten_strides(&self.index_shape, &self.src_strides),
            "idx",
        );
        let scatter_source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void scatter_kernel(
                device {out_ty} *out [[buffer(0)]],
                const device int *indexes [[buffer(1)]],
                const device {src_ty} *src [[buffer(2)]],
                constant uint &n_elements [[buffer(3)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    int scatter_idx = indexes[{index_idx}];
                    out[scatter_idx] = src[{src_idx}];
                }}
            }}
            "#,
            out_ty = out_ty,
            src_ty = src_ty,
            dyn_buffer_index = 4u64
        );
        Some(compile_shader(device, &scatter_source, "scatter_kernel"))
    }

    fn output_size(&self) -> Expression {
        self.dest_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_dest = self
            .dest_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .exec(dyn_map)
            .unwrap() as u32;
        let n_src = self
            .index_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .exec(dyn_map)
            .unwrap() as u32;

        // Dispatch 1: copy dest → output
        let copy_pipeline = self
            .copy_pipeline
            .get()
            .expect("copy pipeline not compiled");
        encoder.set_compute_pipeline_state(copy_pipeline);
        encoder.set_buffer(0, Some(output), 0);
        encoder.set_buffer(1, Some(inputs[0]), 0); // dest
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            &n_dest as *const u32 as *const _,
        );
        let thread_group_size = MTLSize::new(256, 1, 1);
        encoder.dispatch_thread_groups(
            MTLSize::new((n_dest as u64).div_ceil(256), 1, 1),
            thread_group_size,
        );

        // Dispatch 2: scatter src → output[indexes[i]]
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(output), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0); // indexes
        encoder.set_buffer(2, Some(inputs[2]), 0); // src
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &n_src as *const u32 as *const _,
        );
        encoder.dispatch_thread_groups(
            MTLSize::new((n_src as u64).div_ceil(256), 1, 1),
            thread_group_size,
        );
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n_dest = self.output_size().exec(dyn_map).unwrap_or(0);
        let n_src = self
            .index_shape
            .iter()
            .cloned()
            .product::<Expression>()
            .exec(dyn_map)
            .unwrap_or(0);
        n_dest * std::mem::size_of::<f32>()
            + n_src * std::mem::size_of::<i32>()
            + n_src * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self.output_size().exec(dyn_map).unwrap_or(0);
        n * std::mem::size_of::<f32>()
    }

    fn flops(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        0
    }
}

#[derive(Debug, Default, Clone)]
#[allow(dead_code)]
pub struct MetalScatterNoCopy {
    dest_shape: Vec<Expression>,
    dest_strides: Vec<Expression>,
    index_shape: Vec<Expression>,
    index_strides: Vec<Expression>,
    src_strides: Vec<Expression>,
    out_strides: Vec<Expression>,
}

pub(crate) mod metal_scatter_no_copy {
    #[relation]
    pub(crate) fn consumed_buffer_ilist_contains(a0: IList, a1: Ir);
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

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
            .label("metal-consumed-buffer-ilist-contains-head"),
            rule(
                (
                    eq(list, IList::ICons(head, tail)),
                    Fact::from(consumed_buffer_ilist_contains(tail, item)),
                ),
                (consumed_buffer_ilist_contains(list, item),),
            )
            .label("metal-consumed-buffer-ilist-contains-tail"),
            rule(
                (
                    eq(cb, consumed_buffer(a)),
                    Fact::from(consumed_buffer_ilist_contains(ilist1, cb)),
                    ne(op1, op2),
                    Fact::from(consumed_buffer_ilist_contains(ilist2, a)),
                ),
                (delete(consumed_buffer(a)),),
            )
            .label("metal-consumed-buffer-cleanup-shared-op-use"),
        ]
    }

    #[ruleset]
    pub(crate) fn buffer_reuse(
        ds: &EList,
        dst: &EList,
        dest: &Ir,
        indexes: &Ir,
        is: &EList,
        istr: &EList,
        src: &Ir,
        ss: &EList,
        os: &EList,
        scatter: &Ir,
        dt: &DType,
    ) -> Rule {
        let consumed = &consumed_buffer(dest);
        let nocopy = &Ir::Op(
            metal_scatter_no_copy(ds, dst, is, istr, ss, os),
            [consumed, indexes, src],
        );
        rule(
            (
                eq(
                    scatter,
                    Ir::Op(
                        metal_scatter(ds, dst, is, istr, ss, os),
                        [dest, indexes, src],
                    ),
                ),
                eq(dst, os),
                eq(dt, src.dtype()),
            ),
            (
                consumed,
                nocopy,
                union(scatter, nocopy),
                set(nocopy.dtype(), dt),
            ),
        )
        .label("metal-scatter-to-scatter-no-copy")
    }

    #[ruleset]
    pub(crate) fn dtype_prop(a: &Ir, cb: &Ir, dt: &DType) -> Rule {
        rule(
            (eq(cb, consumed_buffer(a)), eq(dt, a.dtype())),
            (set(cb.dtype(), dt),),
        )
        .label("metal-consumed-buffer-dtype")
    }

    #[ruleset]
    pub(crate) fn post_cleanup(
        dest: &Ir,
        cb: &Ir,
        ds: &EList,
        dst: &EList,
        indexes: &Ir,
        is: &EList,
        istr: &EList,
        src: &Ir,
        ss: &EList,
        os: &EList,
        scatter: &Ir,
    ) -> Rule {
        let source = &Ir::Op(
            metal_scatter(ds, dst, is, istr, ss, os),
            [dest, indexes, src],
        );
        rule(
            (
                eq(cb, consumed_buffer(dest)),
                eq(scatter, source),
                Ir::Op(
                    metal_scatter_no_copy(ds, dst, is, istr, ss, os),
                    [cb, indexes, src],
                ),
            ),
            (subsume(source),),
        )
        .label("metal-scatter-no-copy-dominates-valid-consumed-buffer")
    }

    #[ruleset]
    pub(crate) fn base_cleanup(a: &Ir, cb: &Ir) -> Rule {
        rule(
            (eq(cb, consumed_buffer(a)),),
            (union(cb, a), delete(consumed_buffer(a))),
        )
        .label("metal-consumed-buffer-resolve")
    }
}

impl MetalScatterNoCopy {
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
        )) = typed::get_args(
            kind,
            |dest_shape: &EList,
             dest_strides: &EList,
             index_shape: &EList,
             index_strides: &EList,
             src_strides: &EList,
             out_strides: &EList| {
                crate::typed_rules::metal_scatter_no_copy(
                    dest_shape,
                    dest_strides,
                    index_shape,
                    index_strides,
                    src_strides,
                    out_strides,
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
        Ok(Some(LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
            dest_shape: decoded_dest_shape,
            dest_strides: decoded_dest_strides,
            index_shape: decoded_index_shape,
            index_strides: decoded_index_strides,
            src_strides: decoded_src_strides,
            out_strides: decoded_out_strides,
        }))))
    }
}

impl MetalKernelOp for MetalScatterNoCopy {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        let src_dtype = input_dtypes.get(2).copied().unwrap_or(output_dtype);
        let src_ty = metal_buffer_type(src_dtype);
        let out_ty = metal_buffer_type(output_dtype);
        let index_idx = lower_expression_for_metal(
            &flatten_strides(&self.index_shape, &self.index_strides),
            "idx",
        );
        let src_idx = lower_expression_for_metal(
            &flatten_strides(&self.index_shape, &self.src_strides),
            "idx",
        );
        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;
            kernel void scatter_no_copy_kernel(
                device {out_ty} *out [[buffer(0)]],
                const device int *indexes [[buffer(1)]],
                const device {src_ty} *src [[buffer(2)]],
                constant uint &n_elements [[buffer(3)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    int scatter_idx = indexes[{index_idx}];
                    out[scatter_idx] = src[{src_idx}];
                }}
            }}
            "#,
            out_ty = out_ty,
            src_ty = src_ty,
            dyn_buffer_index = 4u64
        );
        Some(compile_shader(device, &source, "scatter_no_copy_kernel"))
    }

    fn infer_output_dtype(&self, input_dtypes: &[DType]) -> DType {
        input_dtypes.first().copied().unwrap_or(DType::F32)
    }

    fn output_size(&self) -> Expression {
        self.dest_shape
            .iter()
            .copied()
            .product::<Expression>()
            .max(Expression::from(1))
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_src = self
            .index_shape
            .iter()
            .copied()
            .product::<Expression>()
            .exec(dyn_map)
            .unwrap() as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(output), 0);
        encoder.set_buffer(1, Some(inputs[1]), 0);
        encoder.set_buffer(2, Some(inputs[2]), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &n_src as *const u32 as *const _,
        );
        let thread_group_size = MTLSize::new(256, 1, 1);
        encoder.dispatch_thread_groups(
            MTLSize::new((n_src as u64).div_ceil(256), 1, 1),
            thread_group_size,
        );
    }

    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n_src = self
            .index_shape
            .iter()
            .copied()
            .product::<Expression>()
            .exec(dyn_map)
            .unwrap_or(0);
        n_src * std::mem::size_of::<i32>() + n_src * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n_src = self
            .index_shape
            .iter()
            .copied()
            .product::<Expression>()
            .exec(dyn_map)
            .unwrap_or(0);
        n_src * std::mem::size_of::<f32>()
    }

    fn flops(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        0
    }

    fn output_aliases_input(&self) -> Option<usize> {
        Some(0)
    }
}

// ============================================================================
// Type Conversion Operations
// ============================================================================

// MetalCast: convert between data types (Int <-> F32, F16, etc.)
// This is a pure element-wise operation with no data movement or reshaping.
#[derive(Debug, Default, Clone)]
pub struct MetalCast {
    size: Expression,
    target_dtype: DType,
}

pub(crate) mod metal_cast {
    use crate::typed_rules::*;
    use egglog_experimental::typed::prelude::*;
    use luminal::typed_reference::schema::*;

    #[ruleset]
    pub(crate) fn kernel_lower(size: &DimExpr, dtype: &DType, inp: &Ir) -> Rule {
        let source = &Ir::Op(OpKind::Cast(size, dtype), [inp]);
        let replacement = &Ir::Op(metal_cast(size, dtype), [inp]);
        rule(
            source,
            (
                union(source, replacement),
                subsume(source),
                set(replacement.dtype(), dtype),
            ),
        )
    }
}

impl MetalCast {
    pub fn decode(
        kind: &OpKind,
        _inputs: &mut Vec<Ir>,
        context: &mut DecodeContext<'_, '_>,
    ) -> Result<Option<LLIROp>, TypedError> {
        let Some((field_size, field_dtype)) = typed::get_args(
            kind,
            |size: &DimExpr, dtype: &luminal::typed_reference::schema::DType| {
                crate::typed_rules::metal_cast(size, dtype)
            },
        )?
        else {
            return Ok(None);
        };
        Ok(Some(LLIROp::new::<dyn MetalKernelOp>(Box::new(Self {
            size: context.expression(&field_size)?,
            target_dtype: context.dtype(&field_dtype)?,
        }))))
    }
}

impl MetalKernelOp for MetalCast {
    fn compile(
        &self,
        device: &Device,
        input_dtypes: &[DType],
        output_dtype: DType,
    ) -> Option<ComputePipelineState> {
        let input_dtype = input_dtypes.first().copied().unwrap_or(DType::F32);
        let input_ty = metal_buffer_type(input_dtype);
        let output_ty = metal_buffer_type(output_dtype);
        let cast_expr = match (input_dtype, output_dtype) {
            (DType::F32, DType::F32)
            | (DType::F16, DType::F16)
            | (DType::Int, DType::Int)
            | (DType::F32, DType::F16)
            | (DType::F16, DType::F32)
            | (DType::F32, DType::Int)
            | (DType::Int, DType::F32)
            | (DType::F16, DType::Int)
            | (DType::Int, DType::F16) => format!("({output_ty})(inp[idx])"),
            _ => panic!(
                "MetalCast does not support runtime cast from {input_dtype:?} to {output_dtype:?}"
            ),
        };
        let source = format!(
            r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void mkernel(
                device {input_ty} *inp [[buffer(0)]],
                device {output_ty} *out [[buffer(1)]],
                constant int *dyn [[buffer({dyn_buffer_index})]],
                constant uint &n_elements [[buffer({n_elements_index})]],
                uint idx [[thread_position_in_grid]]
            ) {{
                if (idx < n_elements) {{
                    out[idx] = {cast_expr};
                }}
            }}
            "#,
            input_ty = input_ty,
            output_ty = output_ty,
            cast_expr = cast_expr,
            dyn_buffer_index = 2u64,
            n_elements_index = 3u64,
        );
        Some(compile_shader(device, &source, "mkernel"))
    }

    fn output_size(&self) -> Expression {
        self.size
    }

    fn infer_output_dtype(&self, _input_dtypes: &[DType]) -> DType {
        self.target_dtype
    }

    fn encode_compute(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        inputs: &[&Buffer],
        output: &Buffer,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        let n_elements = self.size.exec(dyn_map).unwrap_or(0) as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(inputs[0]), 0);
        encoder.set_buffer(1, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &n_elements as *const u32 as *const _,
        );

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n_elements as u64).div_ceil(256), 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
    }

    // Cast is memory-bound: 1 read, 1 write, minimal compute
    fn bytes_loaded(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self.size.exec(dyn_map).unwrap_or(0);
        // TODO: input dtype is not encoded; treat as 4B for now (matches current MetalRuntime IO path).
        n * std::mem::size_of::<f32>()
    }

    fn bytes_stored(&self, dyn_map: &FxHashMap<char, usize>) -> usize {
        let n = self.size.exec(dyn_map).unwrap_or(0);
        // TODO: output dtype sizing should be wired through MetalRuntime allocation.
        n * std::mem::size_of::<f32>()
    }

    fn flops(&self, _dyn_map: &FxHashMap<char, usize>) -> usize {
        0 // Type conversion has negligible compute cost
    }
}
