use egglog_experimental::typed::{builtins as egg, prelude::*};
use luminal::typed_reference::schema::*;

#[constructor(name = "KernelMod")]
pub fn kernel_mod(
    shape: EList,
    a_strides: EList,
    b_strides: EList,
    out_strides: EList,
    dtype: DType,
) -> OpKind;

#[constructor(name = "KernelLessThan")]
pub fn kernel_less_than(
    shape: EList,
    a_strides: EList,
    b_strides: EList,
    out_strides: EList,
    dtype: DType,
) -> OpKind;

#[constructor(name = "KernelIota")]
pub fn kernel_iota(expr: DimExpr, range: DimExpr) -> OpKind;

#[constructor(name = "KernelGather")]
pub fn kernel_gather(
    out_shape: EList,
    index_strides: EList,
    data_shape: EList,
    data_strides: EList,
    out_strides: EList,
    dtype: DType,
) -> OpKind;

#[allow(clippy::too_many_arguments)] // Positional fields of the native kernel constructor.
#[constructor(name = "KernelScatter")]
pub fn kernel_scatter(
    dest_shape: EList,
    dest_strides: EList,
    index_shape: EList,
    index_strides: EList,
    src_strides: EList,
    out_strides: EList,
    dtype: DType,
) -> OpKind;

#[constructor(name = "KernelSum")]
pub fn kernel_sum(
    shape: EList,
    iters: DimExpr,
    strides: EList,
    iter_stride: DimExpr,
    out_strides: EList,
    dtype: DType,
) -> OpKind;

#[constructor(name = "KernelCastSum")]
pub fn kernel_cast_sum(
    shape: EList,
    iters: DimExpr,
    strides: EList,
    iter_stride: DimExpr,
    out_strides: EList,
    dtype: DType,
) -> OpKind;

#[constructor(name = "KernelMax")]
pub fn kernel_max(
    shape: EList,
    iters: DimExpr,
    strides: EList,
    iter_stride: DimExpr,
    out_strides: EList,
    dtype: DType,
) -> OpKind;

#[constructor(name = "KernelConstant")]
pub fn kernel_constant(value: egg::F64, dtype: DType) -> OpKind;

#[constructor(name = "KernelCast")]
pub fn kernel_cast(size: DimExpr, dtype: DType, src_dtype: DType) -> OpKind;

#[constructor(name = "KernelEmbed")]
pub fn kernel_embed(
    batch_shape: EList,
    token_stride: EList,
    out_stride: EList,
    embed_dim: DimExpr,
    dtype: DType,
) -> OpKind;

#[constructor(name = "KernelArgmax")]
pub fn kernel_argmax(shape: EList, cols: DimExpr, dtype: DType) -> OpKind;

#[constructor(name = "KernelGemv")]
pub fn kernel_gemv(n: DimExpr, k: DimExpr, dtype: DType) -> OpKind;

#[constructor(name = "KernelGemvF8", args = KernelGemvF8Args)]
pub fn kernel_gemv_f8(n: DimExpr, k: DimExpr) -> OpKind;

#[constructor(name = "KernelMoEGemv")]
pub fn kernel_mo_e_gemv(
    out_shape: EList,
    k_dim: DimExpr,
    w_shape: EList,
    x_strides: EList,
    topk_strides: EList,
) -> OpKind;

#[constructor(name = "KernelRMSNorm")]
pub fn kernel_rms_norm(out_shape: EList, eps: egg::F64) -> OpKind;

#[constructor(name = "KernelRMSNormQuant")]
pub fn kernel_rms_norm_quant(out_shape: EList, eps: egg::F64) -> OpKind;

#[constructor(name = "KernelRoPE", args = KernelRoPEArgs)]
pub fn kernel_ro_pe(
    out_shape: EList,
    width: DimExpr,
    ln_theta: egg::F64,
    inv_hd: egg::F64,
) -> OpKind;

#[constructor(name = "KernelSwiglu")]
pub fn kernel_swiglu(out_shape: EList) -> OpKind;

#[constructor(name = "KernelSwigluQuant")]
pub fn kernel_swiglu_quant(out_shape: EList) -> OpKind;

#[constructor(name = "KernelStableSortIdx")]
pub fn kernel_stable_sort_idx(out_shape: EList) -> OpKind;

#[constructor(name = "KernelQuantF8")]
pub fn kernel_quant_f8(size: DimExpr) -> OpKind;

#[constructor(name = "KernelMean")]
pub fn kernel_mean(
    shape: EList,
    iters: DimExpr,
    strides: EList,
    iter_stride: DimExpr,
    out_strides: EList,
    dtype: DType,
) -> OpKind;

#[allow(clippy::too_many_arguments)] // Positional fields of the native kernel constructor.
#[constructor(name = "KernelScatterNoCopy")]
pub fn kernel_scatter_no_copy(
    dest_shape: EList,
    dest_strides: EList,
    index_shape: EList,
    index_strides: EList,
    src_strides: EList,
    out_strides: EList,
    dtype: DType,
) -> OpKind;

#[allow(clippy::too_many_arguments)] // Positional fields of the native kernel constructor.
#[constructor(name = "KernelConv2D", args = KernelConv2DArgs)]
pub fn kernel_conv2_d(
    out_shape: EList,
    input_shape: EList,
    input_stride: EList,
    weight_co_stride: DimExpr,
    weight_inner_stride: DimExpr,
    bias_c_stride: DimExpr,
    out_stride: EList,
    kernel_h: DimExpr,
    kernel_w: DimExpr,
    stride_h: DimExpr,
    stride_w: DimExpr,
    dilation_h: DimExpr,
    dilation_w: DimExpr,
    pad_h: DimExpr,
    pad_w: DimExpr,
    dtype: DType,
) -> OpKind;

#[allow(clippy::too_many_arguments)] // Positional fields of the native kernel constructor.
#[constructor(name = "GenericMatmul", args = GenericMatmulArgs)]
pub fn generic_matmul(
    out_shape: EList,
    mul_shape: EList,
    k: DimExpr,
    lhs_strides: EList,
    rhs_strides: EList,
    sum_input_strides: EList,
    sum_iter_stride: DimExpr,
    out_strides: EList,
    dtype: DType,
) -> OpKind;

#[constructor(name = "FusionStart", args = FusionStartArgs)]
pub fn fusion_start(shape: EList, strides: EList, dtype: DType) -> OpKind;

#[constructor(name = "FusionEnd", args = FusionEndArgs)]
pub fn fusion_end(shape: EList, strides: EList, dtype: DType) -> OpKind;

#[constructor(name = "CudaUnaryElementwise")]
pub fn cuda_unary_elementwise(
    op: egg::String,
    shape: EList,
    strides: EList,
    out_strides: EList,
    dtype: DType,
) -> OpKind;

#[constructor(name = "CudaBinaryElementwise", args = CudaBinaryElementwiseArgs)]
pub fn cuda_binary_elementwise(
    op: egg::String,
    shape: EList,
    a_strides: EList,
    b_strides: EList,
    out_strides: EList,
    dtype: DType,
) -> OpKind;

#[allow(clippy::too_many_arguments)] // Positional fields of the native kernel constructor.
#[constructor(name = "cublaslt", args = CuBlasLtArgs)]
pub fn cublaslt(
    m: DimExpr,
    n: DimExpr,
    k: DimExpr,
    a_layout: egg::String,
    b_layout: egg::String,
    a_order: egg::String,
    b_order: egg::String,
    c_order: egg::String,
    d_order: egg::String,
    lda: DimExpr,
    ldb: DimExpr,
    ldc: DimExpr,
    ldd: DimExpr,
    batch_count: DimExpr,
    stride_a: DimExpr,
    stride_b: DimExpr,
    stride_c: DimExpr,
    stride_d: DimExpr,
    a_dtype: DType,
    b_dtype: DType,
    c_dtype: DType,
    d_dtype: DType,
    compute_type: egg::String,
    scale_dtype: egg::String,
    alpha: egg::F64,
    beta: egg::F64,
    epilogue: egg::String,
) -> OpKind;

#[allow(clippy::too_many_arguments)] // Positional fields of the native kernel constructor.
#[constructor(name = "cublaslt_scaled", args = CuBlasLtScaledArgs)]
pub fn cublaslt_scaled(
    m: DimExpr,
    n: DimExpr,
    k: DimExpr,
    a_layout: egg::String,
    b_layout: egg::String,
    a_order: egg::String,
    b_order: egg::String,
    c_order: egg::String,
    d_order: egg::String,
    lda: DimExpr,
    ldb: DimExpr,
    ldc: DimExpr,
    ldd: DimExpr,
    batch_count: DimExpr,
    stride_a: DimExpr,
    stride_b: DimExpr,
    stride_c: DimExpr,
    stride_d: DimExpr,
    a_dtype: DType,
    b_dtype: DType,
    c_dtype: DType,
    d_dtype: DType,
    compute_type: egg::String,
    scale_dtype: egg::String,
    alpha: egg::F64,
    beta: egg::F64,
    epilogue: egg::String,
) -> OpKind;

#[allow(clippy::too_many_arguments)] // Positional fields of the native kernel constructor.
#[constructor(name = "GLUMoE", args = GLUMoEArgs)]
pub fn glu_mo_e(
    gu_io: DimExpr,
    dn_io: DimExpr,
    gu_matmul_k: DimExpr,
    dn_matmul_k: DimExpr,
    output_k: DimExpr,
    gu_within_range: DimExpr,
    dn_within_range: DimExpr,
    mode: DimExpr,
) -> OpKind;

#[allow(clippy::too_many_arguments)] // Positional fields of the native kernel constructor.
#[constructor(name = "FlashInferAttention", args = FlashInferAttentionArgs)]
pub fn flash_infer_attention(
    num_qo_heads: DimExpr,
    num_kv_heads: DimExpr,
    head_dim: DimExpr,
    page_size: DimExpr,
    batch_dim: DimExpr,
    dtype: DType,
    sm_scale: egg::F64,
    window_left: egg::F64,
) -> OpKind;

#[cfg(test)]
mod tests;
