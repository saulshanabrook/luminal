use egglog_experimental::typed::{builtins as egg, prelude::*};
use luminal::typed_reference::schema::*;

#[cfg(test)]
mod tests;

#[constructor(name = "MetalExp2", args = MetalExp2Args)]
pub fn metal_exp2(shape: EList, strides: EList, out_strides: EList) -> OpKind;

#[constructor(name = "MetalLog2", args = MetalLog2Args)]
pub fn metal_log2(shape: EList, strides: EList, out_strides: EList) -> OpKind;

#[constructor(name = "MetalSin", args = MetalSinArgs)]
pub fn metal_sin(shape: EList, strides: EList, out_strides: EList) -> OpKind;

#[constructor(name = "MetalSqrt", args = MetalSqrtArgs)]
pub fn metal_sqrt(shape: EList, strides: EList, out_strides: EList) -> OpKind;

#[constructor(name = "MetalRecip", args = MetalRecipArgs)]
pub fn metal_recip(shape: EList, strides: EList, out_strides: EList) -> OpKind;

#[constructor(name = "MetalAdd", args = MetalAddArgs)]
pub fn metal_add(shape: EList, a_strides: EList, b_strides: EList, out_strides: EList) -> OpKind;

#[constructor(name = "MetalMul", args = MetalMulArgs)]
pub fn metal_mul(shape: EList, a_strides: EList, b_strides: EList, out_strides: EList) -> OpKind;

#[constructor(name = "MetalMod", args = MetalModArgs)]
pub fn metal_mod(shape: EList, a_strides: EList, b_strides: EList, out_strides: EList) -> OpKind;

#[constructor(name = "MetalLessThan", args = MetalLessThanArgs)]
pub fn metal_less_than(
    shape: EList,
    a_strides: EList,
    b_strides: EList,
    out_strides: EList,
) -> OpKind;

#[constructor(name = "MetalSum", args = MetalSumArgs)]
pub fn metal_sum(
    shape: EList,
    iters: DimExpr,
    strides: EList,
    iter_stride: DimExpr,
    out_strides: EList,
) -> OpKind;

#[constructor(name = "MetalMax", args = MetalMaxArgs)]
pub fn metal_max(
    shape: EList,
    iters: DimExpr,
    strides: EList,
    iter_stride: DimExpr,
    out_strides: EList,
) -> OpKind;

#[allow(clippy::too_many_arguments)] // Positional fields of the native kernel constructor.
#[constructor(name = "MPSMatmul", args = MPSMatmulArgs)]
pub fn mps_matmul(
    m: DimExpr,
    n: DimExpr,
    k: DimExpr,
    lhs_row_stride: DimExpr,
    rhs_row_stride: DimExpr,
    out_row_stride: DimExpr,
    transpose_lhs: egg::I64,
    transpose_rhs: egg::I64,
) -> OpKind;

#[allow(clippy::too_many_arguments)] // Positional fields of the native kernel constructor.
#[constructor(name = "MPSBatchedMatmul", args = MPSBatchedMatmulArgs)]
pub fn mps_batched_matmul(
    batch: DimExpr,
    m: DimExpr,
    n: DimExpr,
    k: DimExpr,
    lhs_batch_stride: DimExpr,
    lhs_row_stride: DimExpr,
    rhs_batch_stride: DimExpr,
    rhs_row_stride: DimExpr,
    out_batch_stride: DimExpr,
    out_row_stride: DimExpr,
    transpose_lhs: egg::I64,
    transpose_rhs: egg::I64,
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
) -> OpKind;

#[constructor(name = "MetalConstant")]
pub fn metal_constant(value: egg::F64) -> OpKind;

#[constructor(name = "MetalIota")]
pub fn metal_iota(expr: DimExpr, range: DimExpr) -> OpKind;

#[constructor(name = "MetalGather")]
pub fn metal_gather(
    out_shape: EList,
    index_strides: EList,
    data_shape: EList,
    data_strides: EList,
    out_strides: EList,
) -> OpKind;

#[constructor(name = "MetalScatter")]
pub fn metal_scatter(
    dest_shape: EList,
    dest_strides: EList,
    index_shape: EList,
    index_strides: EList,
    src_strides: EList,
    out_strides: EList,
) -> OpKind;

#[constructor(name = "MetalScatterNoCopy", args = MetalScatterNoCopyArgs)]
pub fn metal_scatter_no_copy(
    dest_shape: EList,
    dest_strides: EList,
    index_shape: EList,
    index_strides: EList,
    src_strides: EList,
    out_strides: EList,
) -> OpKind;

#[constructor(name = "MetalCast")]
pub fn metal_cast(size: DimExpr, dtype: DType) -> OpKind;
