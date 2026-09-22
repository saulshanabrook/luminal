use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use crate::egglog_utils::{DecodeContext, PhaseRules};
use crate::prelude::*;
use crate::typed_reference::schema::{Ir, OpKind};
use as_any::{AsAny, Downcast};
use egglog_experimental::typed::{Action, TypedError};
use rustc_hash::FxHashMap;

#[derive(Clone, Copy)]
pub struct ProfileBucketContext<'a> {
    pub dim_buckets: &'a FxHashMap<char, Vec<DimBucket>>,
    pub bucket_indices: &'a FxHashMap<char, usize>,
    pub representative_dyn_map: &'a FxHashMap<char, usize>,
}

#[derive(Clone, Copy)]
pub struct CandidateFilterContext<'a> {
    pub search_options: &'a crate::graph::CompileOptions,
    pub dyn_map: &'a FxHashMap<char, usize>,
    pub bucket_context: Option<ProfileBucketContext<'a>>,
}

#[derive(Debug, Clone, Default)]
pub struct CandidateFilterResult {
    pub accepted: bool,
    pub display: Option<String>,
}

impl CandidateFilterResult {
    pub fn accept() -> Self {
        Self {
            accepted: true,
            display: None,
        }
    }

    pub fn accept_with_display(display: impl Into<String>) -> Self {
        Self {
            accepted: true,
            display: Some(display.into()),
        }
    }

    pub fn reject() -> Self {
        Self {
            accepted: false,
            display: None,
        }
    }

    pub fn reject_with_display(display: impl Into<String>) -> Self {
        Self {
            accepted: false,
            display: Some(display.into()),
        }
    }
}

pub trait Runtime {
    /// Match and prepare selected backend expressions, before core fallbacks.
    const DECODERS: &'static [DecodeFn] = &[];

    /// Normalized kinds that were direct IR nodes in the previous schema.
    /// They must not become new survivor evidence for conditional HLIR cleanup.
    fn cleanup_ignored(_kind: &OpKind) -> Result<bool, TypedError> {
        Ok(false)
    }

    /// Complete default recipe: phase rules and ordered initialization actions.
    /// Customize these ordinary values before building a search space.
    fn rules() -> (PhaseRules, Vec<Action>) {
        (crate::egglog_utils::base_phase_rules(), Vec::new())
    }
    type CompileArg;
    type ExecReturn;
    type ProfileMetric: PartialOrd + Clone + Debug;
    fn initialize(arg: Self::CompileArg) -> Self;
    fn load_llir(&mut self, llir_graph: &LLIRGraph);
    fn execute(&mut self, dyn_map: &FxHashMap<char, usize>) -> Self::ExecReturn;
    fn profile(
        &mut self,
        llir_graph: &LLIRGraph,
        dyn_map: &FxHashMap<char, usize>,
        trials: usize,
        timeout: Option<std::time::Duration>,
    ) -> (Self::ProfileMetric, String);
    /// Profile one candidate in the context of a specific dynamic-dimension
    /// bucket. Runtimes with bucket-sensitive lowering can override this so
    /// search ranks candidates under the same execution model used after
    /// final bucket compilation.
    fn profile_with_bucket_context(
        &mut self,
        llir_graph: &LLIRGraph,
        dyn_map: &FxHashMap<char, usize>,
        trials: usize,
        timeout: Option<std::time::Duration>,
        _bucket_context: ProfileBucketContext<'_>,
    ) -> (Self::ProfileMetric, String) {
        self.profile(llir_graph, dyn_map, trials, timeout)
    }
    /// Aggregate multiple profile metrics into one comparable metric.
    /// Used for regionalized profiling where one candidate maps to multiple LLIR regions.
    fn aggregate_profile_metrics(metrics: &[Self::ProfileMetric]) -> Self::ProfileMetric {
        metrics
            .first()
            .unwrap_or_else(|| panic!("aggregate_profile_metrics called with empty metrics"))
            .clone()
    }
    /// Allocate a dummy input buffer for a boundary node during per-chunk profiling.
    /// `node_index` is the HLIR node index used in the Input op's `node` field.
    /// `num_bytes` is the number of bytes to allocate.
    fn allocate_dummy_input(&mut self, _node_index: usize, _num_bytes: usize) {}
    /// Check if an HLIR buffer already exists for a given node index.
    fn has_hlir_buffer(&self, _node_index: usize) -> bool {
        false
    }
    /// Clear intermediate buffers to prepare for loading a different chunk's LLIR.
    fn clear_intermediate_buffers(&mut self) {}
    /// Total bytes of intermediate buffers currently allocated.
    fn intermediate_buffer_bytes(&self) -> usize {
        0
    }
    /// Check if the most recent execution produced NaN in any output buffer.
    /// Used by the search to reject NaN-producing graph variants.
    fn has_nan_outputs(&self, _llir_graph: &LLIRGraph, _dyn_map: &FxHashMap<char, usize>) -> bool {
        false
    }
    /// Runtime-specific pre-profile candidate filter. Backends can reject an
    /// extracted LLIR graph before profiling it, for example because it exceeds
    /// a backend-specific resource budget. Core treats this as an opaque
    /// accept/reject decision and optional display text.
    fn filter_llir_candidate(
        &mut self,
        _llir_graph: &LLIRGraph,
        _context: CandidateFilterContext<'_>,
    ) -> CandidateFilterResult {
        CandidateFilterResult::accept()
    }
    /// Load multiple compiled LLIR graphs, one per bucket combination.
    /// Each entry is (bucket_indices, representative_dyn_map, stitched_llir).
    /// The runtime dispatches between them in execute() based on dyn_map values.
    fn load_llir_buckets(
        &mut self,
        _dim_buckets: &FxHashMap<char, Vec<DimBucket>>,
        bucket_llirs: &[BucketLLIR],
    ) {
        if bucket_llirs.len() == 1 {
            self.load_llir(&bucket_llirs[0].2);
        } else {
            panic!("This runtime does not support bucketed compilation");
        }
    }
}

/// Optional runtime instrumentation for collecting execution statistics.
pub trait RuntimeStats: Runtime {
    fn execute_with_stats(&mut self, dyn_map: &FxHashMap<char, usize>) -> Option<ExecutionStats>;
}

/// Timing method used for execution statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimingMethod {
    /// Device-side timing (e.g. GPU timestamps / CUDA events).
    DeviceTimestamp,
    /// Host-side wall-clock timing.
    /// Includes any host/device synchronization overhead.
    #[default]
    WallClock,
}

impl std::fmt::Display for TimingMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimingMethod::DeviceTimestamp => write!(f, "Device"),
            TimingMethod::WallClock => write!(f, "Wall"),
        }
    }
}

/// Detailed execution statistics from a single run.
///
/// This struct captures basic counters and timing.
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Execution time in microseconds.
    pub execution_time_us: f64,
    /// Total bytes read.
    pub bytes_loaded: usize,
    /// Total bytes written.
    pub bytes_stored: usize,
    /// Total floating-point operations.
    pub flops: usize,
    /// Timing method used for this measurement.
    pub timing_method: TimingMethod,
}

impl ExecutionStats {
    pub fn new(
        execution_time_us: f64,
        bytes_loaded: usize,
        bytes_stored: usize,
        flops: usize,
    ) -> Self {
        Self {
            execution_time_us,
            bytes_loaded,
            bytes_stored,
            flops,
            timing_method: TimingMethod::DeviceTimestamp,
        }
    }

    /// Create new execution stats with explicit timing method.
    pub fn with_timing_method(
        execution_time_us: f64,
        bytes_loaded: usize,
        bytes_stored: usize,
        flops: usize,
        timing_method: TimingMethod,
    ) -> Self {
        Self {
            execution_time_us,
            bytes_loaded,
            bytes_stored,
            flops,
            timing_method,
        }
    }

    /// Total bytes transferred (loaded + stored).
    pub fn total_bytes(&self) -> usize {
        self.bytes_loaded + self.bytes_stored
    }

    pub fn merge(&mut self, other: &ExecutionStats) {
        self.execution_time_us += other.execution_time_us;
        self.bytes_loaded += other.bytes_loaded;
        self.bytes_stored += other.bytes_stored;
        self.flops += other.flops;
    }
}

impl std::fmt::Display for ExecutionStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ExecutionStats {{ time: {:.2}µs ({}), bytes: {:.2}MB, flops: {:.2}M }}",
            self.execution_time_us,
            self.timing_method,
            self.total_bytes() as f64 / 1_000_000.0,
            self.flops as f64 / 1_000_000.0
        )
    }
}

/// Prepare one selected constructor and its ordered, selected input producers.
/// A decoder may replace dependencies, but cannot add reachable producers.
/// Match with `get_args` before decoding, mutation, or preparation: `Ok(None)`
/// must leave inputs and context unchanged. Dispatch stops at the first `Some`
/// or error; an error is never a request to try another decoder.
pub type DecodeFn = for<'a, 'b> fn(
    &OpKind,
    &mut Vec<Ir>,
    &mut DecodeContext<'a, 'b>,
) -> Result<Option<LLIROp>, TypedError>;

pub trait CustomOp: Debug {
    fn to_llir_op(&self) -> LLIROp;
}

/// The main HLIROp trait.
///
/// Defines an HLIROp that implements a logical operation.
pub trait HLIROp: Debug + Display + as_any::AsAny {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir;
}

impl<T: HLIROp> HLIROp for Box<T> {
    fn to_ir(&self, inputs: &[&Ir]) -> Ir {
        <T as HLIROp>::to_ir(self, inputs)
    }
}

#[derive(Debug, Clone)]
pub struct LLIROp(Arc<Box<dyn DialectOpTrait>>);

impl LLIROp {
    /// Store an op in a generic LLIR op. **Make sure to erase type into your dialect trait!** i.e. `as Box<dyn BlockOp>`
    pub fn new<T: ?Sized>(op: Box<T>) -> Self
    where
        Box<T>: Debug + 'static,
    {
        assert!(
            op.type_name().contains("dyn")
                || op.type_name().contains("Input")
                || op.type_name().contains("Output")
                || op.type_name().contains("LoopStart")
                || op.type_name().contains("LoopEnd")
                || op.type_name().contains("LoopInput")
                || op.type_name().contains("LoopOutput"),
            "op types must be erased into dialect traits for dialect casting to work!"
        );
        Self(Arc::new(Box::new(DialectOp(Arc::new(op)))))
    }

    pub fn to_dialect<T: ?Sized + 'static>(&self) -> Option<&Arc<Box<T>>> {
        (**self.0).downcast_ref::<DialectOp<Box<T>>>().map(|i| &i.0)
    }

    pub fn to_op<T: 'static>(&self) -> Option<&T> {
        (**self.0)
            .downcast_ref::<DialectOp<Box<T>>>()
            .map(|d| &**d.0)
    }
}

impl std::fmt::Display for LLIROp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Debug)]
struct DialectOp<T>(pub Arc<T>);

impl<T: Debug + 'static> DialectOpTrait for DialectOp<T> {}

pub trait DialectOpTrait: AsAny + Debug {}
