//! Typed rule execution and search directly over exact frozen observations.
use crate::{
    graph::LLIRGraph,
    op::{CustomOp, DecodeFn, LLIROp},
    shape::Expression,
};
use egglog_experimental::typed::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

mod driver;
pub(crate) mod matmul_flattening;
mod search;
#[cfg(test)]
mod tests;
pub use driver::*;
pub use search::*;

pub(crate) const MAIN_SCHEDULE_MAX_TUPLES: usize = 10_000_000;

/// Resolve a user option and its environment override independently for each log channel.
pub fn log_channel_enabled(option_enabled: bool, channel_env: &str) -> bool {
    let enabled = |value: &str| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    };
    if std::env::var("LUMINAL_LOG").is_ok_and(|value| enabled(&value)) {
        return true;
    }
    std::env::var(channel_env)
        .map(|value| enabled(&value))
        .unwrap_or(option_enabled)
}
