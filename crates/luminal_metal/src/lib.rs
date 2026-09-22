pub mod dyn_backend;
pub mod kernel;
mod memory_analysis;
pub mod runtime;
pub mod typed_rules;

#[cfg(test)]
mod tests;

pub use metal::{Buffer, Device, MTLResourceOptions};
pub use objc::rc::autoreleasepool;
pub use runtime::MetalRuntime;
