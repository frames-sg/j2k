// SPDX-License-Identifier: MIT OR Apache-2.0

mod cpu;
mod execute;
mod heuristics;
mod request;
mod routes;
mod session;
#[cfg(test)]
mod tests;

use self::cpu::decode_cpu_host_batch;
pub(crate) use self::request::BatchOp;
use self::request::{batch_scheduler_invariant, QueuedRequest};
pub use self::request::{benchmark_group_region_scaled_requests, BenchmarkGroupedRequests};
use self::routes::{
    decode_distinct_full_color_batch, decode_distinct_full_grayscale_batch,
    decode_distinct_region_scaled_direct_batch,
    decode_distinct_region_scaled_direct_batch_prechecked, decode_individual,
    decode_repeated_full_color, decode_repeated_full_grayscale,
    decode_repeated_region_scaled_direct_batch_prechecked,
};
pub use self::session::MetalSubmission;
use self::session::SessionState;
pub(crate) use self::session::{
    queue_tile_request, queue_tile_request_shared_with_retained, SharedSession,
};
