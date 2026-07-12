// SPDX-License-Identifier: MIT OR Apache-2.0

mod idwt;
mod idwt_launch;
mod store;
mod store_launch;
mod trace;
mod types;
mod validation;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CudaJ2kIdwtBatchKernelMode {
    Generic,
    Cooperative53,
    Cooperative97,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CudaJ2kIdwtBatchTraceRow {
    pub(crate) stage_index: usize,
    pub(crate) mode: CudaJ2kIdwtBatchKernelMode,
    pub(crate) job_count: usize,
    pub(crate) max_width: u32,
    pub(crate) max_height: u32,
    pub(crate) min_width: u32,
    pub(crate) min_height: u32,
    pub(crate) total_pixels: u64,
    pub(crate) irreversible_jobs: usize,
    pub(crate) elapsed_us: u128,
}

#[cfg(test)]
pub(crate) use self::trace::idwt_batch_uses_cooperative_53;
pub use self::types::{
    CudaJ2kIdwtJob, CudaJ2kIdwtTarget, CudaJ2kInverseMctJob, CudaJ2kRect, CudaJ2kStoreGray16Job,
    CudaJ2kStoreGray8Job, CudaJ2kStoreRgb16Job, CudaJ2kStoreRgb16MctJob, CudaJ2kStoreRgb8Job,
    CudaJ2kStoreRgb8MctJob, CudaJ2kStoreRgb8MctTarget, CudaJ2kStridedInterleavedPixels,
};
#[cfg(test)]
pub(crate) use self::validation::checked_f32_words_byte_len;
pub(crate) use self::{
    trace::{format_idwt_batch_trace_row, idwt_batch_kernel_mode, idwt_batch_trace_row},
    types::{CudaJ2kIdwtMultiKernelJob, CudaJ2kStoreRgb8MctBatchJob},
    validation::{
        active_dwt53_buffers, append_j2k_idwt_multi_kernel_jobs, j2k_idwt_multi_kernel_jobs,
    },
};

#[cfg(test)]
mod structure_tests;
