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
pub(crate) use self::{
    trace::{format_idwt_batch_trace_row, idwt_batch_kernel_mode, idwt_batch_trace_row},
    types::{CudaJ2kIdwtMultiKernelJob, CudaJ2kStoreRgb8MctBatchJob},
    validation::{
        active_dwt53_buffers, checked_f32_words_byte_len, ensure_idwt_buffer_len,
        j2k_idwt_multi_kernel_jobs, validate_store_rgb8_plane,
    },
};

#[cfg(test)]
mod structure_tests {
    const ROOT: &str = include_str!("j2k_decode.rs");
    const MODULES: &[(&str, &str, usize)] = &[
        (
            "j2k_decode/idwt.rs",
            include_str!("j2k_decode/idwt.rs"),
            700,
        ),
        (
            "j2k_decode/idwt_launch.rs",
            include_str!("j2k_decode/idwt_launch.rs"),
            350,
        ),
        (
            "j2k_decode/store.rs",
            include_str!("j2k_decode/store.rs"),
            625,
        ),
        (
            "j2k_decode/store_launch.rs",
            include_str!("j2k_decode/store_launch.rs"),
            225,
        ),
        (
            "j2k_decode/trace.rs",
            include_str!("j2k_decode/trace.rs"),
            175,
        ),
        (
            "j2k_decode/types.rs",
            include_str!("j2k_decode/types.rs"),
            375,
        ),
        (
            "j2k_decode/validation.rs",
            include_str!("j2k_decode/validation.rs"),
            150,
        ),
    ];

    #[test]
    fn cuda_j2k_decode_uses_focused_real_modules() {
        let include_macro = ["include", "!("].concat();
        let wildcard_import = ["use super::", "*"].concat();
        assert!(
            ROOT.lines().count() < 150,
            "j2k_decode.rs must remain a focused module shell"
        );
        for module in [
            "mod idwt;",
            "mod idwt_launch;",
            "mod store;",
            "mod store_launch;",
            "mod trace;",
            "mod types;",
            "mod validation;",
        ] {
            assert!(ROOT.contains(module), "j2k_decode.rs must contain {module}");
        }
        assert!(!ROOT.contains(&include_macro));

        for (path, source, max_lines) in MODULES {
            assert!(
                source.lines().count() < *max_lines,
                "{path} must stay below its focused-module line-count ratchet"
            );
            assert!(
                !source.contains(&include_macro),
                "{path} must be a real module"
            );
            assert!(
                !source.contains(&wildcard_import),
                "{path} must use explicit imports"
            );
        }
    }
}
