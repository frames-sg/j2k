// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(test)]
mod abi_tests;
mod decode;
mod diagnostics;
mod encode;
mod types;
mod validation;

pub(crate) use self::types::{
    CudaJpeg420Params, CudaJpegBaselineEncodeStatus, CudaJpegDecodeStatus,
    CudaJpegEntropyChunkParams,
};
pub use self::types::{
    CudaJpegBaselineEncodeFormat, CudaJpegBaselineEncodeHuffmanTable, CudaJpegBaselineEncodeParams,
    CudaJpegBaselineEntropyEncodeBatchJob, CudaJpegBaselineEntropyEncodeJob,
    CudaJpegChunkedEntropyConfig, CudaJpegChunkedEntropyPlan, CudaJpegChunkedEntropyReport,
    CudaJpegEntropyCheckpoint, CudaJpegEntropyOverflowState, CudaJpegEntropySyncState,
    CudaJpegHuffmanTable, CudaJpegRgb8DecodePlan, CudaJpegRgb8Sampling,
};
#[cfg(feature = "cuda-oxide-jpeg-decode")]
pub(crate) use self::{
    types::CudaJpegRgb8ValidatedPlan,
    validation::{
        jpeg_rgb8_kernel, validate_jpeg_entropy_chunk_plan, validate_jpeg_rgb8_plan,
        validate_jpeg_rgb8_plan_with_pitch,
    },
};

const _: [(); 32] = [(); core::mem::size_of::<CudaJpeg420Params>()];
const _: [(); 32] = [(); core::mem::size_of::<CudaJpegEntropyChunkParams>()];

#[cfg_attr(
    all(not(feature = "cuda-oxide-jpeg-decode"), not(test)),
    expect(
        dead_code,
        reason = "overflow accounting is used only by the JPEG decode path"
    )
)]
pub(crate) fn jpeg_entropy_overflow_count(subsequence_count: usize) -> usize {
    subsequence_count.saturating_sub(1)
}

#[cfg(test)]
mod structure_tests {
    const ROOT: &str = include_str!("jpeg.rs");
    const MODULES: &[(&str, &str, usize)] = &[
        ("jpeg/abi_tests.rs", include_str!("jpeg/abi_tests.rs"), 100),
        ("jpeg/decode.rs", include_str!("jpeg/decode.rs"), 350),
        (
            "jpeg/diagnostics.rs",
            include_str!("jpeg/diagnostics.rs"),
            325,
        ),
        ("jpeg/encode.rs", include_str!("jpeg/encode.rs"), 425),
        ("jpeg/types.rs", include_str!("jpeg/types.rs"), 575),
        (
            "jpeg/validation.rs",
            include_str!("jpeg/validation.rs"),
            175,
        ),
    ];

    #[test]
    fn cuda_jpeg_runtime_uses_focused_real_modules() {
        let include_macro = ["include", "!("].concat();
        let wildcard_import = ["use super::", "*"].concat();
        assert!(
            ROOT.lines().count() < 100,
            "jpeg.rs must remain a focused module shell"
        );
        for module in [
            "mod decode;",
            "mod diagnostics;",
            "mod encode;",
            "mod types;",
            "mod validation;",
        ] {
            assert!(ROOT.contains(module), "jpeg.rs must contain {module}");
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
