// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::build_flags::PINNED_POOLED_I16_UPLOAD_MAX_BYTES;

#[cfg(test)]
mod abi_tests;
mod dwt97;
mod htj2k97;
mod launch;
mod readback;
mod reversible53;
mod types;
mod validation;

pub use self::types::{
    CudaDwt97BatchGeometry, CudaDwt97BatchWithPoolRequest, CudaHtj2k97CodeblockBands,
    CudaHtj2k97CodeblockBatchWithPoolRequest, CudaHtj2k97DeviceCodeblockBands,
    CudaHtj2k97I16CodeblockBatchWithPoolRequest, CudaHtj2k97QuantizeParams,
    CudaTranscodeDwt97Bands, CudaTranscodeReversible53Bands,
};
pub(crate) use self::{
    types::{
        DctBlockGrid, Dwt97BatchDeviceBands, Dwt97BatchInput, Dwt97CodeblockBandBuffers,
        Reversible53Dims,
    },
    validation::{checked_i32, validate_dct_block_grid},
};

pub(crate) fn should_use_pinned_pooled_i16_upload(byte_len: usize) -> bool {
    byte_len <= PINNED_POOLED_I16_UPLOAD_MAX_BYTES
}

#[cfg(test)]
mod structure_tests {
    const ROOT: &str = include_str!("transcode.rs");
    const MODULES: &[(&str, &str, usize)] = &[
        (
            "transcode/abi_tests.rs",
            include_str!("transcode/abi_tests.rs"),
            75,
        ),
        (
            "transcode/dwt97.rs",
            include_str!("transcode/dwt97.rs"),
            350,
        ),
        (
            "transcode/htj2k97.rs",
            include_str!("transcode/htj2k97.rs"),
            450,
        ),
        (
            "transcode/launch.rs",
            include_str!("transcode/launch.rs"),
            475,
        ),
        (
            "transcode/readback.rs",
            include_str!("transcode/readback.rs"),
            150,
        ),
        (
            "transcode/reversible53.rs",
            include_str!("transcode/reversible53.rs"),
            175,
        ),
        (
            "transcode/types.rs",
            include_str!("transcode/types.rs"),
            350,
        ),
        (
            "transcode/validation.rs",
            include_str!("transcode/validation.rs"),
            150,
        ),
    ];

    #[test]
    fn cuda_transcode_host_uses_focused_real_modules() {
        let include_macro = ["include", "!("].concat();
        let wildcard_import = ["use super::", "*"].concat();
        assert!(
            ROOT.lines().count() < 125,
            "transcode.rs must remain a focused module shell"
        );
        for module in [
            "mod dwt97;",
            "mod htj2k97;",
            "mod launch;",
            "mod readback;",
            "mod reversible53;",
            "mod types;",
            "mod validation;",
        ] {
            assert!(ROOT.contains(module), "transcode.rs must contain {module}");
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
