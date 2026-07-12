// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::read_runtime;

pub(super) struct Htj2kOutputSources {
    pub(super) context: String,
    pub(super) decode_root: String,
    pub(super) decode: String,
    pub(super) decode_api: String,
    pub(super) decode_completion: String,
    pub(super) decode_dequant: String,
    pub(super) decode_planning: String,
    pub(super) decode_types: String,
    pub(super) context_validation: String,
    pub(super) output_regions: String,
    pub(super) output_region_sweep: String,
    pub(super) output_region_cross_stride: String,
    pub(super) output_region_tests: String,
}

impl Htj2kOutputSources {
    pub(super) fn read() -> Self {
        let decode_root = read_runtime("htj2k_decode.rs");
        let decode_api = read_runtime("htj2k_decode/api.rs");
        let decode_completion = read_runtime("htj2k_decode/completion.rs");
        let decode_dequant = read_runtime("htj2k_decode/completion/dequant.rs");
        let decode_planning = read_runtime("htj2k_decode/planning.rs");
        let decode_types = read_runtime("htj2k_decode/types.rs");
        let decode = [
            decode_root.as_str(),
            decode_api.as_str(),
            decode_completion.as_str(),
            decode_dequant.as_str(),
            decode_planning.as_str(),
            decode_types.as_str(),
        ]
        .concat();
        Self {
            context: read_runtime("context.rs"),
            decode_root,
            decode,
            decode_api,
            decode_completion,
            decode_dequant,
            decode_planning,
            decode_types,
            context_validation: read_runtime("htj2k_decode/context_validation.rs"),
            output_regions: read_runtime("htj2k_decode/output_regions.rs"),
            output_region_sweep: read_runtime("htj2k_decode/output_regions/sweep.rs"),
            output_region_cross_stride: read_runtime(
                "htj2k_decode/output_regions/sweep/cross_stride.rs",
            ),
            output_region_tests: read_runtime("htj2k_decode/output_regions/tests.rs"),
        }
    }
}
