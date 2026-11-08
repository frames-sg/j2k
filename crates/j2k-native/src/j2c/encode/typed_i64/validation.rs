// SPDX-License-Identifier: MIT OR Apache-2.0

//! Non-allocating option validation shared by exact-i64 routes.

use super::super::{
    validate_common_encode_options, EncodeOptions, NativeEncodePipelineError,
    NativeEncodePipelineResult,
};

pub(super) fn validate_high_bit_options(options: &EncodeOptions) -> NativeEncodePipelineResult<()> {
    validate_common_encode_options(options)?;
    if options.irreversible_quantization_profile.is_some() {
        return Err(NativeEncodePipelineError::invalid_input(
            "component quantization profile requires irreversible encoding",
        ));
    }
    Ok(())
}
