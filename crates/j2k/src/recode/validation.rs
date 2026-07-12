// SPDX-License-Identifier: MIT OR Apache-2.0

//! Recode round-trip comparison isolated from route and output ownership.

use alloc::vec::Vec;

use super::map_native_decode_error;
use crate::{BackendError, BackendErrorKind, J2kError};
use j2k_native::{DecodeSettings, Image};

// This intentionally receives the `Vec`: paired validation budgets its
// retained allocation capacity, not only the initialized codestream bytes.
pub(super) fn roundtrip(
    source: &[u8],
    encoded: &Vec<u8>,
    context: &'static str,
) -> Result<(), J2kError> {
    let settings = DecodeSettings::default();
    let source_image = Image::new_with_retained_baseline(source, &settings, encoded.capacity())
        .map_err(|err| map_native_decode_error(err, "source JPEG 2000 validation parse failed"))?;
    let source_metadata_bytes = source_image.retained_allocation_bytes().map_err(|err| {
        map_native_decode_error(err, "source JPEG 2000 validation metadata failed")
    })?;
    let encoded_parse_baseline = encoded.capacity().saturating_add(source_metadata_bytes);
    let encoded_image =
        Image::new_with_retained_baseline(encoded, &settings, encoded_parse_baseline)
            .map_err(|err| map_native_decode_error(err, "HTJ2K validation parse failed"))?;

    let equal = source_image
        .decoded_samples_equal_with_retained_bytes(&encoded_image, encoded)
        .map_err(|err| map_native_decode_error(err, "JPEG 2000 paired validation decode failed"))?;
    if !equal {
        return Err(J2kError::Backend(BackendError::new(
            BackendErrorKind::Validation,
            format!("{context} failed decoded-sample validation"),
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use j2k_native::EncodeOptions;

    fn codestream(sample: u8) -> Vec<u8> {
        j2k_native::encode(
            &[sample],
            1,
            1,
            1,
            8,
            false,
            &EncodeOptions {
                num_decomposition_levels: 0,
                ..EncodeOptions::default()
            },
        )
        .expect("encode validation fixture")
    }

    #[test]
    fn decoded_sample_mismatch_is_a_validation_backend_error() {
        let source = codestream(0);
        let encoded = codestream(1);
        let error = roundtrip(&source, &encoded, "test recode")
            .expect_err("different samples must fail round-trip validation");

        assert!(matches!(
            error,
            J2kError::Backend(ref backend)
                if backend.kind() == BackendErrorKind::Validation
                    && backend.message() == "test recode failed decoded-sample validation"
        ));
    }
}
