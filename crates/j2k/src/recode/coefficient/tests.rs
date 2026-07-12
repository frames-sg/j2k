// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn coefficient_encode_errors_preserve_native_source() {
    let error = map_coefficient_encode_error(j2k_native::EncodeError::CodestreamValidation {
        detail: "coefficient validation fixture",
    });

    assert!(matches!(
        error,
        J2kError::NativeEncode {
            context: "native HTJ2K coefficient recode failed",
            source,
        }
        if source == crate::NativeBackendError::encode(
            j2k_native::EncodeError::CodestreamValidation {
                detail: "coefficient validation fixture",
            },
        )
    ));
}
