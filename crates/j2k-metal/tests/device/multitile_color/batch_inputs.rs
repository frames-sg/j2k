// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

pub(super) fn independent_multitile_inputs(
    encoded: &[u8],
    request: DecodeRequest,
) -> Vec<EncodedImage> {
    let first = Arc::<[u8]>::from(encoded);
    let second = Arc::<[u8]>::from(encoded);
    assert!(
        !Arc::ptr_eq(&first, &second),
        "batch regression requires independent encoded-byte owners"
    );
    vec![
        EncodedImage::new(first, request),
        EncodedImage::new(second, request),
    ]
}
