// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::borrow::Cow;

use crate::error::JpegError;
use crate::internal::checkpoint::{terminated_scan_bytes, terminated_scan_bytes_with_cap};

#[test]
fn terminated_scan_borrows_an_existing_eoi_and_bounds_required_copy() {
    let terminated = [0x00, 0xff, 0xd9];
    let borrowed = terminated_scan_bytes_with_cap(&terminated, 0)
        .expect("an already terminated scan needs no allocation");
    assert!(matches!(borrowed, Cow::Borrowed(_)));
    assert_eq!(borrowed.as_ptr(), terminated.as_ptr());

    let with_trailing_container_bytes = [0x00, 0xff, 0xd9, 0xaa, 0xbb];
    let prefix = terminated_scan_bytes_with_cap(&with_trailing_container_bytes, 0)
        .expect("bytes after the first EOI require no allocation");
    assert!(matches!(prefix, Cow::Borrowed(_)));
    assert_eq!(prefix.as_ref(), &[0x00, 0xff, 0xd9]);

    let unterminated = [0x00];
    let error = terminated_scan_bytes_with_cap(&unterminated, 2)
        .expect_err("copy plus EOI exceeds the supplied cap");
    assert_eq!(
        error,
        JpegError::MemoryCapExceeded {
            requested: 3,
            cap: 2,
        }
    );
}

#[test]
fn terminated_scan_appends_only_the_missing_eoi_bytes() {
    let unterminated = [0x12, 0x34];
    let missing_marker = terminated_scan_bytes(&unterminated).expect("append EOI marker");
    assert_eq!(missing_marker.as_ref(), &[0x12, 0x34, 0xff, 0xd9]);

    let marker_without_code = [0x12, 0xff];
    let missing_code = terminated_scan_bytes(&marker_without_code).expect("append EOI code");
    assert_eq!(missing_code.as_ref(), &[0x12, 0xff, 0xd9]);
}
