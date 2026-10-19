// SPDX-License-Identifier: MIT OR Apache-2.0

//! APP14 "Adobe" color-transform detection.
//!
//! Layout per libjpeg-turbo's `jdmarker.c` and the Adobe DCT filters guide:
//!
//! ```text
//! APP14 payload:
//!   bytes[0..5]  = "Adobe" ASCII signature
//!   bytes[5..7]  = DCTEncodeVersion (unused)
//!   bytes[7..9]  = APP14Flags0
//!   bytes[9..11] = APP14Flags1
//!   bytes[11]    = ColorTransform:  0 = Unknown (no transform applied),
//!                                   1 = YCbCr,
//!                                   2 = YCCK,
//!                                   3.. reserved
//! ```
//!
//! We return `None` for non-Adobe APP14 segments (any APP14 not beginning
//! with `"Adobe"` — those may be Adobe DCT but in an older container).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AdobeTransform {
    Unknown,  // 0 — no transform: RGB for 3 components, CMYK for 4
    YCbCr,    // 1
    Ycck,     // 2
    Reserved, // 3.. — libjpeg-turbo warns and assumes YCbCr (3) or YCCK (4)
}

/// Whether an APP0 payload is a JFIF header, as libjpeg-turbo's
/// `examine_app0` detects it: the `JFIF\0` identifier and at least the
/// 14-byte JFIF APP0 body.
pub(crate) fn is_jfif_app0(payload: &[u8]) -> bool {
    payload.len() >= 14 && payload.starts_with(b"JFIF\0")
}

pub(crate) fn parse_adobe_app14(payload: &[u8]) -> Option<AdobeTransform> {
    if payload.len() < 12 || &payload[0..5] != b"Adobe" {
        return None;
    }
    match payload[11] {
        1 => Some(AdobeTransform::YCbCr),
        2 => Some(AdobeTransform::Ycck),
        0 => Some(AdobeTransform::Unknown),
        _ => Some(AdobeTransform::Reserved), // emit warning at caller
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn adobe_payload(transform: u8) -> Vec<u8> {
        let mut v = vec![b'A', b'd', b'o', b'b', b'e'];
        v.extend_from_slice(&[0x00, 0x64]); // DCTEncodeVersion 100
        v.extend_from_slice(&[0x00, 0x00]); // flags0
        v.extend_from_slice(&[0x00, 0x00]); // flags1
        v.push(transform);
        v
    }

    #[test]
    fn parses_adobe_ycbcr_transform() {
        assert_eq!(
            parse_adobe_app14(&adobe_payload(1)),
            Some(AdobeTransform::YCbCr)
        );
    }

    #[test]
    fn parses_adobe_ycck_transform() {
        assert_eq!(
            parse_adobe_app14(&adobe_payload(2)),
            Some(AdobeTransform::Ycck)
        );
    }

    #[test]
    fn parses_adobe_unknown_transform_zero() {
        assert_eq!(
            parse_adobe_app14(&adobe_payload(0)),
            Some(AdobeTransform::Unknown)
        );
    }

    #[test]
    fn parses_adobe_reserved_transform_out_of_range() {
        assert_eq!(
            parse_adobe_app14(&adobe_payload(99)),
            Some(AdobeTransform::Reserved)
        );
    }

    #[test]
    fn detects_jfif_app0_only_with_identifier_and_full_body() {
        let jfif = b"JFIF\0\x01\x02\x00\x00\x01\x00\x01\x00\x00";
        assert!(is_jfif_app0(jfif));
        assert!(!is_jfif_app0(&jfif[..13]));
        assert!(!is_jfif_app0(b"JFXX\0\x10\x00\x00\x00\x00\x00\x00\x00\x00"));
    }

    #[test]
    fn rejects_non_adobe_signature() {
        let mut p = adobe_payload(1);
        p[0] = b'X';
        assert_eq!(parse_adobe_app14(&p), None);
    }

    #[test]
    fn rejects_short_payload() {
        assert_eq!(parse_adobe_app14(b"Adobe"), None);
    }
}
