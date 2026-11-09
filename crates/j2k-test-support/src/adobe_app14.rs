// SPDX-License-Identifier: MIT OR Apache-2.0

//! Adobe APP14 transform rewrites for JPEG color-interpretation tests.

/// Reserved Adobe APP14 transform values: anything but 0 (no transform),
/// 1 (YCbCr) and 2 (YCCK).
pub const RESERVED_ADOBE_TRANSFORMS: [u8; 3] = [3, 7, 255];

/// `jpeg` with its Adobe APP14 transform byte set to `transform`.
///
/// A stream without an Adobe marker gets one after SOI, and its JFIF APP0
/// marker, which would fix a three-component frame as YCbCr, is dropped.
/// Everything from the first SOS on is copied unchanged.
///
/// # Panics
///
/// Panics if `jpeg` does not start with SOI or its marker segments before the
/// first SOS are malformed.
#[must_use]
pub fn with_adobe_transform(jpeg: &[u8], transform: u8) -> Vec<u8> {
    assert!(jpeg.starts_with(&[0xFF, 0xD8]), "JPEG must start with SOI");
    let mut out = jpeg[..2].to_vec();
    let mut offset = 2;
    let mut rewritten = false;
    while jpeg[offset + 1] != 0xDA {
        assert_eq!(jpeg[offset], 0xFF, "marker expected at byte {offset}");
        let length = usize::from(u16::from_be_bytes([jpeg[offset + 2], jpeg[offset + 3]]));
        let segment = &jpeg[offset..offset + 2 + length];
        let payload = &segment[4..];
        match segment[1] {
            0xE0 if payload.starts_with(b"JFIF\0") => {}
            // The transform is payload byte 11, after "Adobe", version and flags.
            0xEE if payload.starts_with(b"Adobe") && payload.len() >= 12 => {
                out.extend_from_slice(&segment[..15]);
                out.push(transform);
                out.extend_from_slice(&segment[16..]);
                rewritten = true;
            }
            _ => out.extend_from_slice(segment),
        }
        offset += 2 + length;
    }
    if !rewritten {
        let adobe = [
            0xFF, 0xEE, 0x00, 0x0E, b'A', b'd', b'o', b'b', b'e', 0x00, 0x64, 0x00, 0x00, 0x00,
            0x00, transform,
        ];
        out.splice(2..2, adobe);
    }
    out.extend_from_slice(&jpeg[offset..]);
    out
}

#[cfg(test)]
mod tests {
    use super::with_adobe_transform;

    #[test]
    fn rewrites_an_existing_adobe_transform_in_place() {
        let jpeg = crate::cmyk_8x8_jpeg();
        let rewritten = with_adobe_transform(&jpeg, 7);
        assert_eq!(rewritten.len(), jpeg.len());
        let adobe = jpeg.windows(5).position(|part| part == b"Adobe").unwrap();
        let differing: Vec<usize> = (0..jpeg.len())
            .filter(|&index| jpeg[index] != rewritten[index])
            .collect();
        assert_eq!(differing, [adobe + 11]);
        assert_eq!(rewritten[adobe + 11], 7);
    }

    #[test]
    fn replaces_jfif_with_an_adobe_marker() {
        let jpeg = crate::JPEG_BASELINE_420_16X16;
        assert!(jpeg.windows(5).any(|part| part == b"JFIF\0"));
        let rewritten = with_adobe_transform(jpeg, 255);
        assert!(!rewritten.windows(5).any(|part| part == b"JFIF\0"));
        assert_eq!(&rewritten[2..6], [0xFF, 0xEE, 0x00, 0x0E]);
        assert_eq!(&rewritten[6..11], b"Adobe");
        assert_eq!(rewritten[17], 255);
        let sos = |bytes: &[u8]| {
            bytes
                .windows(2)
                .position(|pair| pair == [0xFF, 0xDA])
                .unwrap()
        };
        assert_eq!(&rewritten[sos(&rewritten)..], &jpeg[sos(jpeg)..]);
    }
}
