//! Binary file comparison — port of `OpenJPH/tests/compare_files.cpp`.
//!
//! Compares two byte streams, ignoring COM (comment) marker segments in
//! the JPEG 2000 codestream header (before the first SOT marker).

/// Result of a file comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareResult {
    /// Files match (ignoring comments).
    Match,
    /// Files differ at the given byte offset.
    Mismatch { offset: usize },
    /// One file ended before the other.
    LengthMismatch,
}

/// Compare two byte slices, ignoring COM (0xFF64) marker segments in the
/// JPEG 2000 main header (before the first SOT = 0xFF90 marker).
///
/// This mirrors the C++ `compare_files` logic:
/// - Before a tile starts (SOT marker), when a COM marker (0xFF64) is
///   encountered, its length-prefixed body is skipped in both streams.
/// - After a tile starts, comments are no longer skipped.
/// - All other bytes must match exactly.
pub fn compare_j2k_bytes(data1: &[u8], data2: &[u8]) -> CompareResult {
    let mut i1 = 0usize;
    let mut i2 = 0usize;
    let mut tile_started = false;
    let mut old_c1: u8 = b' ';

    loop {
        let eof1 = i1 >= data1.len();
        let eof2 = i2 >= data2.len();

        if eof1 && eof2 {
            return CompareResult::Match;
        }
        if eof1 || eof2 {
            return CompareResult::LengthMismatch;
        }

        let c1 = data1[i1];
        let c2 = data2[i2];

        if c1 != c2 {
            return CompareResult::Mismatch { offset: i1 };
        }

        i1 += 1;
        i2 += 1;

        // Check for COM marker (0xFF 0x64) before tile starts
        if !tile_started && old_c1 == 0xFF && c1 == 0x64 {
            // Skip the comment body in both streams
            if !eat_comment(data1, &mut i1) || !eat_comment(data2, &mut i2) {
                return CompareResult::LengthMismatch;
            }
        }

        // Check for SOT marker (0xFF 0x90) — stop skipping comments
        if !tile_started && old_c1 == 0xFF && c1 == 0x90 {
            tile_started = true;
        }

        old_c1 = c1;
    }
}

/// Skip a COM marker segment body (reads length, then skips that many bytes).
fn eat_comment(data: &[u8], pos: &mut usize) -> bool {
    if *pos + 2 > data.len() {
        return false;
    }
    let length = ((data[*pos] as usize) << 8) | (data[*pos + 1] as usize);
    *pos += 2;
    let skip = length.saturating_sub(2);
    if *pos + skip > data.len() {
        return false;
    }
    *pos += skip;
    true
}

/// Compare two files on disk, ignoring COM markers in the JPEG 2000 header.
///
/// Returns the comparison result.
#[allow(dead_code)]
pub fn compare_j2k_files(path1: &str, path2: &str) -> std::io::Result<CompareResult> {
    let data1 = std::fs::read(path1)?;
    let data2 = std::fs::read(path2)?;
    Ok(compare_j2k_bytes(&data1, &data2))
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_files_match() {
        let data = vec![0xFF, 0x4F, 0xFF, 0x51, 0x00, 0x0A, 1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(compare_j2k_bytes(&data, &data), CompareResult::Match);
    }

    #[test]
    fn different_files_mismatch() {
        let data1 = vec![0xFF, 0x4F, 0xFF, 0x51, 0x00, 0x0A];
        let data2 = vec![0xFF, 0x4F, 0xFF, 0x51, 0x00, 0x0B];
        assert!(matches!(
            compare_j2k_bytes(&data1, &data2),
            CompareResult::Mismatch { .. }
        ));
    }

    #[test]
    fn length_mismatch() {
        let data1 = vec![0xFF, 0x4F, 0xFF];
        let data2 = vec![0xFF, 0x4F];
        assert_eq!(
            compare_j2k_bytes(&data1, &data2),
            CompareResult::LengthMismatch
        );
    }

    #[test]
    fn skip_comments_before_tile() {
        // SOC + COM marker with body "AB" + COD marker byte
        // COM = 0xFF 0x64, length = 0x0004 (includes 2 length bytes + 2 data bytes)
        let base = vec![0xFF, 0x4F]; // SOC
        let com1 = vec![0xFF, 0x64, 0x00, 0x04, b'A', b'B'];
        let com2 = vec![0xFF, 0x64, 0x00, 0x05, b'X', b'Y', b'Z'];
        let tail = vec![0xFF, 0x52, 0x01]; // COD + one byte

        let mut data1 = base.clone();
        data1.extend_from_slice(&com1);
        data1.extend_from_slice(&tail);

        let mut data2 = base.clone();
        data2.extend_from_slice(&com2);
        data2.extend_from_slice(&tail);

        assert_eq!(compare_j2k_bytes(&data1, &data2), CompareResult::Match);
    }

    #[test]
    fn no_skip_comments_after_tile() {
        // After SOT (0xFF90), COM markers should NOT be skipped
        let mut data1 = vec![0xFF, 0x4F, 0xFF, 0x90]; // SOC + SOT
        let mut data2 = vec![0xFF, 0x4F, 0xFF, 0x90];
        // Add COM in both — but with different content
        data1.extend_from_slice(&[0xFF, 0x64, 0x00, 0x04, b'A', b'B']);
        data2.extend_from_slice(&[0xFF, 0x64, 0x00, 0x04, b'X', b'Y']);
        // After SOT, these should NOT be skipped, so they should mismatch
        assert!(matches!(
            compare_j2k_bytes(&data1, &data2),
            CompareResult::Mismatch { .. }
        ));
    }
}
