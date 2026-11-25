// SPDX-License-Identifier: MIT OR Apache-2.0

//! Telling a cut-off partial codestream from a malformed one.

use super::codestream::markers;

/// Whether a partial codestream holds a header in full: walking its marker
/// segments from `start` reaches `terminator` before the data ends.
///
/// A header that is present but malformed (a byte that is not a marker, a
/// segment length below two, an EOC) counts as present, so the parser
/// reports it as the error it is rather than as missing data.
pub(crate) fn header_is_present(data: &[u8], start: usize, terminator: u8) -> bool {
    let mut pos = start;
    loop {
        let Some(&[prefix, marker]) = data.get(pos..pos.saturating_add(2)) else {
            return false;
        };
        if prefix != 0xff || marker == terminator || marker == markers::EOC {
            return true;
        }
        if (0x30..=0x3f).contains(&marker) {
            // Reserved marker codes without marker segment parameters.
            pos += 2;
            continue;
        }
        let Some(&[high, low]) = data.get(pos + 2..pos.saturating_add(4)) else {
            return false;
        };
        let length = usize::from(u16::from_be_bytes([high, low]));
        if length < 2 {
            return true;
        }
        pos = pos.saturating_add(2 + length);
        if pos > data.len() {
            return false;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::header_is_present;
    use crate::j2c::codestream::markers;

    #[test]
    fn reports_where_the_data_ends() {
        // SOC, a COM segment with two payload bytes, then SOT.
        let header = [0xff, 0x4f, 0xff, 0x64, 0x00, 0x04, 0xaa, 0xbb, 0xff, 0x90];
        assert!(header_is_present(&header, 2, markers::SOT));
        for len in 0..header.len() {
            assert!(!header_is_present(&header[..len], 2, markers::SOT), "{len}");
        }
    }

    #[test]
    fn malformed_headers_count_as_present() {
        assert!(header_is_present(
            &[0xff, 0x4f, 0x12, 0x34],
            2,
            markers::SOT
        ));
        assert!(header_is_present(
            &[0xff, 0x4f, 0xff, 0x64, 0x00, 0x01],
            2,
            markers::SOT
        ));
        assert!(header_is_present(
            &[0xff, 0x4f, 0xff, 0xd9],
            2,
            markers::SOT
        ));
    }
}
