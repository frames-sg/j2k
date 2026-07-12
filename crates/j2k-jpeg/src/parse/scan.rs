// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parse the Start-of-Scan segment payload into per-component DC/AC Huffman
//! table selectors plus the spectral selection (`Ss`, `Se`) and successive-
//! approximation (`Ah`, `Al`) bytes. See T.81 §B.2.3.
//!
//! Layout:
//! ```text
//! SOS payload:
//!   byte[0]         = Ns (component count in scan)
//!   bytes[1..1+2*Ns] = [Cs_i (1), Td_i<<4 | Ta_i (1)] × Ns
//!   bytes[last-3..last] = Ss, Se, Ah<<4 | Al
//! ```

use crate::error::JpegError;
use core::ops::{Deref, Index};

const MAX_SCAN_COMPONENTS: usize = 4;

/// One entry in a scan's component list: which of the parsed components (by
/// `component_id` from SOF) participates, and which Huffman tables it uses.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ScanComponent {
    /// `Cs_i` from the SOS payload — must match a component id from SOF.
    pub(crate) id: u8,
    /// DC Huffman table selector, 0..=3. `Td_i` from the high nibble.
    pub(crate) dc_table: u8,
    /// AC Huffman table selector, 0..=3. `Ta_i` from the low nibble.
    pub(crate) ac_table: u8,
}

/// Inline SOS component selectors. The decoder accepts at most four frame
/// components, so a heap owner per scan only adds allocator failure surface.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct ScanComponents {
    entries: [ScanComponent; MAX_SCAN_COMPONENTS],
    len: u8,
}

impl ScanComponents {
    fn push(&mut self, component: ScanComponent) -> Result<(), JpegError> {
        let index = usize::from(self.len);
        let slot = self
            .entries
            .get_mut(index)
            .ok_or(JpegError::UnsupportedComponentCount {
                count: self.len.saturating_add(1),
            })?;
        *slot = component;
        self.len += 1;
        Ok(())
    }

    pub(crate) fn as_slice(&self) -> &[ScanComponent] {
        &self.entries[..usize::from(self.len)]
    }

    pub(crate) fn len(&self) -> usize {
        usize::from(self.len)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub(crate) fn iter(&self) -> core::slice::Iter<'_, ScanComponent> {
        self.as_slice().iter()
    }
}

impl Deref for ScanComponents {
    type Target = [ScanComponent];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl Index<usize> for ScanComponents {
    type Output = ScanComponent;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<'a> IntoIterator for &'a ScanComponents {
    type Item = &'a ScanComponent;
    type IntoIter = core::slice::Iter<'a, ScanComponent>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Parsed SOS header (not including the entropy-coded scan bytes that follow).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParsedScan {
    pub(crate) components: ScanComponents,
    /// Spectral start. Baseline sequential always has `ss == 0`.
    pub(crate) ss: u8,
    /// Spectral end. Baseline sequential always has `se == 63`.
    pub(crate) se: u8,
    /// Successive-approximation high nibble. Baseline sequential always `0`.
    pub(crate) ah: u8,
    /// Successive-approximation low nibble. Baseline sequential always `0`.
    pub(crate) al: u8,
}

/// Parse the SOS payload. `payload` is the bytes after the 2-byte SOS length
/// field — i.e. starts with `Ns`. Returns a structural error on wrong length
/// or out-of-range selectors.
#[expect(
    clippy::cast_possible_truncation,
    reason = "SOS payload lengths are bounded by the JPEG 16-bit segment-length field"
)]
pub(crate) fn parse_scan_header(payload: &[u8], offset: usize) -> Result<ParsedScan, JpegError> {
    if payload.is_empty() {
        return Err(JpegError::InvalidSegmentLength {
            offset,
            marker: 0xDA,
            length: 2,
        });
    }
    let ns = payload[0] as usize;
    let expected = 1 + ns * 2 + 3;
    if payload.len() != expected {
        return Err(JpegError::InvalidSegmentLength {
            offset,
            marker: 0xDA,
            length: (payload.len() + 2) as u16,
        });
    }
    if ns > MAX_SCAN_COMPONENTS {
        return Err(JpegError::UnsupportedComponentCount { count: payload[0] });
    }
    let mut components = ScanComponents::default();
    for i in 0..ns {
        let base = 1 + i * 2;
        let id = payload[base];
        let td_ta = payload[base + 1];
        let dc_table = td_ta >> 4;
        let ac_table = td_ta & 0x0F;
        if dc_table > 3 || ac_table > 3 {
            return Err(JpegError::InvalidSegmentLength {
                offset: offset + base + 1,
                marker: 0xDA,
                length: (payload.len() + 2) as u16,
            });
        }
        components.push(ScanComponent {
            id,
            dc_table,
            ac_table,
        })?;
    }
    let last = 1 + ns * 2;
    let ss = payload[last];
    let se = payload[last + 1];
    let ahal = payload[last + 2];
    Ok(ParsedScan {
        components,
        ss,
        se,
        ah: ahal >> 4,
        al: ahal & 0x0F,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn parses_three_component_baseline_scan() {
        // Ns=3; Y uses DC0/AC0, Cb uses DC1/AC1, Cr uses DC1/AC1; Ss=0 Se=63 Ah=0 Al=0
        let payload = vec![3u8, 1, 0x00, 2, 0x11, 3, 0x11, 0, 63, 0];
        let scan = parse_scan_header(&payload, 0).unwrap();
        assert_eq!(scan.components.len(), 3);
        assert_eq!(
            scan.components[0],
            ScanComponent {
                id: 1,
                dc_table: 0,
                ac_table: 0
            }
        );
        assert_eq!(
            scan.components[1],
            ScanComponent {
                id: 2,
                dc_table: 1,
                ac_table: 1
            }
        );
        assert_eq!(
            scan.components[2],
            ScanComponent {
                id: 3,
                dc_table: 1,
                ac_table: 1
            }
        );
        assert_eq!((scan.ss, scan.se, scan.ah, scan.al), (0, 63, 0, 0));
    }

    #[test]
    fn parses_single_component_grayscale_scan() {
        let payload = vec![1u8, 1, 0x00, 0, 63, 0];
        let scan = parse_scan_header(&payload, 0).unwrap();
        assert_eq!(scan.components.len(), 1);
        assert_eq!(scan.components[0].id, 1);
    }

    #[test]
    fn rejects_empty_payload() {
        let err = parse_scan_header(&[], 10).unwrap_err();
        assert!(matches!(err, JpegError::InvalidSegmentLength { .. }));
    }

    #[test]
    fn rejects_length_mismatch() {
        // ns=2 declared but payload sized for ns=1
        let payload = vec![2u8, 1, 0x00, 0, 63, 0];
        let err = parse_scan_header(&payload, 0).unwrap_err();
        assert!(matches!(err, JpegError::InvalidSegmentLength { .. }));
    }

    #[test]
    fn rejects_out_of_range_table_selector() {
        // Ta=5 (>3)
        let payload = vec![1u8, 1, 0x05, 0, 63, 0];
        let err = parse_scan_header(&payload, 0).unwrap_err();
        assert!(matches!(err, JpegError::InvalidSegmentLength { .. }));
    }

    #[test]
    fn rejects_more_than_four_scan_components_without_allocating() {
        let payload = vec![5u8, 1, 0x00, 2, 0x00, 3, 0x00, 4, 0x00, 5, 0x00, 0, 63, 0];
        assert_eq!(
            parse_scan_header(&payload, 0),
            Err(JpegError::UnsupportedComponentCount { count: 5 })
        );
    }
}
