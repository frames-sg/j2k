// SPDX-License-Identifier: MIT OR Apache-2.0

//! Single-pass scan validation and temporary missing-EOI ownership.

use alloc::borrow::Cow;
use alloc::vec::Vec;

use super::allocation::{checked_live_bytes, element_capacity_bytes, host_allocation_error};
use crate::error::{JpegError, MarkerKind};

#[derive(Debug, Clone, Copy)]
pub(crate) struct ValidatedScanBytes<'a> {
    bytes: &'a [u8],
    eoi_end: Option<usize>,
}

impl<'a> ValidatedScanBytes<'a> {
    pub(crate) fn payload(&self) -> &'a [u8] {
        self.eoi_end.map_or(self.bytes, |end| &self.bytes[..end])
    }

    pub(crate) fn is_missing_eoi(&self) -> bool {
        self.eoi_end.is_none()
    }

    pub(crate) fn terminated_copy_len(&self, allocation_cap: usize) -> Result<usize, JpegError> {
        if self.eoi_end.is_some() {
            return Ok(0);
        }
        self.bytes
            .len()
            .checked_add(self.missing_eoi_suffix().len())
            .ok_or(JpegError::MemoryCapExceeded {
                requested: usize::MAX,
                cap: allocation_cap,
            })
    }

    pub(crate) fn terminated_with_live_budget(
        &self,
        initial_live_bytes: usize,
        allocation_cap: usize,
    ) -> Result<Cow<'a, [u8]>, JpegError> {
        if let Some(eoi_end) = self.eoi_end {
            return Ok(Cow::Borrowed(&self.bytes[..eoi_end]));
        }

        let suffix = self.missing_eoi_suffix();
        let requested = self.terminated_copy_len(allocation_cap)?;
        checked_live_bytes([initial_live_bytes, requested], allocation_cap)?;
        let mut reader_bytes = Vec::new();
        reader_bytes
            .try_reserve_exact(requested)
            .map_err(|_| host_allocation_error(requested))?;
        let actual_bytes = element_capacity_bytes::<u8>(reader_bytes.capacity(), allocation_cap)?;
        checked_live_bytes([initial_live_bytes, actual_bytes], allocation_cap)?;
        reader_bytes.extend_from_slice(self.bytes);
        reader_bytes.extend_from_slice(suffix);
        Ok(Cow::Owned(reader_bytes))
    }

    fn missing_eoi_suffix(&self) -> &'static [u8] {
        if self.bytes.last() == Some(&0xff) {
            &[0xd9]
        } else {
            &[0xff, 0xd9]
        }
    }
}

pub(crate) fn validate_scan_bytes(
    scan_bytes: &[u8],
    allow_restart_markers: bool,
    marker_offset_base: usize,
) -> Result<ValidatedScanBytes<'_>, JpegError> {
    let mut index = 0usize;
    while index < scan_bytes.len() {
        if scan_bytes[index] != 0xff {
            index += 1;
            continue;
        }

        let marker_start = index;
        let next = index + 1;
        if next >= scan_bytes.len() {
            return Ok(ValidatedScanBytes {
                bytes: scan_bytes,
                eoi_end: None,
            });
        }

        match scan_bytes[next] {
            0x00 => index = next + 1,
            0xd0..=0xd7 if allow_restart_markers => index = next + 1,
            0xd9 => {
                return Ok(ValidatedScanBytes {
                    bytes: scan_bytes,
                    eoi_end: Some(next + 1),
                });
            }
            found => {
                let offset = marker_offset_base.checked_add(marker_start).ok_or(
                    JpegError::MemoryCapExceeded {
                        requested: usize::MAX,
                        cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
                    },
                )?;
                return Err(JpegError::UnexpectedMarker {
                    offset,
                    expected: MarkerKind::Eoi,
                    found,
                });
            }
        }
    }

    Ok(ValidatedScanBytes {
        bytes: scan_bytes,
        eoi_end: None,
    })
}

#[cfg(test)]
pub(super) fn terminated_scan_bytes(scan_bytes: &[u8]) -> Result<Cow<'_, [u8]>, JpegError> {
    terminated_scan_bytes_with_cap(scan_bytes, j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES)
}

#[cfg(test)]
pub(super) fn terminated_scan_bytes_with_cap(
    scan_bytes: &[u8],
    allocation_cap: usize,
) -> Result<Cow<'_, [u8]>, JpegError> {
    validate_scan_bytes(scan_bytes, true, 0)?.terminated_with_live_budget(0, allocation_cap)
}
