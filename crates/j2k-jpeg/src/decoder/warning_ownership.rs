// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fallible ownership transfer for parsed and entropy-scan warnings.

use alloc::vec::Vec;

use crate::allocation::{
    checked_add_allocation_bytes, checked_allocation_bytes, try_reserve_for_len_with_live_budget,
};
use crate::error::{JpegError, Warning};

use super::MAX_DECODE_SCAN_WARNINGS;

pub(super) fn merged_warnings(
    header_warnings: &Vec<Warning>,
    scan_warnings: Vec<Warning>,
) -> Result<Vec<Warning>, JpegError> {
    if header_warnings.is_empty() {
        return Ok(scan_warnings);
    }
    if scan_warnings.is_empty() {
        return try_clone_warnings(header_warnings);
    }

    let warning_count = header_warnings
        .len()
        .checked_add(scan_warnings.len())
        .ok_or_else(cap_overflow)?;
    let mut live_bytes = checked_add_allocation_bytes(
        warning_capacity_bytes(header_warnings.capacity())?,
        warning_capacity_bytes(scan_warnings.capacity())?,
    )?;
    let cap = warning_merge_peak_bytes(header_warnings.capacity())?;
    ensure_warning_capacity_peak(header_warnings.capacity(), scan_warnings.capacity(), 0)?;

    let mut warnings = Vec::new();
    try_reserve_for_len_with_live_budget(&mut warnings, warning_count, &mut live_bytes, cap)?;
    ensure_warning_capacity_peak(
        header_warnings.capacity(),
        scan_warnings.capacity(),
        warnings.capacity(),
    )?;
    warnings.extend_from_slice(header_warnings);
    warnings.extend(scan_warnings);
    Ok(warnings)
}

pub(super) fn try_clone_warnings(warnings: &Vec<Warning>) -> Result<Vec<Warning>, JpegError> {
    let mut live_bytes = warning_capacity_bytes(warnings.capacity())?;
    let cap = warning_merge_peak_bytes(warnings.capacity())?;
    let mut cloned = Vec::new();
    try_reserve_for_len_with_live_budget(&mut cloned, warnings.len(), &mut live_bytes, cap)?;
    ensure_warning_capacity_peak(warnings.capacity(), 0, cloned.capacity())?;
    cloned.extend_from_slice(warnings);
    Ok(cloned)
}

pub(super) fn merged_warning_capacity_bytes(header_capacity: usize) -> Result<usize, JpegError> {
    let capacity = header_capacity
        .checked_add(MAX_DECODE_SCAN_WARNINGS)
        .ok_or_else(cap_overflow)?;
    warning_capacity_bytes(capacity)
}

pub(super) fn warning_merge_peak_bytes(header_capacity: usize) -> Result<usize, JpegError> {
    let header_bytes = warning_capacity_bytes(header_capacity)?;
    let scan_bytes = warning_capacity_bytes(MAX_DECODE_SCAN_WARNINGS)?;
    let output_bytes = merged_warning_capacity_bytes(header_capacity)?;
    let retained = checked_add_allocation_bytes(header_bytes, scan_bytes)?;
    checked_add_allocation_bytes(retained, output_bytes)
}

fn ensure_warning_capacity_peak(
    header_capacity: usize,
    scan_capacity: usize,
    output_capacity: usize,
) -> Result<(), JpegError> {
    let retained = checked_add_allocation_bytes(
        warning_capacity_bytes(header_capacity)?,
        warning_capacity_bytes(scan_capacity)?,
    )?;
    let requested =
        checked_add_allocation_bytes(retained, warning_capacity_bytes(output_capacity)?)?;
    let cap = warning_merge_peak_bytes(header_capacity)?;
    if requested > cap {
        return Err(JpegError::MemoryCapExceeded { requested, cap });
    }
    Ok(())
}

fn warning_capacity_bytes(capacity: usize) -> Result<usize, JpegError> {
    checked_allocation_bytes::<Warning>(capacity)
}

fn cap_overflow() -> JpegError {
    JpegError::MemoryCapExceeded {
        requested: usize::MAX,
        cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ensure_warning_capacity_peak, merged_warnings, warning_merge_peak_bytes, JpegError,
        Warning, MAX_DECODE_SCAN_WARNINGS,
    };
    use core::mem::size_of;

    #[test]
    fn warning_capacity_peak_accepts_exact_and_rejects_one_over() {
        let header_capacity = 4;
        let exact_output_capacity = header_capacity + MAX_DECODE_SCAN_WARNINGS;
        let exact_bytes =
            (header_capacity * 2 + MAX_DECODE_SCAN_WARNINGS * 2) * size_of::<Warning>();
        assert_eq!(
            warning_merge_peak_bytes(header_capacity).unwrap(),
            exact_bytes
        );
        ensure_warning_capacity_peak(
            header_capacity,
            MAX_DECODE_SCAN_WARNINGS,
            exact_output_capacity,
        )
        .expect("exact warning peak");
        assert!(matches!(
            ensure_warning_capacity_peak(
                header_capacity,
                MAX_DECODE_SCAN_WARNINGS,
                exact_output_capacity + 1,
            ),
            Err(JpegError::MemoryCapExceeded { requested, cap })
                if requested == cap + size_of::<Warning>()
        ));
    }

    #[test]
    fn warning_merge_preserves_values_under_the_shared_capacity() {
        let mut header = Vec::with_capacity(4);
        header.push(Warning::MissingEoi);
        let mut scan = Vec::with_capacity(MAX_DECODE_SCAN_WARNINGS);
        scan.push(Warning::MissingEoi);

        let merged = merged_warnings(&header, scan).expect("merge warnings");
        assert_eq!(merged, [Warning::MissingEoi, Warning::MissingEoi]);
        ensure_warning_capacity_peak(header.capacity(), 0, merged.capacity())
            .expect("merged owner remains under the planned peak");
    }
}
