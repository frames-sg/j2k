// SPDX-License-Identifier: MIT OR Apache-2.0

//! Progressive scan entropy traversal and coefficient refinement.

use alloc::vec::Vec;

use crate::allocation::try_reserve_for_len_with_live_budget;
use crate::entropy::huffman::{AcHuffmanTable, DcHuffmanTable};
use crate::entropy::ZIGZAG;
use crate::error::{HuffmanFailure, JpegError};
use crate::internal::bit_reader::BitReader;

use super::allocation::{
    allocate_coefficients, allocate_nonzero_masks, checked_phase_capacity,
    coefficient_capacity_bytes,
};
use super::model::{
    PreparedProgressiveComponentPlan, PreparedProgressivePlan, PreparedProgressiveScan,
    PreparedProgressiveScanComponent, ProgressiveDctBlocks,
};
use super::terminal::finish_progressive_scan;

struct ProgressiveBlockTarget<'a, 't> {
    component: &'a PreparedProgressiveComponentPlan,
    scan_component: &'a PreparedProgressiveScanComponent,
    tables: ScanTables<'t>,
    block_x: u32,
    block_y: u32,
}

/// Huffman tables one scan component uses, resolved once per scan: DC first
/// scans need the DC table, AC scans the AC table, DC refinement neither.
#[derive(Clone, Copy, Default)]
struct ScanTables<'t> {
    dc: Option<DcHuffmanTable<'t>>,
    ac: Option<AcHuffmanTable<'t>>,
}

/// Largest component count of one JPEG scan (T.81 B.2.3).
const MAX_SCAN_COMPONENTS: usize = 4;

fn resolve_scan_tables<'t>(
    plan: &'t PreparedProgressivePlan,
    scan: &PreparedProgressiveScan,
    scan_components: &[PreparedProgressiveScanComponent],
) -> Result<[ScanTables<'t>; MAX_SCAN_COMPONENTS], JpegError> {
    let mut tables = [ScanTables::default(); MAX_SCAN_COMPONENTS];
    if scan_components.len() > MAX_SCAN_COMPONENTS {
        return Err(JpegError::InternalInvariant {
            reason: "prepared progressive scan has more than four components",
        });
    }
    for (slot, scan_component) in tables.iter_mut().zip(scan_components) {
        if scan.ss != 0 {
            slot.ac = Some(plan.ac_table(scan_component.ac_table)?);
        } else if scan.ah == 0 {
            slot.dc = Some(plan.dc_table(scan_component.dc_table)?);
        }
    }
    Ok(tables)
}

fn missing_table() -> JpegError {
    JpegError::InternalInvariant {
        reason: "progressive scan references a missing prepared Huffman table",
    }
}

pub(crate) fn decode_progressive_dct_blocks(
    plan: &PreparedProgressivePlan,
    bytes: &[u8],
    external_live_bytes: usize,
) -> Result<ProgressiveDctBlocks, JpegError> {
    let mut coeffs = allocate_coefficients(plan, external_live_bytes)?;
    let coefficient_live_bytes = checked_phase_capacity(
        external_live_bytes,
        coefficient_capacity_bytes(coeffs.capacity(), &coeffs)?,
        plan.scratch_bytes,
    )?;
    let (mut nonzero, scan_live_bytes) =
        allocate_nonzero_masks(plan, &coeffs, coefficient_live_bytes)?;
    for scan in &plan.scans {
        let mut state = CoefficientState {
            blocks: &mut coeffs,
            nonzero: &mut nonzero,
        };
        decode_progressive_scan(plan, scan, bytes, &mut state, scan_live_bytes)?;
    }
    Ok(ProgressiveDctBlocks { quantized: coeffs })
}

/// Coefficient blocks, plus a mask per block whose bit `k` is set when the
/// coefficient at zigzag position `k` (1..=63) is nonzero. The masks let AC
/// refinement visit only nonzero coefficients.
struct CoefficientState<'a> {
    blocks: &'a mut [Vec<[i32; 64]>],
    nonzero: &'a mut [Vec<u64>],
}

fn decode_progressive_scan(
    plan: &PreparedProgressivePlan,
    scan: &PreparedProgressiveScan,
    bytes: &[u8],
    coeffs: &mut CoefficientState<'_>,
    scan_live_bytes: usize,
) -> Result<(), JpegError> {
    let scan_bytes = bytes
        .get(scan.entropy_offset..)
        .ok_or(JpegError::Truncated {
            offset: scan.entropy_offset,
            expected: 1,
        })?;
    let mut br = BitReader::new_with_eof_padding(scan_bytes, scan.terminal_code == 0);
    let mut live_bytes = scan_live_bytes;
    let mut dc_predictors = Vec::new();
    try_reserve_for_len_with_live_budget(
        &mut dc_predictors,
        plan.components.len(),
        &mut live_bytes,
        plan.scratch_bytes,
    )?;
    dc_predictors.resize(plan.components.len(), 0i32);
    let mut eob_run = 0u32;
    let restart = u32::from(scan.restart_interval.unwrap_or(0));
    let total_mcus = scan_mcu_count(plan, scan)?;
    let scan_components = plan.scan_components(scan)?;
    let tables = if total_mcus == 0 {
        [ScanTables::default(); MAX_SCAN_COMPONENTS]
    } else {
        resolve_scan_tables(plan, scan, scan_components)?
    };
    let context = ScanContext {
        plan,
        scan,
        scan_components,
        tables,
    };

    let state = (&mut dc_predictors[..], &mut eob_run);
    let counts = (restart, total_mcus);
    match (scan.ss == 0, scan.ah == 0) {
        (true, true) => decode_scan_mcus::<DC_FIRST>(&context, &mut br, coeffs, state, counts),
        (true, false) => decode_scan_mcus::<DC_REFINE>(&context, &mut br, coeffs, state, counts),
        (false, true) => decode_scan_mcus::<AC_FIRST>(&context, &mut br, coeffs, state, counts),
        (false, false) => decode_scan_mcus::<AC_REFINE>(&context, &mut br, coeffs, state, counts),
    }?;

    finish_progressive_scan(&mut br, scan_bytes, scan, eob_run)
}

/// Scan kinds (T.81 G.1.2): `Ss == 0` selects DC, `Ah == 0` a first pass.
/// Each kind gets its own monomorphized MCU loop, so a loop only carries
/// the one block decoder it needs.
const DC_FIRST: u8 = 0;
const DC_REFINE: u8 = 1;
const AC_FIRST: u8 = 2;
const AC_REFINE: u8 = 3;

/// Decode every MCU of one scan of kind `KIND`. Kept out of line per kind:
/// its callee chain is inlined so the reader stays in registers.
fn decode_scan_mcus<const KIND: u8>(
    context: &ScanContext<'_>,
    br: &mut BitReader<'_>,
    coeffs: &mut CoefficientState<'_>,
    (dc_predictors, eob_run): (&mut [i32], &mut u32),
    (restart, total_mcus): (u32, u32),
) -> Result<(), JpegError> {
    // Decode through a local copy so the accumulator stays in registers; it
    // is written back even on error.
    let mut local = br.clone();
    let decoded = decode_scan_mcus_local::<KIND>(
        context,
        &mut local,
        coeffs,
        (dc_predictors, eob_run),
        (restart, total_mcus),
    );
    *br = local;
    decoded
}

#[expect(
    clippy::inline_always,
    reason = "inlining keeps the caller's local bit reader in registers for the whole scan"
)]
#[inline(always)]
fn decode_scan_mcus_local<const KIND: u8>(
    context: &ScanContext<'_>,
    br: &mut BitReader<'_>,
    coeffs: &mut CoefficientState<'_>,
    (dc_predictors, eob_run): (&mut [i32], &mut u32),
    (restart, total_mcus): (u32, u32),
) -> Result<(), JpegError> {
    if let [scan_component] = context.scan_components {
        return decode_single_component_scan::<KIND>(
            context,
            scan_component,
            br,
            coeffs,
            (dc_predictors, eob_run),
            restart,
        );
    }
    let mut mcus_since_restart = 0u32;
    let mut expected_rst = 0u8;
    for mcu_index in 0..total_mcus {
        if restart > 0 && mcus_since_restart == restart {
            let (restarted, next_rst) =
                br.clone().restarted(expected_rst, mcu_index, total_mcus)?;
            *br = restarted;
            expected_rst = next_rst;
            dc_predictors.fill(0);
            *eob_run = 0;
            mcus_since_restart = 0;
        }
        decode_progressive_mcu::<KIND>(context, br, coeffs, dc_predictors, eob_run, mcu_index)?;
        mcus_since_restart += 1;
    }
    Ok(())
}

/// Per-scan state shared by every block of the scan.
struct ScanContext<'p> {
    plan: &'p PreparedProgressivePlan,
    scan: &'p PreparedProgressiveScan,
    scan_components: &'p [PreparedProgressiveScanComponent],
    tables: [ScanTables<'p>; MAX_SCAN_COMPONENTS],
}

fn scan_mcu_count(
    plan: &PreparedProgressivePlan,
    scan: &PreparedProgressiveScan,
) -> Result<u32, JpegError> {
    let scan_components = plan.scan_components(scan)?;
    if scan_components.len() > 1 {
        Ok(plan.mcu_cols.saturating_mul(plan.mcu_rows))
    } else {
        let scan_component = scan_components
            .first()
            .ok_or(JpegError::InternalInvariant {
                reason: "prepared progressive scan has no components",
            })?;
        let component = plan.components.get(scan_component.component_index).ok_or(
            JpegError::InternalInvariant {
                reason: "prepared progressive scan references an unknown component",
            },
        )?;
        Ok(progressive_coded_block_cols(component)
            .saturating_mul(progressive_coded_block_rows(component)))
    }
}

/// A non-interleaved scan, whose MCU is one block: walk the coded blocks row
/// by row (the order of `mcu_index`) without per-block index arithmetic.
#[expect(
    clippy::inline_always,
    reason = "inlining keeps the caller's local bit reader in registers for the whole scan"
)]
#[inline(always)]
fn decode_single_component_scan<const KIND: u8>(
    context: &ScanContext<'_>,
    scan_component: &PreparedProgressiveScanComponent,
    br: &mut BitReader<'_>,
    coeffs: &mut CoefficientState<'_>,
    (dc_predictors, eob_run): (&mut [i32], &mut u32),
    restart: u32,
) -> Result<(), JpegError> {
    let component = context
        .plan
        .components
        .get(scan_component.component_index)
        .ok_or(JpegError::InternalInvariant {
            reason: "prepared progressive scan references an unknown component",
        })?;
    let (coded_cols, rows) = (
        progressive_coded_block_cols(component),
        progressive_coded_block_rows(component),
    );
    let total_mcus = coded_cols.saturating_mul(rows);
    let cols = coded_cols as usize;
    let block_cols = component.block_cols as usize;
    let (Some(component_coeffs), Some(component_masks)) = (
        coeffs.blocks.get_mut(scan_component.component_index),
        coeffs.nonzero.get_mut(scan_component.component_index),
    ) else {
        return Err(invalid_symbol());
    };
    let tables = context.tables[0];
    let mut mcus_since_restart = 0u32;
    let mut expected_rst = 0u8;
    let mut mcu_index = 0u32;
    for by in 0..rows as usize {
        let row_start = by * block_cols;
        let (Some(row), Some(masks)) = (
            component_coeffs.get_mut(row_start..row_start + cols),
            component_masks.get_mut(row_start..row_start + cols),
        ) else {
            return Err(invalid_symbol());
        };
        for (block, nonzero) in row.iter_mut().zip(masks) {
            if restart > 0 && mcus_since_restart == restart {
                let (restarted, next_rst) =
                    br.clone().restarted(expected_rst, mcu_index, total_mcus)?;
                *br = restarted;
                expected_rst = next_rst;
                dc_predictors.fill(0);
                *eob_run = 0;
                mcus_since_restart = 0;
            }
            if KIND == DC_FIRST || KIND == AC_FIRST {
                decode_progressive_block_first::<KIND>(
                    context.scan,
                    tables,
                    br,
                    (block, nonzero),
                    &mut dc_predictors[scan_component.component_index],
                    eob_run,
                )?;
            } else {
                decode_progressive_block_refine::<KIND>(
                    context.scan,
                    tables,
                    br,
                    (block, nonzero),
                    eob_run,
                )?;
            }
            mcus_since_restart += 1;
            mcu_index += 1;
        }
    }
    Ok(())
}

#[expect(
    clippy::inline_always,
    reason = "inlining keeps the caller's local bit reader in registers for the whole scan"
)]
#[inline(always)]
fn decode_progressive_mcu<const KIND: u8>(
    context: &ScanContext<'_>,
    br: &mut BitReader<'_>,
    coeffs: &mut CoefficientState<'_>,
    dc_predictors: &mut [i32],
    eob_run: &mut u32,
    mcu_index: u32,
) -> Result<(), JpegError> {
    let plan = context.plan;
    if context.scan_components.len() > 1 {
        let mcu_x = mcu_index % plan.mcu_cols;
        let mcu_y = mcu_index / plan.mcu_cols;
        for (scan_component, &tables) in context.scan_components.iter().zip(&context.tables) {
            let component = plan.components.get(scan_component.component_index).ok_or(
                JpegError::InternalInvariant {
                    reason: "prepared progressive scan references an unknown component",
                },
            )?;
            for by in 0..u32::from(component.v) {
                for bx in 0..u32::from(component.h) {
                    let target = ProgressiveBlockTarget {
                        component,
                        scan_component,
                        tables,
                        block_x: mcu_x * u32::from(component.h) + bx,
                        block_y: mcu_y * u32::from(component.v) + by,
                    };
                    decode_progressive_block_at::<KIND>(
                        context.scan,
                        &target,
                        br,
                        coeffs,
                        dc_predictors,
                        eob_run,
                    )?;
                }
            }
        }
    } else {
        let scan_component =
            context
                .scan_components
                .first()
                .ok_or(JpegError::InternalInvariant {
                    reason: "prepared progressive scan has no components",
                })?;
        let component = plan.components.get(scan_component.component_index).ok_or(
            JpegError::InternalInvariant {
                reason: "prepared progressive scan references an unknown component",
            },
        )?;
        let coded_cols = progressive_coded_block_cols(component);
        let target = ProgressiveBlockTarget {
            component,
            scan_component,
            tables: context.tables[0],
            block_x: mcu_index % coded_cols,
            block_y: mcu_index / coded_cols,
        };
        decode_progressive_block_at::<KIND>(
            context.scan,
            &target,
            br,
            coeffs,
            dc_predictors,
            eob_run,
        )?;
    }

    Ok(())
}

#[expect(
    clippy::inline_always,
    reason = "inlining keeps the caller's local bit reader in registers for the whole scan"
)]
#[inline(always)]
fn decode_progressive_block_at<const KIND: u8>(
    scan: &PreparedProgressiveScan,
    target: &ProgressiveBlockTarget<'_, '_>,
    br: &mut BitReader<'_>,
    coeffs: &mut CoefficientState<'_>,
    dc_predictors: &mut [i32],
    eob_run: &mut u32,
) -> Result<(), JpegError> {
    let block_index = (target.block_y as usize)
        .checked_mul(target.component.block_cols as usize)
        .and_then(|base| base.checked_add(target.block_x as usize))
        .ok_or(JpegError::HuffmanDecode {
            mcu: 0,
            reason: HuffmanFailure::InvalidSymbol,
        })?;
    let component = target.scan_component.component_index;
    let (Some(block), Some(nonzero)) = (
        coeffs
            .blocks
            .get_mut(component)
            .and_then(|blocks| blocks.get_mut(block_index)),
        coeffs
            .nonzero
            .get_mut(component)
            .and_then(|masks| masks.get_mut(block_index)),
    ) else {
        return Err(invalid_symbol());
    };

    if KIND == DC_FIRST || KIND == AC_FIRST {
        decode_progressive_block_first::<KIND>(
            scan,
            target.tables,
            br,
            (block, nonzero),
            &mut dc_predictors[component],
            eob_run,
        )
    } else {
        decode_progressive_block_refine::<KIND>(scan, target.tables, br, (block, nonzero), eob_run)
    }
}

#[expect(
    clippy::inline_always,
    reason = "inlining keeps the caller's local bit reader in registers for the whole scan"
)]
#[inline(always)]
fn decode_progressive_block_first<const KIND: u8>(
    scan: &PreparedProgressiveScan,
    tables: ScanTables<'_>,
    br: &mut BitReader<'_>,
    (block, nonzero): (&mut [i32; 64], &mut u64),
    dc_predictor: &mut i32,
    eob_run: &mut u32,
) -> Result<(), JpegError> {
    if KIND == DC_FIRST {
        let dc_table = tables.dc.ok_or_else(missing_table)?;
        let ssss = dc_table.decode(br)?;
        if ssss > 15 {
            return Err(invalid_symbol());
        }
        let diff = br.receive_extend(ssss)?;
        *dc_predictor = dc_predictor.wrapping_add(diff);
        block[0] = dc_predictor.wrapping_shl(u32::from(scan.al));
        return Ok(());
    }

    let ac_table = tables.ac.ok_or_else(missing_table)?;
    if *eob_run > 0 {
        *eob_run -= 1;
        return Ok(());
    }

    // `usize` band positions: `k <= se <= 63` wherever a coefficient is
    // stored, so the mask only drops a bounds check.
    let (mut k, se) = (usize::from(scan.ss), usize::from(scan.se));
    while k <= se {
        let symbol = ac_table.decode(br)?;
        let run = symbol >> 4;
        let ssss = symbol & 0x0F;
        if ssss == 0 {
            if run == 15 {
                k += 16;
            } else {
                *eob_run = decode_eob_run(br, run)?;
                break;
            }
        } else {
            k += usize::from(run);
            if k > se {
                return Err(invalid_symbol());
            }
            let value = br.receive_extend(ssss)?.wrapping_shl(u32::from(scan.al));
            block[usize::from(ZIGZAG[k & 63]) & 63] = value;
            // `receive_extend` values are nonzero for SSSS >= 1.
            *nonzero |= 1 << (k & 63);
            k += 1;
        }
    }

    Ok(())
}

fn progressive_coded_block_cols(component: &PreparedProgressiveComponentPlan) -> u32 {
    component
        .sample_width
        .div_ceil(8)
        .max(1)
        .min(component.block_cols)
}

fn progressive_coded_block_rows(component: &PreparedProgressiveComponentPlan) -> u32 {
    component
        .sample_height
        .div_ceil(8)
        .max(1)
        .min(component.block_rows)
}

#[expect(
    clippy::inline_always,
    reason = "inlining keeps the caller's local bit reader in registers for the whole scan"
)]
#[inline(always)]
fn decode_progressive_block_refine<const KIND: u8>(
    scan: &PreparedProgressiveScan,
    tables: ScanTables<'_>,
    br: &mut BitReader<'_>,
    (block, nonzero): (&mut [i32; 64], &mut u64),
    eob_run: &mut u32,
) -> Result<(), JpegError> {
    let bit = 1i32 << scan.al;
    if KIND == DC_REFINE {
        if br.read_bits(1)? != 0 {
            block[0] |= bit;
        }
        return Ok(());
    }

    let ac_table = tables.ac.ok_or_else(missing_table)?;
    // Correction bits are read with deferred top-ups; every path finishes
    // them before the next Huffman decode or before returning.
    if *eob_run > 0 {
        *eob_run -= 1;
        let refined = refine_band(br, block, *nonzero, scan.ss, scan.se, 64, bit);
        br.finish_deferred_bits();
        refined?;
        return Ok(());
    }

    let mut k = scan.ss;
    while k <= scan.se {
        let symbol = ac_table.decode(br)?;
        let run = symbol >> 4;
        let ssss = symbol & 0x0F;
        let mut zero_run_length = usize::from(run);
        let mut value = 0i32;

        match ssss {
            0 => {
                if run == 15 {
                    zero_run_length = 15;
                } else {
                    *eob_run = decode_eob_run(br, run)?;
                    zero_run_length = 64;
                }
            }
            1 => {
                let positive = br.read_bit_deferred();
                if positive.is_err() {
                    br.finish_deferred_bits();
                }
                value = if positive? { bit } else { -bit };
            }
            _ => return Err(invalid_symbol()),
        }

        let refined = refine_band(br, block, *nonzero, k, scan.se, zero_run_length, bit);
        br.finish_deferred_bits();
        k = refined?;
        if value != 0 {
            if k > scan.se {
                return Err(invalid_symbol());
            }
            block[usize::from(ZIGZAG[usize::from(k)])] = value;
            *nonzero |= 1 << (k & 63);
        }
        k += 1;
    }

    Ok(())
}

#[expect(
    clippy::inline_always,
    reason = "inlining keeps the caller's local bit reader in registers for the whole scan"
)]
#[inline(always)]
pub(super) fn decode_eob_run(br: &mut BitReader<'_>, run_bits: u8) -> Result<u32, JpegError> {
    let mut eob_run = (1u32 << run_bits) - 1;
    if run_bits > 0 {
        eob_run += br.read_bits(run_bits)?;
    }
    Ok(eob_run)
}

/// Coefficient-by-coefficient refinement (T.81 G.1.2.3), kept as the
/// reference that [`refine_band`] is tested against.
#[cfg(test)]
pub(super) fn refine_non_zeroes(
    br: &mut BitReader<'_>,
    block: &mut [i32; 64],
    start: u8,
    end: u8,
    mut zero_run_length: usize,
    bit: i32,
) -> Result<u8, JpegError> {
    // Reads correction bits deferred; the caller finishes them. `ZIGZAG`
    // entries are below 64, so the mask only drops a bounds check.
    let band = ZIGZAG
        .get(usize::from(start)..=usize::from(end))
        .unwrap_or(&[]);
    for (offset, &natural) in band.iter().enumerate() {
        let coeff = &mut block[usize::from(natural) & 63];
        if *coeff == 0 {
            if zero_run_length == 0 {
                return Ok(start + band_offset(offset));
            }
            zero_run_length -= 1;
        } else if br.read_bit_deferred()? && (*coeff & bit) == 0 {
            if *coeff > 0 {
                *coeff = coeff.wrapping_add(bit);
            } else {
                *coeff = coeff.wrapping_sub(bit);
            }
        }
    }
    Ok(end)
}

/// Refine a band (T.81 G.1.2.3) driven by the block's nonzero mask, like the
/// coefficient-by-coefficient `refine_non_zeroes` test oracle: it visits only
/// nonzero coefficients, and finds the zero that ends the run with bit
/// arithmetic. Returns the same position and reads the same correction bits,
/// in the same order.
#[expect(
    clippy::inline_always,
    reason = "inlining keeps the caller's local bit reader in registers for the whole scan"
)]
#[inline(always)]
pub(super) fn refine_band(
    br: &mut BitReader<'_>,
    block: &mut [i32; 64],
    nonzero: u64,
    start: u8,
    end: u8,
    zero_run_length: usize,
    bit: i32,
) -> Result<u8, JpegError> {
    let (first, last) = (usize::from(start), usize::from(end));
    if first > last || last > 63 {
        return Ok(end);
    }
    let band = (u64::MAX >> (63 - last)) & (u64::MAX << first);
    // The run ends at the zero that follows `zero_run_length` skipped zeros.
    let mut zeros = !nonzero & band;
    let stop = if zero_run_length < zeros.count_ones() as usize {
        for _ in 0..zero_run_length {
            zeros &= zeros - 1;
        }
        Some(zeros.trailing_zeros())
    } else {
        None
    };
    let mut pending = nonzero
        & match stop {
            Some(position) => band & !(u64::MAX << position),
            None => band,
        };
    while pending != 0 {
        let k = pending.trailing_zeros() as usize;
        pending &= pending - 1;
        let coeff = &mut block[usize::from(ZIGZAG[k]) & 63];
        if br.read_bit_deferred()? && (*coeff & bit) == 0 {
            if *coeff > 0 {
                *coeff = coeff.wrapping_add(bit);
            } else {
                *coeff = coeff.wrapping_sub(bit);
            }
        }
    }
    Ok(stop.map_or(end, |position| band_offset(position as usize)))
}

/// Offset within a spectral band, which has at most 64 coefficients.
#[expect(
    clippy::cast_possible_truncation,
    reason = "band offsets index a 64-entry zigzag table"
)]
fn band_offset(offset: usize) -> u8 {
    offset as u8
}

fn invalid_symbol() -> JpegError {
    JpegError::HuffmanDecode {
        mcu: 0,
        reason: HuffmanFailure::InvalidSymbol,
    }
}
