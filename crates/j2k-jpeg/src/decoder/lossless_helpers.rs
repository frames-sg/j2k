// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    downscale_profile_name, emit_jpeg_profile_fields, BitReader, ColorSpace, DownscaleFactor,
    Duration, Info, JpegError, LosslessSample, MarkerKind, PreparedLosslessPlan, ProfileField,
    Rect, RestartIndex, RestartSegment, SofKind,
};
use crate::allocation::{checked_allocation_bytes, try_vec_with_capacity};
use crate::entropy::sequential::PreparedDecodePlan;

pub(crate) fn restart_index_allocation_bytes(
    info: &Info,
    restart_interval: Option<u16>,
) -> Result<usize, JpegError> {
    let Some(interval_mcus) = restart_interval
        .filter(|&interval| interval > 0)
        .map(u32::from)
    else {
        return Ok(0);
    };
    let capacity = restart_segment_capacity(info.mcu_geometry.count, interval_mcus)?;
    checked_allocation_bytes::<RestartSegment>(capacity)
}

pub(super) fn restart_index_for_stream(
    bytes: &[u8],
    scan_data_offset: Option<usize>,
    info: &Info,
    restart_interval: Option<u16>,
) -> Result<Option<RestartIndex>, JpegError> {
    let Some(interval_mcus) = restart_interval
        .filter(|&interval| interval > 0)
        .map(u32::from)
    else {
        return Ok(None);
    };
    let scan_data_offset = scan_data_offset.ok_or(JpegError::MissingMarker {
        marker: MarkerKind::Sos,
    })?;
    if !matches!(info.sof_kind, SofKind::Baseline8 | SofKind::Extended8) || info.scan_count != 1 {
        return Err(JpegError::NotImplemented { sof: info.sof_kind });
    }
    let total_mcus = info.mcu_geometry.count;
    let expected_restarts = total_mcus.saturating_sub(1) / interval_mcus;
    let segment_capacity = restart_segment_capacity(total_mcus, interval_mcus)?;
    let mut segments = try_vec_with_capacity(segment_capacity)?;
    segments.push(RestartSegment {
        start_mcu: 0,
        entropy_offset: scan_data_offset,
        marker_offset: None,
        marker: None,
    });

    let mut found_restarts = 0u32;
    let mut expected_rst = 0xd0u8;
    let mut pos = scan_data_offset;
    while pos < bytes.len() {
        if bytes[pos] != 0xff {
            pos += 1;
            continue;
        }

        let mut marker_code_pos = pos + 1;
        while marker_code_pos < bytes.len() && bytes[marker_code_pos] == 0xff {
            marker_code_pos += 1;
        }
        if marker_code_pos >= bytes.len() {
            return Err(JpegError::Truncated {
                offset: pos,
                expected: 1,
            });
        }

        let marker = bytes[marker_code_pos];
        let marker_offset = marker_code_pos - 1;
        match marker {
            0x00 => pos = marker_code_pos + 1,
            0xd0..=0xd7 => {
                if found_restarts >= expected_restarts {
                    return Err(JpegError::UnexpectedMarker {
                        offset: marker_offset,
                        expected: MarkerKind::Eoi,
                        found: marker,
                    });
                }
                if marker != expected_rst {
                    return Err(JpegError::RestartMismatch {
                        offset: marker_offset,
                        expected: expected_rst & 0x07,
                        found: marker,
                    });
                }
                found_restarts += 1;
                segments.push(RestartSegment {
                    start_mcu: found_restarts.saturating_mul(interval_mcus),
                    entropy_offset: marker_code_pos + 1,
                    marker_offset: Some(marker_offset),
                    marker: Some(marker),
                });
                expected_rst = if expected_rst == 0xd7 {
                    0xd0
                } else {
                    expected_rst + 1
                };
                pos = marker_code_pos + 1;
            }
            0xd9 => {
                if found_restarts != expected_restarts {
                    return Err(JpegError::UnexpectedEoi {
                        mcu_at: found_restarts
                            .saturating_add(1)
                            .saturating_mul(interval_mcus),
                        mcu_total: total_mcus,
                    });
                }
                return Ok(Some(RestartIndex {
                    scan_data_offset,
                    interval_mcus,
                    segments,
                }));
            }
            found => {
                return Err(JpegError::UnexpectedMarker {
                    offset: marker_offset,
                    expected: MarkerKind::Eoi,
                    found,
                });
            }
        }
    }

    Err(JpegError::MissingMarker {
        marker: MarkerKind::Eoi,
    })
}

fn restart_segment_capacity(total_mcus: u32, interval_mcus: u32) -> Result<usize, JpegError> {
    let expected_restarts = total_mcus.saturating_sub(1) / interval_mcus;
    let capacity = usize::try_from(expected_restarts)
        .ok()
        .and_then(|count| count.checked_add(1))
        .ok_or(JpegError::MemoryCapExceeded {
            requested: usize::MAX,
            cap: j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        })?;
    checked_allocation_bytes::<RestartSegment>(capacity)?;
    Ok(capacity)
}

pub(super) fn emit_decode_scan_profile(
    scan_path: &str,
    dimensions: (u32, u32),
    decoded: Rect,
    downscale: DownscaleFactor,
    elapsed: Duration,
) {
    emit_jpeg_profile_fields("jpeg_lossless_scan_fields", "decode_scan", "cpu", || {
        Ok([
            ProfileField::label("scan_path", scan_path)?,
            ProfileField::label("downscale", downscale_profile_name(downscale))?,
            ProfileField::metric_with_summary("source_width", dimensions.0, false)?,
            ProfileField::metric_with_summary("source_height", dimensions.1, false)?,
            ProfileField::metric_with_summary("decoded_x", decoded.x, false)?,
            ProfileField::metric_with_summary("decoded_y", decoded.y, false)?,
            ProfileField::metric_with_summary("decoded_w", decoded.w, false)?,
            ProfileField::metric_with_summary("decoded_h", decoded.h, false)?,
            ProfileField::metric("scan_us", elapsed.as_micros())?,
        ])
    });
}

pub(super) fn consume_extended12_restart(
    br: &mut BitReader<'_>,
    mcu_index: u32,
    total_mcus: u32,
    expected_rst: &mut u8,
) -> Result<(), JpegError> {
    *expected_rst = br.consume_restart_marker(*expected_rst, mcu_index, total_mcus)?;
    Ok(())
}

pub(super) struct Extended12RestartTracker {
    restart_interval: u32,
    total_mcus: u32,
    mcus_since_restart: u32,
    expected_rst: u8,
}

impl Extended12RestartTracker {
    pub(super) fn new(restart_interval: Option<u16>, total_mcus: u32) -> Self {
        Self {
            restart_interval: u32::from(restart_interval.unwrap_or(0)),
            total_mcus,
            mcus_since_restart: 0,
            expected_rst: 0,
        }
    }

    pub(super) fn begin_mcu(
        &mut self,
        br: &mut BitReader<'_>,
        mcu_index: u32,
    ) -> Result<bool, JpegError> {
        if self.restart_interval > 0 && self.mcus_since_restart == self.restart_interval {
            consume_extended12_restart(br, mcu_index, self.total_mcus, &mut self.expected_rst)?;
            self.mcus_since_restart = 0;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    pub(super) fn finish_mcu(&mut self) {
        self.mcus_since_restart += 1;
    }
}

/// Scan-component index feeding each of the three output channels.
pub(super) fn lossless_output_order(
    decode_plan: &PreparedDecodePlan,
) -> Result<[usize; 3], JpegError> {
    let unsupported = || JpegError::UnsupportedComponentCount {
        count: u8::try_from(decode_plan.components.len()).unwrap_or(u8::MAX),
    };
    if decode_plan.components.len() != 3 {
        return Err(unsupported());
    }
    let mut order = [usize::MAX; 3];
    for (index, component) in decode_plan.components.iter().enumerate() {
        let slot = order
            .get_mut(component.output_index)
            .ok_or_else(unsupported)?;
        if *slot != usize::MAX {
            return Err(unsupported());
        }
        *slot = index;
    }
    Ok(order)
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "component counts are validated against the JPEG maximum before error reporting"
)]
pub(super) fn validate_lossless_color_plan<P: LosslessSample>(
    plan: &PreparedLosslessPlan,
    decode_plan: &PreparedDecodePlan,
    info: &Info,
    color_space: ColorSpace,
) -> Result<(), JpegError> {
    if !P::holds_precision(plan.bit_depth) {
        return Err(JpegError::UnsupportedBitDepth {
            depth: plan.bit_depth,
        });
    }
    if info.color_space != color_space {
        return Err(JpegError::NotImplemented { sof: info.sof_kind });
    }
    if decode_plan.components.len() != 3 {
        return Err(JpegError::UnsupportedComponentCount {
            count: decode_plan.components.len() as u8,
        });
    }
    if !(1..=7).contains(&plan.predictor) {
        return Err(JpegError::UnsupportedPredictor {
            predictor: plan.predictor,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests;
