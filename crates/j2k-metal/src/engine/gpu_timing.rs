// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use std::time::Duration;

use crate::metal_types::{CommandBuffer, CommandBufferRef};

pub(super) fn completed_command_buffers_gpu_duration(
    retained: &[CommandBuffer],
    final_buffer: &CommandBufferRef,
) -> Option<Duration> {
    completed_command_buffers_gpu_duration_and_elapsed_window(retained, final_buffer)
        .map(|(duration, _window)| duration)
}

pub(super) fn completed_command_buffers_gpu_duration_and_elapsed_window(
    retained: &[CommandBuffer],
    final_buffer: &CommandBufferRef,
) -> Option<(Duration, Duration)> {
    let mut total = Duration::ZERO;
    let mut min_start = f64::INFINITY;
    let mut max_end = f64::NEG_INFINITY;
    for (index, command_buffer) in retained.iter().enumerate() {
        if retained[..index].iter().any(|previous| {
            objc2::rc::Retained::as_ptr(previous) == objc2::rc::Retained::as_ptr(command_buffer)
        }) {
            continue;
        }
        let (start, end) = completed_command_buffer_gpu_times(command_buffer)?;
        total = total.saturating_add(Duration::from_secs_f64(end - start));
        min_start = min_start.min(start);
        max_end = max_end.max(end);
    }
    if !retained
        .iter()
        .any(|command_buffer| core::ptr::eq(command_buffer.as_ref(), final_buffer))
    {
        let (start, end) = completed_command_buffer_gpu_times(final_buffer)?;
        total = total.saturating_add(Duration::from_secs_f64(end - start));
        min_start = min_start.min(start);
        max_end = max_end.max(end);
    }
    if min_start.is_finite() && max_end.is_finite() && max_end > min_start {
        Some((total, Duration::from_secs_f64(max_end - min_start)))
    } else {
        None
    }
}

pub(super) fn completed_command_buffer_gpu_duration(
    command_buffer: &CommandBufferRef,
) -> Option<Duration> {
    let (start, end) = completed_command_buffer_gpu_times(command_buffer)?;
    Some(Duration::from_secs_f64(end - start))
}

fn completed_command_buffer_gpu_times(command_buffer: &CommandBufferRef) -> Option<(f64, f64)> {
    #[cfg(test)]
    super::test_counters::record_resident_gpu_timestamp_query();

    let start = command_buffer.GPUStartTime();
    let end = command_buffer.GPUEndTime();
    if start.is_finite() && end.is_finite() && end > start {
        Some((start, end))
    } else {
        None
    }
}
