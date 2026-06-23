// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::Duration;

use metal::{
    foreign_types::{ForeignType, ForeignTypeRef},
    objc::{runtime::Sel, Message},
    CommandBuffer, CommandBufferRef,
};

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
    let mut seen = Vec::with_capacity(retained.len().saturating_add(1));
    for command_buffer in retained {
        let ptr = command_buffer.as_ptr();
        if seen.contains(&ptr) {
            continue;
        }
        seen.push(ptr);
        let (start, end) = completed_command_buffer_gpu_times(command_buffer)?;
        total = total.saturating_add(Duration::from_secs_f64(end - start));
        min_start = min_start.min(start);
        max_end = max_end.max(end);
    }
    let final_ptr = final_buffer.as_ptr();
    if !seen.contains(&final_ptr) {
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

    // SAFETY: Objective-C timestamp access is queried after command-buffer completion.
    let start: f64 = unsafe {
        command_buffer
            .send_message::<(), f64>(Sel::register("GPUStartTime"), ())
            .ok()?
    };
    // SAFETY: Objective-C timestamp access is queried after command-buffer completion.
    let end: f64 = unsafe {
        command_buffer
            .send_message::<(), f64>(Sel::register("GPUEndTime"), ())
            .ok()?
    };
    if start.is_finite() && end.is_finite() && end > start {
        Some((start, end))
    } else {
        None
    }
}
