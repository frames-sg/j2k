// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::adapter::{
    assemble_jpeg_baseline_frame, baseline_encode_tables, checked_encode_host_live_bytes,
    validate_jpeg_baseline_dimensions, validate_jpeg_baseline_restart_interval,
};
use crate::encoded_output::checked_jpeg_baseline_frame_capacity;
use crate::profile::jpeg_profile_stages_enabled;
use std::time::Instant;

mod api;
mod entropy;
mod planning;
mod profile;
mod sample_planes;
mod transform;

pub use self::api::{
    EncodedJpeg, JpegBackend, JpegEncodeError, JpegEncodeOptions, JpegSamples, JpegSubsampling,
};
use self::entropy::encode_entropy;
use self::planning::{checked_cpu_encode_capacity_plan, component_plane_capacity_bytes};
use self::profile::{emit_cpu_encode_profile, JpegEncodeProfile};
use self::sample_planes::{component_planes, validate_sample_layout};
use self::transform::cosine_table;

/// Encode grayscale or RGB samples as a baseline JPEG codestream.
///
/// # Errors
///
/// Returns an error for invalid dimensions, sample layout, quality, restart
/// configuration, or an unavailable explicitly requested backend.
pub fn encode_jpeg_baseline(
    samples: JpegSamples<'_>,
    options: JpegEncodeOptions,
) -> Result<EncodedJpeg, JpegEncodeError> {
    match options.backend {
        JpegBackend::Auto | JpegBackend::Cpu => encode_jpeg_baseline_cpu(samples, options),
        JpegBackend::Metal | JpegBackend::Cuda => Err(JpegEncodeError::UnsupportedBackend {
            backend: options.backend,
        }),
    }
}

fn encode_jpeg_baseline_cpu(
    samples: JpegSamples<'_>,
    options: JpegEncodeOptions,
) -> Result<EncodedJpeg, JpegEncodeError> {
    let profile_enabled = jpeg_profile_stages_enabled();
    let total_start = profile_enabled.then(Instant::now);
    let mut profile = JpegEncodeProfile::default();

    let validation_start = profile_enabled.then(Instant::now);
    validate_jpeg_baseline_restart_interval(options.restart_interval)?;
    let (width, height) = samples.dimensions();
    let sample_format = samples.name();
    validate_jpeg_baseline_dimensions(width, height)?;
    let expected_sample_len = validate_sample_layout(samples, options.subsampling)?;
    if let Some(start) = validation_start {
        profile.validation = start.elapsed();
    }

    let setup_start = profile_enabled.then(Instant::now);
    let tables = baseline_encode_tables(options)?;
    let sampling = tables.sampling;
    let capacity_plan = checked_cpu_encode_capacity_plan(
        samples,
        sampling,
        expected_sample_len,
        options.restart_interval,
    )?;
    if samples.data_len() != expected_sample_len {
        return Err(JpegEncodeError::SampleLength {
            expected: expected_sample_len,
            actual: samples.data_len(),
        });
    }
    let cosine = cosine_table();
    if let Some(start) = setup_start {
        profile.setup = start.elapsed();
    }

    let planes_start = profile_enabled.then(Instant::now);
    let planes = component_planes(
        samples,
        options.subsampling,
        capacity_plan.plane_capacity_limit,
    )?;
    let plane_live_bytes = component_plane_capacity_bytes(planes.capacity(), &planes)?;
    checked_encode_host_live_bytes([plane_live_bytes, capacity_plan.entropy_workspace_bytes])?;
    if let Some(start) = planes_start {
        profile.planes = start.elapsed();
    }

    let entropy_start = profile_enabled.then(Instant::now);
    let entropy = encode_entropy(
        &planes,
        width,
        height,
        sampling,
        &tables.q_luma,
        &tables.q_chroma,
        [&tables.huff_dc_luma, &tables.huff_dc_chroma],
        [&tables.huff_ac_luma, &tables.huff_ac_chroma],
        &cosine,
        options.restart_interval,
        capacity_plan.entropy_capacity,
        plane_live_bytes,
    )?;
    if let Some(start) = entropy_start {
        profile.entropy = start.elapsed();
    }

    let header_start = profile_enabled.then(Instant::now);
    let frame_capacity = checked_jpeg_baseline_frame_capacity(entropy.len())?;
    checked_encode_host_live_bytes([plane_live_bytes, entropy.capacity(), frame_capacity])?;
    let encoded =
        assemble_jpeg_baseline_frame(&entropy, width, height, &tables, options, JpegBackend::Cpu)?;
    checked_encode_host_live_bytes([
        plane_live_bytes,
        entropy.capacity(),
        encoded.data.capacity(),
    ])?;
    if let Some(start) = header_start {
        profile.header = start.elapsed();
    }
    drop(entropy);
    drop(planes);

    if let Some(start) = total_start {
        emit_cpu_encode_profile(
            start,
            &profile,
            (width, height),
            sample_format,
            options,
            sampling,
            &encoded,
        );
    }

    Ok(encoded)
}

#[cfg(test)]
mod tests;
