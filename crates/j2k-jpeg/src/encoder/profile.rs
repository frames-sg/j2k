// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::{Duration, Instant};

use crate::adapter::JpegBaselineSampling;
use crate::profile::{emit_jpeg_profile_fields, ProfileField};

use super::{EncodedJpeg, JpegEncodeOptions};

#[derive(Default)]
pub(super) struct JpegEncodeProfile {
    pub(super) validation: Duration,
    pub(super) setup: Duration,
    pub(super) planes: Duration,
    pub(super) header: Duration,
    pub(super) entropy: Duration,
}

pub(super) fn emit_cpu_encode_profile(
    start: Instant,
    profile: &JpegEncodeProfile,
    (width, height): (u32, u32),
    sample_format: &str,
    options: JpegEncodeOptions,
    sampling: JpegBaselineSampling,
    encoded: &EncodedJpeg,
) {
    emit_jpeg_profile_fields("jpeg_cpu_encode_fields", "encode", "cpu", || {
        Ok([
            ProfileField::metric_with_summary("sample", sample_format, false)?,
            ProfileField::metric_with_summary("width", width, false)?,
            ProfileField::metric_with_summary("height", height, false)?,
            ProfileField::metric_with_summary("components", sampling.components, false)?,
            ProfileField::metric_with_summary("quality", options.quality, false)?,
            ProfileField::metric_with_summary(
                "subsampling",
                format_args!("{:?}", options.subsampling),
                false,
            )?,
            ProfileField::metric_with_summary(
                "restart_interval",
                options.restart_interval.unwrap_or(0),
                false,
            )?,
            ProfileField::metric("validation_us", profile.validation.as_micros())?,
            ProfileField::metric("setup_us", profile.setup.as_micros())?,
            ProfileField::metric("planes_us", profile.planes.as_micros())?,
            ProfileField::metric("header_us", profile.header.as_micros())?,
            ProfileField::metric("entropy_us", profile.entropy.as_micros())?,
            ProfileField::metric("total_us", start.elapsed().as_micros())?,
            ProfileField::metric_with_summary("output_bytes", encoded.data.len(), false)?,
            ProfileField::metric_with_summary(
                "rayon_threads",
                rayon::current_num_threads(),
                false,
            )?,
        ])
    });
}
