// SPDX-License-Identifier: MIT OR Apache-2.0

// The 9/7 lifting, scaling and IDCT constants come from
// `j2k_codec_math::transcode97`; only the shared-memory layout lives here.
pub(crate) const DWT97_ROW_LIFT_MAX_WIDTH: usize = 1024;
pub(crate) const DWT97_ROW_LIFT_ROWS_PER_BLOCK: usize = 4;
pub(crate) const DWT97_ROW_LIFT_SHARED_SAMPLES: usize =
    DWT97_ROW_LIFT_MAX_WIDTH * DWT97_ROW_LIFT_ROWS_PER_BLOCK;
