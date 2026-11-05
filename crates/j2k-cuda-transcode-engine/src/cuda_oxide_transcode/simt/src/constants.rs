// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) const DWT97_ALPHA: f32 = j2k_codec_math::dwt::DWT97_ALPHA_F32;
pub(crate) const DWT97_BETA: f32 = j2k_codec_math::dwt::DWT97_BETA_F32;
pub(crate) const DWT97_GAMMA: f32 = j2k_codec_math::dwt::DWT97_GAMMA_F32;
pub(crate) const DWT97_DELTA: f32 = j2k_codec_math::dwt::DWT97_DELTA_F32;
pub(crate) const DWT97_KAPPA: f32 = j2k_codec_math::dwt::DWT97_KAPPA_F32;
pub(crate) const DWT97_INV_KAPPA: f32 = j2k_codec_math::dwt::DWT97_INV_KAPPA_F32;
pub(crate) const DWT97_ROW_LIFT_MAX_WIDTH: usize = 1024;
pub(crate) const DWT97_ROW_LIFT_ROWS_PER_BLOCK: usize = 4;
pub(crate) const DWT97_ROW_LIFT_SHARED_SAMPLES: usize =
    DWT97_ROW_LIFT_MAX_WIDTH * DWT97_ROW_LIFT_ROWS_PER_BLOCK;

pub(crate) const IDCT_C0: f32 = 0.353_553_38;
pub(crate) const IDCT_C1: f32 = 0.490_392_65;
pub(crate) const IDCT_C2: f32 = 0.461_939_75;
pub(crate) const IDCT_C3: f32 = 0.415_734_8;
pub(crate) const IDCT_C5: f32 = 0.277_785_12;
pub(crate) const IDCT_C6: f32 = 0.191_341_71;
pub(crate) const IDCT_C7: f32 = 0.097_545_16;
