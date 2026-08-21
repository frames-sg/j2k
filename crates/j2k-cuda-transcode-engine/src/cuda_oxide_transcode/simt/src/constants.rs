// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_codec_math::jpeg::idct;

pub(crate) const CONST_BITS: i32 = idct::CONST_BITS as i32;
pub(crate) const PASS1_BITS: i32 = idct::PASS1_BITS as i32;
pub(crate) const FIX_0_298631336: i32 = idct::FIX_0_298631336;
pub(crate) const FIX_0_390180644: i32 = idct::FIX_0_390180644;
pub(crate) const FIX_0_541196100: i32 = idct::FIX_0_541196100;
pub(crate) const FIX_0_765366865: i32 = idct::FIX_0_765366865;
pub(crate) const FIX_0_899976223: i32 = idct::FIX_0_899976223;
pub(crate) const FIX_1_175875602: i32 = idct::FIX_1_175875602;
pub(crate) const FIX_1_501321110: i32 = idct::FIX_1_501321110;
pub(crate) const FIX_1_847759065: i32 = idct::FIX_1_847759065;
pub(crate) const FIX_1_961570560: i32 = idct::FIX_1_961570560;
pub(crate) const FIX_2_053119869: i32 = idct::FIX_2_053119869;
pub(crate) const FIX_2_562915447: i32 = idct::FIX_2_562915447;
pub(crate) const FIX_3_072711026: i32 = idct::FIX_3_072711026;

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
