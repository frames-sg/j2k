// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) const J2K_FDWT97_ALPHA: f32 = j2k_codec_math::dwt::DWT97_ALPHA_F32;
pub(crate) const J2K_FDWT97_BETA: f32 = j2k_codec_math::dwt::DWT97_BETA_F32;
pub(crate) const J2K_FDWT97_GAMMA: f32 = j2k_codec_math::dwt::DWT97_GAMMA_F32;
pub(crate) const J2K_FDWT97_DELTA: f32 = j2k_codec_math::dwt::DWT97_DELTA_F32;
pub(crate) const J2K_FDWT97_KAPPA: f32 = j2k_codec_math::dwt::DWT97_KAPPA_F32;
pub(crate) const J2K_FDWT97_INV_KAPPA: f32 = j2k_codec_math::dwt::DWT97_INV_KAPPA_F32;
pub(crate) const J2K_HT_MEL_SIZE: u32 = 192;
pub(crate) const J2K_HT_VLC_SIZE: u32 = 3072 - J2K_HT_MEL_SIZE;
pub(crate) const J2K_HT_MS_SIZE: u32 = ((16384 * 16) + 14) / 15;
pub(crate) const J2K_HT_MEL_OFFSET: u32 = J2K_HT_MS_SIZE;
pub(crate) const J2K_HT_VLC_OFFSET: u32 = J2K_HT_MS_SIZE + J2K_HT_MEL_SIZE;
pub(crate) const J2K_HT_COMPACT_ASSEMBLE_FLAG: u32 = 0x8000_0000;
pub(crate) const J2K_HT_COMPACT_LENGTH_MASK: u32 = 0x7fff;
pub(crate) const J2K_ENCODE_STATUS_OK: u32 = 0;
pub(crate) const J2K_ENCODE_STATUS_FAIL: u32 = 1;
pub(crate) const J2K_ENCODE_STATUS_UNSUPPORTED: u32 = 2;
pub(crate) const J2K_PACKET_TAG_INF: u32 = 0x7fff_ffff;
pub(crate) const J2K_PACKET_MAX_TAG_NODES: usize = 2048;
pub(crate) const J2K_PACKET_MAX_TAG_LEVELS: usize = 16;
