//! Performing the inverse discrete wavelet transform, as specified in Annex F.

mod direct;
mod filter_common;
mod horizontal;
mod interleave;
mod interleave_i64;
mod model;
mod orchestrate;
mod roi;
mod vertical;

pub(crate) use direct::apply_codestream_single_decomposition_idwt_job;
#[cfg(test)]
pub(crate) use horizontal::test_irreversible_filter_97i;
pub(crate) use model::{idwt_buffer_size, IDWTOutput};
pub(crate) use orchestrate::apply;

// Decoded high-pass sub-bands include JPEG 2000's orientation gain. OpenJPEG's
// synthesis constant is therefore halved at this internal codestream boundary.
const OPENJPEG_NORMALIZED_HIGH_PASS_F32: f32 =
    j2k_codec_math::dwt::IDWT97_OPENJPEG_TWO_INV_KAPPA_F32 * 0.5;

#[cfg(test)]
mod tests;
