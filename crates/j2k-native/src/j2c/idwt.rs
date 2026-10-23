//! Performing the inverse discrete wavelet transform, as specified in Annex F.

use alloc::vec::Vec;

use crate::{try_resize_decode_elements, Result};

mod banded;
mod direct;
mod filter_common;
mod horizontal;
mod interleave;
mod interleave_i64;
mod lines;
mod model;
mod orchestrate;
mod roi;
mod vertical;

pub(crate) use banded::{
    banded_scratch_elements_for_execution, pack_last_level, planned_band_scratch_elements,
    LastLevel,
};
pub(crate) use direct::apply_codestream_single_decomposition_idwt_job;
pub(crate) use direct::apply_single_decomposition_idwt_job;
#[cfg(test)]
pub(crate) use horizontal::test_irreversible_filter_97i;
pub(crate) use model::{idwt_buffer_size, IDWTOutput, IdwtPlacement};
pub(crate) use orchestrate::apply;
pub(crate) use roi::{last_level_fits_channels, roi_idwt_peak_elements};

// Decoded high-pass sub-bands include JPEG 2000's orientation gain. OpenJPEG's
// synthesis constant is therefore halved at this internal codestream boundary.
const OPENJPEG_NORMALIZED_HIGH_PASS_F32: f32 =
    j2k_codec_math::dwt::IDWT97_OPENJPEG_TWO_INV_KAPPA_F32 * 0.5;

#[cfg(test)]
std::thread_local! {
    static ZERO_INITIALIZED_SAMPLES: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_zero_initialized_samples() {
    ZERO_INITIALIZED_SAMPLES.set(0);
}

/// IDWT level and band-scratch samples zero-initialized on this thread since
/// the last reset.
#[cfg(test)]
pub(crate) fn zero_initialized_samples() -> usize {
    ZERO_INITIALIZED_SAMPLES.get()
}

/// Make the first `len` samples of an IDWT level or band-scratch buffer
/// addressable; callers use `buffer[..len]`.
///
/// The buffer never shrinks. Every level and band writes all `len` samples
/// before reading any of them, so samples a previous level or component left
/// behind need no zeroing; only newly grown samples are initialized.
fn try_grow_level_buffer(buffer: &mut Vec<f32>, len: usize) -> Result<()> {
    if buffer.len() >= len {
        return Ok(());
    }
    #[cfg(test)]
    ZERO_INITIALIZED_SAMPLES.set(ZERO_INITIALIZED_SAMPLES.get() + len - buffer.len());
    try_resize_decode_elements(buffer, len, 0.0)
}

#[cfg(test)]
mod tests;
