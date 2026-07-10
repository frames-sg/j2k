// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::{vec, vec::Vec};

use super::cleanup::{
    cleanup_segment_suffix_length, cleanup_symbol_stride, decode_cleanup_symbols,
};
use super::segments::HtCodeBlockSegments;
use super::significance::{
    apply_significance_propagation_phase, build_sigma_from_cleanup_phase, sigma_stride,
};
use super::validation::validate_combined_decode;
use crate::error::{bail, DecodingError, Result};

pub(crate) struct HtSigPropBenchmarkState {
    refinement_data: Vec<u8>,
    sigma: Vec<u16>,
    prev_row_sig: Vec<u16>,
    width: u32,
    height: u32,
    stride: u32,
    mstr: usize,
    stripe_causal: bool,
    p: u32,
}

impl HtSigPropBenchmarkState {
    pub(crate) fn output_len(&self) -> usize {
        if self.height == 0 {
            0
        } else {
            (self.stride as usize * (self.height as usize - 1)) + self.width as usize
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "the benchmark setup mirrors the stable validated HT decode entry-point signature"
)]
pub(crate) fn prepare_sigprop_benchmark_state(
    segments: &HtCodeBlockSegments<'_>,
    missing_bit_planes: u8,
    total_bitplanes: u8,
    number_of_coding_passes: u8,
    stripe_causal: bool,
    strict: bool,
    width: u32,
    height: u32,
    stride: u32,
) -> Result<HtSigPropBenchmarkState> {
    if !validate_combined_decode(
        missing_bit_planes,
        total_bitplanes,
        number_of_coding_passes,
        strict,
    )? {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }
    if number_of_coding_passes < 2 || segments.refinement.is_empty() || missing_bit_planes > 28 {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }

    let lcup = segments.cleanup.len();
    let scup = cleanup_segment_suffix_length(segments.cleanup, lcup)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;
    let sstr = cleanup_symbol_stride(width);
    let quad_rows = height.div_ceil(2) as usize;
    let mut cleanup = vec![0u16; sstr * (quad_rows + 1)];
    decode_cleanup_symbols(
        segments.cleanup,
        lcup,
        scup,
        width,
        height,
        sstr,
        &mut cleanup,
    )
    .ok_or(DecodingError::CodeBlockDecodeFailure)?;

    let mstr = sigma_stride(width);
    let sigma_rows = height.div_ceil(4) as usize + 1;
    let mut sigma = vec![0u16; sigma_rows * mstr];
    build_sigma_from_cleanup_phase(&cleanup, &mut sigma, width, height, sstr, mstr)
        .ok_or(DecodingError::CodeBlockDecodeFailure)?;

    Ok(HtSigPropBenchmarkState {
        refinement_data: segments.refinement.to_vec(),
        sigma,
        prev_row_sig: vec![0u16; width.div_ceil(4) as usize + 8],
        width,
        height,
        stride,
        mstr,
        stripe_causal,
        p: 30 - u32::from(missing_bit_planes),
    })
}

pub(crate) fn decode_sigprop_benchmark_state(
    state: &mut HtSigPropBenchmarkState,
    decoded_data: &mut [u32],
) -> Result<()> {
    if decoded_data.len() < state.output_len() {
        bail!(DecodingError::CodeBlockDecodeFailure);
    }

    apply_significance_propagation_phase(
        &state.refinement_data,
        &state.sigma,
        decoded_data,
        state.width,
        state.height,
        state.stride,
        state.mstr,
        state.stripe_causal,
        state.p,
        &mut state.prev_row_sig,
    )
    .ok_or(DecodingError::CodeBlockDecodeFailure.into())
}
