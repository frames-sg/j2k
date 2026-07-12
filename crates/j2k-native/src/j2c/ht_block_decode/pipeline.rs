// SPDX-License-Identifier: MIT OR Apache-2.0

use super::cleanup::{
    cleanup_segment_suffix_length, cleanup_symbol_stride, decode_cleanup_symbols,
};
use super::magnitude::{decode_cleanup_and_magnitude_sign_phase, decode_magnitude_sign_phase};
use super::refinement::apply_magnitude_refinement_phase;
use super::significance::{
    apply_significance_propagation_phase, build_sigma_from_cleanup_phase, sigma_stride,
};
use super::state::{
    resized_u16_scratch, resized_u32_scratch, zeroed_u16_scratch, HtBlockDecodeScratch,
    HtDecodeObserver,
};

mod scratch;
pub(crate) use self::scratch::ht_decode_workspace_bytes;
pub(super) use self::scratch::prepare_scratch;

pub(crate) const PHASE_LIMIT_CLEANUP: u8 = 0;
pub(crate) const PHASE_LIMIT_SIGPROP: u8 = 1;
pub(crate) const PHASE_LIMIT_MAGREF: u8 = 2;

#[expect(
    clippy::inline_always,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "the monomorphized HT phase pipeline has a stable hot signature and preserves normative pass order"
)]
#[inline(always)]
pub(super) fn decode_impl<const PHASE_LIMIT: u8, O: HtDecodeObserver>(
    cleanup_data: &[u8],
    refinement_data: &[u8],
    decoded_data: &mut [u32],
    missing_msbs: u32,
    mut num_passes: u32,
    width: u32,
    height: u32,
    stride: u32,
    stripe_causal: bool,
    scratch_buffers: &mut HtBlockDecodeScratch,
    observer: &mut O,
) -> Option<()> {
    observer.record_block(cleanup_data.len(), refinement_data.len());

    if num_passes > 1 && refinement_data.is_empty() {
        num_passes = 1;
    }

    if num_passes > 3 || missing_msbs > 30 {
        return None;
    }

    if missing_msbs == 29 && num_passes > 1 {
        num_passes = 1;
    }

    let p = 30 - missing_msbs;
    let lcup = cleanup_data.len();

    if lcup < 2 {
        return None;
    }

    let scup = cleanup_segment_suffix_length(cleanup_data, lcup)?;

    let quad_rows = height.div_ceil(2) as usize;
    let sstr = cleanup_symbol_stride(width);
    let v_n_width = width.div_ceil(2) as usize + 2;
    let v_n_scratch = resized_u32_scratch(&mut scratch_buffers.v_n, v_n_width)?;
    let cleanup_only = PHASE_LIMIT == PHASE_LIMIT_CLEANUP || num_passes == 1;
    let scratch = if cleanup_only {
        resized_u16_scratch(&mut scratch_buffers.cleanup, sstr * (quad_rows + 1))?
    } else {
        zeroed_u16_scratch(&mut scratch_buffers.cleanup, sstr * (quad_rows + 1))?
    };

    if cleanup_only {
        decode_cleanup_and_magnitude_sign_phase(
            cleanup_data,
            lcup,
            scup,
            decoded_data,
            missing_msbs,
            width,
            height,
            stride,
            sstr,
            scratch,
            v_n_scratch,
            observer,
        )?;
    } else {
        let phase_start = observer.phase_start();
        decode_cleanup_symbols(cleanup_data, lcup, scup, width, height, sstr, scratch)?;
        observer.add_cleanup_us(phase_start);

        let phase_start = observer.phase_start();
        decode_magnitude_sign_phase(
            cleanup_data,
            lcup,
            scup,
            scratch,
            decoded_data,
            missing_msbs,
            width,
            height,
            stride,
            sstr,
            v_n_scratch,
        )?;
        observer.add_mag_sgn_us(phase_start);
    }

    if PHASE_LIMIT == PHASE_LIMIT_CLEANUP {
        return Some(());
    }

    if num_passes > 1 {
        let sigma_rows = height.div_ceil(4) as usize + 1;
        let mstr = sigma_stride(width);
        let sigma = zeroed_u16_scratch(&mut scratch_buffers.sigma, sigma_rows * mstr)?;
        let phase_start = observer.phase_start();
        build_sigma_from_cleanup_phase(scratch, sigma, width, height, sstr, mstr)?;
        observer.add_sigma_us(phase_start);

        let prev_row_sig = resized_u16_scratch(
            &mut scratch_buffers.prev_row_sig,
            width.div_ceil(4) as usize + 8,
        )?;
        let phase_start = observer.phase_start();
        apply_significance_propagation_phase(
            refinement_data,
            sigma,
            decoded_data,
            width,
            height,
            stride,
            mstr,
            stripe_causal,
            p,
            prev_row_sig,
        )?;
        observer.add_sigprop_us(phase_start);

        if PHASE_LIMIT == PHASE_LIMIT_SIGPROP {
            return Some(());
        }

        if num_passes > 2 {
            let phase_start = observer.phase_start();
            apply_magnitude_refinement_phase(
                refinement_data,
                sigma,
                decoded_data,
                width,
                height,
                stride,
                mstr,
                p,
            )?;
            observer.add_magref_us(phase_start);
        }
    }

    Some(())
}
