// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::arithmetic_decoder::ArithmeticDecoder;
use super::super::build::CodeBlock;
use super::super::decode::DecompositionStorage;
use super::arithmetic::{
    cleanup_pass_arithmetic_with_neighbors, magnitude_refinement_pass_arithmetic_with_neighbors,
    significance_propagation_pass_arithmetic_with_neighbors,
};
use super::bypass::{BitDecoder, BypassDecoder, SafeScalarTier1};
use super::observer::{J2kDecodeObserver, NoJ2kDecodeStats};
use super::state::{
    extend_preallocated, push_preallocated, BitPlaneDecodeBuffers, BitPlaneDecodeContext,
};
use crate::J2kCodeBlockSegment;

pub(super) fn decode_inner(
    code_block: &CodeBlock,
    storage: &DecompositionStorage<'_>,
    ctx: &mut BitPlaneDecodeContext,
    bp_buffers: &mut BitPlaneDecodeBuffers,
) -> Option<()> {
    let mut observer = NoJ2kDecodeStats;
    decode_inner_with_observer(code_block, storage, ctx, bp_buffers, &mut observer)
}

fn decode_inner_with_observer<O: J2kDecodeObserver>(
    code_block: &CodeBlock,
    storage: &DecompositionStorage<'_>,
    ctx: &mut BitPlaneDecodeContext,
    bp_buffers: &mut BitPlaneDecodeBuffers,
    observer: &mut O,
) -> Option<()> {
    bp_buffers.reset()?;

    let mut last_segment_idx = 0;
    let mut coding_passes = 0;

    // Build a list so that we can associate coding passes with their segments
    // and data more easily.
    for layer in storage.layers.get(code_block.layers.clone())? {
        if let Some(range) = layer.segments.clone() {
            let layer_segments = storage.segments.get(range)?;
            for segment in layer_segments {
                if segment.idx != last_segment_idx {
                    if last_segment_idx.checked_add(1) != Some(segment.idx) {
                        return None;
                    }

                    push_preallocated(
                        &mut bp_buffers.segment_ranges,
                        bp_buffers.combined_layers.len(),
                    )?;
                    push_preallocated(&mut bp_buffers.segment_coding_passes, coding_passes)?;
                    last_segment_idx += 1;
                }

                extend_preallocated(&mut bp_buffers.combined_layers, segment.data)?;
                coding_passes = coding_passes.checked_add(segment.coding_pases)?;
            }
        }
    }

    if coding_passes != code_block.number_of_coding_passes {
        return None;
    }

    push_preallocated(
        &mut bp_buffers.segment_ranges,
        bp_buffers.combined_layers.len(),
    )?;
    push_preallocated(&mut bp_buffers.segment_coding_passes, coding_passes)?;

    let is_normal_mode =
        !ctx.style.selective_arithmetic_coding_bypass && !ctx.style.termination_on_each_pass;

    if is_normal_mode {
        // Only one termination per code block, so we can just decode the
        // whole range in one single go, processing all coding passes at once.
        let mut decoder = ArithmeticDecoder::new(&bp_buffers.combined_layers);
        let end = code_block
            .number_of_coding_passes
            .min(ctx.max_coding_passes);
        if ctx.uses_normal_arithmetic_neighbor_path() {
            handle_normal_arithmetic_coding_passes(0, end, ctx, &mut decoder, observer)?;
        } else {
            handle_arithmetic_coding_passes(0, end, ctx, &mut decoder, observer)?;
        }
    } else {
        // Otherwise, each segment introduces a termination. For "termination on
        // each pass", each segment only covers one coding pass
        // and a termination is introduced every time. Otherwise, for only
        // arithmetic coding bypass, terminations are introduced based on the
        // exact index of the covered coding passes (see Table D.9).
        for segment in 0..bp_buffers.segment_coding_passes.len() - 1 {
            let start_coding_pass = bp_buffers.segment_coding_passes[segment];
            let end_coding_pass =
                bp_buffers.segment_coding_passes[segment + 1].min(ctx.max_coding_passes);

            let data = &bp_buffers.combined_layers
                [bp_buffers.segment_ranges[segment]..bp_buffers.segment_ranges[segment + 1]];

            let use_arithmetic = if ctx.style.selective_arithmetic_coding_bypass {
                if start_coding_pass <= 9 {
                    true
                } else {
                    // Only for cleanup pass.
                    start_coding_pass.is_multiple_of(3)
                }
            } else {
                true
            };

            if use_arithmetic {
                let mut decoder = ArithmeticDecoder::new(data);
                handle_arithmetic_coding_passes(
                    start_coding_pass,
                    end_coding_pass,
                    ctx,
                    &mut decoder,
                    observer,
                )?;
            } else {
                let mut decoder = BypassDecoder::new(data, ctx.strict);
                handle_bypass_coding_passes(
                    start_coding_pass,
                    end_coding_pass,
                    ctx,
                    &mut decoder,
                    observer,
                )?;
            }
        }
    }

    Some(())
}

fn handle_arithmetic_coding_passes(
    start: u8,
    end: u8,
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
    observer: &mut impl J2kDecodeObserver,
) -> Option<()> {
    handle_arithmetic_coding_passes_with_neighbors::<false>(start, end, ctx, decoder, observer)
}

fn handle_normal_arithmetic_coding_passes(
    start: u8,
    end: u8,
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
    observer: &mut impl J2kDecodeObserver,
) -> Option<()> {
    handle_arithmetic_coding_passes_with_neighbors::<true>(start, end, ctx, decoder, observer)
}

fn handle_arithmetic_coding_passes_with_neighbors<const NORMAL_NEIGHBORS: bool>(
    start: u8,
    end: u8,
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut ArithmeticDecoder<'_>,
    observer: &mut impl J2kDecodeObserver,
) -> Option<()> {
    for coding_pass in start..end {
        let current_bitplane = coding_pass.div_ceil(3);
        ctx.current_bit_position = ctx.bitplanes - 1 - current_bitplane;

        // The first bitplane only has a cleanup pass, all other bitplanes
        // are in the order SPP -> MRR -> C.
        match coding_pass % 3 {
            0 => {
                let phase_start = observer.phase_start();
                cleanup_pass_arithmetic_with_neighbors::<NORMAL_NEIGHBORS>(ctx, decoder);

                if ctx.style.segmentation_symbols {
                    let b0 = decoder.read_bit(ctx.arithmetic_decoder_context(18));
                    let b1 = decoder.read_bit(ctx.arithmetic_decoder_context(18));
                    let b2 = decoder.read_bit(ctx.arithmetic_decoder_context(18));
                    let b3 = decoder.read_bit(ctx.arithmetic_decoder_context(18));

                    if (b0 != 1 || b1 != 0 || b2 != 1 || b3 != 0) && ctx.strict {
                        return None;
                    }
                }

                ctx.reset_for_next_bitplane();
                observer.add_cleanup_us(phase_start);
            }
            1 => {
                let phase_start = observer.phase_start();
                significance_propagation_pass_arithmetic_with_neighbors::<NORMAL_NEIGHBORS>(
                    ctx, decoder,
                );
                observer.add_sigprop_us(phase_start);
            }
            2 => {
                let phase_start = observer.phase_start();
                magnitude_refinement_pass_arithmetic_with_neighbors::<NORMAL_NEIGHBORS>(
                    ctx, decoder,
                );
                observer.add_magref_us(phase_start);
            }
            _ => unreachable!(),
        }

        if ctx.style.reset_context_probabilities {
            ctx.reset_contexts();
        }
    }

    Some(())
}

fn handle_bypass_coding_passes(
    start: u8,
    end: u8,
    ctx: &mut BitPlaneDecodeContext,
    decoder: &mut BypassDecoder<'_>,
    observer: &mut impl J2kDecodeObserver,
) -> Option<()> {
    for coding_pass in start..end {
        let phase_start = observer.phase_start();
        let current_bitplane = coding_pass.div_ceil(3);
        ctx.current_bit_position = ctx.bitplanes - 1 - current_bitplane;

        match coding_pass % 3 {
            0 => {
                SafeScalarTier1::cleanup_pass_bypass(ctx, decoder)?;

                if ctx.style.segmentation_symbols {
                    let b0 = decoder.read_bit(ctx.arithmetic_decoder_context(18))?;
                    let b1 = decoder.read_bit(ctx.arithmetic_decoder_context(18))?;
                    let b2 = decoder.read_bit(ctx.arithmetic_decoder_context(18))?;
                    let b3 = decoder.read_bit(ctx.arithmetic_decoder_context(18))?;

                    if (b0 != 1 || b1 != 0 || b2 != 1 || b3 != 0) && ctx.strict {
                        return None;
                    }
                }

                ctx.reset_for_next_bitplane();
            }
            1 => {
                SafeScalarTier1::significance_propagation_pass_bypass(ctx, decoder)?;
            }
            2 => {
                SafeScalarTier1::magnitude_refinement_pass_bypass(ctx, decoder)?;
            }
            _ => unreachable!(),
        }

        if ctx.style.reset_context_probabilities {
            ctx.reset_contexts();
        }
        observer.add_bypass_us(phase_start);
    }

    Some(())
}

pub(super) fn decode_code_block_segments_inner(
    data: &[u8],
    segments: &[J2kCodeBlockSegment],
    number_of_coding_passes: u8,
    ctx: &mut BitPlaneDecodeContext,
    observer: &mut impl J2kDecodeObserver,
) -> Option<()> {
    let mut expected_start = 0u8;

    for segment in segments {
        if segment.start_coding_pass != expected_start
            || segment.start_coding_pass > segment.end_coding_pass
        {
            return None;
        }
        expected_start = segment.end_coding_pass;

        let start_coding_pass = segment.start_coding_pass;
        let end_coding_pass = segment.end_coding_pass.min(ctx.max_coding_passes);
        let data_start = usize::try_from(segment.data_offset).ok()?;
        let data_length = usize::try_from(segment.data_length).ok()?;
        let data_end = data_start.checked_add(data_length)?;
        let segment_data = data.get(data_start..data_end)?;

        if segment.use_arithmetic {
            let mut decoder = ArithmeticDecoder::new(segment_data);
            if ctx.uses_normal_arithmetic_neighbor_path() {
                handle_normal_arithmetic_coding_passes(
                    start_coding_pass,
                    end_coding_pass,
                    ctx,
                    &mut decoder,
                    observer,
                )?;
            } else {
                handle_arithmetic_coding_passes(
                    start_coding_pass,
                    end_coding_pass,
                    ctx,
                    &mut decoder,
                    observer,
                )?;
            }
        } else {
            let mut decoder = BypassDecoder::new(segment_data, ctx.strict);
            handle_bypass_coding_passes(
                start_coding_pass,
                end_coding_pass,
                ctx,
                &mut decoder,
                observer,
            )?;
        }
    }

    if expected_start != number_of_coding_passes {
        return None;
    }

    Some(())
}
