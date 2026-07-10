// SPDX-License-Identifier: MIT OR Apache-2.0

// Coding-pass scheduling and classic segment state management.

use alloc::vec;
use alloc::vec::Vec;

use super::super::arithmetic_encoder::{ArithmeticEncoder, ArithmeticEncoderContext};
use super::super::build::SubBandType;
use super::super::codestream::CodeBlockStyle;
use crate::math::bit_width_u64;
use crate::writer::BitWriter;

use super::distortion::segment_distortion_delta;
use super::passes::{
    cleanup_pass, clear_coded_in_current_pass, magnitude_refinement_pass,
    magnitude_refinement_pass_raw, prepare_padded_coefficients, significance_propagation_pass,
    significance_propagation_pass_raw,
};
use super::{
    encode_code_block_with_style_i64, EncodedCodeBlockSegment, EncodedCodeBlockWithSegments,
};

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "segmented scheduling shares one borrowed style across every coding pass"
)]
pub(super) fn encode_segmented_code_block(
    coefficients: &[i64],
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    style: &CodeBlockStyle,
) -> EncodedCodeBlockWithSegments {
    if let Some(encoded) = encode_unsegmented_code_block(
        coefficients,
        width,
        height,
        sub_band_type,
        total_bitplanes,
        style,
    ) {
        return encoded;
    }

    let max_magnitude = coefficients
        .iter()
        .map(|coefficient| coefficient.unsigned_abs())
        .max()
        .unwrap_or(0);
    if max_magnitude == 0 {
        return empty_segmented_code_block(total_bitplanes);
    }

    let num_bitplanes = bit_width_u64(max_magnitude);
    debug_assert!(num_bitplanes <= total_bitplanes);
    let num_zero_bitplanes = total_bitplanes.saturating_sub(num_bitplanes);
    SegmentedCodeBlockEncoder::new(
        coefficients,
        width as usize,
        height as usize,
        sub_band_type,
        num_bitplanes,
        style,
    )
    .encode_all_passes(num_zero_bitplanes)
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the adapter shares the caller's style with the stable encoder entrypoint"
)]
fn encode_unsegmented_code_block(
    coefficients: &[i64],
    width: u32,
    height: u32,
    sub_band_type: SubBandType,
    total_bitplanes: u8,
    style: &CodeBlockStyle,
) -> Option<EncodedCodeBlockWithSegments> {
    if style.termination_on_each_pass || style.selective_arithmetic_coding_bypass {
        return None;
    }

    let encoded = encode_code_block_with_style_i64(
        coefficients,
        width,
        height,
        sub_band_type,
        total_bitplanes,
        style,
    );
    let segments = if encoded.num_coding_passes == 0 {
        Vec::new()
    } else {
        vec![EncodedCodeBlockSegment {
            data_offset: 0,
            data_length: u32::try_from(encoded.data.len())
                .expect("classic code-block payload length fits in u32"),
            start_coding_pass: 0,
            end_coding_pass: encoded.num_coding_passes,
            distortion_delta: segment_distortion_delta(
                coefficients,
                0,
                encoded.num_coding_passes,
                total_bitplanes,
            ),
            use_arithmetic: true,
        }]
    };
    Some(EncodedCodeBlockWithSegments {
        data: encoded.data,
        segments,
        num_coding_passes: encoded.num_coding_passes,
        num_zero_bitplanes: encoded.num_zero_bitplanes,
    })
}

fn empty_segmented_code_block(total_bitplanes: u8) -> EncodedCodeBlockWithSegments {
    EncodedCodeBlockWithSegments {
        data: Vec::new(),
        segments: Vec::new(),
        num_coding_passes: 0,
        num_zero_bitplanes: total_bitplanes,
    }
}

struct SegmentedCodeBlockEncoder<'a> {
    coefficients: &'a [i64],
    sub_band_type: SubBandType,
    style: &'a CodeBlockStyle,
    width: usize,
    height: usize,
    padded_width: usize,
    num_bitplanes: u8,
    magnitudes: Vec<u64>,
    states: Vec<u8>,
    neighbors: Vec<u8>,
    contexts: [ArithmeticEncoderContext; 19],
    data: Vec<u8>,
    segments: Vec<EncodedCodeBlockSegment>,
    current_segment_idx: Option<u8>,
    current_segment_start_pass: u8,
    current_use_arithmetic: bool,
    arithmetic_encoder: Option<ArithmeticEncoder>,
    bypass_writer: Option<BitWriter>,
    coded_indices: Vec<usize>,
}

impl<'a> SegmentedCodeBlockEncoder<'a> {
    fn new(
        coefficients: &'a [i64],
        width: usize,
        height: usize,
        sub_band_type: SubBandType,
        num_bitplanes: u8,
        style: &'a CodeBlockStyle,
    ) -> Self {
        let padded_width = width + 2;
        let (magnitudes, states) =
            prepare_padded_coefficients(coefficients, width, height, padded_width);
        let neighbors = vec![0u8; magnitudes.len()];
        let mut contexts = [ArithmeticEncoderContext::default(); 19];
        reset_contexts(&mut contexts);

        Self {
            coefficients,
            sub_band_type,
            style,
            width,
            height,
            padded_width,
            num_bitplanes,
            magnitudes,
            states,
            neighbors,
            contexts,
            data: Vec::new(),
            segments: Vec::new(),
            current_segment_idx: None,
            current_segment_start_pass: 0,
            current_use_arithmetic: true,
            arithmetic_encoder: None,
            bypass_writer: None,
            coded_indices: Vec::new(),
        }
    }

    fn encode_all_passes(mut self, num_zero_bitplanes: u8) -> EncodedCodeBlockWithSegments {
        let total_passes = 1 + 3 * (self.num_bitplanes - 1);
        for coding_pass in 0..total_passes {
            self.begin_segment_for_pass(coding_pass);
            self.encode_coding_pass(coding_pass);
            if self.style.reset_context_probabilities {
                reset_contexts(&mut self.contexts);
            }
        }
        if self.current_segment_idx.is_some() {
            self.finish_current_segment(total_passes, true);
        }

        EncodedCodeBlockWithSegments {
            data: self.data,
            segments: self.segments,
            num_coding_passes: total_passes,
            num_zero_bitplanes,
        }
    }

    fn begin_segment_for_pass(&mut self, coding_pass: u8) {
        let segment_idx = segment_index(self.style, coding_pass);
        let use_arithmetic = pass_uses_arithmetic(self.style, coding_pass);
        if self.current_segment_idx == Some(segment_idx) {
            return;
        }

        if let Some(previous_idx) = self.current_segment_idx {
            self.finish_current_segment(coding_pass, false);
            debug_assert!(previous_idx < segment_idx);
        }

        self.current_segment_idx = Some(segment_idx);
        self.current_segment_start_pass = coding_pass;
        self.current_use_arithmetic = use_arithmetic;
        if use_arithmetic {
            self.arithmetic_encoder = Some(ArithmeticEncoder::new());
            self.bypass_writer = None;
        } else {
            self.arithmetic_encoder = None;
            self.bypass_writer = Some(BitWriter::new());
        }
    }

    fn finish_current_segment(&mut self, end_coding_pass: u8, final_segment: bool) {
        let segment_data = if self.current_use_arithmetic {
            let message = if final_segment {
                "final arithmetic segment encoder exists"
            } else {
                "arithmetic segment encoder exists"
            };
            self.arithmetic_encoder.take().expect(message).finish()
        } else {
            let message = if final_segment {
                "final bypass segment writer exists"
            } else {
                "bypass segment writer exists"
            };
            self.bypass_writer.take().expect(message).finish()
        };
        push_segment(
            &mut self.data,
            &mut self.segments,
            self.current_segment_start_pass,
            end_coding_pass,
            segment_data,
            segment_distortion_delta(
                self.coefficients,
                self.current_segment_start_pass,
                end_coding_pass,
                self.num_bitplanes,
            ),
            self.current_use_arithmetic,
        );
    }

    fn encode_coding_pass(&mut self, coding_pass: u8) {
        let current_bitplane = usize::from(coding_pass.div_ceil(3));
        let bit_mask = 1u64 << (usize::from(self.num_bitplanes) - 1 - current_bitplane);
        match coding_pass % 3 {
            0 => self.encode_cleanup(bit_mask),
            1 => self.encode_significance_propagation(bit_mask),
            2 => self.encode_magnitude_refinement(bit_mask),
            _ => unreachable!(),
        }
    }

    fn encode_cleanup(&mut self, bit_mask: u64) {
        let encoder = self
            .arithmetic_encoder
            .as_mut()
            .expect("cleanup pass uses arithmetic encoder");
        cleanup_pass(
            &self.magnitudes,
            &mut self.states,
            &mut self.neighbors,
            encoder,
            &mut self.contexts,
            self.width,
            self.height,
            self.padded_width,
            bit_mask,
            self.sub_band_type,
            self.style,
        );
        if self.style.segmentation_symbols {
            encode_segmentation_symbols(encoder, &mut self.contexts);
        }
        clear_coded_in_current_pass(&mut self.states, &mut self.coded_indices);
    }

    fn encode_significance_propagation(&mut self, bit_mask: u64) {
        if self.current_use_arithmetic {
            significance_propagation_pass(
                &self.magnitudes,
                &mut self.states,
                &mut self.neighbors,
                &mut self.coded_indices,
                self.arithmetic_encoder
                    .as_mut()
                    .expect("arithmetic encoder exists for significance pass"),
                &mut self.contexts,
                self.width,
                self.height,
                self.padded_width,
                bit_mask,
                self.sub_band_type,
                self.style,
            );
        } else {
            significance_propagation_pass_raw(
                &self.magnitudes,
                &mut self.states,
                &mut self.neighbors,
                &mut self.coded_indices,
                self.bypass_writer
                    .as_mut()
                    .expect("bypass writer exists for significance pass"),
                self.width,
                self.height,
                self.padded_width,
                bit_mask,
                self.style,
            );
        }
    }

    fn encode_magnitude_refinement(&mut self, bit_mask: u64) {
        if self.current_use_arithmetic {
            magnitude_refinement_pass(
                &self.magnitudes,
                &mut self.states,
                &mut self.neighbors,
                self.arithmetic_encoder
                    .as_mut()
                    .expect("arithmetic encoder exists for refinement pass"),
                &mut self.contexts,
                self.width,
                self.height,
                self.padded_width,
                bit_mask,
                self.style,
            );
        } else {
            magnitude_refinement_pass_raw(
                &self.magnitudes,
                &mut self.states,
                &mut self.neighbors,
                self.bypass_writer
                    .as_mut()
                    .expect("bypass writer exists for refinement pass"),
                self.width,
                self.height,
                self.padded_width,
                bit_mask,
                self.style,
            );
        }
    }
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the pass loop keeps one borrowed style rather than copying it for each query"
)]
fn segment_index(style: &CodeBlockStyle, coding_pass: u8) -> u8 {
    if style.termination_on_each_pass {
        coding_pass
    } else if style.selective_arithmetic_coding_bypass {
        bypass_segment_idx(coding_pass)
    } else {
        0
    }
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "the pass loop keeps one borrowed style rather than copying it for each query"
)]
fn pass_uses_arithmetic(style: &CodeBlockStyle, coding_pass: u8) -> bool {
    !style.selective_arithmetic_coding_bypass || coding_pass <= 9 || coding_pass.is_multiple_of(3)
}

pub(super) fn reset_contexts(contexts: &mut [ArithmeticEncoderContext; 19]) {
    *contexts = [ArithmeticEncoderContext::default(); 19];
    contexts[0].reset_with_index(4);
    contexts[17].reset_with_index(3);
    contexts[18].reset_with_index(46);
}

pub(super) fn arithmetic_encoder_capacity(width: usize, height: usize, bitplanes: usize) -> usize {
    1 + width
        .saturating_mul(height)
        .saturating_mul(bitplanes)
        .checked_div(16)
        .unwrap_or(usize::MAX)
        .max(32)
}

pub(super) fn encode_segmentation_symbols(
    encoder: &mut ArithmeticEncoder,
    contexts: &mut [ArithmeticEncoderContext; 19],
) {
    encoder.encode(1, &mut contexts[18]);
    encoder.encode(0, &mut contexts[18]);
    encoder.encode(1, &mut contexts[18]);
    encoder.encode(0, &mut contexts[18]);
}

#[inline]
fn bypass_segment_idx(pass_idx: u8) -> u8 {
    if pass_idx < 10 {
        0
    } else {
        1 + (2 * ((pass_idx - 10) / 3)) + u8::from(((pass_idx - 10) % 3) == 2)
    }
}

pub(super) fn push_segment(
    data: &mut Vec<u8>,
    segments: &mut Vec<EncodedCodeBlockSegment>,
    start_coding_pass: u8,
    end_coding_pass: u8,
    segment_data: Vec<u8>,
    distortion_delta: f64,
    use_arithmetic: bool,
) {
    let data_offset =
        u32::try_from(data.len()).expect("classic code-block data offset fits in u32");
    let data_length =
        u32::try_from(segment_data.len()).expect("classic code-block segment length fits in u32");
    data.extend(segment_data);
    segments.push(EncodedCodeBlockSegment {
        data_offset,
        data_length,
        start_coding_pass,
        end_coding_pass,
        distortion_delta,
        use_arithmetic,
    });
}
