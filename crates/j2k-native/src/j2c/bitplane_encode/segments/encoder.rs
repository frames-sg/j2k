// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use super::super::super::arithmetic_encoder::{ArithmeticEncoder, ArithmeticEncoderContext};
use super::super::super::build::SubBandType;
use super::super::super::codestream::CodeBlockStyle;
use super::super::super::coefficient_view::{CoefficientBlockView, SignedCoefficient};
use super::super::super::encode::allocation::{try_untracked_vec, try_untracked_vec_filled};
use crate::writer::BitWriter;
use crate::{EncodeError, EncodeResult};

use super::super::allocation::ClassicWorkerAllocation;
use super::super::distortion::segment_distortion_delta_view;
use super::super::passes::{
    cleanup_pass, clear_coded_in_current_pass, magnitude_refinement_pass,
    magnitude_refinement_pass_raw, significance_propagation_pass,
    significance_propagation_pass_raw,
};
use super::super::preparation::try_prepare_padded_coefficients_from_view;
use super::super::{EncodedCodeBlockSegment, EncodedCodeBlockWithSegments};
use super::{
    encode_segmentation_symbols, pass_uses_arithmetic, reset_contexts, segment_index,
    try_push_segment, PendingSegment,
};

pub(super) struct SegmentedCodeBlockEncoder<'a, T> {
    coefficients: CoefficientBlockView<'a, T>,
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
    payload_limit: usize,
    segment_limit: usize,
}

impl<'a, T: SignedCoefficient> SegmentedCodeBlockEncoder<'a, T> {
    pub(super) fn try_new(
        coefficients: CoefficientBlockView<'a, T>,
        sub_band_type: SubBandType,
        num_bitplanes: u8,
        style: &'a CodeBlockStyle,
        allocation: ClassicWorkerAllocation,
    ) -> EncodeResult<Self> {
        let width = coefficients.width();
        let height = coefficients.height();
        let padded_width = width
            .checked_add(2)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "classic Tier-1 padded width",
            })?;
        let (magnitudes, states) =
            try_prepare_padded_coefficients_from_view(coefficients, padded_width)?;
        let neighbors = try_untracked_vec_filled(
            allocation.padded_coefficients,
            0_u8,
            "classic Tier-1 neighbor states",
        )?;
        let mut contexts = [ArithmeticEncoderContext::default(); 19];
        reset_contexts(&mut contexts);

        Ok(Self {
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
            data: try_untracked_vec(allocation.payload_bytes.min(256), "classic Tier-1 payload")?,
            segments: try_untracked_vec(
                allocation.coding_passes,
                "classic Tier-1 segment metadata",
            )?,
            current_segment_idx: None,
            current_segment_start_pass: 0,
            current_use_arithmetic: true,
            arithmetic_encoder: None,
            bypass_writer: None,
            coded_indices: try_untracked_vec(
                allocation.padded_coefficients,
                "classic Tier-1 coded-index scratch",
            )?,
            payload_limit: allocation.payload_bytes,
            segment_limit: allocation.coding_passes,
        })
    }

    pub(super) fn try_encode_all_passes(
        mut self,
        num_zero_bitplanes: u8,
    ) -> EncodeResult<EncodedCodeBlockWithSegments> {
        let total_passes = 1 + 3 * (self.num_bitplanes - 1);
        for coding_pass in 0..total_passes {
            self.try_begin_segment_for_pass(coding_pass)?;
            self.encode_coding_pass(coding_pass);
            if self.style.reset_context_probabilities {
                reset_contexts(&mut self.contexts);
            }
        }
        if self.current_segment_idx.is_some() {
            self.try_finish_current_segment(total_passes, true)?;
        }

        Ok(EncodedCodeBlockWithSegments {
            data: self.data,
            segments: self.segments,
            num_coding_passes: total_passes,
            num_zero_bitplanes,
        })
    }

    fn try_begin_segment_for_pass(&mut self, coding_pass: u8) -> EncodeResult<()> {
        let segment_idx = segment_index(self.style, coding_pass);
        let use_arithmetic = pass_uses_arithmetic(self.style, coding_pass);
        if self.current_segment_idx == Some(segment_idx) {
            return Ok(());
        }

        if let Some(previous_idx) = self.current_segment_idx {
            self.try_finish_current_segment(coding_pass, false)?;
            debug_assert!(previous_idx < segment_idx);
        }

        self.current_segment_idx = Some(segment_idx);
        self.current_segment_start_pass = coding_pass;
        self.current_use_arithmetic = use_arithmetic;
        if use_arithmetic {
            self.arithmetic_encoder =
                Some(ArithmeticEncoder::try_with_byte_limit(self.payload_limit)?);
            self.bypass_writer = None;
        } else {
            self.arithmetic_encoder = None;
            self.bypass_writer = Some(BitWriter::try_with_byte_limit(self.payload_limit)?);
        }
        Ok(())
    }

    fn try_finish_current_segment(
        &mut self,
        end_coding_pass: u8,
        _final_segment: bool,
    ) -> EncodeResult<()> {
        let segment_data = if self.current_use_arithmetic {
            self.arithmetic_encoder
                .take()
                .ok_or(EncodeError::InternalInvariant {
                    what: "classic arithmetic segment encoder is missing",
                })?
                .finish_checked()?
        } else {
            self.bypass_writer
                .take()
                .ok_or(EncodeError::InternalInvariant {
                    what: "classic bypass segment writer is missing",
                })?
                .finish_checked()?
        };
        try_push_segment(
            &mut self.data,
            &mut self.segments,
            self.payload_limit,
            self.segment_limit,
            PendingSegment {
                start_coding_pass: self.current_segment_start_pass,
                end_coding_pass,
                data: segment_data,
                distortion_delta: segment_distortion_delta_view(
                    self.coefficients,
                    self.current_segment_start_pass,
                    end_coding_pass,
                    self.num_bitplanes,
                ),
                use_arithmetic: self.current_use_arithmetic,
            },
        )
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
