// SPDX-License-Identifier: Apache-2.0

//! Shared public-to-native JPEG 2000 adapter conversions.

use alloc::vec::Vec;

use crate::{
    EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, IrreversibleQuantizationSubbandScales,
    J2kCodeBlockSegment, J2kCodeBlockStyle, J2kEncodeDispatchReport, J2kForwardDwt53Level,
    J2kForwardDwt53Output, J2kForwardDwt97Level, J2kForwardDwt97Output, J2kHtCodeBlockEncodeJob,
    J2kHtj2kTileEncodeJob, J2kPacketizationBlockCodingMode, J2kPacketizationCodeBlock,
    J2kPacketizationPacketDescriptor, J2kPacketizationProgressionOrder, J2kPacketizationResolution,
    J2kPacketizationSubband, J2kProgressionOrder, J2kSubBandType, J2kTier1CodeBlockEncodeJob,
};

/// Convert a native sub-band identifier into the public adapter type.
pub const fn subband_from_native(subband: signinum_j2k_native::J2kSubBandType) -> J2kSubBandType {
    match subband {
        signinum_j2k_native::J2kSubBandType::LowLow => J2kSubBandType::LowLow,
        signinum_j2k_native::J2kSubBandType::HighLow => J2kSubBandType::HighLow,
        signinum_j2k_native::J2kSubBandType::LowHigh => J2kSubBandType::LowHigh,
        signinum_j2k_native::J2kSubBandType::HighHigh => J2kSubBandType::HighHigh,
    }
}

/// Convert a public adapter sub-band identifier into the native type.
pub const fn subband_to_native(subband: J2kSubBandType) -> signinum_j2k_native::J2kSubBandType {
    match subband {
        J2kSubBandType::LowLow => signinum_j2k_native::J2kSubBandType::LowLow,
        J2kSubBandType::HighLow => signinum_j2k_native::J2kSubBandType::HighLow,
        J2kSubBandType::LowHigh => signinum_j2k_native::J2kSubBandType::LowHigh,
        J2kSubBandType::HighHigh => signinum_j2k_native::J2kSubBandType::HighHigh,
    }
}

/// Convert native classic code-block style flags into public adapter flags.
pub const fn style_from_native(style: signinum_j2k_native::J2kCodeBlockStyle) -> J2kCodeBlockStyle {
    J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: style.selective_arithmetic_coding_bypass,
        reset_context_probabilities: style.reset_context_probabilities,
        termination_on_each_pass: style.termination_on_each_pass,
        vertically_causal_context: style.vertically_causal_context,
        segmentation_symbols: style.segmentation_symbols,
    }
}

/// Convert public classic code-block style flags into native flags.
pub const fn style_to_native(style: J2kCodeBlockStyle) -> signinum_j2k_native::J2kCodeBlockStyle {
    signinum_j2k_native::J2kCodeBlockStyle {
        selective_arithmetic_coding_bypass: style.selective_arithmetic_coding_bypass,
        reset_context_probabilities: style.reset_context_probabilities,
        termination_on_each_pass: style.termination_on_each_pass,
        vertically_causal_context: style.vertically_causal_context,
        segmentation_symbols: style.segmentation_symbols,
    }
}

/// Convert a native classic code-block segment into the public adapter type.
pub const fn segment_from_native(
    segment: signinum_j2k_native::J2kCodeBlockSegment,
) -> J2kCodeBlockSegment {
    J2kCodeBlockSegment {
        data_offset: segment.data_offset,
        data_length: segment.data_length,
        start_coding_pass: segment.start_coding_pass,
        end_coding_pass: segment.end_coding_pass,
        use_arithmetic: segment.use_arithmetic,
    }
}

/// Convert a public classic code-block segment into the native type.
pub const fn segment_to_native(
    segment: J2kCodeBlockSegment,
) -> signinum_j2k_native::J2kCodeBlockSegment {
    signinum_j2k_native::J2kCodeBlockSegment {
        data_offset: segment.data_offset,
        data_length: segment.data_length,
        start_coding_pass: segment.start_coding_pass,
        end_coding_pass: segment.end_coding_pass,
        use_arithmetic: segment.use_arithmetic,
    }
}

/// Convert a native classic encoded code-block into the public adapter type.
pub fn encoded_j2k_from_native(
    block: signinum_j2k_native::EncodedJ2kCodeBlock,
) -> EncodedJ2kCodeBlock {
    EncodedJ2kCodeBlock {
        data: block.data,
        segments: block
            .segments
            .into_iter()
            .map(segment_from_native)
            .collect(),
        number_of_coding_passes: block.number_of_coding_passes,
        missing_bit_planes: block.missing_bit_planes,
    }
}

/// Convert a public classic encoded code-block into the native type.
pub fn encoded_j2k_to_native(
    block: EncodedJ2kCodeBlock,
) -> signinum_j2k_native::EncodedJ2kCodeBlock {
    signinum_j2k_native::EncodedJ2kCodeBlock {
        data: block.data,
        segments: block.segments.into_iter().map(segment_to_native).collect(),
        number_of_coding_passes: block.number_of_coding_passes,
        missing_bit_planes: block.missing_bit_planes,
    }
}

/// Convert a native HTJ2K encoded code-block into the public adapter type.
pub fn encoded_ht_from_native(
    block: signinum_j2k_native::EncodedHtJ2kCodeBlock,
) -> EncodedHtJ2kCodeBlock {
    EncodedHtJ2kCodeBlock {
        data: block.data,
        cleanup_length: block.cleanup_length,
        refinement_length: block.refinement_length,
        num_coding_passes: block.num_coding_passes,
        num_zero_bitplanes: block.num_zero_bitplanes,
    }
}

/// Convert a public HTJ2K encoded code-block into the native type.
pub fn encoded_ht_to_native(
    block: EncodedHtJ2kCodeBlock,
) -> signinum_j2k_native::EncodedHtJ2kCodeBlock {
    signinum_j2k_native::EncodedHtJ2kCodeBlock {
        data: block.data,
        cleanup_length: block.cleanup_length,
        refinement_length: block.refinement_length,
        num_coding_passes: block.num_coding_passes,
        num_zero_bitplanes: block.num_zero_bitplanes,
    }
}

/// Convert a native classic Tier-1 code-block job into the public adapter type.
pub const fn tier1_job_from_native(
    job: signinum_j2k_native::J2kTier1CodeBlockEncodeJob<'_>,
) -> J2kTier1CodeBlockEncodeJob<'_> {
    J2kTier1CodeBlockEncodeJob {
        coefficients: job.coefficients,
        width: job.width,
        height: job.height,
        sub_band_type: subband_from_native(job.sub_band_type),
        total_bitplanes: job.total_bitplanes,
        style: style_from_native(job.style),
    }
}

/// Convert a public classic Tier-1 code-block job into the native type.
pub const fn tier1_job_to_native(
    job: J2kTier1CodeBlockEncodeJob<'_>,
) -> signinum_j2k_native::J2kTier1CodeBlockEncodeJob<'_> {
    signinum_j2k_native::J2kTier1CodeBlockEncodeJob {
        coefficients: job.coefficients,
        width: job.width,
        height: job.height,
        sub_band_type: subband_to_native(job.sub_band_type),
        total_bitplanes: job.total_bitplanes,
        style: style_to_native(job.style),
    }
}

/// Convert a native HTJ2K code-block job into the public adapter type.
pub const fn ht_job_from_native(
    job: signinum_j2k_native::J2kHtCodeBlockEncodeJob<'_>,
) -> J2kHtCodeBlockEncodeJob<'_> {
    J2kHtCodeBlockEncodeJob {
        coefficients: job.coefficients,
        width: job.width,
        height: job.height,
        total_bitplanes: job.total_bitplanes,
        target_coding_passes: job.target_coding_passes,
    }
}

/// Convert a public HTJ2K code-block job into the native type.
pub const fn ht_job_to_native(
    job: J2kHtCodeBlockEncodeJob<'_>,
) -> signinum_j2k_native::J2kHtCodeBlockEncodeJob<'_> {
    signinum_j2k_native::J2kHtCodeBlockEncodeJob {
        coefficients: job.coefficients,
        width: job.width,
        height: job.height,
        total_bitplanes: job.total_bitplanes,
        target_coding_passes: job.target_coding_passes,
    }
}

/// Convert a public whole-tile HTJ2K job into the native type.
pub const fn htj2k_tile_job_to_native(
    job: J2kHtj2kTileEncodeJob<'_>,
) -> signinum_j2k_native::J2kHtj2kTileEncodeJob<'_> {
    signinum_j2k_native::J2kHtj2kTileEncodeJob {
        pixels: job.pixels,
        width: job.width,
        height: job.height,
        num_components: job.num_components,
        bit_depth: job.bit_depth,
        signed: job.signed,
        num_decomposition_levels: job.num_decomposition_levels,
        reversible: job.reversible,
        use_mct: job.use_mct,
        guard_bits: job.guard_bits,
        code_block_width: job.code_block_width,
        code_block_height: job.code_block_height,
        progression_order: packet_progression_to_native(job.progression_order),
        component_sampling: job.component_sampling,
        quantization_steps: job.quantization_steps,
    }
}

/// Convert a native reversible 5/3 DWT output into the public adapter type.
pub fn dwt53_output_from_native(
    output: signinum_j2k_native::J2kForwardDwt53Output,
) -> J2kForwardDwt53Output {
    J2kForwardDwt53Output {
        ll: output.ll,
        ll_width: output.ll_width,
        ll_height: output.ll_height,
        levels: output
            .levels
            .into_iter()
            .map(|level| J2kForwardDwt53Level {
                hl: level.hl,
                lh: level.lh,
                hh: level.hh,
                width: level.width,
                height: level.height,
                low_width: level.low_width,
                low_height: level.low_height,
                high_width: level.high_width,
                high_height: level.high_height,
            })
            .collect(),
    }
}

/// Convert a public reversible 5/3 DWT output into the native type.
pub fn dwt53_output_to_native(
    output: J2kForwardDwt53Output,
) -> signinum_j2k_native::J2kForwardDwt53Output {
    signinum_j2k_native::J2kForwardDwt53Output {
        ll: output.ll,
        ll_width: output.ll_width,
        ll_height: output.ll_height,
        levels: output
            .levels
            .into_iter()
            .map(|level| signinum_j2k_native::J2kForwardDwt53Level {
                hl: level.hl,
                lh: level.lh,
                hh: level.hh,
                width: level.width,
                height: level.height,
                low_width: level.low_width,
                low_height: level.low_height,
                high_width: level.high_width,
                high_height: level.high_height,
            })
            .collect(),
    }
}

/// Convert a native irreversible 9/7 DWT output into the public adapter type.
pub fn dwt97_output_from_native(
    output: signinum_j2k_native::J2kForwardDwt97Output,
) -> J2kForwardDwt97Output {
    J2kForwardDwt97Output {
        ll: output.ll,
        ll_width: output.ll_width,
        ll_height: output.ll_height,
        levels: output
            .levels
            .into_iter()
            .map(|level| J2kForwardDwt97Level {
                hl: level.hl,
                lh: level.lh,
                hh: level.hh,
                width: level.width,
                height: level.height,
                low_width: level.low_width,
                low_height: level.low_height,
                high_width: level.high_width,
                high_height: level.high_height,
            })
            .collect(),
    }
}

/// Convert a public irreversible 9/7 DWT output into the native type.
pub fn dwt97_output_to_native(
    output: J2kForwardDwt97Output,
) -> signinum_j2k_native::J2kForwardDwt97Output {
    signinum_j2k_native::J2kForwardDwt97Output {
        ll: output.ll,
        ll_width: output.ll_width,
        ll_height: output.ll_height,
        levels: output
            .levels
            .into_iter()
            .map(|level| signinum_j2k_native::J2kForwardDwt97Level {
                hl: level.hl,
                lh: level.lh,
                hh: level.hh,
                width: level.width,
                height: level.height,
                low_width: level.low_width,
                low_height: level.low_height,
                high_width: level.high_width,
                high_height: level.high_height,
            })
            .collect(),
    }
}

/// Convert a native packetization descriptor into the public adapter type.
pub const fn packet_descriptor_from_native(
    descriptor: signinum_j2k_native::J2kPacketizationPacketDescriptor,
) -> J2kPacketizationPacketDescriptor {
    J2kPacketizationPacketDescriptor {
        packet_index: descriptor.packet_index,
        state_index: descriptor.state_index,
        layer: descriptor.layer,
        resolution: descriptor.resolution,
        component: descriptor.component,
        precinct: descriptor.precinct,
    }
}

/// Convert a public packetization descriptor into the native type.
pub const fn packet_descriptor_to_native(
    descriptor: J2kPacketizationPacketDescriptor,
) -> signinum_j2k_native::J2kPacketizationPacketDescriptor {
    signinum_j2k_native::J2kPacketizationPacketDescriptor {
        packet_index: descriptor.packet_index,
        state_index: descriptor.state_index,
        layer: descriptor.layer,
        resolution: descriptor.resolution,
        component: descriptor.component,
        precinct: descriptor.precinct,
    }
}

/// Convert a native packetization resolution into the public adapter type.
pub fn packet_resolution_from_native<'a>(
    resolution: &signinum_j2k_native::J2kPacketizationResolution<'a>,
) -> J2kPacketizationResolution<'a> {
    J2kPacketizationResolution {
        subbands: resolution
            .subbands
            .iter()
            .map(|subband| J2kPacketizationSubband {
                code_blocks: subband
                    .code_blocks
                    .iter()
                    .copied()
                    .map(packet_code_block_from_native)
                    .collect(),
                num_cbs_x: subband.num_cbs_x,
                num_cbs_y: subband.num_cbs_y,
            })
            .collect(),
    }
}

/// Convert a public packetization resolution into the native type.
pub fn packet_resolution_to_native<'a>(
    resolution: &J2kPacketizationResolution<'a>,
) -> signinum_j2k_native::J2kPacketizationResolution<'a> {
    signinum_j2k_native::J2kPacketizationResolution {
        subbands: resolution
            .subbands
            .iter()
            .map(|subband| signinum_j2k_native::J2kPacketizationSubband {
                code_blocks: subband
                    .code_blocks
                    .iter()
                    .copied()
                    .map(packet_code_block_to_native)
                    .collect(),
                num_cbs_x: subband.num_cbs_x,
                num_cbs_y: subband.num_cbs_y,
            })
            .collect(),
    }
}

/// Convert public packetization resolutions into native resolution values.
pub fn packet_resolutions_to_native<'a>(
    resolutions: &'a [J2kPacketizationResolution<'a>],
) -> Vec<signinum_j2k_native::J2kPacketizationResolution<'a>> {
    resolutions
        .iter()
        .map(packet_resolution_to_native)
        .collect()
}

/// Convert a native packetization code-block into the public adapter type.
pub const fn packet_code_block_from_native(
    code_block: signinum_j2k_native::J2kPacketizationCodeBlock<'_>,
) -> J2kPacketizationCodeBlock<'_> {
    J2kPacketizationCodeBlock {
        data: code_block.data,
        ht_cleanup_length: code_block.ht_cleanup_length,
        ht_refinement_length: code_block.ht_refinement_length,
        num_coding_passes: code_block.num_coding_passes,
        num_zero_bitplanes: code_block.num_zero_bitplanes,
        previously_included: code_block.previously_included,
        l_block: code_block.l_block,
        block_coding_mode: packet_block_mode_from_native(code_block.block_coding_mode),
    }
}

/// Convert a public packetization code-block into the native type.
pub const fn packet_code_block_to_native(
    code_block: J2kPacketizationCodeBlock<'_>,
) -> signinum_j2k_native::J2kPacketizationCodeBlock<'_> {
    signinum_j2k_native::J2kPacketizationCodeBlock {
        data: code_block.data,
        ht_cleanup_length: code_block.ht_cleanup_length,
        ht_refinement_length: code_block.ht_refinement_length,
        num_coding_passes: code_block.num_coding_passes,
        num_zero_bitplanes: code_block.num_zero_bitplanes,
        previously_included: code_block.previously_included,
        l_block: code_block.l_block,
        block_coding_mode: packet_block_mode_to_native(code_block.block_coding_mode),
    }
}

/// Convert a native packetization block coding mode into the public adapter type.
pub const fn packet_block_mode_from_native(
    mode: signinum_j2k_native::J2kPacketizationBlockCodingMode,
) -> J2kPacketizationBlockCodingMode {
    match mode {
        signinum_j2k_native::J2kPacketizationBlockCodingMode::Classic => {
            J2kPacketizationBlockCodingMode::Classic
        }
        signinum_j2k_native::J2kPacketizationBlockCodingMode::HighThroughput => {
            J2kPacketizationBlockCodingMode::HighThroughput
        }
    }
}

/// Convert a public packetization block coding mode into the native type.
pub const fn packet_block_mode_to_native(
    mode: J2kPacketizationBlockCodingMode,
) -> signinum_j2k_native::J2kPacketizationBlockCodingMode {
    match mode {
        J2kPacketizationBlockCodingMode::Classic => {
            signinum_j2k_native::J2kPacketizationBlockCodingMode::Classic
        }
        J2kPacketizationBlockCodingMode::HighThroughput => {
            signinum_j2k_native::J2kPacketizationBlockCodingMode::HighThroughput
        }
    }
}

/// Convert a native packet progression order into the public adapter type.
pub const fn packet_progression_from_native(
    progression: signinum_j2k_native::J2kPacketizationProgressionOrder,
) -> J2kPacketizationProgressionOrder {
    match progression {
        signinum_j2k_native::J2kPacketizationProgressionOrder::Lrcp => {
            J2kPacketizationProgressionOrder::Lrcp
        }
        signinum_j2k_native::J2kPacketizationProgressionOrder::Rlcp => {
            J2kPacketizationProgressionOrder::Rlcp
        }
        signinum_j2k_native::J2kPacketizationProgressionOrder::Rpcl => {
            J2kPacketizationProgressionOrder::Rpcl
        }
        signinum_j2k_native::J2kPacketizationProgressionOrder::Pcrl => {
            J2kPacketizationProgressionOrder::Pcrl
        }
        signinum_j2k_native::J2kPacketizationProgressionOrder::Cprl => {
            J2kPacketizationProgressionOrder::Cprl
        }
    }
}

/// Convert a public packet progression order into the native type.
pub const fn packet_progression_to_native(
    progression: J2kPacketizationProgressionOrder,
) -> signinum_j2k_native::J2kPacketizationProgressionOrder {
    match progression {
        J2kPacketizationProgressionOrder::Lrcp => {
            signinum_j2k_native::J2kPacketizationProgressionOrder::Lrcp
        }
        J2kPacketizationProgressionOrder::Rlcp => {
            signinum_j2k_native::J2kPacketizationProgressionOrder::Rlcp
        }
        J2kPacketizationProgressionOrder::Rpcl => {
            signinum_j2k_native::J2kPacketizationProgressionOrder::Rpcl
        }
        J2kPacketizationProgressionOrder::Pcrl => {
            signinum_j2k_native::J2kPacketizationProgressionOrder::Pcrl
        }
        J2kPacketizationProgressionOrder::Cprl => {
            signinum_j2k_native::J2kPacketizationProgressionOrder::Cprl
        }
    }
}

/// Convert a public encode progression order into the native encode option type.
pub const fn progression_order_to_native(
    progression: J2kProgressionOrder,
) -> signinum_j2k_native::EncodeProgressionOrder {
    match progression {
        J2kProgressionOrder::Lrcp => signinum_j2k_native::EncodeProgressionOrder::Lrcp,
        J2kProgressionOrder::Rlcp => signinum_j2k_native::EncodeProgressionOrder::Rlcp,
        J2kProgressionOrder::Rpcl => signinum_j2k_native::EncodeProgressionOrder::Rpcl,
        J2kProgressionOrder::Pcrl => signinum_j2k_native::EncodeProgressionOrder::Pcrl,
        J2kProgressionOrder::Cprl => signinum_j2k_native::EncodeProgressionOrder::Cprl,
    }
}

/// Convert public irreversible quantization sub-band scales into native scales.
pub const fn quantization_scales_to_native(
    scales: IrreversibleQuantizationSubbandScales,
) -> signinum_j2k_native::IrreversibleQuantizationSubbandScales {
    signinum_j2k_native::IrreversibleQuantizationSubbandScales {
        low_low: scales.low_low,
        high_low: scales.high_low,
        low_high: scales.low_high,
        high_high: scales.high_high,
    }
}

/// Convert a public encode-stage dispatch report into the native report type.
pub const fn dispatch_report_to_native(
    report: J2kEncodeDispatchReport,
) -> signinum_j2k_native::J2kEncodeDispatchReport {
    signinum_j2k_native::J2kEncodeDispatchReport {
        deinterleave: report.deinterleave,
        forward_rct: report.forward_rct,
        forward_ict: report.forward_ict,
        forward_dwt53: report.forward_dwt53,
        forward_dwt97: report.forward_dwt97,
        quantize_subband: report.quantize_subband,
        tier1_code_block: report.tier1_code_block,
        ht_code_block: report.ht_code_block,
        packetization: report.packetization,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        dwt53_output_from_native, dwt53_output_to_native, encoded_ht_from_native,
        encoded_ht_to_native, encoded_j2k_from_native, encoded_j2k_to_native,
        packet_code_block_from_native, packet_code_block_to_native, packet_descriptor_from_native,
        packet_descriptor_to_native, packet_progression_from_native, packet_progression_to_native,
        progression_order_to_native, quantization_scales_to_native, style_from_native,
        style_to_native, subband_from_native, subband_to_native,
    };
    use crate::{
        EncodedHtJ2kCodeBlock, EncodedJ2kCodeBlock, IrreversibleQuantizationSubbandScales,
        J2kCodeBlockSegment, J2kCodeBlockStyle, J2kForwardDwt53Level, J2kForwardDwt53Output,
        J2kPacketizationBlockCodingMode, J2kPacketizationCodeBlock,
        J2kPacketizationPacketDescriptor, J2kPacketizationProgressionOrder, J2kProgressionOrder,
        J2kSubBandType,
    };
    use signinum_j2k_native as native;

    #[test]
    fn subband_and_progression_orders_round_trip_through_native() {
        for subband in [
            J2kSubBandType::LowLow,
            J2kSubBandType::HighLow,
            J2kSubBandType::LowHigh,
            J2kSubBandType::HighHigh,
        ] {
            assert_eq!(subband_from_native(subband_to_native(subband)), subband);
        }

        for progression in [
            J2kPacketizationProgressionOrder::Lrcp,
            J2kPacketizationProgressionOrder::Rlcp,
            J2kPacketizationProgressionOrder::Rpcl,
            J2kPacketizationProgressionOrder::Pcrl,
            J2kPacketizationProgressionOrder::Cprl,
        ] {
            assert_eq!(
                packet_progression_from_native(packet_progression_to_native(progression)),
                progression
            );
        }

        assert_eq!(
            progression_order_to_native(J2kProgressionOrder::Cprl),
            native::EncodeProgressionOrder::Cprl
        );
    }

    #[test]
    fn style_and_encoded_blocks_round_trip_through_native() {
        let style = J2kCodeBlockStyle {
            selective_arithmetic_coding_bypass: true,
            reset_context_probabilities: false,
            termination_on_each_pass: true,
            vertically_causal_context: true,
            segmentation_symbols: false,
        };
        let round_trip_style = style_from_native(style_to_native(style));
        assert_eq!(
            round_trip_style.selective_arithmetic_coding_bypass,
            style.selective_arithmetic_coding_bypass
        );
        assert_eq!(
            round_trip_style.reset_context_probabilities,
            style.reset_context_probabilities
        );
        assert_eq!(
            round_trip_style.termination_on_each_pass,
            style.termination_on_each_pass
        );
        assert_eq!(
            round_trip_style.vertically_causal_context,
            style.vertically_causal_context
        );
        assert_eq!(
            round_trip_style.segmentation_symbols,
            style.segmentation_symbols
        );

        let classic = EncodedJ2kCodeBlock {
            data: vec![1, 2, 3, 4],
            segments: vec![J2kCodeBlockSegment {
                data_offset: 1,
                data_length: 2,
                start_coding_pass: 3,
                end_coding_pass: 4,
                use_arithmetic: true,
            }],
            number_of_coding_passes: 5,
            missing_bit_planes: 6,
        };
        let classic_round_trip = encoded_j2k_from_native(encoded_j2k_to_native(classic.clone()));
        assert_eq!(classic_round_trip.data, classic.data);
        assert_eq!(classic_round_trip.segments, classic.segments);
        assert_eq!(
            classic_round_trip.number_of_coding_passes,
            classic.number_of_coding_passes
        );
        assert_eq!(
            classic_round_trip.missing_bit_planes,
            classic.missing_bit_planes
        );

        let ht = EncodedHtJ2kCodeBlock {
            data: vec![9, 8, 7],
            cleanup_length: 2,
            refinement_length: 1,
            num_coding_passes: 2,
            num_zero_bitplanes: 3,
        };
        let ht_round_trip = encoded_ht_from_native(encoded_ht_to_native(ht.clone()));
        assert_eq!(ht_round_trip.data, ht.data);
        assert_eq!(ht_round_trip.cleanup_length, ht.cleanup_length);
        assert_eq!(ht_round_trip.refinement_length, ht.refinement_length);
        assert_eq!(ht_round_trip.num_coding_passes, ht.num_coding_passes);
        assert_eq!(ht_round_trip.num_zero_bitplanes, ht.num_zero_bitplanes);
    }

    #[test]
    fn dwt_and_packetization_conversions_preserve_fields() {
        let dwt = J2kForwardDwt53Output {
            ll: vec![1.0, 2.0],
            ll_width: 2,
            ll_height: 1,
            levels: vec![J2kForwardDwt53Level {
                hl: vec![3.0],
                lh: vec![4.0],
                hh: vec![5.0],
                width: 3,
                height: 4,
                low_width: 2,
                low_height: 2,
                high_width: 1,
                high_height: 2,
            }],
        };
        let dwt_round_trip = dwt53_output_from_native(dwt53_output_to_native(dwt.clone()));
        assert_eq!(dwt_round_trip.ll, dwt.ll);
        assert_eq!(dwt_round_trip.ll_width, dwt.ll_width);
        assert_eq!(dwt_round_trip.ll_height, dwt.ll_height);
        assert_eq!(dwt_round_trip.levels[0].hl, dwt.levels[0].hl);
        assert_eq!(dwt_round_trip.levels[0].lh, dwt.levels[0].lh);
        assert_eq!(dwt_round_trip.levels[0].hh, dwt.levels[0].hh);
        assert_eq!(
            dwt_round_trip.levels[0].high_height,
            dwt.levels[0].high_height
        );

        let descriptor = J2kPacketizationPacketDescriptor {
            packet_index: 1,
            state_index: 2,
            layer: 3,
            resolution: 4,
            component: 5,
            precinct: 6,
        };
        assert_eq!(
            packet_descriptor_from_native(packet_descriptor_to_native(descriptor)),
            descriptor
        );

        let payload = [11, 12, 13];
        let block = J2kPacketizationCodeBlock {
            data: &payload,
            ht_cleanup_length: 1,
            ht_refinement_length: 2,
            num_coding_passes: 3,
            num_zero_bitplanes: 4,
            previously_included: true,
            l_block: 5,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        };
        assert_eq!(
            packet_code_block_from_native(packet_code_block_to_native(block)),
            block
        );
    }

    #[test]
    fn quantization_scales_convert_to_native_shape() {
        let native = quantization_scales_to_native(IrreversibleQuantizationSubbandScales {
            low_low: 1.0,
            high_low: 2.0,
            low_high: 3.0,
            high_high: 4.0,
        });
        assert_eq!(native.low_low.to_bits(), 1.0f32.to_bits());
        assert_eq!(native.high_low.to_bits(), 2.0f32.to_bits());
        assert_eq!(native.low_high.to_bits(), 3.0f32.to_bits());
        assert_eq!(native.high_high.to_bits(), 4.0f32.to_bits());
    }
}
