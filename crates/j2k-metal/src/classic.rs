// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::engine as compute;
#[cfg(target_os = "macos")]
use j2k_native::DecodingError;
use j2k_native::{
    decode_j2k_code_block_scalar, decode_j2k_sub_band_scalar, HtCodeBlockDecoder,
    J2kClassicReconstruction, J2kCodeBlockDecodeJob, J2kSubBandDecodeJob, Result,
};

#[derive(Default)]
pub(crate) struct MetalClassicBlockDecoder {
    blocks_decoded: usize,
    #[cfg(target_os = "macos")]
    kernel_dispatches: usize,
    sub_band_batches: usize,
    #[cfg(target_os = "macos")]
    batched_kernel_dispatches: usize,
}

const fn classic_irreversible_midpoint(reconstruction: J2kClassicReconstruction) -> Option<bool> {
    match reconstruction {
        J2kClassicReconstruction::ReversibleExact
        | J2kClassicReconstruction::ReversibleMidpoint => Some(false),
        J2kClassicReconstruction::IrreversibleMidpoint => Some(true),
        _ => None,
    }
}

impl MetalClassicBlockDecoder {
    #[cfg(test)]
    pub(crate) fn blocks_decoded(&self) -> usize {
        self.blocks_decoded
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn kernel_dispatches(&self) -> usize {
        self.kernel_dispatches
    }

    #[cfg(test)]
    pub(crate) fn sub_band_batches(&self) -> usize {
        self.sub_band_batches
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn batched_kernel_dispatches(&self) -> usize {
        self.batched_kernel_dispatches
    }

    fn decode_j2k_sub_band_inner(
        &mut self,
        job: J2kSubBandDecodeJob<'_>,
        output: &mut [f32],
        irreversible_midpoint: bool,
    ) -> Result<bool> {
        if job.jobs.len() <= 1 {
            return Ok(false);
        }

        self.sub_band_batches = self.sub_band_batches.saturating_add(1);
        #[cfg(target_os = "macos")]
        if job
            .jobs
            .iter()
            .all(|batch_job| supports_metal_classic_kernel(&batch_job.code_block))
        {
            compute::decode_classic_cleanup_sub_band_with_midpoint(
                job,
                output,
                irreversible_midpoint,
            )
            .map_err(metal_classic_sub_band_decode_error)?;
            self.batched_kernel_dispatches = self.batched_kernel_dispatches.saturating_add(1);
            return Ok(true);
        }

        if irreversible_midpoint {
            return Ok(false);
        }
        decode_j2k_sub_band_scalar(job, output)?;
        Ok(true)
    }

    fn decode_j2k_code_block_inner(
        &mut self,
        job: J2kCodeBlockDecodeJob<'_>,
        output: &mut [f32],
        irreversible_midpoint: bool,
    ) -> Result<bool> {
        self.blocks_decoded = self.blocks_decoded.saturating_add(1);
        #[cfg(target_os = "macos")]
        if supports_metal_classic_kernel(&job) {
            compute::decode_classic_cleanup_code_block_with_midpoint(
                job,
                output,
                irreversible_midpoint,
            )
            .map_err(metal_classic_code_block_decode_error)?;
            self.kernel_dispatches = self.kernel_dispatches.saturating_add(1);
            return Ok(true);
        }

        if irreversible_midpoint {
            return Ok(false);
        }
        decode_j2k_code_block_scalar(job, output)?;
        Ok(true)
    }
}

impl HtCodeBlockDecoder for MetalClassicBlockDecoder {
    fn decode_j2k_sub_band(
        &mut self,
        job: J2kSubBandDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<bool> {
        self.decode_j2k_sub_band_inner(job, output, false)
    }

    fn decode_j2k_sub_band_with_midpoint(
        &mut self,
        job: J2kSubBandDecodeJob<'_>,
        output: &mut [f32],
        irreversible_midpoint: bool,
    ) -> Result<bool> {
        self.decode_j2k_sub_band_inner(job, output, irreversible_midpoint)
    }

    fn decode_j2k_sub_band_with_reconstruction(
        &mut self,
        job: J2kSubBandDecodeJob<'_>,
        output: &mut [f32],
        reconstruction: J2kClassicReconstruction,
    ) -> Result<bool> {
        let Some(irreversible_midpoint) = classic_irreversible_midpoint(reconstruction) else {
            return Ok(false);
        };
        self.decode_j2k_sub_band_inner(job, output, irreversible_midpoint)
    }

    fn decode_j2k_code_block(
        &mut self,
        job: J2kCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> Result<bool> {
        self.decode_j2k_code_block_inner(job, output, false)
    }

    fn decode_j2k_code_block_with_midpoint(
        &mut self,
        job: J2kCodeBlockDecodeJob<'_>,
        output: &mut [f32],
        irreversible_midpoint: bool,
    ) -> Result<bool> {
        self.decode_j2k_code_block_inner(job, output, irreversible_midpoint)
    }

    fn decode_j2k_code_block_with_reconstruction(
        &mut self,
        job: J2kCodeBlockDecodeJob<'_>,
        output: &mut [f32],
        reconstruction: J2kClassicReconstruction,
    ) -> Result<bool> {
        let Some(irreversible_midpoint) = classic_irreversible_midpoint(reconstruction) else {
            return Ok(false);
        };
        self.decode_j2k_code_block_inner(job, output, irreversible_midpoint)
    }
}

#[cfg(target_os = "macos")]
fn metal_classic_sub_band_decode_error(_error: crate::Error) -> j2k_native::DecodeError {
    DecodingError::CodeBlockDecodeFailureWithContext("Metal classic sub-band decode kernel failed")
        .into()
}

#[cfg(target_os = "macos")]
fn metal_classic_code_block_decode_error(_error: crate::Error) -> j2k_native::DecodeError {
    DecodingError::CodeBlockDecodeFailureWithContext(
        "Metal classic code-block decode kernel failed",
    )
    .into()
}

#[cfg(target_os = "macos")]
fn supports_metal_classic_kernel(job: &J2kCodeBlockDecodeJob<'_>) -> bool {
    if job.width == 0 || job.height == 0 {
        return false;
    }
    if job.width > 64 || job.height > 64 {
        return false;
    }
    if job.output_stride < job.width as usize {
        return false;
    }
    if job.number_of_coding_passes == 0 {
        return false;
    }
    if job.data.is_empty() {
        return false;
    }
    let Some(coded_bitplanes) = job.total_bitplanes.checked_add(job.roi_shift) else {
        return false;
    };
    if job.total_bitplanes == 0 || coded_bitplanes > 31 || job.missing_bit_planes >= coded_bitplanes
    {
        return false;
    }
    let bitplanes = coded_bitplanes - job.missing_bit_planes;
    if bitplanes == 0 {
        return false;
    }
    let max_coding_passes = 1 + 3 * (bitplanes - 1);
    if job.number_of_coding_passes > max_coding_passes {
        return false;
    }
    if job.segments.is_empty() {
        return false;
    }

    let uses_bypass = job.style.selective_arithmetic_coding_bypass;
    let mut expected_start = 0u8;
    let mut expected_offset = 0usize;
    for segment in job.segments {
        if segment.start_coding_pass != expected_start
            || segment.start_coding_pass > segment.end_coding_pass
        {
            return false;
        }
        if uses_bypass {
            let expected_arithmetic =
                segment.start_coding_pass <= 9 || segment.start_coding_pass % 3 == 0;
            if segment.use_arithmetic != expected_arithmetic {
                return false;
            }
            if !segment.use_arithmetic {
                if segment.start_coding_pass % 3 != 1 {
                    return false;
                }
                if segment
                    .end_coding_pass
                    .saturating_sub(segment.start_coding_pass)
                    > 2
                {
                    return false;
                }
                if (segment.start_coding_pass..segment.end_coding_pass).any(|pass| pass % 3 == 0) {
                    return false;
                }
            }
        } else if !segment.use_arithmetic {
            return false;
        }
        let Some(data_start) = usize::try_from(segment.data_offset).ok() else {
            return false;
        };
        let Some(data_len) = usize::try_from(segment.data_length).ok() else {
            return false;
        };
        let Some(data_end) = data_start.checked_add(data_len) else {
            return false;
        };
        if data_start != expected_offset || data_end > job.data.len() {
            return false;
        }
        expected_offset = data_end;
        expected_start = segment.end_coding_pass;
    }
    expected_start == job.number_of_coding_passes && expected_offset == job.data.len()
}

#[cfg(all(test, target_os = "macos"))]
mod error_context_tests {
    #[test]
    fn metal_classic_decode_error_includes_kernel_context() {
        let error = super::metal_classic_code_block_decode_error(crate::Error::MetalKernel {
            message: "detail=4".to_string(),
        });

        assert!(
            error
                .to_string()
                .contains("Metal classic code-block decode kernel failed"),
            "Metal decode collapse must preserve kernel context, got: {error}"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{classic_irreversible_midpoint, MetalClassicBlockDecoder};
    #[cfg(target_os = "macos")]
    use crate::engine as compute;
    #[cfg(target_os = "macos")]
    use j2k_native::{
        decode_j2k_code_block_scalar, encode_j2k_code_block_scalar_with_style,
        J2kCodeBlockBatchJob, J2kCodeBlockDecodeJob, J2kCodeBlockSegment, J2kCodeBlockStyle,
        J2kSubBandDecodeJob, J2kSubBandType,
    };
    use j2k_native::{
        encode, ColorSpace, DecodeSettings, DecoderContext, EncodeOptions, HtCodeBlockDecoder,
        Image, J2kClassicReconstruction,
    };

    #[test]
    fn explicit_classic_reconstruction_modes_map_without_guessing() {
        assert_eq!(
            classic_irreversible_midpoint(J2kClassicReconstruction::ReversibleExact),
            Some(false)
        );
        assert_eq!(
            classic_irreversible_midpoint(J2kClassicReconstruction::ReversibleMidpoint),
            Some(false)
        );
        assert_eq!(
            classic_irreversible_midpoint(J2kClassicReconstruction::IrreversibleMidpoint),
            Some(true)
        );
    }

    #[cfg(target_os = "macos")]
    fn should_run_metal_runtime() -> bool {
        j2k_test_support::metal_runtime_gate(module_path!())
    }

    struct CpuOnlyCodeBlockDecoder;

    impl HtCodeBlockDecoder for CpuOnlyCodeBlockDecoder {}

    #[cfg(target_os = "macos")]
    #[derive(Clone)]
    struct OwnedClassicJob {
        data: Vec<u8>,
        segments: Vec<j2k_native::J2kCodeBlockSegment>,
        width: u32,
        height: u32,
        output_stride: usize,
        missing_bit_planes: u8,
        number_of_coding_passes: u8,
        total_bitplanes: u8,
        roi_shift: u8,
        sub_band_type: j2k_native::J2kSubBandType,
        style: j2k_native::J2kCodeBlockStyle,
        strict: bool,
        dequantization_step: f32,
    }

    #[cfg(target_os = "macos")]
    impl OwnedClassicJob {
        fn as_job(&self) -> J2kCodeBlockDecodeJob<'_> {
            J2kCodeBlockDecodeJob {
                data: &self.data,
                segments: &self.segments,
                width: self.width,
                height: self.height,
                output_stride: self.output_stride,
                missing_bit_planes: self.missing_bit_planes,
                number_of_coding_passes: self.number_of_coding_passes,
                total_bitplanes: self.total_bitplanes,
                roi_shift: self.roi_shift,
                sub_band_type: self.sub_band_type,
                style: self.style,
                strict: self.strict,
                dequantization_step: self.dequantization_step,
            }
        }

        fn output_len(&self) -> usize {
            self.output_stride * self.height as usize
        }
    }

    #[cfg(target_os = "macos")]
    fn split_job_into_two_valid_segments(job: &OwnedClassicJob) -> OwnedClassicJob {
        let mut baseline = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut baseline).expect("baseline decode");

        for split_pass in 1..job.number_of_coding_passes {
            for split_offset in 1..job.data.len() {
                let mut candidate = job.clone();
                let split_offset_u32 =
                    u32::try_from(split_offset).expect("split offset fits in u32");
                let remaining_len_u32 = u32::try_from(job.data.len() - split_offset)
                    .expect("remaining segment length fits in u32");
                candidate.segments = vec![
                    J2kCodeBlockSegment {
                        data_offset: 0,
                        data_length: split_offset_u32,
                        start_coding_pass: 0,
                        end_coding_pass: split_pass,
                        use_arithmetic: true,
                    },
                    J2kCodeBlockSegment {
                        data_offset: split_offset_u32,
                        data_length: remaining_len_u32,
                        start_coding_pass: split_pass,
                        end_coding_pass: job.number_of_coding_passes,
                        use_arithmetic: true,
                    },
                ];
                let mut candidate_output = vec![0.0f32; candidate.output_len()];
                if decode_j2k_code_block_scalar(candidate.as_job(), &mut candidate_output).is_ok()
                    && candidate_output == baseline
                {
                    return candidate;
                }
            }
        }

        panic!("expected to find a valid two-segment classic codeblock split");
    }

    #[cfg(target_os = "macos")]
    #[derive(Default)]
    struct CaptureFirstClassicJob {
        first: Option<OwnedClassicJob>,
    }

    #[cfg(target_os = "macos")]
    impl HtCodeBlockDecoder for CaptureFirstClassicJob {
        fn decode_j2k_code_block(
            &mut self,
            job: J2kCodeBlockDecodeJob<'_>,
            output: &mut [f32],
        ) -> j2k_native::Result<bool> {
            if self.first.is_none() {
                self.first = Some(OwnedClassicJob {
                    data: job.data.to_vec(),
                    segments: job.segments.to_vec(),
                    width: job.width,
                    height: job.height,
                    output_stride: job.output_stride,
                    missing_bit_planes: job.missing_bit_planes,
                    number_of_coding_passes: job.number_of_coding_passes,
                    total_bitplanes: job.total_bitplanes,
                    roi_shift: job.roi_shift,
                    sub_band_type: job.sub_band_type,
                    style: job.style,
                    strict: job.strict,
                    dequantization_step: job.dequantization_step,
                });
            }
            decode_j2k_code_block_scalar(job, output)?;
            Ok(true)
        }
    }

    fn fixture_j2k_gray8() -> Vec<u8> {
        let pixels: Vec<u8> = (0..16).collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        };
        encode(&pixels, 4, 4, 1, 8, false, &options).expect("encode classic gray8")
    }

    fn fixture_j2k_gray1_cleanup_only() -> Vec<u8> {
        let pixels = vec![0u8, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0];
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            ..EncodeOptions::default()
        };
        encode(&pixels, 4, 4, 1, 1, false, &options).expect("encode classic gray1")
    }

    fn fixture_j2k_gray1_cleanup_only_multi_block() -> Vec<u8> {
        let pixels = vec![
            0u8, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0,
            1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1,
            0, 1, 0, 1, 0, 1, 0,
        ];
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            code_block_width_exp: 0,
            code_block_height_exp: 0,
            ..EncodeOptions::default()
        };
        encode(&pixels, 8, 8, 1, 1, false, &options).expect("encode classic gray1 multi-block")
    }

    #[cfg(target_os = "macos")]
    fn fixture_j2k_gray8_tall() -> Vec<u8> {
        let pixels: Vec<u8> = (0u16..32u16)
            .map(|value| u8::try_from((value * 17) & 0xFF).expect("fixture sample fits in u8"))
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            ..EncodeOptions::default()
        };
        encode(&pixels, 4, 8, 1, 8, false, &options).expect("encode classic tall gray8")
    }

    #[cfg(target_os = "macos")]
    fn fixture_j2k_gray16_bypass() -> Vec<u8> {
        let pixels: Vec<u8> = (0u16..16u16)
            .flat_map(|index| ((index * 0x111) | 0x123).to_le_bytes())
            .collect();
        let options = EncodeOptions {
            reversible: true,
            num_decomposition_levels: 0,
            ..EncodeOptions::default()
        };
        encode(&pixels, 4, 4, 1, 16, false, &options).expect("encode classic gray16 bypass")
    }

    #[cfg(target_os = "macos")]
    fn integral_coefficients(values: &[f32]) -> Vec<i32> {
        values
            .iter()
            .map(|value| {
                assert!(value.is_finite(), "coefficient must be finite");
                let rounded = value.round();
                assert!(
                    (rounded - *value).abs() <= f32::EPSILON,
                    "reversible classic coefficients must be integral"
                );
                format!("{rounded:.0}")
                    .parse::<i32>()
                    .expect("integral coefficient fits in i32")
            })
            .collect()
    }

    #[test]
    fn metal_classic_decoder_matches_native_decode() {
        #[cfg(target_os = "macos")]
        if !should_run_metal_runtime() {
            return;
        }

        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");

        let mut baseline_context = DecoderContext::default();
        let baseline = image
            .decode_components_with_context(&mut baseline_context)
            .expect("baseline decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalClassicBlockDecoder::default();
        let hooked = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert!(
            decoder.blocks_decoded() > 0,
            "classic J2K hook must be used"
        );
        #[cfg(target_os = "macos")]
        assert!(decoder.kernel_dispatches() > 0);
        assert_eq!(hooked.dimensions(), baseline.dimensions());
        assert!(matches!(hooked.color_space(), ColorSpace::Gray));
        assert_eq!(
            core::mem::discriminant(hooked.color_space()),
            core::mem::discriminant(baseline.color_space())
        );
        assert_eq!(hooked.has_alpha(), baseline.has_alpha());
        assert_eq!(hooked.planes().len(), baseline.planes().len());

        for (hooked_plane, baseline_plane) in hooked.planes().iter().zip(baseline.planes()) {
            assert_eq!(hooked_plane.bit_depth(), baseline_plane.bit_depth());
            assert_eq!(hooked_plane.samples(), baseline_plane.samples());
        }
    }

    #[test]
    fn metal_classic_decoder_matches_native_region_for_openjpeg_irreversible_rgb() {
        #[cfg(target_os = "macos")]
        if !should_run_metal_runtime() {
            return;
        }

        let image = Image::new(
            j2k_test_support::OPENJPEG_IRREVERSIBLE_RGB8_8X8,
            &DecodeSettings::default(),
        )
        .expect("image");
        let roi = (2, 2, 4, 4);
        let mut expected_context = DecoderContext::default();
        let expected = image
            .decode_region_components_with_ht_decoder(
                &mut expected_context,
                roi,
                &mut CpuOnlyCodeBlockDecoder,
            )
            .expect("native region decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalClassicBlockDecoder::default();
        let actual = image
            .decode_region_components_with_ht_decoder(&mut hooked_context, roi, &mut decoder)
            .expect("Metal classic region decode");

        assert_eq!(actual.dimensions(), expected.dimensions());
        for (component, (actual_plane, expected_plane)) in
            actual.planes().iter().zip(expected.planes()).enumerate()
        {
            assert_eq!(
                actual_plane.samples(),
                expected_plane.samples(),
                "Metal classic component {component} must match native region decode"
            );
        }
        assert!(decoder.blocks_decoded() > 0);
        #[cfg(target_os = "macos")]
        assert!(
            decoder.kernel_dispatches() + decoder.batched_kernel_dispatches() > 0,
            "OpenJPEG RGB region must exercise a Metal classic kernel"
        );
    }

    #[test]
    fn metal_classic_decoder_batches_multi_block_subbands() {
        #[cfg(target_os = "macos")]
        if !should_run_metal_runtime() {
            return;
        }

        let bytes = fixture_j2k_gray1_cleanup_only_multi_block();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");

        let mut baseline_context = DecoderContext::default();
        let baseline = image
            .decode_components_with_context(&mut baseline_context)
            .expect("baseline decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalClassicBlockDecoder::default();
        let hooked = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert!(
            decoder.sub_band_batches() > 0,
            "multi-block classic fixture must exercise the batched classic path"
        );
        #[cfg(target_os = "macos")]
        assert!(decoder.batched_kernel_dispatches() > 0);
        assert_eq!(hooked.dimensions(), baseline.dimensions());
        assert_eq!(hooked.planes().len(), baseline.planes().len());

        for (hooked_plane, baseline_plane) in hooked.planes().iter().zip(baseline.planes()) {
            assert_eq!(hooked_plane.bit_depth(), baseline_plane.bit_depth());
            assert_eq!(hooked_plane.samples(), baseline_plane.samples());
        }
    }

    #[test]
    fn metal_classic_decoder_matches_native_decode_for_cleanup_only_fixture() {
        #[cfg(target_os = "macos")]
        if !should_run_metal_runtime() {
            return;
        }

        let bytes = fixture_j2k_gray1_cleanup_only();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");

        let mut baseline_context = DecoderContext::default();
        let baseline = image
            .decode_components_with_context(&mut baseline_context)
            .expect("baseline decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalClassicBlockDecoder::default();
        let hooked = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert!(decoder.blocks_decoded() > 0);
        #[cfg(target_os = "macos")]
        assert!(decoder.kernel_dispatches() > 0);

        for (hooked_plane, baseline_plane) in hooked.planes().iter().zip(baseline.planes()) {
            assert_eq!(hooked_plane.samples(), baseline_plane.samples());
        }
    }

    #[test]
    fn metal_classic_decoder_matches_native_for_early_stopped_reversible_blocks() {
        #[cfg(target_os = "macos")]
        if !should_run_metal_runtime() {
            return;
        }

        // Both fixtures need the T.800 Equation E-8 midpoint (r = 1/2). Their
        // one-block sub-bands reach the per-block kernel; the rate-limited
        // fixture's four-block sub-bands reach the batched kernel.
        for (label, bytes, batched) in [
            (
                "lossless",
                j2k_test_support::REVERSIBLE_EARLY_STOP_LOSSLESS,
                false,
            ),
            (
                "rate-limited",
                j2k_test_support::REVERSIBLE_EARLY_STOP_RATE_LIMITED,
                true,
            ),
        ] {
            let image = Image::new(bytes, &DecodeSettings::default()).expect("image");

            let mut baseline_context = DecoderContext::default();
            let baseline = image
                .decode_components_with_context(&mut baseline_context)
                .expect("baseline decode");

            let mut hooked_context = DecoderContext::default();
            let mut decoder = MetalClassicBlockDecoder::default();
            let hooked = image
                .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
                .expect("hooked decode");

            #[cfg(target_os = "macos")]
            {
                assert!(decoder.kernel_dispatches() > 0, "{label}");
                assert_eq!(decoder.batched_kernel_dispatches() > 0, batched, "{label}");
            }
            #[cfg(not(target_os = "macos"))]
            let _ = batched;
            assert_eq!(
                hooked.planes()[0].samples(),
                baseline.planes()[0].samples(),
                "{label}"
            );
        }
    }

    #[cfg(target_os = "macos")]
    fn patterned_coefficients(count: u32) -> Vec<i32> {
        let mut state = 0x9e37_79b9_u32;
        (0..count)
            .map(|index| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                if index % 11 == 0 {
                    0
                } else {
                    i32::try_from((state >> 16) & 0x1ff).expect("masked coefficient") - 255
                }
            })
            .collect()
    }

    /// The pass count, segments and data of `encoded` as if its encoder had
    /// stopped after `passes` passes.
    #[cfg(target_os = "macos")]
    fn truncate_coding_passes(
        encoded: &j2k_native::EncodedJ2kCodeBlock,
        passes: u8,
    ) -> (u8, Vec<J2kCodeBlockSegment>, &[u8]) {
        let segments = encoded
            .segments
            .iter()
            .filter(|segment| segment.start_coding_pass < passes)
            .map(|segment| J2kCodeBlockSegment {
                end_coding_pass: segment.end_coding_pass.min(passes),
                ..*segment
            })
            .collect::<Vec<_>>();
        let last = segments.last().expect("a segment holds the first pass");
        let data_len = (last.data_offset + last.data_length) as usize;
        (passes, segments, &encoded.data[..data_len])
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_kernels_match_scalar_at_every_reversible_pass_count() {
        if !should_run_metal_runtime() {
            return;
        }

        // Stopping a block after each of its coding passes ends it on every
        // kind of pass, so reversible coefficients take the T.800 Equation
        // E-8 midpoint below each bit-plane on the per-block and batched
        // kernels. The truncations sit side by side in one sub-band.
        let (width, height, total_bitplanes) = (13_u32, 9_u32, 10_u8);
        let coefficients = patterned_coefficients(width * height);
        let plain = J2kCodeBlockStyle {
            selective_arithmetic_coding_bypass: false,
            reset_context_probabilities: false,
            termination_on_each_pass: false,
            vertically_causal_context: false,
            segmentation_symbols: false,
        };
        for style in [
            plain,
            J2kCodeBlockStyle {
                selective_arithmetic_coding_bypass: true,
                ..plain
            },
            J2kCodeBlockStyle {
                termination_on_each_pass: true,
                reset_context_probabilities: true,
                ..plain
            },
        ] {
            let encoded = encode_j2k_code_block_scalar_with_style(
                &coefficients,
                width,
                height,
                J2kSubBandType::LowLow,
                total_bitplanes,
                style,
            )
            .expect("encode classic code block");
            let truncations = (1..=encoded.number_of_coding_passes)
                .map(|passes| truncate_coding_passes(&encoded, passes))
                .collect::<Vec<_>>();
            let sub_band_width = width * u32::from(encoded.number_of_coding_passes);
            let job = |passes: u8, segments, data, output_stride| J2kCodeBlockDecodeJob {
                data,
                segments,
                width,
                height,
                output_stride,
                missing_bit_planes: encoded.missing_bit_planes,
                number_of_coding_passes: passes,
                total_bitplanes,
                roi_shift: 0,
                sub_band_type: J2kSubBandType::LowLow,
                style,
                strict: true,
                dequantization_step: 1.0,
            };

            let mut expected = vec![0.0f32; (sub_band_width * height) as usize];
            let mut batch = Vec::with_capacity(truncations.len());
            for (index, (passes, segments, data)) in truncations.iter().enumerate() {
                let output_x = u32::try_from(index).expect("truncation index") * width;
                let block = job(*passes, segments, data, width as usize);
                let mut scalar = vec![0.0f32; (width * height) as usize];
                decode_j2k_code_block_scalar(block, &mut scalar).expect("scalar decode");
                let mut metal = vec![0.0f32; scalar.len()];
                compute::decode_classic_cleanup_code_block_with_midpoint(block, &mut metal, false)
                    .expect("Metal per-block decode");
                assert_eq!(metal, scalar, "{style:?} after {passes} passes");

                let in_sub_band = job(*passes, segments, data, sub_band_width as usize);
                decode_j2k_code_block_scalar(in_sub_band, &mut expected[output_x as usize..])
                    .expect("scalar decode into the sub-band");
                batch.push(J2kCodeBlockBatchJob {
                    output_x,
                    output_y: 0,
                    code_block: in_sub_band,
                });
            }
            let mut actual = vec![0.0f32; expected.len()];
            compute::decode_classic_cleanup_sub_band_with_midpoint(
                J2kSubBandDecodeJob {
                    width: sub_band_width,
                    height,
                    jobs: &batch,
                },
                &mut actual,
                false,
            )
            .expect("Metal batched decode");
            assert_eq!(actual, expected, "{style:?} batched");
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_kernel_matches_scalar_for_captured_cleanup_job() {
        if !should_run_metal_runtime() {
            return;
        }

        let bytes = fixture_j2k_gray1_cleanup_only();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureFirstClassicJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture classic job");

        let job = capture.first.expect("captured classic job");
        assert_eq!(job.number_of_coding_passes, 1);
        assert!(super::supports_metal_classic_kernel(&job.as_job()));

        let mut expected = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut expected).expect("scalar decode");
        let mut actual = vec![0.0f32; job.output_len()];
        compute::decode_classic_cleanup_code_block_with_midpoint(job.as_job(), &mut actual, false)
            .expect("metal decode");
        assert_eq!(actual, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_kernel_matches_scalar_for_cleanup_job_with_segmentation_symbols() {
        if !should_run_metal_runtime() {
            return;
        }

        let bytes = fixture_j2k_gray1_cleanup_only();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureFirstClassicJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture classic job");

        let mut job = capture.first.expect("captured classic job");
        assert_eq!(job.number_of_coding_passes, 1);
        let mut original_coefficients = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut original_coefficients)
            .expect("decode original cleanup job");
        let original_coefficients = integral_coefficients(&original_coefficients);
        let encoded = j2k_native::encode_j2k_code_block_scalar_with_style(
            &original_coefficients,
            job.width,
            job.height,
            job.sub_band_type,
            job.total_bitplanes,
            j2k_native::J2kCodeBlockStyle {
                segmentation_symbols: true,
                ..job.style
            },
        )
        .expect("encode segmentation-symbol cleanup job");
        job.style.segmentation_symbols = true;
        job.data = encoded.data;
        job.segments = encoded.segments;
        job.number_of_coding_passes = encoded.number_of_coding_passes;
        job.missing_bit_planes = encoded.missing_bit_planes;

        assert!(super::supports_metal_classic_kernel(&job.as_job()));

        let mut expected = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut expected).expect("scalar decode");
        let mut actual = vec![0.0f32; job.output_len()];
        compute::decode_classic_cleanup_code_block_with_midpoint(job.as_job(), &mut actual, false)
            .expect("metal decode");
        assert_eq!(actual, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_kernel_matches_scalar_for_captured_multi_pass_job() {
        if !should_run_metal_runtime() {
            return;
        }

        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureFirstClassicJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture classic job");

        let job = capture.first.expect("captured classic job");
        assert!(job.number_of_coding_passes > 1);
        assert!(super::supports_metal_classic_kernel(&job.as_job()));

        let mut expected = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut expected).expect("scalar decode");
        let mut actual = vec![0.0f32; job.output_len()];
        compute::decode_classic_cleanup_code_block_with_midpoint(job.as_job(), &mut actual, false)
            .expect("metal decode");
        assert_eq!(actual, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_kernel_matches_scalar_for_multi_pass_job_with_reset_contexts() {
        if !should_run_metal_runtime() {
            return;
        }

        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureFirstClassicJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture classic job");

        let mut job = capture.first.expect("captured classic job");
        assert!(job.number_of_coding_passes > 1);
        let mut original_coefficients = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut original_coefficients)
            .expect("decode original multi-pass job");
        let original_coefficients = integral_coefficients(&original_coefficients);
        let encoded = j2k_native::encode_j2k_code_block_scalar_with_style(
            &original_coefficients,
            job.width,
            job.height,
            job.sub_band_type,
            job.total_bitplanes,
            j2k_native::J2kCodeBlockStyle {
                reset_context_probabilities: true,
                ..job.style
            },
        )
        .expect("encode reset-context classic job");
        job.style.reset_context_probabilities = true;
        job.data = encoded.data;
        job.segments = encoded.segments;
        job.number_of_coding_passes = encoded.number_of_coding_passes;
        job.missing_bit_planes = encoded.missing_bit_planes;

        assert!(super::supports_metal_classic_kernel(&job.as_job()));

        let mut expected = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut expected).expect("scalar decode");
        let mut actual = vec![0.0f32; job.output_len()];
        compute::decode_classic_cleanup_code_block_with_midpoint(job.as_job(), &mut actual, false)
            .expect("metal decode");
        assert_eq!(actual, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_kernel_matches_scalar_for_tall_job_with_vertically_causal_context() {
        if !should_run_metal_runtime() {
            return;
        }

        let bytes = fixture_j2k_gray8_tall();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureFirstClassicJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture classic job");

        let mut job = capture.first.expect("captured classic job");
        assert!(job.height > 4, "fixture must cross a stripe boundary");
        let mut original_coefficients = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut original_coefficients)
            .expect("decode original tall job");
        let original_coefficients = integral_coefficients(&original_coefficients);
        let encoded = j2k_native::encode_j2k_code_block_scalar_with_style(
            &original_coefficients,
            job.width,
            job.height,
            job.sub_band_type,
            job.total_bitplanes,
            j2k_native::J2kCodeBlockStyle {
                vertically_causal_context: true,
                ..job.style
            },
        )
        .expect("encode vertically-causal classic job");
        job.style.vertically_causal_context = true;
        job.data = encoded.data;
        job.segments = encoded.segments;
        job.number_of_coding_passes = encoded.number_of_coding_passes;
        job.missing_bit_planes = encoded.missing_bit_planes;

        assert!(super::supports_metal_classic_kernel(&job.as_job()));

        let mut expected = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut expected).expect("scalar decode");
        let mut actual = vec![0.0f32; job.output_len()];
        compute::decode_classic_cleanup_code_block_with_midpoint(job.as_job(), &mut actual, false)
            .expect("metal decode");
        assert_eq!(actual, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_kernel_matches_scalar_for_multi_pass_job_with_termination_on_each_pass() {
        if !should_run_metal_runtime() {
            return;
        }

        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureFirstClassicJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture classic job");

        let mut job = capture.first.expect("captured classic job");
        assert!(job.number_of_coding_passes > 1);
        let mut original_coefficients = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut original_coefficients)
            .expect("decode original multi-pass job");
        let original_coefficients = integral_coefficients(&original_coefficients);
        let encoded = j2k_native::encode_j2k_code_block_scalar_with_style(
            &original_coefficients,
            job.width,
            job.height,
            job.sub_band_type,
            job.total_bitplanes,
            j2k_native::J2kCodeBlockStyle {
                termination_on_each_pass: true,
                ..job.style
            },
        )
        .expect("encode terminated classic job");
        job.style.termination_on_each_pass = true;
        job.data = encoded.data;
        job.segments = encoded.segments;
        job.number_of_coding_passes = encoded.number_of_coding_passes;
        job.missing_bit_planes = encoded.missing_bit_planes;

        assert!(
            job.segments.len() > 1,
            "termination must produce multiple segments"
        );
        assert!(super::supports_metal_classic_kernel(&job.as_job()));

        let mut expected = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut expected).expect("scalar decode");
        let mut actual = vec![0.0f32; job.output_len()];
        compute::decode_classic_cleanup_code_block_with_midpoint(job.as_job(), &mut actual, false)
            .expect("metal decode");
        assert_eq!(actual, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_kernel_matches_scalar_for_bypass_job() {
        if !should_run_metal_runtime() {
            return;
        }

        let bytes = fixture_j2k_gray16_bypass();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureFirstClassicJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture classic job");

        let mut job = capture.first.expect("captured classic job");
        assert!(
            job.number_of_coding_passes > 10,
            "fixture must reach bypass-eligible coding passes"
        );
        let mut original_coefficients = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut original_coefficients)
            .expect("decode original bypass job");
        let original_coefficients = integral_coefficients(&original_coefficients);
        let encoded = j2k_native::encode_j2k_code_block_scalar_with_style(
            &original_coefficients,
            job.width,
            job.height,
            job.sub_band_type,
            job.total_bitplanes,
            j2k_native::J2kCodeBlockStyle {
                selective_arithmetic_coding_bypass: true,
                ..job.style
            },
        )
        .expect("encode bypass classic job");
        job.style.selective_arithmetic_coding_bypass = true;
        job.data = encoded.data;
        job.segments = encoded.segments;
        job.number_of_coding_passes = encoded.number_of_coding_passes;
        job.missing_bit_planes = encoded.missing_bit_planes;

        assert!(
            job.segments.iter().any(|segment| !segment.use_arithmetic),
            "bypass job must contain non-arithmetic segments"
        );
        assert!(super::supports_metal_classic_kernel(&job.as_job()));

        let mut expected = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut expected).expect("scalar decode");
        let mut actual = vec![0.0f32; job.output_len()];
        compute::decode_classic_cleanup_code_block_with_midpoint(job.as_job(), &mut actual, false)
            .expect("metal decode");
        assert_eq!(actual, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_kernel_accepts_zero_length_prefix_segment() {
        if !should_run_metal_runtime() {
            return;
        }

        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureFirstClassicJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture classic job");

        let mut job = capture.first.expect("captured classic job");
        let real = job.segments[0];
        job.segments = vec![
            J2kCodeBlockSegment {
                data_offset: 0,
                data_length: 0,
                start_coding_pass: 0,
                end_coding_pass: 0,
                use_arithmetic: true,
            },
            real,
        ];

        assert!(super::supports_metal_classic_kernel(&job.as_job()));

        let mut expected = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut expected).expect("scalar decode");
        let mut actual = vec![0.0f32; job.output_len()];
        compute::decode_classic_cleanup_code_block_with_midpoint(job.as_job(), &mut actual, false)
            .expect("metal decode");
        assert_eq!(actual, expected);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_classic_kernel_matches_scalar_for_valid_two_segment_job() {
        if !should_run_metal_runtime() {
            return;
        }

        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut capture = CaptureFirstClassicJob::default();
        image
            .decode_components_with_ht_decoder(&mut context, &mut capture)
            .expect("capture classic job");

        let captured = capture.first.expect("captured classic job");
        let job = split_job_into_two_valid_segments(&captured);
        assert!(job.segments.len() > 1);
        assert!(super::supports_metal_classic_kernel(&job.as_job()));

        let mut expected = vec![0.0f32; job.output_len()];
        decode_j2k_code_block_scalar(job.as_job(), &mut expected).expect("scalar decode");
        let mut actual = vec![0.0f32; job.output_len()];
        compute::decode_classic_cleanup_code_block_with_midpoint(job.as_job(), &mut actual, false)
            .expect("metal decode");
        assert_eq!(actual, expected);
    }
}
