// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{
    HtCodeBlockDecodeJob, HtCodeBlockDecoder, HtSubBandDecodeJob, J2kCodeBlockDecodeJob,
    J2kIdwtNormalization, J2kInverseMctJob, J2kSingleDecompositionIdwtJob, J2kStoreComponentJob,
    J2kSubBandDecodeJob,
};

use crate::{
    classic::MetalClassicBlockDecoder, ht::MetalHtBlockDecoder, idwt::MetalIdwtDecoder,
    mct::MetalMctDecoder, store::MetalStoreDecoder,
};

#[derive(Default)]
pub(super) struct MetalCodeBlockDecoder {
    classic: MetalClassicBlockDecoder,
    ht: MetalHtBlockDecoder,
    idwt: MetalIdwtDecoder,
    pub(super) mct: MetalMctDecoder,
    pub(super) store: MetalStoreDecoder,
}

impl HtCodeBlockDecoder for MetalCodeBlockDecoder {
    fn decode_j2k_sub_band(
        &mut self,
        job: J2kSubBandDecodeJob<'_>,
        output: &mut [f32],
    ) -> j2k_native::Result<bool> {
        self.classic.decode_j2k_sub_band(job, output)
    }

    fn decode_j2k_sub_band_with_midpoint(
        &mut self,
        job: J2kSubBandDecodeJob<'_>,
        output: &mut [f32],
        irreversible_midpoint: bool,
    ) -> j2k_native::Result<bool> {
        self.classic
            .decode_j2k_sub_band_with_midpoint(job, output, irreversible_midpoint)
    }

    fn decode_j2k_code_block(
        &mut self,
        job: J2kCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> j2k_native::Result<bool> {
        self.classic.decode_j2k_code_block(job, output)
    }

    fn decode_j2k_code_block_with_midpoint(
        &mut self,
        job: J2kCodeBlockDecodeJob<'_>,
        output: &mut [f32],
        irreversible_midpoint: bool,
    ) -> j2k_native::Result<bool> {
        self.classic
            .decode_j2k_code_block_with_midpoint(job, output, irreversible_midpoint)
    }

    fn decode_sub_band(
        &mut self,
        job: HtSubBandDecodeJob<'_>,
        output: &mut [f32],
    ) -> j2k_native::Result<bool> {
        self.ht.decode_sub_band(job, output)
    }

    fn decode_code_block(
        &mut self,
        job: HtCodeBlockDecodeJob<'_>,
        output: &mut [f32],
    ) -> j2k_native::Result<()> {
        self.ht.decode_code_block(job, output)
    }

    fn decode_single_decomposition_idwt(
        &mut self,
        job: J2kSingleDecompositionIdwtJob<'_>,
        output: &mut [f32],
    ) -> j2k_native::Result<bool> {
        self.idwt.decode_single_decomposition_idwt(job, output)
    }

    fn decode_single_decomposition_idwt_with_normalization(
        &mut self,
        job: J2kSingleDecompositionIdwtJob<'_>,
        normalization: J2kIdwtNormalization,
        output: &mut [f32],
    ) -> j2k_native::Result<bool> {
        self.idwt
            .decode_single_decomposition_idwt_with_normalization(job, normalization, output)
    }

    fn decode_inverse_mct(&mut self, job: J2kInverseMctJob<'_>) -> j2k_native::Result<bool> {
        self.mct.decode_inverse_mct(job)
    }

    fn decode_store_component(
        &mut self,
        job: J2kStoreComponentJob<'_>,
    ) -> j2k_native::Result<bool> {
        self.store.decode_store_component(job)
    }
}

#[cfg(test)]
mod tests {
    use super::MetalCodeBlockDecoder;
    use j2k_native::{DecodeSettings, DecoderContext, HtCodeBlockDecoder, Image};

    struct CpuOnlyCodeBlockDecoder;

    impl HtCodeBlockDecoder for CpuOnlyCodeBlockDecoder {}

    #[test]
    fn composite_decoder_retains_exact_openjpeg_irreversible_rgb_region_planes() {
        #[cfg(target_os = "macos")]
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
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
        let mut decoder = MetalCodeBlockDecoder::default();
        let actual = image
            .decode_region_components_with_ht_decoder(&mut hooked_context, roi, &mut decoder)
            .expect("composite Metal region decode");

        assert_eq!(actual.dimensions(), expected.dimensions());
        for (component, (actual_plane, expected_plane)) in
            actual.planes().iter().zip(expected.planes()).enumerate()
        {
            assert_eq!(
                actual_plane.samples(),
                expected_plane.samples(),
                "composite Metal component {component} must match native decode"
            );
        }

        #[cfg(target_os = "macos")]
        {
            let captured = decoder.mct.take_captured_planes();
            assert_eq!(captured.len(), actual.planes().len());
            for (component, (buffer, plane)) in captured.iter().zip(actual.planes()).enumerate() {
                let retained = crate::compute::checked_buffer_slice::<f32>(
                    buffer,
                    plane.samples().len(),
                    "composite MCT plane",
                )
                .expect("retained MCT plane readback");
                assert_eq!(
                    retained,
                    plane.samples(),
                    "retained Metal component {component} must match host plane"
                );
            }
        }
    }
}
