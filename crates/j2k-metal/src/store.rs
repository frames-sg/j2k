// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::compute;
use j2k_native::{HtCodeBlockDecoder, J2kStoreComponentJob, Result};
#[cfg(target_os = "macos")]
use metal::Buffer;

#[derive(Default)]
pub(crate) struct MetalStoreDecoder {
    #[cfg(target_os = "macos")]
    kernel_dispatches: usize,
    #[cfg(target_os = "macos")]
    captured_planes: Vec<Buffer>,
}

impl MetalStoreDecoder {
    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn kernel_dispatches(&self) -> usize {
        self.kernel_dispatches
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(crate) fn captured_plane_count(&self) -> usize {
        self.captured_planes.len()
    }

    #[cfg(target_os = "macos")]
    pub(crate) fn take_captured_planes(&mut self) -> Vec<Buffer> {
        core::mem::take(&mut self.captured_planes)
    }
}

impl HtCodeBlockDecoder for MetalStoreDecoder {
    fn decode_store_component(&mut self, job: J2kStoreComponentJob<'_>) -> Result<bool> {
        #[cfg(target_os = "macos")]
        if supports_metal_store(&job) {
            let captured = compute::decode_store_component_and_capture(job)
                .map_err(|_| j2k_native::DecodingError::CodeBlockDecodeFailure)?;
            self.captured_planes.push(captured);
            self.kernel_dispatches = self.kernel_dispatches.saturating_add(1);
            return Ok(true);
        }
        #[cfg(not(target_os = "macos"))]
        let _ = job;

        Ok(false)
    }
}

#[cfg(target_os = "macos")]
fn supports_metal_store(job: &J2kStoreComponentJob<'_>) -> bool {
    job.copy_width > 0
        && job.copy_height > 0
        && job.input_width > 0
        && job.output_width > 0
        && job.input.len() >= job.input_width as usize
        && job.output.len() >= job.output_width as usize
}

#[cfg(test)]
mod tests {
    use super::MetalStoreDecoder;
    #[cfg(target_os = "macos")]
    use crate::compute;
    use j2k_native::{
        encode, DecodeSettings, DecoderContext, EncodeOptions, HtCodeBlockDecoder, Image,
    };

    #[cfg(target_os = "macos")]
    fn should_run_metal_runtime() -> bool {
        j2k_test_support::metal_runtime_gate(module_path!())
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

    #[test]
    fn metal_store_decoder_matches_native_decode() {
        #[cfg(target_os = "macos")]
        if !should_run_metal_runtime() {
            return;
        }

        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut expected_context = DecoderContext::default();
        let expected = image
            .decode_components_with_context(&mut expected_context)
            .expect("native decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalStoreDecoder::default();
        let actual = image
            .decode_components_with_ht_decoder(&mut hooked_context, &mut decoder)
            .expect("hooked decode");

        assert_eq!(actual.dimensions(), expected.dimensions());
        assert_eq!(
            actual.planes()[0].samples(),
            expected.planes()[0].samples(),
            "Metal store output must match native decode"
        );
        #[cfg(target_os = "macos")]
        assert!(
            decoder.kernel_dispatches() > 0,
            "grayscale fixture must exercise the Metal store kernel"
        );
    }

    struct CpuOnlyCodeBlockDecoder;

    impl HtCodeBlockDecoder for CpuOnlyCodeBlockDecoder {}

    #[test]
    fn default_decoder_without_store_kernel_still_decodes() {
        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut decoder = CpuOnlyCodeBlockDecoder;
        let image_components = image
            .decode_components_with_ht_decoder(&mut context, &mut decoder)
            .expect("decode without store override");
        assert_eq!(image_components.dimensions(), (4, 4));
    }

    #[test]
    fn metal_store_decoder_matches_native_region_decode() {
        #[cfg(target_os = "macos")]
        if !should_run_metal_runtime() {
            return;
        }

        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let roi = (1, 1, 2, 2);
        let mut expected_context = DecoderContext::default();
        let expected = image
            .decode_region_components_with_ht_decoder(
                &mut expected_context,
                roi,
                &mut CpuOnlyCodeBlockDecoder,
            )
            .expect("native region decode");

        let mut hooked_context = DecoderContext::default();
        let mut decoder = MetalStoreDecoder::default();
        let actual = image
            .decode_region_components_with_ht_decoder(&mut hooked_context, roi, &mut decoder)
            .expect("hooked region decode");

        assert_eq!(actual.dimensions(), expected.dimensions());
        assert_eq!(
            actual.planes()[0].samples(),
            expected.planes()[0].samples(),
            "Metal region store output must match native region decode"
        );
        #[cfg(target_os = "macos")]
        assert!(
            decoder.kernel_dispatches() > 0,
            "region fixture must exercise the Metal store kernel"
        );
    }

    #[test]
    fn metal_store_decoder_captures_device_plane_for_full_decode() {
        #[cfg(target_os = "macos")]
        if !should_run_metal_runtime() {
            return;
        }

        let bytes = fixture_j2k_gray8();
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("image");
        let mut context = DecoderContext::default();
        let mut decoder = MetalStoreDecoder::default();
        let _decoded = image
            .decode_components_with_ht_decoder(&mut context, &mut decoder)
            .expect("hooked decode");
        #[cfg(target_os = "macos")]
        assert_eq!(
            decoder.captured_plane_count(),
            1,
            "full grayscale decode should capture one Metal-backed plane"
        );
    }

    #[test]
    fn captured_store_plane_owns_storage_after_caller_output_is_reused() {
        #[cfg(target_os = "macos")]
        {
            if !should_run_metal_runtime() {
                return;
            }

            let input = [1.0f32, 2.0, 3.0, 4.0];
            let input_width = u32::try_from(input.len()).expect("test input width fits u32");
            let mut caller_output = vec![0.0f32; input.len()];
            let captured =
                compute::decode_store_component_and_capture(j2k_native::J2kStoreComponentJob {
                    input: &input,
                    input_width,
                    source_x: 0,
                    source_y: 0,
                    copy_width: input_width,
                    copy_height: 1,
                    output: &mut caller_output,
                    output_width: input_width,
                    output_x: 0,
                    output_y: 0,
                    addend: 0.0,
                })
                .expect("dispatch Metal component store");
            assert_eq!(
                caller_output, input,
                "host readback must preserve output parity"
            );

            caller_output.fill(-99.0);
            let captured_after_mutation = compute::checked_buffer_slice::<f32>(
                &captured,
                input.len(),
                "captured store ownership regression",
            )
            .expect("read Metal-owned capture after caller mutation");
            assert_eq!(captured_after_mutation, input);

            drop(caller_output);
            let captured_after_drop = compute::checked_buffer_slice::<f32>(
                &captured,
                input.len(),
                "captured store ownership regression after caller drop",
            )
            .expect("read Metal-owned capture after caller drop");
            assert_eq!(captured_after_drop, input);
        }
    }
}
