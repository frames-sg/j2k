// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(all(test, target_os = "macos"))]
use std::sync::Arc;

#[cfg(target_os = "macos")]
use j2k::DecodeRequest;
#[cfg(any(test, target_os = "macos"))]
use j2k::{BatchColor, BatchLayout};
use j2k::{
    BatchDecodeOptions, BatchGroupInfo, EncodedImage, IndexedBatchError, J2kDecodeWarning,
    PreparedBatch, PreparedBatchGroup, PreparedImage,
};
#[cfg(target_os = "macos")]
use j2k_core::DeviceSubmission;
#[cfg(any(test, target_os = "macos"))]
use j2k_core::PixelFormat;
use j2k_core::Rect;

#[cfg(target_os = "macos")]
use j2k_metal_support::{MetalImageDestination, MetalImageLayout, ResidentMetalImage};

#[cfg(target_os = "macos")]
use metal::{Buffer, DeviceRef};

use crate::{Error, MetalBackendSession, Surface};

mod contracts;
mod decoder;
#[cfg(all(test, target_os = "macos"))]
mod encoder_count_tests;
#[cfg(target_os = "macos")]
mod external;
#[cfg(target_os = "macos")]
mod plan_cache;
#[cfg(all(test, target_os = "macos"))]
mod queue_ordering_tests;
mod resident;
#[cfg(target_os = "macos")]
mod submission;

#[cfg(test)]
mod decoder_ownership_tests {
    #[test]
    fn persistent_decoder_does_not_layer_the_legacy_metal_session() {
        let decoder_source = include_str!("batch_decoder/decoder.rs");
        let legacy_session_type = ["Metal", "Session"].concat();

        assert!(
            !decoder_source.contains(&legacy_session_type),
            "MetalBatchDecoder must own only MetalBackendSession and its direct counter"
        );
        assert!(decoder_source.contains("pub(super) backend: MetalBackendSession"));
        assert!(decoder_source.contains("submission_count: u64"));
    }
}

#[cfg(target_os = "macos")]
pub use self::contracts::MetalResidentBatch;
pub use self::contracts::{
    MetalBatchDecodeResult, MetalBatchGroup, MetalBatchGroupCompletion, MetalBatchGroupError,
    MetalBatchGroupParts,
};
pub use self::decoder::MetalBatchDecoder;
#[cfg(target_os = "macos")]
pub use self::submission::{SubmittedMetalGroupDecodeInto, SubmittedMetalPreparedBatch};

#[cfg(any(test, target_os = "macos"))]
use self::contracts::validate_group_contract;
#[cfg(target_os = "macos")]
use self::plan_cache::{PreparedColorPlanCache, PreparedGrayPlanCache};
#[cfg(target_os = "macos")]
use self::submission::{
    allocate_codec_owned_group_destination, validate_codec_owned_resident_group,
    MetalResidentGroupMetadata, SubmittedMetalResidentGroup,
};
#[cfg(test)]
mod batch_contract_tests {
    use super::*;
    use j2k::{BatchAlpha, NativeSampleType};

    fn color_info(
        color: BatchColor,
        sample_type: NativeSampleType,
        signed: bool,
    ) -> BatchGroupInfo {
        BatchGroupInfo {
            dimensions: (2, 2),
            color,
            alpha: if color == BatchColor::Rgba {
                BatchAlpha::Straight
            } else {
                BatchAlpha::None
            },
            precision: if sample_type == NativeSampleType::U8 {
                8
            } else {
                12
            },
            signed,
            sample_type,
            layout: BatchLayout::Nchw,
            colorspace: j2k_core::Colorspace::Rgb,
            route: j2k::BatchCodecRoute::Htj2k,
            transform: j2k::BatchWaveletTransform::Reversible53,
            transfer_syntax: j2k_core::CompressedTransferSyntax::HtJpeg2000Lossless,
            payload_kind: j2k_core::CompressedPayloadKind::Jpeg2000Codestream,
        }
    }

    #[test]
    fn exact_color_contract_maps_all_native_rgb_and_rgba_formats() {
        assert_eq!(
            validate_group_contract(&color_info(BatchColor::Rgb, NativeSampleType::I16, true))
                .expect("signed RGB must have an exact native Metal format"),
            PixelFormat::RgbI16
        );
        for (sample_type, signed, expected) in [
            (NativeSampleType::U8, false, PixelFormat::Rgba8),
            (NativeSampleType::U16, false, PixelFormat::Rgba16),
            (NativeSampleType::I16, true, PixelFormat::RgbaI16),
        ] {
            assert_eq!(
                validate_group_contract(&color_info(BatchColor::Rgba, sample_type, signed))
                    .expect("RGBA must have an exact native Metal format"),
                expected
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn external_ht_gray_group_uses_one_stacked_component_graph() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }

        let bytes = j2k_test_support::openhtj2k_refinement_fixture();
        let options = BatchDecodeOptions {
            settings: j2k::DecodeSettings::lenient(),
            ..BatchDecodeOptions::default()
        };
        let mut decoder = MetalBatchDecoder::system_default_with_options(options)
            .expect("persistent Metal decoder");
        let prepared = decoder
            .prepare(vec![
                EncodedImage::full(Arc::<[u8]>::from(bytes)),
                EncodedImage::full(Arc::<[u8]>::from(bytes)),
            ])
            .expect("prepare independent HT grayscale group");
        let group = &prepared.groups()[0];
        let (width, height) = group.info().dimensions;
        let image_bytes = usize::try_from(width)
            .expect("width")
            .checked_mul(usize::try_from(height).expect("height"))
            .expect("image bytes");
        let output = j2k_metal_support::checked_shared_buffer(
            decoder.backend_session().device(),
            image_bytes.checked_mul(2).expect("group bytes"),
        )
        .expect("external group output");
        let layout = MetalImageLayout::new_batch(
            0,
            (width, height),
            width as usize,
            PixelFormat::Gray8,
            2,
            image_bytes,
        )
        .expect("dense grayscale group layout");
        // SAFETY: the fresh allocation remains exclusively owned by the
        // submitted codec work until completion.
        let destination = unsafe {
            MetalImageDestination::from_exclusive_buffer(output.clone(), layout)
                .expect("external group destination")
        };

        crate::compute::reset_stacked_component_batches_for_test();
        let completion = decoder
            .submit_prepared_group_into(group, destination)
            .expect("submit external group")
            .wait()
            .expect("complete external group");

        assert_eq!(
            crate::compute::stacked_component_batches_for_test(),
            1,
            "two homogeneous HT images must share one stacked component graph"
        );
        assert_eq!(
            completion.decoded_rects(),
            group
                .images()
                .iter()
                .map(|image| image.plan().output_rect())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            completion.warnings(),
            [
                vec![J2kDecodeWarning::LenientDecodeMode],
                vec![J2kDecodeWarning::LenientDecodeMode],
            ]
        );

        // The synchronous convenience API must use the same coalesced graph,
        // not the legacy per-image destination loop.
        // SAFETY: the prior submission completed and released this allocation;
        // the synchronous call retires its work before returning.
        let destination = unsafe {
            MetalImageDestination::from_exclusive_buffer(output, layout)
                .expect("reused external group destination")
        };
        crate::compute::reset_stacked_component_batches_for_test();
        decoder
            .decode_prepared_group_into(group, &destination)
            .expect("synchronous external group decode");
        assert_eq!(crate::compute::stacked_component_batches_for_test(), 1);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn external_ht_rgb_group_stacks_each_component_across_images() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }

        let fixture = j2k_test_support::openjph_batch_fixtures()
            .iter()
            .find(|fixture| fixture.name == "openjph-rgb-s12-53-single-raw")
            .expect("independent signed RGB fixture");
        let mut decoder = MetalBatchDecoder::system_default().expect("persistent Metal decoder");
        let prepared = decoder
            .prepare(vec![
                EncodedImage::full(Arc::<[u8]>::from(fixture.encoded)),
                EncodedImage::full(Arc::<[u8]>::from(fixture.encoded)),
            ])
            .expect("prepare independent HT RGB group");
        let group = &prepared.groups()[0];
        let (width, height) = group.info().dimensions;
        let row_bytes = usize::try_from(width)
            .expect("width")
            .checked_mul(PixelFormat::RgbI16.bytes_per_pixel())
            .expect("row bytes");
        let image_bytes = row_bytes
            .checked_mul(usize::try_from(height).expect("height"))
            .expect("image bytes");
        let output = j2k_metal_support::checked_shared_buffer(
            decoder.backend_session().device(),
            image_bytes.checked_mul(2).expect("group bytes"),
        )
        .expect("external RGB group output");
        let layout = MetalImageLayout::new_batch(
            0,
            (width, height),
            row_bytes,
            PixelFormat::RgbI16,
            2,
            image_bytes,
        )
        .expect("dense RGB group layout");
        // SAFETY: the fresh allocation remains exclusively owned by the
        // submitted codec work until completion.
        let destination = unsafe {
            MetalImageDestination::from_exclusive_buffer(output.clone(), layout)
                .expect("external RGB group destination")
        };

        crate::compute::reset_stacked_component_batches_for_test();
        decoder
            .submit_prepared_group_into(group, destination)
            .expect("submit external RGB group")
            .wait()
            .expect("complete external RGB group");

        assert_eq!(
            crate::compute::stacked_component_batches_for_test(),
            3,
            "RGB components must each coalesce the two image plans"
        );
        // SAFETY: codec completion released the exclusive destination before
        // this host-only parity check.
        let bytes = unsafe {
            j2k_metal_support::checked_buffer_read_vec::<u8>(
                &output,
                layout.byte_offset(),
                layout.byte_len(),
            )
            .expect("completed RGB group bytes")
        };
        assert_eq!(
            &bytes[..image_bytes],
            &bytes[image_bytes..],
            "independently prepared identical inputs must remain identical after stacking"
        );
    }
}
