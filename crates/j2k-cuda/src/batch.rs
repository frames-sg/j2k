// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    prepare_batch, prepare_batch_from_images, BatchDecodeOptions, BatchDecoder, BatchGroupInfo,
    EncodedImage, IndexedBatchError, J2kDecodeWarning, PreparedBatch, PreparedBatchGroup,
    PreparedImage,
};
#[cfg(feature = "cuda-runtime")]
use j2k::{BatchColor, BatchLayout};
#[cfg(feature = "cuda-runtime")]
use j2k_core::{BackendKind, PixelFormat};
use j2k_core::{BatchInfrastructureError, Rect};

#[cfg(feature = "cuda-runtime")]
use std::sync::Arc;

#[cfg(feature = "cuda-runtime")]
use crate::surface::cuda_range_storage;
use crate::{CudaSession, Error, Surface};
#[cfg(feature = "cuda-runtime")]
use crate::{CudaSurfaceStats, SurfaceResidency};

mod decoder;
#[cfg(feature = "cuda-runtime")]
mod external;
#[cfg(feature = "cuda-runtime")]
mod input;
#[cfg(feature = "cuda-runtime")]
mod resident_submission;
mod types;
pub use self::decoder::CudaBatchDecoder;
#[cfg(feature = "cuda-runtime")]
pub use self::external::{
    CudaExternalBatchGroup, CudaExternalBatchTryFinish, SubmittedCudaExternalBatch,
};
#[cfg(feature = "cuda-runtime")]
pub use self::resident_submission::SubmittedCudaResidentBatch;
#[cfg(feature = "cuda-runtime")]
pub use self::types::CudaResidentBatchBuffer;
pub use self::types::{CudaBatchDecodeResult, CudaBatchError, CudaBatchGroup, CudaBatchGroupError};

#[cfg(feature = "cuda-runtime")]
use self::external::SubmittedCudaCodecBatch;
#[cfg(feature = "cuda-runtime")]
use self::input::{
    decode_warnings, group_pixel_format, native_color_group_storage, native_color_inputs,
    native_decode_settings, native_referenced_classic_plan, native_referenced_htj2k_plan,
    validate_layout,
};

#[cfg(all(test, feature = "cuda-runtime"))]
mod tests {
    use std::sync::Arc;

    use j2k::{
        prepare_batch, BatchAlpha, BatchCodecRoute, BatchColor, BatchDecodeOptions, BatchGroupInfo,
        BatchLayout, BatchWaveletTransform, EncodedImage, NativeSampleType,
    };
    use j2k_core::{Colorspace, CompressedPayloadKind, CompressedTransferSyntax, PixelFormat};

    use super::{group_pixel_format, native_color_inputs, validate_layout};

    #[test]
    fn signed_grayscale_group_selects_native_i16_cuda_store() {
        let info = BatchGroupInfo {
            dimensions: (8, 8),
            color: BatchColor::Gray,
            alpha: BatchAlpha::None,
            precision: 12,
            signed: true,
            sample_type: NativeSampleType::I16,
            layout: BatchLayout::Nchw,
            colorspace: Colorspace::Grayscale,
            route: BatchCodecRoute::Htj2k,
            transform: BatchWaveletTransform::Reversible53,
            transfer_syntax: CompressedTransferSyntax::HtJpeg2000Lossless,
            payload_kind: CompressedPayloadKind::Jpeg2000Codestream,
        };

        assert_eq!(
            group_pixel_format(&info).expect("signed grayscale CUDA format"),
            PixelFormat::GrayI16
        );
    }

    #[test]
    fn native_rgb_batch_selects_exact_unsigned_cuda_stores_for_both_layouts() {
        for (sample_type, precision, expected) in [
            (NativeSampleType::U8, 7, PixelFormat::Rgb8),
            (NativeSampleType::U16, 12, PixelFormat::Rgb16),
        ] {
            for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
                let info = BatchGroupInfo {
                    dimensions: (8, 8),
                    color: BatchColor::Rgb,
                    alpha: BatchAlpha::None,
                    precision,
                    signed: false,
                    sample_type,
                    layout,
                    colorspace: Colorspace::SRgb,
                    route: BatchCodecRoute::Htj2k,
                    transform: BatchWaveletTransform::Reversible53,
                    transfer_syntax: CompressedTransferSyntax::HtJpeg2000Lossless,
                    payload_kind: CompressedPayloadKind::Jpeg2000Codestream,
                };

                assert_eq!(group_pixel_format(&info).unwrap(), expected);
                validate_layout(&info).expect("exact CUDA RGB layout");
            }
        }
    }

    #[cfg(feature = "cuda-runtime")]
    #[test]
    fn classic_rgb_group_enters_the_exact_native_color_pipeline_without_an_ht_plan() {
        let pixels = (0_u8..8 * 8 * 3).collect::<Vec<_>>();
        let encoded = j2k_native::encode(
            &pixels,
            8,
            8,
            3,
            8,
            false,
            &j2k_native::EncodeOptions {
                reversible: true,
                num_decomposition_levels: 1,
                ..j2k_native::EncodeOptions::default()
            },
        )
        .expect("encode classic RGB fixture");
        let prepared = prepare_batch(
            vec![EncodedImage::full(Arc::from(encoded))],
            BatchDecodeOptions::default(),
        )
        .expect("prepare classic RGB fixture");
        let [group] = prepared.groups() else {
            panic!("expected one classic RGB group")
        };
        assert!(group.images()[0].htj2k_plan().is_none());

        native_color_inputs(group).expect("classic RGB exact CUDA input");
    }

    #[test]
    fn native_color_batch_selects_signed_rgb_and_exact_rgba_formats() {
        for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
            let info = BatchGroupInfo {
                dimensions: (8, 8),
                color: BatchColor::Rgb,
                alpha: BatchAlpha::None,
                precision: 12,
                signed: true,
                sample_type: NativeSampleType::I16,
                layout,
                colorspace: Colorspace::SRgb,
                route: BatchCodecRoute::Htj2k,
                transform: BatchWaveletTransform::Reversible53,
                transfer_syntax: CompressedTransferSyntax::HtJpeg2000Lossless,
                payload_kind: CompressedPayloadKind::Jpeg2000Codestream,
            };

            assert_eq!(
                group_pixel_format(&info).expect("signed RGB CUDA format"),
                PixelFormat::RgbI16
            );
            validate_layout(&info).expect("signed exact CUDA RGB layout");
        }

        for (sample_type, precision, expected) in [
            (NativeSampleType::U8, 8, PixelFormat::Rgba8),
            (NativeSampleType::U16, 12, PixelFormat::Rgba16),
            (NativeSampleType::I16, 12, PixelFormat::RgbaI16),
        ] {
            let info = BatchGroupInfo {
                dimensions: (8, 8),
                color: BatchColor::Rgba,
                alpha: BatchAlpha::Straight,
                precision,
                signed: sample_type == NativeSampleType::I16,
                sample_type,
                layout: BatchLayout::Nhwc,
                colorspace: Colorspace::SRgb,
                route: BatchCodecRoute::Htj2k,
                transform: BatchWaveletTransform::Reversible53,
                transfer_syntax: CompressedTransferSyntax::HtJpeg2000Lossless,
                payload_kind: CompressedPayloadKind::Jpeg2000Codestream,
            };

            assert_eq!(group_pixel_format(&info).unwrap(), expected);
            validate_layout(&info).expect("exact CUDA RGBA layout");
        }
    }
}
