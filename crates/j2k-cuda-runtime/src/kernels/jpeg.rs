// SPDX-License-Identifier: MIT OR Apache-2.0

use super::CudaKernel;

impl CudaKernel {
    pub(crate) fn is_jpeg_entropy_stage(self) -> bool {
        matches!(
            self,
            Self::JpegEntropySync420 | Self::JpegEntropyOverflow420
        )
    }

    #[cfg_attr(
        all(not(feature = "cuda-oxide-jpeg-decode"), not(test)),
        expect(
            dead_code,
            reason = "classifier is used only by the JPEG decode kernel feature"
        )
    )]
    pub(crate) fn is_cuda_oxide_jpeg_decode_stage(self) -> bool {
        self.is_jpeg_entropy_stage()
            || matches!(
                self,
                Self::JpegDecodeFast420Rgb8
                    | Self::JpegDecodeFast422Rgb8
                    | Self::JpegDecodeFast444Rgb8
                    | Self::JpegSubsampledPlanesToRgb8
            )
    }

    #[cfg_attr(
        all(not(feature = "cuda-oxide-jpeg-encode"), not(test)),
        expect(
            dead_code,
            reason = "classifier is used only by the JPEG encode kernel feature"
        )
    )]
    pub(crate) fn is_cuda_oxide_jpeg_encode_stage(self) -> bool {
        matches!(
            self,
            Self::JpegEncodeBaselineEntropy | Self::JpegEncodeBaselineEntropyBatch
        )
    }
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
const CUDA_OXIDE_JPEG_DECODE_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_jpeg_decode.ptx"));
#[cfg(feature = "cuda-oxide-jpeg-encode")]
const CUDA_OXIDE_JPEG_ENCODE_PTX: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_jpeg_encode.ptx"));

#[cfg(feature = "cuda-oxide-jpeg-decode")]
pub(crate) fn cuda_oxide_jpeg_decode_ptx() -> &'static [u8] {
    CUDA_OXIDE_JPEG_DECODE_PTX
}

#[cfg(feature = "cuda-oxide-jpeg-encode")]
pub(crate) fn cuda_oxide_jpeg_encode_ptx() -> &'static [u8] {
    CUDA_OXIDE_JPEG_ENCODE_PTX
}
