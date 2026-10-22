// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) use j2k_cuda_runtime::CudaLaunchGeometry;
#[cfg(j2k_cuda_oxide_enabled)]
use j2k_cuda_runtime::{CudaError, CudaKernelSpec};

#[cfg(test)]
pub(crate) const CUDA_MAX_GRID_DIM_X: u32 = 2_147_483_647;

#[cfg(feature = "cuda-oxide-jpeg-decode")]
use crate::build_flags::ensure_jpeg_decode_ptx_built;
#[cfg(feature = "cuda-oxide-jpeg-encode")]
use crate::build_flags::ensure_jpeg_encode_ptx_built;
#[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
use crate::build_flags::ensure_jpeg_lossless_encode_ptx_built;

/// Each kernel exists only with the feature that builds its PTX module.
#[cfg(j2k_cuda_oxide_enabled)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CudaKernel {
    #[cfg(feature = "cuda-oxide-jpeg-decode")]
    JpegDecodeFast420Rgb8,
    #[cfg(feature = "cuda-oxide-jpeg-decode")]
    JpegDecodeFast422Rgb8,
    #[cfg(feature = "cuda-oxide-jpeg-decode")]
    JpegDecodeFast444Rgb8,
    #[cfg(feature = "cuda-oxide-jpeg-decode")]
    JpegSubsampledPlanesToRgb8,
    #[cfg(feature = "cuda-oxide-jpeg-decode")]
    JpegEntropySync420,
    #[cfg(feature = "cuda-oxide-jpeg-decode")]
    JpegEntropyOverflow420,
    #[cfg(feature = "cuda-oxide-jpeg-encode")]
    JpegEncodeBaselinePrecomputeBatch,
    #[cfg(feature = "cuda-oxide-jpeg-encode")]
    JpegEncodeBaselineEntropyFromCoeffsBatch,
    #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
    JpegLosslessEncodeClassify,
    #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
    JpegLosslessEncodeReadSample,
    #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
    JpegLosslessEncodeChunkBits,
    #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
    JpegLosslessEncodeRowOffsets,
    #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
    JpegLosslessEncodeIntervalOffsets,
    #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
    JpegLosslessEncodeIntervalBases,
    #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
    JpegLosslessEncodeClear,
    #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
    JpegLosslessEncodePack,
}

#[cfg(j2k_cuda_oxide_enabled)]
impl CudaKernel {
    pub(crate) const fn entrypoint(self) -> &'static [u8] {
        match self {
            #[cfg(feature = "cuda-oxide-jpeg-decode")]
            Self::JpegDecodeFast420Rgb8 => b"j2k_jpeg_decode_fast420_rgb8\0",
            #[cfg(feature = "cuda-oxide-jpeg-decode")]
            Self::JpegDecodeFast422Rgb8 => b"j2k_jpeg_decode_fast422_rgb8\0",
            #[cfg(feature = "cuda-oxide-jpeg-decode")]
            Self::JpegDecodeFast444Rgb8 => b"j2k_jpeg_decode_fast444_rgb8\0",
            #[cfg(feature = "cuda-oxide-jpeg-decode")]
            Self::JpegSubsampledPlanesToRgb8 => b"j2k_jpeg_subsampled_planes_to_rgb8\0",
            #[cfg(feature = "cuda-oxide-jpeg-decode")]
            Self::JpegEntropySync420 => b"j2k_jpeg_entropy_sync420\0",
            #[cfg(feature = "cuda-oxide-jpeg-decode")]
            Self::JpegEntropyOverflow420 => b"j2k_jpeg_entropy_overflow420\0",
            #[cfg(feature = "cuda-oxide-jpeg-encode")]
            Self::JpegEncodeBaselinePrecomputeBatch => {
                b"j2k_jpeg_encode_baseline_precompute_batch\0"
            }
            #[cfg(feature = "cuda-oxide-jpeg-encode")]
            Self::JpegEncodeBaselineEntropyFromCoeffsBatch => {
                b"j2k_jpeg_encode_baseline_entropy_from_coeffs_batch\0"
            }
            #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
            Self::JpegLosslessEncodeClassify => b"j2k_jpeg_lossless_encode_classify\0",
            #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
            Self::JpegLosslessEncodeReadSample => b"j2k_jpeg_lossless_encode_read_sample\0",
            #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
            Self::JpegLosslessEncodeChunkBits => b"j2k_jpeg_lossless_encode_chunk_bits\0",
            #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
            Self::JpegLosslessEncodeRowOffsets => b"j2k_jpeg_lossless_encode_row_offsets\0",
            #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
            Self::JpegLosslessEncodeIntervalOffsets => {
                b"j2k_jpeg_lossless_encode_interval_offsets\0"
            }
            #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
            Self::JpegLosslessEncodeIntervalBases => b"j2k_jpeg_lossless_encode_interval_bases\0",
            #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
            Self::JpegLosslessEncodeClear => b"j2k_jpeg_lossless_encode_clear\0",
            #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
            Self::JpegLosslessEncodePack => b"j2k_jpeg_lossless_encode_pack\0",
        }
    }

    pub(crate) fn spec(self) -> Result<CudaKernelSpec, CudaError> {
        let (module_id, ptx) = match self {
            #[cfg(feature = "cuda-oxide-jpeg-decode")]
            Self::JpegDecodeFast420Rgb8
            | Self::JpegDecodeFast422Rgb8
            | Self::JpegDecodeFast444Rgb8
            | Self::JpegSubsampledPlanesToRgb8
            | Self::JpegEntropySync420
            | Self::JpegEntropyOverflow420 => {
                ensure_jpeg_decode_ptx_built()?;
                ("jpeg-decode", jpeg_decode_ptx())
            }
            #[cfg(feature = "cuda-oxide-jpeg-encode")]
            Self::JpegEncodeBaselinePrecomputeBatch
            | Self::JpegEncodeBaselineEntropyFromCoeffsBatch => {
                ensure_jpeg_encode_ptx_built()?;
                ("jpeg-encode", jpeg_encode_ptx())
            }
            #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
            Self::JpegLosslessEncodeClassify
            | Self::JpegLosslessEncodeReadSample
            | Self::JpegLosslessEncodeChunkBits
            | Self::JpegLosslessEncodeRowOffsets
            | Self::JpegLosslessEncodeIntervalOffsets
            | Self::JpegLosslessEncodeIntervalBases
            | Self::JpegLosslessEncodeClear
            | Self::JpegLosslessEncodePack => {
                ensure_jpeg_lossless_encode_ptx_built()?;
                ("jpeg-lossless-encode", jpeg_lossless_encode_ptx())
            }
        };
        CudaKernelSpec::new(module_id, ptx, self.entrypoint())
    }
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
fn jpeg_decode_ptx() -> &'static [u8] {
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_jpeg_decode.ptx"))
}

#[cfg(feature = "cuda-oxide-jpeg-encode")]
fn jpeg_encode_ptx() -> &'static [u8] {
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_jpeg_encode.ptx"))
}

#[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
fn jpeg_lossless_encode_ptx() -> &'static [u8] {
    include_bytes!(concat!(
        env!("OUT_DIR"),
        "/cuda_oxide_jpeg_lossless_encode.ptx"
    ))
}

#[cfg(all(test, j2k_cuda_oxide_enabled))]
mod tests {
    use super::CudaKernel;

    #[cfg(feature = "cuda-oxide-jpeg-decode")]
    const DECODE_KERNELS: [(CudaKernel, &[u8]); 6] = [
        (
            CudaKernel::JpegDecodeFast420Rgb8,
            b"j2k_jpeg_decode_fast420_rgb8\0",
        ),
        (
            CudaKernel::JpegDecodeFast422Rgb8,
            b"j2k_jpeg_decode_fast422_rgb8\0",
        ),
        (
            CudaKernel::JpegDecodeFast444Rgb8,
            b"j2k_jpeg_decode_fast444_rgb8\0",
        ),
        (
            CudaKernel::JpegSubsampledPlanesToRgb8,
            b"j2k_jpeg_subsampled_planes_to_rgb8\0",
        ),
        (
            CudaKernel::JpegEntropySync420,
            b"j2k_jpeg_entropy_sync420\0",
        ),
        (
            CudaKernel::JpegEntropyOverflow420,
            b"j2k_jpeg_entropy_overflow420\0",
        ),
    ];

    #[cfg(feature = "cuda-oxide-jpeg-encode")]
    const ENCODE_KERNELS: [(CudaKernel, &[u8]); 2] = [
        (
            CudaKernel::JpegEncodeBaselinePrecomputeBatch,
            b"j2k_jpeg_encode_baseline_precompute_batch\0",
        ),
        (
            CudaKernel::JpegEncodeBaselineEntropyFromCoeffsBatch,
            b"j2k_jpeg_encode_baseline_entropy_from_coeffs_batch\0",
        ),
    ];

    #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
    const LOSSLESS_ENCODE_KERNELS: [(CudaKernel, &[u8]); 8] = [
        (
            CudaKernel::JpegLosslessEncodeClassify,
            b"j2k_jpeg_lossless_encode_classify\0",
        ),
        (
            CudaKernel::JpegLosslessEncodeReadSample,
            b"j2k_jpeg_lossless_encode_read_sample\0",
        ),
        (
            CudaKernel::JpegLosslessEncodeChunkBits,
            b"j2k_jpeg_lossless_encode_chunk_bits\0",
        ),
        (
            CudaKernel::JpegLosslessEncodeRowOffsets,
            b"j2k_jpeg_lossless_encode_row_offsets\0",
        ),
        (
            CudaKernel::JpegLosslessEncodeIntervalOffsets,
            b"j2k_jpeg_lossless_encode_interval_offsets\0",
        ),
        (
            CudaKernel::JpegLosslessEncodeIntervalBases,
            b"j2k_jpeg_lossless_encode_interval_bases\0",
        ),
        (
            CudaKernel::JpegLosslessEncodeClear,
            b"j2k_jpeg_lossless_encode_clear\0",
        ),
        (
            CudaKernel::JpegLosslessEncodePack,
            b"j2k_jpeg_lossless_encode_pack\0",
        ),
    ];

    fn assert_entrypoints_are_stable(kernels: &[(CudaKernel, &[u8])]) {
        for (kernel, entrypoint) in kernels {
            assert_eq!(kernel.entrypoint(), *entrypoint);
        }
    }

    #[cfg(any(
        j2k_cuda_oxide_jpeg_decode_built,
        j2k_cuda_oxide_jpeg_encode_built,
        j2k_cuda_oxide_jpeg_lossless_encode_built
    ))]
    fn assert_entrypoints_in_ptx(ptx: &[u8], kernels: &[(CudaKernel, &[u8])]) {
        for (kernel, _) in kernels {
            let entrypoint = &kernel.entrypoint()[..kernel.entrypoint().len() - 1];
            assert!(ptx
                .windows(entrypoint.len())
                .any(|window| window == entrypoint));
        }
    }

    #[cfg(feature = "cuda-oxide-jpeg-decode")]
    #[test]
    fn decode_entrypoints_are_stable() {
        assert_entrypoints_are_stable(&DECODE_KERNELS);
    }

    #[cfg(feature = "cuda-oxide-jpeg-encode")]
    #[test]
    fn encode_entrypoints_are_stable() {
        assert_entrypoints_are_stable(&ENCODE_KERNELS);
    }

    #[cfg(feature = "cuda-oxide-jpeg-lossless-encode")]
    #[test]
    fn lossless_encode_entrypoints_are_stable() {
        assert_entrypoints_are_stable(&LOSSLESS_ENCODE_KERNELS);
    }

    #[cfg(all(
        feature = "cuda-oxide-jpeg-lossless-encode",
        j2k_cuda_oxide_jpeg_lossless_encode_built
    ))]
    #[test]
    fn lossless_encode_metadata_matches_generated_ptx() {
        assert_entrypoints_in_ptx(super::jpeg_lossless_encode_ptx(), &LOSSLESS_ENCODE_KERNELS);
    }

    #[cfg(all(feature = "cuda-oxide-jpeg-decode", j2k_cuda_oxide_jpeg_decode_built))]
    #[test]
    fn decode_metadata_matches_generated_ptx() {
        assert_entrypoints_in_ptx(super::jpeg_decode_ptx(), &DECODE_KERNELS);
    }

    #[cfg(all(feature = "cuda-oxide-jpeg-encode", j2k_cuda_oxide_jpeg_encode_built))]
    #[test]
    fn encode_metadata_matches_generated_ptx() {
        assert_entrypoints_in_ptx(super::jpeg_encode_ptx(), &ENCODE_KERNELS);
    }
}
