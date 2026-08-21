// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) use j2k_cuda_runtime::CudaLaunchGeometry;
use j2k_cuda_runtime::{CudaError, CudaKernelSpec};

#[cfg(test)]
pub(crate) const CUDA_MAX_GRID_DIM_X: u32 = 2_147_483_647;

#[cfg(feature = "cuda-oxide-jpeg-decode")]
use crate::build_flags::ensure_jpeg_decode_ptx_built;
#[cfg(feature = "cuda-oxide-jpeg-encode")]
use crate::build_flags::ensure_jpeg_encode_ptx_built;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CudaKernel {
    JpegDecodeFast420Rgb8,
    JpegDecodeFast422Rgb8,
    JpegDecodeFast444Rgb8,
    JpegSubsampledPlanesToRgb8,
    JpegEntropySync420,
    JpegEntropyOverflow420,
    JpegEncodeBaselinePrecomputeBatch,
    JpegEncodeBaselineEntropyFromCoeffsBatch,
}

impl CudaKernel {
    pub(crate) const fn entrypoint(self) -> &'static [u8] {
        match self {
            Self::JpegDecodeFast420Rgb8 => b"j2k_jpeg_decode_fast420_rgb8\0",
            Self::JpegDecodeFast422Rgb8 => b"j2k_jpeg_decode_fast422_rgb8\0",
            Self::JpegDecodeFast444Rgb8 => b"j2k_jpeg_decode_fast444_rgb8\0",
            Self::JpegSubsampledPlanesToRgb8 => b"j2k_jpeg_subsampled_planes_to_rgb8\0",
            Self::JpegEntropySync420 => b"j2k_jpeg_entropy_sync420\0",
            Self::JpegEntropyOverflow420 => b"j2k_jpeg_entropy_overflow420\0",
            Self::JpegEncodeBaselinePrecomputeBatch => {
                b"j2k_jpeg_encode_baseline_precompute_batch\0"
            }
            Self::JpegEncodeBaselineEntropyFromCoeffsBatch => {
                b"j2k_jpeg_encode_baseline_entropy_from_coeffs_batch\0"
            }
        }
    }

    pub(crate) fn spec(self) -> Result<CudaKernelSpec, CudaError> {
        let (module_id, ptx) = match self {
            Self::JpegDecodeFast420Rgb8
            | Self::JpegDecodeFast422Rgb8
            | Self::JpegDecodeFast444Rgb8
            | Self::JpegSubsampledPlanesToRgb8
            | Self::JpegEntropySync420
            | Self::JpegEntropyOverflow420 => {
                #[cfg(feature = "cuda-oxide-jpeg-decode")]
                ensure_jpeg_decode_ptx_built()?;
                ("jpeg-decode", jpeg_decode_ptx())
            }
            Self::JpegEncodeBaselinePrecomputeBatch
            | Self::JpegEncodeBaselineEntropyFromCoeffsBatch => {
                #[cfg(feature = "cuda-oxide-jpeg-encode")]
                ensure_jpeg_encode_ptx_built()?;
                ("jpeg-encode", jpeg_encode_ptx())
            }
        };
        CudaKernelSpec::new(module_id, ptx, self.entrypoint())
    }
}

#[cfg(feature = "cuda-oxide-jpeg-decode")]
fn jpeg_decode_ptx() -> &'static [u8] {
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_jpeg_decode.ptx"))
}

#[cfg(not(feature = "cuda-oxide-jpeg-decode"))]
fn jpeg_decode_ptx() -> &'static [u8] {
    unreachable!("JPEG decode kernels are gated by cuda-oxide-jpeg-decode")
}

#[cfg(feature = "cuda-oxide-jpeg-encode")]
fn jpeg_encode_ptx() -> &'static [u8] {
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_jpeg_encode.ptx"))
}

#[cfg(not(feature = "cuda-oxide-jpeg-encode"))]
fn jpeg_encode_ptx() -> &'static [u8] {
    unreachable!("JPEG encode kernels are gated by cuda-oxide-jpeg-encode")
}

#[cfg(test)]
mod tests {
    use super::CudaKernel;

    #[test]
    fn jpeg_entrypoints_are_stable() {
        let expected = [
            (
                CudaKernel::JpegDecodeFast420Rgb8,
                b"j2k_jpeg_decode_fast420_rgb8\0".as_slice(),
            ),
            (
                CudaKernel::JpegDecodeFast422Rgb8,
                b"j2k_jpeg_decode_fast422_rgb8\0".as_slice(),
            ),
            (
                CudaKernel::JpegDecodeFast444Rgb8,
                b"j2k_jpeg_decode_fast444_rgb8\0".as_slice(),
            ),
            (
                CudaKernel::JpegSubsampledPlanesToRgb8,
                b"j2k_jpeg_subsampled_planes_to_rgb8\0".as_slice(),
            ),
            (
                CudaKernel::JpegEntropySync420,
                b"j2k_jpeg_entropy_sync420\0".as_slice(),
            ),
            (
                CudaKernel::JpegEntropyOverflow420,
                b"j2k_jpeg_entropy_overflow420\0".as_slice(),
            ),
            (
                CudaKernel::JpegEncodeBaselinePrecomputeBatch,
                b"j2k_jpeg_encode_baseline_precompute_batch\0".as_slice(),
            ),
            (
                CudaKernel::JpegEncodeBaselineEntropyFromCoeffsBatch,
                b"j2k_jpeg_encode_baseline_entropy_from_coeffs_batch\0".as_slice(),
            ),
        ];
        for (kernel, entrypoint) in expected {
            assert_eq!(kernel.entrypoint(), entrypoint);
        }
    }

    #[cfg(all(feature = "cuda-oxide-jpeg-decode", j2k_cuda_oxide_jpeg_decode_built))]
    #[test]
    fn decode_metadata_matches_generated_ptx() {
        let ptx = super::jpeg_decode_ptx();
        for kernel in [
            CudaKernel::JpegDecodeFast420Rgb8,
            CudaKernel::JpegDecodeFast422Rgb8,
            CudaKernel::JpegDecodeFast444Rgb8,
            CudaKernel::JpegSubsampledPlanesToRgb8,
            CudaKernel::JpegEntropySync420,
            CudaKernel::JpegEntropyOverflow420,
        ] {
            let entrypoint = &kernel.entrypoint()[..kernel.entrypoint().len() - 1];
            assert!(ptx
                .windows(entrypoint.len())
                .any(|window| window == entrypoint));
        }
    }

    #[cfg(all(feature = "cuda-oxide-jpeg-encode", j2k_cuda_oxide_jpeg_encode_built))]
    #[test]
    fn encode_metadata_matches_generated_ptx() {
        let ptx = super::jpeg_encode_ptx();
        for kernel in [
            CudaKernel::JpegEncodeBaselinePrecomputeBatch,
            CudaKernel::JpegEncodeBaselineEntropyFromCoeffsBatch,
        ] {
            let entrypoint = &kernel.entrypoint()[..kernel.entrypoint().len() - 1];
            assert!(ptx
                .windows(entrypoint.len())
                .any(|window| window == entrypoint));
        }
    }
}
