// SPDX-License-Identifier: MIT OR Apache-2.0

//! Device-resident component workspace for cross-MCU chroma interpolation.

use super::super::{CudaJpeg420Params, CudaJpegRgb8Sampling};
use crate::{error::CudaError, kernels::CudaLaunchGeometry};

const CONVERSION_THREADS: u32 = 256;
const U32_ADDRESSABLE_BYTES: u64 = u32::MAX as u64 + 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct CudaJpegSubsampledWorkspacePlan {
    pub(super) byte_len: usize,
    pub(super) conversion_geometry: CudaLaunchGeometry,
    pub(super) sampling_code: u32,
}

pub(super) fn subsampled_workspace_plan(
    sampling: CudaJpegRgb8Sampling,
    params: CudaJpeg420Params,
) -> Result<Option<CudaJpegSubsampledWorkspacePlan>, CudaError> {
    let sampling_code = match sampling {
        CudaJpegRgb8Sampling::Fast420 => 0,
        CudaJpegRgb8Sampling::Fast422 => 1,
        CudaJpegRgb8Sampling::Fast444 => return Ok(None),
    };
    let pixel_count = params
        .width
        .checked_mul(params.height)
        .ok_or(CudaError::ImageTooLarge {
            width: params.width,
            height: params.height,
            channels: 3,
        })?;
    if pixel_count == 0 {
        return Err(CudaError::InvalidArgument {
            message: "JPEG CUDA component workspace requires nonzero dimensions".to_string(),
        });
    }
    let chroma_width = params.width.div_ceil(2);
    let chroma_height = if sampling == CudaJpegRgb8Sampling::Fast420 {
        params.height.div_ceil(2)
    } else {
        params.height
    };
    let byte_len_u64 = u64::from(pixel_count)
        .checked_add(
            u64::from(chroma_width)
                .checked_mul(u64::from(chroma_height))
                .and_then(|bytes| bytes.checked_mul(2))
                .ok_or(CudaError::ImageTooLarge {
                    width: params.width,
                    height: params.height,
                    channels: 3,
                })?,
        )
        .ok_or(CudaError::ImageTooLarge {
            width: params.width,
            height: params.height,
            channels: 3,
        })?;
    if byte_len_u64 > U32_ADDRESSABLE_BYTES {
        return Err(CudaError::InvalidArgument {
            message: "JPEG CUDA component workspace exceeds u32 byte addressing".to_string(),
        });
    }
    let byte_len = usize::try_from(byte_len_u64).map_err(|_| CudaError::ImageTooLarge {
        width: params.width,
        height: params.height,
        channels: 3,
    })?;
    let conversion_geometry = CudaLaunchGeometry::new(
        (pixel_count.div_ceil(CONVERSION_THREADS), 1, 1),
        (CONVERSION_THREADS, 1, 1),
    )
    .ok_or(CudaError::InvalidArgument {
        message: "JPEG CUDA component conversion launch exceeds static limits".to_string(),
    })?;
    Ok(Some(CudaJpegSubsampledWorkspacePlan {
        byte_len,
        conversion_geometry,
        sampling_code,
    }))
}

#[cfg(test)]
mod tests {
    use super::{subsampled_workspace_plan, CudaJpeg420Params, CudaJpegRgb8Sampling};

    fn params(width: u32, height: u32) -> CudaJpeg420Params {
        CudaJpeg420Params {
            width,
            height,
            mcus_per_row: 1,
            mcu_rows: 1,
            entropy_len: 1,
            checkpoint_count: 1,
            out_stride: width.saturating_mul(3),
            reserved: 0,
        }
    }

    #[test]
    fn component_workspace_accounts_exact_subsampled_plane_shapes() {
        let plan_420 = subsampled_workspace_plan(CudaJpegRgb8Sampling::Fast420, params(32, 32))
            .unwrap()
            .unwrap();
        assert_eq!(plan_420.byte_len, 32 * 32 + 2 * 16 * 16);
        assert_eq!(plan_420.sampling_code, 0);

        let odd_420 = subsampled_workspace_plan(CudaJpegRgb8Sampling::Fast420, params(17, 17))
            .unwrap()
            .unwrap();
        assert_eq!(odd_420.byte_len, 17 * 17 + 2 * 9 * 9);

        let plan_422 = subsampled_workspace_plan(CudaJpegRgb8Sampling::Fast422, params(32, 8))
            .unwrap()
            .unwrap();
        assert_eq!(plan_422.byte_len, 32 * 8 + 2 * 16 * 8);
        assert_eq!(plan_422.sampling_code, 1);

        assert!(
            subsampled_workspace_plan(CudaJpegRgb8Sampling::Fast444, params(8, 8))
                .unwrap()
                .is_none()
        );
    }
}
