// SPDX-License-Identifier: MIT OR Apache-2.0

use std::os::raw::c_uint;

use crate::{build_flags::ensure_transcode_ptx_built, error::CudaError};
use j2k_cuda_runtime::CudaKernelSpec;
pub(crate) use j2k_cuda_runtime::CudaLaunchGeometry;

const SAMPLE_THREADS: usize = 256;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum CudaKernel {
    TranscodeReversible53Idct,
    TranscodeReversible53VerticalLow,
    TranscodeReversible53VerticalHigh,
    TranscodeReversible53HorizontalLow,
    TranscodeReversible53HorizontalHigh,
    TranscodeDwt97Idct,
    TranscodeDwt97RowLift,
    TranscodeDwt97ColumnLift,
    TranscodeDwt97IdctBatch,
    TranscodeDwt97IdctI16Batch,
    TranscodeDwt97RowLiftBatch,
    TranscodeDwt97RowLiftBatchCoop,
    TranscodeDwt97ColumnLiftBatch,
    TranscodeDwt97QuantizeCodeblocks,
}

impl CudaKernel {
    pub(crate) fn spec(self) -> Result<CudaKernelSpec, CudaError> {
        ensure_transcode_ptx_built()?;
        CudaKernelSpec::new("transcode", transcode_ptx(), self.entrypoint())
    }

    pub(crate) const fn entrypoint(self) -> &'static [u8] {
        match self {
            Self::TranscodeReversible53Idct => b"transcode_reversible53_idct\0",
            Self::TranscodeReversible53VerticalLow => b"transcode_reversible53_vertical_low\0",
            Self::TranscodeReversible53VerticalHigh => b"transcode_reversible53_vertical_high\0",
            Self::TranscodeReversible53HorizontalLow => b"transcode_reversible53_horizontal_low\0",
            Self::TranscodeReversible53HorizontalHigh => {
                b"transcode_reversible53_horizontal_high\0"
            }
            Self::TranscodeDwt97Idct => b"transcode_dwt97_idct\0",
            Self::TranscodeDwt97RowLift => b"transcode_dwt97_row_lift\0",
            Self::TranscodeDwt97ColumnLift => b"transcode_dwt97_column_lift\0",
            Self::TranscodeDwt97IdctBatch => b"transcode_dwt97_idct_batch\0",
            Self::TranscodeDwt97IdctI16Batch => b"transcode_dwt97_idct_i16_batch\0",
            Self::TranscodeDwt97RowLiftBatch => b"transcode_dwt97_row_lift_batch\0",
            Self::TranscodeDwt97RowLiftBatchCoop => b"transcode_dwt97_row_lift_batch_coop\0",
            Self::TranscodeDwt97ColumnLiftBatch => b"transcode_dwt97_column_lift_batch\0",
            Self::TranscodeDwt97QuantizeCodeblocks => b"transcode_dwt97_quantize_codeblocks\0",
        }
    }
}

pub(crate) fn copy_u8_launch_geometry(len: usize) -> Option<CudaLaunchGeometry> {
    x_blocks_launch_geometry(len, 1, SAMPLE_THREADS)
}

fn x_blocks_launch_geometry(
    work_items: usize,
    grid_y: usize,
    threads_per_block: usize,
) -> Option<CudaLaunchGeometry> {
    if threads_per_block == 0 {
        return None;
    }
    let blocks = c_uint::try_from(work_items.div_ceil(threads_per_block)).ok()?;
    let grid_y = c_uint::try_from(grid_y).ok()?;
    let block_x = c_uint::try_from(threads_per_block).ok()?;
    CudaLaunchGeometry::new((blocks, grid_y, 1), (block_x, 1, 1))
}

pub(crate) fn transcode_dwt53_launch_geometry(
    width: u32,
    height: u32,
) -> Option<CudaLaunchGeometry> {
    CudaLaunchGeometry::new((width.div_ceil(16), height.div_ceil(16), 1), (16, 16, 1))
}

pub(crate) fn with_grid_y(base: CudaLaunchGeometry, grid_y: c_uint) -> Option<CudaLaunchGeometry> {
    let (grid_x, _, grid_z) = base.grid();
    CudaLaunchGeometry::new((grid_x, grid_y, grid_z), base.block())
}

pub(crate) fn with_grid_z(base: CudaLaunchGeometry, grid_z: c_uint) -> Option<CudaLaunchGeometry> {
    let (grid_x, grid_y, _) = base.grid();
    CudaLaunchGeometry::new((grid_x, grid_y, grid_z), base.block())
}

#[cfg(feature = "cuda-oxide-transcode")]
fn transcode_ptx() -> &'static [u8] {
    include_bytes!(concat!(env!("OUT_DIR"), "/cuda_oxide_transcode.ptx"))
}

#[cfg(not(feature = "cuda-oxide-transcode"))]
fn transcode_ptx() -> &'static [u8] {
    b".version 7.0\n.target sm_52\n.address_size 64\n\0"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcode_kernel_entrypoints_are_stable() {
        let cases = [
            (
                CudaKernel::TranscodeReversible53Idct,
                "transcode_reversible53_idct",
            ),
            (
                CudaKernel::TranscodeReversible53VerticalLow,
                "transcode_reversible53_vertical_low",
            ),
            (
                CudaKernel::TranscodeReversible53VerticalHigh,
                "transcode_reversible53_vertical_high",
            ),
            (
                CudaKernel::TranscodeReversible53HorizontalLow,
                "transcode_reversible53_horizontal_low",
            ),
            (
                CudaKernel::TranscodeReversible53HorizontalHigh,
                "transcode_reversible53_horizontal_high",
            ),
            (CudaKernel::TranscodeDwt97Idct, "transcode_dwt97_idct"),
            (
                CudaKernel::TranscodeDwt97RowLift,
                "transcode_dwt97_row_lift",
            ),
            (
                CudaKernel::TranscodeDwt97ColumnLift,
                "transcode_dwt97_column_lift",
            ),
            (
                CudaKernel::TranscodeDwt97IdctBatch,
                "transcode_dwt97_idct_batch",
            ),
            (
                CudaKernel::TranscodeDwt97IdctI16Batch,
                "transcode_dwt97_idct_i16_batch",
            ),
            (
                CudaKernel::TranscodeDwt97RowLiftBatch,
                "transcode_dwt97_row_lift_batch",
            ),
            (
                CudaKernel::TranscodeDwt97RowLiftBatchCoop,
                "transcode_dwt97_row_lift_batch_coop",
            ),
            (
                CudaKernel::TranscodeDwt97ColumnLiftBatch,
                "transcode_dwt97_column_lift_batch",
            ),
            (
                CudaKernel::TranscodeDwt97QuantizeCodeblocks,
                "transcode_dwt97_quantize_codeblocks",
            ),
        ];
        for (kernel, expected) in cases {
            let entrypoint = kernel.entrypoint();
            assert_eq!(&entrypoint[..entrypoint.len() - 1], expected.as_bytes());
            assert_eq!(entrypoint.last(), Some(&0));
        }
    }

    #[test]
    fn grid_axis_overrides_preserve_the_other_axes_and_block() {
        let base = CudaLaunchGeometry::new((2, 3, 4), (16, 8, 1)).unwrap();
        let grid_y = with_grid_y(base, 9).unwrap();
        let grid_z = with_grid_z(base, 11).unwrap();
        assert_eq!(grid_y.grid(), (2, 9, 4));
        assert_eq!(grid_z.grid(), (2, 3, 11));
        assert_eq!(grid_y.block(), base.block());
        assert_eq!(grid_z.block(), base.block());
        assert!(!include_str!("transcode/launch.rs").contains("CudaLaunchGeometry {"));
    }

    #[cfg(all(feature = "cuda-oxide-transcode", j2k_cuda_oxide_transcode_built))]
    #[test]
    fn transcode_kernel_metadata_matches_generated_ptx() {
        let source =
            std::str::from_utf8(&transcode_ptx()[..transcode_ptx().len() - 1]).expect("PTX UTF-8");
        for kernel in [
            CudaKernel::TranscodeReversible53Idct,
            CudaKernel::TranscodeReversible53VerticalLow,
            CudaKernel::TranscodeReversible53VerticalHigh,
            CudaKernel::TranscodeReversible53HorizontalLow,
            CudaKernel::TranscodeReversible53HorizontalHigh,
            CudaKernel::TranscodeDwt97Idct,
            CudaKernel::TranscodeDwt97RowLift,
            CudaKernel::TranscodeDwt97ColumnLift,
            CudaKernel::TranscodeDwt97IdctBatch,
            CudaKernel::TranscodeDwt97IdctI16Batch,
            CudaKernel::TranscodeDwt97RowLiftBatch,
            CudaKernel::TranscodeDwt97RowLiftBatchCoop,
            CudaKernel::TranscodeDwt97ColumnLiftBatch,
            CudaKernel::TranscodeDwt97QuantizeCodeblocks,
        ] {
            let raw = kernel.entrypoint();
            let entrypoint = std::str::from_utf8(&raw[..raw.len() - 1]).expect("entrypoint UTF-8");
            assert!(source.contains(&format!(".visible .entry {entrypoint}(")));
        }
    }
}
