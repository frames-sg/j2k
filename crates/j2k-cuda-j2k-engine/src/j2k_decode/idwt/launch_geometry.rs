// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::kernels::{
    j2k_idwt_multi_coop_axis_launch_geometry, j2k_idwt_multi_coop_columns_launch_geometry,
    CudaKernel, CudaLaunchGeometry,
};

pub(in crate::j2k_decode) fn idwt_vertical_97_multi_launch_geometry(
    max_columns: usize,
    max_height: usize,
    job_count: usize,
) -> Option<(CudaKernel, CudaLaunchGeometry)> {
    const COLUMNS_PER_BLOCK: usize = 4;
    const MIN_COLS4_JOBS: usize = 64;
    if job_count >= MIN_COLS4_JOBS && max_height <= 256 {
        let geometry = j2k_idwt_multi_coop_columns_launch_geometry(
            max_columns,
            max_height,
            job_count,
            COLUMNS_PER_BLOCK,
        )?;
        Some((CudaKernel::J2kIdwtVertical97MultiCols4, geometry))
    } else {
        let geometry =
            j2k_idwt_multi_coop_axis_launch_geometry(max_columns, max_height, job_count)?;
        Some((CudaKernel::J2kIdwtVertical97Multi, geometry))
    }
}
