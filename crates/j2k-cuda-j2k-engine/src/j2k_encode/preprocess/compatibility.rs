// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    error::CudaError, execution::CudaExecutionStats, j2k_decode::CudaJ2kStridedInterleavedPixels,
};

use super::super::{CudaJ2kDeinterleavedComponents, CudaJ2kResidentComponents};

impl crate::J2kCudaEngine<'_> {
    /// Deinterleave RGB pixels, then apply forward RCT or ICT.
    ///
    /// This compatibility convenience method deliberately uses the portable
    /// two-dispatch route: one deinterleave dispatch followed by one MCT dispatch.
    #[doc(hidden)]
    pub fn j2k_deinterleave_mct_to_f32(
        &self,
        pixels: &[u8],
        num_pixels: usize,
        bit_depth: u8,
        signed: bool,
        reversible: bool,
    ) -> Result<CudaJ2kDeinterleavedComponents, CudaError> {
        let resident = self.j2k_deinterleave_mct_to_f32_resident(
            pixels, num_pixels, bit_depth, signed, reversible,
        )?;
        let execution = resident.execution();
        let components = resident.download_components()?;
        Ok(CudaJ2kDeinterleavedComponents {
            components,
            execution,
        })
    }

    /// Deinterleave RGB pixels, then apply forward RCT or ICT to the resident planes.
    ///
    /// The method remains available for source compatibility and records both
    /// dispatches in the returned execution counters.
    #[doc(hidden)]
    pub fn j2k_deinterleave_mct_to_f32_resident(
        &self,
        pixels: &[u8],
        num_pixels: usize,
        bit_depth: u8,
        signed: bool,
        reversible: bool,
    ) -> Result<CudaJ2kResidentComponents, CudaError> {
        let mut components =
            self.j2k_deinterleave_to_f32_resident(pixels, num_pixels, 3, bit_depth, signed)?;
        self.apply_separate_forward_mct(&mut components, reversible)?;
        Ok(components)
    }

    /// Deinterleave device-resident strided RGB pixels, then apply forward RCT or ICT.
    ///
    /// This source-compatible wrapper uses separate CUDA dispatches.
    #[doc(hidden)]
    pub fn j2k_deinterleave_mct_strided_to_f32_resident(
        &self,
        image: CudaJ2kStridedInterleavedPixels<'_>,
        reversible: bool,
    ) -> Result<CudaJ2kResidentComponents, CudaError> {
        if image.num_components != 3 {
            return Err(CudaError::InvalidArgument {
                message: "combined input MCT requires exactly three components".to_string(),
            });
        }
        let mut components = self.j2k_deinterleave_strided_to_f32_resident_impl(image)?;
        self.apply_separate_forward_mct(&mut components, reversible)?;
        Ok(components)
    }

    fn apply_separate_forward_mct(
        &self,
        components: &mut CudaJ2kResidentComponents,
        reversible: bool,
    ) -> Result<(), CudaError> {
        let deinterleave = components.execution;
        let mct = if reversible {
            self.j2k_forward_rct_resident(components)?
        } else {
            self.j2k_forward_ict_resident(components)?
        };
        components.execution = CudaExecutionStats::new(
            deinterleave
                .kernel_dispatches()
                .saturating_add(mct.kernel_dispatches()),
            deinterleave
                .copy_kernel_dispatches()
                .saturating_add(mct.copy_kernel_dispatches()),
            deinterleave
                .decode_kernel_dispatches()
                .saturating_add(mct.decode_kernel_dispatches()),
            deinterleave.used_hardware_decode() || mct.used_hardware_decode(),
        );
        Ok(())
    }
}
