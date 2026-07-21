use j2k_core::PixelFormat;
use j2k_native::{J2kDirectGrayscalePlan, J2kDirectGrayscaleStep, J2kWaveletTransform};

use crate::{allocation::HostPhaseBudget, Error};

mod accessors;
mod classic;
mod ht;
mod required_regions;
mod shared;
#[cfg(test)]
mod tests;

use self::{
    classic::append_classic_subband,
    ht::append_ht_subband,
    required_regions::required_regions_for_direct_plan,
    shared::{convert_store_step, CudaPlanOwners},
};

const EMPTY_CUDA_PLAN: &str = "strict CUDA plan contains no entropy code blocks";
const MIXED_TRANSFORMS_UNSUPPORTED: &str = "strict CUDA HTJ2K plan contains mixed DWT transforms";
const PLAN_PAYLOAD_TOO_LARGE: &str = "strict CUDA HTJ2K plan payload is too large";
const PLAN_OUTPUT_RECT_MISMATCH: &str =
    "strict CUDA HTJ2K plan store does not fit the requested output rectangle";

/// CUDA-side DWT transform selector for a flat HTJ2K plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub(crate) enum CudaHtj2kTransform {
    /// Reversible 5/3 transform.
    Reversible53,
    /// Irreversible 9/7 transform.
    Irreversible97,
}

/// Stable CUDA-side identifier for a direct-plan coefficient band.
pub(crate) type CudaHtj2kBandId = u32;

impl CudaHtj2kTransform {
    pub(crate) fn from_native(value: J2kWaveletTransform) -> Self {
        match value {
            J2kWaveletTransform::Reversible53 => Self::Reversible53,
            J2kWaveletTransform::Irreversible97 => Self::Irreversible97,
        }
    }
}

/// Flat POD HTJ2K code-block metadata consumed by CUDA kernels.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub(crate) struct CudaHtj2kCodeBlock {
    /// Index of the parent sub-band in [`CudaHtj2kDecodePlan::subbands`].
    pub(crate) subband_index: u32,
    /// Byte offset into [`CudaHtj2kDecodePlan::payload`].
    pub(crate) payload_offset: u64,
    /// Total payload byte length for this code block.
    pub(crate) payload_len: u32,
    /// Cleanup segment length in bytes.
    pub(crate) cleanup_length: u32,
    /// Refinement segment length in bytes.
    pub(crate) refinement_length: u32,
    /// X offset within the target sub-band coefficient buffer.
    pub(crate) output_x: u32,
    /// Y offset within the target sub-band coefficient buffer.
    pub(crate) output_y: u32,
    /// Code-block width in samples.
    pub(crate) width: u32,
    /// Code-block height in samples.
    pub(crate) height: u32,
    /// Output row stride, in samples.
    pub(crate) output_stride: u32,
    /// Missing most-significant bit planes.
    pub(crate) missing_bit_planes: u8,
    /// Number of coding passes present.
    pub(crate) number_of_coding_passes: u8,
    /// Total coded bitplanes for the parent sub-band.
    pub(crate) num_bitplanes: u8,
    /// Nonzero when vertically causal context was enabled.
    pub(crate) stripe_causal: u8,
    /// Dequantization step to apply to decoded coefficients.
    pub(crate) dequantization_step: f32,
}

/// Flat POD sub-band geometry consumed by CUDA kernels.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub(crate) struct CudaHtj2kSubband {
    /// Stable CUDA direct-plan band id.
    pub(crate) band_id: CudaHtj2kBandId,
    /// Absolute x0 coordinate in component space.
    pub(crate) x0: u32,
    /// Absolute y0 coordinate in component space.
    pub(crate) y0: u32,
    /// Absolute x1 coordinate in component space.
    pub(crate) x1: u32,
    /// Absolute y1 coordinate in component space.
    pub(crate) y1: u32,
    /// Sub-band width in samples.
    pub(crate) width: u32,
    /// Sub-band height in samples.
    pub(crate) height: u32,
    /// First code-block index for this sub-band.
    pub(crate) code_block_start: u32,
    /// Number of code blocks for this sub-band.
    pub(crate) code_block_count: u32,
}

/// Flat classic JPEG 2000 code-block metadata consumed by CUDA kernels.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub(crate) struct CudaClassicCodeBlock {
    pub(crate) subband_index: u32,
    pub(crate) payload_offset: u64,
    pub(crate) payload_len: u32,
    pub(crate) segment_start: u32,
    pub(crate) segment_count: u32,
    pub(crate) output_x: u32,
    pub(crate) output_y: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) output_stride: u32,
    pub(crate) missing_bit_planes: u8,
    pub(crate) number_of_coding_passes: u8,
    pub(crate) total_bitplanes: u8,
    pub(crate) sub_band_type: u8,
    pub(crate) style_flags: u32,
    pub(crate) strict: bool,
    pub(crate) dequantization_step: f32,
}

/// Flat classic JPEG 2000 segment metadata consumed by CUDA kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct CudaClassicSegment {
    pub(crate) data_offset: u32,
    pub(crate) data_length: u32,
    pub(crate) start_coding_pass: u8,
    pub(crate) end_coding_pass: u8,
    pub(crate) use_arithmetic: bool,
}

/// Flat classic JPEG 2000 sub-band geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct CudaClassicSubband {
    pub(crate) band_id: CudaHtj2kBandId,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) code_block_start: u32,
    pub(crate) code_block_count: u32,
}

/// Flat POD IDWT step consumed by CUDA kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct CudaHtj2kIdwtStep {
    /// Stable identifier of the output coefficient band produced by this step.
    pub(crate) output_band_id: CudaHtj2kBandId,
    /// DWT transform to apply.
    pub(crate) transform: CudaHtj2kTransform,
    /// Output rectangle.
    pub(crate) rect: CudaHtj2kRect,
    /// LL input band id.
    pub(crate) ll_band_id: CudaHtj2kBandId,
    /// LL input rectangle.
    pub(crate) ll_rect: CudaHtj2kRect,
    /// HL input band id.
    pub(crate) hl_band_id: CudaHtj2kBandId,
    /// HL input rectangle.
    pub(crate) hl_rect: CudaHtj2kRect,
    /// LH input band id.
    pub(crate) lh_band_id: CudaHtj2kBandId,
    /// LH input rectangle.
    pub(crate) lh_rect: CudaHtj2kRect,
    /// HH input band id.
    pub(crate) hh_band_id: CudaHtj2kBandId,
    /// HH input rectangle.
    pub(crate) hh_rect: CudaHtj2kRect,
}

/// Flat POD store step consumed by CUDA kernels.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub(crate) struct CudaHtj2kStoreStep {
    /// Stable identifier of the input coefficient band.
    pub(crate) input_band_id: CudaHtj2kBandId,
    /// Source rectangle.
    pub(crate) input_rect: CudaHtj2kRect,
    /// Source x offset.
    pub(crate) source_x: u32,
    /// Source y offset.
    pub(crate) source_y: u32,
    /// Number of samples copied per row.
    pub(crate) copy_width: u32,
    /// Number of rows copied.
    pub(crate) copy_height: u32,
    /// Destination row width.
    pub(crate) output_width: u32,
    /// Destination height.
    pub(crate) output_height: u32,
    /// Destination x offset.
    pub(crate) output_x: u32,
    /// Destination y offset.
    pub(crate) output_y: u32,
    /// Constant level-shift addend.
    pub(crate) addend: f32,
}

/// Flat POD rectangle used inside CUDA HTJ2K plan metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct CudaHtj2kRect {
    /// Inclusive left coordinate.
    pub(crate) x0: u32,
    /// Inclusive top coordinate.
    pub(crate) y0: u32,
    /// Exclusive right coordinate.
    pub(crate) x1: u32,
    /// Exclusive bottom coordinate.
    pub(crate) y1: u32,
}

/// Flat CUDA HTJ2K decode plan.
///
/// The plan is move-only because its payload and descriptor vectors can
/// approach the shared host-allocation cap. Borrow it after construction.
#[derive(Debug)]
pub(crate) struct CudaHtj2kDecodePlan {
    #[cfg_attr(
        not(feature = "cuda-runtime"),
        expect(
            dead_code,
            reason = "output dimensions are consumed only by CUDA decode routes"
        )
    )]
    dimensions: (u32, u32),
    #[cfg_attr(
        not(feature = "cuda-runtime"),
        expect(
            dead_code,
            reason = "sample metadata is consumed only by CUDA decode routes"
        )
    )]
    bit_depth: u8,
    #[cfg_attr(
        not(feature = "cuda-runtime"),
        expect(
            dead_code,
            reason = "output format is consumed only by CUDA decode routes"
        )
    )]
    output_format: PixelFormat,
    #[cfg_attr(
        not(feature = "cuda-runtime"),
        expect(
            dead_code,
            reason = "output origin is consumed only by CUDA decode routes"
        )
    )]
    output_origin: (u32, u32),
    #[cfg_attr(
        not(feature = "cuda-runtime"),
        expect(
            dead_code,
            reason = "transform metadata is consumed only by CUDA decode routes"
        )
    )]
    transform: CudaHtj2kTransform,
    payload: Vec<u8>,
    code_blocks: Vec<CudaHtj2kCodeBlock>,
    classic_code_blocks: Vec<CudaClassicCodeBlock>,
    classic_segments: Vec<CudaClassicSegment>,
    #[cfg_attr(
        not(feature = "cuda-runtime"),
        expect(
            dead_code,
            reason = "classic subband metadata is consumed only by CUDA decode routes"
        )
    )]
    classic_subbands: Vec<CudaClassicSubband>,
    #[cfg_attr(
        not(feature = "cuda-runtime"),
        expect(
            dead_code,
            reason = "subband metadata is consumed only by CUDA decode routes"
        )
    )]
    subbands: Vec<CudaHtj2kSubband>,
    #[cfg_attr(
        not(feature = "cuda-runtime"),
        expect(
            dead_code,
            reason = "IDWT metadata is consumed only by CUDA decode routes"
        )
    )]
    idwt_steps: Vec<CudaHtj2kIdwtStep>,
    #[cfg_attr(
        not(feature = "cuda-runtime"),
        expect(
            dead_code,
            reason = "store metadata is consumed only by CUDA decode routes"
        )
    )]
    store_steps: Vec<CudaHtj2kStoreStep>,
}

impl CudaHtj2kDecodePlan {
    pub(crate) fn from_grayscale_direct_plan(
        plan: &J2kDirectGrayscalePlan,
        output_format: PixelFormat,
        output_origin: (u32, u32),
    ) -> Result<Self, Error> {
        Self::from_grayscale_direct_plan_region(plan, output_format, output_origin, plan.dimensions)
    }

    pub(crate) fn from_grayscale_direct_plan_region(
        plan: &J2kDirectGrayscalePlan,
        output_format: PixelFormat,
        output_origin: (u32, u32),
        output_dimensions: (u32, u32),
    ) -> Result<Self, Error> {
        let (mut owners, retained_plan_capacity) = CudaPlanOwners::from_plan(plan)?;
        let required_regions = if output_origin == (0, 0) && output_dimensions == plan.dimensions {
            None
        } else {
            Some(required_regions_for_direct_plan(
                plan,
                retained_plan_capacity,
            )?)
        };

        for step in &plan.steps {
            match step {
                J2kDirectGrayscaleStep::HtSubBand(subband) => {
                    append_ht_subband(&mut owners, subband, required_regions.as_ref())?;
                }
                J2kDirectGrayscaleStep::ClassicSubBand(subband) => {
                    append_classic_subband(&mut owners, subband, required_regions.as_ref())?;
                }
                J2kDirectGrayscaleStep::Idwt(step) => owners.append_idwt(*step)?,
                J2kDirectGrayscaleStep::Store(step) => {
                    owners.store_steps.push(convert_store_step(
                        *step,
                        output_origin,
                        output_dimensions,
                    )?);
                }
            }
        }

        owners.finish(plan, output_format, output_origin, output_dimensions)
    }
}
