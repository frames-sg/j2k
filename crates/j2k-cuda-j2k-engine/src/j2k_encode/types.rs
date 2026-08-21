// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{execution::CudaExecutionStats, memory::CudaDeviceBuffer};

#[derive(Clone, Copy)]
pub(super) struct J2kStridedDeinterleaveLaunch<'a> {
    pub(super) pixels: &'a CudaDeviceBuffer,
    pub(super) output: &'a CudaDeviceBuffer,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) byte_offset: usize,
    pub(super) pitch_bytes: usize,
    pub(super) num_components: u8,
    pub(super) bit_depth: u8,
    pub(super) signed: bool,
}

/// Resident f32 component planes produced by CUDA JPEG 2000 encode preparation.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaJ2kResidentComponents {
    pub(crate) buffer: CudaDeviceBuffer,
    pub(crate) num_pixels: usize,
    pub(crate) num_components: u8,
    pub(crate) execution: CudaExecutionStats,
}

/// Host-visible component planes produced by CUDA pixel deinterleave.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaJ2kDeinterleavedComponents {
    pub(crate) components: Vec<Vec<f32>>,
    pub(crate) execution: CudaExecutionStats,
}

/// Forward 5/3 DWT output and level metadata.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaDwt53Output {
    pub(crate) transformed: Vec<f32>,
    pub(crate) levels: Vec<CudaDwt53LevelShape>,
    pub(crate) ll_width: u32,
    pub(crate) ll_height: u32,
    pub(crate) execution: CudaExecutionStats,
}

/// Resident forward 5/3 DWT output and level metadata.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaResidentDwt53Output {
    pub(crate) buffer: CudaDeviceBuffer,
    pub(crate) sample_count: usize,
    pub(crate) levels: Vec<CudaDwt53LevelShape>,
    pub(crate) ll_width: u32,
    pub(crate) ll_height: u32,
    pub(crate) execution: CudaExecutionStats,
}

/// Forward 9/7 DWT output and level metadata.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaDwt97Output {
    pub(crate) transformed: Vec<f32>,
    pub(crate) levels: Vec<CudaDwt53LevelShape>,
    pub(crate) ll_width: u32,
    pub(crate) ll_height: u32,
    pub(crate) execution: CudaExecutionStats,
}

/// Resident forward 9/7 DWT output and level metadata.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaResidentDwt97Output {
    pub(crate) buffer: CudaDeviceBuffer,
    pub(crate) sample_count: usize,
    pub(crate) levels: Vec<CudaDwt53LevelShape>,
    pub(crate) ll_width: u32,
    pub(crate) ll_height: u32,
    pub(crate) execution: CudaExecutionStats,
}

/// JPEG 2000 sub-band quantization parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub struct CudaJ2kQuantizeJob {
    /// Quantization step-size exponent.
    pub step_exponent: u16,
    /// Quantization step-size mantissa.
    pub step_mantissa: u16,
    /// Nominal range bits for this sub-band.
    pub range_bits: u8,
    /// Whether to use reversible integer quantization.
    pub reversible: bool,
}

/// Resident strided sub-band rectangle and quantization parameters.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub struct CudaJ2kQuantizeSubbandRegionJob {
    /// X offset, in f32 samples, of the sub-band rectangle inside the resident plane.
    pub x0: u32,
    /// Y offset, in f32 samples, of the sub-band rectangle inside the resident plane.
    pub y0: u32,
    /// Sub-band rectangle width in samples.
    pub width: u32,
    /// Sub-band rectangle height in samples.
    pub height: u32,
    /// Resident source row stride in f32 samples.
    pub stride: u32,
    /// Quantization parameters applied to every source sample.
    pub quantization: CudaJ2kQuantizeJob,
}

/// Quantized JPEG 2000 sub-band coefficients and execution metadata.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaJ2kQuantizedSubband {
    pub(crate) coefficients: Vec<i32>,
    pub(crate) execution: CudaExecutionStats,
}

/// Device-resident quantized JPEG 2000 sub-band coefficients and execution metadata.
#[derive(Debug)]
#[doc(hidden)]
pub struct CudaJ2kResidentQuantizedSubband {
    pub(crate) coefficients: CudaDeviceBuffer,
    pub(crate) coefficient_count: usize,
    pub(crate) execution: CudaExecutionStats,
}

/// Shape metadata for one forward 5/3 DWT level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub struct CudaDwt53LevelShape {
    /// Input level width.
    pub width: u32,
    /// Input level height.
    pub height: u32,
    /// Low-pass width.
    pub low_width: u32,
    /// Low-pass height.
    pub low_height: u32,
    /// High-pass width.
    pub high_width: u32,
    /// High-pass height.
    pub high_height: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaDwt53Pass {
    pub(crate) full_width: u32,
    pub(crate) current_width: u32,
    pub(crate) current_height: u32,
    pub(crate) low_extent: u32,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaDwt53LevelPass {
    pub(crate) full_width: u32,
    pub(crate) current_width: u32,
    pub(crate) current_height: u32,
}
