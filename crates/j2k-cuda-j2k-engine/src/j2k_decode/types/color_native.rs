// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{driver::CuDevicePtr, memory::CudaDeviceBuffer};

/// Exact-native RGB store geometry shared by the U8 and U16 batch kernels.
///
/// Unlike the legacy display-oriented RGB stores, these jobs clamp to the
/// declared component precision without scaling samples to fill the storage
/// type. `layout` is zero for NHWC and one for NCHW. `transform` is zero for
/// no color transform, one for reversible RCT, and two for irreversible ICT.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[doc(hidden)]
pub struct CudaJ2kStoreRgbNativeJob {
    /// Source width for component 0.
    pub input_width0: u32,
    /// Source width for component 1.
    pub input_width1: u32,
    /// Source width for component 2.
    pub input_width2: u32,
    /// Source x offset for component 0.
    pub source_x0: u32,
    /// Source y offset for component 0.
    pub source_y0: u32,
    /// Source x offset for component 1.
    pub source_x1: u32,
    /// Source y offset for component 1.
    pub source_y1: u32,
    /// Source x offset for component 2.
    pub source_x2: u32,
    /// Source y offset for component 2.
    pub source_y2: u32,
    /// Number of pixels copied per row.
    pub copy_width: u32,
    /// Number of rows copied.
    pub copy_height: u32,
    /// Destination output width in pixels.
    pub output_width: u32,
    /// Destination output height in rows.
    pub output_height: u32,
    /// Destination x offset.
    pub output_x: u32,
    /// Destination y offset.
    pub output_y: u32,
    /// Addend applied to component 0 after an optional inverse color transform.
    pub addend0: f32,
    /// Addend applied to component 1 after an optional inverse color transform.
    pub addend1: f32,
    /// Addend applied to component 2 after an optional inverse color transform.
    pub addend2: f32,
    /// Declared precision for component 0.
    pub bit_depth0: u32,
    /// Declared precision for component 1.
    pub bit_depth1: u32,
    /// Declared precision for component 2.
    pub bit_depth2: u32,
    /// Zero for NHWC, one for NCHW.
    pub layout: u32,
    /// Zero for none, one for reversible RCT, two for irreversible ICT.
    pub transform: u32,
    /// Reserved and initialized to zero.
    pub reserved: u32,
}

/// One exact-native RGB store item for a batched dispatch.
#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct CudaJ2kStoreRgbNativeTarget<'a> {
    /// Dense output image receiving this store. Tile stores for one image use
    /// the same index and disjoint destination rectangles.
    pub output_index: usize,
    /// Source component plane 0.
    pub plane0: &'a CudaDeviceBuffer,
    /// Source component plane 1.
    pub plane1: &'a CudaDeviceBuffer,
    /// Source component plane 2.
    pub plane2: &'a CudaDeviceBuffer,
    /// Exact-native store geometry, precision, transform, and layout.
    pub job: CudaJ2kStoreRgbNativeJob,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CudaJ2kStoreRgbNativeBatchJob {
    pub(crate) plane0_ptr: CuDevicePtr,
    pub(crate) plane1_ptr: CuDevicePtr,
    pub(crate) plane2_ptr: CuDevicePtr,
    pub(crate) output_ptr: CuDevicePtr,
    pub(crate) job: CudaJ2kStoreRgbNativeJob,
}
