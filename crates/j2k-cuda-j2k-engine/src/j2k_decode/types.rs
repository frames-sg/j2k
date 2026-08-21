// SPDX-License-Identifier: MIT OR Apache-2.0

mod color_native;
mod color_native_rgba;

pub(crate) use color_native::CudaJ2kStoreRgbNativeBatchJob;
pub use color_native::{CudaJ2kStoreRgbNativeJob, CudaJ2kStoreRgbNativeTarget};
pub(crate) use color_native_rgba::CudaJ2kStoreRgbaNativeBatchJob;
pub use color_native_rgba::{CudaJ2kStoreRgbaNativeJob, CudaJ2kStoreRgbaNativeTarget};

use crate::{driver::CuDevicePtr, memory::CudaDeviceBuffer};

/// CUDA-side integer rectangle for JPEG 2000 direct-plan kernels.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[doc(hidden)]
pub struct CudaJ2kRect {
    /// Inclusive minimum x coordinate.
    pub x0: u32,
    /// Inclusive minimum y coordinate.
    pub y0: u32,
    /// Exclusive maximum x coordinate.
    pub x1: u32,
    /// Exclusive maximum y coordinate.
    pub y1: u32,
}

/// One single-decomposition inverse DWT dispatch over device coefficient bands.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[doc(hidden)]
pub struct CudaJ2kIdwtJob {
    /// Output rectangle produced by the IDWT stage.
    pub rect: CudaJ2kRect,
    /// LL input band rectangle.
    pub ll_rect: CudaJ2kRect,
    /// HL input band rectangle.
    pub hl_rect: CudaJ2kRect,
    /// LH input band rectangle.
    pub lh_rect: CudaJ2kRect,
    /// HH input band rectangle.
    pub hh_rect: CudaJ2kRect,
    /// Nonzero for irreversible 9/7; zero for reversible 5/3.
    pub irreversible97: u32,
}

/// One output buffer and input band set for batched inverse DWT.
#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct CudaJ2kIdwtTarget<'a> {
    /// LL input band.
    pub ll: &'a CudaDeviceBuffer,
    /// HL input band.
    pub hl: &'a CudaDeviceBuffer,
    /// LH input band.
    pub lh: &'a CudaDeviceBuffer,
    /// HH input band.
    pub hh: &'a CudaDeviceBuffer,
    /// Output buffer for the reconstructed band.
    pub output: &'a CudaDeviceBuffer,
    /// IDWT geometry and transform metadata.
    pub job: CudaJ2kIdwtJob,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CudaJ2kIdwtMultiKernelJob {
    pub(crate) ll_ptr: u64,
    pub(crate) hl_ptr: u64,
    pub(crate) lh_ptr: u64,
    pub(crate) hh_ptr: u64,
    pub(crate) output_ptr: u64,
    pub(crate) job: CudaJ2kIdwtJob,
    pub(crate) reserved_tail: u32,
}

/// Grayscale store dispatch from f32 component samples to tightly packed Gray8.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[doc(hidden)]
pub struct CudaJ2kStoreGray8Job {
    /// Source component buffer width in samples.
    pub input_width: u32,
    /// Source x offset in samples.
    pub source_x: u32,
    /// Source y offset in samples.
    pub source_y: u32,
    /// Number of samples copied per row.
    pub copy_width: u32,
    /// Number of rows copied.
    pub copy_height: u32,
    /// Destination output width in samples.
    pub output_width: u32,
    /// Destination output height in rows.
    pub output_height: u32,
    /// Destination x offset in samples.
    pub output_x: u32,
    /// Destination y offset in samples.
    pub output_y: u32,
    /// Level-shift addend applied before quantizing to Gray8.
    pub addend: f32,
    /// Source component bit depth.
    pub bit_depth: u32,
}

/// Grayscale store dispatch from f32 component samples to tightly packed Gray16.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[doc(hidden)]
pub struct CudaJ2kStoreGray16Job {
    /// Source component buffer width in samples.
    pub input_width: u32,
    /// Source x offset in samples.
    pub source_x: u32,
    /// Source y offset in samples.
    pub source_y: u32,
    /// Number of samples copied per row.
    pub copy_width: u32,
    /// Number of rows copied.
    pub copy_height: u32,
    /// Destination output width in samples.
    pub output_width: u32,
    /// Destination output height in rows.
    pub output_height: u32,
    /// Destination x offset in samples.
    pub output_x: u32,
    /// Destination y offset in samples.
    pub output_y: u32,
    /// Level-shift addend applied before quantizing to Gray16.
    pub addend: f32,
    /// Source component bit depth.
    pub bit_depth: u32,
}

/// One Gray8 store item for a batched dispatch.
#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct CudaJ2kStoreGray8Target<'a> {
    /// Dense output index; consecutive tile stores may write disjoint rectangles.
    pub output_index: usize,
    /// Source reconstructed component plane.
    pub input: &'a CudaDeviceBuffer,
    /// Store geometry, level shift, and precision.
    pub job: CudaJ2kStoreGray8Job,
}

/// One Gray16 store item for a batched dispatch.
#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct CudaJ2kStoreGray16Target<'a> {
    /// Dense output image receiving this store.
    pub output_index: usize,
    /// Source reconstructed component plane.
    pub input: &'a CudaDeviceBuffer,
    /// Store geometry, level shift, and precision.
    pub job: CudaJ2kStoreGray16Job,
}

/// One signed GrayI16 store item for a batched dispatch.
#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct CudaJ2kStoreGrayI16Target<'a> {
    /// Dense output image receiving this store.
    pub output_index: usize,
    /// Source reconstructed component plane.
    pub input: &'a CudaDeviceBuffer,
    /// Store geometry, zero signed level shift, and precision.
    pub job: CudaJ2kStoreGray16Job,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CudaJ2kStoreGray8BatchJob {
    pub(crate) input_ptr: CuDevicePtr,
    pub(crate) output_ptr: CuDevicePtr,
    pub(crate) job: CudaJ2kStoreGray8Job,
    pub(crate) reserved_tail: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CudaJ2kStoreGray16BatchJob {
    pub(crate) input_ptr: CuDevicePtr,
    pub(crate) output_ptr: CuDevicePtr,
    pub(crate) job: CudaJ2kStoreGray16Job,
    pub(crate) reserved_tail: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CudaJ2kStoreGrayI16BatchJob {
    pub(crate) input_ptr: CuDevicePtr,
    pub(crate) output_ptr: CuDevicePtr,
    pub(crate) job: CudaJ2kStoreGray16Job,
    pub(crate) reserved_tail: u32,
}

/// In-place inverse MCT dispatch over three device f32 component planes.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[doc(hidden)]
pub struct CudaJ2kInverseMctJob {
    /// Number of samples in each component plane.
    pub len: u32,
    /// Nonzero for irreversible ICT; zero for reversible RCT.
    pub irreversible97: u32,
    /// Addend applied to output channel 0 after inverse MCT.
    pub addend0: f32,
    /// Addend applied to output channel 1 after inverse MCT.
    pub addend1: f32,
    /// Addend applied to output channel 2 after inverse MCT.
    pub addend2: f32,
}

/// RGB/RGBA store dispatch from three f32 component planes to packed 8-bit pixels.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[doc(hidden)]
pub struct CudaJ2kStoreRgb8Job {
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
    /// Addend applied to component 0 before quantizing.
    pub addend0: f32,
    /// Addend applied to component 1 before quantizing.
    pub addend1: f32,
    /// Addend applied to component 2 before quantizing.
    pub addend2: f32,
    /// Source bit depth for component 0.
    pub bit_depth0: u32,
    /// Source bit depth for component 1.
    pub bit_depth1: u32,
    /// Source bit depth for component 2.
    pub bit_depth2: u32,
    /// Nonzero to write RGBA8 with opaque alpha; zero writes RGB8.
    pub rgba: u32,
}

/// RGB/RGBA store dispatch from three f32 component planes to packed 16-bit pixels.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[doc(hidden)]
pub struct CudaJ2kStoreRgb16Job {
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
    /// Addend applied to component 0 before quantizing.
    pub addend0: f32,
    /// Addend applied to component 1 before quantizing.
    pub addend1: f32,
    /// Addend applied to component 2 before quantizing.
    pub addend2: f32,
    /// Source bit depth for component 0.
    pub bit_depth0: u32,
    /// Source bit depth for component 1.
    pub bit_depth1: u32,
    /// Source bit depth for component 2.
    pub bit_depth2: u32,
    /// Nonzero to write RGBA16 with opaque alpha; zero writes RGB16.
    pub rgba: u32,
}

/// Fused inverse RCT/ICT and packed RGB8/RGBA8 store dispatch.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[doc(hidden)]
pub struct CudaJ2kStoreRgb8MctJob {
    /// RGB/RGBA store geometry, addends, bit depths, and alpha mode.
    pub store: CudaJ2kStoreRgb8Job,
    /// Nonzero for irreversible ICT; zero for reversible RCT.
    pub irreversible97: u32,
}

/// One fused inverse MCT plus RGB8/RGBA8 store item for a batched dispatch.
#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct CudaJ2kStoreRgb8MctTarget<'a> {
    /// Source component plane 0.
    pub plane0: &'a CudaDeviceBuffer,
    /// Source component plane 1.
    pub plane1: &'a CudaDeviceBuffer,
    /// Source component plane 2.
    pub plane2: &'a CudaDeviceBuffer,
    /// Store geometry and inverse MCT parameters.
    pub job: CudaJ2kStoreRgb8MctJob,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct CudaJ2kStoreRgb8MctBatchJob {
    pub(crate) plane0_ptr: CuDevicePtr,
    pub(crate) plane1_ptr: CuDevicePtr,
    pub(crate) plane2_ptr: CuDevicePtr,
    pub(crate) output_ptr: CuDevicePtr,
    pub(crate) job: CudaJ2kStoreRgb8MctJob,
    pub(crate) reserved_tail: u32,
}

/// Fused inverse RCT/ICT and packed RGB16/RGBA16 store dispatch.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[doc(hidden)]
pub struct CudaJ2kStoreRgb16MctJob {
    /// RGB/RGBA store geometry, addends, bit depths, and alpha mode.
    pub store: CudaJ2kStoreRgb16Job,
    /// Nonzero for irreversible ICT; zero for reversible RCT.
    pub irreversible97: u32,
}

/// Device-resident interleaved JPEG 2000 input pixels with row stride metadata.
#[derive(Clone, Copy, Debug)]
#[doc(hidden)]
pub struct CudaJ2kStridedInterleavedPixels<'a> {
    /// Backing CUDA device byte buffer.
    pub buffer: &'a CudaDeviceBuffer,
    /// Byte offset to the first pixel in `buffer`.
    pub byte_offset: usize,
    /// Active input width in pixels.
    pub width: u32,
    /// Active input height in pixels.
    pub height: u32,
    /// Bytes between the start of consecutive rows.
    pub pitch_bytes: usize,
    /// Number of interleaved components per pixel.
    pub num_components: u8,
    /// Integer sample precision.
    pub bit_depth: u8,
    /// Whether integer samples are signed.
    pub signed: bool,
}
