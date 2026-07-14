// SPDX-License-Identifier: MIT OR Apache-2.0

//! Independent Burn tensor integration for JPEG 2000 and HTJ2K.

#![deny(missing_docs)]

use burn_core::tensor::Tensor;
use j2k::{DeviceDecodeRequest, J2kDecodeWarning, Rect};

#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "metal")]
pub mod metal;

/// Tensor memory layout.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TensorLayout {
    /// Channels precede spatial dimensions (`CHW` or `NCHW`).
    #[default]
    ChannelsFirst,
    /// Channels follow spatial dimensions (`HWC` or `NHWC`).
    ChannelsLast,
}

/// Output channel selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ChannelSelection {
    /// Preserve grayscale as one channel and otherwise produce RGB.
    #[default]
    Auto,
    /// Produce one grayscale channel.
    Gray,
    /// Produce three RGB channels.
    Rgb,
    /// Produce four RGBA channels.
    Rgba,
}

/// Floating-point sample normalization.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum FloatNormalization {
    /// Scale integer samples into the inclusive range `0..=1`.
    #[default]
    Unit,
    /// Cast integer samples without scaling.
    Raw,
    /// Unit-scale, then apply per-channel `(x - mean) / std`.
    MeanStd {
        /// Per-channel means.
        mean: Vec<f32>,
        /// Per-channel standard deviations.
        std: Vec<f32>,
    },
}

/// Options shared by all tensor decode routes.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TensorDecodeOptions {
    /// Requested tensor layout.
    pub layout: TensorLayout,
    /// Requested output channels.
    pub channels: ChannelSelection,
    /// Floating-point normalization.
    pub normalization: FloatNormalization,
}

/// Route that produced a tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorRoute {
    /// Host decode followed by a Burn upload.
    CpuStaged,
    /// Decode and conversion directly into a CUDA tensor allocation.
    CudaDirect,
    /// Metal decode followed by one packed readback and Burn upload.
    MetalStaged,
}

/// Borrowed compressed input and its decode geometry request.
#[derive(Debug, Clone, Copy)]
pub struct TensorInput<'a> {
    /// JP2, JPH, raw J2K, or raw HTJ2K bytes.
    pub encoded: &'a [u8],
    /// Full-frame, ROI, scaled, or ROI-scaled request.
    pub request: DeviceDecodeRequest,
}

impl<'a> TensorInput<'a> {
    /// Construct a full-resolution input.
    #[must_use]
    pub const fn full(encoded: &'a [u8]) -> Self {
        Self {
            encoded,
            request: DeviceDecodeRequest::Full,
        }
    }
}

/// Successful single-image tensor decode.
#[derive(Debug)]
pub struct TensorDecode<T> {
    /// Decoded Burn tensor.
    pub tensor: T,
    /// Rectangle actually decoded.
    pub decoded: Rect,
    /// Non-fatal codec warnings.
    pub warnings: Vec<J2kDecodeWarning>,
    /// Route actually used.
    pub route: TensorRoute,
}

/// Successful batch tensor decode.
#[derive(Debug)]
pub struct TensorBatchDecode<T> {
    /// Decoded Burn tensor.
    pub tensor: T,
    /// Rectangle decoded for each item, in input order.
    pub decoded: Vec<Rect>,
    /// Codec warnings for each item, in input order.
    pub warnings: Vec<Vec<J2kDecodeWarning>>,
    /// Route actually used.
    pub route: TensorRoute,
}

/// Tensor decode failure.
#[derive(Debug, thiserror::Error)]
pub enum TensorDecodeError {
    /// The codec rejected the compressed input or decode request.
    #[error("JPEG 2000 decode failed: {0}")]
    Codec(#[from] j2k::J2kError),
    /// A requested integer dtype is unsupported by the selected Burn backend.
    #[error("Burn backend does not support requested dtype {dtype:?}")]
    UnsupportedDType {
        /// Unsupported dtype.
        dtype: burn_core::tensor::DType,
    },
    /// Normalization parameters are invalid.
    #[error("invalid float normalization: {message}")]
    InvalidNormalization {
        /// Actionable validation detail.
        message: String,
    },
    /// A batch contained no inputs.
    #[error("cannot decode an empty tensor batch")]
    EmptyBatch,
    /// A batch item shape differs from the first item.
    #[error("batch item {index} has shape {actual:?}; expected {expected:?}")]
    BatchShapeMismatch {
        /// Index of the mismatching item.
        index: usize,
        /// Expected HWC shape.
        expected: [usize; 3],
        /// Actual HWC shape.
        actual: [usize; 3],
    },
    /// A particular batch item failed to decode or convert.
    #[error("batch item {index} failed: {source}")]
    BatchItem {
        /// Input index.
        index: usize,
        /// Item-specific failure.
        #[source]
        source: Box<Self>,
    },
    /// A requested allocation size overflowed `usize`.
    #[error("tensor size overflow")]
    SizeOverflow,
    /// Accelerator route failed without falling back.
    #[error("strict {route:?} route failed: {message}")]
    StrictRoute {
        /// Requested route.
        route: TensorRoute,
        /// Backend failure detail.
        message: String,
    },
}

/// Infallible Burn batcher that intentionally panics on float decode errors.
#[derive(Debug, Clone)]
pub struct PanicOnDecodeError<B> {
    options: TensorDecodeOptions,
    backend: core::marker::PhantomData<B>,
}

impl<B> PanicOnDecodeError<B> {
    /// Construct an adapter with explicit decode options.
    #[must_use]
    pub fn new(options: TensorDecodeOptions) -> Self {
        Self {
            options,
            backend: core::marker::PhantomData,
        }
    }
}

#[cfg(feature = "cpu")]
impl<'a, B> burn_core::data::dataloader::batcher::Batcher<B, TensorInput<'a>, Tensor<B, 4>>
    for PanicOnDecodeError<B>
where
    B: burn_core::tensor::backend::Backend,
{
    fn batch(&self, items: Vec<TensorInput<'a>>, device: &B::Device) -> Tensor<B, 4> {
        cpu::decode_float_batch(&items, &self.options, device)
            .unwrap_or_else(|error| panic!("j2k-ml batch decode failed: {error}"))
            .tensor
    }
}
