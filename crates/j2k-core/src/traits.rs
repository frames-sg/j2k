// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::{
    accelerator::{DeviceMemoryRange, ExecutionStats, SurfaceResidency},
    backend::{BackendKind, BackendRequest},
    batch::{TileRegionScaledDecodeJob, TileRegionScaledDeviceDecodeRequest},
    context::{CodecContext, DecoderContext},
    error::CodecError,
    pixel::PixelFormat,
    row_sink::RowSink,
    sample::Sample,
    scale::Downscale,
    scratch::ScratchPool,
    types::{DecodeOutcome, Info, Rect},
};

/// Error wrapper used by row-streaming decode when either the codec or the
/// caller-provided row sink can fail.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DecodeRowsError<D, E>
where
    D: core::error::Error + 'static,
    E: core::error::Error + 'static,
{
    #[error(transparent)]
    /// Codec decode failure.
    Decode(D),
    #[error(transparent)]
    /// Caller-provided row sink failure.
    Sink(E),
}

/// Common associated types shared by image codecs.
pub trait ImageCodec {
    /// Codec-specific error type.
    type Error: CodecError;
    /// Non-fatal warning type returned in successful decode outcomes.
    type Warning: core::fmt::Debug + core::fmt::Display + Send + Sync + 'static;
    /// Caller-owned scratch pool type used to reuse allocations.
    type Pool: ScratchPool;
}

/// Decoded image data resident on a specific backend.
pub trait DeviceSurface {
    /// Backend that owns or produced the surface.
    fn backend_kind(&self) -> BackendKind;
    /// Memory residency of the surface.
    fn residency(&self) -> SurfaceResidency {
        SurfaceResidency::for_backend(self.backend_kind())
    }
    /// Surface dimensions in pixels.
    fn dimensions(&self) -> (u32, u32);
    /// Pixel format stored by the surface.
    fn pixel_format(&self) -> PixelFormat;
    /// Number of bytes represented by the surface.
    fn byte_len(&self) -> usize;
    /// Execution statistics attached to the surface.
    fn execution_stats(&self) -> ExecutionStats {
        ExecutionStats::default()
    }
    /// Backend-visible memory range, when the backend can expose one safely.
    fn memory_range(&self) -> Option<DeviceMemoryRange> {
        None
    }
}

/// Submitted device decode operation that can be waited on for completion.
pub trait DeviceSubmission {
    /// Completed output type.
    type Output;
    /// Submission or decode error type.
    type Error;

    /// Wait for the submission and return its output.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if submission or device execution fails.
    fn wait(self) -> Result<Self::Output, Self::Error>;
}

/// Already-completed submission used by synchronous fallback paths.
#[derive(Debug)]
#[doc(hidden)]
pub struct ReadySubmission<T, E>(Result<T, E>);

impl<T, E> ReadySubmission<T, E> {
    /// Wrap an immediate result as a submission.
    pub fn from_result(result: Result<T, E>) -> Self {
        Self(result)
    }
}

impl<T, E> DeviceSubmission for ReadySubmission<T, E> {
    type Output = T;
    type Error = E;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        self.0
    }
}

/// Mutable device session that tracks submitted backend work.
#[doc(hidden)]
pub trait DeviceSubmitSession {
    /// Record a submitted device operation.
    fn record_submit(&mut self);
}

/// Record a device submission and wrap an immediate result as ready.
#[doc(hidden)]
pub fn submit_ready_device<S, T, E>(
    session: &mut S,
    submit: impl FnOnce(&mut S) -> Result<T, E>,
) -> ReadySubmission<T, E>
where
    S: DeviceSubmitSession + ?Sized,
{
    session.record_submit();
    ReadySubmission::from_result(submit(session))
}

/// Borrowed-image decode API for codecs that parse compressed bytes directly.
pub trait ImageDecode<'a>: ImageCodec + Sized + 'a {
    /// Borrowed parse product that can later construct a decoder.
    type View: 'a;

    /// Inspect metadata without decoding pixels.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the compressed input is invalid or unsupported.
    fn inspect(input: &'a [u8]) -> Result<Info, Self::Error>;
    /// Parse compressed bytes into a borrowed view.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the compressed input cannot be parsed.
    fn parse(input: &'a [u8]) -> Result<Self::View, Self::Error>;
    /// Build a decoder from a parsed view.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the parsed view is unsupported or inconsistent.
    fn from_view(view: Self::View) -> Result<Self, Self::Error>;

    /// Decode the full image into caller-owned output.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] for invalid input, unsupported output, or an
    /// undersized or invalid output layout.
    fn decode_into(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    /// Decode the full image into caller-owned output with reusable scratch.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] for invalid input, unsupported output, scratch
    /// failure, or an invalid output layout.
    fn decode_into_with_scratch(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    /// Decode a source-coordinate region into caller-owned output.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the input, region, output layout, or scratch
    /// state cannot be decoded.
    fn decode_region_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    /// Decode the full image at reduced resolution into caller-owned output.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the input, scale, output layout, or scratch
    /// state cannot be decoded.
    fn decode_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    /// Decode a source-coordinate region at reduced resolution into caller-owned output.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the input, region, scale, output layout, or
    /// scratch state cannot be decoded.
    fn decode_region_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;
}

/// Adapter hook for decoders whose host-output path delegates to a CPU decoder.
///
/// GPU adapter crates implement this trait when their public decoder wraps a
/// CPU decoder for host output but has backend-specific device submission
/// methods. The blanket [`ImageDecode`] impl below keeps the CPU-host
/// delegation in one place.
#[doc(hidden)]
pub trait CpuBackedImageDecode<'a>: ImageCodec + Sized + 'a {
    /// CPU decoder that owns the host-output implementation.
    type Cpu: ImageDecode<'a, Pool = Self::Pool>;
    /// Borrowed parse product used by this adapter.
    type View: 'a;

    /// Inspect metadata through the CPU codec and map it to core info.
    fn inspect_cpu(input: &'a [u8]) -> Result<Info, Self::Error>;
    /// Parse compressed bytes through the CPU codec or adapter view.
    fn parse_cpu(input: &'a [u8]) -> Result<Self::View, Self::Error>;
    /// Build this adapter from a parsed CPU view.
    fn from_cpu_view(view: Self::View) -> Result<Self, Self::Error>;
    /// Borrow the wrapped CPU decoder mutably.
    fn cpu_decoder_mut(&mut self) -> &mut Self::Cpu;
    /// Convert a CPU decode outcome into this adapter's warning type.
    fn map_cpu_outcome(
        outcome: DecodeOutcome<<Self::Cpu as ImageCodec>::Warning>,
    ) -> DecodeOutcome<Self::Warning>;
}

#[doc(hidden)]
impl<'a, T> ImageDecode<'a> for T
where
    T: CpuBackedImageDecode<'a>,
    <T::Cpu as ImageCodec>::Error: Into<T::Error>,
{
    type View = T::View;

    fn inspect(input: &'a [u8]) -> Result<Info, Self::Error> {
        T::inspect_cpu(input)
    }

    fn parse(input: &'a [u8]) -> Result<Self::View, Self::Error> {
        T::parse_cpu(input)
    }

    fn from_view(view: Self::View) -> Result<Self, Self::Error> {
        T::from_cpu_view(view)
    }

    fn decode_into(
        &mut self,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        let outcome = self
            .cpu_decoder_mut()
            .decode_into(out, stride, fmt)
            .map_err(Into::into)?;
        Ok(T::map_cpu_outcome(outcome))
    }

    fn decode_into_with_scratch(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        let outcome = self
            .cpu_decoder_mut()
            .decode_into_with_scratch(pool, out, stride, fmt)
            .map_err(Into::into)?;
        Ok(T::map_cpu_outcome(outcome))
    }

    fn decode_region_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        let outcome = self
            .cpu_decoder_mut()
            .decode_region_into(pool, out, stride, fmt, roi)
            .map_err(Into::into)?;
        Ok(T::map_cpu_outcome(outcome))
    }

    fn decode_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        let outcome = self
            .cpu_decoder_mut()
            .decode_scaled_into(pool, out, stride, fmt, scale)
            .map_err(Into::into)?;
        Ok(T::map_cpu_outcome(outcome))
    }

    fn decode_region_scaled_into(
        &mut self,
        pool: &mut Self::Pool,
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error> {
        let outcome = self
            .cpu_decoder_mut()
            .decode_region_scaled_into(pool, out, stride, fmt, roi, scale)
            .map_err(Into::into)?;
        Ok(T::map_cpu_outcome(outcome))
    }
}

/// Decode API for implementations that can submit work to a device backend.
pub trait ImageDecodeSubmit<'a>: ImageDecode<'a> {
    /// Mutable session state shared across submissions.
    type Session: Default + Send;
    /// Device surface returned by completed submissions.
    type DeviceSurface: DeviceSurface;
    /// Submission handle type.
    type SubmittedSurface: DeviceSubmission<Output = Self::DeviceSurface, Error = Self::Error>;

    /// Submit full-image decode to the requested backend.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the request is invalid, unsupported, or
    /// cannot be submitted.
    fn submit_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;

    /// Submit region decode to the requested backend.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the region or backend request is invalid,
    /// unsupported, or cannot be submitted.
    fn submit_region_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;

    /// Submit reduced-resolution decode to the requested backend.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the scale or backend request is invalid,
    /// unsupported, or cannot be submitted.
    fn submit_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;

    /// Submit region decode at reduced resolution to the requested backend.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the region, scale, or backend request is
    /// invalid, unsupported, or cannot be submitted.
    fn submit_region_scaled_to_device(
        &mut self,
        session: &mut Self::Session,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;
}

/// Synchronous device-output decode API.
pub trait ImageDecodeDevice<'a>: ImageDecode<'a> {
    /// Device surface returned by decode calls.
    type DeviceSurface: DeviceSurface;

    /// Decode the full image to the requested backend.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if submission or device execution fails.
    fn decode_to_device(
        &mut self,
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<<Self as ImageDecodeDevice<'a>>::DeviceSurface, Self::Error>
    where
        Self: ImageDecodeSubmit<'a, DeviceSurface = <Self as ImageDecodeDevice<'a>>::DeviceSurface>,
    {
        let mut session = <Self as ImageDecodeSubmit<'a>>::Session::default();
        <Self as ImageDecodeSubmit<'a>>::submit_to_device(self, &mut session, fmt, backend)?.wait()
    }

    /// Decode a source-coordinate region to the requested backend.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if the region is invalid or submission or device
    /// execution fails.
    fn decode_region_to_device(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<<Self as ImageDecodeDevice<'a>>::DeviceSurface, Self::Error>
    where
        Self: ImageDecodeSubmit<'a, DeviceSurface = <Self as ImageDecodeDevice<'a>>::DeviceSurface>,
    {
        let mut session = <Self as ImageDecodeSubmit<'a>>::Session::default();
        <Self as ImageDecodeSubmit<'a>>::submit_region_to_device(
            self,
            &mut session,
            fmt,
            roi,
            backend,
        )?
        .wait()
    }

    /// Decode the full image at reduced resolution to the requested backend.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if the scale is unsupported or submission or
    /// device execution fails.
    fn decode_scaled_to_device(
        &mut self,
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<<Self as ImageDecodeDevice<'a>>::DeviceSurface, Self::Error>
    where
        Self: ImageDecodeSubmit<'a, DeviceSurface = <Self as ImageDecodeDevice<'a>>::DeviceSurface>,
    {
        let mut session = <Self as ImageDecodeSubmit<'a>>::Session::default();
        <Self as ImageDecodeSubmit<'a>>::submit_scaled_to_device(
            self,
            &mut session,
            fmt,
            scale,
            backend,
        )?
        .wait()
    }

    /// Decode a source-coordinate region at reduced resolution to the requested backend.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if the region or scale is invalid or submission
    /// or device execution fails.
    fn decode_region_scaled_to_device(
        &mut self,
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<<Self as ImageDecodeDevice<'a>>::DeviceSurface, Self::Error>
    where
        Self: ImageDecodeSubmit<'a, DeviceSurface = <Self as ImageDecodeDevice<'a>>::DeviceSurface>,
    {
        let mut session = <Self as ImageDecodeSubmit<'a>>::Session::default();
        <Self as ImageDecodeSubmit<'a>>::submit_region_scaled_to_device(
            self,
            &mut session,
            fmt,
            roi,
            scale,
            backend,
        )?
        .wait()
    }
}

/// Row-streaming decode API for large images or stripe-oriented callers.
pub trait ImageDecodeRows<'a, S: Sample>: ImageDecode<'a> {
    /// Decode rows into `sink` without requiring one contiguous output buffer.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeRowsError::Decode`] for codec failures or
    /// [`DecodeRowsError::Sink`] when the destination rejects a row.
    #[expect(
        clippy::type_complexity,
        reason = "the public contract must preserve distinct codec and sink error types"
    )]
    fn decode_rows<R: RowSink<S>>(
        &mut self,
        sink: &mut R,
    ) -> Result<DecodeOutcome<Self::Warning>, DecodeRowsError<Self::Error, R::Error>>;
}

/// Stateless tile-batch decode helpers that reuse caller-owned context.
pub trait TileBatchDecode: ImageCodec {
    /// Codec-specific context cached across tiles.
    type Context: CodecContext;

    /// Decode one tile into caller-owned output.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the tile input or output layout cannot be decoded.
    fn decode_tile(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    /// Decode one tile region into caller-owned output.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the tile, region, or output layout cannot be decoded.
    fn decode_tile_region(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        roi: Rect,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    /// Decode one tile at reduced resolution into caller-owned output.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the tile, scale, or output layout cannot be decoded.
    fn decode_tile_scaled(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
        stride: usize,
        fmt: PixelFormat,
        scale: Downscale,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;

    /// Decode one tile region at reduced resolution into caller-owned output.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the tile, region, scale, or output layout
    /// cannot be decoded.
    fn decode_tile_region_scaled(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        fmt: PixelFormat,
        job: TileRegionScaledDecodeJob<'_, '_>,
    ) -> Result<DecodeOutcome<Self::Warning>, Self::Error>;
}

/// Tile-batch helpers that return synchronous device surfaces.
pub trait TileBatchDecodeDevice: ImageCodec {
    /// Codec-specific context cached across tiles.
    type Context: CodecContext;
    /// Device surface returned by decode calls.
    type DeviceSurface: DeviceSurface;

    /// Decode one tile to the requested backend.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if the tile cannot be submitted or device execution fails.
    fn decode_tile_to_device(
        ctx: &mut DecoderContext<<Self as TileBatchDecodeDevice>::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<<Self as TileBatchDecodeDevice>::DeviceSurface, Self::Error>
    where
        Self: TileBatchDecodeSubmit<
            Context = <Self as TileBatchDecodeDevice>::Context,
            DeviceSurface = <Self as TileBatchDecodeDevice>::DeviceSurface,
        >,
    {
        let mut session = <Self as TileBatchDecodeSubmit>::Session::default();
        <Self as TileBatchDecodeSubmit>::submit_tile_to_device(
            ctx,
            &mut session,
            pool,
            input,
            fmt,
            backend,
        )?
        .wait()
    }

    /// Decode one tile region to the requested backend.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if the region is invalid or submission or device
    /// execution fails.
    fn decode_tile_region_to_device(
        ctx: &mut DecoderContext<<Self as TileBatchDecodeDevice>::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<<Self as TileBatchDecodeDevice>::DeviceSurface, Self::Error>
    where
        Self: TileBatchDecodeSubmit<
            Context = <Self as TileBatchDecodeDevice>::Context,
            DeviceSurface = <Self as TileBatchDecodeDevice>::DeviceSurface,
        >,
    {
        let mut session = <Self as TileBatchDecodeSubmit>::Session::default();
        <Self as TileBatchDecodeSubmit>::submit_tile_region_to_device(
            ctx,
            &mut session,
            pool,
            input,
            fmt,
            roi,
            backend,
        )?
        .wait()
    }

    /// Decode one tile at reduced resolution to the requested backend.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if the scale is unsupported or submission or
    /// device execution fails.
    fn decode_tile_scaled_to_device(
        ctx: &mut DecoderContext<<Self as TileBatchDecodeDevice>::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<<Self as TileBatchDecodeDevice>::DeviceSurface, Self::Error>
    where
        Self: TileBatchDecodeSubmit<
            Context = <Self as TileBatchDecodeDevice>::Context,
            DeviceSurface = <Self as TileBatchDecodeDevice>::DeviceSurface,
        >,
    {
        let mut session = <Self as TileBatchDecodeSubmit>::Session::default();
        <Self as TileBatchDecodeSubmit>::submit_tile_scaled_to_device(
            ctx,
            &mut session,
            pool,
            input,
            fmt,
            scale,
            backend,
        )?
        .wait()
    }

    /// Decode one tile region at reduced resolution to the requested backend.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if the region or scale is invalid or submission
    /// or device execution fails.
    fn decode_tile_region_scaled_to_device(
        ctx: &mut DecoderContext<<Self as TileBatchDecodeDevice>::Context>,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<<Self as TileBatchDecodeDevice>::DeviceSurface, Self::Error>
    where
        Self: TileBatchDecodeSubmit<
            Context = <Self as TileBatchDecodeDevice>::Context,
            DeviceSurface = <Self as TileBatchDecodeDevice>::DeviceSurface,
        >,
    {
        let mut session = <Self as TileBatchDecodeSubmit>::Session::default();
        <Self as TileBatchDecodeSubmit>::submit_tile_region_scaled_to_device(
            ctx,
            &mut session,
            pool,
            TileRegionScaledDeviceDecodeRequest {
                input,
                fmt,
                roi,
                scale,
                backend,
            },
        )?
        .wait()
    }
}

/// Full-tile batch helpers that decode many independent tiles to device surfaces.
pub trait TileBatchDecodeManyDevice: ImageCodec {
    /// Codec-specific context cached across tiles.
    type Context: CodecContext;
    /// Device surface returned by decode calls.
    type DeviceSurface: DeviceSurface;

    /// Decode many full tiles to the requested backend, preserving input order.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] if any tile cannot be decoded by the requested backend.
    fn decode_tiles_to_device(
        ctx: &mut DecoderContext<Self::Context>,
        pool: &mut Self::Pool,
        inputs: &[&[u8]],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Vec<Self::DeviceSurface>, Self::Error>;
}

/// Tile-batch helpers that queue device submissions.
pub trait TileBatchDecodeSubmit: ImageCodec {
    /// Codec-specific context cached across tiles.
    type Context: CodecContext;
    /// Mutable session state shared across submissions.
    type Session: Default + Send;
    /// Device surface returned by completed submissions.
    type DeviceSurface: DeviceSurface;
    /// Submission handle type.
    type SubmittedSurface: DeviceSubmission<Output = Self::DeviceSurface, Error = Self::Error>;

    /// Submit one full tile to the requested backend.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the request is invalid, unsupported, or
    /// cannot be submitted.
    fn submit_tile_to_device(
        ctx: &mut DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;

    /// Submit one tile region to the requested backend.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the region or backend request is invalid,
    /// unsupported, or cannot be submitted.
    fn submit_tile_region_to_device(
        ctx: &mut DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        roi: Rect,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;

    /// Submit one tile at reduced resolution to the requested backend.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the scale or backend request is invalid,
    /// unsupported, or cannot be submitted.
    fn submit_tile_scaled_to_device(
        ctx: &mut DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        input: &[u8],
        fmt: PixelFormat,
        scale: Downscale,
        backend: BackendRequest,
    ) -> Result<Self::SubmittedSurface, Self::Error>;

    /// Submit one tile region at reduced resolution to the requested backend.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the request is invalid, unsupported, or
    /// cannot be submitted.
    fn submit_tile_region_scaled_to_device(
        ctx: &mut DecoderContext<Self::Context>,
        session: &mut Self::Session,
        pool: &mut Self::Pool,
        request: TileRegionScaledDeviceDecodeRequest<'_>,
    ) -> Result<Self::SubmittedSurface, Self::Error>;
}

/// Tile payload decompression API for container codecs such as Deflate, Zstd,
/// LZW, and uncompressed data.
pub trait TileDecompress {
    /// Codec-specific error type.
    type Error: CodecError;
    /// Caller-owned scratch pool type.
    type Pool: ScratchPool;

    /// Return the expected decoded size when the compressed payload encodes it.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the payload header is invalid or unsupported.
    fn expected_size(input: &[u8]) -> Result<Option<usize>, Self::Error>;

    /// Decompress `input` into `out`, returning the number of bytes written.
    ///
    /// # Errors
    ///
    /// Returns [`Self::Error`] when the payload is invalid, scratch allocation
    /// fails, or `out` cannot hold the decoded bytes.
    fn decompress_into(
        pool: &mut Self::Pool,
        input: &[u8],
        out: &mut [u8],
    ) -> Result<usize, Self::Error>;
}
