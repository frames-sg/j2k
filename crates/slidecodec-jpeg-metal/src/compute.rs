// SPDX-License-Identifier: Apache-2.0

#[cfg(target_os = "macos")]
use std::{cell::RefCell, mem::size_of};

#[cfg(target_os = "macos")]
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use slidecodec_core::{PixelFormat, Rect};
use slidecodec_jpeg::{
    ColorSpace as JpegColorSpace, ComponentRowWriter, Decoder as CpuDecoder,
    __private::{JpegMetalFast420PacketV1, MetalHuffmanTable as PacketHuffmanTable},
};

use crate::viewport::ViewportTile;
use crate::{Error, Surface};

#[cfg(target_os = "macos")]
const SHADER_SOURCE: &str = include_str!("shaders.metal");

#[cfg(target_os = "macos")]
const MODE_GRAY: u32 = 0;
#[cfg(target_os = "macos")]
const MODE_YCBCR: u32 = 1;
#[cfg(target_os = "macos")]
const MODE_RGB: u32 = 2;

#[cfg(target_os = "macos")]
const OUT_GRAY: u32 = 0;
#[cfg(target_os = "macos")]
const OUT_RGB: u32 = 1;
#[cfg(target_os = "macos")]
const OUT_RGBA: u32 = 2;

#[cfg(target_os = "macos")]
const FAST420_STATUS_OK: u32 = 0;
#[cfg(target_os = "macos")]
const FAST420_STATUS_TRUNCATED: u32 = 1;
#[cfg(target_os = "macos")]
const FAST420_STATUS_HUFFMAN: u32 = 2;

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct JpegPackParams {
    width: u32,
    height: u32,
    out_stride: u32,
    alpha: u32,
    mode: u32,
    out_format: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct JpegFast420Params {
    width: u32,
    height: u32,
    chroma_width: u32,
    chroma_height: u32,
    mcus_per_row: u32,
    mcu_rows: u32,
    entropy_len: u32,
    out_stride: u32,
    alpha: u32,
    out_format: u32,
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct MetalHuffmanTableHost {
    bits: [u8; 16],
    values_len: u16,
    reserved: u16,
    values: [u8; 256],
}

#[cfg(target_os = "macos")]
impl From<&PacketHuffmanTable> for MetalHuffmanTableHost {
    fn from(value: &PacketHuffmanTable) -> Self {
        Self {
            bits: value.bits,
            values_len: value.values_len,
            reserved: 0,
            values: value.values,
        }
    }
}

#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct JpegDecodeStatus {
    code: u32,
    detail: u32,
    position: u32,
    reserved: u32,
}

#[cfg(target_os = "macos")]
thread_local! {
    static METAL_RUNTIME: RefCell<Option<Result<MetalRuntime, String>>> = const { RefCell::new(None) };
    static VIEWPORT_PLANE_CACHE: RefCell<Option<CachedViewportPlanes>> = const { RefCell::new(None) };
}

#[cfg(target_os = "macos")]
struct MetalRuntime {
    device: Device,
    queue: CommandQueue,
    pack_pipeline: ComputePipelineState,
    pack_420_pipeline: ComputePipelineState,
    fast420_decode_pipeline: ComputePipelineState,
}

#[cfg(target_os = "macos")]
impl MetalRuntime {
    fn new() -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or_else(|| "Metal is unavailable on this host".to_string())?;
        let options = CompileOptions::new();
        let library = device.new_library_with_source(SHADER_SOURCE, &options)?;
        let pack_function = library.get_function("jpeg_pack", None)?;
        let pack_pipeline = device.new_compute_pipeline_state_with_function(&pack_function)?;
        let pack_420_function = library.get_function("jpeg_pack_420", None)?;
        let pack_420_pipeline =
            device.new_compute_pipeline_state_with_function(&pack_420_function)?;
        let fast420_decode_function = library.get_function("jpeg_decode_fast420", None)?;
        let fast420_decode_pipeline =
            device.new_compute_pipeline_state_with_function(&fast420_decode_function)?;
        let queue = device.new_command_queue();
        Ok(Self {
            device,
            queue,
            pack_pipeline,
            pack_420_pipeline,
            fast420_decode_pipeline,
        })
    }
}

#[cfg(target_os = "macos")]
fn with_runtime<R>(f: impl FnOnce(&MetalRuntime) -> Result<R, Error>) -> Result<R, Error> {
    METAL_RUNTIME.with(|runtime| {
        let mut runtime = runtime.borrow_mut();
        if runtime.is_none() {
            *runtime = Some(MetalRuntime::new());
        }
        match runtime.as_ref().expect("runtime initialized") {
            Ok(runtime) => f(runtime),
            Err(message) => Err(Error::MetalKernel {
                message: message.clone(),
            }),
        }
    })
}

#[cfg(target_os = "macos")]
#[derive(Clone, Copy, PartialEq, Eq)]
enum PlaneMode {
    Gray,
    YCbCr,
    Rgb,
}

#[cfg(target_os = "macos")]
struct PlaneStage {
    dims: (u32, u32),
    mode: PlaneMode,
    plane0: Buffer,
    plane1: Option<Buffer>,
    plane2: Option<Buffer>,
}

#[cfg(target_os = "macos")]
struct ViewportPlaneWriter<'a> {
    stage: &'a mut PlaneStage,
    dest: Rect,
}

#[cfg(target_os = "macos")]
struct CachedViewportPlanes {
    dims: (u32, u32),
    mode: PlaneMode,
    plane0: Buffer,
    plane1: Option<Buffer>,
    plane2: Option<Buffer>,
}

#[cfg(target_os = "macos")]
impl PlaneStage {
    fn new(device: &Device, color_space: JpegColorSpace, dims: (u32, u32)) -> Result<Self, Error> {
        let len = dims.0 as usize * dims.1 as usize;
        let plane0 = device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared);
        let (mode, plane1, plane2) = match color_space {
            JpegColorSpace::Grayscale => (PlaneMode::Gray, None, None),
            JpegColorSpace::YCbCr => (
                PlaneMode::YCbCr,
                Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
                Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
            ),
            JpegColorSpace::Rgb => (
                PlaneMode::Rgb,
                Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
                Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
            ),
            JpegColorSpace::Cmyk | JpegColorSpace::Ycck => {
                return Err(Error::MetalKernel {
                    message: "Metal compute path does not support CMYK/YCCK JPEG output"
                        .to_string(),
                })
            }
        };

        Ok(Self {
            dims,
            mode,
            plane0,
            plane1,
            plane2,
        })
    }

    fn finish_with_runtime(
        self,
        runtime: &MetalRuntime,
        fmt: PixelFormat,
    ) -> Result<Surface, Error> {
        match (self.mode, fmt) {
            (PlaneMode::Gray | PlaneMode::YCbCr, PixelFormat::Gray8) => {
                Ok(Surface::from_metal_buffer(self.plane0, self.dims, fmt))
            }
            (
                PlaneMode::Gray | PlaneMode::YCbCr | PlaneMode::Rgb,
                PixelFormat::Rgb8 | PixelFormat::Rgba8,
            )
            | (PlaneMode::Rgb, PixelFormat::Gray8) => Ok(self.dispatch_with_runtime(runtime, fmt)),
            _ => Err(Error::MetalKernel {
                message: format!("unsupported JPEG Metal pixel format {fmt:?}"),
            }),
        }
    }

    fn dispatch_with_runtime(self, runtime: &MetalRuntime, fmt: PixelFormat) -> Surface {
        let pitch_bytes = self.dims.0 as usize * fmt.bytes_per_pixel();
        let out_buffer = runtime.device.new_buffer(
            (pitch_bytes * self.dims.1 as usize) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params = JpegPackParams {
            width: self.dims.0,
            height: self.dims.1,
            out_stride: u32::try_from(pitch_bytes).expect("JPEG Metal output stride fits in u32"),
            alpha: u32::from(u8::MAX),
            mode: match self.mode {
                PlaneMode::Gray => MODE_GRAY,
                PlaneMode::YCbCr => MODE_YCBCR,
                PlaneMode::Rgb => MODE_RGB,
            },
            out_format: match fmt {
                PixelFormat::Gray8 => OUT_GRAY,
                PixelFormat::Rgb8 => OUT_RGB,
                PixelFormat::Rgba8 => OUT_RGBA,
                _ => unreachable!("validated by finish"),
            },
        };

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.pack_pipeline);
        encoder.set_buffer(0, Some(&self.plane0), 0);
        encoder.set_buffer(1, self.plane1.as_ref().map(std::convert::AsRef::as_ref), 0);
        encoder.set_buffer(2, self.plane2.as_ref().map(std::convert::AsRef::as_ref), 0);
        encoder.set_buffer(3, Some(&out_buffer), 0);
        encoder.set_bytes(
            4,
            size_of::<JpegPackParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(encoder, &runtime.pack_pipeline, self.dims);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Surface::from_metal_buffer(out_buffer, self.dims, fmt)
    }
}

#[cfg(target_os = "macos")]
impl ComponentRowWriter for PlaneStage {
    fn write_gray_row(
        &mut self,
        y: u32,
        gray_row: &[u8],
    ) -> Result<(), slidecodec_jpeg::JpegError> {
        let width = self.dims.0 as usize;
        write_row_u8(&self.plane0, y, width, gray_row);
        Ok(())
    }

    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        chroma_blue_row: &[u8],
        chroma_red_row: &[u8],
    ) -> Result<(), slidecodec_jpeg::JpegError> {
        let width = self.dims.0 as usize;
        write_row_u8(&self.plane0, y, width, y_row);
        write_row_u8(
            self.plane1.as_ref().expect("Cb plane"),
            y,
            width,
            chroma_blue_row,
        );
        write_row_u8(
            self.plane2.as_ref().expect("Cr plane"),
            y,
            width,
            chroma_red_row,
        );
        Ok(())
    }

    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), slidecodec_jpeg::JpegError> {
        let width = self.dims.0 as usize;
        write_row_u8(&self.plane0, y, width, r_row);
        write_row_u8(self.plane1.as_ref().expect("G plane"), y, width, g_row);
        write_row_u8(self.plane2.as_ref().expect("B plane"), y, width, b_row);
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn write_row_u8(buffer: &Buffer, y: u32, width: usize, src: &[u8]) {
    let row_start = y as usize * width;
    let row_end = row_start + width;
    let len = width * (y as usize + 1);
    let dst = unsafe {
        core::slice::from_raw_parts_mut(buffer.contents().cast::<u8>(), len.max(row_end))
    };
    dst[row_start..row_end].copy_from_slice(&src[..width]);
}

#[cfg(target_os = "macos")]
fn write_row_u8_at(buffer: &Buffer, y: u32, x: u32, full_width: usize, src: &[u8]) {
    let row_start = y as usize * full_width + x as usize;
    let row_end = row_start + src.len();
    let len = full_width * (y as usize + 1);
    let dst = unsafe {
        core::slice::from_raw_parts_mut(buffer.contents().cast::<u8>(), len.max(row_end))
    };
    dst[row_start..row_end].copy_from_slice(src);
}

#[cfg(target_os = "macos")]
fn plane_mode_for_color_space(color_space: JpegColorSpace) -> Result<PlaneMode, Error> {
    match color_space {
        JpegColorSpace::Grayscale => Ok(PlaneMode::Gray),
        JpegColorSpace::YCbCr => Ok(PlaneMode::YCbCr),
        JpegColorSpace::Rgb => Ok(PlaneMode::Rgb),
        JpegColorSpace::Cmyk | JpegColorSpace::Ycck => Err(Error::MetalKernel {
            message: "Metal compute path does not support CMYK/YCCK JPEG output".to_string(),
        }),
    }
}

#[cfg(target_os = "macos")]
fn clear_buffer(buffer: &Buffer, len: usize) {
    unsafe {
        core::ptr::write_bytes(buffer.contents().cast::<u8>(), 0, len);
    }
}

#[cfg(target_os = "macos")]
fn cached_viewport_stage(
    device: &Device,
    color_space: JpegColorSpace,
    dims: (u32, u32),
) -> Result<PlaneStage, Error> {
    let mode = plane_mode_for_color_space(color_space)?;
    VIEWPORT_PLANE_CACHE.with(|slot| {
        let mut slot = slot.borrow_mut();
        let len = dims.0 as usize * dims.1 as usize;
        let refresh = slot
            .as_ref()
            .is_none_or(|cached| cached.dims != dims || cached.mode != mode);
        if refresh {
            let plane0 = device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared);
            let (plane1, plane2) = match mode {
                PlaneMode::Gray => (None, None),
                PlaneMode::YCbCr | PlaneMode::Rgb => (
                    Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
                    Some(device.new_buffer(len as u64, MTLResourceOptions::StorageModeShared)),
                ),
            };
            *slot = Some(CachedViewportPlanes {
                dims,
                mode,
                plane0,
                plane1,
                plane2,
            });
        }

        let cached = slot.as_ref().expect("viewport plane cache");
        let stage = PlaneStage {
            dims,
            mode,
            plane0: cached.plane0.clone(),
            plane1: cached.plane1.clone(),
            plane2: cached.plane2.clone(),
        };
        clear_buffer(&stage.plane0, len);
        if let Some(plane1) = &stage.plane1 {
            clear_buffer(plane1, len);
        }
        if let Some(plane2) = &stage.plane2 {
            clear_buffer(plane2, len);
        }
        Ok(stage)
    })
}

#[cfg(target_os = "macos")]
impl ComponentRowWriter for ViewportPlaneWriter<'_> {
    fn write_gray_row(
        &mut self,
        y: u32,
        gray_row: &[u8],
    ) -> Result<(), slidecodec_jpeg::JpegError> {
        write_row_u8_at(
            &self.stage.plane0,
            self.dest.y + y,
            self.dest.x,
            self.stage.dims.0 as usize,
            gray_row,
        );
        Ok(())
    }

    fn write_ycbcr_row(
        &mut self,
        y: u32,
        y_row: &[u8],
        chroma_blue_row: &[u8],
        chroma_red_row: &[u8],
    ) -> Result<(), slidecodec_jpeg::JpegError> {
        let width = self.stage.dims.0 as usize;
        write_row_u8_at(
            &self.stage.plane0,
            self.dest.y + y,
            self.dest.x,
            width,
            y_row,
        );
        write_row_u8_at(
            self.stage.plane1.as_ref().expect("Cb plane"),
            self.dest.y + y,
            self.dest.x,
            width,
            chroma_blue_row,
        );
        write_row_u8_at(
            self.stage.plane2.as_ref().expect("Cr plane"),
            self.dest.y + y,
            self.dest.x,
            width,
            chroma_red_row,
        );
        Ok(())
    }

    fn write_rgb_row(
        &mut self,
        y: u32,
        r_row: &[u8],
        g_row: &[u8],
        b_row: &[u8],
    ) -> Result<(), slidecodec_jpeg::JpegError> {
        let width = self.stage.dims.0 as usize;
        write_row_u8_at(
            &self.stage.plane0,
            self.dest.y + y,
            self.dest.x,
            width,
            r_row,
        );
        write_row_u8_at(
            self.stage.plane1.as_ref().expect("G plane"),
            self.dest.y + y,
            self.dest.x,
            width,
            g_row,
        );
        write_row_u8_at(
            self.stage.plane2.as_ref().expect("B plane"),
            self.dest.y + y,
            self.dest.x,
            width,
            b_row,
        );
        Ok(())
    }
}

#[cfg(target_os = "macos")]
fn dispatch_2d_pipeline(
    encoder: &metal::ComputeCommandEncoderRef,
    pipeline: &ComputePipelineState,
    dims: (u32, u32),
) {
    let width = pipeline.thread_execution_width().max(1);
    let max_threads = pipeline.max_total_threads_per_threadgroup().max(width);
    let height = (max_threads / width).max(1);
    encoder.dispatch_threads(
        MTLSize {
            width: u64::from(dims.0),
            height: u64::from(dims.1),
            depth: 1,
        },
        MTLSize {
            width,
            height,
            depth: 1,
        },
    );
}

#[cfg(target_os = "macos")]
fn pixel_format_to_out_format(fmt: PixelFormat) -> Option<u32> {
    match fmt {
        PixelFormat::Gray8 => Some(OUT_GRAY),
        PixelFormat::Rgb8 => Some(OUT_RGB),
        PixelFormat::Rgba8 => Some(OUT_RGBA),
        _ => None,
    }
}

#[cfg(target_os = "macos")]
fn fast420_params(
    packet: &JpegMetalFast420PacketV1,
    fmt: PixelFormat,
) -> Result<JpegFast420Params, Error> {
    let out_format = pixel_format_to_out_format(fmt).ok_or_else(|| Error::MetalKernel {
        message: format!("unsupported JPEG Metal fast420 pixel format {fmt:?}"),
    })?;
    let out_stride = packet.dimensions.0 as usize * fmt.bytes_per_pixel();
    Ok(JpegFast420Params {
        width: packet.dimensions.0,
        height: packet.dimensions.1,
        chroma_width: packet.dimensions.0.div_ceil(2),
        chroma_height: packet.dimensions.1.div_ceil(2),
        mcus_per_row: packet.mcus_per_row,
        mcu_rows: packet.mcu_rows,
        entropy_len: u32::try_from(packet.entropy_bytes.len())
            .expect("JPEG Metal entropy payload fits in u32"),
        out_stride: u32::try_from(out_stride).expect("JPEG Metal output stride fits in u32"),
        alpha: u32::from(u8::MAX),
        out_format,
    })
}

#[cfg(target_os = "macos")]
fn decode_error_from_cpu(
    decoder: &CpuDecoder<'_>,
    fmt: PixelFormat,
    status: JpegDecodeStatus,
) -> Error {
    if let Err(err) = decoder.decode(fmt) {
        Error::Decode(err)
    } else {
        let reason = match status.code {
            FAST420_STATUS_TRUNCATED => "truncated entropy stream",
            FAST420_STATUS_HUFFMAN => "invalid Huffman stream",
            _ => "unexpected Metal fast420 failure",
        };
        Error::MetalKernel {
            message: format!("{reason} at entropy byte {}", status.position),
        }
    }
}

#[cfg(target_os = "macos")]
fn try_decode_fast420_to_surface(
    runtime: &MetalRuntime,
    decoder: &CpuDecoder<'_>,
    packet: Option<&JpegMetalFast420PacketV1>,
    fmt: PixelFormat,
) -> Result<Option<Surface>, Error> {
    let Some(packet) = packet else {
        return Ok(None);
    };
    let Some(_out_format) = pixel_format_to_out_format(fmt) else {
        return Ok(None);
    };

    let params = fast420_params(packet, fmt)?;
    let y_len = params.width as usize * params.height as usize;
    let chroma_len = params.chroma_width as usize * params.chroma_height as usize;
    let y_plane = runtime
        .device
        .new_buffer(y_len as u64, MTLResourceOptions::StorageModeShared);
    let chroma_buffers = [
        runtime
            .device
            .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared),
        runtime
            .device
            .new_buffer(chroma_len as u64, MTLResourceOptions::StorageModeShared),
    ];
    let status = JpegDecodeStatus::default();
    let status_buffer = runtime.device.new_buffer_with_data(
        (&raw const status).cast(),
        size_of::<JpegDecodeStatus>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let entropy_buffer = runtime.device.new_buffer_with_data(
        packet.entropy_bytes.as_ptr().cast(),
        packet.entropy_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let dc_tables = [
        MetalHuffmanTableHost::from(&packet.y_dc_table),
        MetalHuffmanTableHost::from(&packet.cb_dc_table),
        MetalHuffmanTableHost::from(&packet.cr_dc_table),
    ];
    let ac_tables = [
        MetalHuffmanTableHost::from(&packet.y_ac_table),
        MetalHuffmanTableHost::from(&packet.cb_ac_table),
        MetalHuffmanTableHost::from(&packet.cr_ac_table),
    ];

    let out_buffer = (fmt != PixelFormat::Gray8).then(|| {
        runtime.device.new_buffer(
            (params.out_stride as usize * params.height as usize) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    });

    let command_buffer = runtime.queue.new_command_buffer();
    let decoder_encoder = command_buffer.new_compute_command_encoder();
    decoder_encoder.set_compute_pipeline_state(&runtime.fast420_decode_pipeline);
    decoder_encoder.set_buffer(0, Some(&entropy_buffer), 0);
    decoder_encoder.set_buffer(1, Some(&y_plane), 0);
    decoder_encoder.set_buffer(2, Some(&chroma_buffers[0]), 0);
    decoder_encoder.set_buffer(3, Some(&chroma_buffers[1]), 0);
    decoder_encoder.set_bytes(
        4,
        size_of::<JpegFast420Params>() as u64,
        (&raw const params).cast(),
    );
    decoder_encoder.set_bytes(
        5,
        size_of::<[u16; 64]>() as u64,
        packet.y_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        6,
        size_of::<[u16; 64]>() as u64,
        packet.cb_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        7,
        size_of::<[u16; 64]>() as u64,
        packet.cr_quant.as_ptr().cast(),
    );
    decoder_encoder.set_bytes(
        8,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        9,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[0]).cast(),
    );
    decoder_encoder.set_bytes(
        10,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        11,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[1]).cast(),
    );
    decoder_encoder.set_bytes(
        12,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const dc_tables[2]).cast(),
    );
    decoder_encoder.set_bytes(
        13,
        size_of::<MetalHuffmanTableHost>() as u64,
        (&raw const ac_tables[2]).cast(),
    );
    decoder_encoder.set_buffer(14, Some(&status_buffer), 0);
    decoder_encoder.dispatch_threads(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    decoder_encoder.end_encoding();

    if let Some(out_buffer) = out_buffer.as_ref() {
        let pack_encoder = command_buffer.new_compute_command_encoder();
        pack_encoder.set_compute_pipeline_state(&runtime.pack_420_pipeline);
        pack_encoder.set_buffer(0, Some(&y_plane), 0);
        pack_encoder.set_buffer(1, Some(&chroma_buffers[0]), 0);
        pack_encoder.set_buffer(2, Some(&chroma_buffers[1]), 0);
        pack_encoder.set_buffer(3, Some(out_buffer), 0);
        pack_encoder.set_bytes(
            4,
            size_of::<JpegFast420Params>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(pack_encoder, &runtime.pack_420_pipeline, packet.dimensions);
        pack_encoder.end_encoding();
    }

    command_buffer.commit();
    command_buffer.wait_until_completed();

    let status = unsafe { *status_buffer.contents().cast::<JpegDecodeStatus>() };
    if status.code != FAST420_STATUS_OK {
        return Err(decode_error_from_cpu(decoder, fmt, status));
    }

    Ok(Some(match out_buffer {
        Some(out_buffer) => Surface::from_metal_buffer(out_buffer, packet.dimensions, fmt),
        None => Surface::from_metal_buffer(y_plane, packet.dimensions, fmt),
    }))
}

#[cfg(target_os = "macos")]
fn scaled_rect_covering(rect: Rect, scale: slidecodec_core::Downscale) -> Rect {
    let denom = scale.denominator();
    let x_end = rect.x + rect.w;
    let y_end = rect.y + rect.h;
    let x0 = rect.x / denom;
    let y0 = rect.y / denom;
    let x1 = x_end.div_ceil(denom);
    let y1 = y_end.div_ceil(denom);
    Rect {
        x: x0,
        y: y0,
        w: x1.saturating_sub(x0),
        h: y1.saturating_sub(y0),
    }
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut slidecodec_jpeg::ScratchPool,
    fmt: PixelFormat,
    fast420_packet: Option<&JpegMetalFast420PacketV1>,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        if let Some(surface) = try_decode_fast420_to_surface(runtime, decoder, fast420_packet, fmt)?
        {
            return Ok(surface);
        }
        let mut stage = PlaneStage::new(
            &runtime.device,
            decoder.info().color_space,
            decoder.info().dimensions,
        )?;
        decoder.decode_component_rows_with_scratch(pool, &mut stage)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut slidecodec_jpeg::ScratchPool,
    fmt: PixelFormat,
    roi: slidecodec_jpeg::Rect,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let dims = (roi.w, roi.h);
        let mut stage = PlaneStage::new(&runtime.device, decoder.info().color_space, dims)?;
        decoder.decode_region_component_rows_with_scratch(
            pool,
            &mut stage,
            roi,
            slidecodec_core::Downscale::None,
        )?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_scaled_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut slidecodec_jpeg::ScratchPool,
    fmt: PixelFormat,
    scale: slidecodec_core::Downscale,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let full = decoder.info().dimensions;
        let roi = slidecodec_jpeg::Rect {
            x: 0,
            y: 0,
            w: full.0,
            h: full.1,
        };
        let scaled = scaled_rect_covering(
            Rect {
                x: 0,
                y: 0,
                w: full.0,
                h: full.1,
            },
            scale,
        );
        let mut stage = PlaneStage::new(
            &runtime.device,
            decoder.info().color_space,
            (scaled.w, scaled.h),
        )?;
        decoder.decode_region_component_rows_with_scratch(pool, &mut stage, roi, scale)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn decode_region_scaled_to_surface(
    decoder: &CpuDecoder<'_>,
    pool: &mut slidecodec_jpeg::ScratchPool,
    fmt: PixelFormat,
    roi: slidecodec_jpeg::Rect,
    scale: slidecodec_core::Downscale,
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let scaled = scaled_rect_covering(
            Rect {
                x: roi.x,
                y: roi.y,
                w: roi.w,
                h: roi.h,
            },
            scale,
        );
        let mut stage = PlaneStage::new(
            &runtime.device,
            decoder.info().color_space,
            (scaled.w, scaled.h),
        )?;
        decoder.decode_region_component_rows_with_scratch(pool, &mut stage, roi, scale)?;
        stage.finish_with_runtime(runtime, fmt)
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn compose_rgb_viewport_from_regions(
    decoder: &CpuDecoder<'_>,
    pool: &mut slidecodec_jpeg::ScratchPool,
    scale: slidecodec_core::Downscale,
    viewport_dims: (u32, u32),
    tiles: &[ViewportTile],
) -> Result<Surface, Error> {
    with_runtime(|runtime| {
        let mut stage =
            cached_viewport_stage(&runtime.device, decoder.info().color_space, viewport_dims)?;
        for tile in tiles {
            let dims = scaled_rect_covering(tile.source_roi, scale);
            if (dims.w, dims.h) != (tile.dest.w, tile.dest.h) {
                return Err(Error::MetalKernel {
                    message: format!(
                        "viewport tile dims {:?} do not match destination rect {:?}",
                        (dims.w, dims.h),
                        tile.dest
                    ),
                });
            }
            let mut writer = ViewportPlaneWriter {
                stage: &mut stage,
                dest: tile.dest,
            };
            decoder.decode_region_component_rows_with_scratch(
                pool,
                &mut writer,
                slidecodec_jpeg::Rect {
                    x: tile.source_roi.x,
                    y: tile.source_roi.y,
                    w: tile.source_roi.w,
                    h: tile.source_roi.h,
                },
                scale,
            )?;
        }
        stage.finish_with_runtime(runtime, PixelFormat::Rgb8)
    })
}
