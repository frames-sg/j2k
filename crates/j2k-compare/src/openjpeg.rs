// SPDX-License-Identifier: MIT OR Apache-2.0

use openjpeg_sys::{
    opj_codec_set_threads, opj_create_decompress, opj_decode, opj_destroy_codec, opj_dparameters_t,
    opj_end_decompress, opj_image, opj_image_destroy, opj_image_t, opj_read_header,
    opj_set_decode_area, opj_set_decoded_resolution_factor, opj_set_default_decoder_parameters,
    opj_setup_decoder, opj_stream_create, opj_stream_destroy, opj_stream_set_read_function,
    opj_stream_set_seek_function, opj_stream_set_skip_function, opj_stream_set_user_data,
    opj_stream_set_user_data_length, opj_stream_t, opj_version, OPJ_BOOL, OPJ_CODEC_FORMAT,
    OPJ_FALSE, OPJ_OFF_T, OPJ_SIZE_T, OPJ_STREAM_READ, OPJ_TRUE,
};
use std::{
    ffi::{c_void, CStr},
    ptr, slice,
};

use crate::ExternalDecodeRequest;

const MAX_EXTERNAL_OUTPUT_BYTES: usize = 512 * 1024 * 1024;
const MAX_COMPONENT_SAMPLES: usize = MAX_EXTERNAL_OUTPUT_BYTES / std::mem::size_of::<i32>();

pub fn is_available() -> bool {
    true
}

pub fn version() -> String {
    // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
    unsafe {
        let ptr = opj_version();
        if ptr.is_null() {
            "unknown".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

pub fn library_path() -> &'static str {
    "openjpeg-sys vendored openjp2"
}

crate::external_decode_wrappers!(decode);

/// Owns an `OpenJPEG` stream; never null.
struct StreamGuard(*mut opj_stream_t);

impl Drop for StreamGuard {
    fn drop(&mut self) {
        // SAFETY: the pointer came from a successful `opj_stream_create` and
        // this guard is its single owner.
        // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
        unsafe { opj_stream_destroy(self.0) };
    }
}

/// Owns an `OpenJPEG` codec; never null.
struct CodecGuard(*mut openjpeg_sys::opj_codec_t);

impl Drop for CodecGuard {
    fn drop(&mut self) {
        // SAFETY: the pointer came from a successful `opj_create_decompress`
        // and this guard is its single owner.
        // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
        unsafe { opj_destroy_codec(self.0) };
    }
}

/// Owns the image pointer `opj_read_header` fills in; null until then.
struct ImageGuard(*mut opj_image_t);

impl Drop for ImageGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: non-null only after OpenJPEG allocated the image, and
            // this guard is its single owner.
            // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
            unsafe { opj_image_destroy(self.0) };
        }
    }
}

fn decode(bytes: &[u8], request: ExternalDecodeRequest) -> Result<Vec<u8>, String> {
    let codec_format = codec_format(bytes)?;
    let channels = request.color.channels() as usize;
    // Declaration order matters: drops run in reverse (image, codec, stream),
    // and the RAII guards free the FFI resources on every exit path,
    // including the early error returns below.
    let stream = StreamGuard(create_stream(bytes)?);
    let codec = CodecGuard(create_codec(codec_format)?);
    let mut image = ImageGuard(ptr::null_mut());
    // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
    unsafe {
        if opj_read_header(stream.0, codec.0, &raw mut image.0) == bool_false() {
            return Err("openjpeg: failed to read header".to_string());
        }
        if let Some(reduce) = request.reduce {
            if opj_set_decoded_resolution_factor(codec.0, reduce) == bool_false() {
                return Err("openjpeg: failed to set reduction factor".to_string());
            }
        }
        if let Some(roi) = request.region {
            if opj_set_decode_area(
                codec.0,
                image.0,
                roi.x as i32,
                roi.y as i32,
                (roi.x + roi.w) as i32,
                (roi.y + roi.h) as i32,
            ) == bool_false()
            {
                return Err("openjpeg: failed to set decode area".to_string());
            }
        }
        if opj_decode(codec.0, stream.0, image.0) == bool_false() {
            return Err("openjpeg: decode failed".to_string());
        }
        let packed = pack_image(image.0, channels)?;
        if opj_end_decompress(codec.0, stream.0) == bool_false() {
            return Err("openjpeg: end_decompress failed".to_string());
        }
        Ok(packed)
    }
}

fn codec_format(bytes: &[u8]) -> Result<OPJ_CODEC_FORMAT, String> {
    if bytes.starts_with(&[0, 0, 0, 12, b'j', b'P', b' ', b' ']) {
        Ok(OPJ_CODEC_FORMAT::OPJ_CODEC_JP2)
    } else if bytes.starts_with(&[0xFF, 0x4F]) {
        Ok(OPJ_CODEC_FORMAT::OPJ_CODEC_J2K)
    } else {
        Err("openjpeg: unsupported container, expected JP2 or raw J2K".to_string())
    }
}

fn create_stream(bytes: &[u8]) -> Result<*mut opj_stream_t, String> {
    // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
    let stream = unsafe { opj_stream_create(64 * 1024, OPJ_STREAM_READ as OPJ_BOOL) };
    if stream.is_null() {
        return Err("openjpeg: failed to create stream".to_string());
    }
    let user = Box::into_raw(Box::new(MemoryStream::new(bytes)));
    // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
    unsafe {
        opj_stream_set_user_data(stream, user.cast(), Some(drop_memory_stream));
        opj_stream_set_user_data_length(stream, bytes.len() as u64);
        opj_stream_set_read_function(stream, Some(read_memory));
        opj_stream_set_skip_function(stream, Some(skip_memory));
        opj_stream_set_seek_function(stream, Some(seek_memory));
    }
    Ok(stream)
}

fn create_codec(codec_format: OPJ_CODEC_FORMAT) -> Result<*mut openjpeg_sys::opj_codec_t, String> {
    // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
    let mut params = unsafe { std::mem::zeroed::<opj_dparameters_t>() };
    // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
    unsafe { opj_set_default_decoder_parameters(&raw mut params) };
    // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
    let codec = unsafe { opj_create_decompress(codec_format) };
    if codec.is_null() {
        return Err("openjpeg: failed to create codec".to_string());
    }
    // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
    let setup_ok = unsafe { opj_setup_decoder(codec, &raw mut params) };
    if setup_ok == bool_false() {
        // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
        unsafe { opj_destroy_codec(codec) };
        return Err("openjpeg: setup_decoder failed".to_string());
    }
    // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
    let threading_ok = unsafe { opj_codec_set_threads(codec, 1) };
    if threading_ok == bool_false() {
        // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
        unsafe { opj_destroy_codec(codec) };
        return Err("openjpeg: codec_set_threads failed".to_string());
    }
    Ok(codec)
}

fn pack_image(image: *mut opj_image_t, channels: usize) -> Result<Vec<u8>, String> {
    if image.is_null() {
        return Err("openjpeg: null image".to_string());
    }
    // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
    let image_ref = unsafe { &*image };
    if image_ref.numcomps == 0 || image_ref.comps.is_null() {
        return Err("openjpeg: image has no components".to_string());
    }
    validate_component_count(image_ref.numcomps as usize, channels)?;
    // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
    let comp0 = unsafe { &*image_ref.comps };
    let width = comp0.w as usize;
    let height = comp0.h as usize;
    let output_len = checked_output_len(width, height, channels)?;
    let mut out = vec![0_u8; output_len];
    for row in 0..height {
        for col in 0..width {
            let dst = (row * width + col) * channels;
            let sample0 = read_component(image_ref, 0, row, col, width, height)?;
            if channels == 1 {
                out[dst] = sample0;
                continue;
            }
            if image_ref.numcomps == 1 {
                out[dst] = sample0;
                out[dst + 1] = sample0;
                out[dst + 2] = sample0;
                continue;
            }
            out[dst] = sample0;
            out[dst + 1] = read_component(image_ref, 1, row, col, width, height)?;
            out[dst + 2] = read_component(image_ref, 2, row, col, width, height)?;
        }
    }
    Ok(out)
}

fn read_component(
    image: &opj_image,
    index: usize,
    row: usize,
    col: usize,
    full_width: usize,
    full_height: usize,
) -> Result<u8, String> {
    if index >= image.numcomps as usize {
        return Err(format!("openjpeg: component {index} missing"));
    }
    // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
    let comp = unsafe {
        image
            .comps
            .add(index)
            .as_ref()
            .ok_or_else(|| "openjpeg: component missing".to_string())?
    };
    if comp.data.is_null() {
        return Err("openjpeg: component data missing".to_string());
    }
    let stride = comp.w as usize;
    let height = comp.h as usize;
    if stride == 0 || height == 0 {
        return Err("openjpeg: component has zero-sized output".to_string());
    }
    let sample_len = checked_component_sample_len(stride, height)?;
    // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
    let data = unsafe { slice::from_raw_parts(comp.data, sample_len) };
    let comp_col = col.saturating_mul(stride) / full_width.max(1);
    let comp_row = row.saturating_mul(height) / full_height.max(1);
    let value = data[comp_row.min(height - 1) * stride + comp_col.min(stride - 1)];
    Ok(scale_to_u8(value, comp.prec, comp.sgnd != 0))
}

fn validate_component_count(numcomps: usize, channels: usize) -> Result<(), String> {
    match channels {
        1 => Ok(()),
        3 if numcomps == 1 || numcomps >= 3 => Ok(()),
        3 => Err(format!(
            "openjpeg: RGB output requires 1 or at least 3 components, got {numcomps}"
        )),
        _ => Err(format!(
            "openjpeg: unsupported channel count {channels}, expected 1 or 3"
        )),
    }
}

fn checked_output_len(width: usize, height: usize, channels: usize) -> Result<usize, String> {
    if !matches!(channels, 1 | 3) {
        return Err(format!(
            "openjpeg: unsupported channel count {channels}, expected 1 or 3"
        ));
    }
    if width == 0 || height == 0 {
        return Err("openjpeg: image has zero-sized output".to_string());
    }
    let pixels = width
        .checked_mul(height)
        .ok_or_else(|| "openjpeg: output pixel count overflow".to_string())?;
    let len = pixels
        .checked_mul(channels)
        .ok_or_else(|| "openjpeg: output byte count overflow".to_string())?;
    if len > MAX_EXTERNAL_OUTPUT_BYTES {
        return Err(format!(
            "openjpeg: output exceeds {MAX_EXTERNAL_OUTPUT_BYTES} byte cap"
        ));
    }
    Ok(len)
}

fn checked_component_sample_len(stride: usize, height: usize) -> Result<usize, String> {
    let len = stride
        .checked_mul(height)
        .ok_or_else(|| "openjpeg: component sample count overflow".to_string())?;
    if len > MAX_COMPONENT_SAMPLES {
        return Err(format!(
            "openjpeg: component sample count exceeds {MAX_COMPONENT_SAMPLES} sample cap"
        ));
    }
    Ok(len)
}

fn scale_to_u8(value: i32, precision: u32, signed: bool) -> u8 {
    let adjusted = if signed {
        value.saturating_add(1_i32 << precision.saturating_sub(1))
    } else {
        value
    };
    if precision <= 8 {
        adjusted.clamp(0, 255) as u8
    } else {
        let max = i64::from((1_u32 << precision.min(31)) - 1);
        let scaled = (i64::from(adjusted.max(0)) * 255 + max / 2) / max.max(1);
        scaled.clamp(0, 255) as u8
    }
}

struct MemoryStream {
    ptr: *const u8,
    len: usize,
    offset: usize,
}

impl MemoryStream {
    fn new(bytes: &[u8]) -> Self {
        Self {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
            offset: 0,
        }
    }
}

unsafe extern "C" fn read_memory(
    buffer: *mut c_void,
    bytes: OPJ_SIZE_T,
    user_data: *mut c_void,
) -> OPJ_SIZE_T {
    // SAFETY: OpenJPEG passes back the `MemoryStream` pointer registered in
    // `create_stream`; the stream owns it until `drop_memory_stream` runs.
    // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
    let state = unsafe { &mut *user_data.cast::<MemoryStream>() };
    let remaining = state.len.saturating_sub(state.offset);
    if remaining == 0 {
        return usize::MAX;
    }
    let count = remaining.min(bytes);
    // SAFETY: `count` is bounded by the remaining source length, and OpenJPEG
    // provides a writable buffer of the requested size for the read callback.
    // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
    unsafe {
        ptr::copy_nonoverlapping(state.ptr.add(state.offset), buffer.cast::<u8>(), count);
    }
    state.offset += count;
    count
}

unsafe extern "C" fn skip_memory(bytes: OPJ_OFF_T, user_data: *mut c_void) -> OPJ_OFF_T {
    // SAFETY: OpenJPEG passes back the `MemoryStream` pointer registered in
    // `create_stream`; the stream owns it until `drop_memory_stream` runs.
    // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
    let state = unsafe { &mut *user_data.cast::<MemoryStream>() };
    if bytes < 0 {
        return -1;
    }
    let next = state.offset.saturating_add(bytes as usize);
    if next > state.len {
        return -1;
    }
    state.offset = next;
    bytes
}

unsafe extern "C" fn seek_memory(bytes: OPJ_OFF_T, user_data: *mut c_void) -> OPJ_BOOL {
    // SAFETY: OpenJPEG passes back the `MemoryStream` pointer registered in
    // `create_stream`; the stream owns it until `drop_memory_stream` runs.
    // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
    let state = unsafe { &mut *user_data.cast::<MemoryStream>() };
    if bytes < 0 || bytes as usize > state.len {
        return bool_false();
    }
    state.offset = bytes as usize;
    bool_true()
}

unsafe extern "C" fn drop_memory_stream(user_data: *mut c_void) {
    // SAFETY: `create_stream` allocated this pointer with `Box::into_raw` and
    // registered this callback as the single owner responsible for freeing it.
    // SAFETY: OpenJPEG FFI calls use checked handles and validated component buffers.
    unsafe { drop(Box::from_raw(user_data.cast::<MemoryStream>())) };
}

const fn bool_false() -> OPJ_BOOL {
    OPJ_FALSE as OPJ_BOOL
}

const fn bool_true() -> OPJ_BOOL {
    OPJ_TRUE as OPJ_BOOL
}

#[cfg(test)]
mod tests {
    use super::{
        checked_component_sample_len, checked_output_len, validate_component_count,
        MAX_EXTERNAL_OUTPUT_BYTES,
    };

    #[test]
    fn output_len_rejects_invalid_channels_and_zero_dimensions() {
        assert!(checked_output_len(1, 1, 2).is_err());
        assert!(checked_output_len(0, 1, 1).is_err());
        assert!(checked_output_len(1, 0, 3).is_err());
    }

    #[test]
    fn output_len_rejects_overflow_and_cap_excess() {
        assert!(checked_output_len(usize::MAX, 2, 3).is_err());
        assert!(checked_output_len(MAX_EXTERNAL_OUTPUT_BYTES + 1, 1, 1).is_err());
    }

    #[test]
    fn component_count_rejects_two_component_rgb_output() {
        assert!(validate_component_count(2, 3).is_err());
        assert!(validate_component_count(1, 3).is_ok());
        assert!(validate_component_count(3, 3).is_ok());
    }

    #[test]
    fn component_sample_len_rejects_overflow_and_cap_excess() {
        assert!(checked_component_sample_len(usize::MAX, 2).is_err());
        assert!(checked_component_sample_len(
            MAX_EXTERNAL_OUTPUT_BYTES / std::mem::size_of::<i32>() + 1,
            1
        )
        .is_err());
    }
}
