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

use crate::{ExternalDecodeRequest, MAX_EXTERNAL_OUTPUT_BYTES};

const MAX_COMPONENT_SAMPLES: usize = MAX_EXTERNAL_OUTPUT_BYTES / std::mem::size_of::<i32>();

pub fn is_available() -> bool {
    true
}

#[expect(
    unsafe_code,
    reason = "OpenJPEG exposes its version through a process-lifetime C string pointer"
)]
pub fn version() -> String {
    // SAFETY: `opj_version` returns either null or a process-lifetime NUL-terminated string.
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
    #[expect(
        unsafe_code,
        reason = "the guard owns and releases one OpenJPEG stream handle"
    )]
    fn drop(&mut self) {
        // SAFETY: the pointer came from a successful `opj_stream_create` and
        // this guard is its single owner.
        unsafe { opj_stream_destroy(self.0) };
    }
}

/// Owns an `OpenJPEG` codec; never null.
struct CodecGuard(*mut openjpeg_sys::opj_codec_t);

impl Drop for CodecGuard {
    #[expect(
        unsafe_code,
        reason = "the guard owns and releases one OpenJPEG codec handle"
    )]
    fn drop(&mut self) {
        // SAFETY: the pointer came from a successful `opj_create_decompress`
        // and this guard is its single owner.
        unsafe { opj_destroy_codec(self.0) };
    }
}

/// Owns the image pointer `opj_read_header` fills in; null until then.
struct ImageGuard(*mut opj_image_t);

impl Drop for ImageGuard {
    #[expect(
        unsafe_code,
        reason = "the guard conditionally owns OpenJPEG's decoded image handle"
    )]
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: non-null only after OpenJPEG allocated the image, and
            // this guard is its single owner.
            unsafe { opj_image_destroy(self.0) };
        }
    }
}

#[expect(
    unsafe_code,
    reason = "decode coordinates checked RAII-owned OpenJPEG stream, codec, and image handles"
)]
fn decode(bytes: &[u8], request: ExternalDecodeRequest) -> Result<Vec<u8>, String> {
    let codec_format = codec_format(bytes)?;
    let channels = usize::try_from(request.color.channels())
        .map_err(|_| "openjpeg: channel count exceeds platform usize".to_string())?;
    let decode_area = request.region.map(checked_decode_area).transpose()?;
    // Declaration order matters: drops run in reverse (image, codec, stream),
    // and the RAII guards free the FFI resources on every exit path,
    // including the early error returns below.
    let stream = StreamGuard(create_stream(bytes)?);
    let codec = CodecGuard(create_codec(codec_format)?);
    let mut image = ImageGuard(ptr::null_mut());
    // SAFETY: all three handles are either non-null RAII-owned values or an image out-pointer
    // owned by `ImageGuard`; OpenJPEG receives the source buffer through `StreamGuard` callbacks.
    unsafe {
        if opj_read_header(stream.0, codec.0, &raw mut image.0) == bool_false() {
            return Err("openjpeg: failed to read header".to_string());
        }
        if let Some(reduce) = request.reduce {
            if opj_set_decoded_resolution_factor(codec.0, reduce) == bool_false() {
                return Err("openjpeg: failed to set reduction factor".to_string());
            }
        }
        if let Some([x0, y0, x1, y1]) = decode_area {
            if opj_set_decode_area(codec.0, image.0, x0, y0, x1, y1) == bool_false() {
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

fn checked_decode_area(roi: j2k_core::Rect) -> Result<[i32; 4], String> {
    let x1 = roi
        .x
        .checked_add(roi.w)
        .ok_or_else(|| "openjpeg: decode area x coordinate overflow".to_string())?;
    let y1 = roi
        .y
        .checked_add(roi.h)
        .ok_or_else(|| "openjpeg: decode area y coordinate overflow".to_string())?;
    let coordinate = |value| {
        i32::try_from(value)
            .map_err(|_| "openjpeg: decode area exceeds OpenJPEG i32 coordinates".to_string())
    };
    Ok([
        coordinate(roi.x)?,
        coordinate(roi.y)?,
        coordinate(x1)?,
        coordinate(y1)?,
    ])
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

#[expect(
    unsafe_code,
    reason = "stream creation registers Rust-owned memory callbacks with OpenJPEG"
)]
fn create_stream(bytes: &[u8]) -> Result<*mut opj_stream_t, String> {
    let source_len = u64::try_from(bytes.len())
        .map_err(|_| "openjpeg: source length exceeds OpenJPEG u64 length".to_string())?;
    // SAFETY: the call has no borrowed pointer arguments and its nullable result is checked.
    let stream = unsafe { opj_stream_create(64 * 1024, OPJ_STREAM_READ.cast_signed()) };
    if stream.is_null() {
        return Err("openjpeg: failed to create stream".to_string());
    }
    let user = Box::into_raw(Box::new(MemoryStream::new(bytes)));
    // SAFETY: `user` is a unique Box allocation transferred to OpenJPEG and reclaimed exactly
    // once by `drop_memory_stream`; the callbacks use the same registered state and source length.
    unsafe {
        opj_stream_set_user_data(stream, user.cast(), Some(drop_memory_stream));
        opj_stream_set_user_data_length(stream, source_len);
        opj_stream_set_read_function(stream, Some(read_memory));
        opj_stream_set_skip_function(stream, Some(skip_memory));
        opj_stream_set_seek_function(stream, Some(seek_memory));
    }
    Ok(stream)
}

#[expect(
    unsafe_code,
    reason = "codec creation initializes and validates OpenJPEG's opaque decoder state"
)]
fn create_codec(codec_format: OPJ_CODEC_FORMAT) -> Result<*mut openjpeg_sys::opj_codec_t, String> {
    // SAFETY: OpenJPEG's parameter POD is valid when zeroed before its default initializer runs.
    let mut params = unsafe { std::mem::zeroed::<opj_dparameters_t>() };
    // SAFETY: `params` is live, writable, and correctly typed for the initializer.
    unsafe { opj_set_default_decoder_parameters(&raw mut params) };
    // SAFETY: `codec_format` was selected from a recognized JP2/J2K signature; null is checked.
    let codec = unsafe { opj_create_decompress(codec_format) };
    if codec.is_null() {
        return Err("openjpeg: failed to create codec".to_string());
    }
    // SAFETY: `codec` is non-null and uniquely owned here; `params` remains live for the call.
    let setup_ok = unsafe { opj_setup_decoder(codec, &raw mut params) };
    if setup_ok == bool_false() {
        // SAFETY: setup failure leaves this non-null codec owned by this function.
        unsafe { opj_destroy_codec(codec) };
        return Err("openjpeg: setup_decoder failed".to_string());
    }
    // SAFETY: `codec` remains non-null and configured; a single decoder thread is requested.
    let threading_ok = unsafe { opj_codec_set_threads(codec, 1) };
    if threading_ok == bool_false() {
        // SAFETY: thread-configuration failure leaves this non-null codec owned here.
        unsafe { opj_destroy_codec(codec) };
        return Err("openjpeg: codec_set_threads failed".to_string());
    }
    Ok(codec)
}

#[expect(
    unsafe_code,
    reason = "OpenJPEG returns image/component metadata through validated C pointers"
)]
fn pack_image(image: *mut opj_image_t, channels: usize) -> Result<Vec<u8>, String> {
    if image.is_null() {
        return Err("openjpeg: null image".to_string());
    }
    // SAFETY: `image` was checked non-null and remains owned by the caller's `ImageGuard`.
    let image_ref = unsafe { &*image };
    if image_ref.numcomps == 0 || image_ref.comps.is_null() {
        return Err("openjpeg: image has no components".to_string());
    }
    let num_components = usize::try_from(image_ref.numcomps)
        .map_err(|_| "openjpeg: component count exceeds platform usize".to_string())?;
    validate_component_count(num_components, channels)?;
    // SAFETY: `numcomps > 0` and `comps` was checked non-null before reading component zero.
    let comp0 = unsafe { &*image_ref.comps };
    let width = usize::try_from(comp0.w)
        .map_err(|_| "openjpeg: width exceeds platform usize".to_string())?;
    let height = usize::try_from(comp0.h)
        .map_err(|_| "openjpeg: height exceeds platform usize".to_string())?;
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

#[expect(
    unsafe_code,
    reason = "component lookup validates OpenJPEG's component array and sample buffer bounds"
)]
fn read_component(
    image: &opj_image,
    index: usize,
    row: usize,
    col: usize,
    full_width: usize,
    full_height: usize,
) -> Result<u8, String> {
    let num_components = usize::try_from(image.numcomps)
        .map_err(|_| "openjpeg: component count exceeds platform usize".to_string())?;
    if index >= num_components {
        return Err(format!("openjpeg: component {index} missing"));
    }
    // SAFETY: `index < numcomps`; OpenJPEG owns a contiguous component array for this image.
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
    let stride = usize::try_from(comp.w)
        .map_err(|_| "openjpeg: component width exceeds platform usize".to_string())?;
    let height = usize::try_from(comp.h)
        .map_err(|_| "openjpeg: component height exceeds platform usize".to_string())?;
    if stride == 0 || height == 0 {
        return Err("openjpeg: component has zero-sized output".to_string());
    }
    let precision = component_precision(comp.prec)?;
    let sample_len = checked_component_sample_len(stride, height)?;
    // SAFETY: `comp.data` is non-null and `sample_len` is the checked component allocation shape.
    let data = unsafe { slice::from_raw_parts(comp.data, sample_len) };
    let comp_col = col.saturating_mul(stride) / full_width.max(1);
    let comp_row = row.saturating_mul(height) / full_height.max(1);
    let value = data[comp_row.min(height - 1) * stride + comp_col.min(stride - 1)];
    Ok(scale_to_u8(value, precision, comp.sgnd != 0))
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

#[derive(Clone, Copy)]
struct ComponentPrecision(u32);

fn component_precision(precision: u32) -> Result<ComponentPrecision, String> {
    if !(1..=31).contains(&precision) {
        return Err(format!(
            "openjpeg: unsupported component precision {precision}, expected 1..=31"
        ));
    }
    Ok(ComponentPrecision(precision))
}

fn scale_to_u8(value: i32, precision: ComponentPrecision, signed: bool) -> u8 {
    let precision = precision.0;
    let adjusted = if signed {
        value.saturating_add(1_i32 << (precision - 1))
    } else {
        value
    };
    if precision <= 8 {
        u8::try_from(adjusted.clamp(0, 255)).expect("sample was clamped to the u8 range")
    } else {
        let max = i64::from((1_u32 << precision.min(31)) - 1);
        let scaled = (i64::from(adjusted.max(0)) * 255 + max / 2) / max.max(1);
        u8::try_from(scaled.clamp(0, 255)).expect("sample was clamped to the u8 range")
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

#[expect(
    unsafe_code,
    reason = "OpenJPEG invokes this callback with the registered MemoryStream and output buffer"
)]
unsafe extern "C" fn read_memory(
    buffer: *mut c_void,
    bytes: OPJ_SIZE_T,
    user_data: *mut c_void,
) -> OPJ_SIZE_T {
    if user_data.is_null() {
        return usize::MAX;
    }
    // SAFETY: OpenJPEG passes back the `MemoryStream` pointer registered in
    // `create_stream`; null was rejected and the stream owns it until cleanup.
    let state = unsafe { &mut *user_data.cast::<MemoryStream>() };
    let remaining = state.len.saturating_sub(state.offset);
    if remaining == 0 {
        return usize::MAX;
    }
    let count = remaining.min(bytes);
    if count == 0 {
        return 0;
    }
    if buffer.is_null() {
        return usize::MAX;
    }
    // SAFETY: `count` is bounded by the remaining source length, and OpenJPEG
    // provides a non-null writable buffer of the requested size for the callback.
    unsafe {
        ptr::copy_nonoverlapping(state.ptr.add(state.offset), buffer.cast::<u8>(), count);
    }
    state.offset += count;
    count
}

#[expect(
    unsafe_code,
    reason = "OpenJPEG invokes this callback with the registered MemoryStream state"
)]
unsafe extern "C" fn skip_memory(bytes: OPJ_OFF_T, user_data: *mut c_void) -> OPJ_OFF_T {
    if user_data.is_null() {
        return -1;
    }
    // SAFETY: OpenJPEG passes back the `MemoryStream` pointer registered in
    // `create_stream`; null was rejected and the stream owns it until cleanup.
    let state = unsafe { &mut *user_data.cast::<MemoryStream>() };
    if bytes < 0 {
        return -1;
    }
    let Ok(delta) = usize::try_from(bytes) else {
        return -1;
    };
    let Some(next) = state.offset.checked_add(delta) else {
        return -1;
    };
    if next > state.len {
        return -1;
    }
    state.offset = next;
    bytes
}

#[expect(
    unsafe_code,
    reason = "OpenJPEG invokes this callback with the registered MemoryStream state"
)]
unsafe extern "C" fn seek_memory(bytes: OPJ_OFF_T, user_data: *mut c_void) -> OPJ_BOOL {
    if user_data.is_null() {
        return bool_false();
    }
    // SAFETY: OpenJPEG passes back the `MemoryStream` pointer registered in
    // `create_stream`; null was rejected and the stream owns it until cleanup.
    let state = unsafe { &mut *user_data.cast::<MemoryStream>() };
    let Ok(offset) = usize::try_from(bytes) else {
        return bool_false();
    };
    if offset > state.len {
        return bool_false();
    }
    state.offset = offset;
    bool_true()
}

#[expect(
    unsafe_code,
    reason = "OpenJPEG transfers the registered MemoryStream allocation back to this callback"
)]
unsafe extern "C" fn drop_memory_stream(user_data: *mut c_void) {
    if user_data.is_null() {
        return;
    }
    // SAFETY: `create_stream` allocated this pointer with `Box::into_raw` and
    // registered this callback as the single owner responsible for freeing it;
    // the null case was rejected above.
    unsafe { drop(Box::from_raw(user_data.cast::<MemoryStream>())) };
}

const fn bool_false() -> OPJ_BOOL {
    OPJ_FALSE.cast_signed()
}

const fn bool_true() -> OPJ_BOOL {
    OPJ_TRUE.cast_signed()
}

#[cfg(test)]
mod tests {
    use j2k_core::Rect;

    use super::{
        bool_false, bool_true, checked_component_sample_len, checked_decode_area,
        checked_output_len, component_precision, drop_memory_stream, read_memory, seek_memory,
        skip_memory, validate_component_count, MemoryStream, MAX_EXTERNAL_OUTPUT_BYTES,
    };
    use openjpeg_sys::{opj_image, opj_image_comp, COLOR_SPACE};

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

    #[test]
    fn component_precision_rejects_unsafe_shift_ranges() {
        assert!(component_precision(0).is_err());
        assert!(component_precision(32).is_err());
        assert!(component_precision(u32::MAX).is_err());
        assert!(component_precision(1).is_ok());
        assert!(component_precision(31).is_ok());
    }

    #[test]
    fn read_component_propagates_invalid_precision_errors() {
        fn read_at_precision(precision: u32) -> Result<u8, String> {
            let mut sample = 0_i32;
            let mut component = opj_image_comp {
                dx: 1,
                dy: 1,
                w: 1,
                h: 1,
                x0: 0,
                y0: 0,
                prec: precision,
                bpp: precision,
                sgnd: 0,
                resno_decoded: 0,
                factor: 0,
                data: &raw mut sample,
                alpha: 0,
            };
            let image = opj_image {
                x0: 0,
                y0: 0,
                x1: 1,
                y1: 1,
                numcomps: 1,
                color_space: COLOR_SPACE::OPJ_CLRSPC_UNSPECIFIED,
                comps: &raw mut component,
                icc_profile_buf: std::ptr::null_mut(),
                icc_profile_len: 0,
            };
            super::read_component(&image, 0, 0, 0, 1, 1)
        }

        assert!(read_at_precision(0).is_err());
        assert!(read_at_precision(32).is_err());
        assert!(read_at_precision(31).is_ok());
    }

    #[test]
    #[expect(
        unsafe_code,
        reason = "direct callback tests exercise null arguments at the OpenJPEG C ABI boundary"
    )]
    fn stream_callbacks_reject_null_arguments_without_dereferencing() {
        let null = std::ptr::null_mut();
        // SAFETY: each callback is invoked specifically with null inputs to verify
        // its defensive sentinel path, which must return before dereferencing.
        unsafe {
            assert_eq!(read_memory(null, 1, null), usize::MAX);
            assert_eq!(skip_memory(1, null), -1);
            assert_eq!(seek_memory(1, null), bool_false());
            drop_memory_stream(null);
        }

        let bytes = [1_u8];
        let mut stream = MemoryStream::new(&bytes);
        let user_data = (&raw mut stream).cast();
        // SAFETY: `user_data` points to a live stack `MemoryStream`; the null output
        // buffer must be rejected before a non-empty copy and does not take ownership.
        unsafe {
            assert_eq!(read_memory(null, 0, user_data), 0);
            assert_eq!(read_memory(null, 1, user_data), usize::MAX);
            assert_eq!(seek_memory(1, user_data), bool_true());
        }
        assert_eq!(stream.offset, 1);
    }

    #[test]
    fn decode_area_rejects_coordinate_addition_and_i32_overflow() {
        assert!(checked_decode_area(Rect {
            x: u32::MAX,
            y: 0,
            w: 1,
            h: 1,
        })
        .is_err());
        assert!(checked_decode_area(Rect {
            x: 2_147_483_647,
            y: 0,
            w: 1,
            h: 1,
        })
        .is_err());
        assert_eq!(
            checked_decode_area(Rect {
                x: 1,
                y: 2,
                w: 3,
                h: 4,
            })
            .expect("bounded decode area"),
            [1, 2, 4, 6]
        );
    }
}
