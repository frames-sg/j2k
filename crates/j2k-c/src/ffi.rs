// SPDX-License-Identifier: MIT OR Apache-2.0

//! Raw pointer validation, ownership transfer, and unwind containment.

use crate::{
    abi::{
        J2kCColorInfo, J2kCComponentInfo, J2kCDecodeOptions, J2kCEncodeOptions, J2kCEncodeParams,
        J2kCEncodeReport, J2kCError, J2kCHeader, J2kCImageInfo, J2kCOpenOptions,
        J2kCOperationOptions, J2kCOutputInfo, J2kCRegionPlan, J2K_ABI_MISMATCH, J2K_ABI_VERSION,
        J2K_BUFFER_TOO_SMALL, J2K_INTERNAL_ERROR, J2K_INVALID_ARGUMENT, J2K_INVALID_INPUT,
        J2K_PANIC,
    },
    decoder::Decoder,
    encode::{encode, EncodeParams},
    error::{Error, Result},
    operations::OperationRequest,
};
use std::{
    ffi::{c_char, c_void},
    mem::{align_of, size_of},
    panic::{catch_unwind, AssertUnwindSafe},
    ptr, slice,
};

/// Opaque decoder. Never instantiate, copy, dereference, or free it directly.
#[repr(C)]
pub struct J2kCDecoder {
    _private: [u8; 0],
}

/// Opaque shared cancellation token. Cancel may run concurrently; free may not.
#[repr(C)]
pub struct J2kCCancellation {
    _private: [u8; 0],
}

/// Create an uncancelled token. On failure, the output slot is null.
///
/// # Safety
/// `out` is a valid writable slot that owns no live token. Optional `error`
/// storage must be valid, aligned, writable, and disjoint from the slot.
#[no_mangle]
pub unsafe extern "C" fn j2k_cancellation_create(
    out: *mut *mut J2kCCancellation,
    error: *mut J2kCError,
) -> u32 {
    if check_pointer(out).is_ok() {
        // SAFETY: Caller supplies a writable pointer slot; alignment was checked.
        unsafe { out.write(ptr::null_mut()) };
    }
    // SAFETY: The caller supplies valid disjoint storage, checked before use.
    unsafe {
        boundary(error, || {
            check_pointer(out)?;
            out.write(Box::into_raw(Box::new(j2k::J2kCancellationToken::new())).cast());
            Ok(())
        })
    }
}

/// Permanently cancel a token. This may run concurrently with decoding or cancellation.
///
/// # Safety
/// The token must be live for this call, and may not be freed concurrently.
/// Optional writable error storage must be disjoint and exclusively used.
#[no_mangle]
pub unsafe extern "C" fn j2k_cancellation_cancel(
    cancellation: *const J2kCCancellation,
    error: *mut J2kCError,
) -> u32 {
    // SAFETY: The caller keeps the token alive; a shared borrow uses atomic cancellation.
    unsafe {
        boundary(error, || {
            let token = cancellation.cast::<j2k::J2kCancellationToken>();
            check_pointer(token)?;
            (&*token).cancel();
            Ok(())
        })
    }
}

/// Free a token and null the caller's slot. A null handle is a no-op.
///
/// # Safety
/// The slot must be valid and contain null or a token created by this library.
/// No call may use or cancel the token concurrently with free. Optional error
/// storage must be valid and disjoint from the slot and token.
#[no_mangle]
pub unsafe extern "C" fn j2k_cancellation_free(
    cancellation: *mut *mut J2kCCancellation,
    error: *mut J2kCError,
) -> u32 {
    // SAFETY: The caller transfers exclusive token ownership through a valid slot.
    unsafe {
        boundary(error, || {
            check_pointer(cancellation)?;
            let token = cancellation.read().cast::<j2k::J2kCancellationToken>();
            if !token.is_null() {
                check_pointer(token)?;
                cancellation.write(ptr::null_mut());
                drop(Box::from_raw(token));
            }
            Ok(())
        })
    }
}

unsafe fn operation_request(
    options: *const J2kCOperationOptions,
    dimensions: (u32, u32),
) -> Result<OperationRequest> {
    // SAFETY: Caller guarantees a readable ABI prefix and any referenced live token.
    unsafe {
        let options = read_options(options)?;
        let cancellation = if options.cancellation.is_null() {
            None
        } else {
            let token = options.cancellation.cast::<j2k::J2kCancellationToken>();
            check_pointer(token)?;
            // Own a token clone before any long-running codec work begins.
            Some((&*token).clone())
        };
        OperationRequest::new(options, dimensions, cancellation)
    }
}

/// Prepare immutable input facts for later native exact operations.
/// This explicit input preparation is outside per-operation limits/cancellation;
/// without it, the first native operation prepares the source under its own controls.
///
/// # Safety
/// The decoder is live and exclusively used. Optional writable error storage
/// must be valid, aligned, and disjoint from the decoder.
#[no_mangle]
pub unsafe extern "C" fn j2k_decoder_prepare_operations(
    decoder: *mut J2kCDecoder,
    error: *mut J2kCError,
) -> u32 {
    // SAFETY: The caller owns a live decoder exclusively; the boundary validates it.
    unsafe { boundary(error, || with_decoder(decoder, Decoder::prepare_operations)) }
}

/// Plan an exact native or sRGB region without allocating pixel output.
///
/// # Safety
/// The decoder is live and exclusively used. Options (or null) and their token
/// are readable and live for the call. `out` is a writable initialized ABI
/// structure. Optional error storage and every argument are valid and disjoint.
#[no_mangle]
pub unsafe extern "C" fn j2k_decoder_plan_region(
    decoder: *mut J2kCDecoder,
    options: *const J2kCOperationOptions,
    out: *mut J2kCRegionPlan,
    error: *mut J2kCError,
) -> u32 {
    // SAFETY: Valid disjoint storage is the caller's contract; check ABI prefixes first.
    unsafe {
        boundary(error, || {
            validate_struct(out)?;
            with_decoder(decoder, |state| {
                let request = operation_request(options, state.image_size())?;
                out.write(state.plan_operation(&request)?);
                Ok(())
            })
        })
    }
}

/// Decode an exact region into a caller-owned strided buffer. No bytes are
/// written on failure. Optional `out` receives actual output metadata/warnings.
///
/// # Safety
/// Same handle/options/token requirements as [`j2k_decoder_plan_region`].
/// `output` covers `output_len` writable bytes. Optional `out` is an initialized
/// writable ABI structure. All input/output/diagnostic storage is disjoint.
#[no_mangle]
pub unsafe extern "C" fn j2k_decoder_decode_region_into(
    decoder: *mut J2kCDecoder,
    options: *const J2kCOperationOptions,
    output: *mut u8,
    output_len: usize,
    stride: usize,
    out: *mut J2kCOutputInfo,
    error: *mut J2kCError,
) -> u32 {
    // SAFETY: The caller supplies live non-overlapping storage; validate spans before slicing.
    unsafe {
        boundary(error, || {
            if !out.is_null() {
                validate_struct(out)?;
            }
            check_bytes(output, output_len)?;
            with_decoder(decoder, |state| {
                let request = operation_request(options, state.image_size())?;
                let plan = state.plan_operation(&request)?;
                let required = state.operation_output_span(&request, plan.output, stride)?;
                if output_len < required {
                    return Err(Error::new(
                        J2K_BUFFER_TOO_SMALL,
                        "output buffer is shorter than the requested region",
                    ));
                }
                let info = state.decode_operation(
                    &request,
                    plan,
                    slice::from_raw_parts_mut(output, required),
                    stride,
                )?;
                if !out.is_null() {
                    out.write(info);
                }
                Ok(())
            })
        })
    }
}

/// Bound derived native operation plans retained by this decoder; zero disables caching.
/// Original input preparation and encoded input are separate owners.
///
/// # Safety
/// The decoder is live and exclusively used. Optional error storage is valid,
/// aligned, writable, and disjoint from decoder storage.
#[no_mangle]
pub unsafe extern "C" fn j2k_decoder_set_operation_cache_limit(
    decoder: *mut J2kCDecoder,
    bytes: usize,
    error: *mut J2kCError,
) -> u32 {
    // SAFETY: The caller keeps the decoder live and grants exclusive access.
    unsafe {
        boundary(error, || {
            with_decoder(decoder, |state| {
                state.session.set_operation_cache_limit_bytes(bytes);
                Ok(())
            })
        })
    }
}

/// Return the C ABI revision.
#[no_mangle]
pub extern "C" fn j2k_abi_version() -> u32 {
    J2K_ABI_VERSION
}

/// Return an immutable, process-lifetime NUL-terminated library version string.
#[no_mangle]
pub extern "C" fn j2k_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// Inspect and copy an encoded input into a new CPU decoder.
///
/// # Safety
/// Non-null arguments must point to valid, appropriately aligned allocations.
/// `input` must cover `input_len` readable bytes. `out` must be a writable pointer
/// slot that does not own a live decoder. Options must contain a readable ABI
/// header and the prefix declared by it. All output arguments must be disjoint
/// from each other and all inputs. `options` and `error` may be null.
#[no_mangle]
pub unsafe extern "C" fn j2k_decoder_open(
    input: *const u8,
    input_len: usize,
    options: *const J2kCOpenOptions,
    out: *mut *mut J2kCDecoder,
    error: *mut J2kCError,
) -> u32 {
    if check_pointer(out).is_ok() {
        // SAFETY: `out` is a non-null, aligned, writable slot by contract. Clearing it before
        // `error` is validated keeps the null-on-failure promise for every failure.
        unsafe { out.write(ptr::null_mut()) };
    }
    // SAFETY: Caller provides the valid storage documented above; prefixes and lengths are checked before access.
    unsafe {
        boundary(error, || {
            check_pointer(out)?;
            let options = read_options(options)?;
            if input_len as u64 > Decoder::input_limit(options)? {
                return Err(Error::limit("encoded input exceeds max_input_bytes"));
            }
            check_bytes(input, input_len)?;
            if input_len == 0 {
                return Err(Error::new(J2K_INVALID_INPUT, "encoded input is empty"));
            }
            let state = Decoder::open(slice::from_raw_parts(input, input_len), options)?;
            out.write(Box::into_raw(Box::new(state)).cast());
            Ok(())
        })
    }
}

/// Release a decoder and set the caller's slot to null. A null handle is a no-op.
///
/// # Safety
/// `decoder` must point to a writable slot containing null or a live handle from
/// this library. No other call may use the handle during destruction or later.
/// `error`, if non-null, must be writable and disjoint from the handle and slot.
#[no_mangle]
pub unsafe extern "C" fn j2k_decoder_close(
    decoder: *mut *mut J2kCDecoder,
    error: *mut J2kCError,
) -> u32 {
    // SAFETY: Slot ownership and exclusivity are the caller's obligations.
    unsafe {
        boundary(error, || {
            check_pointer(decoder)?;
            let handle = decoder.read().cast::<Decoder>();
            if !handle.is_null() {
                check_pointer(handle)?;
                decoder.write(ptr::null_mut());
                drop(Box::from_raw(handle));
            }
            Ok(())
        })
    }
}

/// Read source image metadata.
///
/// # Safety
/// The handle must be live and exclusively used for this call. `out` must be a
/// writable, initialized ABI structure. `error` may be null; otherwise it must
/// be writable. Arguments must be aligned, valid, and non-overlapping.
#[no_mangle]
pub unsafe extern "C" fn j2k_decoder_info(
    decoder: *mut J2kCDecoder,
    out: *mut J2kCImageInfo,
    error: *mut J2kCError,
) -> u32 {
    // SAFETY: Caller supplies exclusive live storage; validate the output prefix before mutation.
    unsafe {
        boundary(error, || {
            validate_struct(out)?;
            with_decoder(decoder, |state| {
                out.write(state.info());
                Ok(())
            })
        })
    }
}

/// Read precision and sampling of a zero-based source component.
///
/// # Safety
/// Same storage and exclusivity requirements as [`j2k_decoder_info`].
#[no_mangle]
pub unsafe extern "C" fn j2k_decoder_component_info(
    decoder: *mut J2kCDecoder,
    index: u32,
    out: *mut J2kCComponentInfo,
    error: *mut J2kCError,
) -> u32 {
    // SAFETY: Caller supplies live disjoint storage; validate the output prefix before mutation.
    unsafe {
        boundary(error, || {
            validate_struct(out)?;
            with_decoder(decoder, |state| {
                out.write(state.component(index)?);
                Ok(())
            })
        })
    }
}

/// Read the colour specification decoding applies: the file's first COLR box.
///
/// # Safety
/// Same storage and exclusivity requirements as [`j2k_decoder_info`].
#[no_mangle]
pub unsafe extern "C" fn j2k_decoder_color_info(
    decoder: *mut J2kCDecoder,
    out: *mut J2kCColorInfo,
    error: *mut J2kCError,
) -> u32 {
    // SAFETY: Caller supplies exclusive live storage; validate the output prefix before mutation.
    unsafe {
        boundary(error, || {
            validate_struct(out)?;
            with_decoder(decoder, |state| {
                out.write(state.color_info());
                Ok(())
            })
        })
    }
}

/// Copy the ICC profile decoding applies into caller-owned storage.
///
/// # Safety
/// The decoder must be live and exclusively used. `output` must cover
/// `output_len` writable bytes disjoint from every other argument. `error` may
/// be null; otherwise it must be writable.
#[no_mangle]
pub unsafe extern "C" fn j2k_decoder_icc_profile(
    decoder: *mut J2kCDecoder,
    output: *mut u8,
    output_len: usize,
    error: *mut J2kCError,
) -> u32 {
    // SAFETY: Caller supplies live disjoint storage; the copy is bounded by the checked length.
    unsafe {
        boundary(error, || {
            check_bytes(output, output_len)?;
            with_decoder(decoder, |state| {
                let profile = state.icc_profile().ok_or_else(|| {
                    Error::argument("the applied colour specification has no ICC profile")
                })?;
                if output_len < profile.len() {
                    return Err(Error::new(
                        J2K_BUFFER_TOO_SMALL,
                        "output is shorter than the ICC profile",
                    ));
                }
                ptr::copy_nonoverlapping(profile.as_ptr(), output, profile.len());
                Ok(())
            })
        })
    }
}

/// Prepare a decode request and report its exact native output layout.
///
/// # Safety
/// Same requirements as [`j2k_decoder_info`]. `options` may be null, otherwise
/// it must contain a valid readable structure prefix disjoint from outputs.
#[no_mangle]
pub unsafe extern "C" fn j2k_decoder_output_info(
    decoder: *mut J2kCDecoder,
    options: *const J2kCDecodeOptions,
    out: *mut J2kCOutputInfo,
    error: *mut J2kCError,
) -> u32 {
    // SAFETY: Caller supplies live disjoint storage; validate versioned prefixes first.
    unsafe {
        boundary(error, || {
            validate_struct(out)?;
            let options = read_options(options)?;
            with_decoder(decoder, |state| {
                out.write(state.prepare(options)?);
                Ok(())
            })
        })
    }
}

/// Decode native interleaved samples into a caller-owned strided byte buffer.
///
/// # Safety
/// The decoder must be live and exclusively used. `output` must cover
/// `output_len` writable bytes and must not overlap any other argument. Options
/// and optional error storage follow [`j2k_decoder_output_info`].
#[no_mangle]
pub unsafe extern "C" fn j2k_decoder_decode_into(
    decoder: *mut J2kCDecoder,
    options: *const J2kCDecodeOptions,
    output: *mut u8,
    output_len: usize,
    stride: usize,
    error: *mut J2kCError,
) -> u32 {
    // SAFETY: Caller owns the destination and handle; validate sizes before making a mutable slice.
    unsafe {
        boundary(error, || {
            let options = read_options(options)?;
            check_bytes(output, output_len)?;
            with_decoder(decoder, |state| {
                let info = state.prepare(options)?;
                let required = state.output_span(info, stride)?;
                if output_len < required {
                    return Err(Error::new(
                        J2K_BUFFER_TOO_SMALL,
                        "output buffer is shorter than the requested strided image",
                    ));
                }
                state.decode_prepared(slice::from_raw_parts_mut(output, required), stride)
            })
        })
    }
}

/// `j2k_row_callback`: receives one row of `row_bytes` bytes at `row`, valid
/// only during the call. Returns zero to continue and nonzero to stop.
pub type J2kCRowCallback = Option<
    unsafe extern "C" fn(user_data: *mut c_void, y: u32, row: *const u8, row_bytes: usize) -> i32,
>;

/// Report the layout `j2k_decoder_decode_rows` delivers: the full image at full
/// resolution in the native samples `j2k_decoder_output_info` describes.
///
/// # Safety
/// Same requirements as [`j2k_decoder_output_info`].
#[no_mangle]
pub unsafe extern "C" fn j2k_decoder_row_info(
    decoder: *mut J2kCDecoder,
    options: *const J2kCDecodeOptions,
    out: *mut J2kCOutputInfo,
    error: *mut J2kCError,
) -> u32 {
    // SAFETY: Caller supplies live disjoint storage; validate versioned prefixes first.
    unsafe {
        boundary(error, || {
            validate_struct(out)?;
            let options = read_options(options)?;
            with_decoder(decoder, |state| {
                out.write(state.row_info(options)?);
                Ok(())
            })
        })
    }
}

/// Decode the full image and hand its rows to `callback` in order, once each.
///
/// # Safety
/// The decoder must be live and exclusively used. `callback` must be a valid
/// function that may be called with `user_data` from a thread of this library
/// while the call blocks, and must not unwind. `out`, if non-null, must be a
/// writable, initialized ABI structure. Options and `error` follow
/// [`j2k_decoder_output_info`].
#[no_mangle]
pub unsafe extern "C" fn j2k_decoder_decode_rows(
    decoder: *mut J2kCDecoder,
    options: *const J2kCDecodeOptions,
    max_rows_per_stripe: u32,
    callback: J2kCRowCallback,
    user_data: *mut c_void,
    out: *mut J2kCOutputInfo,
    error: *mut J2kCError,
) -> u32 {
    // SAFETY: Caller supplies live disjoint storage and a callable callback;
    // prefixes are validated before any row is decoded.
    unsafe {
        boundary(error, || {
            if !out.is_null() {
                validate_struct(out)?;
            }
            let options = read_options(options)?;
            let function = callback.ok_or_else(|| Error::argument("row callback is null"))?;
            let target = RowTarget {
                function,
                user_data,
            };
            let mut consumer = move |y: u32, row: &[u8]| target.call(y, row);
            let info = with_decoder(decoder, |state| {
                state.decode_rows(options, max_rows_per_stripe, &mut consumer)
            })?;
            if !out.is_null() {
                out.write(info);
            }
            Ok(())
        })
    }
}

/// The caller's row callback and its data.
#[derive(Clone, Copy)]
struct RowTarget {
    function: unsafe extern "C" fn(*mut c_void, u32, *const u8, usize) -> i32,
    user_data: *mut c_void,
}

// SAFETY: `j2k_decoder_decode_rows` documents that the callback and its data
// may be used from a library thread while the call blocks; rows are delivered
// one at a time, never concurrently.
unsafe impl Send for RowTarget {}

impl RowTarget {
    /// Hand one row to the callback: whether to continue.
    fn call(self, y: u32, row: &[u8]) -> bool {
        // SAFETY: The caller provides a valid callback; `row` stays valid for the call.
        unsafe { (self.function)(self.user_data, y, row.as_ptr(), row.len()) == 0 }
    }
}

/// Encode interleaved samples losslessly into a buffer this library allocates:
/// [`j2k_encode`] with default lossless parameters and no report.
///
/// # Safety
/// `samples` must cover `samples_len` readable bytes. `options` must be a valid
/// ABI structure. `out` and `out_len` must be writable and disjoint from all
/// other arguments. `error` may be null. Release a successful result with
/// [`j2k_buffer_free`] exactly once.
#[no_mangle]
pub unsafe extern "C" fn j2k_encode_lossless(
    samples: *const u8,
    samples_len: usize,
    options: *const J2kCEncodeOptions,
    out: *mut *mut u8,
    out_len: *mut usize,
    error: *mut J2kCError,
) -> u32 {
    // SAFETY: The caller's contract for this call is j2k_encode's with null
    // parameters (lossless defaults) and a null report.
    unsafe {
        j2k_encode(
            samples,
            samples_len,
            options,
            ptr::null(),
            out,
            out_len,
            ptr::null_mut(),
            error,
        )
    }
}

/// Encode interleaved samples with coding parameters into a buffer this
/// library allocates, optionally reporting what the codec measured.
///
/// # Safety
/// As [`j2k_encode_lossless`]. `params` may be null for lossless defaults;
/// otherwise it must be a valid ABI structure whose `targets` cover
/// `target_count` readable values and whose `icc_profile` covers
/// `icc_profile_len` readable bytes. `report` may be null; otherwise it must
/// be a writable, initialized ABI structure disjoint from all other arguments.
#[no_mangle]
pub unsafe extern "C" fn j2k_encode(
    samples: *const u8,
    samples_len: usize,
    options: *const J2kCEncodeOptions,
    params: *const J2kCEncodeParams,
    out: *mut *mut u8,
    out_len: *mut usize,
    report: *mut J2kCEncodeReport,
    error: *mut J2kCError,
) -> u32 {
    if check_pointer(out).is_ok() && check_pointer(out_len).is_ok() {
        // SAFETY: Both slots are non-null, aligned and writable by contract;
        // clearing them first keeps the empty-on-failure promise.
        unsafe {
            out.write(ptr::null_mut());
            out_len.write(0);
        }
    }
    // SAFETY: Caller provides the valid storage documented above; prefixes and lengths are checked before access.
    unsafe {
        boundary(error, || {
            check_pointer(out)?;
            check_pointer(out_len)?;
            if !report.is_null() {
                validate_struct(report)?;
            }
            if options.is_null() {
                return Err(Error::argument("encode options are required"));
            }
            let options = read_options(options)?;
            let params = encode_params(params)?;
            check_bytes(samples, samples_len)?;
            let encoded = encode(
                slice::from_raw_parts(samples, samples_len),
                &options,
                &params,
            )?;
            let bytes = encoded.bytes.into_boxed_slice();
            let length = bytes.len();
            out.write(Box::into_raw(bytes).cast::<u8>());
            out_len.write(length);
            if !report.is_null() {
                report.write(encoded.report);
            }
            Ok(())
        })
    }
}

/// The parameters `params` points at, with its arrays as slices, or lossless
/// defaults for a null pointer.
///
/// # Safety
/// `params` is null or a readable ABI structure whose `targets` and
/// `icc_profile` arrays are readable for their declared lengths while the
/// returned borrow is used.
unsafe fn encode_params<'a>(params: *const J2kCEncodeParams) -> Result<EncodeParams<'a>> {
    if params.is_null() {
        return Ok(EncodeParams::default());
    }
    // SAFETY: Caller supplies a readable structure prefix.
    let fields = unsafe { read_options(params)? };
    let targets = match usize::try_from(fields.target_count) {
        Ok(0) => &[][..],
        Ok(count) => {
            check_slice(fields.targets, count)?;
            // SAFETY: The checked, aligned pointer covers `count` values by contract.
            unsafe { slice::from_raw_parts(fields.targets, count) }
        }
        Err(_) => return Err(Error::argument("target count exceeds the address space")),
    };
    let icc_profile = match (fields.icc_profile.is_null(), fields.icc_profile_len) {
        (true, 0) => None,
        (true, _) => return Err(Error::argument("an ICC profile length without a profile")),
        (false, 0) => return Err(Error::argument("an empty ICC profile")),
        (false, length) => {
            check_bytes(fields.icc_profile, length)?;
            // SAFETY: The checked pointer covers `length` bytes by contract.
            Some(unsafe { slice::from_raw_parts(fields.icc_profile, length) })
        }
    };
    Ok(EncodeParams {
        fields,
        targets,
        icc_profile,
    })
}

/// Release a buffer from [`j2k_encode`] or [`j2k_encode_lossless`]. A null
/// buffer is a no-op.
///
/// # Safety
/// `data` must be null or a buffer this library returned, with the length it
/// returned, not yet released.
#[no_mangle]
pub unsafe extern "C" fn j2k_buffer_free(data: *mut u8, length: usize) {
    if !data.is_null() {
        // SAFETY: `j2k_encode` made this pointer and length from a
        // boxed slice, which the caller hands back once.
        drop(unsafe { Box::from_raw(ptr::slice_from_raw_parts_mut(data, length)) });
    }
}

fn check_pointer<T>(pointer: *const T) -> Result<()> {
    if pointer.is_null() || !(pointer as usize).is_multiple_of(align_of::<T>()) {
        return Err(Error::argument("required pointer is null or misaligned"));
    }
    Ok(())
}

fn check_bytes(pointer: *const u8, length: usize) -> Result<()> {
    check_slice(pointer, length)
}

/// A non-null, aligned pointer to `length` values within the address space.
fn check_slice<T>(pointer: *const T, length: usize) -> Result<()> {
    check_pointer(pointer)?;
    let bytes = length.saturating_mul(size_of::<T>());
    if bytes > isize::MAX as usize || (pointer as usize).checked_add(bytes).is_none() {
        return Err(Error::argument(
            "buffer length exceeds the host address space",
        ));
    }
    Ok(())
}

// T must be one of this crate's repr(C) structs starting with J2kCHeader.
unsafe fn validate_struct<T>(pointer: *const T) -> Result<()> {
    check_pointer(pointer)?;
    // SAFETY: Caller promises a readable header; all internal calls use header-prefixed ABI structs.
    let header = unsafe { pointer.cast::<J2kCHeader>().read() };
    if header.abi_version != J2K_ABI_VERSION || (header.struct_size as usize) < size_of::<T>() {
        return Err(Error::new(
            J2K_ABI_MISMATCH,
            "unsupported ABI version or structure too small",
        ));
    }
    Ok(())
}

unsafe fn read_options<T: Copy + Default>(pointer: *const T) -> Result<T> {
    if pointer.is_null() {
        return Ok(T::default());
    }
    // SAFETY: Caller supplies a readable prefix; internal uses are restricted to header-prefixed option types.
    unsafe {
        validate_struct(pointer)?;
        Ok(pointer.read())
    }
}

unsafe fn with_decoder<T>(
    pointer: *mut J2kCDecoder,
    operation: impl FnOnce(&mut Decoder) -> Result<T>,
) -> Result<T> {
    let pointer = pointer.cast::<Decoder>();
    check_pointer(pointer)?;
    // SAFETY: A live handle was allocated as Box<Decoder>; caller guarantees exclusive access.
    let decoder = unsafe { &mut *pointer };
    if decoder.poisoned {
        return Err(Error::new(
            J2K_PANIC,
            "decoder is poisoned; close it and create a new one",
        ));
    }
    decoder.poisoned = true;
    let result = operation(decoder);
    decoder.poisoned =
        matches!(&result, Err(error) if matches!(error.code, J2K_PANIC | J2K_INTERNAL_ERROR));
    result
}

unsafe fn boundary(error: *mut J2kCError, operation: impl FnOnce() -> Result<()>) -> u32 {
    if !error.is_null() && check_pointer(error).is_err() {
        return J2K_INVALID_ARGUMENT;
    }
    let result = match catch_unwind(AssertUnwindSafe(operation)) {
        Ok(result) => result,
        Err(payload) => {
            // A custom panic payload may panic again during Drop.
            std::mem::forget(payload);
            Err(Error::new(
                J2K_PANIC,
                "Rust panic contained at the C ABI boundary",
            ))
        }
    };
    let diagnostic = result
        .err()
        .map_or_else(J2kCError::default, |value| value.diagnostic());
    let code = diagnostic.code;
    if !error.is_null() {
        // SAFETY: Caller provides disjoint writable error storage; alignment was checked above.
        unsafe {
            error.write(diagnostic);
        }
    }
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unwinding_is_contained_and_poisoned_decoder_cannot_be_reused() {
        let samples = [1_u8; 16];
        let image = j2k::encode_j2k_lossless(
            j2k::J2kLosslessSamples::new(&samples, 4, 4, 1, 8, false).unwrap(),
            &j2k::J2kLosslessEncodeOptions::default(),
        )
        .unwrap();
        let mut decoder = Decoder::open(&image.codestream, J2kCOpenOptions::default()).unwrap();
        let handle = ptr::from_mut(&mut decoder).cast();
        let mut error = J2kCError::default();
        // SAFETY: Stack decoder is exclusively borrowed and is not passed to close.
        let status = unsafe {
            boundary(&raw mut error, || {
                with_decoder(handle, |_| panic!("test panic"))
            })
        };
        assert_eq!(status, J2K_PANIC);
        // SAFETY: Same live exclusive stack decoder, now poisoned.
        let status = unsafe { boundary(&raw mut error, || with_decoder(handle, |_| Ok(()))) };
        assert_eq!(status, J2K_PANIC);
        assert_eq!(error.code, status);
        // SAFETY: Reset through the same raw handle so the stack borrow stays valid.
        unsafe { (*handle.cast::<Decoder>()).poisoned = false };
        // SAFETY: Same live exclusive stack decoder; an invariant failure also poisons it.
        let status = unsafe {
            boundary(&raw mut error, || {
                with_decoder(handle, |_| Err::<(), _>(Error::internal("test")))
            })
        };
        assert_eq!(status, J2K_INTERNAL_ERROR);
        // SAFETY: Same live exclusive stack decoder, now poisoned.
        let status = unsafe { boundary(&raw mut error, || with_decoder(handle, |_| Ok(()))) };
        assert_eq!(status, J2K_PANIC);
        // Diagnostics truncate at a UTF-8 boundary and stay NUL-terminated.
        let message = Error::new(J2K_INTERNAL_ERROR, "\u{e9}".repeat(200))
            .diagnostic()
            .message
            .map(|c| u8::from_ne_bytes(c.to_ne_bytes()));
        assert_eq!(
            (&message[252..254], &message[254..]),
            ("\u{e9}".as_bytes(), &[0_u8, 0][..])
        );
    }
}
