// SPDX-License-Identifier: MIT OR Apache-2.0

//! C ABI over the `j2k` JPEG 2000 / HTJ2K codec.
//!
//! `include/j2k.h` is the C contract and `include/j2k.hpp` its C++17
//! ownership wrapper; `docs/api.md` documents both.

mod abi;
mod decoder;
mod encode;
mod error;
mod ffi;
mod operations;
mod rows;
pub use abi::{
    J2kCComponentInfo, J2kCDecodeOptions, J2kCEncodeOptions, J2kCError, J2kCHeader, J2kCImageInfo,
    J2kCOpenOptions, J2kCOperationOptions, J2kCOutputInfo, J2kCRegionPlan, J2K_ABI_MISMATCH,
    J2K_ABI_VERSION, J2K_ALLOCATION_FAILED, J2K_ALPHA_NONE, J2K_ALPHA_PREMULTIPLIED,
    J2K_ALPHA_STRAIGHT, J2K_BUFFER_TOO_SMALL, J2K_CANCELLED, J2K_DECODE_FAILED, J2K_ENCODE_FAILED,
    J2K_ENCODE_FILE, J2K_ENCODE_HTJ2K, J2K_ENCODE_SIGNED, J2K_I16_LE, J2K_INTERNAL_ERROR,
    J2K_INTERPRETATION_NATIVE, J2K_INTERPRETATION_SRGB8, J2K_INVALID_ARGUMENT, J2K_INVALID_INPUT,
    J2K_OK, J2K_OPEN_LENIENT, J2K_PANIC, J2K_RESOURCE_LIMIT, J2K_ROUTE_NATIVE_CPU,
    J2K_ROUTE_REGION_PLANES, J2K_STOPPED, J2K_U16_LE, J2K_U8, J2K_UNSUPPORTED,
    J2K_WARNING_LENIENT_RECOVERY,
};
pub use ffi::{
    j2k_abi_version, j2k_buffer_free, j2k_cancellation_cancel, j2k_cancellation_create,
    j2k_cancellation_free, j2k_decoder_close, j2k_decoder_component_info, j2k_decoder_decode_into,
    j2k_decoder_decode_region_into, j2k_decoder_decode_rows, j2k_decoder_info, j2k_decoder_open,
    j2k_decoder_output_info, j2k_decoder_plan_region, j2k_decoder_prepare_operations,
    j2k_decoder_row_info, j2k_decoder_set_operation_cache_limit, j2k_encode_lossless, j2k_version,
    J2kCCancellation, J2kCDecoder, J2kCRowCallback,
};
