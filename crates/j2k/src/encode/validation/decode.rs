// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained-capacity decode and typed error mapping for generated output.

use alloc::{string::String, vec::Vec};

use j2k_native::{
    DecodeError, DecodeSettings, DecodedNativeComponents, Image, RawBitmap,
    DEFAULT_MAX_DECODE_BYTES,
};

use crate::{BackendError, BackendErrorKind, J2kError};

fn validation_image<'a>(
    codestream: &'a [u8],
    retained_capacity: usize,
    context: &'static str,
) -> Result<Image<'a>, J2kError> {
    Image::new_with_retained_baseline(codestream, &DecodeSettings::default(), retained_capacity)
        .map_err(|source| validation_decode_error(source, context))
}

// This intentionally receives the `Vec`: validation budgets its retained
// allocation capacity before passing the initialized bytes to the parser.
pub(super) fn decode_native_for_validation(
    codestream: &Vec<u8>,
    context: &'static str,
) -> Result<RawBitmap, J2kError> {
    let retained_capacity = validation_retained_capacity(codestream.capacity(), context)?;
    validation_image(codestream, retained_capacity, context)?
        .decode_native_with_retained_capacity(retained_capacity)
        .map_err(|source| validation_decode_error(source, context))
}

// See `decode_native_for_validation` for the `Vec` ownership contract.
pub(super) fn decode_native_components_for_validation(
    codestream: &Vec<u8>,
    context: &'static str,
) -> Result<DecodedNativeComponents, J2kError> {
    let retained_capacity = validation_retained_capacity(codestream.capacity(), context)?;
    validation_image(codestream, retained_capacity, context)?
        .decode_native_components_with_retained_capacity(retained_capacity)
        .map_err(|source| validation_decode_error(source, context))
}

pub(super) fn validation_retained_capacity(
    codestream_capacity: usize,
    context: &'static str,
) -> Result<usize, J2kError> {
    let requested = codestream_capacity;
    if requested > DEFAULT_MAX_DECODE_BYTES {
        return Err(validation_decode_error(
            DecodeError::AllocationTooLarge {
                what: "facade encode validation retained codestreams",
                requested,
                cap: DEFAULT_MAX_DECODE_BYTES,
            },
            context,
        ));
    }
    Ok(requested)
}

pub(super) fn validation_decode_error(source: DecodeError, context: &'static str) -> J2kError {
    J2kError::NativeValidation {
        context,
        source: crate::NativeBackendError::decode(source),
    }
}

pub(super) fn output_validation_error(message: impl Into<String>) -> J2kError {
    J2kError::Backend(BackendError::new(BackendErrorKind::Validation, message))
}

pub(super) fn raw_bitmap_metadata_matches(
    decoded: &RawBitmap,
    width: u32,
    height: u32,
    components: u16,
    bit_depth: u8,
    signed: bool,
) -> bool {
    decoded.width == width
        && decoded.height == height
        && decoded.num_components == components
        && decoded.bit_depth == bit_depth
        && decoded.signed == signed
        && decoded.component_signed.len() == usize::from(components)
        && decoded
            .component_signed
            .iter()
            .all(|component_signed| *component_signed == signed)
}
