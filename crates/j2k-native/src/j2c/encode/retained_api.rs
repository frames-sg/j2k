// SPDX-License-Identifier: MIT OR Apache-2.0

//! Internal adapter seams for codec-owned allocations retained during encode.

use super::{
    encode_with_accelerator_and_component_sample_info_for_session, EncodeOptions,
    J2kEncodeStageAccelerator, NativeEncodeRetainedInput, NativeEncodeSession, Vec,
};

/// Encode pixels while accounting caller-owned allocations that remain live.
///
/// The retained-input token immutably borrows its owners for this complete
/// call. Ordinary public entrypoints use a zero baseline.
///
/// # Errors
///
/// Returns a typed error for invalid input, checked size or cap failure, host
/// allocation failure, accelerator failure, or codestream validation failure.
#[expect(
    clippy::too_many_arguments,
    reason = "this codec boundary keeps geometry, state buffers, and validated options explicit without allocation or indirection"
)]
pub(crate) fn encode_with_accelerator_and_retained_input(
    pixels: &[u8],
    width: u32,
    height: u32,
    num_components: u16,
    bit_depth: u8,
    signed: bool,
    options: &EncodeOptions,
    retained_input: NativeEncodeRetainedInput<'_>,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> crate::EncodeResult<Vec<u8>> {
    let session = NativeEncodeSession::try_new(retained_input)?;
    encode_with_accelerator_and_component_sample_info_for_session(
        pixels,
        width,
        height,
        num_components,
        bit_depth,
        signed,
        options,
        &[],
        &session,
        accelerator,
    )
}
