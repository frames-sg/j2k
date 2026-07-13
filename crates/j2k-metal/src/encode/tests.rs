// SPDX-License-Identifier: MIT OR Apache-2.0

use super::MetalEncodeStageAccelerator;
#[cfg(target_os = "macos")]
use crate::compute;
use j2k::{
    encode_j2k_lossless_with_accelerator, EncodeBackendPreference, EncodedJ2k,
    J2kLosslessEncodeOptions, J2kLosslessSamples,
};
#[cfg(target_os = "macos")]
use j2k::{
    encode_j2k_lossy_with_accelerator, J2kBlockCodingMode, J2kEncodeValidation,
    J2kLossyEncodeOptions, J2kLossySamples, J2kMarkerSegment, J2kProgressionOrder,
    ReversibleTransform,
};
#[cfg(target_os = "macos")]
use j2k::{J2kDeinterleaveToF32Job, J2kForwardDwt53Job, J2kForwardIctJob, J2kQuantizeSubbandJob};
use j2k::{J2kEncodeDispatchReport, J2kEncodeStageAccelerator, J2kForwardRctJob};
#[cfg(target_os = "macos")]
use j2k_core::CodecError;
use j2k_core::DeviceSubmission;
#[cfg(target_os = "macos")]
use j2k_core::{BackendKind, PixelFormat};
#[cfg(target_os = "macos")]
use j2k_native::{
    forward_dwt53_reference, quantize_reversible_reference as quantize_reference,
    try_deinterleave_reference, EncodeOptions, J2kCodeBlockStyle,
};
use j2k_native::{DecodeSettings, Image};
#[cfg(target_os = "macos")]
use metal::foreign_types::ForeignType;
use std::time::Duration;

#[cfg(target_os = "macos")]
macro_rules! lossless_options {
    ($($field:ident: $value:expr),+ $(,)?) => {{
        let mut options = J2kLosslessEncodeOptions::default();
        $(options.$field = $value;)+
        options
    }};
}

mod batch;
#[cfg(target_os = "macos")]
mod dwt_parity;
#[cfg(target_os = "macos")]
mod kernels;
#[cfg(target_os = "macos")]
mod layouts;
#[cfg(target_os = "macos")]
mod resident_batches;
#[cfg(target_os = "macos")]
mod resident_buffers;
#[cfg(target_os = "macos")]
mod routing;
mod stage_validation;
mod stats_inflight;

#[cfg(target_os = "macos")]
fn assert_decoded_bytes_match(actual: &[u8], expected: &[u8]) {
    if actual == expected {
        return;
    }
    let mismatch = actual
        .iter()
        .zip(expected.iter())
        .position(|(actual, expected)| actual != expected)
        .unwrap_or_else(|| actual.len().min(expected.len()));
    let actual_value = actual.get(mismatch).copied();
    let expected_value = expected.get(mismatch).copied();
    panic!(
        "decoded bytes mismatch at byte {mismatch}: actual={actual_value:?} expected={expected_value:?} actual_len={} expected_len={}",
        actual.len(),
        expected.len()
    );
}

#[cfg(target_os = "macos")]
fn should_run_metal_runtime() -> bool {
    j2k_test_support::metal_runtime_gate(module_path!())
}
