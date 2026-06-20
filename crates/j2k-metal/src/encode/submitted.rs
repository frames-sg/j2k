// SPDX-License-Identifier: Apache-2.0

use j2k::{EncodedJ2k, J2kLosslessEncodeOptions};
use j2k_core::DeviceSubmission;
#[cfg(target_os = "macos")]
use j2k_core::PixelFormat;
#[cfg(target_os = "macos")]
use metal::Buffer;

use super::MetalLosslessBufferEncodeBatchOutcome;
#[cfg(target_os = "macos")]
use super::{
    encode_lossless_owned_tiles_with_report,
    encode_owned_lossless_tiles_to_metal_buffer_fallback_batch,
    wait_submitted_resident_lossless_buffer_encode_batch, MetalEncodeInputStaging,
    MetalLosslessEncodeConfig, MetalLosslessEncodeTile,
    SubmittedResidentLosslessMetalBufferEncodeBatch,
};

#[cfg(target_os = "macos")]
/// Submitted multi-tile Metal encode that resolves to host codestream bytes.
pub struct SubmittedJ2kLosslessMetalEncodeBatch {
    pub(super) state: SubmittedJ2kLosslessMetalEncodeBatchState,
}

#[cfg(target_os = "macos")]
/// Submitted multi-tile Metal encode that resolves to Metal-backed codestreams.
pub struct SubmittedJ2kLosslessMetalBufferEncodeBatch {
    pub(super) state: SubmittedJ2kLosslessMetalBufferEncodeBatchState,
}

#[cfg(target_os = "macos")]
pub(super) enum SubmittedJ2kLosslessMetalEncodeBatchState {
    Ready(Vec<EncodedJ2k>),
    Deferred {
        tiles: Vec<OwnedMetalLosslessEncodeTile>,
        options: J2kLosslessEncodeOptions,
        session: crate::MetalBackendSession,
        staging: MetalEncodeInputStaging,
        config: MetalLosslessEncodeConfig,
    },
}

#[cfg(target_os = "macos")]
pub(super) enum SubmittedJ2kLosslessMetalBufferEncodeBatchState {
    Resident(Box<SubmittedResidentLosslessMetalBufferEncodeBatch>),
    Deferred {
        tiles: Vec<OwnedMetalLosslessEncodeTile>,
        options: J2kLosslessEncodeOptions,
        session: crate::MetalBackendSession,
        staging: MetalEncodeInputStaging,
    },
}

#[cfg(target_os = "macos")]
pub(super) struct OwnedMetalLosslessEncodeTile {
    pub(super) buffer: Buffer,
    pub(super) byte_offset: usize,
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) pitch_bytes: usize,
    pub(super) output_width: u32,
    pub(super) output_height: u32,
    pub(super) format: PixelFormat,
}

#[cfg(target_os = "macos")]
impl OwnedMetalLosslessEncodeTile {
    pub(super) fn from_tile(tile: MetalLosslessEncodeTile<'_>) -> Self {
        Self {
            buffer: tile.buffer.to_owned(),
            byte_offset: tile.byte_offset,
            width: tile.width,
            height: tile.height,
            pitch_bytes: tile.pitch_bytes,
            output_width: tile.output_width,
            output_height: tile.output_height,
            format: tile.format,
        }
    }

    pub(super) fn as_tile(&self) -> MetalLosslessEncodeTile<'_> {
        MetalLosslessEncodeTile {
            buffer: &self.buffer,
            byte_offset: self.byte_offset,
            width: self.width,
            height: self.height,
            pitch_bytes: self.pitch_bytes,
            output_width: self.output_width,
            output_height: self.output_height,
            format: self.format,
        }
    }
}

#[cfg(not(target_os = "macos"))]
/// Placeholder submitted multi-tile encode for non-macOS builds.
pub struct SubmittedJ2kLosslessMetalEncodeBatch {
    _private: (),
}

#[cfg(not(target_os = "macos"))]
/// Placeholder submitted Metal-buffer encode for non-macOS builds.
pub struct SubmittedJ2kLosslessMetalBufferEncodeBatch {
    _private: (),
}

#[cfg(target_os = "macos")]
impl DeviceSubmission for SubmittedJ2kLosslessMetalEncodeBatch {
    type Output = Vec<EncodedJ2k>;
    type Error = crate::Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        match self.state {
            SubmittedJ2kLosslessMetalEncodeBatchState::Ready(encoded) => Ok(encoded),
            SubmittedJ2kLosslessMetalEncodeBatchState::Deferred {
                tiles,
                options,
                session,
                staging,
                config,
            } => {
                encode_lossless_owned_tiles_with_report(&tiles, options, &session, staging, config)
                    .map(|outcomes| {
                        outcomes
                            .into_iter()
                            .map(|outcome| outcome.encoded)
                            .collect()
                    })
            }
        }
    }
}

#[cfg(target_os = "macos")]
impl DeviceSubmission for SubmittedJ2kLosslessMetalBufferEncodeBatch {
    type Output = MetalLosslessBufferEncodeBatchOutcome;
    type Error = crate::Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        match self.state {
            SubmittedJ2kLosslessMetalBufferEncodeBatchState::Resident(submitted) => {
                wait_submitted_resident_lossless_buffer_encode_batch(*submitted)
            }
            SubmittedJ2kLosslessMetalBufferEncodeBatchState::Deferred {
                tiles,
                options,
                session,
                staging,
            } => encode_owned_lossless_tiles_to_metal_buffer_fallback_batch(
                &tiles, options, &session, staging,
            ),
        }
    }
}

#[cfg(not(target_os = "macos"))]
impl DeviceSubmission for SubmittedJ2kLosslessMetalEncodeBatch {
    type Output = Vec<EncodedJ2k>;
    type Error = crate::Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        let _ = self;
        Err(crate::Error::MetalUnavailable)
    }
}

#[cfg(not(target_os = "macos"))]
impl DeviceSubmission for SubmittedJ2kLosslessMetalBufferEncodeBatch {
    type Output = MetalLosslessBufferEncodeBatchOutcome;
    type Error = crate::Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        let _ = self;
        Err(crate::Error::MetalUnavailable)
    }
}
