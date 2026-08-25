// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{BatchDecodeOptions, EncodedImage, PreparedBatch, PreparedBatchGroup, PreparedImage};

use crate::{Error, MpsGraphTensorSpec};

/// Unavailable `MPSGraph` decoder on non-Apple-Silicon targets.
#[derive(Debug)]
pub struct MpsGraphBatchDecoder;

/// Unavailable `MPSGraph` input group on non-Apple-Silicon targets.
#[derive(Debug)]
pub struct MpsGraphInputGroup;

/// Unavailable `MPSGraph` batch result on non-Apple-Silicon targets.
#[derive(Debug)]
pub struct MpsGraphBatchDecode;

/// Unavailable graph program on non-Apple-Silicon targets.
#[derive(Debug)]
pub struct MpsGraphProgram;

/// Unavailable submitted graph run on non-Apple-Silicon targets.
#[derive(Debug)]
pub struct SubmittedMpsGraphRun;

/// Unavailable completed graph output on non-Apple-Silicon targets.
#[derive(Debug)]
pub struct MpsGraphRunOutput;

impl MpsGraphBatchDecoder {
    /// Return the typed platform error on unsupported targets.
    pub fn system_default(_options: BatchDecodeOptions) -> Result<Self, Error> {
        Err(Error::UnsupportedPlatform)
    }

    /// Return the typed platform error on unsupported targets.
    pub fn prepare(&self, _inputs: Vec<EncodedImage>) -> Result<PreparedBatch, Error> {
        Err(Error::UnsupportedPlatform)
    }

    /// Return the typed platform error on unsupported targets.
    pub fn prepare_prepared_images(
        &self,
        _images: Vec<PreparedImage>,
    ) -> Result<PreparedBatch, Error> {
        Err(Error::UnsupportedPlatform)
    }

    /// Return the typed platform error on unsupported targets.
    pub fn decode(&mut self, _inputs: Vec<EncodedImage>) -> Result<MpsGraphBatchDecode, Error> {
        Err(Error::UnsupportedPlatform)
    }

    /// Return the typed platform error on unsupported targets.
    pub fn decode_prepared(
        &mut self,
        _prepared: &PreparedBatch,
    ) -> Result<MpsGraphBatchDecode, Error> {
        Err(Error::UnsupportedPlatform)
    }

    /// Return the typed platform error on unsupported targets.
    pub fn decode_prepared_images(
        &mut self,
        _images: Vec<PreparedImage>,
    ) -> Result<MpsGraphBatchDecode, Error> {
        Err(Error::UnsupportedPlatform)
    }

    /// Return the typed platform error on unsupported targets.
    pub fn submit_prepared_group(
        &mut self,
        _program: &MpsGraphProgram,
        _group: &PreparedBatchGroup,
    ) -> Result<SubmittedMpsGraphRun, Error> {
        Err(Error::UnsupportedPlatform)
    }

    /// Return the typed platform error on unsupported targets.
    pub fn run_prepared_group(
        &mut self,
        _program: &MpsGraphProgram,
        _group: &PreparedBatchGroup,
    ) -> Result<MpsGraphRunOutput, Error> {
        Err(Error::UnsupportedPlatform)
    }
}

impl MpsGraphProgram {
    /// Return the typed platform error on unsupported targets.
    pub fn identity(_input_spec: MpsGraphTensorSpec) -> Result<Self, Error> {
        Err(Error::UnsupportedPlatform)
    }

    /// Return the typed platform error on unsupported targets.
    pub fn rgb8_nhwc_reference(
        _batch: usize,
        _height: usize,
        _width: usize,
    ) -> Result<Self, Error> {
        Err(Error::UnsupportedPlatform)
    }
}

impl SubmittedMpsGraphRun {
    /// No graph can be submitted on this target.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        false
    }

    /// Return the typed platform error on unsupported targets.
    pub fn wait(self) -> Result<MpsGraphRunOutput, Error> {
        Err(Error::UnsupportedPlatform)
    }
}
