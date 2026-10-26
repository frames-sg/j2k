// SPDX-License-Identifier: MIT OR Apache-2.0

//! Candidate sources for the lossy rate and PSNR searches.

use alloc::vec::Vec;

use j2k_native::{EncodeRoiRegion as NativeEncodeRoiRegion, IrreversibleScaleEncoder};

use super::super::contracts::J2kLossyEncodeOptions;
use super::super::cpu::native_lossy_options;
use super::super::samples::J2kLossySamples;
use super::super::validation::decoded_psnr;
use super::{
    encode_cpu_lossy, encode_cpu_lossy_with_roi_regions, LOSSY_ENCODE_CONTEXT,
    LOSSY_ROI_ENCODE_CONTEXT,
};
use crate::J2kError;

/// One encode and one quality measurement per probed quantization scale.
pub(in crate::encode) trait LossyCandidates {
    /// Whether the byte search may return its last probe's codestream when
    /// that probe is the selected scale, instead of encoding it again. Only
    /// sound when `encode_candidate` is deterministic per scale. The `false`
    /// default keeps closure sources, including accelerator encodes, on the
    /// final re-encode.
    fn reuse_terminal_byte_output(&self) -> bool {
        false
    }

    fn encode_candidate(&mut self, quantization_scale: f32) -> Result<Vec<u8>, J2kError>;

    // See `validate_lossy_roundtrip` for the `Vec` ownership contract.
    fn candidate_psnr(
        &mut self,
        samples: J2kLossySamples<'_>,
        codestream: &Vec<u8>,
    ) -> Result<f64, J2kError> {
        decoded_psnr(samples, codestream, 0)
    }
}

impl<F: FnMut(f32) -> Result<Vec<u8>, J2kError>> LossyCandidates for F {
    fn encode_candidate(&mut self, quantization_scale: f32) -> Result<Vec<u8>, J2kError> {
        self(quantization_scale)
    }
}

/// CPU candidates. With `reuse`, one colour transform and DWT serve every
/// scale, and quality decodes charge the bytes it keeps live; each candidate
/// still equals the independent CPU encode at its scale.
pub(super) struct CpuCandidates<'a> {
    samples: J2kLossySamples<'a>,
    options: &'a J2kLossyEncodeOptions,
    roi_regions: Option<&'a [NativeEncodeRoiRegion]>,
    encoder: Option<IrreversibleScaleEncoder<'a>>,
    reuse: bool,
}

impl<'a> CpuCandidates<'a> {
    pub(super) fn new(
        samples: J2kLossySamples<'a>,
        options: &'a J2kLossyEncodeOptions,
        roi_regions: Option<&'a [NativeEncodeRoiRegion]>,
        reuse: bool,
    ) -> Self {
        Self {
            samples,
            options,
            roi_regions,
            encoder: None,
            reuse,
        }
    }

    fn measure(
        &mut self,
        samples: J2kLossySamples<'_>,
        codestream: &Vec<u8>,
        retained_bytes: usize,
    ) -> Result<f64, J2kError> {
        let measured = decoded_psnr(samples, codestream, retained_bytes);
        if measured.is_ok() || retained_bytes == 0 {
            return measured;
        }
        // Not a silent failure: the retained bytes can refuse a decode the
        // independent search runs. Release them; the plain decode's outcome,
        // Ok or Err, is authoritative.
        self.release();
        decoded_psnr(samples, codestream, 0)
    }

    /// Later candidates encode independently, which is byte-identical.
    fn release(&mut self) {
        self.encoder = None;
        self.reuse = false;
    }
}

/// Above this, a retained transform held beside each quality decode costs more
/// peak memory than its reuse saves time (no gain measured at 4096^2 RGB).
const RELEASE_BEFORE_DECODE_BYTES: usize = j2k_native::DEFAULT_MAX_DECODE_BYTES / 4;

impl LossyCandidates for CpuCandidates<'_> {
    fn reuse_terminal_byte_output(&self) -> bool {
        true
    }

    fn encode_candidate(&mut self, quantization_scale: f32) -> Result<Vec<u8>, J2kError> {
        let (samples, options) = (self.samples, self.options);
        if !self.reuse {
            return self.roi_regions.map_or_else(
                || encode_cpu_lossy(samples, options, quantization_scale),
                |roi| encode_cpu_lossy_with_roi_regions(samples, options, quantization_scale, roi),
            );
        }
        let encoder = match self.encoder.take() {
            Some(encoder) => encoder,
            // The same call, and error point, as an independent candidate's.
            None => IrreversibleScaleEncoder::new(
                samples.data,
                samples.width,
                samples.height,
                samples.components,
                samples.bit_depth,
                samples.signed,
                native_lossy_options(samples, options, quantization_scale)?,
                self.roi_regions.unwrap_or(&[]),
            ),
        };
        let context = if self.roi_regions.is_some() {
            LOSSY_ROI_ENCODE_CONTEXT
        } else {
            LOSSY_ENCODE_CONTEXT
        };
        self.encoder
            .insert(encoder)
            .encode_at_scale(quantization_scale)
            .map_err(|source| J2kError::from_native_encode_error_with_context(source, context))
    }

    fn candidate_psnr(
        &mut self,
        samples: J2kLossySamples<'_>,
        codestream: &Vec<u8>,
    ) -> Result<f64, J2kError> {
        let mut retained = self
            .encoder
            .as_ref()
            .map_or(0, IrreversibleScaleEncoder::retained_bytes);
        if retained > RELEASE_BEFORE_DECODE_BYTES {
            self.release();
            retained = 0;
        }
        self.measure(samples, codestream, retained)
    }
}

#[cfg(test)]
mod tests;
