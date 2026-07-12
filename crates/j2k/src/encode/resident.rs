// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendKind, Unsupported};

use super::contracts::{
    EncodeBackendPreference, EncodedJ2k, J2kBlockCodingMode, J2kEncodeValidation,
    J2kLosslessEncodeOptions, MAX_RAW_PIXEL_ENCODE_BIT_DEPTH,
};
use super::native::encode_resident_with_native_accelerator;
use super::routing::{required_resident_encode_stages, resolve_accelerated_encode_backend};
use crate::{J2kEncodeStageAccelerator, J2kError, J2kResidentEncodeInput};

/// Encode a lossless HTJ2K codestream from backend-resident pixels.
///
/// This adapter-facing entry point has no host pixel slice and therefore has
/// no CPU fallback. The resident whole-tile hook must dispatch successfully,
/// report every required stage, and return a tile body. Validation must be
/// performed externally by the owning backend.
#[doc(hidden)]
pub fn encode_j2k_lossless_resident_with_accelerator(
    input: J2kResidentEncodeInput,
    options: &J2kLosslessEncodeOptions,
    accelerated_backend: BackendKind,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<EncodedJ2k, J2kError> {
    if accelerated_backend == BackendKind::Cpu {
        return Err(J2kError::Unsupported(Unsupported {
            what: "resident JPEG 2000 encode requires a device backend kind",
        }));
    }
    if options.backend == EncodeBackendPreference::CpuOnly {
        return Err(J2kError::Unsupported(Unsupported {
            what: "resident JPEG 2000 encode has no CPU input or fallback",
        }));
    }
    if options.block_coding_mode != J2kBlockCodingMode::HighThroughput {
        return Err(J2kError::Unsupported(Unsupported {
            what: "resident JPEG 2000 encode requires HTJ2K block coding",
        }));
    }
    if options.validation != J2kEncodeValidation::External {
        return Err(J2kError::Unsupported(Unsupported {
            what: "resident JPEG 2000 encode requires external validation",
        }));
    }
    if input.bit_depth() > MAX_RAW_PIXEL_ENCODE_BIT_DEPTH {
        return Err(J2kError::Unsupported(Unsupported {
            what: "resident JPEG 2000 encode supports at most 24 bits per sample",
        }));
    }

    let before = accelerator.dispatch_report();
    let required_stages = required_resident_encode_stages(input, *options, accelerated_backend);
    let codestream = encode_resident_with_native_accelerator(input, *options, accelerator)?;
    let dispatch = accelerator.dispatch_report().saturating_delta(before);
    let backend = resolve_accelerated_encode_backend(
        EncodeBackendPreference::RequireDevice,
        accelerated_backend,
        dispatch,
        required_stages,
    )?;

    Ok(EncodedJ2k {
        codestream,
        backend,
        dispatch_report: dispatch,
        width: input.width(),
        height: input.height(),
        components: input.num_components(),
        bit_depth: input.bit_depth(),
        signed: input.signed(),
    })
}

#[cfg(test)]
mod tests;
