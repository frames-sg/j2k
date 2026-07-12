// SPDX-License-Identifier: MIT OR Apache-2.0
// j2k-coverage: shared-accelerator-host

use alloc::vec::Vec;

use j2k_core::{BackendKind, Unsupported};

use super::contracts::{
    EncodeBackendPreference, J2kBlockCodingMode, J2kLosslessEncodeOptions, J2kLossyEncodeOptions,
    J2kMarkerSegment, ReversibleTransform,
};
use super::lossy::lossy_quality_layer_count;
use super::native::{native_lossless_options, native_lossy_options};
use super::samples::{J2kLosslessSamples, J2kLossySamples};
use super::{
    j2k_lossless_decomposition_levels_for_resident_geometry,
    j2k_lossy_decomposition_levels_for_options,
};
use crate::{J2kEncodeDispatchReport, J2kEncodeStageAccelerator, J2kError, J2kResidentEncodeInput};

pub(super) fn resolve_encode_backend(
    preference: EncodeBackendPreference,
) -> Result<BackendKind, J2kError> {
    match preference {
        EncodeBackendPreference::Auto | EncodeBackendPreference::CpuOnly => Ok(BackendKind::Cpu),
        EncodeBackendPreference::RequireDevice => Err(J2kError::Unsupported(Unsupported {
            what: "device JPEG 2000 lossless encode backend is unavailable",
        })),
    }
}

pub(super) fn resolve_accelerated_encode_backend(
    preference: EncodeBackendPreference,
    accelerated_backend: BackendKind,
    dispatch: J2kEncodeDispatchReport,
    required_stages: RequiredEncodeStages,
) -> Result<BackendKind, J2kError> {
    if required_stages.satisfied_by(dispatch) {
        return Ok(accelerated_backend);
    }
    match preference {
        EncodeBackendPreference::RequireDevice => Err(J2kError::Unsupported(Unsupported {
            what: required_stages.missing_message(dispatch),
        })),
        EncodeBackendPreference::Auto | EncodeBackendPreference::CpuOnly => Ok(BackendKind::Cpu),
    }
}

pub(super) fn encode_with_native_accelerator(
    samples: J2kLosslessSamples<'_>,
    options: J2kLosslessEncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, J2kError> {
    let options = native_lossless_options(samples, options);
    j2k_native::encode_with_accelerator(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &options,
        accelerator,
    )
    .map_err(|source| {
        J2kError::from_native_encode_error_with_context(
            source,
            "accelerated native JPEG 2000 lossless encode failed",
        )
    })
}

pub(super) fn encode_lossy_with_native_accelerator(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    quantization_scale: f32,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<u8>, J2kError> {
    let options = native_lossy_options(samples, options, quantization_scale)?;
    j2k_native::encode_with_accelerator(
        samples.data,
        samples.width,
        samples.height,
        samples.components,
        samples.bit_depth,
        samples.signed,
        &options,
        accelerator,
    )
    .map_err(|source| {
        J2kError::from_native_encode_error_with_context(
            source,
            "accelerated native JPEG 2000 lossy encode failed",
        )
    })
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RequiredEncodeStages {
    bits: u16,
}

impl RequiredEncodeStages {
    const DEINTERLEAVE: u16 = 1 << 0;
    const FORWARD_RCT: u16 = 1 << 1;
    const FORWARD_DWT53: u16 = 1 << 2;
    const TIER1_CODE_BLOCK: u16 = 1 << 3;
    const HT_CODE_BLOCK: u16 = 1 << 4;
    const PACKETIZATION: u16 = 1 << 5;
    const QUANTIZE_SUBBAND: u16 = 1 << 6;
    const FORWARD_ICT: u16 = 1 << 7;
    const FORWARD_DWT97: u16 = 1 << 8;

    fn satisfied_by(self, dispatch: J2kEncodeDispatchReport) -> bool {
        self.missing_stage(dispatch).is_none()
    }

    fn missing_message(self, dispatch: J2kEncodeDispatchReport) -> &'static str {
        match self.missing_stage(dispatch) {
            Some("deinterleave") => {
                "requested JPEG 2000 device encode backend did not dispatch deinterleave"
            }
            Some("forward_rct") => {
                "requested JPEG 2000 device encode backend did not dispatch forward_rct"
            }
            Some("forward_ict") => {
                "requested JPEG 2000 device encode backend did not dispatch forward_ict"
            }
            Some("forward_dwt53") => {
                "requested JPEG 2000 device encode backend did not dispatch forward_dwt53"
            }
            Some("forward_dwt97") => {
                "requested JPEG 2000 device encode backend did not dispatch forward_dwt97"
            }
            Some("tier1_code_block") => {
                "requested JPEG 2000 device encode backend did not dispatch tier1_code_block"
            }
            Some("ht_code_block") => {
                "requested JPEG 2000 device encode backend did not dispatch ht_code_block"
            }
            Some("quantize_subband") => {
                "requested JPEG 2000 device encode backend did not dispatch quantize_subband"
            }
            Some("packetization") => {
                "requested JPEG 2000 device encode backend did not dispatch packetization"
            }
            _ => "requested JPEG 2000 device encode backend did not dispatch",
        }
    }

    fn missing_stage(self, dispatch: J2kEncodeDispatchReport) -> Option<&'static str> {
        if self.contains(Self::DEINTERLEAVE) && dispatch.deinterleave == 0 {
            return Some("deinterleave");
        }
        if self.contains(Self::FORWARD_RCT) && dispatch.forward_rct == 0 {
            return Some("forward_rct");
        }
        if self.contains(Self::FORWARD_ICT) && dispatch.forward_ict == 0 {
            return Some("forward_ict");
        }
        if self.contains(Self::FORWARD_DWT53) && dispatch.forward_dwt53 == 0 {
            return Some("forward_dwt53");
        }
        if self.contains(Self::FORWARD_DWT97) && dispatch.forward_dwt97 == 0 {
            return Some("forward_dwt97");
        }
        if self.contains(Self::TIER1_CODE_BLOCK) && dispatch.tier1_code_block == 0 {
            return Some("tier1_code_block");
        }
        if self.contains(Self::HT_CODE_BLOCK) && dispatch.ht_code_block == 0 {
            return Some("ht_code_block");
        }
        if self.contains(Self::QUANTIZE_SUBBAND) && dispatch.quantize_subband == 0 {
            return Some("quantize_subband");
        }
        if self.contains(Self::PACKETIZATION) && dispatch.packetization == 0 {
            return Some("packetization");
        }
        None
    }

    fn contains(self, stage: u16) -> bool {
        self.bits & stage != 0
    }
}

pub(super) fn required_encode_stages(
    samples: J2kLosslessSamples<'_>,
    options: J2kLosslessEncodeOptions,
    accelerated_backend: BackendKind,
) -> RequiredEncodeStages {
    required_encode_stages_for_geometry(
        samples.width,
        samples.height,
        samples.components,
        options,
        accelerated_backend,
    )
}

pub(super) fn required_resident_encode_stages(
    input: J2kResidentEncodeInput,
    options: J2kLosslessEncodeOptions,
    accelerated_backend: BackendKind,
) -> RequiredEncodeStages {
    required_encode_stages_for_geometry(
        input.width(),
        input.height(),
        input.num_components(),
        options,
        accelerated_backend,
    )
}

fn required_encode_stages_for_geometry(
    width: u32,
    height: u32,
    components: u16,
    options: J2kLosslessEncodeOptions,
    accelerated_backend: BackendKind,
) -> RequiredEncodeStages {
    let decomposition_levels =
        j2k_lossless_decomposition_levels_for_resident_geometry(width, height, options);
    let high_throughput = options.block_coding_mode == J2kBlockCodingMode::HighThroughput;

    let mut bits = RequiredEncodeStages::PACKETIZATION;
    if matches!(accelerated_backend, BackendKind::Cuda | BackendKind::Metal) {
        bits |= RequiredEncodeStages::DEINTERLEAVE | RequiredEncodeStages::QUANTIZE_SUBBAND;
    }
    if matches!(components, 3 | 4) && options.reversible_transform == ReversibleTransform::Rct53 {
        bits |= RequiredEncodeStages::FORWARD_RCT;
    }
    if decomposition_levels > 0 {
        bits |= RequiredEncodeStages::FORWARD_DWT53;
    }
    if high_throughput {
        bits |= RequiredEncodeStages::HT_CODE_BLOCK;
    } else {
        bits |= RequiredEncodeStages::TIER1_CODE_BLOCK;
    }

    RequiredEncodeStages { bits }
}

pub(super) fn required_lossy_encode_stages(
    samples: J2kLossySamples<'_>,
    options: &J2kLossyEncodeOptions,
    accelerated_backend: BackendKind,
) -> RequiredEncodeStages {
    let decomposition_levels = j2k_lossy_decomposition_levels_for_options(samples, options);
    let high_throughput = options.block_coding_mode == J2kBlockCodingMode::HighThroughput;

    let scalar_packetization_required = lossy_quality_layer_count(options) > 1
        || options.marker_segments.contains(&J2kMarkerSegment::Plt)
        || options.marker_segments.contains(&J2kMarkerSegment::Plm)
        || options.marker_segments.contains(&J2kMarkerSegment::Sop)
        || options.marker_segments.contains(&J2kMarkerSegment::Eph);
    let mut bits = 0;
    if !scalar_packetization_required || accelerated_backend == BackendKind::Metal {
        bits |= RequiredEncodeStages::PACKETIZATION;
    }
    if matches!(accelerated_backend, BackendKind::Cuda | BackendKind::Metal) {
        bits |= RequiredEncodeStages::DEINTERLEAVE | RequiredEncodeStages::QUANTIZE_SUBBAND;
        if matches!(samples.components, 3 | 4) {
            bits |= RequiredEncodeStages::FORWARD_ICT;
        }
        if decomposition_levels > 0 {
            bits |= RequiredEncodeStages::FORWARD_DWT97;
        }
    }
    if high_throughput {
        bits |= RequiredEncodeStages::HT_CODE_BLOCK;
    } else {
        bits |= RequiredEncodeStages::TIER1_CODE_BLOCK;
    }

    RequiredEncodeStages { bits }
}
