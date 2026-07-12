// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CudaHtj2kEncodeStageTimings, CudaHtj2kEncodeTables, CudaTranscodeError,
    Htj2k97CodeBlockOptions, PreencodedHtj2k97CompactSubband, PreencodedHtj2k97Subband,
};
use j2k_transcode::Htj2k97CodeBlockOptionsError;

mod orchestration;
mod output;
mod planning;

pub(super) use orchestration::{
    device_band_groups_to_compact_preencoded_components,
    device_band_groups_to_preencoded_components, encode_resident_compact_subbands,
    encode_resident_subbands,
};
pub(super) use output::{
    assemble_compact_preencoded_components, assemble_preencoded_components, Htj2k97ComponentJob,
};
pub(super) use planning::ResidentDeviceGroup;

pub(super) type ResidentSubbands = (
    Vec<PreencodedHtj2k97Subband>,
    Vec<PreencodedHtj2k97Subband>,
    Vec<PreencodedHtj2k97Subband>,
    Vec<PreencodedHtj2k97Subband>,
    CudaHtj2kEncodeStageTimings,
    usize,
);

pub(super) type CompactResidentSubbands = (
    Vec<u8>,
    Vec<PreencodedHtj2k97CompactSubband>,
    Vec<PreencodedHtj2k97CompactSubband>,
    Vec<PreencodedHtj2k97CompactSubband>,
    Vec<PreencodedHtj2k97CompactSubband>,
    CudaHtj2kEncodeStageTimings,
    usize,
);

pub(super) fn to_u32(value: usize) -> Result<u32, CudaTranscodeError> {
    u32::try_from(value).map_err(|_| CudaTranscodeError::Kernel("CUDA value exceeds u32"))
}

pub(super) fn cuda_htj2k_encode_tables() -> CudaHtj2kEncodeTables<'static> {
    CudaHtj2kEncodeTables {
        vlc_table0: j2k_native::ht_vlc_encode_table0(),
        vlc_table1: j2k_native::ht_vlc_encode_table1(),
        uvlc_table: j2k_native::ht_uvlc_encode_table_bytes(),
    }
}

pub(super) fn validate_band_len(
    band: &[i32],
    item_count: usize,
    item_size: usize,
) -> Result<(), CudaTranscodeError> {
    let expected = item_count
        .checked_mul(item_size)
        .ok_or(CudaTranscodeError::Kernel(
            "CUDA 9/7 code-block band length overflow",
        ))?;
    if band.len() != expected {
        return Err(CudaTranscodeError::Kernel(
            "CUDA 9/7 code-block band output length mismatch",
        ));
    }
    Ok(())
}

pub(super) fn validate_htj2k97_codeblock_options(
    options: Htj2k97CodeBlockOptions,
) -> Result<(usize, usize), CudaTranscodeError> {
    j2k_transcode::validate_htj2k97_codeblock_options(options)
        .map_err(unsupported_htj2k97_codeblock_options)
}

fn unsupported_htj2k97_codeblock_options(
    error: Htj2k97CodeBlockOptionsError,
) -> CudaTranscodeError {
    match error {
        Htj2k97CodeBlockOptionsError::NumericOptionsOutOfRange => {
            CudaTranscodeError::UnsupportedJob(
                Htj2k97CodeBlockOptionsError::NumericOptionsOutOfRange.reason(),
            )
        }
        Htj2k97CodeBlockOptionsError::QuantizationOptionsOutOfRange => {
            CudaTranscodeError::UnsupportedJob(
                Htj2k97CodeBlockOptionsError::QuantizationOptionsOutOfRange.reason(),
            )
        }
        Htj2k97CodeBlockOptionsError::DimensionExponentUnsupported {
            axis,
            exponent_minus_two,
        } => CudaTranscodeError::UnsupportedJob(
            Htj2k97CodeBlockOptionsError::DimensionExponentUnsupported {
                axis,
                exponent_minus_two,
            }
            .reason(),
        ),
        Htj2k97CodeBlockOptionsError::DimensionsExceedLimits { width, height } => {
            CudaTranscodeError::UnsupportedJob(
                Htj2k97CodeBlockOptionsError::DimensionsExceedLimits { width, height }.reason(),
            )
        }
        _ => CudaTranscodeError::UnsupportedJob(error.reason()),
    }
}

pub(super) fn htj2k97_code_block_dim(exp_minus_two: u8) -> Result<usize, CudaTranscodeError> {
    1usize
        .checked_shl(u32::from(exp_minus_two) + 2)
        .ok_or(CudaTranscodeError::UnsupportedJob(
            "CUDA 9/7 code-block exponent is too large",
        ))
}

#[cfg(test)]
mod tests;
