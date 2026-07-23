// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host staging followed by ordinary Burn tensor materialization.

use burn_core::tensor::{backend::Backend, DType, Tensor, TensorData};
use j2k::{BatchGroupInfo, J2kDecodeWarning, NativeSampleType, Rect};

use crate::batch_contract::tensor_shape;
use crate::{BurnBatchGroup, BurnBatchTensor, BurnDecodeError};

pub(crate) fn byte_len(batch: usize, info: &BatchGroupInfo) -> Result<usize, BurnDecodeError> {
    let samples = tensor_shape(batch, info)?
        .into_iter()
        .try_fold(1usize, usize::checked_mul)
        .ok_or(BurnDecodeError::SizeOverflow)?;
    let bytes_per_sample = match info.sample_type {
        NativeSampleType::U8 => 1,
        NativeSampleType::U16 | NativeSampleType::I16 => 2,
        _ => return Err(BurnDecodeError::UnsupportedCodecContract),
    };
    samples
        .checked_mul(bytes_per_sample)
        .ok_or(BurnDecodeError::SizeOverflow)
}

pub(crate) fn materialize<B: Backend>(
    info: BatchGroupInfo,
    source_indices: Vec<usize>,
    decoded_rects: Vec<Rect>,
    warnings: Vec<Vec<J2kDecodeWarning>>,
    bytes: Vec<u8>,
    device: &B::Device,
) -> Result<BurnBatchGroup<B>, BurnDecodeError> {
    let shape = tensor_shape(source_indices.len(), &info)?;
    let expected = byte_len(source_indices.len(), &info)?;
    if bytes.len() != expected {
        return Err(BurnDecodeError::StagingSizeMismatch {
            expected,
            actual: bytes.len(),
        });
    }
    let tensor = match info.sample_type {
        NativeSampleType::U8 => BurnBatchTensor::U8(Tensor::from_data(
            TensorData::new(bytes, shape),
            (device, DType::U8),
        )),
        NativeSampleType::U16 => {
            let samples = bytes
                .chunks_exact(2)
                .map(|bytes| u16::from_ne_bytes([bytes[0], bytes[1]]))
                .collect();
            BurnBatchTensor::U16(Tensor::from_data(
                TensorData::new(samples, shape),
                (device, DType::U16),
            ))
        }
        NativeSampleType::I16 => {
            let samples = bytes
                .chunks_exact(2)
                .map(|bytes| i16::from_ne_bytes([bytes[0], bytes[1]]))
                .collect();
            BurnBatchTensor::I16(Tensor::from_data(
                TensorData::new(samples, shape),
                (device, DType::I16),
            ))
        }
        _ => return Err(BurnDecodeError::UnsupportedCodecContract),
    };
    Ok(BurnBatchGroup {
        tensor,
        info,
        source_indices,
        decoded_rects,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use j2k::{
        BatchAlpha, BatchCodecRoute, BatchColor, BatchLayout, BatchWaveletTransform,
        CompressedPayloadKind, CompressedTransferSyntax,
    };
    use j2k_core::Colorspace;

    use super::byte_len;

    #[test]
    fn staged_byte_length_tracks_native_width_and_layout() {
        let info = j2k::BatchGroupInfo {
            dimensions: (3, 2),
            sample_type: j2k::NativeSampleType::U16,
            precision: 12,
            signed: false,
            color: BatchColor::Rgb,
            alpha: BatchAlpha::None,
            transform: BatchWaveletTransform::Reversible53,
            route: BatchCodecRoute::Htj2k,
            layout: BatchLayout::Nchw,
            colorspace: Colorspace::SRgb,
            transfer_syntax: CompressedTransferSyntax::HtJpeg2000Lossless,
            payload_kind: CompressedPayloadKind::Jpeg2000Codestream,
        };
        assert_eq!(byte_len(4, &info).unwrap(), 4 * 3 * 2 * 3 * 2);
    }
}
