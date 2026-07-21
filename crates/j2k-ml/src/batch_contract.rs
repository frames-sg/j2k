// SPDX-License-Identifier: MIT OR Apache-2.0

use burn_core::tensor::DType;
use j2k::{BatchGroupInfo, BatchLayout, NativeSampleType};

use crate::BurnDecodeError;

pub(crate) fn tensor_shape(
    batch: usize,
    info: &BatchGroupInfo,
) -> Result<[usize; 4], BurnDecodeError> {
    let width = usize::try_from(info.dimensions.0).map_err(|_| BurnDecodeError::SizeOverflow)?;
    let height = usize::try_from(info.dimensions.1).map_err(|_| BurnDecodeError::SizeOverflow)?;
    let channels = info.color.channels();
    let samples = batch
        .checked_mul(width)
        .and_then(|value| value.checked_mul(height))
        .and_then(|value| value.checked_mul(channels))
        .ok_or(BurnDecodeError::SizeOverflow)?;
    if samples == 0 {
        return Err(BurnDecodeError::SizeOverflow);
    }
    Ok(match info.layout {
        BatchLayout::Nchw => [batch, channels, height, width],
        BatchLayout::Nhwc => [batch, height, width, channels],
        _ => return Err(BurnDecodeError::UnsupportedCodecContract),
    })
}

pub(crate) const fn dtype(sample_type: NativeSampleType) -> Result<DType, BurnDecodeError> {
    Ok(match sample_type {
        NativeSampleType::U8 => DType::U8,
        NativeSampleType::U16 => DType::U16,
        NativeSampleType::I16 => DType::I16,
        _ => return Err(BurnDecodeError::UnsupportedCodecContract),
    })
}
