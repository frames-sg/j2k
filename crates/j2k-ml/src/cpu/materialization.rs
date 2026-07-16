// SPDX-License-Identifier: MIT OR Apache-2.0

use burn_core::tensor::{backend::Backend, DType, Int, Tensor, TensorData};

use super::{PackedBatch, PackedImage};
use crate::{FloatNormalization, TensorDecodeError, TensorLayout};

#[derive(Debug, Clone, Copy)]
pub(crate) enum SampleWidth {
    U8,
    U16,
}

impl SampleWidth {
    pub(crate) const fn bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
        }
    }

    pub(crate) const fn dtype(self) -> DType {
        match self {
            Self::U8 => DType::U8,
            Self::U16 => DType::U16,
        }
    }

    const fn unit_denominator(self) -> f32 {
        match self {
            Self::U8 => 255.0,
            Self::U16 => 65_535.0,
        }
    }
}

pub(super) fn integer_tensor_3<B: Backend>(
    packed: &PackedImage,
    layout: TensorLayout,
    device: &B::Device,
    dtype: DType,
) -> Tensor<B, 3, Int> {
    integer_tensor_3_from_bytes(packed.bytes.clone(), packed.shape, layout, device, dtype)
}

pub(crate) fn integer_tensor_3_from_bytes<B: Backend>(
    bytes: Vec<u8>,
    shape: [usize; 3],
    layout: TensorLayout,
    device: &B::Device,
    dtype: DType,
) -> Tensor<B, 3, Int> {
    let tensor = match dtype {
        DType::U8 => Tensor::from_data(TensorData::new(bytes, shape), (device, DType::U8)),
        DType::U16 => Tensor::from_data(
            TensorData::new(bytes_to_u16(&bytes), shape),
            (device, DType::U16),
        ),
        _ => unreachable!("portable integer decode only uses U8 or U16"),
    };
    match layout {
        TensorLayout::ChannelsFirst => tensor.permute([2, 0, 1]),
        TensorLayout::ChannelsLast => tensor,
    }
}

pub(super) fn integer_tensor_4<B: Backend>(
    packed: &PackedBatch,
    layout: TensorLayout,
    device: &B::Device,
    dtype: DType,
) -> Tensor<B, 4, Int> {
    integer_tensor_4_from_bytes(
        packed.bytes.clone(),
        packed.outcomes.len(),
        packed.shape,
        layout,
        device,
        dtype,
    )
}

pub(crate) fn integer_tensor_4_from_bytes<B: Backend>(
    bytes: Vec<u8>,
    batch: usize,
    item_shape: [usize; 3],
    layout: TensorLayout,
    device: &B::Device,
    dtype: DType,
) -> Tensor<B, 4, Int> {
    let shape = [batch, item_shape[0], item_shape[1], item_shape[2]];
    let tensor = match dtype {
        DType::U8 => Tensor::from_data(TensorData::new(bytes, shape), (device, DType::U8)),
        DType::U16 => Tensor::from_data(
            TensorData::new(bytes_to_u16(&bytes), shape),
            (device, DType::U16),
        ),
        _ => unreachable!("portable integer decode only uses U8 or U16"),
    };
    match layout {
        TensorLayout::ChannelsFirst => tensor.permute([0, 3, 1, 2]),
        TensorLayout::ChannelsLast => tensor,
    }
}

fn bytes_to_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(2)
        .map(|sample| u16::from_ne_bytes([sample[0], sample[1]]))
        .collect()
}

pub(crate) fn validate_normalization_values(
    normalization: &FloatNormalization,
) -> Result<(), TensorDecodeError> {
    let FloatNormalization::MeanStd { mean, std } = normalization else {
        return Ok(());
    };
    if mean.iter().chain(std).any(|value| !value.is_finite()) {
        return Err(invalid_normalization("mean and std values must be finite"));
    }
    if std.contains(&0.0) {
        return Err(invalid_normalization("standard deviations must be nonzero"));
    }
    Ok(())
}

pub(crate) fn validate_normalization_channels(
    normalization: &FloatNormalization,
    channels: usize,
) -> Result<(), TensorDecodeError> {
    let FloatNormalization::MeanStd { mean, std } = normalization else {
        return Ok(());
    };
    if mean.len() != channels || std.len() != channels {
        return Err(invalid_normalization(format!(
            "mean and std must each contain {channels} values; got {} and {}",
            mean.len(),
            std.len()
        )));
    }
    Ok(())
}

fn invalid_normalization(message: impl Into<String>) -> TensorDecodeError {
    TensorDecodeError::InvalidNormalization {
        message: message.into(),
    }
}

pub(crate) fn normalize_3<B: Backend>(
    tensor: Tensor<B, 3>,
    normalization: &FloatNormalization,
    layout: TensorLayout,
    channels: usize,
    width: SampleWidth,
    device: &B::Device,
) -> Tensor<B, 3> {
    match normalization {
        FloatNormalization::Raw => tensor,
        FloatNormalization::Unit => tensor.div_scalar(width.unit_denominator()),
        FloatNormalization::MeanStd { mean, std } => {
            let unit = tensor.div_scalar(width.unit_denominator());
            let (mean, std) = normalization_tensors(mean, std, channels, layout, device);
            unit.sub(mean.reshape(shape_3(channels, layout)))
                .div(std.reshape(shape_3(channels, layout)))
        }
    }
}

pub(crate) fn normalize_4<B: Backend>(
    tensor: Tensor<B, 4>,
    normalization: &FloatNormalization,
    layout: TensorLayout,
    channels: usize,
    width: SampleWidth,
    device: &B::Device,
) -> Tensor<B, 4> {
    match normalization {
        FloatNormalization::Raw => tensor,
        FloatNormalization::Unit => tensor.div_scalar(width.unit_denominator()),
        FloatNormalization::MeanStd { mean, std } => {
            let unit = tensor.div_scalar(width.unit_denominator());
            let (mean, std) = normalization_tensors(mean, std, channels, layout, device);
            unit.sub(mean.reshape(shape_4(channels, layout)))
                .div(std.reshape(shape_4(channels, layout)))
        }
    }
}

fn normalization_tensors<B: Backend>(
    mean: &[f32],
    std: &[f32],
    channels: usize,
    _layout: TensorLayout,
    device: &B::Device,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    (
        Tensor::from_data(
            TensorData::new(mean.to_vec(), [channels]),
            (device, DType::F32),
        ),
        Tensor::from_data(
            TensorData::new(std.to_vec(), [channels]),
            (device, DType::F32),
        ),
    )
}

const fn shape_3(channels: usize, layout: TensorLayout) -> [usize; 3] {
    match layout {
        TensorLayout::ChannelsFirst => [channels, 1, 1],
        TensorLayout::ChannelsLast => [1, 1, channels],
    }
}

const fn shape_4(channels: usize, layout: TensorLayout) -> [usize; 4] {
    match layout {
        TensorLayout::ChannelsFirst => [1, channels, 1, 1],
        TensorLayout::ChannelsLast => [1, 1, 1, channels],
    }
}
