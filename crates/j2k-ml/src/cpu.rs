// SPDX-License-Identifier: MIT OR Apache-2.0

//! Portable decode into any Burn backend.

use burn_core::tensor::{backend::Backend, DType, FloatDType, Int, Tensor, TensorData};
use j2k::{
    DecodeOutcome, DeviceDecodePlan, DeviceDecodeRequest, J2kDecodeWarning, J2kDecoder,
    J2kScratchPool, PixelFormat,
};

use crate::{
    ChannelSelection, FloatNormalization, TensorBatchDecode, TensorDecode, TensorDecodeError,
    TensorDecodeOptions, TensorInput, TensorLayout, TensorRoute,
};

#[derive(Debug)]
struct PackedImage {
    bytes: Vec<u8>,
    shape: [usize; 3],
    outcome: DecodeOutcome<J2kDecodeWarning>,
}

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

/// Decode one image into a rank-3 U8 tensor.
pub fn decode_u8<B: Backend>(
    input: TensorInput<'_>,
    options: &TensorDecodeOptions,
    device: &B::Device,
) -> Result<TensorDecode<Tensor<B, 3, Int>>, TensorDecodeError> {
    ensure_dtype::<B>(device, DType::U8)?;
    let packed = decode_packed(input, options.channels, SampleWidth::U8)?;
    let tensor = integer_tensor_3::<B>(&packed, options.layout, device, DType::U8);
    Ok(single_result(tensor, packed.outcome))
}

/// Decode one image into a rank-3 U16 tensor.
pub fn decode_u16<B: Backend>(
    input: TensorInput<'_>,
    options: &TensorDecodeOptions,
    device: &B::Device,
) -> Result<TensorDecode<Tensor<B, 3, Int>>, TensorDecodeError> {
    ensure_dtype::<B>(device, DType::U16)?;
    let packed = decode_packed(input, options.channels, SampleWidth::U16)?;
    let tensor = integer_tensor_3::<B>(&packed, options.layout, device, DType::U16);
    Ok(single_result(tensor, packed.outcome))
}

/// Decode one image into a rank-3 F32 tensor.
pub fn decode_float<B: Backend>(
    input: TensorInput<'_>,
    options: &TensorDecodeOptions,
    device: &B::Device,
) -> Result<TensorDecode<Tensor<B, 3>>, TensorDecodeError> {
    validate_normalization_values(&options.normalization)?;
    ensure_dtype::<B>(device, DType::F32)?;

    let info = J2kDecoder::inspect(input.encoded)?;
    let width = if info.bit_depth <= 8 {
        SampleWidth::U8
    } else {
        SampleWidth::U16
    };
    ensure_dtype::<B>(device, width.dtype())?;
    validate_normalization_channels(
        &options.normalization,
        selected_channels(options.channels, info.components),
    )?;
    let packed = decode_packed(input, options.channels, width)?;
    let tensor =
        integer_tensor_3::<B>(&packed, options.layout, device, width.dtype()).cast(FloatDType::F32);
    let tensor = normalize_3(
        tensor,
        &options.normalization,
        options.layout,
        packed.shape[2],
        width,
        device,
    );
    Ok(single_result(tensor, packed.outcome))
}

/// Decode a batch into a rank-4 U8 tensor.
pub fn decode_u8_batch<B: Backend>(
    inputs: &[TensorInput<'_>],
    options: &TensorDecodeOptions,
    device: &B::Device,
) -> Result<TensorBatchDecode<Tensor<B, 4, Int>>, TensorDecodeError> {
    ensure_dtype::<B>(device, DType::U8)?;
    let packed = decode_batch(inputs, options.channels, SampleWidth::U8)?;
    let tensor = integer_tensor_4::<B>(&packed, options.layout, device, DType::U8);
    Ok(batch_result(tensor, packed))
}

/// Decode a batch into a rank-4 U16 tensor.
pub fn decode_u16_batch<B: Backend>(
    inputs: &[TensorInput<'_>],
    options: &TensorDecodeOptions,
    device: &B::Device,
) -> Result<TensorBatchDecode<Tensor<B, 4, Int>>, TensorDecodeError> {
    ensure_dtype::<B>(device, DType::U16)?;
    let packed = decode_batch(inputs, options.channels, SampleWidth::U16)?;
    let tensor = integer_tensor_4::<B>(&packed, options.layout, device, DType::U16);
    Ok(batch_result(tensor, packed))
}

/// Decode a batch into a rank-4 F32 tensor.
pub fn decode_float_batch<B: Backend>(
    inputs: &[TensorInput<'_>],
    options: &TensorDecodeOptions,
    device: &B::Device,
) -> Result<TensorBatchDecode<Tensor<B, 4>>, TensorDecodeError> {
    validate_normalization_values(&options.normalization)?;
    ensure_dtype::<B>(device, DType::F32)?;
    let first = inputs.first().ok_or(TensorDecodeError::EmptyBatch)?;
    let first_info =
        J2kDecoder::inspect(first.encoded).map_err(|source| indexed(0, source.into()))?;
    let bit_depth = first_info.bit_depth;
    let width = if bit_depth <= 8 {
        SampleWidth::U8
    } else {
        SampleWidth::U16
    };
    ensure_dtype::<B>(device, width.dtype())?;
    validate_normalization_channels(
        &options.normalization,
        selected_channels(options.channels, first_info.components),
    )
    .map_err(|error| indexed(0, error))?;

    for (index, input) in inputs.iter().enumerate().skip(1) {
        let item_info =
            J2kDecoder::inspect(input.encoded).map_err(|source| indexed(index, source.into()))?;
        let item_depth = item_info.bit_depth;
        if (item_depth <= 8) != (bit_depth <= 8) {
            return Err(TensorDecodeError::BatchItem {
                index,
                source: Box::new(TensorDecodeError::StrictRoute {
                    route: TensorRoute::CpuStaged,
                    message: format!(
                        "mixed canonical integer widths are unsupported: first item is {bit_depth}-bit, item is {item_depth}-bit"
                    ),
                }),
            });
        }
        validate_normalization_channels(
            &options.normalization,
            selected_channels(options.channels, item_info.components),
        )
        .map_err(|error| indexed(index, error))?;
    }

    let packed = decode_batch(inputs, options.channels, width)?;
    let tensor =
        integer_tensor_4::<B>(&packed, options.layout, device, width.dtype()).cast(FloatDType::F32);
    let tensor = normalize_4(
        tensor,
        &options.normalization,
        options.layout,
        packed.shape[2],
        width,
        device,
    );
    Ok(batch_result(tensor, packed))
}

pub(crate) fn ensure_dtype<B: Backend>(
    device: &B::Device,
    dtype: DType,
) -> Result<(), TensorDecodeError> {
    if B::supports_dtype(device, dtype) {
        Ok(())
    } else {
        Err(TensorDecodeError::UnsupportedDType { dtype })
    }
}

fn decode_packed(
    input: TensorInput<'_>,
    selection: ChannelSelection,
    width: SampleWidth,
) -> Result<PackedImage, TensorDecodeError> {
    let mut decoder = J2kDecoder::new(input.encoded)?;
    let plan = DeviceDecodePlan::for_image(decoder.info().dimensions, input.request)?;
    let channels = selected_channels(selection, decoder.info().components);
    let format = pixel_format(channels, width);
    let (output_width, output_height) = plan.output_dims();
    let width_usize = usize::try_from(output_width).map_err(|_| TensorDecodeError::SizeOverflow)?;
    let height_usize =
        usize::try_from(output_height).map_err(|_| TensorDecodeError::SizeOverflow)?;
    let stride = width_usize
        .checked_mul(format.bytes_per_pixel())
        .ok_or(TensorDecodeError::SizeOverflow)?;
    let byte_len = stride
        .checked_mul(height_usize)
        .ok_or(TensorDecodeError::SizeOverflow)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(byte_len)
        .map_err(|_| TensorDecodeError::SizeOverflow)?;
    bytes.resize(byte_len, 0);
    let mut pool = J2kScratchPool::new();
    let outcome = match input.request {
        DeviceDecodeRequest::Full => decoder.decode_into(&mut bytes, stride, format)?,
        DeviceDecodeRequest::Region { roi } => {
            decoder.decode_region_into(&mut pool, &mut bytes, stride, format, roi)?
        }
        DeviceDecodeRequest::Scaled { scale } => {
            decoder.decode_scaled_into(&mut pool, &mut bytes, stride, format, scale)?
        }
        DeviceDecodeRequest::RegionScaled { roi, scale } => {
            decoder.decode_region_scaled_into(&mut pool, &mut bytes, stride, format, roi, scale)?
        }
    };
    Ok(PackedImage {
        bytes,
        shape: [height_usize, width_usize, channels],
        outcome,
    })
}

pub(crate) fn selected_channels(selection: ChannelSelection, components: u16) -> usize {
    match selection {
        ChannelSelection::Auto if components == 1 => 1,
        ChannelSelection::Auto | ChannelSelection::Rgb => 3,
        ChannelSelection::Gray => 1,
        ChannelSelection::Rgba => 4,
    }
}

pub(crate) fn planned_shape(
    input: TensorInput<'_>,
    selection: ChannelSelection,
) -> Result<[usize; 3], TensorDecodeError> {
    let info = J2kDecoder::inspect(input.encoded)?;
    let plan = DeviceDecodePlan::for_image(info.dimensions, input.request)?;
    let (width, height) = plan.output_dims();
    Ok([
        usize::try_from(height).map_err(|_| TensorDecodeError::SizeOverflow)?,
        usize::try_from(width).map_err(|_| TensorDecodeError::SizeOverflow)?,
        selected_channels(selection, info.components),
    ])
}

pub(crate) fn pixel_format(channels: usize, width: SampleWidth) -> PixelFormat {
    match (channels, width) {
        (1, SampleWidth::U8) => PixelFormat::Gray8,
        (3, SampleWidth::U8) => PixelFormat::Rgb8,
        (4, SampleWidth::U8) => PixelFormat::Rgba8,
        (1, SampleWidth::U16) => PixelFormat::Gray16,
        (3, SampleWidth::U16) => PixelFormat::Rgb16,
        (4, SampleWidth::U16) => PixelFormat::Rgba16,
        _ => unreachable!("channel selection is confined to 1, 3, or 4"),
    }
}

fn integer_tensor_3<B: Backend>(
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

fn integer_tensor_4<B: Backend>(
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
            let (mean, std) = match layout {
                TensorLayout::ChannelsFirst => (
                    Tensor::<B, 1>::from_data(
                        TensorData::new(mean.clone(), [channels]),
                        (device, DType::F32),
                    )
                    .reshape([channels, 1, 1]),
                    Tensor::<B, 1>::from_data(
                        TensorData::new(std.clone(), [channels]),
                        (device, DType::F32),
                    )
                    .reshape([channels, 1, 1]),
                ),
                TensorLayout::ChannelsLast => (
                    Tensor::<B, 1>::from_data(
                        TensorData::new(mean.clone(), [channels]),
                        (device, DType::F32),
                    )
                    .reshape([1, 1, channels]),
                    Tensor::<B, 1>::from_data(
                        TensorData::new(std.clone(), [channels]),
                        (device, DType::F32),
                    )
                    .reshape([1, 1, channels]),
                ),
            };
            unit.sub(mean).div(std)
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
            let (mean, std) = match layout {
                TensorLayout::ChannelsFirst => (
                    Tensor::<B, 1>::from_data(
                        TensorData::new(mean.clone(), [channels]),
                        (device, DType::F32),
                    )
                    .reshape([1, channels, 1, 1]),
                    Tensor::<B, 1>::from_data(
                        TensorData::new(std.clone(), [channels]),
                        (device, DType::F32),
                    )
                    .reshape([1, channels, 1, 1]),
                ),
                TensorLayout::ChannelsLast => (
                    Tensor::<B, 1>::from_data(
                        TensorData::new(mean.clone(), [channels]),
                        (device, DType::F32),
                    )
                    .reshape([1, 1, 1, channels]),
                    Tensor::<B, 1>::from_data(
                        TensorData::new(std.clone(), [channels]),
                        (device, DType::F32),
                    )
                    .reshape([1, 1, 1, channels]),
                ),
            };
            unit.sub(mean).div(std)
        }
    }
}

#[derive(Debug)]
struct PackedBatch {
    bytes: Vec<u8>,
    shape: [usize; 3],
    outcomes: Vec<DecodeOutcome<J2kDecodeWarning>>,
}

fn decode_batch(
    inputs: &[TensorInput<'_>],
    channels: ChannelSelection,
    width: SampleWidth,
) -> Result<PackedBatch, TensorDecodeError> {
    let first = inputs.first().ok_or(TensorDecodeError::EmptyBatch)?;
    let expected_shape = planned_shape(*first, channels).map_err(|error| indexed(0, error))?;
    for (index, input) in inputs.iter().enumerate().skip(1) {
        let actual = planned_shape(*input, channels).map_err(|error| indexed(index, error))?;
        if actual != expected_shape {
            return Err(TensorDecodeError::BatchShapeMismatch {
                index,
                expected: expected_shape,
                actual,
            });
        }
    }
    let first = decode_packed(*first, channels, width).map_err(|error| indexed(0, error))?;
    let shape = first.shape;
    let item_bytes = shape
        .iter()
        .try_fold(width.bytes(), |size, dim| size.checked_mul(*dim))
        .ok_or(TensorDecodeError::SizeOverflow)?;
    let capacity = item_bytes
        .checked_mul(inputs.len())
        .ok_or(TensorDecodeError::SizeOverflow)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(capacity)
        .map_err(|_| TensorDecodeError::SizeOverflow)?;
    bytes.extend_from_slice(&first.bytes);
    let mut outcomes = Vec::new();
    outcomes
        .try_reserve_exact(inputs.len())
        .map_err(|_| TensorDecodeError::SizeOverflow)?;
    outcomes.push(first.outcome);

    for (index, input) in inputs.iter().enumerate().skip(1) {
        let image =
            decode_packed(*input, channels, width).map_err(|error| indexed(index, error))?;
        if image.shape != shape {
            return Err(TensorDecodeError::BatchShapeMismatch {
                index,
                expected: shape,
                actual: image.shape,
            });
        }
        bytes.extend_from_slice(&image.bytes);
        outcomes.push(image.outcome);
    }
    Ok(PackedBatch {
        bytes,
        shape,
        outcomes,
    })
}

fn indexed(index: usize, source: TensorDecodeError) -> TensorDecodeError {
    TensorDecodeError::BatchItem {
        index,
        source: Box::new(source),
    }
}

fn single_result<T>(tensor: T, outcome: DecodeOutcome<J2kDecodeWarning>) -> TensorDecode<T> {
    TensorDecode {
        tensor,
        decoded: outcome.decoded,
        warnings: outcome.warnings,
        route: TensorRoute::CpuStaged,
    }
}

fn batch_result<T>(tensor: T, packed: PackedBatch) -> TensorBatchDecode<T> {
    let (decoded, warnings) = packed
        .outcomes
        .into_iter()
        .map(|outcome| (outcome.decoded, outcome.warnings))
        .unzip();
    TensorBatchDecode {
        tensor,
        decoded,
        warnings,
        route: TensorRoute::CpuStaged,
    }
}
