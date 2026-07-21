// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_cuda_runtime::{
    CudaJ2kMlKernelConfig, CudaJ2kMlLayout, CudaJ2kMlNormalization, CudaJ2kMlSample,
};

use super::PlannedImage;
use crate::cpu::SampleWidth;
use crate::{FloatNormalization, TensorDecodeError, TensorDecodeOptions, TensorLayout};

pub(super) fn kernel_config(
    plan: &PlannedImage<'_>,
    options: &TensorDecodeOptions,
    width: SampleWidth,
    integer_output: bool,
    index: usize,
    item_elements: usize,
) -> Result<CudaJ2kMlKernelConfig, TensorDecodeError> {
    let sample = match width {
        SampleWidth::U8 => CudaJ2kMlSample::U8,
        SampleWidth::U16 => CudaJ2kMlSample::U16,
    };
    let layout = match options.layout {
        TensorLayout::ChannelsFirst => CudaJ2kMlLayout::ChannelsFirst,
        TensorLayout::ChannelsLast => CudaJ2kMlLayout::ChannelsLast,
    };
    let normalization = if integer_output {
        CudaJ2kMlNormalization::Integer
    } else {
        match &options.normalization {
            FloatNormalization::Raw => CudaJ2kMlNormalization::Raw,
            FloatNormalization::Unit => CudaJ2kMlNormalization::Unit,
            FloatNormalization::MeanStd { mean, std } => {
                let mut means = [0.0; 4];
                let mut deviations = [1.0; 4];
                means[..plan.shape[2]].copy_from_slice(mean);
                deviations[..plan.shape[2]].copy_from_slice(std);
                CudaJ2kMlNormalization::MeanStd {
                    mean: means,
                    std: deviations,
                }
            }
        }
    };
    Ok(CudaJ2kMlKernelConfig {
        width: u32::try_from(plan.shape[1]).map_err(|_| TensorDecodeError::SizeOverflow)?,
        height: u32::try_from(plan.shape[0]).map_err(|_| TensorDecodeError::SizeOverflow)?,
        channels: u32::try_from(plan.shape[2]).map_err(|_| TensorDecodeError::SizeOverflow)?,
        sample,
        layout,
        destination_offset_elements: index
            .checked_mul(item_elements)
            .ok_or(TensorDecodeError::SizeOverflow)?,
        normalization,
    })
}

pub(super) fn tensor_shape_3(shape: [usize; 3], layout: TensorLayout) -> [usize; 3] {
    match layout {
        TensorLayout::ChannelsFirst => [shape[2], shape[0], shape[1]],
        TensorLayout::ChannelsLast => shape,
    }
}

pub(super) fn tensor_shape_4(batch: usize, shape: [usize; 3], layout: TensorLayout) -> [usize; 4] {
    match layout {
        TensorLayout::ChannelsFirst => [batch, shape[2], shape[0], shape[1]],
        TensorLayout::ChannelsLast => [batch, shape[0], shape[1], shape[2]],
    }
}
