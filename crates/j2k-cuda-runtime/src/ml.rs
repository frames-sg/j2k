// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    driver::CuDevicePtr, execution::cuda_kernel_param, kernels::copy_u8_launch_geometry,
    CudaContext, CudaError, CudaExternalDeviceBufferViewMut,
};

/// Integer sample representation read from a resident J2K surface.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaJ2kMlSample {
    /// Unsigned 8-bit sample.
    U8,
    /// Unsigned 16-bit sample.
    U16,
}

impl CudaJ2kMlSample {
    const fn byte_width(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
        }
    }

    const fn flag(self) -> u32 {
        match self {
            Self::U8 => 1,
            Self::U16 => 2,
        }
    }
}

/// Destination layout understood by the low-level j2k-ml CUDA kernel.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaJ2kMlLayout {
    /// Channels-first output.
    ChannelsFirst,
    /// Channels-last output.
    ChannelsLast,
}

/// Fused floating-point normalization understood by the CUDA kernel.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CudaJ2kMlNormalization {
    /// Integer output with no floating-point conversion.
    Integer,
    /// Cast integer samples to F32 without scaling.
    Raw,
    /// Scale into `0..=1`.
    Unit,
    /// Unit-scale and apply per-channel mean and standard deviation.
    MeanStd {
        /// Up to four channel means.
        mean: [f32; 4],
        /// Up to four channel standard deviations.
        std: [f32; 4],
    },
}

/// Validated launch configuration for one resident surface.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CudaJ2kMlKernelConfig {
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Channel count (1, 3, or 4).
    pub channels: u32,
    /// Source and integer-output sample representation.
    pub sample: CudaJ2kMlSample,
    /// Destination layout.
    pub layout: CudaJ2kMlLayout,
    /// Destination element offset for this batch item.
    pub destination_offset_elements: usize,
    /// Fused conversion and normalization.
    pub normalization: CudaJ2kMlNormalization,
}

impl CudaContext {
    /// Convert one resident interleaved J2K surface into externally owned CUDA memory.
    #[doc(hidden)]
    pub fn j2k_ml_convert_into_external(
        &self,
        source_ptr: u64,
        source_byte_len: usize,
        destination: &mut CudaExternalDeviceBufferViewMut<'_>,
        config: CudaJ2kMlKernelConfig,
    ) -> Result<(), CudaError> {
        if !self.is_same_context(destination.context()) {
            return Err(CudaError::InvalidArgument {
                message: "j2k-ml destination belongs to a different CUDA context".to_string(),
            });
        }
        if config.width == 0 || config.height == 0 {
            return Err(CudaError::InvalidArgument {
                message: "j2k-ml CUDA dimensions must be nonzero".to_string(),
            });
        }
        if !matches!(config.channels, 1 | 3 | 4) {
            return Err(CudaError::InvalidArgument {
                message: "j2k-ml CUDA channels must be 1, 3, or 4".to_string(),
            });
        }
        if source_ptr == 0 {
            return Err(CudaError::InvalidArgument {
                message: "j2k-ml CUDA source pointer must not be null".to_string(),
            });
        }
        let sample_count = usize::try_from(config.width)
            .ok()
            .and_then(|width| width.checked_mul(config.height as usize))
            .and_then(|pixels| pixels.checked_mul(config.channels as usize))
            .ok_or(CudaError::LengthTooLarge {
                len: source_byte_len,
            })?;
        let source_required = sample_count
            .checked_mul(config.sample.byte_width())
            .ok_or(CudaError::LengthTooLarge { len: sample_count })?;
        if source_required > source_byte_len {
            return Err(CudaError::OutputTooSmall {
                required: source_required,
                have: source_byte_len,
            });
        }
        if !source_ptr.is_multiple_of(config.sample.byte_width() as u64) {
            return Err(CudaError::InvalidArgument {
                message: "j2k-ml CUDA source pointer is misaligned".to_string(),
            });
        }
        self.inner.validate_pointer_context(source_ptr)?;

        let output_width = match config.normalization {
            CudaJ2kMlNormalization::Integer => config.sample.byte_width(),
            CudaJ2kMlNormalization::Raw
            | CudaJ2kMlNormalization::Unit
            | CudaJ2kMlNormalization::MeanStd { .. } => 4,
        };
        let destination_end = config
            .destination_offset_elements
            .checked_add(sample_count)
            .and_then(|elements| elements.checked_mul(output_width))
            .ok_or(CudaError::LengthTooLarge { len: sample_count })?;
        if destination_end > destination.byte_len() {
            return Err(CudaError::OutputTooSmall {
                required: destination_end,
                have: destination.byte_len(),
            });
        }
        validate_normalization(config.normalization, config.channels as usize)?;
        let geometry = copy_u8_launch_geometry(sample_count)
            .ok_or(CudaError::LengthTooLarge { len: sample_count })?;
        let function = self.inner.cuda_oxide_j2k_ml_kernel_function()?;

        let mut destination_ptr: CuDevicePtr = destination.device_ptr();
        let mut source_ptr: CuDevicePtr = source_ptr;
        let mut sample_count = sample_count as u64;
        let mut channels = config.channels;
        let mut source_sample = config.sample.flag();
        let mut output_kind = match config.normalization {
            CudaJ2kMlNormalization::Integer => config.sample.flag(),
            _ => 4,
        };
        let mut layout = match config.layout {
            CudaJ2kMlLayout::ChannelsFirst => 0u32,
            CudaJ2kMlLayout::ChannelsLast => 1u32,
        };
        let mut destination_offset = config.destination_offset_elements as u64;
        let (mut normalization, mean, std) = normalization_args(config.normalization);
        let [mut mean0, mut mean1, mut mean2, mut mean3] = mean;
        let [mut std0, mut std1, mut std2, mut std3] = std;
        let mut params = cuda_kernel_params!(
            destination_ptr,
            source_ptr,
            sample_count,
            channels,
            source_sample,
            output_kind,
            layout,
            destination_offset,
            normalization,
            mean0,
            mean1,
            mean2,
            mean3,
            std0,
            std1,
            std2,
            std3,
        );
        self.launch_kernel(function, geometry, &mut params)
    }
}

fn validate_normalization(
    normalization: CudaJ2kMlNormalization,
    channels: usize,
) -> Result<(), CudaError> {
    let CudaJ2kMlNormalization::MeanStd { mean, std } = normalization else {
        return Ok(());
    };
    if mean[..channels]
        .iter()
        .chain(&std[..channels])
        .any(|value| !value.is_finite())
    {
        return Err(CudaError::InvalidArgument {
            message: "j2k-ml CUDA normalization values must be finite".to_string(),
        });
    }
    if std[..channels].contains(&0.0) {
        return Err(CudaError::InvalidArgument {
            message: "j2k-ml CUDA standard deviations must be nonzero".to_string(),
        });
    }
    Ok(())
}

fn normalization_args(normalization: CudaJ2kMlNormalization) -> (u32, [f32; 4], [f32; 4]) {
    match normalization {
        CudaJ2kMlNormalization::Integer => (0, [0.0; 4], [1.0; 4]),
        CudaJ2kMlNormalization::Raw => (1, [0.0; 4], [1.0; 4]),
        CudaJ2kMlNormalization::Unit => (2, [0.0; 4], [1.0; 4]),
        CudaJ2kMlNormalization::MeanStd { mean, std } => (3, mean, std),
    }
}
