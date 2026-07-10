// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_command_queue, idct8_basis_table, size_of_val, system_default_device, Arc, Buffer,
    CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MetalPipelineLoader,
    MetalTranscodeError, METAL_DCT_RUNTIME_FAILED,
};

pub(super) fn shader_source() -> String {
    [
        r"
#include <metal_stdlib>
using namespace metal;
",
        j2k_codec_math::generated::DWT97_CONSTANTS_METAL,
        "\n",
        include_str!("../dct97.metal"),
    ]
    .concat()
}
pub(super) struct MetalRuntime {
    pub(super) device: Device,
    pub(super) queue: CommandQueue,
    pub(super) dct_project_band: ComputePipelineState,
    pub(super) dct_project_band_batch: ComputePipelineState,
    pub(super) dct97_idct_row_lift_batch: ComputePipelineState,
    pub(super) dct97_column_lift_batch: ComputePipelineState,
    pub(super) dct97_quantize_codeblocks_batch: ComputePipelineState,
    pub(super) reversible53_project_band: ComputePipelineState,
    pub(super) idct_basis: Buffer,
}

#[derive(Clone, Default)]
/// Reusable Metal session for transcode-stage accelerator dispatch.
pub struct MetalTranscodeSession {
    pub(super) device: Option<Device>,
    pub(super) runtime: Option<Arc<MetalRuntime>>,
}

impl MetalTranscodeSession {
    /// Create a transcode session bound to an existing Metal device.
    pub fn new(device: Device) -> Self {
        Self {
            device: Some(device),
            runtime: None,
        }
    }

    /// Create a transcode session bound to the system default Metal device.
    pub fn system_default() -> Result<Self, MetalTranscodeError> {
        system_default_device()
            .map(Self::new)
            .map_err(|_| MetalTranscodeError::MetalUnavailable)
    }

    pub(super) fn runtime(&mut self) -> Result<Arc<MetalRuntime>, MetalTranscodeError> {
        if let Some(runtime) = &self.runtime {
            return Ok(Arc::clone(runtime));
        }
        let runtime = Arc::new(match &self.device {
            Some(device) => MetalRuntime::new_with_device(device.clone())?,
            None => MetalRuntime::new()?,
        });
        self.runtime = Some(Arc::clone(&runtime));
        Ok(runtime)
    }

    pub(super) fn with_runtime<R>(
        &mut self,
        f: impl FnOnce(&MetalRuntime) -> Result<R, MetalTranscodeError>,
    ) -> Result<R, MetalTranscodeError> {
        let runtime = self.runtime()?;
        f(&runtime)
    }
}

impl core::fmt::Debug for MetalTranscodeSession {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalTranscodeSession")
            .field("device", &self.device.as_ref().map(|device| device.name()))
            .field("runtime_initialized", &self.runtime.is_some())
            .finish()
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct DctProjectionParams {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) block_cols: u32,
    pub(super) band_width: u32,
    pub(super) band_height: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct DctBatchProjectionParams {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) block_cols: u32,
    pub(super) blocks_per_item: u32,
    pub(super) band_width: u32,
    pub(super) band_height: u32,
    pub(super) output_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Dct97IdctRowLiftParams {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) block_cols: u32,
    pub(super) blocks_per_item: u32,
    pub(super) low_width: u32,
    pub(super) high_width: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Dct97ColumnLiftParams {
    pub(super) height: u32,
    pub(super) low_width: u32,
    pub(super) high_width: u32,
    pub(super) low_height: u32,
    pub(super) high_height: u32,
    pub(super) row_low_stride: u32,
    pub(super) row_high_stride: u32,
    pub(super) ll_stride: u32,
    pub(super) hl_stride: u32,
    pub(super) lh_stride: u32,
    pub(super) hh_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Dct97QuantizeCodeblocksParams {
    pub(super) band_width: u32,
    pub(super) band_height: u32,
    pub(super) output_stride: u32,
    pub(super) code_block_width: u32,
    pub(super) code_block_height: u32,
    pub(super) inv_delta: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Reversible53ProjectionParams {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) block_cols: u32,
    pub(super) blocks_per_item: u32,
    pub(super) band_width: u32,
    pub(super) band_height: u32,
    pub(super) output_stride: u32,
    pub(super) vertical_low: u32,
    pub(super) horizontal_low: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct MetalSparseRow {
    pub(super) offset: u32,
    pub(super) count: u32,
}

#[expect(
    unsafe_code,
    reason = "the repr(C) sparse-row record is an audited plain-data Metal ABI type"
)]
// SAFETY: Metal ABI structs are repr(C) plain data matching shader layouts.
unsafe impl j2k_core::accelerator::GpuAbi for MetalSparseRow {
    const NAME: &'static str = "MetalSparseRow";
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct MetalWeightTap {
    pub(super) sample_idx: u32,
    pub(super) weight: f32,
}

#[expect(
    unsafe_code,
    reason = "the repr(C) weight-tap record is an audited plain-data Metal ABI type"
)]
// SAFETY: Metal ABI structs are repr(C) plain data matching shader layouts.
unsafe impl j2k_core::accelerator::GpuAbi for MetalWeightTap {
    const NAME: &'static str = "MetalWeightTap";
}

pub(super) struct MetalSparseRows {
    pub(super) rows: Vec<MetalSparseRow>,
    pub(super) taps: Vec<MetalWeightTap>,
}

impl MetalRuntime {
    fn new() -> Result<Self, MetalTranscodeError> {
        let device = system_default_device().map_err(|_| MetalTranscodeError::MetalUnavailable)?;
        Self::new_with_device(device)
    }

    fn new_with_device(device: Device) -> Result<Self, MetalTranscodeError> {
        let shader_source = shader_source();
        let loader = MetalPipelineLoader::new(&device, &shader_source)
            .map_err(|_| MetalTranscodeError::Runtime(METAL_DCT_RUNTIME_FAILED))?;
        let pipeline = |name| {
            loader
                .pipeline(name)
                .map_err(|_| MetalTranscodeError::Runtime(METAL_DCT_RUNTIME_FAILED))
        };
        let dct_project_band = pipeline("dct97_project_band")?;
        let dct_project_band_batch = pipeline("dct97_project_band_batch")?;
        let dct97_idct_row_lift_batch = pipeline("dct97_idct_row_lift_batch")?;
        let dct97_column_lift_batch = pipeline("dct97_column_lift_batch")?;
        let dct97_quantize_codeblocks_batch = pipeline("dct97_quantize_codeblocks_batch")?;
        let reversible53_project_band = pipeline("reversible53_project_band")?;
        let queue = checked_command_queue(&device)
            .map_err(|_| MetalTranscodeError::Runtime(METAL_DCT_RUNTIME_FAILED))?;
        let idct_basis_data = idct8_basis_table();
        let idct_basis = device.new_buffer_with_data(
            idct_basis_data.as_ptr().cast(),
            size_of_val(&idct_basis_data) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            device,
            queue,
            dct_project_band,
            dct_project_band_batch,
            dct97_idct_row_lift_batch,
            dct97_column_lift_batch,
            dct97_quantize_codeblocks_batch,
            reversible53_project_band,
            idct_basis,
        })
    }
}
