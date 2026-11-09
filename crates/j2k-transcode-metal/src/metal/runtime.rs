// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_codec_math::transcode97::{
    DWT97_HIGH_SCALE_F32, DWT97_LIFT_STEPS_F32, DWT97_LOW_SCALE_F32, IDCT8_BASIS_F32,
};

use super::{
    checked_command_queue, idct8_basis_table, shared_buffer_with_slice, system_default_device, Arc,
    Buffer, CommandQueue, ComputePipelineState, Device, MetalPipelineLoader, MetalTranscodeError,
};
use objc2_metal::MTLDevice as _;

pub(super) fn shader_source() -> String {
    [
        r"
#include <metal_stdlib>
using namespace metal;
",
        &transcode97_constants_metal(),
        include_str!("../dct97.metal"),
    ]
    .concat()
}

/// MSL declarations of the canonical 9/7 transcode constants, emitted from
/// `j2k_codec_math::transcode97` so the shader never carries its own copy.
///
/// Each value is written as its shortest round-trip decimal with an `f`
/// suffix, which the Metal compiler rounds back to the identical binary32.
fn transcode97_constants_metal() -> String {
    let basis_rows = IDCT8_BASIS_F32
        .iter()
        .map(|row| format!("    {{{}}},", msl_float_list(row)))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "// Generated at runtime from j2k_codec_math::transcode97.\n\
         constant float TRANSCODE97_IDCT_BASIS[8][8] = {{\n{basis_rows}\n}};\n\
         constant float TRANSCODE97_LIFT_STEPS[4] = {{{}}};\n\
         constant float TRANSCODE97_LOW_SCALE = {};\n\
         constant float TRANSCODE97_HIGH_SCALE = {};\n",
        msl_float_list(&DWT97_LIFT_STEPS_F32),
        msl_float_literal(DWT97_LOW_SCALE_F32),
        msl_float_literal(DWT97_HIGH_SCALE_F32),
    )
}

fn msl_float_list(values: &[f32]) -> String {
    values
        .iter()
        .map(|&value| msl_float_literal(value))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Shortest round-trip decimal for a finite `f32`, as an MSL float literal.
fn msl_float_literal(value: f32) -> String {
    format!("{value:?}f")
}
pub(super) struct MetalRuntime {
    pub(super) device: Device,
    pub(super) queue: CommandQueue,
    pub(super) dct_project_band: ComputePipelineState,
    pub(super) dct97_idct_row_lift_batch: ComputePipelineState,
    pub(super) dct97_column_lift_batch: ComputePipelineState,
    pub(super) dct97_idct_plane_batch: ComputePipelineState,
    pub(super) dct97_lift_plane_step_batch: ComputePipelineState,
    pub(super) dct97_scale_plane_rows_batch: ComputePipelineState,
    pub(super) dct97_split_plane_batch: ComputePipelineState,
    pub(super) dct97_quantize_codeblocks_batch: ComputePipelineState,
    pub(super) reversible53_project_band: ComputePipelineState,
    /// IDCT basis consumed by the 5/3 float projection kernels; the 9/7
    /// kernels use the canonical `TRANSCODE97_IDCT_BASIS` constants instead.
    pub(super) idct_basis: Buffer,
}

#[expect(
    unsafe_code,
    reason = "objc2 Metal protocol objects require an audited cross-thread runtime boundary"
)]
#[expect(
    clippy::non_send_fields_in_send_ty,
    reason = "the immutable IDCT basis MTLBuffer is covered by the SAFETY comment's cross-thread contract"
)]
// SAFETY: Retained Metal devices, queues, pipelines, and buffers support
// cross-thread ownership. This runtime is immutable after construction and
// contains no unsynchronized host-side mutable cache.
unsafe impl Send for MetalRuntime {}
#[expect(
    unsafe_code,
    reason = "objc2 Metal protocol objects require an audited cross-thread runtime boundary"
)]
// SAFETY: Shared runtime references expose only immutable retained handles;
// Metal command ordering is established by independent command buffers on the
// retained queue.
unsafe impl Sync for MetalRuntime {}

#[derive(Clone, Default)]
/// Reusable Metal session for transcode-stage accelerator dispatch.
pub struct MetalTranscodeSession {
    pub(super) device: Option<Device>,
    pub(super) runtime: Option<Arc<MetalRuntime>>,
}

impl MetalTranscodeSession {
    /// Create a transcode session bound to an existing Metal device.
    pub fn new(
        device: objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>,
    ) -> Self {
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

/// Parameters shared by the global-memory canonical 9/7 plane kernels.
#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct Dct97PlaneParams {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) block_cols: u32,
    pub(super) blocks_per_item: u32,
    pub(super) low_width: u32,
    pub(super) high_width: u32,
    pub(super) low_height: u32,
    pub(super) high_height: u32,
    pub(super) ll_stride: u32,
    pub(super) hl_stride: u32,
    pub(super) lh_stride: u32,
    pub(super) hh_stride: u32,
    /// Nonzero for a column lifting step.
    pub(super) vertical: u32,
    /// Lifting step index `0..4`.
    pub(super) step: u32,
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

macro_rules! prove_gpu_parameter_layout {
    ($ty:ty, $offset:expr;) => {
        let _: [(); core::mem::size_of::<$ty>()] = [(); $offset];
    };
    (
        $ty:ty,
        $offset:expr;
        $field:ident: $field_ty:ty
        $(, $remaining_field:ident: $remaining_field_ty:ty)*
    ) => {
        let _: [(); core::mem::offset_of!($ty, $field)] = [(); $offset];
        prove_gpu_parameter_layout!(
            $ty,
            $offset + core::mem::size_of::<$field_ty>();
            $($remaining_field: $remaining_field_ty),*
        );
    };
}

macro_rules! impl_gpu_parameter_abi {
    ($(
        $ty:ty {
            $first_field:ident: $first_field_ty:ty
            $(, $field:ident: $field_ty:ty)*
            $(,)?
        }
    ),+ $(,)?) => {
        $(
            const _: () = {
                fn assert_field_types(value: &$ty) {
                    let _: &$first_field_ty = &value.$first_field;
                    $(let _: &$field_ty = &value.$field;)*
                }
                let _ = assert_field_types;

                prove_gpu_parameter_layout!(
                    $ty,
                    0;
                    $first_field: $first_field_ty
                    $(, $field: $field_ty)*
                );
            };

            #[expect(
                unsafe_code,
                reason = "the compile-time field walk proves a padding-free Metal ABI value"
            )]
            // SAFETY: The type is repr(C), Copy, contains only numeric scalar
            // fields for which every bit pattern is valid, and the compile-time
            // field walk proves that its complete representation has no padding.
            unsafe impl j2k_core::accelerator::GpuAbi for $ty {
                const NAME: &'static str = stringify!($ty);
            }
        )+
    };
}

impl_gpu_parameter_abi!(
    DctProjectionParams {
        width: u32,
        height: u32,
        block_cols: u32,
        band_width: u32,
        band_height: u32,
    },
    Dct97IdctRowLiftParams {
        width: u32,
        height: u32,
        block_cols: u32,
        blocks_per_item: u32,
        low_width: u32,
        high_width: u32,
    },
    Dct97ColumnLiftParams {
        height: u32,
        low_width: u32,
        high_width: u32,
        low_height: u32,
        high_height: u32,
        row_low_stride: u32,
        row_high_stride: u32,
        ll_stride: u32,
        hl_stride: u32,
        lh_stride: u32,
        hh_stride: u32,
    },
    Dct97PlaneParams {
        width: u32,
        height: u32,
        block_cols: u32,
        blocks_per_item: u32,
        low_width: u32,
        high_width: u32,
        low_height: u32,
        high_height: u32,
        ll_stride: u32,
        hl_stride: u32,
        lh_stride: u32,
        hh_stride: u32,
        vertical: u32,
        step: u32,
    },
    Dct97QuantizeCodeblocksParams {
        band_width: u32,
        band_height: u32,
        output_stride: u32,
        code_block_width: u32,
        code_block_height: u32,
        inv_delta: f32,
    },
    Reversible53ProjectionParams {
        width: u32,
        height: u32,
        block_cols: u32,
        blocks_per_item: u32,
        band_width: u32,
        band_height: u32,
        output_stride: u32,
        vertical_low: u32,
        horizontal_low: u32,
    },
);

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
        let loader = MetalPipelineLoader::new(&device, &shader_source).map_err(|error| {
            MetalTranscodeError::support("Metal transcode shader compilation", error)
        })?;
        let pipeline = |name| {
            loader
                .pipeline(name)
                .map_err(|error| MetalTranscodeError::support("Metal pipeline creation", error))
        };
        let dct_project_band = pipeline("dct97_project_band")?;
        let dct97_idct_row_lift_batch = pipeline("dct97_idct_row_lift_batch")?;
        let dct97_column_lift_batch = pipeline("dct97_column_lift_batch")?;
        let dct97_idct_plane_batch = pipeline("dct97_idct_plane_batch")?;
        let dct97_lift_plane_step_batch = pipeline("dct97_lift_plane_step_batch")?;
        let dct97_scale_plane_rows_batch = pipeline("dct97_scale_plane_rows_batch")?;
        let dct97_split_plane_batch = pipeline("dct97_split_plane_batch")?;
        let dct97_quantize_codeblocks_batch = pipeline("dct97_quantize_codeblocks_batch")?;
        let reversible53_project_band = pipeline("reversible53_project_band")?;
        let queue = checked_command_queue(&device).map_err(|error| {
            MetalTranscodeError::support("Metal transcode command queue creation", error)
        })?;
        let idct_basis_data = idct8_basis_table();
        let idct_basis =
            shared_buffer_with_slice(&device, &idct_basis_data, "Metal IDCT basis table")?;

        Ok(Self {
            device,
            queue,
            dct_project_band,
            dct97_idct_row_lift_batch,
            dct97_column_lift_batch,
            dct97_idct_plane_batch,
            dct97_lift_plane_step_batch,
            dct97_scale_plane_rows_batch,
            dct97_split_plane_batch,
            dct97_quantize_codeblocks_batch,
            reversible53_project_band,
            idct_basis,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        transcode97_constants_metal, DWT97_HIGH_SCALE_F32, DWT97_LIFT_STEPS_F32,
        DWT97_LOW_SCALE_F32, IDCT8_BASIS_F32,
    };

    /// Every literal in the emitted prelude, in declaration order.
    fn prelude_literals(source: &str) -> Vec<f32> {
        source
            .lines()
            .filter(|line| !line.trim_start().starts_with("//"))
            .flat_map(|line| line.split(['{', '}', ',', '=', ';']))
            .filter_map(|token| token.trim().strip_suffix('f'))
            .map(|literal| literal.parse::<f32>().expect("prelude literal parses"))
            .collect()
    }

    #[test]
    fn transcode97_prelude_literals_round_trip_to_the_codec_math_constants() {
        let literals = prelude_literals(&transcode97_constants_metal());
        let expected: Vec<f32> = IDCT8_BASIS_F32
            .iter()
            .flatten()
            .chain(&DWT97_LIFT_STEPS_F32)
            .chain([&DWT97_LOW_SCALE_F32, &DWT97_HIGH_SCALE_F32])
            .copied()
            .collect();
        assert_eq!(literals.len(), expected.len());
        for (index, (literal, constant)) in literals.iter().zip(&expected).enumerate() {
            assert_eq!(literal.to_bits(), constant.to_bits(), "literal {index}");
        }
    }
}
