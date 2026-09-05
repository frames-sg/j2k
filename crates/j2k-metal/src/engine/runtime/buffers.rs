// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::metal_types::{ComputePipelineState, Device};
use j2k_metal_support::{MetalPipelineLoader, MetalSupportError};

pub(in crate::engine) struct BufferKernels {
    pub(in crate::engine) zero_u32_buffer: ComputePipelineState,
    pub(in crate::engine) validate_bytes_equal: ComputePipelineState,
    pub(in crate::engine) copy_interleaved_padded: ComputePipelineState,
}

impl BufferKernels {
    pub(super) fn new(device: &Device) -> Result<Self, MetalSupportError> {
        let source = super::super::shader_source::buffers_shader_source();
        let loader = MetalPipelineLoader::new(device, &source)?;
        Ok(Self {
            zero_u32_buffer: loader.pipeline("j2k_zero_u32_buffer")?,
            validate_bytes_equal: loader.pipeline("j2k_validate_bytes_equal")?,
            copy_interleaved_padded: loader.pipeline("j2k_copy_interleaved_padded")?,
        })
    }
}
