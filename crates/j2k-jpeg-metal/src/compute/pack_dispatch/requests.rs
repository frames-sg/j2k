// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    batch, BatchDeviceBufferCache, CommandBufferRef, FastSubsampledMetal, JpegFast444PacketV1,
    MetalRuntime, PixelFormat, PlaneMode, Rect,
};

#[cfg(target_os = "macos")]
pub(in crate::compute) struct FastSubsampledScaledRegionBatchItemRequest<'a, P: FastSubsampledMetal>
{
    pub(in crate::compute) runtime: &'a MetalRuntime,
    pub(in crate::compute) command_buffer: &'a CommandBufferRef,
    pub(in crate::compute) device_buffer_cache: &'a mut BatchDeviceBufferCache,
    pub(in crate::compute) packet: &'a P,
    pub(in crate::compute) fmt: PixelFormat,
    pub(in crate::compute) roi: Rect,
    pub(in crate::compute) scale: j2k_core::Downscale,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) struct FastSubsampledOpBatchItemRequest<'a, P: FastSubsampledMetal> {
    pub(in crate::compute) runtime: &'a MetalRuntime,
    pub(in crate::compute) command_buffer: &'a CommandBufferRef,
    pub(in crate::compute) device_buffer_cache: &'a mut BatchDeviceBufferCache,
    pub(in crate::compute) packet: &'a P,
    pub(in crate::compute) fmt: PixelFormat,
    pub(in crate::compute) op: batch::BatchOp,
}

#[cfg(target_os = "macos")]
pub(in crate::compute) struct Fast444ScaledRegionBatchItemRequest<'a> {
    pub(in crate::compute) runtime: &'a MetalRuntime,
    pub(in crate::compute) command_buffer: &'a CommandBufferRef,
    pub(in crate::compute) device_buffer_cache: &'a mut BatchDeviceBufferCache,
    pub(in crate::compute) packet: &'a JpegFast444PacketV1,
    pub(in crate::compute) mode: PlaneMode,
    pub(in crate::compute) fmt: PixelFormat,
    pub(in crate::compute) roi: Rect,
    pub(in crate::compute) scale: j2k_core::Downscale,
}
