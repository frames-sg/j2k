// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{cell::RefCell, sync::Arc};

use j2k_metal_support::{
    checked_command_queue, checked_shared_buffer, checked_shared_buffer_with_slice,
    commit_and_wait, wait_for_completion, MetalPipelineLoader, MetalSupportError,
};
use j2k_native::{
    ht_uvlc_encode_table, ht_uvlc_table0, ht_uvlc_table1, ht_vlc_encode_table0,
    ht_vlc_encode_table1, ht_vlc_table0, ht_vlc_table1,
};
use metal::{
    foreign_types::ForeignType, Buffer, CommandBufferRef, CommandQueue, ComputePipelineState,
    Device,
};

use crate::{
    buffer_pool::{MetalBufferPools, PooledBuffer},
    error::{metal_kernel_support_error, metal_runtime_support_error},
    Error,
};

use super::{abi::J2kHtUvlcEncodeTableEntry, shader_source::shader_source};

#[cfg(test)]
use j2k_metal_support::system_default_device;

thread_local! {
    static DEFAULT_METAL_SESSION: RefCell<Option<Result<crate::MetalBackendSession, MetalSupportError>>> = const { RefCell::new(None) };
    static METAL_RUNTIME_OVERRIDE: RefCell<Option<Arc<MetalRuntime>>> = const { RefCell::new(None) };
}

pub(crate) struct MetalRuntime {
    pub(super) device: Device,
    pub(super) queue: CommandQueue,
    pub(super) zero_u32_buffer: ComputePipelineState,
    pub(super) validate_bytes_equal: ComputePipelineState,
    pub(super) copy_interleaved_padded: ComputePipelineState,
    pub(super) lossless_deinterleave_to_planes: ComputePipelineState,
    pub(super) lossless_deinterleave_rct_rgb8_to_planes: ComputePipelineState,
    pub(super) lossless_extract_coefficients: ComputePipelineState,
    pub(super) pack_gray8: ComputePipelineState,
    pub(super) pack_rgb8: ComputePipelineState,
    pub(super) pack_mct_rgb8: ComputePipelineState,
    pub(super) pack_mct_rgb8_batched: ComputePipelineState,
    pub(super) pack_rgb_opaque_rgba8: ComputePipelineState,
    pub(super) pack_rgba8: ComputePipelineState,
    pub(super) pack_gray16: ComputePipelineState,
    pub(super) pack_rgb16: ComputePipelineState,
    pub(super) pack_u8_repeated_gray: ComputePipelineState,
    pub(super) pack_u16_repeated_gray: ComputePipelineState,
    pub(super) classic_cleanup_plain_batched: ComputePipelineState,
    pub(super) classic_cleanup_batched: ComputePipelineState,
    pub(super) classic_cleanup_plain_repeated_batched: ComputePipelineState,
    pub(super) classic_cleanup_plain_dev_repeated_batched: ComputePipelineState,
    pub(super) classic_cleanup_repeated_batched: ComputePipelineState,
    pub(super) classic_store_repeated_batched: ComputePipelineState,
    pub(super) idwt_interleave: ComputePipelineState,
    pub(super) idwt_reversible53_horizontal: ComputePipelineState,
    pub(super) idwt_reversible53_vertical: ComputePipelineState,
    pub(super) idwt_interleave_batched: ComputePipelineState,
    pub(super) idwt_reversible53_horizontal_batched: ComputePipelineState,
    pub(super) idwt_reversible53_vertical_batched: ComputePipelineState,
    pub(super) idwt_irreversible97_single_decomposition: ComputePipelineState,
    pub(super) fdwt53_horizontal: ComputePipelineState,
    pub(super) fdwt53_vertical: ComputePipelineState,
    pub(super) fdwt53_horizontal_batched: ComputePipelineState,
    pub(super) fdwt53_vertical_batched: ComputePipelineState,
    pub(super) fdwt97_lift_horizontal: ComputePipelineState,
    pub(super) fdwt97_lift_vertical: ComputePipelineState,
    pub(super) fdwt97_deinterleave_horizontal: ComputePipelineState,
    pub(super) fdwt97_deinterleave_vertical: ComputePipelineState,
    pub(super) inverse_mct: ComputePipelineState,
    pub(super) forward_rct: ComputePipelineState,
    pub(super) forward_ict: ComputePipelineState,
    pub(super) quantize_subband: ComputePipelineState,
    pub(super) store_component: ComputePipelineState,
    pub(super) store_component_repeated: ComputePipelineState,
    pub(super) store_component_repeated_gray_u8: ComputePipelineState,
    pub(super) store_component_repeated_gray_u16: ComputePipelineState,
    pub(super) store_component_repeated_gray_u8_contiguous: ComputePipelineState,
    pub(super) store_component_repeated_gray_u16_contiguous: ComputePipelineState,
    pub(super) store_component_gray_u8: ComputePipelineState,
    pub(super) store_component_gray_u16: ComputePipelineState,
    pub(super) ht_cleanup: ComputePipelineState,
    pub(super) ht_cleanup_batched: ComputePipelineState,
    pub(super) ht_cleanup_repeated_batched: ComputePipelineState,
    pub(super) classic_encode_code_block: ComputePipelineState,
    pub(super) classic_encode_code_blocks: ComputePipelineState,
    pub(super) classic_encode_code_blocks_32: ComputePipelineState,
    pub(super) classic_encode_code_blocks_bypass_32: ComputePipelineState,
    pub(super) classic_encode_code_blocks_bypass_u16_32: ComputePipelineState,
    pub(super) classic_tier1_density_bypass_u16_32: ComputePipelineState,
    pub(super) classic_tier1_raw_pack_bypass_u16_32: ComputePipelineState,
    pub(super) classic_tier1_arithmetic_pack_bypass_u16_32: ComputePipelineState,
    pub(super) classic_tier1_symbol_plan_bypass_u16_32: ComputePipelineState,
    pub(super) classic_tier1_pass_plan_bypass_u16_32: ComputePipelineState,
    pub(super) classic_tier1_token_emit_bypass_u16_32: ComputePipelineState,
    pub(super) classic_tier1_split_token_emit_bypass_u16_32: ComputePipelineState,
    pub(super) classic_tier1_split_mq_byte_token_emit_bypass_u16_32: ComputePipelineState,
    pub(super) classic_tier1_token_pack_bypass_u16_32: ComputePipelineState,
    pub(super) classic_tier1_split_token_pack_bypass_u16_32: ComputePipelineState,
    pub(super) classic_encode_code_blocks_style0: ComputePipelineState,
    pub(super) classic_encode_code_blocks_style0_32: ComputePipelineState,
    pub(super) ht_encode_code_block: ComputePipelineState,
    pub(super) ht_encode_code_blocks: ComputePipelineState,
    pub(super) packet_block_prepare_resident_classic: ComputePipelineState,
    pub(super) packet_block_prepare_resident_ht: ComputePipelineState,
    pub(super) packet_encode: ComputePipelineState,
    pub(super) packet_encode_batched: ComputePipelineState,
    pub(super) packet_encode_resident_classic_batched: ComputePipelineState,
    pub(super) packet_payload_copy_batched: ComputePipelineState,
    pub(super) lossless_codestream_assemble: ComputePipelineState,
    pub(super) lossless_codestream_assemble_batched: ComputePipelineState,
    pub(super) ht_vlc_table0: Buffer,
    pub(super) ht_vlc_table1: Buffer,
    pub(super) ht_uvlc_table0: Buffer,
    pub(super) ht_uvlc_table1: Buffer,
    pub(super) ht_vlc_encode_table0: Buffer,
    pub(super) ht_vlc_encode_table1: Buffer,
    pub(super) ht_uvlc_encode_table: Buffer,
    pub(super) tier1_dummy_buffer: Buffer,
    pub(super) buffer_pools: MetalBufferPools,
}

impl MetalRuntime {
    #[cfg(test)]
    pub(crate) fn new() -> Result<Self, MetalSupportError> {
        let device = system_default_device()?;
        Self::new_with_device(&device)
    }

    #[expect(
        clippy::too_many_lines,
        reason = "pipeline inventory construction mirrors the fixed Metal runtime ABI"
    )]
    pub(crate) fn new_with_device(device: &Device) -> Result<Self, MetalSupportError> {
        let shader_source = shader_source();
        let loader = MetalPipelineLoader::new(device, &shader_source)?;
        let pipeline = |name: &str| loader.pipeline(name);
        let queue = checked_command_queue(device)?;
        let ht_uvlc_encode_rows = (*ht_uvlc_encode_table()).map(J2kHtUvlcEncodeTableEntry::from);
        Ok(Self {
            device: device.clone(),
            queue,
            zero_u32_buffer: pipeline("j2k_zero_u32_buffer")?,
            validate_bytes_equal: pipeline("j2k_validate_bytes_equal")?,
            copy_interleaved_padded: pipeline("j2k_copy_interleaved_padded")?,
            lossless_deinterleave_to_planes: pipeline("j2k_lossless_deinterleave_to_planes")?,
            lossless_deinterleave_rct_rgb8_to_planes: pipeline(
                "j2k_lossless_deinterleave_rct_rgb8_to_planes",
            )?,
            lossless_extract_coefficients: pipeline("j2k_lossless_extract_coefficients")?,
            pack_gray8: pipeline("j2k_pack_gray8")?,
            pack_rgb8: pipeline("j2k_pack_rgb8")?,
            pack_mct_rgb8: pipeline("j2k_pack_mct_rgb8")?,
            pack_mct_rgb8_batched: pipeline("j2k_pack_mct_rgb8_batched")?,
            pack_rgb_opaque_rgba8: pipeline("j2k_pack_rgb_opaque_rgba8")?,
            pack_rgba8: pipeline("j2k_pack_rgba8")?,
            pack_gray16: pipeline("j2k_pack_gray16")?,
            pack_rgb16: pipeline("j2k_pack_rgb16")?,
            pack_u8_repeated_gray: pipeline("j2k_pack_u8_repeated_gray")?,
            pack_u16_repeated_gray: pipeline("j2k_pack_u16_repeated_gray")?,
            classic_cleanup_plain_batched: pipeline("j2k_decode_classic_cleanup_plain_batched")?,
            classic_cleanup_batched: pipeline("j2k_decode_classic_cleanup_batched")?,
            classic_cleanup_plain_repeated_batched: pipeline(
                "j2k_decode_classic_cleanup_plain_repeated_batched",
            )?,
            classic_cleanup_plain_dev_repeated_batched: pipeline(
                "j2k_decode_classic_cleanup_plain_dev_repeated_batched",
            )?,
            classic_cleanup_repeated_batched: pipeline(
                "j2k_decode_classic_cleanup_repeated_batched",
            )?,
            classic_store_repeated_batched: pipeline("j2k_store_classic_repeated_batched")?,
            idwt_interleave: pipeline("j2k_idwt_interleave")?,
            idwt_reversible53_horizontal: pipeline("j2k_idwt_reversible53_horizontal_pass")?,
            idwt_reversible53_vertical: pipeline("j2k_idwt_reversible53_vertical_pass")?,
            idwt_interleave_batched: pipeline("j2k_idwt_interleave_batched")?,
            idwt_reversible53_horizontal_batched: pipeline(
                "j2k_idwt_reversible53_horizontal_pass_batched",
            )?,
            idwt_reversible53_vertical_batched: pipeline(
                "j2k_idwt_reversible53_vertical_pass_batched",
            )?,
            idwt_irreversible97_single_decomposition: pipeline(
                "j2k_idwt_irreversible97_single_decomposition",
            )?,
            fdwt53_horizontal: pipeline("j2k_forward_dwt53_horizontal")?,
            fdwt53_vertical: pipeline("j2k_forward_dwt53_vertical")?,
            fdwt53_horizontal_batched: pipeline("j2k_forward_dwt53_horizontal_batched")?,
            fdwt53_vertical_batched: pipeline("j2k_forward_dwt53_vertical_batched")?,
            fdwt97_lift_horizontal: pipeline("j2k_forward_dwt97_lift_horizontal")?,
            fdwt97_lift_vertical: pipeline("j2k_forward_dwt97_lift_vertical")?,
            fdwt97_deinterleave_horizontal: pipeline("j2k_forward_dwt97_deinterleave_horizontal")?,
            fdwt97_deinterleave_vertical: pipeline("j2k_forward_dwt97_deinterleave_vertical")?,
            inverse_mct: pipeline("j2k_inverse_mct")?,
            forward_rct: pipeline("j2k_forward_rct")?,
            forward_ict: pipeline("j2k_forward_ict")?,
            quantize_subband: pipeline("j2k_quantize_subband")?,
            store_component: pipeline("j2k_store_component")?,
            store_component_repeated: pipeline("j2k_store_component_repeated")?,
            store_component_repeated_gray_u8: pipeline("j2k_store_component_repeated_gray_u8")?,
            store_component_repeated_gray_u16: pipeline("j2k_store_component_repeated_gray_u16")?,
            store_component_repeated_gray_u8_contiguous: pipeline(
                "j2k_store_component_repeated_gray_u8_contiguous",
            )?,
            store_component_repeated_gray_u16_contiguous: pipeline(
                "j2k_store_component_repeated_gray_u16_contiguous",
            )?,
            store_component_gray_u8: pipeline("j2k_store_component_gray_u8")?,
            store_component_gray_u16: pipeline("j2k_store_component_gray_u16")?,
            ht_cleanup: pipeline("j2k_decode_ht_cleanup")?,
            ht_cleanup_batched: pipeline("j2k_decode_ht_cleanup_batched")?,
            ht_cleanup_repeated_batched: pipeline("j2k_decode_ht_cleanup_repeated_batched")?,
            classic_encode_code_block: pipeline("j2k_encode_classic_code_block")?,
            classic_encode_code_blocks: pipeline("j2k_encode_classic_code_blocks")?,
            classic_encode_code_blocks_32: pipeline("j2k_encode_classic_code_blocks_32")?,
            classic_encode_code_blocks_bypass_32: pipeline(
                "j2k_encode_classic_code_blocks_bypass_32",
            )?,
            classic_encode_code_blocks_bypass_u16_32: pipeline(
                "j2k_encode_classic_code_blocks_bypass_u16_32",
            )?,
            classic_tier1_density_bypass_u16_32: pipeline(
                "j2k_profile_classic_tier1_density_bypass_u16_32",
            )?,
            classic_tier1_raw_pack_bypass_u16_32: pipeline(
                "j2k_profile_classic_tier1_raw_pack_bypass_u16_32",
            )?,
            classic_tier1_arithmetic_pack_bypass_u16_32: pipeline(
                "j2k_profile_classic_tier1_arithmetic_pack_bypass_u16_32",
            )?,
            classic_tier1_symbol_plan_bypass_u16_32: pipeline(
                "j2k_plan_classic_tier1_symbols_bypass_u16_32",
            )?,
            classic_tier1_pass_plan_bypass_u16_32: pipeline(
                "j2k_plan_classic_tier1_passes_bypass_u16_32",
            )?,
            classic_tier1_token_emit_bypass_u16_32: pipeline(
                "j2k_emit_classic_tier1_tokens_bypass_u16_32",
            )?,
            classic_tier1_split_token_emit_bypass_u16_32: pipeline(
                "j2k_emit_classic_tier1_split_tokens_bypass_u16_32",
            )?,
            classic_tier1_split_mq_byte_token_emit_bypass_u16_32: pipeline(
                "j2k_emit_classic_tier1_split_mq_byte_raw_tokens_bypass_u16_32",
            )?,
            classic_tier1_token_pack_bypass_u16_32: pipeline(
                "j2k_pack_classic_tier1_tokens_bypass_u16_32",
            )?,
            classic_tier1_split_token_pack_bypass_u16_32: pipeline(
                "j2k_pack_classic_tier1_split_tokens_bypass_u16_32",
            )?,
            classic_encode_code_blocks_style0: pipeline("j2k_encode_classic_code_blocks_style0")?,
            classic_encode_code_blocks_style0_32: pipeline(
                "j2k_encode_classic_code_blocks_style0_32",
            )?,
            ht_encode_code_block: pipeline("j2k_encode_ht_code_block")?,
            ht_encode_code_blocks: pipeline("j2k_encode_ht_code_blocks")?,
            packet_block_prepare_resident_classic: pipeline(
                "j2k_prepare_packet_blocks_from_classic_status",
            )?,
            packet_block_prepare_resident_ht: pipeline("j2k_prepare_packet_blocks_from_ht_status")?,
            packet_encode: pipeline("j2k_encode_packetization")?,
            packet_encode_batched: pipeline("j2k_encode_packetization_batched")?,
            packet_encode_resident_classic_batched: pipeline(
                "j2k_encode_packetization_resident_classic_batched",
            )?,
            packet_payload_copy_batched: pipeline("j2k_copy_packet_payload_batched")?,
            lossless_codestream_assemble: pipeline("j2k_assemble_lossless_classic_codestream")?,
            lossless_codestream_assemble_batched: pipeline(
                "j2k_assemble_lossless_codestream_batched",
            )?,
            ht_vlc_table0: checked_shared_buffer_with_slice(device, ht_vlc_table0())?,
            ht_vlc_table1: checked_shared_buffer_with_slice(device, ht_vlc_table1())?,
            ht_uvlc_table0: checked_shared_buffer_with_slice(device, ht_uvlc_table0())?,
            ht_uvlc_table1: checked_shared_buffer_with_slice(device, ht_uvlc_table1())?,
            ht_vlc_encode_table0: checked_shared_buffer_with_slice(device, ht_vlc_encode_table0())?,
            ht_vlc_encode_table1: checked_shared_buffer_with_slice(device, ht_vlc_encode_table1())?,
            ht_uvlc_encode_table: checked_shared_buffer_with_slice(device, &ht_uvlc_encode_rows)?,
            tier1_dummy_buffer: checked_shared_buffer(device, 1)?,
            buffer_pools: MetalBufferPools::new(device),
        })
    }

    pub(crate) fn command_queue(&self) -> &metal::CommandQueueRef {
        self.queue.as_ref()
    }

    pub(super) fn take_private_buffer(&self, bytes: usize) -> Result<PooledBuffer, Error> {
        self.buffer_pools.take_private(&self.device, bytes)
    }

    pub(super) fn recycle_private_buffer(&self, buffer: PooledBuffer) -> Result<(), Error> {
        self.buffer_pools.recycle_private(buffer)
    }

    pub(super) fn take_shared_buffer(&self, bytes: usize) -> Result<PooledBuffer, Error> {
        self.buffer_pools.take_shared(&self.device, bytes)
    }

    pub(super) fn recycle_shared_buffer(&self, buffer: PooledBuffer) -> Result<(), Error> {
        self.buffer_pools.recycle_shared(buffer)
    }

    pub(crate) fn buffer_pool_diagnostics(
        &self,
    ) -> Result<crate::MetalBufferPoolsDiagnostics, Error> {
        self.buffer_pools.diagnostics()
    }
}

pub(super) fn with_runtime<R>(
    f: impl FnOnce(&MetalRuntime) -> Result<R, Error>,
) -> Result<R, Error> {
    let override_runtime = METAL_RUNTIME_OVERRIDE.with(|slot| slot.borrow().clone());
    if let Some(runtime) = override_runtime {
        return f(&runtime);
    }

    DEFAULT_METAL_SESSION.with(|session| {
        let mut session = session.borrow_mut();
        if session.is_none() {
            *session = Some(
                j2k_metal_support::system_default_device().map(crate::MetalBackendSession::new),
            );
        }
        let Some(session) = session.as_ref() else {
            return Err(Error::MetalRuntime {
                message: "J2K Metal default session was not initialized".to_string(),
            });
        };
        match session {
            Ok(session) => with_runtime_for_session(session, f),
            Err(error) => Err(runtime_initialization_error(error)),
        }
    })
}

pub(crate) fn runtime_initialization_error(error: &MetalSupportError) -> Error {
    metal_runtime_support_error(error)
}

pub(super) fn commit_and_wait_metal(command_buffer: &CommandBufferRef) -> Result<(), Error> {
    commit_and_wait(command_buffer)
        .map_err(|error| metal_kernel_support_error(error.to_string(), error))
}

pub(super) fn wait_for_completion_metal(command_buffer: &CommandBufferRef) -> Result<(), Error> {
    wait_for_completion(command_buffer)
        .map_err(|error| metal_kernel_support_error(error.to_string(), error))
}

struct RuntimeOverrideGuard {
    previous: Option<Arc<MetalRuntime>>,
}

impl Drop for RuntimeOverrideGuard {
    fn drop(&mut self) {
        let previous = self.previous.take();
        METAL_RUNTIME_OVERRIDE.with(|slot| {
            slot.replace(previous);
        });
    }
}

pub(crate) fn with_runtime_for_session<R>(
    session: &crate::MetalBackendSession,
    f: impl FnOnce(&MetalRuntime) -> Result<R, Error>,
) -> Result<R, Error> {
    let runtime = session.runtime()?;
    let previous = METAL_RUNTIME_OVERRIDE.with(|slot| slot.replace(Some(runtime.clone())));
    let _guard = RuntimeOverrideGuard { previous };
    f(&runtime)
}

pub(super) fn with_runtime_for_device<R>(
    device: &Device,
    f: impl FnOnce(&MetalRuntime) -> Result<R, Error>,
) -> Result<R, Error> {
    let override_runtime = METAL_RUNTIME_OVERRIDE.with(|slot| slot.borrow().clone());
    if let Some(runtime) = override_runtime {
        if runtime.device.as_ptr() == device.as_ptr() {
            return f(&runtime);
        }
    }

    let session = crate::MetalBackendSession::new(device.clone());
    with_runtime_for_session(&session, f)
}

#[cfg(test)]
pub(crate) fn with_isolated_runtime_for_device_for_test<R>(
    device: &Device,
    f: impl FnOnce() -> Result<R, Error>,
) -> Result<R, Error> {
    let runtime = Arc::new(
        MetalRuntime::new_with_device(device)
            .map_err(|error| runtime_initialization_error(&error))?,
    );
    let previous = METAL_RUNTIME_OVERRIDE.with(|slot| slot.replace(Some(runtime)));
    let _guard = RuntimeOverrideGuard { previous };
    f()
}
