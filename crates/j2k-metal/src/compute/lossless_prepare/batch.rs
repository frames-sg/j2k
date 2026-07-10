// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    lossless_prepare_sizes, metal_profile_coefficient_prep_split_commands_enabled,
    prepare_lossless_batch_item, take_recyclable_private_buffer, with_runtime_for_session,
    BatchPrepareItemRequest, Error, J2kLosslessDeviceBatchPrepareItem,
    J2kPreparedLosslessDeviceCodeBlocks,
};

#[cfg(target_os = "macos")]
pub(crate) fn prepare_lossless_device_code_blocks_batch(
    session: &crate::MetalBackendSession,
    items: Vec<J2kLosslessDeviceBatchPrepareItem<'_>>,
) -> Result<Vec<J2kPreparedLosslessDeviceCodeBlocks>, Error> {
    if items.is_empty() {
        return Ok(Vec::new());
    }

    let mut sizes = Vec::with_capacity(items.len());
    let mut coefficient_byte_offsets = Vec::with_capacity(items.len());
    let mut total_coefficient_bytes = 0usize;
    for item in &items {
        let item_sizes = lossless_prepare_sizes(item.job).map_err(|err| Error::MetalKernel {
            message: format!(
                "J2K Metal resident batch coefficient prep failed at tile {}: {err}",
                item.tile_index
            ),
        })?;
        coefficient_byte_offsets.push(total_coefficient_bytes);
        total_coefficient_bytes = total_coefficient_bytes
            .checked_add(item_sizes.coefficient_bytes)
            .ok_or_else(|| Error::MetalKernel {
                message: "J2K Metal resident batch coefficient size overflow".to_string(),
            })?;
        sizes.push(item_sizes);
    }

    with_runtime_for_session(session, |runtime| {
        let mut shared_recyclable_private_buffers = Vec::new();
        let coefficient_buffer = take_recyclable_private_buffer(
            runtime,
            total_coefficient_bytes.max(1),
            &mut shared_recyclable_private_buffers,
        )?;
        let split_prepare_command_buffers = metal_profile_coefficient_prep_split_commands_enabled();
        let shared_command_buffer = if split_prepare_command_buffers {
            None
        } else {
            Some(runtime.queue.new_command_buffer().to_owned())
        };
        let mut prepared = Vec::with_capacity(items.len());

        for ((item, item_sizes), coefficient_byte_offset) in
            items.into_iter().zip(sizes).zip(coefficient_byte_offsets)
        {
            prepared.push(prepare_lossless_batch_item(BatchPrepareItemRequest {
                runtime,
                item,
                item_sizes,
                coefficient_buffer: &coefficient_buffer,
                coefficient_byte_offset,
                split_prepare_command_buffers,
                shared_command_buffer: &shared_command_buffer,
                shared_recyclable_private_buffers: &mut shared_recyclable_private_buffers,
            })?);
        }

        if let Some(command_buffer) = shared_command_buffer {
            command_buffer.commit();
        }
        Ok(prepared)
    })
}
