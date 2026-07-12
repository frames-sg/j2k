// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    compute, duration_share, Instant, PlannedResidentLosslessBufferEncode,
    PreparedResidentLosslessBufferEncode,
};

pub(super) struct PreparedResidentLosslessBatchItem {
    pub(super) prepared: PreparedResidentLosslessBufferEncode,
    pub(super) prepare_duration: std::time::Duration,
}

pub(super) fn prepare_planned_resident_lossless_tiles_batch(
    planned: Vec<PlannedResidentLosslessBufferEncode>,
    session: &crate::MetalBackendSession,
) -> Result<Vec<PreparedResidentLosslessBatchItem>, crate::Error> {
    struct BatchPlanInfo {
        index: usize,
        coefficient_count: usize,
        bytes_per_sample: u8,
        code_blocks: Vec<compute::J2kLosslessDeviceCodeBlock>,
    }

    let started = Instant::now();
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal resident preparation plan");
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<
            super::ResidentLosslessBufferEncodeMetadata,
        >(planned.len()),
        crate::batch_allocation::BatchMetadataRequest::of::<BatchPlanInfo>(planned.len()),
        crate::batch_allocation::BatchMetadataRequest::of::<
            compute::J2kLosslessDeviceBatchPrepareItem<'_>,
        >(planned.len()),
        crate::batch_allocation::BatchMetadataRequest::of::<PreparedResidentLosslessBatchItem>(
            planned.len(),
        ),
    ])?;
    let mut metadatas = budget.try_vec(planned.len(), "J2K Metal resident preparation metadata")?;
    let mut plan_infos =
        budget.try_vec(planned.len(), "J2K Metal resident preparation plan items")?;
    for mut planned in planned {
        #[cfg(test)]
        if planned.failure_injection_index == Some(planned.index) {
            return Err(crate::Error::MetalKernel {
                message: format!(
                    "injected J2K Metal resident encode failure at tile {}",
                    planned.index
                ),
            });
        }

        plan_infos.push(BatchPlanInfo {
            index: planned.index,
            coefficient_count: planned.coefficient_count,
            bytes_per_sample: planned.bytes_per_sample,
            code_blocks: planned.metadata.plan.take_code_blocks(),
        });
        metadatas.push(planned.metadata);
    }

    let mut batch_items = budget.try_vec(
        metadatas.len(),
        "J2K Metal resident device preparation items",
    )?;
    for (metadata, plan_info) in metadatas.iter().zip(plan_infos) {
        let tile = metadata.tile.as_tile();
        batch_items.push(compute::J2kLosslessDeviceBatchPrepareItem {
            tile_index: plan_info.index,
            job: compute::J2kLosslessDevicePrepareJob {
                input: tile.buffer,
                input_byte_offset: tile.byte_offset,
                input_width: tile.width,
                input_height: tile.height,
                input_pitch_bytes: tile.pitch_bytes,
                output_width: tile.output_width,
                output_height: tile.output_height,
                component_count: metadata.components,
                bytes_per_sample: plan_info.bytes_per_sample,
                bit_depth: metadata.bit_depth,
                num_decomposition_levels: metadata.plan.num_decomposition_levels,
                coefficient_count: plan_info.coefficient_count,
            },
            code_blocks: plan_info.code_blocks,
        });
    }

    let prepared = compute::prepare_lossless_device_code_blocks_batch(session, batch_items)?;
    let prepare_duration = duration_share(started.elapsed(), prepared.len());
    let mut results = budget.try_vec(metadatas.len(), "J2K Metal prepared resident batch items")?;
    for (metadata, prepared) in metadatas.into_iter().zip(prepared) {
        results.push(PreparedResidentLosslessBatchItem {
            prepared: PreparedResidentLosslessBufferEncode { metadata, prepared },
            prepare_duration,
        });
    }
    Ok(results)
}
