// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{PixelFormat, Rect};
use j2k_jpeg::adapter::JpegFast444PacketV1;

use crate::{batch, Error};

use super::{
    checked_u32, fast444_scaled_region_params, fast_subsampled_full_mcu_scaled_window,
    fast_subsampled_scaled_params, fast_subsampled_scaled_region_params,
    fast_subsampled_windowed_pack_params_for_dims, plane_mode_to_u32, FastRegionScaledMetal,
    FastSubsampledPacket, JpegFast444ScaledParams, JpegFastRegionScaledBatchParams,
    JpegWindowedPackBatchParams, JpegWindowedTexturePackBatchParams, PlaneMode, OUT_RGB,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct RegionScaledBatchPlan {
    pub(super) decode_params: JpegFastRegionScaledBatchParams,
    pub(super) pack_params: JpegWindowedPackBatchParams,
    pub(super) y_len: usize,
    pub(super) chroma_len: usize,
    pub(super) out_tile_len: usize,
    pub(super) out_dims: (u32, u32),
}

pub(super) fn windowed_texture_pack_params(
    plan: RegionScaledBatchPlan,
) -> JpegWindowedTexturePackBatchParams {
    JpegWindowedTexturePackBatchParams {
        src_width: plan.pack_params.src_width,
        src_height: plan.pack_params.src_height,
        chroma_width: plan.pack_params.chroma_width,
        chroma_height: plan.pack_params.chroma_height,
        src_x: plan.pack_params.src_x,
        src_y: plan.pack_params.src_y,
        width: plan.pack_params.width,
        height: plan.pack_params.height,
        tile_index: 0,
        alpha: u32::from(u8::MAX),
    }
}

fn fast_packets_share_batch_shape<P: FastSubsampledPacket>(
    first: &P,
    packet: &P,
    segment_count: usize,
    restart_packets_supported: bool,
) -> bool {
    (restart_packets_supported || packet.restart_interval_mcus() == 0)
        && packet.dimensions() == first.dimensions()
        && packet.mcus_per_row() == first.mcus_per_row()
        && packet.mcu_rows() == first.mcu_rows()
        && packet.entropy_checkpoints().len() == segment_count
        && packet.y_quant() == first.y_quant()
        && packet.cb_quant() == first.cb_quant()
        && packet.cr_quant() == first.cr_quant()
        && packet.y_dc_table() == first.y_dc_table()
        && packet.y_ac_table() == first.y_ac_table()
        && packet.cb_dc_table() == first.cb_dc_table()
        && packet.cb_ac_table() == first.cb_ac_table()
        && packet.cr_dc_table() == first.cr_dc_table()
        && packet.cr_ac_table() == first.cr_ac_table()
}

pub(super) fn fast_subsampled_packets_share_full_rgb_batch_shape<P: FastSubsampledPacket>(
    first: &P,
    packet: &P,
    segment_count: usize,
) -> bool {
    fast_packets_share_batch_shape(
        first,
        packet,
        segment_count,
        P::FULL_RGB_BATCH_SUPPORTS_RESTART,
    )
}

fn fast_full_rgb_batch_groups<P, K>(
    requests: &[batch::QueuedRequest],
    packets: &[(&P, K)],
    restart_packets_supported: bool,
) -> Result<Option<Vec<Vec<usize>>>, Error>
where
    P: FastSubsampledPacket,
    K: Copy + Eq,
{
    let mut budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal full RGB batch grouping",
        requests,
    )?;
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<Vec<usize>>(packets.len()),
        crate::batch_allocation::BatchMetadataRequest::of::<usize>(packets.len()),
    ])?;
    let mut groups: Vec<Vec<usize>> =
        budget.try_vec(packets.len(), "JPEG Metal full RGB batch groups")?;
    'packet: for (index, (packet, key)) in packets.iter().copied().enumerate() {
        if packet.entropy_bytes().is_empty() || packet.entropy_checkpoints().is_empty() {
            return Ok(None);
        }

        for group in &mut groups {
            let (first, first_key) = packets[group[0]];
            if key == first_key
                && fast_packets_share_batch_shape(
                    first,
                    packet,
                    first.entropy_checkpoints().len(),
                    restart_packets_supported,
                )
            {
                crate::batch_allocation::try_reserve_for_push(
                    group,
                    "JPEG Metal full RGB grouped indices",
                )?;
                group.push(index);
                continue 'packet;
            }
        }
        let mut indices = budget.try_vec(1, "JPEG Metal full RGB grouped indices")?;
        indices.push(index);
        groups.push(indices);
    }
    let mut actual = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal full RGB batch grouping",
        requests,
    )?;
    actual.account_capacity::<Vec<usize>>(groups.capacity())?;
    for group in &groups {
        actual.account_capacity::<usize>(group.capacity())?;
    }
    Ok(Some(groups))
}

pub(super) fn fast_subsampled_full_rgb_batch_groups<P: FastSubsampledPacket>(
    requests: &[batch::QueuedRequest],
    packets: &[&P],
) -> Result<Option<Vec<Vec<usize>>>, Error> {
    let mut budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal full RGB keyed packet plan",
        requests,
    )?;
    let mut keyed_packets =
        budget.try_vec(packets.len(), "JPEG Metal full RGB keyed packet references")?;
    keyed_packets.extend(packets.iter().copied().map(|packet| (packet, ())));
    fast_full_rgb_batch_groups(requests, &keyed_packets, P::FULL_RGB_BATCH_SUPPORTS_RESTART)
}

pub(super) fn fast_subsampled_region_scaled_batch_plan<P: FastSubsampledPacket>(
    packet: &P,
    roi: Rect,
    scale: j2k_core::Downscale,
    tile_count: u32,
    segment_count: u32,
    mode: PlaneMode,
) -> Option<RegionScaledBatchPlan> {
    let full_params = fast_subsampled_scaled_params(packet, scale)?;
    let scaled_roi = roi.scaled_covering(scale);
    let scaled_roi = j2k_jpeg::Rect {
        x: scaled_roi.x,
        y: scaled_roi.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let source_window = fast_subsampled_full_mcu_scaled_window::<P>(
        (full_params.scaled_width, full_params.scaled_height),
        scaled_roi,
        full_params.scale_shift,
    );
    let decode_params = fast_subsampled_scaled_region_params(packet, scale, source_window)?;
    let local_roi = j2k_jpeg::Rect {
        x: scaled_roi.x - source_window.x,
        y: scaled_roi.y - source_window.y,
        w: scaled_roi.w,
        h: scaled_roi.h,
    };
    let pack_params = fast_subsampled_windowed_pack_params_for_dims::<P>(
        (source_window.w, source_window.h),
        PixelFormat::Rgb8,
        local_roi,
    )
    .ok()?;
    let out_stride = scaled_roi.w as usize * PixelFormat::Rgb8.bytes_per_pixel();
    Some(RegionScaledBatchPlan {
        decode_params: JpegFastRegionScaledBatchParams {
            scaled_width: decode_params.scaled_width,
            scaled_height: decode_params.scaled_height,
            chroma_width: decode_params.chroma_width,
            chroma_height: decode_params.chroma_height,
            mcus_per_row: decode_params.mcus_per_row,
            mcu_rows: decode_params.mcu_rows,
            segment_count,
            tile_count,
            scale_shift: decode_params.scale_shift,
            origin_x: decode_params.origin_x,
            origin_y: decode_params.origin_y,
        },
        pack_params: JpegWindowedPackBatchParams {
            src_width: pack_params.src_width,
            src_height: pack_params.src_height,
            chroma_width: pack_params.chroma_width,
            chroma_height: pack_params.chroma_height,
            src_x: pack_params.src_x,
            src_y: pack_params.src_y,
            width: pack_params.width,
            height: pack_params.height,
            tile_count,
            out_stride: checked_u32(out_stride, P::REGION_SCALED_BATCH_OUT_STRIDE_CTX).ok()?,
            alpha: u32::from(u8::MAX),
            mode: plane_mode_to_u32(mode),
            out_format: OUT_RGB,
        },
        y_len: source_window.w as usize * source_window.h as usize,
        chroma_len: P::chroma_width(source_window.w) as usize
            * P::chroma_height(source_window.h) as usize,
        out_tile_len: out_stride * scaled_roi.h as usize,
        out_dims: (scaled_roi.w, scaled_roi.h),
    })
}

struct FastRegionScaledGroup {
    indices: Vec<usize>,
    scale: j2k_core::Downscale,
    plan: RegionScaledBatchPlan,
}

pub(super) fn fast_subsampled_region_scaled_batch_groups<P: FastRegionScaledMetal>(
    requests: &[batch::QueuedRequest],
    packets: &[(&P, PlaneMode)],
) -> Result<Option<Vec<Vec<usize>>>, Error> {
    let mut budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal region-scaled batch grouping",
        requests,
    )?;
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<FastRegionScaledGroup>(requests.len()),
        crate::batch_allocation::BatchMetadataRequest::of::<Vec<usize>>(requests.len()),
        crate::batch_allocation::BatchMetadataRequest::of::<usize>(requests.len()),
        crate::batch_allocation::BatchMetadataRequest::of::<Vec<usize>>(requests.len()),
    ])?;
    let mut groups: Vec<FastRegionScaledGroup> =
        budget.try_vec(requests.len(), "JPEG Metal region-scaled batch groups")?;
    'packet: for (index, (request, (packet, mode))) in
        requests.iter().zip(packets.iter().copied()).enumerate()
    {
        if packet.restart_interval_mcus() != 0
            || packet.entropy_bytes().is_empty()
            || packet.entropy_checkpoints().is_empty()
        {
            return Ok(None);
        }
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        let segment_count = packet.entropy_checkpoints().len();
        let segment_count_u32 = checked_u32(
            segment_count,
            &format!(
                "{} region scaled texture batch segment count",
                P::FAMILY_NAME
            ),
        )?;
        let Some(plan) = fast_subsampled_region_scaled_batch_plan(
            packet,
            roi,
            scale,
            1,
            segment_count_u32,
            mode,
        ) else {
            return Ok(None);
        };

        for group in &mut groups {
            let (first, first_mode) = packets[group.indices[0]];
            let first_segment_count = first.entropy_checkpoints().len();
            if mode == first_mode
                && scale == group.scale
                && plan == group.plan
                && fast_subsampled_packets_share_full_rgb_batch_shape(
                    first,
                    packet,
                    first_segment_count,
                )
            {
                crate::batch_allocation::try_reserve_for_push(
                    &mut group.indices,
                    "JPEG Metal region-scaled grouped indices",
                )?;
                group.indices.push(index);
                continue 'packet;
            }
        }
        let mut indices = budget.try_vec(1, "JPEG Metal region-scaled grouped indices")?;
        indices.push(index);
        groups.push(FastRegionScaledGroup {
            indices,
            scale,
            plan,
        });
    }
    let mut output_budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal region-scaled batch grouping",
        requests,
    )?;
    output_budget.account_capacity::<FastRegionScaledGroup>(groups.capacity())?;
    for group in &groups {
        output_budget.account_capacity::<usize>(group.indices.capacity())?;
    }
    let mut indices = output_budget.try_vec(
        groups.len(),
        "JPEG Metal ordered region-scaled index groups",
    )?;
    indices.extend(groups.into_iter().map(|group| group.indices));
    Ok(Some(indices))
}

pub(super) fn fast444_packets_share_region_scaled_batch_shape(
    first: &JpegFast444PacketV1,
    packet: &JpegFast444PacketV1,
    segment_count: usize,
) -> bool {
    fast_packets_share_batch_shape(first, packet, segment_count, false)
}

struct Fast444RegionScaledGroup {
    indices: Vec<usize>,
    mode: PlaneMode,
    scale: j2k_core::Downscale,
    params: JpegFast444ScaledParams,
}

pub(super) fn fast444_region_scaled_batch_groups(
    requests: &[batch::QueuedRequest],
    packets: &[(&JpegFast444PacketV1, PlaneMode)],
) -> Result<Option<Vec<Vec<usize>>>, Error> {
    let mut budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal fast444 region-scaled batch grouping",
        requests,
    )?;
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<Fast444RegionScaledGroup>(
            requests.len(),
        ),
        crate::batch_allocation::BatchMetadataRequest::of::<Vec<usize>>(requests.len()),
        crate::batch_allocation::BatchMetadataRequest::of::<usize>(requests.len()),
        crate::batch_allocation::BatchMetadataRequest::of::<Vec<usize>>(requests.len()),
    ])?;
    let mut groups: Vec<Fast444RegionScaledGroup> = budget.try_vec(
        requests.len(),
        "JPEG Metal fast444 region-scaled batch groups",
    )?;
    'packet: for (index, (request, (packet, mode))) in
        requests.iter().zip(packets.iter().copied()).enumerate()
    {
        if packet.restart_interval_mcus != 0
            || packet.entropy_bytes.is_empty()
            || packet.entropy_checkpoints.is_empty()
        {
            return Ok(None);
        }
        let batch::BatchOp::RegionScaled { roi, scale } = request.op else {
            return Ok(None);
        };
        let scaled = roi.scaled_covering(scale);
        let scaled_roi = j2k_jpeg::Rect {
            x: scaled.x,
            y: scaled.y,
            w: scaled.w,
            h: scaled.h,
        };
        let Some(params) = fast444_scaled_region_params(packet, scale, scaled_roi) else {
            return Ok(None);
        };

        for group in &mut groups {
            let (first, _) = packets[group.indices[0]];
            if mode == group.mode
                && scale == group.scale
                && params == group.params
                && fast444_packets_share_region_scaled_batch_shape(
                    first,
                    packet,
                    first.entropy_checkpoints.len(),
                )
            {
                crate::batch_allocation::try_reserve_for_push(
                    &mut group.indices,
                    "JPEG Metal fast444 region-scaled grouped indices",
                )?;
                group.indices.push(index);
                continue 'packet;
            }
        }
        let mut indices = budget.try_vec(1, "JPEG Metal fast444 region-scaled grouped indices")?;
        indices.push(index);
        groups.push(Fast444RegionScaledGroup {
            indices,
            mode,
            scale,
            params,
        });
    }
    let mut output_budget = crate::plan_owner_ledger::batch_execution_budget(
        "JPEG Metal fast444 region-scaled batch grouping",
        requests,
    )?;
    output_budget.account_capacity::<Fast444RegionScaledGroup>(groups.capacity())?;
    for group in &groups {
        output_budget.account_capacity::<usize>(group.indices.capacity())?;
    }
    let mut indices = output_budget.try_vec(
        groups.len(),
        "JPEG Metal ordered fast444 region-scaled index groups",
    )?;
    indices.extend(groups.into_iter().map(|group| group.indices));
    Ok(Some(indices))
}
