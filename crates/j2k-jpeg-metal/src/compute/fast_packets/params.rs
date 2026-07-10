// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    checked_u32, new_shared_buffer_with_data, pixel_format_to_out_format, Buffer, Device, Error,
    JpegEntropyCheckpointHost, JpegEntropyCheckpointV1, JpegFast420Params, JpegFast420ScaledParams,
    JpegFast420WindowedPackParams, JpegFast444PacketV1, JpegFast444Params, JpegFast444ScaledParams,
    PixelFormat,
};
use super::descriptors::FastSubsampledPacket;

pub(in crate::compute) fn fast_subsampled_params<P: FastSubsampledPacket>(
    packet: &P,
    fmt: PixelFormat,
) -> Result<JpegFast420Params, Error> {
    let out_format = pixel_format_to_out_format(fmt).ok_or_else(|| Error::MetalKernel {
        message: format!(
            "unsupported JPEG Metal {} pixel format {fmt:?}",
            P::FAMILY_NAME
        ),
    })?;
    let out_stride = packet.dimensions().0 as usize * fmt.bytes_per_pixel();
    Ok(JpegFast420Params {
        width: packet.dimensions().0,
        height: packet.dimensions().1,
        chroma_width: P::chroma_width(packet.dimensions().0),
        chroma_height: P::chroma_height(packet.dimensions().1),
        mcus_per_row: packet.mcus_per_row(),
        mcu_rows: packet.mcu_rows(),
        restart_interval_mcus: packet.restart_interval_mcus(),
        restart_offset_count: checked_entropy_segment_count(
            packet.restart_interval_mcus(),
            packet.restart_offsets().len(),
            packet.entropy_checkpoints().len(),
        )?,
        restart_start_mcu: 0,
        entropy_len: checked_u32(packet.entropy_bytes().len(), P::ENTROPY_PAYLOAD_CTX)?,
        out_stride: checked_u32(out_stride, P::OUTPUT_STRIDE_CTX)?,
        alpha: u32::from(u8::MAX),
        out_format,
        origin_x: 0,
        origin_y: 0,
    })
}

pub(in crate::compute) fn fast_subsampled_region_params<P: FastSubsampledPacket>(
    packet: &P,
    fmt: PixelFormat,
    source_window: j2k_jpeg::Rect,
) -> Result<JpegFast420Params, Error> {
    let out_format = pixel_format_to_out_format(fmt).ok_or_else(|| Error::MetalKernel {
        message: format!(
            "unsupported JPEG Metal {} pixel format {fmt:?}",
            P::FAMILY_NAME
        ),
    })?;
    let out_stride = source_window.w as usize * fmt.bytes_per_pixel();
    Ok(JpegFast420Params {
        width: source_window.w,
        height: source_window.h,
        chroma_width: P::chroma_width(source_window.w),
        chroma_height: P::chroma_height(source_window.h),
        mcus_per_row: packet.mcus_per_row(),
        mcu_rows: packet.mcu_rows(),
        restart_interval_mcus: packet.restart_interval_mcus(),
        restart_offset_count: checked_entropy_segment_count(
            packet.restart_interval_mcus(),
            packet.restart_offsets().len(),
            packet.entropy_checkpoints().len(),
        )?,
        restart_start_mcu: 0,
        entropy_len: checked_u32(packet.entropy_bytes().len(), P::ENTROPY_PAYLOAD_CTX)?,
        out_stride: checked_u32(out_stride, P::REGION_OUTPUT_STRIDE_CTX)?,
        alpha: u32::from(u8::MAX),
        out_format,
        origin_x: source_window.x,
        origin_y: source_window.y,
    })
}

pub(in crate::compute) fn fast_subsampled_scaled_params<P: FastSubsampledPacket>(
    packet: &P,
    scale: j2k_core::Downscale,
) -> Option<JpegFast420ScaledParams> {
    let scale_shift = match scale {
        j2k_core::Downscale::None => 0,
        j2k_core::Downscale::Half => 1,
        j2k_core::Downscale::Quarter => 2,
        j2k_core::Downscale::Eighth => 3,
        _ => return None,
    };
    let denom = 1u32 << scale_shift;
    let scaled_width = packet.dimensions().0.div_ceil(denom);
    let scaled_height = packet.dimensions().1.div_ceil(denom);
    Some(JpegFast420ScaledParams {
        scaled_width,
        scaled_height,
        chroma_width: P::chroma_width(scaled_width),
        chroma_height: P::chroma_height(scaled_height),
        mcus_per_row: packet.mcus_per_row(),
        mcu_rows: packet.mcu_rows(),
        restart_interval_mcus: packet.restart_interval_mcus(),
        restart_offset_count: optional_entropy_segment_count(
            packet.restart_interval_mcus(),
            packet.restart_offsets().len(),
            packet.entropy_checkpoints().len(),
        )?,
        restart_start_mcu: 0,
        entropy_len: checked_u32(packet.entropy_bytes().len(), P::SCALED_ENTROPY_PAYLOAD_CTX)
            .ok()?,
        scale_shift,
        origin_x: 0,
        origin_y: 0,
    })
}

pub(in crate::compute) fn fast_subsampled_scaled_region_params<P: FastSubsampledPacket>(
    packet: &P,
    scale: j2k_core::Downscale,
    source_window: j2k_jpeg::Rect,
) -> Option<JpegFast420ScaledParams> {
    let full = fast_subsampled_scaled_params(packet, scale)?;
    Some(JpegFast420ScaledParams {
        scaled_width: source_window.w,
        scaled_height: source_window.h,
        chroma_width: P::chroma_width(source_window.w),
        chroma_height: P::chroma_height(source_window.h),
        origin_x: source_window.x,
        origin_y: source_window.y,
        ..full
    })
}

pub(in crate::compute) fn fast_subsampled_full_mcu_window<P: FastSubsampledPacket>(
    dims: (u32, u32),
    roi: j2k_jpeg::Rect,
) -> j2k_jpeg::Rect {
    let x0 = (roi.x / P::MCU_WIDTH) * P::MCU_WIDTH;
    let y0 = (roi.y / P::MCU_HEIGHT) * P::MCU_HEIGHT;
    let x1 = (roi.x + roi.w).div_ceil(P::MCU_WIDTH) * P::MCU_WIDTH;
    let y1 = (roi.y + roi.h).div_ceil(P::MCU_HEIGHT) * P::MCU_HEIGHT;
    j2k_jpeg::Rect {
        x: x0,
        y: y0,
        w: x1.min(dims.0).saturating_sub(x0),
        h: y1.min(dims.1).saturating_sub(y0),
    }
}

pub(in crate::compute) fn fast_subsampled_full_mcu_scaled_window<P: FastSubsampledPacket>(
    scaled_dims: (u32, u32),
    roi: j2k_jpeg::Rect,
    scale_shift: u32,
) -> j2k_jpeg::Rect {
    let mcu_width = P::MCU_WIDTH >> scale_shift;
    let mcu_height = P::MCU_HEIGHT >> scale_shift;
    let x0 = (roi.x / mcu_width) * mcu_width;
    let y0 = (roi.y / mcu_height) * mcu_height;
    let x1 = (roi.x + roi.w).div_ceil(mcu_width) * mcu_width;
    let y1 = (roi.y + roi.h).div_ceil(mcu_height) * mcu_height;
    j2k_jpeg::Rect {
        x: x0,
        y: y0,
        w: x1.min(scaled_dims.0).saturating_sub(x0),
        h: y1.min(scaled_dims.1).saturating_sub(y0),
    }
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn fast444_params(
    packet: &JpegFast444PacketV1,
) -> Result<JpegFast444Params, Error> {
    Ok(JpegFast444Params {
        width: packet.dimensions.0,
        height: packet.dimensions.1,
        mcus_per_row: packet.mcus_per_row,
        mcu_rows: packet.mcu_rows,
        restart_interval_mcus: packet.restart_interval_mcus,
        restart_offset_count: checked_entropy_segment_count(
            packet.restart_interval_mcus,
            packet.restart_offsets.len(),
            packet.entropy_checkpoints.len(),
        )?,
        restart_start_mcu: 0,
        entropy_len: checked_u32(packet.entropy_bytes.len(), "fast444 entropy payload")?,
        origin_x: 0,
        origin_y: 0,
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn fast444_region_params(
    packet: &JpegFast444PacketV1,
    roi: j2k_jpeg::Rect,
) -> Result<JpegFast444Params, Error> {
    Ok(JpegFast444Params {
        width: roi.w,
        height: roi.h,
        origin_x: roi.x,
        origin_y: roi.y,
        ..fast444_params(packet)?
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn fast444_scaled_params(
    packet: &JpegFast444PacketV1,
    scale: j2k_core::Downscale,
) -> Option<JpegFast444ScaledParams> {
    let scale_shift = match scale {
        j2k_core::Downscale::None => 0,
        j2k_core::Downscale::Half => 1,
        j2k_core::Downscale::Quarter => 2,
        j2k_core::Downscale::Eighth => 3,
        _ => return None,
    };
    let denom = 1u32 << scale_shift;
    Some(JpegFast444ScaledParams {
        scaled_width: packet.dimensions.0.div_ceil(denom),
        scaled_height: packet.dimensions.1.div_ceil(denom),
        mcus_per_row: packet.mcus_per_row,
        mcu_rows: packet.mcu_rows,
        restart_interval_mcus: packet.restart_interval_mcus,
        restart_offset_count: optional_entropy_segment_count(
            packet.restart_interval_mcus,
            packet.restart_offsets.len(),
            packet.entropy_checkpoints.len(),
        )?,
        restart_start_mcu: 0,
        entropy_len: checked_u32(packet.entropy_bytes.len(), "fast444 scaled entropy payload")
            .ok()?,
        scale_shift,
        origin_x: 0,
        origin_y: 0,
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn fast444_scaled_region_params(
    packet: &JpegFast444PacketV1,
    scale: j2k_core::Downscale,
    roi: j2k_jpeg::Rect,
) -> Option<JpegFast444ScaledParams> {
    Some(JpegFast444ScaledParams {
        scaled_width: roi.w,
        scaled_height: roi.h,
        origin_x: roi.x,
        origin_y: roi.y,
        ..fast444_scaled_params(packet, scale)?
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn fast_subsampled_windowed_pack_params_for_dims<P: FastSubsampledPacket>(
    dims: (u32, u32),
    fmt: PixelFormat,
    roi: j2k_jpeg::Rect,
) -> Result<JpegFast420WindowedPackParams, Error> {
    let out_format = pixel_format_to_out_format(fmt).ok_or_else(|| Error::MetalKernel {
        message: format!(
            "unsupported JPEG Metal {} pixel format {fmt:?}",
            P::FAMILY_NAME
        ),
    })?;
    let out_stride = roi.w as usize * fmt.bytes_per_pixel();
    Ok(JpegFast420WindowedPackParams {
        src_width: dims.0,
        src_height: dims.1,
        chroma_width: P::chroma_width(dims.0),
        chroma_height: P::chroma_height(dims.1),
        src_x: roi.x,
        src_y: roi.y,
        width: roi.w,
        height: roi.h,
        out_stride: checked_u32(
            out_stride,
            &format!("{} windowed output stride", P::FAMILY_NAME),
        )?,
        alpha: u32::from(u8::MAX),
        out_format,
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn restart_offsets_buffer(
    device: &Device,
    restart_offsets: &[u32],
) -> Result<Buffer, Error> {
    if restart_offsets.is_empty() {
        return Err(Error::MetalKernel {
            message: "JPEG Metal restart offsets must contain at least one entry".to_string(),
        });
    }
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let bytes = unsafe {
        core::slice::from_raw_parts(
            restart_offsets.as_ptr().cast::<u8>(),
            size_of_val(restart_offsets),
        )
    };
    Ok(new_shared_buffer_with_data(device, bytes))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn entropy_checkpoints_buffer(
    device: &Device,
    entropy_checkpoints: &[JpegEntropyCheckpointV1],
) -> Result<Buffer, Error> {
    if entropy_checkpoints.is_empty() {
        return Err(Error::MetalKernel {
            message: "JPEG Metal entropy checkpoints must contain at least one entry".to_string(),
        });
    }
    let checkpoints = entropy_checkpoint_hosts(entropy_checkpoints)?;
    // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
    let bytes = unsafe {
        core::slice::from_raw_parts(
            checkpoints.as_ptr().cast::<u8>(),
            size_of_val(checkpoints.as_slice()),
        )
    };
    Ok(new_shared_buffer_with_data(device, bytes))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn entropy_checkpoint_hosts(
    entropy_checkpoints: &[JpegEntropyCheckpointV1],
) -> Result<Vec<JpegEntropyCheckpointHost>, Error> {
    if entropy_checkpoints.is_empty() {
        return Err(Error::MetalKernel {
            message: "JPEG Metal entropy checkpoints must contain at least one entry".to_string(),
        });
    }
    Ok(entropy_checkpoints
        .iter()
        .copied()
        .map(JpegEntropyCheckpointHost::from)
        .collect::<Vec<_>>())
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn entropy_segment_count(
    restart_interval_mcus: u32,
    restart_offsets_len: usize,
    entropy_checkpoints_len: usize,
) -> u32 {
    let len = if restart_interval_mcus == 0 {
        entropy_checkpoints_len
    } else {
        restart_offsets_len
    };
    u32::try_from(len)
        .expect("JPEG Metal entropy segment count fits in u32")
        .max(1)
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn optional_entropy_segment_count(
    restart_interval_mcus: u32,
    restart_offsets_len: usize,
    entropy_checkpoints_len: usize,
) -> Option<u32> {
    let len = if restart_interval_mcus == 0 {
        entropy_checkpoints_len
    } else {
        restart_offsets_len
    };
    u32::try_from(len).ok().map(|count| count.max(1))
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn checked_entropy_segment_count(
    restart_interval_mcus: u32,
    restart_offsets_len: usize,
    entropy_checkpoints_len: usize,
) -> Result<u32, Error> {
    optional_entropy_segment_count(
        restart_interval_mcus,
        restart_offsets_len,
        entropy_checkpoints_len,
    )
    .ok_or_else(|| Error::MetalKernel {
        message: "JPEG Metal entropy segment count does not fit in u32".to_string(),
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn restart_work_for_mcu_range(
    restart_offsets: &[u32],
    restart_interval_mcus: u32,
    total_mcus: u32,
    first_mcu: u32,
    end_mcu: u32,
) -> (u32, &[u32]) {
    if restart_interval_mcus == 0 || restart_offsets.len() <= 1 {
        return (0, restart_offsets);
    }

    let first_mcu = first_mcu.min(total_mcus);
    let end_mcu = end_mcu.min(total_mcus).max(first_mcu + 1);
    let restart_offset_count =
        u32::try_from(restart_offsets.len()).expect("JPEG Metal restart offsets fit in u32");
    let first_segment = (first_mcu / restart_interval_mcus).min(restart_offset_count - 1);
    let end_segment = end_mcu
        .div_ceil(restart_interval_mcus)
        .min(restart_offset_count)
        .max(first_segment + 1);
    (
        first_segment * restart_interval_mcus,
        &restart_offsets[first_segment as usize..end_segment as usize],
    )
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn mcu_range_for_rect(
    rect: j2k_jpeg::Rect,
    mcus_per_row: u32,
    mcu_rows: u32,
    mcu_width: u32,
    mcu_height: u32,
) -> (u32, u32) {
    if rect.w == 0 || rect.h == 0 || mcus_per_row == 0 || mcu_rows == 0 {
        return (0, 0);
    }

    let max_col = mcus_per_row - 1;
    let max_row = mcu_rows - 1;
    let last_x = rect.x.saturating_add(rect.w).saturating_sub(1);
    let last_y = rect.y.saturating_add(rect.h).saturating_sub(1);
    let first_col = (rect.x / mcu_width).min(max_col);
    let last_col = (last_x / mcu_width).min(max_col);
    let first_row = (rect.y / mcu_height).min(max_row);
    let last_row = (last_y / mcu_height).min(max_row);
    let first_mcu = first_row * mcus_per_row + first_col;
    let end_mcu = last_row * mcus_per_row + last_col + 1;
    (first_mcu, end_mcu)
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn entropy_decode_thread_count(
    restart_interval_mcus: u32,
    restart_offsets_len: usize,
    entropy_checkpoints_len: usize,
) -> u32 {
    entropy_segment_count(
        restart_interval_mcus,
        restart_offsets_len,
        entropy_checkpoints_len,
    )
}
