// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use j2k_metal_support::{dispatch_2d_pipeline, dispatch_single_thread};
use metal::{Buffer, MTLResourceOptions};

use crate::{profile_env::label_command_buffer, Error};

use super::{
    with_runtime_for_session, J2kCopyInterleavedParams, J2kValidateBytesParams,
    J2kValidateBytesStatus,
};

pub(crate) fn validate_metal_buffer_matches_bytes(
    expected: &[u8],
    actual_buffer: &Buffer,
    actual_byte_offset: usize,
    session: &crate::MetalBackendSession,
) -> Result<(), Error> {
    if expected.is_empty() {
        return Ok(());
    }
    let byte_len = u32::try_from(expected.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal validation buffer exceeds u32 byte length".to_string(),
    })?;
    let actual_offset = u64::try_from(actual_byte_offset).map_err(|_| Error::MetalKernel {
        message: "J2K Metal validation buffer offset exceeds u64".to_string(),
    })?;

    with_runtime_for_session(session, |runtime| {
        let expected_buffer = runtime.device.new_buffer_with_data(
            expected.as_ptr().cast(),
            expected.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let status = J2kValidateBytesStatus::default();
        let status_buffer = runtime.device.new_buffer_with_data(
            (&raw const status).cast(),
            size_of::<J2kValidateBytesStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params = J2kValidateBytesParams { byte_len };

        let command_buffer = runtime.queue.new_command_buffer();
        label_command_buffer(command_buffer, "j2k lossless coefficient prep");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.validate_bytes_equal);
        encoder.set_buffer(0, Some(actual_buffer), actual_offset);
        encoder.set_buffer(1, Some(&expected_buffer), 0);
        encoder.set_buffer(2, Some(&status_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kValidateBytesParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_single_thread(encoder);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let status = unsafe {
            status_buffer
                .contents()
                .cast::<J2kValidateBytesStatus>()
                .read()
        };
        if status.code == 0 {
            return Ok(());
        }

        Err(Error::MetalKernel {
            message: format!(
                "J2K Metal validation mismatch at byte {}: expected {}, got {}",
                status.index, status.expected, status.actual
            ),
        })
    })
}

pub(crate) fn validate_metal_buffers_match(
    expected_buffer: &Buffer,
    expected_byte_offset: usize,
    actual_buffer: &Buffer,
    actual_byte_offset: usize,
    byte_len: usize,
    session: &crate::MetalBackendSession,
) -> Result<(), Error> {
    if byte_len == 0 {
        return Ok(());
    }
    let byte_len_u32 = u32::try_from(byte_len).map_err(|_| Error::MetalKernel {
        message: "J2K Metal validation buffer exceeds u32 byte length".to_string(),
    })?;
    let expected_offset = u64::try_from(expected_byte_offset).map_err(|_| Error::MetalKernel {
        message: "J2K Metal validation expected buffer offset exceeds u64".to_string(),
    })?;
    let actual_offset = u64::try_from(actual_byte_offset).map_err(|_| Error::MetalKernel {
        message: "J2K Metal validation actual buffer offset exceeds u64".to_string(),
    })?;

    with_runtime_for_session(session, |runtime| {
        let status = J2kValidateBytesStatus::default();
        let status_buffer = runtime.device.new_buffer_with_data(
            (&raw const status).cast(),
            size_of::<J2kValidateBytesStatus>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let params = J2kValidateBytesParams {
            byte_len: byte_len_u32,
        };

        let command_buffer = runtime.queue.new_command_buffer();
        label_command_buffer(command_buffer, "j2k lossless coefficient prep batch");
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.validate_bytes_equal);
        encoder.set_buffer(0, Some(actual_buffer), actual_offset);
        encoder.set_buffer(1, Some(expected_buffer), expected_offset);
        encoder.set_buffer(2, Some(&status_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kValidateBytesParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_single_thread(encoder);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // SAFETY: Metal buffer access follows validated sizes and synchronized command completion.
        let status = unsafe {
            status_buffer
                .contents()
                .cast::<J2kValidateBytesStatus>()
                .read()
        };
        if status.code == 0 {
            return Ok(());
        }

        Err(Error::MetalKernel {
            message: format!(
                "J2K Metal validation mismatch at byte {}: expected {}, got {}",
                status.index, status.expected, status.actual
            ),
        })
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_interleaved_padded_to_shared_buffer(
    src_buffer: &Buffer,
    src_byte_offset: usize,
    src_width: u32,
    src_height: u32,
    src_pitch_bytes: usize,
    dst_width: u32,
    dst_height: u32,
    bytes_per_pixel: usize,
    session: &crate::MetalBackendSession,
) -> Result<Buffer, Error> {
    if src_width > dst_width || src_height > dst_height {
        return Err(Error::MetalKernel {
            message: "J2K Metal input tile cannot be larger than encoded tile".to_string(),
        });
    }
    let src_stride = u32::try_from(src_pitch_bytes).map_err(|_| Error::MetalKernel {
        message: "J2K Metal input tile pitch exceeds u32".to_string(),
    })?;
    let dst_stride_usize = (dst_width as usize)
        .checked_mul(bytes_per_pixel)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal padded tile stride overflow".to_string(),
        })?;
    let dst_stride = u32::try_from(dst_stride_usize).map_err(|_| Error::MetalKernel {
        message: "J2K Metal padded tile stride exceeds u32".to_string(),
    })?;
    let dst_len = dst_stride_usize
        .checked_mul(dst_height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal padded tile byte length overflow".to_string(),
        })?;
    let bytes_per_pixel = u32::try_from(bytes_per_pixel).map_err(|_| Error::MetalKernel {
        message: "J2K Metal bytes-per-pixel exceeds u32".to_string(),
    })?;
    let src_offset = u64::try_from(src_byte_offset).map_err(|_| Error::MetalKernel {
        message: "J2K Metal input tile offset exceeds u64".to_string(),
    })?;

    with_runtime_for_session(session, |runtime| {
        let dst_buffer = runtime
            .device
            .new_buffer(dst_len as u64, MTLResourceOptions::StorageModeShared);
        let params = J2kCopyInterleavedParams {
            src_width,
            src_height,
            src_stride,
            dst_width,
            dst_height,
            dst_stride,
            bytes_per_pixel,
        };
        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.copy_interleaved_padded);
        encoder.set_buffer(0, Some(src_buffer), src_offset);
        encoder.set_buffer(1, Some(&dst_buffer), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kCopyInterleavedParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(
            encoder,
            &runtime.copy_interleaved_padded,
            (dst_width, dst_height),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();
        Ok(dst_buffer)
    })
}
