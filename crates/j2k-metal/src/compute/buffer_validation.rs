// SPDX-License-Identifier: MIT OR Apache-2.0

use std::mem::size_of;

use j2k_metal_support::{dispatch_2d_pipeline, dispatch_single_thread};
use metal::Buffer;

use crate::{profile_env::label_command_buffer, Error};

use super::abi::{J2kCopyInterleavedParams, J2kValidateBytesParams, J2kValidateBytesStatus};
use super::{
    commit_and_wait_metal,
    direct_buffers::{checked_buffer_read, zeroed_shared_buffer},
    new_command_buffer, new_compute_command_encoder, new_shared_buffer,
    new_shared_buffer_with_slice, with_runtime_for_session,
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
        let expected_buffer = new_shared_buffer_with_slice(&runtime.device, expected)?;
        let status_buffer =
            zeroed_shared_buffer(&runtime.device, size_of::<J2kValidateBytesStatus>())?;
        let params = J2kValidateBytesParams { byte_len };

        let command_buffer = new_command_buffer(&runtime.queue)?;
        label_command_buffer(&command_buffer, "j2k lossless coefficient prep");
        let encoder = new_compute_command_encoder(&command_buffer)?;
        encoder.set_compute_pipeline_state(&runtime.validate_bytes_equal);
        encoder.set_buffer(0, Some(actual_buffer), actual_offset);
        encoder.set_buffer(1, Some(&expected_buffer), 0);
        encoder.set_buffer(2, Some(&status_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kValidateBytesParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_single_thread(&encoder);
        encoder.end_encoding();
        commit_and_wait_metal(&command_buffer)?;

        let status = checked_buffer_read::<J2kValidateBytesStatus>(
            &status_buffer,
            "byte validation status",
        )?;
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
        let status_buffer =
            zeroed_shared_buffer(&runtime.device, size_of::<J2kValidateBytesStatus>())?;
        let params = J2kValidateBytesParams {
            byte_len: byte_len_u32,
        };

        let command_buffer = new_command_buffer(&runtime.queue)?;
        label_command_buffer(&command_buffer, "j2k lossless coefficient prep batch");
        let encoder = new_compute_command_encoder(&command_buffer)?;
        encoder.set_compute_pipeline_state(&runtime.validate_bytes_equal);
        encoder.set_buffer(0, Some(actual_buffer), actual_offset);
        encoder.set_buffer(1, Some(expected_buffer), expected_offset);
        encoder.set_buffer(2, Some(&status_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kValidateBytesParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_single_thread(&encoder);
        encoder.end_encoding();
        commit_and_wait_metal(&command_buffer)?;

        let status = checked_buffer_read::<J2kValidateBytesStatus>(
            &status_buffer,
            "byte validation status",
        )?;
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

#[derive(Clone, Copy)]
pub(crate) struct PaddedInterleavedCopy<'a> {
    pub(crate) src_buffer: &'a Buffer,
    pub(crate) src_byte_offset: usize,
    pub(crate) src_width: u32,
    pub(crate) src_height: u32,
    pub(crate) src_pitch_bytes: usize,
    pub(crate) dst_width: u32,
    pub(crate) dst_height: u32,
    pub(crate) bytes_per_pixel: usize,
    pub(crate) session: &'a crate::MetalBackendSession,
}

pub(crate) fn copy_interleaved_padded_to_shared_buffer(
    copy: PaddedInterleavedCopy<'_>,
) -> Result<Buffer, Error> {
    if copy.src_width > copy.dst_width || copy.src_height > copy.dst_height {
        return Err(Error::MetalKernel {
            message: "J2K Metal input tile cannot be larger than encoded tile".to_string(),
        });
    }
    let src_stride = u32::try_from(copy.src_pitch_bytes).map_err(|_| Error::MetalKernel {
        message: "J2K Metal input tile pitch exceeds u32".to_string(),
    })?;
    let dst_stride_usize = (copy.dst_width as usize)
        .checked_mul(copy.bytes_per_pixel)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal padded tile stride overflow".to_string(),
        })?;
    let dst_stride = u32::try_from(dst_stride_usize).map_err(|_| Error::MetalKernel {
        message: "J2K Metal padded tile stride exceeds u32".to_string(),
    })?;
    let dst_len = dst_stride_usize
        .checked_mul(copy.dst_height as usize)
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal padded tile byte length overflow".to_string(),
        })?;
    let bytes_per_pixel = u32::try_from(copy.bytes_per_pixel).map_err(|_| Error::MetalKernel {
        message: "J2K Metal bytes-per-pixel exceeds u32".to_string(),
    })?;
    let src_offset = u64::try_from(copy.src_byte_offset).map_err(|_| Error::MetalKernel {
        message: "J2K Metal input tile offset exceeds u64".to_string(),
    })?;

    with_runtime_for_session(copy.session, |runtime| {
        let dst_buffer = new_shared_buffer(&runtime.device, dst_len)?;
        let params = J2kCopyInterleavedParams {
            src_width: copy.src_width,
            src_height: copy.src_height,
            src_stride,
            dst_width: copy.dst_width,
            dst_height: copy.dst_height,
            dst_stride,
            bytes_per_pixel,
        };
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        encoder.set_compute_pipeline_state(&runtime.copy_interleaved_padded);
        encoder.set_buffer(0, Some(copy.src_buffer), src_offset);
        encoder.set_buffer(1, Some(&dst_buffer), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kCopyInterleavedParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_2d_pipeline(
            &encoder,
            &runtime.copy_interleaved_padded,
            (copy.dst_width, copy.dst_height),
        );
        encoder.end_encoding();
        commit_and_wait_metal(&command_buffer)?;
        Ok(dst_buffer)
    })
}
