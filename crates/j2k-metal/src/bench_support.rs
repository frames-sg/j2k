// SPDX-License-Identifier: MIT OR Apache-2.0

//! Benchmark-only Metal upload and direct-plan preparation support.

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;
use crate::{error, hybrid, Error, MetalBackendSession};
use j2k_core::{Downscale, PixelFormat, Rect};
#[cfg(target_os = "macos")]
use j2k_metal_support::{
    checked_blit_command_encoder, checked_command_buffer, checked_private_buffer,
    checked_shared_buffer_with_bytes, commit_and_wait,
};

#[cfg(target_os = "macos")]
#[doc(hidden)]
pub fn benchmark_region_scaled_direct_plan_prepare(
    input: &[u8],
    fmt: PixelFormat,
    roi: Rect,
    scale: Downscale,
) -> Result<(), Error> {
    hybrid::benchmark_region_scaled_direct_plan_prepare(input, fmt, roi, scale)
}

#[cfg(not(target_os = "macos"))]
#[doc(hidden)]
pub fn benchmark_region_scaled_direct_plan_prepare(
    _input: &[u8],
    _fmt: PixelFormat,
    _roi: Rect,
    _scale: Downscale,
) -> Result<(), Error> {
    Err(Error::MetalUnavailable)
}

#[cfg(target_os = "macos")]
#[doc(hidden)]
pub fn benchmark_private_buffer_with_bytes(
    session: &MetalBackendSession,
    bytes: &[u8],
) -> Result<objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>>, Error>
{
    if bytes.is_empty() {
        return Err(Error::MetalKernel {
            message: "J2K Metal benchmark private input upload is empty".to_string(),
        });
    }
    let byte_len = u64::try_from(bytes.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal benchmark private input length exceeds u64".to_string(),
    })?;
    let map_allocation =
        |source| error::metal_kernel_support_error("J2K Metal benchmark buffer allocation", source);
    let upload =
        checked_shared_buffer_with_bytes(session.device(), bytes).map_err(map_allocation)?;
    let private = checked_private_buffer(session.device(), bytes.len()).map_err(map_allocation)?;
    let runtime = session.runtime()?;
    let command_buffer = checked_command_buffer(runtime.command_queue()).map_err(map_allocation)?;
    let blit = checked_blit_command_encoder(&command_buffer).map_err(map_allocation)?;
    blit.copy_from_buffer(&upload, 0, &private, 0, byte_len)?;
    blit.endEncoding();
    commit_and_wait(&command_buffer).map_err(|error| {
        error::metal_kernel_support_error(
            format!("J2K Metal benchmark private input upload failed: {error}"),
            error,
        )
    })?;
    Ok(private)
}

#[cfg(target_os = "macos")]
#[doc(hidden)]
pub fn benchmark_overwrite_private_buffer_with_bytes(
    session: &MetalBackendSession,
    dst: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLBuffer>,
    bytes: &[u8],
) -> Result<(), Error> {
    if bytes.is_empty() {
        return Err(Error::MetalKernel {
            message: "J2K Metal benchmark private input overwrite is empty".to_string(),
        });
    }
    let byte_len = u64::try_from(bytes.len()).map_err(|_| Error::MetalKernel {
        message: "J2K Metal benchmark private input overwrite length exceeds u64".to_string(),
    })?;
    if usize::try_from(byte_len).map_or(true, |byte_len| byte_len > dst.length()) {
        return Err(Error::MetalKernel {
            message: "J2K Metal benchmark private input overwrite exceeds destination buffer"
                .to_string(),
        });
    }
    let map_allocation =
        |source| error::metal_kernel_support_error("J2K Metal benchmark buffer allocation", source);
    let upload =
        checked_shared_buffer_with_bytes(session.device(), bytes).map_err(map_allocation)?;
    let runtime = session.runtime()?;
    let command_buffer = checked_command_buffer(runtime.command_queue()).map_err(map_allocation)?;
    let blit = checked_blit_command_encoder(&command_buffer).map_err(map_allocation)?;
    blit.copy_from_buffer(&upload, 0, dst, 0, byte_len)?;
    blit.endEncoding();
    commit_and_wait(&command_buffer).map_err(|error| {
        error::metal_kernel_support_error(
            format!("J2K Metal benchmark private input overwrite failed: {error}"),
            error,
        )
    })?;
    Ok(())
}
