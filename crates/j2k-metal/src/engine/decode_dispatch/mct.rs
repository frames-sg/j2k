// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use super::{
    checked_buffer_slice, commit_and_wait_metal, copied_slice_buffer, hybrid_stage_signpost,
    new_command_buffer, new_compute_command_encoder, with_runtime, Buffer, CommandBufferRef, Error,
    J2kInverseMctJob, J2kInverseMctParams, J2kWaveletTransform, MetalRuntime,
    SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE,
};

#[cfg(target_os = "macos")]
pub(crate) fn decode_inverse_mct(job: J2kInverseMctJob<'_>) -> Result<Vec<Buffer>, Error> {
    let J2kInverseMctJob {
        transform,
        plane0,
        plane1,
        plane2,
        addend0,
        addend1,
        addend2,
    } = job;
    with_runtime(|runtime| {
        let len = plane0.len();
        if len == 0 {
            return Ok(Vec::new());
        }
        if plane1.len() != len || plane2.len() != len {
            return Err(Error::MetalKernel {
                message: "J2K Metal inverse MCT plane lengths must match".to_string(),
            });
        }

        let transform = match transform {
            J2kWaveletTransform::Reversible53 => 0,
            J2kWaveletTransform::Irreversible97 => 1,
        };
        let params = J2kInverseMctParams {
            _len: u32::try_from(len).map_err(|_| Error::MetalKernel {
                message: "J2K Metal inverse MCT plane length exceeds u32".to_string(),
            })?,
            _transform: transform,
            _addend0: addend0,
            _addend1: addend1,
            _addend2: addend2,
        };
        let plane0_buffer = copied_slice_buffer(&runtime.device, plane0)?;
        let plane1_buffer = copied_slice_buffer(&runtime.device, plane1)?;
        let plane2_buffer = copied_slice_buffer(&runtime.device, plane2)?;
        let command_buffer = new_command_buffer(&runtime.queue)?;
        let encoder = new_compute_command_encoder(&command_buffer)?;
        encoder.setComputePipelineState(&runtime.decode()?.inverse_mct);
        encoder.set_buffer(0, Some(&plane0_buffer), 0);
        encoder.set_buffer(1, Some(&plane1_buffer), 0);
        encoder.set_buffer(2, Some(&plane2_buffer), 0);
        encoder.set_bytes::<J2kInverseMctParams>(3, &params);
        let width = runtime
            .decode()?
            .inverse_mct
            .threadExecutionWidth()
            .max(1)
            .min(len);
        encoder.dispatchThreads_threadsPerThreadgroup(
            j2k_metal_support::mtl_size(len as u64, 1, 1),
            j2k_metal_support::mtl_size(width as u64, 1, 1),
        );
        encoder.endEncoding();
        commit_and_wait_metal(&command_buffer)?;

        let plane0_host = checked_buffer_slice::<f32>(&plane0_buffer, len, "inverse MCT plane 0")?;
        let plane1_host = checked_buffer_slice::<f32>(&plane1_buffer, len, "inverse MCT plane 1")?;
        let plane2_host = checked_buffer_slice::<f32>(&plane2_buffer, len, "inverse MCT plane 2")?;
        plane0.copy_from_slice(&plane0_host);
        plane1.copy_from_slice(&plane1_host);
        plane2.copy_from_slice(&plane2_host);
        crate::batch_allocation::try_vec_from_array(
            [plane0_buffer, plane1_buffer, plane2_buffer],
            "J2K Metal inverse MCT retained buffers",
        )
    })
}

#[cfg(target_os = "macos")]
pub(in crate::engine) fn dispatch_inverse_mct_buffers_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: [&Buffer; 3],
    len: usize,
    transform: J2kWaveletTransform,
    addends: [f32; 3],
) -> Result<(), Error> {
    if len == 0 {
        return Err(Error::MetalKernel {
            message: "J2K MetalDirect color MCT cannot run on an empty plane".to_string(),
        });
    }

    let transform = match transform {
        J2kWaveletTransform::Reversible53 => 0,
        J2kWaveletTransform::Irreversible97 => 1,
    };
    let params = J2kInverseMctParams {
        _len: u32::try_from(len).map_err(|_| Error::MetalKernel {
            message: "J2K MetalDirect color MCT plane length exceeds u32".to_string(),
        })?,
        _transform: transform,
        _addend0: addends[0],
        _addend1: addends[1],
        _addend2: addends[2],
    };
    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE);
    let encoder = new_compute_command_encoder(command_buffer)?;
    encoder.setComputePipelineState(&runtime.decode()?.inverse_mct);
    encoder.set_buffer(0, Some(planes[0]), 0);
    encoder.set_buffer(1, Some(planes[1]), 0);
    encoder.set_buffer(2, Some(planes[2]), 0);
    encoder.set_bytes::<J2kInverseMctParams>(3, &params);
    let width = runtime
        .decode()?
        .inverse_mct
        .threadExecutionWidth()
        .max(1)
        .min(len);
    encoder.dispatchThreads_threadsPerThreadgroup(
        j2k_metal_support::mtl_size(len as u64, 1, 1),
        j2k_metal_support::mtl_size(width as u64, 1, 1),
    );
    encoder.endEncoding();

    Ok(())
}
