// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    checked_buffer_read, checked_buffer_slice, commit_and_wait_metal, copied_slice_buffer,
    decode_mct_status_error, hybrid_stage_signpost, size_of, with_runtime, zeroed_shared_buffer,
    Buffer, CommandBufferRef, DirectStatusCheck, Error, J2kInverseMctJob, J2kInverseMctParams,
    J2kMctStatus, J2kWaveletTransform, MTLSize, MetalRuntime, J2K_MCT_STATUS_OK,
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
        let plane0_buffer = copied_slice_buffer(&runtime.device, plane0);
        let plane1_buffer = copied_slice_buffer(&runtime.device, plane1);
        let plane2_buffer = copied_slice_buffer(&runtime.device, plane2);
        let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kMctStatus>());

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.inverse_mct);
        encoder.set_buffer(0, Some(&plane0_buffer), 0);
        encoder.set_buffer(1, Some(&plane1_buffer), 0);
        encoder.set_buffer(2, Some(&plane2_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kInverseMctParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(4, Some(&status_buffer), 0);
        let width = runtime
            .inverse_mct
            .thread_execution_width()
            .max(1)
            .min(len as u64);
        encoder.dispatch_threads(
            MTLSize {
                width: len as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;

        let status = checked_buffer_read::<J2kMctStatus>(&status_buffer, "inverse MCT status")?;
        if status.code != J2K_MCT_STATUS_OK {
            return Err(decode_mct_status_error(status));
        }

        let plane0_host = checked_buffer_slice::<f32>(&plane0_buffer, len, "inverse MCT plane 0")?;
        let plane1_host = checked_buffer_slice::<f32>(&plane1_buffer, len, "inverse MCT plane 1")?;
        let plane2_host = checked_buffer_slice::<f32>(&plane2_buffer, len, "inverse MCT plane 2")?;
        for (dst, sample) in plane0.iter_mut().zip(plane0_host.iter().copied()) {
            *dst = sample - addend0;
        }
        for (dst, sample) in plane1.iter_mut().zip(plane1_host.iter().copied()) {
            *dst = sample - addend1;
        }
        for (dst, sample) in plane2.iter_mut().zip(plane2_host.iter().copied()) {
            *dst = sample - addend2;
        }
        Ok(vec![plane0_buffer, plane1_buffer, plane2_buffer])
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn dispatch_inverse_mct_buffers_in_command_buffer(
    runtime: &MetalRuntime,
    command_buffer: &CommandBufferRef,
    planes: [&Buffer; 3],
    len: usize,
    transform: J2kWaveletTransform,
    addends: [f32; 3],
) -> Result<DirectStatusCheck, Error> {
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
    let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kMctStatus>());

    let _signpost = hybrid_stage_signpost(SIGNPOST_DECODE_HYBRID_MCT_PACK_COMMAND_ENCODE);
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&runtime.inverse_mct);
    encoder.set_buffer(0, Some(planes[0]), 0);
    encoder.set_buffer(1, Some(planes[1]), 0);
    encoder.set_buffer(2, Some(planes[2]), 0);
    encoder.set_bytes(
        3,
        size_of::<J2kInverseMctParams>() as u64,
        (&raw const params).cast(),
    );
    encoder.set_buffer(4, Some(&status_buffer), 0);
    let width = runtime
        .inverse_mct
        .thread_execution_width()
        .max(1)
        .min(len as u64);
    encoder.dispatch_threads(
        MTLSize {
            width: len as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    Ok(DirectStatusCheck::Mct(status_buffer))
}
