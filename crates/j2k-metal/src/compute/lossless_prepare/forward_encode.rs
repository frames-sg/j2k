// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    borrow_mut_slice_buffer, checked_buffer_read, checked_buffer_slice, commit_and_wait_metal,
    copied_slice_buffer, decode_mct_status_error, dispatch_1d_pipeline, label_command_buffer,
    label_compute_encoder, size_of, with_runtime, zeroed_shared_buffer, Error, J2kForwardIctParams,
    J2kForwardRctParams, J2kMctStatus, J2kQuantizeSubbandJob, J2kQuantizeSubbandParams,
    MTLResourceOptions, MTLSize, J2K_MCT_STATUS_OK,
};

#[cfg(target_os = "macos")]
pub(crate) fn encode_forward_rct(
    plane0: &mut [f32],
    plane1: &mut [f32],
    plane2: &mut [f32],
) -> Result<(), Error> {
    let len = plane0.len();
    if len == 0 {
        return Ok(());
    }
    if plane1.len() != len || plane2.len() != len {
        return Err(Error::MetalKernel {
            message: "J2K Metal forward RCT plane lengths must match".to_string(),
        });
    }
    let len_u32 = u32::try_from(len).map_err(|_| Error::MetalKernel {
        message: "J2K Metal forward RCT plane length exceeds u32".to_string(),
    })?;

    with_runtime(|runtime| {
        let params = J2kForwardRctParams {
            _len: len_u32,
            _reserved0: 0,
            _reserved1: 0,
            _reserved2: 0,
        };
        let plane0_buffer = borrow_mut_slice_buffer(&runtime.device, plane0);
        let plane1_buffer = borrow_mut_slice_buffer(&runtime.device, plane1);
        let plane2_buffer = borrow_mut_slice_buffer(&runtime.device, plane2);
        let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kMctStatus>());

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.forward_rct);
        encoder.set_buffer(0, Some(&plane0_buffer), 0);
        encoder.set_buffer(1, Some(&plane1_buffer), 0);
        encoder.set_buffer(2, Some(&plane2_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kForwardRctParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(4, Some(&status_buffer), 0);
        let width = runtime
            .forward_rct
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

        let status = checked_buffer_read::<J2kMctStatus>(&status_buffer, "forward RCT status")?;
        if status.code != J2K_MCT_STATUS_OK {
            return Err(decode_mct_status_error(status));
        }

        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_forward_ict(
    plane0: &mut [f32],
    plane1: &mut [f32],
    plane2: &mut [f32],
) -> Result<(), Error> {
    let len = plane0.len();
    if len == 0 {
        return Ok(());
    }
    if plane1.len() != len || plane2.len() != len {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal forward ICT plane lengths must match",
        });
    }
    let len_u32 = u32::try_from(len).map_err(|_| Error::UnsupportedMetalRequest {
        reason: "J2K Metal forward ICT plane length exceeds u32",
    })?;

    with_runtime(|runtime| {
        let params = J2kForwardIctParams {
            _len: len_u32,
            _reserved0: 0,
            _reserved1: 0,
            _reserved2: 0,
        };
        let plane0_buffer = borrow_mut_slice_buffer(&runtime.device, plane0);
        let plane1_buffer = borrow_mut_slice_buffer(&runtime.device, plane1);
        let plane2_buffer = borrow_mut_slice_buffer(&runtime.device, plane2);
        let status_buffer = zeroed_shared_buffer(&runtime.device, size_of::<J2kMctStatus>());

        let command_buffer = runtime.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&runtime.forward_ict);
        encoder.set_buffer(0, Some(&plane0_buffer), 0);
        encoder.set_buffer(1, Some(&plane1_buffer), 0);
        encoder.set_buffer(2, Some(&plane2_buffer), 0);
        encoder.set_bytes(
            3,
            size_of::<J2kForwardIctParams>() as u64,
            (&raw const params).cast(),
        );
        encoder.set_buffer(4, Some(&status_buffer), 0);
        let width = runtime
            .forward_ict
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

        let status = checked_buffer_read::<J2kMctStatus>(&status_buffer, "forward ICT status")?;
        if status.code != J2K_MCT_STATUS_OK {
            return Err(decode_mct_status_error(status));
        }

        Ok(())
    })
}

#[cfg(target_os = "macos")]
pub(in crate::compute) fn validate_encode_quantize_subband_job(
    job: J2kQuantizeSubbandJob<'_>,
) -> Result<(), Error> {
    if job.step_exponent > 31 {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode quantize_subband supports step exponents <= 31",
        });
    }
    if job.step_mantissa > 2047 {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode quantize_subband supports step mantissas <= 2047",
        });
    }
    if job.range_bits == 0 || job.range_bits > 31 {
        return Err(Error::UnsupportedMetalRequest {
            reason: "J2K Metal encode quantize_subband supports range bits 1-31",
        });
    }
    Ok(())
}

#[cfg(target_os = "macos")]
pub(crate) fn encode_quantize_subband(job: J2kQuantizeSubbandJob<'_>) -> Result<Vec<i32>, Error> {
    validate_encode_quantize_subband_job(job)?;
    let len = job.coefficients.len();
    if len == 0 {
        return Ok(Vec::new());
    }
    let len_u32 = u32::try_from(len).map_err(|_| Error::UnsupportedMetalRequest {
        reason: "J2K Metal encode quantize_subband coefficient count exceeds u32",
    })?;
    let output_bytes = len
        .checked_mul(size_of::<i32>())
        .ok_or_else(|| Error::MetalKernel {
            message: "J2K Metal encode quantize_subband output length overflow".to_string(),
        })?;

    with_runtime(|runtime| {
        let input_buffer = copied_slice_buffer(&runtime.device, job.coefficients);
        let output_buffer = runtime
            .device
            .new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);
        let params = J2kQuantizeSubbandParams {
            _len: len_u32,
            _step_exponent: u32::from(job.step_exponent),
            _step_mantissa: u32::from(job.step_mantissa),
            _range_bits: u32::from(job.range_bits),
            _reversible: u32::from(job.reversible),
            _reserved0: 0,
            _reserved1: 0,
            _reserved2: 0,
        };

        let command_buffer = runtime.queue.new_command_buffer();
        label_command_buffer(command_buffer, "j2k encode-stage quantize_subband");
        let encoder = command_buffer.new_compute_command_encoder();
        label_compute_encoder(encoder, "J2K encode-stage quantize_subband");
        encoder.set_compute_pipeline_state(&runtime.quantize_subband);
        encoder.set_buffer(0, Some(&input_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            size_of::<J2kQuantizeSubbandParams>() as u64,
            (&raw const params).cast(),
        );
        dispatch_1d_pipeline(encoder, &runtime.quantize_subband, u64::from(len_u32));
        encoder.end_encoding();
        commit_and_wait_metal(command_buffer)?;

        checked_buffer_slice::<i32>(&output_buffer, len, "quantized subband")
    })
}
