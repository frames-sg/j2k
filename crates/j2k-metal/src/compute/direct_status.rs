// SPDX-License-Identifier: MIT OR Apache-2.0

use metal::Buffer;

use crate::{buffer_pool::PooledBuffer, Error, MetalDirectFallbackReason};

use super::abi::{
    J2kClassicStatus, J2kHtStatus, J2kMctStatus, J2K_CLASSIC_STATUS_FAIL, J2K_CLASSIC_STATUS_OK,
    J2K_CLASSIC_STATUS_UNSUPPORTED, J2K_HT_STATUS_FAIL, J2K_HT_STATUS_OK,
    J2K_HT_STATUS_UNSUPPORTED, J2K_MCT_STATUS_FAIL,
};
use super::direct_buffers::checked_buffer_slice;

pub(super) enum DirectStatusCheck {
    Classic {
        buffer: Buffer,
        len: usize,
        source_indices: Option<Vec<usize>>,
    },
    Ht {
        buffer: Buffer,
        len: usize,
        source_indices: Option<Vec<usize>>,
        recyclable_status: Option<PooledBuffer>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DirectStatusRetirementMode {
    Validate,
    RecycleWithoutRead,
}

impl DirectStatusCheck {
    fn attributed_sources_mut(&mut self, state: &'static str) -> Result<&mut Vec<usize>, Error> {
        let (len, source_indices) = match self {
            Self::Classic {
                len,
                source_indices,
                ..
            }
            | Self::Ht {
                len,
                source_indices,
                ..
            } => (*len, source_indices),
        };
        let indices = source_indices.as_mut().ok_or(Error::MetalStateInvariant {
            state,
            reason: "codec status check has no source identity table",
        })?;
        if indices.len() != len {
            return Err(Error::MetalStateInvariant {
                state,
                reason: "source identity count does not match status count",
            });
        }
        Ok(indices)
    }

    pub(super) fn remap_source(&mut self, source_index: usize) -> Result<(), Error> {
        let indices = self.attributed_sources_mut("J2K Metal batch status attribution")?;
        indices.fill(source_index);
        Ok(())
    }

    pub(super) fn remap_sources(&mut self, sources: &[usize]) -> Result<(), Error> {
        let indices = self.attributed_sources_mut("J2K Metal stacked batch status attribution")?;
        for source in indices {
            *source = *sources.get(*source).ok_or(Error::MetalStateInvariant {
                state: "J2K Metal stacked batch status attribution",
                reason: "local codec-status source identity exceeds prepared group",
            })?;
        }
        Ok(())
    }
}

#[cfg(test)]
pub(super) fn validate_direct_status(
    runtime: &super::MetalRuntime,
    status_check: DirectStatusCheck,
) -> Result<(), Error> {
    retire_direct_status_checks(
        runtime,
        core::iter::once(status_check),
        DirectStatusRetirementMode::Validate,
    )
}

pub(super) fn retire_direct_status_checks(
    runtime: &super::MetalRuntime,
    status_checks: impl IntoIterator<Item = DirectStatusCheck>,
    mode: DirectStatusRetirementMode,
) -> Result<(), Error> {
    let mut first_validation_error = None;
    let mut first_recycle_error = None;
    for mut status_check in status_checks {
        if mode == DirectStatusRetirementMode::Validate {
            if let Err(error) = validate_direct_status_contents(&status_check) {
                if first_validation_error.is_none() {
                    first_validation_error = Some(error);
                }
            }
        }
        if let DirectStatusCheck::Ht {
            recyclable_status, ..
        } = &mut status_check
        {
            if let Some(owner) = recyclable_status.take() {
                if let Err(error) = runtime.recycle_shared_buffer(owner) {
                    if first_recycle_error.is_none() {
                        first_recycle_error = Some(error);
                    }
                }
            }
        }
    }
    first_validation_error
        .or(first_recycle_error)
        .map_or(Ok(()), Err)
}

fn validate_direct_status_contents(status_check: &DirectStatusCheck) -> Result<(), Error> {
    match status_check {
        DirectStatusCheck::Classic {
            buffer,
            len,
            source_indices,
        } => {
            let statuses =
                checked_buffer_slice::<J2kClassicStatus>(buffer, *len, "classic direct status")?;
            if let Some((status_index, status)) = statuses
                .iter()
                .copied()
                .enumerate()
                .find(|(_, status)| status.code != J2K_CLASSIC_STATUS_OK)
            {
                let source_index = source_indices
                    .as_ref()
                    .and_then(|indices| indices.get(status_index))
                    .copied();
                return Err(decode_classic_status_error_for_source(status, source_index));
            }
        }
        DirectStatusCheck::Ht {
            buffer,
            len,
            source_indices,
            ..
        } => {
            let statuses = checked_buffer_slice::<J2kHtStatus>(buffer, *len, "HT direct status")?;
            if let Some((status_index, status)) = statuses
                .iter()
                .copied()
                .enumerate()
                .find(|(_, status)| status.code != J2K_HT_STATUS_OK)
            {
                let source_index = source_indices
                    .as_ref()
                    .and_then(|indices| indices.get(status_index))
                    .copied();
                return Err(decode_ht_status_error_for_source(status, source_index));
            }
        }
    }

    Ok(())
}

pub(super) fn decode_classic_status_error(status: J2kClassicStatus) -> Error {
    decode_classic_status_error_for_source(status, None)
}

fn decode_classic_status_error_for_source(
    status: J2kClassicStatus,
    source_index: Option<usize>,
) -> Error {
    let source = source_index.map_or_else(String::new, |index| format!(" for source {index}"));
    if status.code == J2K_CLASSIC_STATUS_UNSUPPORTED {
        return Error::MetalDirectFallback {
            message: format!(
                "classic J2K Metal kernel unsupported classic kernel input{source} (detail={})",
                status.detail
            ),
            reason: MetalDirectFallbackReason::UnsupportedRuntimeInput,
        };
    }
    let kind = match status.code {
        J2K_CLASSIC_STATUS_FAIL => "decode failure",
        _ => "unexpected classic kernel status",
    };
    Error::MetalKernel {
        message: format!(
            "classic J2K Metal kernel {kind}{source} (detail={})",
            status.detail
        ),
    }
}

pub(super) fn decode_mct_status_error(status: J2kMctStatus) -> Error {
    let kind = match status.code {
        J2K_MCT_STATUS_FAIL => "decode failure",
        _ => "unexpected status",
    };
    Error::MetalKernel {
        message: format!(
            "J2K Metal color transform kernel {kind} (detail={})",
            status.detail
        ),
    }
}

pub(super) fn decode_ht_status_error(status: J2kHtStatus) -> Error {
    decode_ht_status_error_for_source(status, None)
}

fn decode_ht_status_error_for_source(status: J2kHtStatus, source_index: Option<usize>) -> Error {
    let source = source_index.map_or_else(String::new, |index| format!(" for source {index}"));
    if status.code == J2K_HT_STATUS_UNSUPPORTED {
        return Error::MetalDirectFallback {
            message: format!(
                "HTJ2K Metal kernel unsupported HT kernel input{source} (detail={})",
                status.detail,
            ),
            reason: MetalDirectFallbackReason::UnsupportedRuntimeInput,
        };
    }
    let kind = match status.code {
        J2K_HT_STATUS_FAIL => "decode failure",
        _ => "unexpected HT kernel status",
    };
    Error::MetalKernel {
        message: format!(
            "HTJ2K Metal kernel {kind}{source} (detail={})",
            status.detail
        ),
    }
}

#[cfg(test)]
mod ht_source_tests {
    use super::{decode_ht_status_error_for_source, Error, J2kHtStatus, J2K_HT_STATUS_FAIL};

    #[test]
    fn ht_status_failure_names_the_responsible_batch_source() {
        let error = decode_ht_status_error_for_source(
            J2kHtStatus {
                code: J2K_HT_STATUS_FAIL,
                detail: 17,
                ..J2kHtStatus::default()
            },
            Some(9),
        );

        assert!(matches!(
            error,
            Error::MetalKernel { message }
                if message.contains("source 9") && message.contains("detail=17")
        ));
    }
}

#[cfg(test)]
mod mct_status_tests {
    use super::{decode_mct_status_error, J2kMctStatus, J2K_MCT_STATUS_FAIL};

    #[test]
    fn shared_mct_status_error_does_not_mislabel_forward_transforms_as_inverse() {
        let message = decode_mct_status_error(J2kMctStatus {
            code: J2K_MCT_STATUS_FAIL,
            detail: 7,
            ..J2kMctStatus::default()
        })
        .to_string();

        assert!(message.contains("color transform kernel decode failure"));
        assert!(!message.contains("inverse MCT"));
    }

    #[test]
    fn unexpected_shared_mct_status_is_transform_neutral() {
        let message = decode_mct_status_error(J2kMctStatus {
            code: u32::MAX,
            detail: 11,
            ..J2kMctStatus::default()
        })
        .to_string();

        assert!(message.contains("color transform kernel unexpected status"));
        assert!(!message.contains("inverse MCT"));
    }
}

#[cfg(test)]
mod retirement_tests {
    use core::mem::size_of;

    use super::{
        retire_direct_status_checks, DirectStatusCheck, DirectStatusRetirementMode, Error,
        J2kHtStatus, J2K_HT_STATUS_FAIL, J2K_HT_STATUS_OK,
    };
    use crate::compute::MetalRuntime;

    fn pooled_ht_status(runtime: &MetalRuntime, status: J2kHtStatus) -> DirectStatusCheck {
        let owner = runtime
            .take_shared_buffer(size_of::<J2kHtStatus>())
            .expect("pooled HT status allocation");
        // SAFETY: This fresh shared buffer is exclusively owned by `owner`,
        // and no GPU command can access it during this host-only regression.
        unsafe { j2k_metal_support::checked_buffer_write(owner.buffer(), 0, &[status]) }
            .expect("write HT status fixture");
        DirectStatusCheck::Ht {
            buffer: owner.buffer().clone(),
            len: 1,
            source_indices: Some(vec![0]),
            recyclable_status: Some(owner),
        }
    }

    #[test]
    fn failed_first_status_still_retires_later_pooled_status() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }

        let runtime = MetalRuntime::new().expect("Metal runtime");
        let first = pooled_ht_status(
            &runtime,
            J2kHtStatus {
                code: J2K_HT_STATUS_FAIL,
                detail: 41,
                ..J2kHtStatus::default()
            },
        );
        let second = pooled_ht_status(
            &runtime,
            J2kHtStatus {
                code: J2K_HT_STATUS_OK,
                ..J2kHtStatus::default()
            },
        );

        let error = retire_direct_status_checks(
            &runtime,
            vec![first, second],
            DirectStatusRetirementMode::Validate,
        )
        .expect_err("first status must fail validation");
        assert!(matches!(
            error,
            Error::MetalKernel { message } if message.contains("detail=41")
        ));
        assert_eq!(
            runtime
                .buffer_pool_diagnostics()
                .expect("pool diagnostics")
                .shared
                .cached_buffers,
            2,
            "both pooled statuses must be retired despite the first validation failure"
        );
    }

    #[test]
    fn recycle_without_read_retires_status_with_unreadable_metadata() {
        if !j2k_test_support::metal_runtime_gate(module_path!()) {
            return;
        }

        let runtime = MetalRuntime::new().expect("Metal runtime");
        let mut status = pooled_ht_status(
            &runtime,
            J2kHtStatus {
                code: J2K_HT_STATUS_OK,
                ..J2kHtStatus::default()
            },
        );
        let DirectStatusCheck::Ht { len, .. } = &mut status else {
            unreachable!("fixture is an HT status")
        };
        *len = usize::MAX;

        retire_direct_status_checks(
            &runtime,
            core::iter::once(status),
            DirectStatusRetirementMode::RecycleWithoutRead,
        )
        .expect("command-failure retirement must not read status contents");
        assert_eq!(
            runtime
                .buffer_pool_diagnostics()
                .expect("pool diagnostics")
                .shared
                .cached_buffers,
            1
        );
    }
}
