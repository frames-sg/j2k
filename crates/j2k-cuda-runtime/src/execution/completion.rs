// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) use crate::error::select_uncertain_completion_error;
use crate::error::CudaError;

/// Whether CUDA established a resource-lifetime completion point.
///
/// Only `CUDA_SUCCESS` is treated as proof of completion. A returned error can
/// be a precondition failure that did not wait, an asynchronous device error,
/// or a fatal context error. The runtime therefore retains queued resources
/// and poisons further context use for every error instead of guessing which
/// allocations the driver may still reference.
pub(crate) enum CudaSynchronizationOutcome {
    Completed,
    CompletionUncertain(CudaError),
}

impl CudaSynchronizationOutcome {
    pub(crate) fn completion_established(&self) -> bool {
        matches!(self, Self::Completed)
    }

    pub(crate) fn into_result(self) -> Result<(), CudaError> {
        match self {
            Self::Completed => Ok(()),
            Self::CompletionUncertain(error) => Err(error),
        }
    }
}

#[cfg(test)]
mod uncertain_completion_error_tests {
    use super::*;

    fn driver_error(operation: &'static str, code: i32) -> CudaError {
        CudaError::Driver {
            operation,
            code,
            name: "CUDA_ERROR_TEST".to_string(),
        }
    }

    #[test]
    fn poisoned_context_preserves_primary_driver_error() {
        let selected = select_uncertain_completion_error(driver_error("cuLaunchKernel", 719), None);
        assert!(matches!(
            selected,
            CudaError::Driver {
                operation: "cuLaunchKernel",
                code: 719,
                ..
            }
        ));
    }

    #[test]
    fn poisoned_context_preserves_existing_state_poison_error() {
        let selected = select_uncertain_completion_error(
            CudaError::StatePoisoned {
                message: "original poison".to_string(),
            },
            None,
        );
        assert!(matches!(
            selected,
            CudaError::StatePoisoned { message } if message == "original poison"
        ));
    }

    #[test]
    fn poisoned_context_preserves_existing_compound_completion_error() {
        let compound = select_uncertain_completion_error(
            driver_error("cuLaunchKernel", 719),
            Some(driver_error("cuCtxSynchronize", 700)),
        );
        let selected = select_uncertain_completion_error(compound, None);

        assert!(matches!(selected, CudaError::CompletionFailed { .. }));
    }

    #[test]
    fn poisoned_context_wraps_primary_validation_error() {
        let selected = select_uncertain_completion_error(
            CudaError::InvalidArgument {
                message: "invalid job".to_string(),
            },
            None,
        );
        assert!(matches!(
            selected,
            CudaError::StatePoisoned { message }
                if message.contains("preceding operation failed")
                    && message.contains("invalid job")
        ));
    }

    #[test]
    fn synchronization_failure_preserves_both_operation_diagnostics() {
        let selected = select_uncertain_completion_error(
            driver_error("cuMemcpyHtoD_v2", 700),
            Some(driver_error("cuCtxSynchronize", 719)),
        );
        let CudaError::CompletionFailed {
            primary,
            completion,
        } = selected
        else {
            panic!("expected compound CUDA completion error");
        };
        assert!(matches!(
            *primary,
            CudaError::Driver {
                operation: "cuMemcpyHtoD_v2",
                code: 700,
                ..
            }
        ));
        assert!(matches!(
            *completion,
            CudaError::Driver {
                operation: "cuCtxSynchronize",
                code: 719,
                ..
            }
        ));

        let selected = select_uncertain_completion_error(
            CudaError::InvalidArgument {
                message: "validation failed".to_string(),
            },
            Some(driver_error("cuCtxSynchronize", 719)),
        );
        assert!(matches!(selected, CudaError::CompletionFailed { .. }));
    }
}
