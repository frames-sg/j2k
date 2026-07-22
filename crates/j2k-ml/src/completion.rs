// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{BurnBatchGroupError, BurnDecodeError};

pub(crate) fn finish_submitted_groups<G, O>(
    groups: Vec<G>,
    mut group_errors: Vec<BurnBatchGroupError>,
    mut finish: impl FnMut(G) -> (Vec<usize>, Result<O, BurnDecodeError>),
) -> Result<(Vec<O>, Vec<BurnBatchGroupError>), BurnDecodeError> {
    let mut outputs = Vec::new();
    outputs
        .try_reserve_exact(groups.len())
        .map_err(|_| BurnDecodeError::SizeOverflow)?;
    group_errors
        .try_reserve_exact(groups.len())
        .map_err(|_| BurnDecodeError::SizeOverflow)?;
    let mut fatal_error = None;
    for group in groups {
        let (source_indices, result) = finish(group);
        match result {
            Ok(output) => outputs.push(output),
            Err(source) if burn_group_error_is_fatal(&source) => {
                if fatal_error.is_none() {
                    fatal_error = Some(source);
                }
            }
            Err(source) => group_errors.push(BurnBatchGroupError::new(source_indices, source)),
        }
    }
    if let Some(error) = fatal_error {
        return Err(error);
    }
    group_errors.sort_by_key(|error| {
        error
            .source_indices()
            .first()
            .copied()
            .unwrap_or(usize::MAX)
    });
    Ok((outputs, group_errors))
}

pub(crate) fn burn_group_error_is_fatal(error: &BurnDecodeError) -> bool {
    match error {
        BurnDecodeError::Infrastructure(_) | BurnDecodeError::AcceleratorInterop { .. } => true,
        #[cfg(feature = "cuda")]
        BurnDecodeError::Cuda(source) => source.session_is_unusable(),
        #[cfg(feature = "metal")]
        BurnDecodeError::Metal(source) => source.session_is_unusable(),
        BurnDecodeError::UnsupportedDType { .. }
        | BurnDecodeError::SampleTypeMismatch
        | BurnDecodeError::SizeOverflow
        | BurnDecodeError::UnsupportedCodecContract => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{burn_group_error_is_fatal, finish_submitted_groups};
    use crate::BurnDecodeError;

    #[test]
    fn finishing_groups_retires_every_group_after_nonfatal_failure() {
        let groups = vec![
            (0, Ok(10)),
            (1, Err(BurnDecodeError::UnsupportedCodecContract)),
            (2, Ok(30)),
        ];
        let mut retired = Vec::new();

        let (outputs, errors) = finish_submitted_groups(groups, Vec::new(), |(index, result)| {
            retired.push(index);
            (vec![index], result)
        })
        .expect("a group-local failure must preserve other completed groups");

        assert_eq!(retired, [0, 1, 2]);
        assert_eq!(outputs, [10, 30]);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].source_indices(), [1]);
        assert!(matches!(
            errors[0].source(),
            BurnDecodeError::UnsupportedCodecContract
        ));
    }

    #[test]
    fn finishing_groups_retires_every_group_before_returning_fatal_failure() {
        let groups = vec![
            (
                0,
                Err(BurnDecodeError::AcceleratorInterop {
                    backend: "test",
                    message: "session ordering failed".to_string(),
                }),
            ),
            (1, Ok(10)),
        ];
        let mut retired = Vec::new();

        let result = finish_submitted_groups(groups, Vec::new(), |(index, result)| {
            retired.push(index);
            (vec![index], result)
        });

        assert_eq!(retired, [0, 1]);
        assert!(matches!(
            result,
            Err(BurnDecodeError::AcceleratorInterop {
                backend: "test",
                ..
            })
        ));
    }

    #[cfg(feature = "metal")]
    #[test]
    fn metal_group_fatality_delegates_to_codec_error_classification() {
        assert!(burn_group_error_is_fatal(&BurnDecodeError::Metal(
            j2k_metal::Error::MetalRuntime {
                message: "test runtime failure".to_string(),
            }
        )));
        assert!(!burn_group_error_is_fatal(&BurnDecodeError::Metal(
            j2k_metal::Error::MetalKernel {
                message: "test group status".to_string(),
            }
        )));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_group_fatality_delegates_to_codec_error_classification() {
        let recoverable = j2k_cuda::CudaBatchError::GroupExecution {
            source_indices: vec![2],
            source: Box::new(j2k_cuda::Error::HtJobChunkPlan(
                j2k_core::HtGpuJobChunkPlanError::InvalidCodingPassCount {
                    source_index: 2,
                    original_job_index: 4,
                    coding_passes: 0,
                },
            )),
        };
        assert!(!burn_group_error_is_fatal(&BurnDecodeError::Cuda(
            recoverable
        )));
        assert!(burn_group_error_is_fatal(&BurnDecodeError::Cuda(
            j2k_cuda::CudaBatchError::Infrastructure(
                j2k_core::BatchInfrastructureError::EmptyBatchPlan,
            )
        )));
    }
}
