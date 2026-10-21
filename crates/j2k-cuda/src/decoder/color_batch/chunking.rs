// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bounded colour batch groups shared by the surface and owned CUDA paths.

use crate::decoder::Error;

// The batch scratch pool retains at most 2,048 buffers. The current 512x512
// RGB8 path uses about 1,250 scratch buffers for 32 images, while larger
// all-at-once batches exceed the retention ceiling and repeatedly enter CUDA
// allocation/free paths. Keep a conservative amount of headroom for other
// color geometries. This bounds the working set; it is not an optimal-size
// claim for every GPU or codestream.
pub(in crate::decoder) const COLOR_BATCH_CHUNK_IMAGES: usize = 32;

/// Rebase a group-local colour decode error to whole-batch indices.
pub(in crate::decoder) fn offset_color_batch_error(
    error: Error,
    source_index_base: usize,
    ht_job_index_base: usize,
    classic_job_index_base: usize,
) -> Error {
    match error {
        Error::CudaTier1JobFailed {
            source_index,
            original_job_index,
            source,
        } => {
            let job_index_base = if cuda_error_is_classic_tier1(&source) {
                classic_job_index_base
            } else {
                ht_job_index_base
            };
            Error::CudaTier1JobFailed {
                source_index: source_index.saturating_add(source_index_base),
                original_job_index: original_job_index.saturating_add(job_index_base),
                source,
            }
        }
        Error::CudaCleanupFailed { primary, cleanup } => Error::CudaCleanupFailed {
            primary: Box::new(offset_color_batch_error(
                *primary,
                source_index_base,
                ht_job_index_base,
                classic_job_index_base,
            )),
            cleanup: Box::new(offset_color_batch_error(
                *cleanup,
                source_index_base,
                ht_job_index_base,
                classic_job_index_base,
            )),
        },
        error => error,
    }
}

fn cuda_error_is_classic_tier1(error: &j2k_cuda_runtime::CudaError) -> bool {
    match error {
        j2k_cuda_runtime::CudaError::KernelJobStatus { kernel, .. } => {
            *kernel == "j2k_decode_classic_codeblocks_multi"
        }
        j2k_cuda_runtime::CudaError::CompletionFailed { primary, .. }
        | j2k_cuda_runtime::CudaError::ResourceReleaseFailed { primary, .. } => {
            cuda_error_is_classic_tier1(primary)
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::{offset_color_batch_error, COLOR_BATCH_CHUNK_IMAGES};

    #[test]
    fn chunked_color_error_rebases_source_and_job_indices() {
        let error = offset_color_batch_error(
            crate::Error::CudaTier1JobFailed {
                source_index: 3,
                original_job_index: 5,
                source: j2k_cuda_runtime::CudaError::InvalidArgument {
                    message: "test failure".to_owned(),
                },
            },
            COLOR_BATCH_CHUNK_IMAGES,
            96,
            7,
        );

        assert!(matches!(
            error,
            crate::Error::CudaTier1JobFailed {
                source_index: 35,
                original_job_index: 101,
                ..
            }
        ));

        let classic = offset_color_batch_error(
            crate::Error::CudaTier1JobFailed {
                source_index: 3,
                original_job_index: 5,
                source: j2k_cuda_runtime::CudaError::KernelJobStatus {
                    kernel: "j2k_decode_classic_codeblocks_multi",
                    job_index: 5,
                    code: 1,
                    detail: 2,
                },
            },
            COLOR_BATCH_CHUNK_IMAGES,
            96,
            7,
        );
        assert!(matches!(
            classic,
            crate::Error::CudaTier1JobFailed {
                source_index: 35,
                original_job_index: 12,
                ..
            }
        ));
    }
}
