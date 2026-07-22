// SPDX-License-Identifier: MIT OR Apache-2.0

//! Owned, reusable batch contracts for native-width JPEG 2000 decode.

use alloc::{sync::Arc, vec::Vec};
use core::{fmt, mem::size_of, num::NonZeroUsize, ops::Range, sync::atomic::Ordering};
use std::sync::Mutex;

use j2k_core::{BatchInfrastructureError, Colorspace, CompressedTransferSyntax, Downscale, Rect};

pub use j2k_core::{PixelLayout as BatchColor, SampleType as NativeSampleType};

use crate::{
    backend,
    batch::{
        allocation::{J2K_BATCH_METADATA_ALLOWANCE_BYTES, MAX_GENERIC_BATCH_WORKERS},
        scheduler::run_retained_chunks,
        worker::BatchWorker,
    },
    decode::decode_warnings_for_settings,
    parse::{extract_j2k_codestream_payload, parse_image_info},
    CpuDecodeParallelism, DecodeSettings, DeviceDecodePlan, J2kDecodeWarning, J2kError,
    J2kSupportInfo, TileBatchOptions,
};

mod contracts;
pub use self::contracts::{
    BatchAlpha, BatchCodecRoute, BatchDecodeOptions, BatchGroupInfo, BatchLayout,
    BatchWaveletTransform, DecodeRequest, EncodedImage, PreparationDepth,
};
use self::contracts::{PrepareImageResult, PrepareJob};

mod errors;
pub use self::errors::{
    BatchErrorStage, BatchItemError, IndexedBatchError, NonRepresentableReason,
};

mod prepared;
pub use self::prepared::{BatchDecoder, PreparedBatch, PreparedBatchGroup, PreparedImage};
use self::prepared::{BatchExecutionShape, PreparedCodecPlan, PreparedImageInner};

mod prepared_plan;
pub use self::prepared_plan::{
    ClassicCodeBlockPayload, Htj2kPayloadRanges, J2kCodestreamRange, PreparedClassicPlan,
    PreparedHtj2kPlan,
};
mod cpu_prepared;
use self::cpu_prepared::{
    decode_prepared_plan_samples, execute_staged_entropy_job, finish_staged_plan_samples,
    finish_staged_tile, prepare_staged_entropy_worker, prepare_staged_image, prepare_staged_tile,
    staged_tile_count,
};
mod cpu_fast;
use self::cpu_fast::CpuGroupFastWorkspace;
mod cpu_staged;
use self::cpu_staged::{CpuEntropyOutcome, CpuStagedWorkspace, CPU_ENTROPY_IMAGES_PER_WINDOW};
mod cpu_execute;
use self::cpu_execute::decode_cpu_group;
mod cpu;
mod cpu_materialize;
mod cpu_staged_execute;
pub use self::cpu::{CpuBatchDecodeResult, CpuBatchDecoder, CpuBatchGroup, CpuBatchSamples};
mod cpu_diagnostics;
pub use self::cpu_diagnostics::CpuBatchWorkspaceStats;
mod prepare;
use self::prepare::prepare_batch_with_workers;
#[cfg(test)]
use self::prepare::prepare_staging_bytes;
pub use self::prepare::{prepare_batch, prepare_batch_from_images};

#[cfg(test)]
mod tests {
    use super::{prepare_staging_bytes, J2K_BATCH_METADATA_ALLOWANCE_BYTES};

    #[test]
    fn preparation_staging_size_rejects_overflowing_input_count() {
        assert!(prepare_staging_bytes(usize::MAX).is_err());
    }

    #[test]
    fn preparation_staging_size_accepts_its_exact_logical_boundary() {
        let bytes_per_input = prepare_staging_bytes(1).expect("one preparation item fits");
        let exact_count = J2K_BATCH_METADATA_ALLOWANCE_BYTES / bytes_per_input;

        assert!(prepare_staging_bytes(exact_count).is_ok());
        assert!(prepare_staging_bytes(exact_count + 1).is_err());
    }
}
