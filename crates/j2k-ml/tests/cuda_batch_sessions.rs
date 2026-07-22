// SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "cuda")]

use std::sync::Arc;

use burn_cuda::CudaDevice;
use j2k::{prepare_batch, BatchDecodeOptions, BatchItemError, DecodeSettings, EncodedImage};
use j2k_ml::{BurnBatchTensor, BurnDecodeError, CudaBurnDecoder};
use j2k_native::{encode, EncodeOptions};
use j2k_test_support::{cuda_runtime_and_strict_oxide_gate, htj2k_gray8_large_fixture};

fn unsupported_classic_roi_rgb() -> Arc<[u8]> {
    let pixels = (0..4_u8)
        .flat_map(|index| [index * 17, index * 29 + 3, index * 41 + 5])
        .collect::<Vec<_>>();
    Arc::from(
        encode(
            &pixels,
            2,
            2,
            3,
            8,
            false,
            &EncodeOptions {
                reversible: true,
                num_decomposition_levels: 1,
                roi_component_shifts: vec![3, 0, 0],
                ..EncodeOptions::default()
            },
        )
        .expect("encode classic RGB8 with unsupported RGN maxshift"),
    )
}

#[test]
fn cuda_burn_decoder_construction_is_infallible_and_lazy() {
    let constructor: fn(CudaDevice, BatchDecodeOptions) -> CudaBurnDecoder = CudaBurnDecoder::new;
    let decoder = constructor(CudaDevice::new(0), BatchDecodeOptions::default());

    assert_eq!(decoder.codec().session().submissions(), 0);
}

#[test]
fn empty_cuda_batch_uses_the_persistent_shared_codec_contract_without_initializing_work() {
    let mut decoder = CudaBurnDecoder::new(CudaDevice::new(0), BatchDecodeOptions::default());

    let prepared = decoder
        .prepare(Vec::<EncodedImage>::new())
        .expect("prepare empty shared batch");
    let submitted = decoder
        .submit_prepared(&prepared)
        .expect("submit empty shared batch");
    assert!(submitted.is_empty());
    let output = submitted.wait().expect("finish empty shared batch");

    assert!(output.groups.is_empty());
    assert!(output.errors.is_empty());
    assert_eq!(decoder.codec().session().submissions(), 0);
}

#[test]
fn cuda_burn_regroups_prepared_images_and_keeps_settings_failures_indexed_without_cuda() {
    let lenient_options = BatchDecodeOptions {
        settings: DecodeSettings::lenient(),
        ..BatchDecodeOptions::default()
    };
    let prepared = prepare_batch(
        vec![EncodedImage::full(Arc::from(
            j2k_test_support::htj2k_gray8_large_fixture(4, 4),
        ))],
        lenient_options,
    )
    .expect("prepare lenient CUDA Burn input");
    let image = prepared.groups()[0].images()[0].clone();
    let mut decoder = CudaBurnDecoder::new(CudaDevice::new(0), BatchDecodeOptions::default());

    let regrouped = decoder
        .prepare_prepared_images(vec![image.clone()])
        .expect("settings mismatch remains indexed preflight data");
    assert!(regrouped.groups().is_empty());
    assert_eq!(regrouped.errors()[0].index, 0);
    assert!(matches!(
        regrouped.errors()[0].source,
        BatchItemError::PreparedDecodeSettingsMismatch {
            prepared,
            requested,
        } if prepared == DecodeSettings::lenient() && requested == DecodeSettings::strict()
    ));

    let output = decoder
        .decode_prepared_images(vec![image])
        .expect("preflight-only batch must not initialize CUDA");
    assert!(output.groups.is_empty());
    assert_eq!(output.errors[0].index, 0);
    assert_eq!(decoder.codec().session().submissions(), 0);
}

#[test]
fn dropping_submitted_burn_batch_retires_cuda_work_and_keeps_session_reusable() {
    if !cuda_runtime_and_strict_oxide_gate("j2k-ml CUDA submitted-batch drop reuse") {
        return;
    }
    let encoded = Arc::<[u8]>::from(htj2k_gray8_large_fixture(8, 8));
    let mut decoder = CudaBurnDecoder::new(CudaDevice::default(), BatchDecodeOptions::default());

    let submitted = decoder
        .submit(vec![EncodedImage::full(Arc::clone(&encoded))])
        .expect("submit CUDA Burn batch to drop");
    assert_eq!(submitted.len(), 1);
    drop(submitted);

    let output = decoder
        .decode(vec![EncodedImage::full(encoded)])
        .expect("reuse CUDA Burn decoder after dropped submission");
    assert!(output.errors.is_empty());
    let [group] = output.groups.as_slice() else {
        panic!("expected one decoded CUDA Burn group")
    };
    let BurnBatchTensor::U8(tensor) = &group.tensor else {
        panic!("expected native U8 CUDA Burn tensor")
    };
    assert_eq!(tensor.dims(), [1, 1, 8, 8]);
    assert!(decoder.codec().session().submissions() >= 2);
}

#[test]
fn burn_direct_session_reuses_events_and_codec_memory_for_one_thousand_batches() {
    if !cuda_runtime_and_strict_oxide_gate("j2k-ml CUDA direct-write session soak") {
        return;
    }
    let encoded = Arc::<[u8]>::from(htj2k_gray8_large_fixture(8, 8));
    let mut decoder = CudaBurnDecoder::new(CudaDevice::default(), BatchDecodeOptions::default());
    let prepared = decoder
        .prepare(vec![EncodedImage::full(encoded)])
        .expect("prepare reusable CUDA Burn batch");

    for _ in 0..16 {
        drop(
            decoder
                .decode_prepared(&prepared)
                .expect("warm CUDA Burn direct-write session"),
        );
    }
    let warm = decoder
        .codec()
        .diagnostics()
        .expect("warm CUDA Burn codec diagnostics");
    let warm_runtime = warm.runtime.expect("CUDA Burn runtime is initialized");
    assert!(warm.pools.retained_bytes() > 0);

    for _ in 0..1_000 {
        drop(
            decoder
                .decode_prepared(&prepared)
                .expect("reuse CUDA Burn direct-write session"),
        );
    }

    let after = decoder
        .codec()
        .diagnostics()
        .expect("post-soak CUDA Burn codec diagnostics");
    let after_runtime = after
        .runtime
        .expect("CUDA Burn runtime remains initialized");
    assert_eq!(after.pools.retained_bytes(), warm.pools.retained_bytes());
    assert_eq!(
        after.pools.peak_retained_bytes_upper_bound(),
        warm.pools.peak_retained_bytes_upper_bound()
    );
    assert_eq!(
        after_runtime.status_device_to_host_operations
            - warm_runtime.status_device_to_host_operations,
        1_000,
        "each Burn-direct group must use one status readback"
    );
    assert_eq!(
        after_runtime.device_to_host_operations - warm_runtime.device_to_host_operations,
        1_000,
        "Burn-direct decode must not download decoded pixels"
    );
    assert_eq!(
        after_runtime.device_to_host_bytes - warm_runtime.device_to_host_bytes,
        after_runtime.status_device_to_host_bytes - warm_runtime.status_device_to_host_bytes
    );
    assert_eq!(
        after_runtime.event_driver_allocations, warm_runtime.event_driver_allocations,
        "Burn stream-bridge events must stabilize after warmup"
    );
    assert!(after_runtime.event_reuses > warm_runtime.event_reuses);
    assert_eq!(
        after_runtime.event_host_synchronizations, warm_runtime.event_host_synchronizations,
        "Burn-direct completion must use the group status boundary"
    );
    assert_eq!(
        after_runtime.context_host_synchronizations, warm_runtime.context_host_synchronizations,
        "Burn-direct completion must not synchronize the whole context"
    );
}

#[test]
fn cuda_burn_batch_continues_after_one_group_submit_failure() {
    if !cuda_runtime_and_strict_oxide_gate("j2k-ml CUDA group submit continuation") {
        return;
    }
    let valid_gray = Arc::<[u8]>::from(htj2k_gray8_large_fixture(8, 8));
    let mut decoder = CudaBurnDecoder::new(CudaDevice::default(), BatchDecodeOptions::default());
    let prepared = decoder
        .prepare(vec![
            EncodedImage::full(unsupported_classic_roi_rgb()),
            EncodedImage::full(valid_gray),
        ])
        .expect("prepare two homogeneous CUDA groups");
    assert_eq!(prepared.groups().len(), 2);

    let submitted = decoder
        .submit_prepared(&prepared)
        .expect("unsupported group must remain a result-level failure");
    assert_eq!(submitted.len(), 1);
    let output = submitted.wait().expect("finish supported CUDA group");

    assert!(output.errors.is_empty());
    assert_eq!(output.groups.len(), 1);
    assert_eq!(output.groups[0].source_indices, [1]);
    assert_eq!(output.group_errors.len(), 1);
    assert_eq!(output.group_errors[0].source_indices(), &[0]);
    assert!(matches!(
        output.group_errors[0].source(),
        BurnDecodeError::Cuda(j2k_cuda::CudaBatchError::GroupExecution { .. })
    ));
}
