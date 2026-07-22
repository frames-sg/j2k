// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

#[cfg(not(feature = "cuda-runtime"))]
use j2k::DecodeRequest;
use j2k::{
    prepare_batch, BatchDecodeOptions, BatchDecoder, BatchItemError, DecodeSettings, EncodedImage,
};
#[cfg(not(feature = "cuda-runtime"))]
use j2k_cuda::Error;
use j2k_cuda::{CudaBatchDecodeResult, CudaBatchDecoder, CudaBatchError};

#[test]
fn persistent_batch_decoder_accepts_an_empty_batch_without_cuda() {
    let mut decoder = CudaBatchDecoder::new();

    let output = decoder
        .decode_batch(Vec::new())
        .expect("empty batch must not initialize CUDA");

    assert!(output.groups().is_empty());
    assert!(output.errors().is_empty());
    assert!(output.group_errors().is_empty());
    assert_eq!(decoder.session().submissions(), 0);
}

#[test]
fn cuda_prepared_image_regrouping_reports_indexed_settings_mismatch_without_cuda() {
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
    .expect("prepare lenient CUDA input");
    let image = prepared.groups()[0].images()[0].clone();
    let mut decoder = CudaBatchDecoder::with_options(BatchDecodeOptions::default());

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
        .expect("all-preflight-error batch must not initialize CUDA");
    assert!(output.groups().is_empty());
    assert_eq!(output.errors()[0].index, 0);
    assert_eq!(decoder.session().submissions(), 0);
}

#[cfg(feature = "cuda-runtime")]
#[test]
fn asynchronous_resident_decoder_accepts_an_empty_batch_without_cuda() {
    let mut decoder = CudaBatchDecoder::new();

    let submitted = decoder
        .submit_batch(Vec::new())
        .expect("empty submission must not initialize CUDA");
    assert_eq!(submitted.pending_group_count(), 0);
    assert!(submitted.is_complete().expect("query empty completion"));

    let output = submitted.wait().expect("wait empty submission");
    assert!(output.groups().is_empty());
    assert!(output.errors().is_empty());
    assert!(output.group_errors().is_empty());
    assert_eq!(decoder.session().submissions(), 0);
}

#[test]
fn persistent_batch_decoder_implements_shared_prepared_contract() {
    fn decode_through_trait<D>(
        decoder: &mut D,
        prepared: &j2k::PreparedBatch,
    ) -> Result<CudaBatchDecodeResult, CudaBatchError>
    where
        D: BatchDecoder<Output = CudaBatchDecodeResult, Error = CudaBatchError>,
    {
        decoder.decode_prepared(prepared)
    }

    let prepared = prepare_batch(Vec::new(), BatchDecodeOptions::default())
        .expect("prepare empty shared batch");
    let mut decoder = CudaBatchDecoder::new();
    let output = decode_through_trait(&mut decoder, &prepared)
        .expect("empty prepared batch must not initialize CUDA");

    assert!(output.groups().is_empty());
    assert!(output.errors().is_empty());
    assert!(output.group_errors().is_empty());
    assert_eq!(decoder.session().submissions(), 0);
}

#[cfg(not(feature = "cuda-runtime"))]
#[test]
fn nonempty_batch_fails_closed_without_cuda_runtime() {
    let pixels = (0_u8..64).collect::<Vec<_>>();
    let encoded = j2k_native::encode_htj2k(
        &pixels,
        8,
        8,
        1,
        8,
        false,
        &j2k_native::EncodeOptions {
            reversible: true,
            num_decomposition_levels: 1,
            ..j2k_native::EncodeOptions::default()
        },
    )
    .expect("encode valid HTJ2K batch fixture");
    let input = EncodedImage::new(Arc::from(encoded), DecodeRequest::Full);
    let mut decoder = CudaBatchDecoder::new();

    let error = decoder
        .decode_batch(vec![input])
        .expect_err("nonempty CUDA batch requires the CUDA runtime feature");
    let CudaBatchError::GroupExecution { source, .. } = error else {
        panic!("expected group execution error")
    };
    assert!(matches!(*source, Error::CudaUnavailable));
}
