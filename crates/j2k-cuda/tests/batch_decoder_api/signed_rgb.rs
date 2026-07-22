// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{
    prepare_batch, BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples,
    DecodeRequest, Downscale, EncodedImage, Rect,
};
use j2k_cuda::{CudaBatchDecoder, CudaSession};
use j2k_cuda_runtime::{CudaContext, CudaExternalDeviceBufferViewMut};
use j2k_test_support::OpenJphBatchFixture;

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "the CUDA-only matrix keeps independent-fixture, external, and resident parity together"
)]
fn signed_rgb_i16_external_and_resident_batches_match_cpu_for_geometry_and_layout_when_runtime_required(
) {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }

    let fixtures = j2k_test_support::openjph_batch_fixtures()
        .iter()
        .filter(|fixture| {
            matches!(
                fixture.name,
                "openjph-rgb-s8-53-single-raw"
                    | "openjph-rgb-s12-53-single-raw"
                    | "openjph-rgb-s16-53-single-raw"
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(fixtures.len(), 3);
    let requests = [
        DecodeRequest::Full,
        DecodeRequest::Region {
            roi: Rect {
                x: 2,
                y: 3,
                w: 9,
                h: 7,
            },
        },
        DecodeRequest::Reduced {
            scale: Downscale::Half,
        },
        DecodeRequest::RegionReduced {
            roi: Rect {
                x: 2,
                y: 4,
                w: 10,
                h: 8,
            },
            scale: Downscale::Half,
        },
    ];
    let context = CudaContext::system_default().expect("CUDA context");

    for layout in [BatchLayout::Nhwc, BatchLayout::Nchw] {
        let options = BatchDecodeOptions {
            layout,
            ..BatchDecodeOptions::default()
        };
        let session = CudaSession::with_context(context.clone());
        let mut decoder = CudaBatchDecoder::with_session_and_options(session, options);
        let mut cpu = CpuBatchDecoder::new(options);

        for fixture in &fixtures {
            let encoded = Arc::<[u8]>::from(fixture.encoded);
            for request in requests {
                let prepared = prepare_batch(
                    vec![EncodedImage::new(Arc::clone(&encoded), request)],
                    options,
                )
                .unwrap_or_else(|error| panic!("{} prepare: {error}", fixture.name));
                assert!(prepared.errors().is_empty(), "{}", fixture.name);
                let [prepared_group] = prepared.groups() else {
                    panic!("{}: expected one signed RGB group", fixture.name)
                };
                assert!(
                    prepared_group.images()[0].htj2k_plan().is_some(),
                    "{}",
                    fixture.name
                );

                let oracle = cpu
                    .decode_prepared(&prepared)
                    .unwrap_or_else(|error| panic!("{} CPU oracle: {error}", fixture.name));
                let expected_samples = match oracle.groups()[0].samples() {
                    CpuBatchSamples::I16(samples) => samples,
                    other => panic!(
                        "{}: unexpected signed RGB oracle type: {other:?}",
                        fixture.name
                    ),
                };
                if layout == BatchLayout::Nhwc && request == DecodeRequest::Full {
                    assert_eq!(
                        expected_samples,
                        &openjph_i16_oracle(fixture),
                        "{} independent OpenJPH oracle",
                        fixture.name
                    );
                }
                let expected = expected_samples
                    .iter()
                    .flat_map(|sample| sample.to_ne_bytes())
                    .collect::<Vec<_>>();

                let mut allocation = context
                    .allocate(expected.len())
                    .expect("signed RGB external destination");
                let submitted = {
                    let ptr = allocation.device_ptr();
                    let len = allocation.byte_len();
                    // SAFETY: the allocation remains live and inaccessible until
                    // the returned completion owner is waited below.
                    let mut destination = unsafe {
                        CudaExternalDeviceBufferViewMut::from_raw_parts(
                            &context,
                            ptr,
                            len,
                            std::mem::align_of::<i16>(),
                            &mut allocation,
                        )
                    }
                    .expect("signed RGB external view");
                    // SAFETY: the view owns the unique destination borrow through
                    // submission and completion is established before host read.
                    unsafe { decoder.submit_batch_into(prepared_group, &mut destination) }
                        .unwrap_or_else(|error| panic!("{} submit external: {error}", fixture.name))
                };
                submitted
                    .wait()
                    .unwrap_or_else(|error| panic!("{} wait external: {error}", fixture.name));
                let mut external = vec![0_u8; expected.len()];
                allocation
                    .copy_to_host(&mut external)
                    .expect("download signed RGB external output");
                assert_eq!(
                    external, expected,
                    "{} external {layout:?} {request:?}",
                    fixture.name
                );

                let resident = decoder
                    .decode_prepared(&prepared)
                    .unwrap_or_else(|error| panic!("{} resident: {error}", fixture.name));
                let [resident_group] = resident.groups() else {
                    panic!("{}: expected one signed RGB resident group", fixture.name)
                };
                let dense = resident_group.dense_output();
                let mut resident_bytes = vec![0_u8; expected.len()];
                dense
                    .buffer()
                    .copy_range_to_host(dense.ranges()[0].offset, &mut resident_bytes)
                    .expect("download signed RGB resident output");
                assert_eq!(
                    resident_bytes, expected,
                    "{} resident {layout:?} {request:?}",
                    fixture.name
                );
            }
        }
    }
}

fn openjph_i16_oracle(fixture: &OpenJphBatchFixture) -> Vec<i16> {
    if fixture.precision <= 8 {
        fixture
            .oracle
            .iter()
            .map(|sample| i16::from(i8::from_ne_bytes([*sample])))
            .collect()
    } else {
        fixture
            .oracle
            .chunks_exact(2)
            .map(|sample| i16::from_le_bytes([sample[0], sample[1]]))
            .collect()
    }
}
