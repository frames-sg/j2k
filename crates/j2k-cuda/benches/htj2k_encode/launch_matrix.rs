// SPDX-License-Identifier: MIT OR Apache-2.0

use criterion::{Criterion, SamplingMode, Throughput};
use j2k::{
    encode_j2k_lossless_with_accelerator, EncodeBackendPreference, J2kBlockCodingMode,
    J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples,
};
use j2k_core::BackendKind;
use j2k_cuda::CudaEncodeStageAccelerator;
use j2k_native::{DecodeSettings, Image};
use sha2::{Digest, Sha256};

#[derive(Clone, Copy)]
struct Workload {
    width: u32,
    height: u32,
    components: u16,
    batch: usize,
}

fn fixtures(workload: Workload) -> Vec<Vec<u8>> {
    (0..workload.batch)
        .map(|index| {
            let mut pixels = if workload.components == 1 {
                j2k_test_support::patterned_gray8(workload.width, workload.height)
            } else {
                j2k_test_support::patterned_rgb8(workload.width, workload.height)
            };
            pixels[0] = pixels[0].wrapping_add(u8::try_from(index).expect("bounded batch"));
            pixels
        })
        .collect()
}

fn framed_sha256(items: &[Vec<u8>]) -> String {
    let mut hash = Sha256::new();
    for item in items {
        hash.update(
            u64::try_from(item.len())
                .expect("fixture length fits u64")
                .to_le_bytes(),
        );
        hash.update(item);
    }
    format!("{:x}", hash.finalize())
}

fn encode_batch(
    workload: Workload,
    pixels: &[Vec<u8>],
    accelerator: &mut CudaEncodeStageAccelerator,
    options: &J2kLosslessEncodeOptions,
) -> Vec<Vec<u8>> {
    pixels
        .iter()
        .map(|pixels| {
            let samples = J2kLosslessSamples::new(
                pixels,
                workload.width,
                workload.height,
                workload.components,
                8,
                false,
            )
            .expect("launch matrix samples");
            let encoded = encode_j2k_lossless_with_accelerator(
                samples,
                options,
                BackendKind::Cuda,
                accelerator,
            )
            .expect("launch matrix CUDA encode");
            assert_eq!(encoded.backend, BackendKind::Cuda);
            encoded.codestream
        })
        .collect()
}

fn verify(codestreams: &[Vec<u8>], pixels: &[Vec<u8>]) {
    assert_eq!(codestreams.len(), pixels.len());
    for (codestream, expected) in codestreams.iter().zip(pixels) {
        let decoded = Image::new(codestream, &DecodeSettings::default())
            .expect("launch matrix codestream parses")
            .decode_native()
            .expect("launch matrix independent CPU decode");
        assert_eq!(&decoded.data, expected, "launch matrix exact pixel parity");
    }
}

pub(super) fn bench(criterion: &mut Criterion) {
    let options = J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(3))
        .with_validation(J2kEncodeValidation::External);
    for (width, height, components, batch) in [
        (128, 128, 3, 1),
        (512, 512, 1, 1),
        (512, 512, 3, 1),
        (640, 480, 3, 1),
        (1024, 1024, 3, 1),
        (512, 512, 3, 16),
    ] {
        let workload = Workload {
            width,
            height,
            components,
            batch,
        };
        let pixels = fixtures(workload);
        let mut accelerator = CudaEncodeStageAccelerator::default();
        let first = encode_batch(workload, &pixels, &mut accelerator, &options);
        let second = encode_batch(workload, &pixels, &mut accelerator, &options);
        verify(&first, &pixels);
        verify(&second, &pixels);
        assert_eq!(first, second, "launch matrix deterministic codestreams");
        let id =
            format!("j2k_cuda_ht_encode_launch_product/{width}x{height}/c{components}/b{batch}");
        eprintln!(
            "{id} input_sha256={} output_sha256={} exact_parity=true",
            framed_sha256(&pixels),
            framed_sha256(&first)
        );
        let mut group = criterion.benchmark_group(id);
        group.sampling_mode(SamplingMode::Flat);
        group.throughput(Throughput::Bytes(
            u64::try_from(pixels.iter().map(Vec::len).sum::<usize>()).expect("bounded pixels"),
        ));
        group.bench_function("encode", |bencher| {
            bencher.iter(|| {
                std::hint::black_box(encode_batch(
                    workload,
                    std::hint::black_box(&pixels),
                    &mut accelerator,
                    &options,
                ))
            });
        });
        group.finish();
    }
}
