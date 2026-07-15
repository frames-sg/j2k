// SPDX-License-Identifier: MIT OR Apache-2.0

use burn_core::tensor::backend::Backend;
#[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
use burn_flex::{Flex as CpuBackend, FlexDevice as CpuDevice};
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
use burn_ndarray::NdArrayDevice::Cpu as CpuDevice;
use criterion::{BenchmarkId, Criterion, Throughput};
use j2k::{DeviceDecodeRequest, Downscale, Rect};
use j2k_ml::{cpu, TensorDecodeOptions, TensorInput};
#[cfg(all(feature = "cuda", not(target_os = "macos")))]
use j2k_test_support::htj2k_gray8_large_fixture;
use j2k_test_support::{classic_j2k_gray8_fixture, openhtj2k_refinement_fixture};

#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
type CpuBackend = burn_ndarray::NdArray<f32, i64, i8>;

const BATCH_SIZES: &[usize] = &[1, 8, 32];
#[cfg(not(all(target_arch = "aarch64", target_os = "linux")))]
const CPU_STAGED_BENCHMARK_GROUP: &str = "j2k_ml_decode_to_ready_tensor_cpu_staged_flex";
#[cfg(all(target_arch = "aarch64", target_os = "linux"))]
const CPU_STAGED_BENCHMARK_GROUP: &str =
    "j2k_ml_decode_to_ready_tensor_cpu_staged_ndarray_arm_linux";

fn main() {
    let mut criterion = Criterion::default().configure_from_args();
    bench_cpu_staged(&mut criterion);
    #[cfg(all(feature = "cuda", not(target_os = "macos")))]
    bench_cuda_routes(&mut criterion);
    #[cfg(all(feature = "metal", target_os = "macos"))]
    bench_metal_staged(&mut criterion);
    criterion.final_summary();
}

fn bench_cpu_staged(criterion: &mut Criterion) {
    let gray8 = classic_j2k_gray8_fixture(128, 128);
    let gray16 = openhtj2k_refinement_fixture();
    let options = TensorDecodeOptions::default();
    let mut group = criterion.benchmark_group(CPU_STAGED_BENCHMARK_GROUP);

    for (label, encoded) in [("gray8", gray8.as_slice()), ("gray16", gray16)] {
        let item_bytes = compact_item_bytes(encoded);
        for &batch_size in BATCH_SIZES {
            let inputs = repeated_inputs(encoded, batch_size, DeviceDecodeRequest::Full);
            let transferred = batch_bytes(item_bytes, batch_size);
            group.throughput(Throughput::Bytes(transferred));
            group.bench_with_input(
                BenchmarkId::new(format!("{label}/compact_upload_bytes"), transferred),
                &inputs,
                |bencher, inputs| {
                    bencher.iter(|| {
                        let decoded = cpu::decode_float_batch::<CpuBackend>(
                            std::hint::black_box(inputs),
                            &options,
                            &CpuDevice,
                        )
                        .expect("CPU-staged tensor decode");
                        CpuBackend::sync(&CpuDevice).expect("sync CPU backend");
                        std::hint::black_box(decoded.tensor)
                    });
                },
            );
        }
    }

    let roi = Rect {
        x: 32,
        y: 32,
        w: 64,
        h: 64,
    };
    let request = DeviceDecodeRequest::RegionScaled {
        roi,
        scale: Downscale::Half,
    };
    let inputs = repeated_inputs(&gray8, 8, request);
    let compact_bytes = 8_u64 * 32 * 32;
    group.throughput(Throughput::Bytes(compact_bytes));
    group.bench_function("roi_tiles_8/compact_upload_bytes_8192", |bencher| {
        bencher.iter(|| {
            let decoded = cpu::decode_float_batch::<CpuBackend>(
                std::hint::black_box(&inputs),
                &options,
                &CpuDevice,
            )
            .expect("CPU-staged ROI batch");
            CpuBackend::sync(&CpuDevice).expect("sync CPU backend");
            std::hint::black_box(decoded.tensor)
        });
    });
    group.finish();
}

#[cfg(all(feature = "cuda", not(target_os = "macos")))]
fn bench_cuda_routes(criterion: &mut Criterion) {
    use burn_cuda::{Cuda, CudaDevice};
    use j2k_ml::cuda;

    type CudaBackend = Cuda;

    let gray8 = htj2k_gray8_large_fixture(128, 128);
    let gray16 = openhtj2k_refinement_fixture();
    let options = TensorDecodeOptions::default();
    let device = CudaDevice::default();
    let mut group = criterion.benchmark_group("j2k_ml_decode_to_ready_tensor_cuda");

    for (label, encoded) in [("gray8", gray8.as_slice()), ("gray16", gray16)] {
        let item_bytes = compact_item_bytes(encoded);
        for &batch_size in BATCH_SIZES {
            let inputs = repeated_inputs(encoded, batch_size, DeviceDecodeRequest::Full);
            let transferred = batch_bytes(item_bytes, batch_size);
            group.throughput(Throughput::Bytes(transferred));
            group.bench_with_input(
                BenchmarkId::new(
                    format!("{label}/staged_baseline_compact_upload_bytes"),
                    transferred,
                ),
                &inputs,
                |bencher, inputs| {
                    bencher.iter(|| {
                        let decoded = cpu::decode_float_batch::<CudaBackend>(
                            std::hint::black_box(inputs),
                            &options,
                            &device,
                        )
                        .expect("CUDA-staged baseline");
                        CudaBackend::sync(&device).expect("sync CUDA");
                        std::hint::black_box(decoded.tensor)
                    });
                },
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{label}/direct_transferred_bytes_0"), batch_size),
                &inputs,
                |bencher, inputs| {
                    bencher.iter(|| {
                        let decoded = cuda::decode_float_batch(
                            std::hint::black_box(inputs),
                            &options,
                            &device,
                        )
                        .expect("CUDA-direct tensor decode");
                        CudaBackend::sync(&device).expect("sync CUDA");
                        std::hint::black_box(decoded.tensor)
                    });
                },
            );
        }
    }

    let roi_request = DeviceDecodeRequest::RegionScaled {
        roi: Rect {
            x: 32,
            y: 32,
            w: 64,
            h: 64,
        },
        scale: Downscale::Half,
    };
    let roi_inputs = repeated_inputs(&gray8, 8, roi_request);
    let roi_bytes = 8_u64 * 32 * 32;
    group.throughput(Throughput::Bytes(roi_bytes));
    group.bench_function(
        "roi_tiles_8/staged_baseline_compact_upload_bytes_8192",
        |bencher| {
            bencher.iter(|| {
                let decoded =
                    cpu::decode_float_batch::<CudaBackend>(&roi_inputs, &options, &device)
                        .expect("CUDA ROI baseline");
                CudaBackend::sync(&device).expect("sync CUDA");
                std::hint::black_box(decoded.tensor)
            });
        },
    );
    group.bench_function("roi_tiles_8/direct_transferred_bytes_0", |bencher| {
        bencher.iter(|| {
            let decoded = cuda::decode_float_batch(&roi_inputs, &options, &device)
                .expect("CUDA-direct ROI decode");
            CudaBackend::sync(&device).expect("sync CUDA");
            std::hint::black_box(decoded.tensor)
        });
    });
    group.finish();
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn bench_metal_staged(criterion: &mut Criterion) {
    use burn_wgpu::{Wgpu, WgpuDevice};
    use j2k_ml::metal;

    let gray8 = classic_j2k_gray8_fixture(128, 128);
    let gray16 = openhtj2k_refinement_fixture();
    let options = TensorDecodeOptions::default();
    let device = WgpuDevice::DefaultDevice;
    let mut group = criterion.benchmark_group("j2k_ml_decode_to_ready_tensor_metal_staged");
    for (label, encoded) in [("gray8", gray8.as_slice()), ("gray16", gray16)] {
        let item_bytes = compact_item_bytes(encoded);
        for &batch_size in BATCH_SIZES {
            let inputs = repeated_inputs(encoded, batch_size, DeviceDecodeRequest::Full);
            let transferred = batch_bytes(item_bytes, batch_size)
                .checked_mul(2)
                .expect("benchmark transfer byte count fits u64");
            group.throughput(Throughput::Bytes(transferred));
            group.bench_with_input(
                BenchmarkId::new(
                    format!("{label}/packed_readback_plus_upload_bytes"),
                    transferred,
                ),
                &inputs,
                |bencher, inputs| {
                    bencher.iter(|| {
                        let decoded = metal::decode_float_batch(
                            std::hint::black_box(inputs),
                            &options,
                            &device,
                        )
                        .expect("Metal-staged tensor decode");
                        <Wgpu<f32, i32, u32> as Backend>::sync(&device).expect("sync Metal");
                        std::hint::black_box(decoded.tensor)
                    });
                },
            );
        }
    }

    let roi_inputs = repeated_inputs(
        &gray8,
        8,
        DeviceDecodeRequest::RegionScaled {
            roi: Rect {
                x: 32,
                y: 32,
                w: 64,
                h: 64,
            },
            scale: Downscale::Half,
        },
    );
    let roi_transferred = 8_u64 * 32 * 32 * 2;
    group.throughput(Throughput::Bytes(roi_transferred));
    group.bench_function(
        "roi_tiles_8/packed_readback_plus_upload_bytes_16384",
        |bencher| {
            bencher.iter(|| {
                let decoded = metal::decode_float_batch(&roi_inputs, &options, &device)
                    .expect("Metal-staged ROI tensor decode");
                <Wgpu<f32, i32, u32> as Backend>::sync(&device).expect("sync Metal");
                std::hint::black_box(decoded.tensor)
            });
        },
    );
    group.finish();
}

fn compact_item_bytes(encoded: &[u8]) -> u64 {
    let info = j2k::J2kDecoder::inspect(encoded).expect("inspect benchmark fixture");
    let sample_bytes = if info.bit_depth <= 8 { 1 } else { 2 };
    u64::from(info.dimensions.0) * u64::from(info.dimensions.1) * sample_bytes
}

fn batch_bytes(item_bytes: u64, batch_size: usize) -> u64 {
    item_bytes
        .checked_mul(u64::try_from(batch_size).expect("benchmark batch size fits u64"))
        .expect("benchmark transfer byte count fits u64")
}

fn repeated_inputs(
    encoded: &[u8],
    batch_size: usize,
    request: DeviceDecodeRequest,
) -> Vec<TensorInput<'_>> {
    (0..batch_size)
        .map(|_| TensorInput { encoded, request })
        .collect()
}
