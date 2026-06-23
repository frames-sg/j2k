// SPDX-License-Identifier: MIT OR Apache-2.0

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use j2k::{
    encode_j2k_lossless, EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation,
    J2kLosslessEncodeOptions, J2kLosslessSamples,
};
use j2k_core::BackendKind;
use j2k_cuda::encode_j2k_lossless_with_cuda;
use j2k_cuda_runtime::{
    CudaContext, CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeCodeBlockRegionJob,
    CudaHtj2kEncodeTables, CudaHtj2kEncodedCodeBlocks,
};
use j2k_native::{
    encode_ht_code_block_scalar, ht_uvlc_encode_table, ht_vlc_encode_table0, ht_vlc_encode_table1,
};

const TILE_DIM: u32 = 512;
const CODE_BLOCK_DIM: u32 = 64;
const CODE_BLOCK_BATCH: usize = 64;
const REGION_BLOCKS_X: u32 = 8;
const REGION_BLOCKS_Y: u32 = 8;

fn bench_htj2k_encode(c: &mut Criterion) {
    let pixels = generate_gray_tile(TILE_DIM, TILE_DIM);
    let coefficients =
        generate_codeblock_coefficients(CODE_BLOCK_DIM, CODE_BLOCK_DIM, CODE_BLOCK_BATCH);
    let jobs = contiguous_jobs(CODE_BLOCK_DIM, CODE_BLOCK_DIM, CODE_BLOCK_BATCH);
    let region_width = CODE_BLOCK_DIM * REGION_BLOCKS_X;
    let region_height = CODE_BLOCK_DIM * REGION_BLOCKS_Y;
    let region_coefficients = generate_region_coefficients(region_width, region_height);
    let region_jobs = strided_region_jobs(CODE_BLOCK_DIM, REGION_BLOCKS_X, REGION_BLOCKS_Y);
    let cuda_available = cuda_encode_available(&pixels, &coefficients, &jobs);

    bench_host_input(c, &pixels, cuda_available);
    bench_codeblock_microkernels(c, &coefficients, &jobs, cuda_available);
    bench_device_input_regions(c, &region_coefficients, &region_jobs, cuda_available);
}

fn bench_host_input(c: &mut Criterion, pixels: &[u8], cuda_available: bool) {
    let mut group = c.benchmark_group("j2k_cuda_htj2k_host_input_encode");
    group.bench_with_input(
        BenchmarkId::new("cpu_gray8", TILE_DIM),
        pixels,
        |b, pixels| {
            let options = cpu_htj2k_options();
            b.iter(|| {
                let samples = J2kLosslessSamples::new(
                    std::hint::black_box(pixels),
                    TILE_DIM,
                    TILE_DIM,
                    1,
                    8,
                    false,
                )
                .expect("valid gray8 samples");
                let encoded =
                    encode_j2k_lossless(samples, &options).expect("CPU HTJ2K lossless encode");
                assert_eq!(encoded.backend, BackendKind::Cpu);
                std::hint::black_box(encoded.codestream.len())
            });
        },
    );

    if cuda_available {
        group.bench_with_input(
            BenchmarkId::new("cuda_gray8", TILE_DIM),
            pixels,
            |b, pixels| {
                let options = cuda_htj2k_options();
                b.iter(|| {
                    let samples = J2kLosslessSamples::new(
                        std::hint::black_box(pixels),
                        TILE_DIM,
                        TILE_DIM,
                        1,
                        8,
                        false,
                    )
                    .expect("valid gray8 samples");
                    let encoded = encode_j2k_lossless_with_cuda(samples, &options)
                        .expect("CUDA HTJ2K lossless encode");
                    assert_eq!(encoded.backend, BackendKind::Cuda);
                    std::hint::black_box(encoded.codestream.len())
                });
            },
        );
    }
    group.finish();
}

fn bench_codeblock_microkernels(
    c: &mut Criterion,
    coefficients: &[i32],
    jobs: &[CudaHtj2kEncodeCodeBlockJob],
    cuda_available: bool,
) {
    let mut group = c.benchmark_group("j2k_cuda_htj2k_codeblock_microkernel");
    group.bench_with_input(
        BenchmarkId::new("cpu_scalar_cleanup", CODE_BLOCK_BATCH),
        &(coefficients, jobs),
        |b, (coefficients, jobs)| {
            b.iter(|| {
                let encoded_bytes = jobs
                    .iter()
                    .map(|job| {
                        let coefficients = contiguous_block(coefficients, *job);
                        encode_ht_code_block_scalar(
                            std::hint::black_box(coefficients),
                            job.width,
                            job.height,
                            job.total_bitplanes,
                        )
                        .expect("native scalar HT code-block encode")
                        .data
                        .len()
                    })
                    .sum::<usize>();
                std::hint::black_box(encoded_bytes)
            });
        },
    );

    if cuda_available {
        group.bench_with_input(
            BenchmarkId::new("cuda_host_staged_cleanup", CODE_BLOCK_BATCH),
            &(coefficients, jobs),
            |b, (coefficients, jobs)| {
                let context = CudaContext::system_default().expect("CUDA context");
                let uvlc_table = uvlc_encode_table_bytes();
                let resources = context
                    .upload_htj2k_encode_resources(cuda_encode_tables(&uvlc_table))
                    .expect("CUDA HTJ2K encode resources");
                b.iter(|| {
                    let encoded = context
                        .encode_htj2k_codeblocks_with_resources(
                            std::hint::black_box(coefficients),
                            std::hint::black_box(jobs),
                            &resources,
                        )
                        .expect("CUDA host-staged HTJ2K code-block encode");
                    std::hint::black_box(assert_cuda_batch(&encoded))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cuda_resident_cleanup", CODE_BLOCK_BATCH),
            &(coefficients, jobs),
            |b, (coefficients, jobs)| {
                let context = CudaContext::system_default().expect("CUDA context");
                let coefficient_bytes = coefficients_as_bytes(coefficients);
                let resident_coefficients = context
                    .upload(&coefficient_bytes)
                    .expect("resident quantized coefficients");
                let uvlc_table = uvlc_encode_table_bytes();
                let resources = context
                    .upload_htj2k_encode_resources(cuda_encode_tables(&uvlc_table))
                    .expect("CUDA HTJ2K encode resources");
                b.iter(|| {
                    let encoded = context
                        .encode_htj2k_codeblocks_resident_with_resources(
                            &resident_coefficients,
                            coefficients.len(),
                            std::hint::black_box(jobs),
                            &resources,
                        )
                        .expect("CUDA resident HTJ2K code-block encode");
                    std::hint::black_box(assert_cuda_batch(&encoded))
                });
            },
        );
    }
    group.finish();
}

fn bench_device_input_regions(
    c: &mut Criterion,
    coefficients: &[i32],
    jobs: &[CudaHtj2kEncodeCodeBlockRegionJob],
    cuda_available: bool,
) {
    let mut group = c.benchmark_group("j2k_cuda_htj2k_device_input_encode");
    group.bench_with_input(
        BenchmarkId::new("cpu_gather_scalar_cleanup", jobs.len()),
        &(coefficients, jobs),
        |b, (coefficients, jobs)| {
            b.iter(|| {
                let encoded_bytes = jobs
                    .iter()
                    .map(|job| {
                        let block = gather_region(coefficients, *job);
                        encode_ht_code_block_scalar(
                            std::hint::black_box(&block),
                            job.width,
                            job.height,
                            job.total_bitplanes,
                        )
                        .expect("native scalar strided HT encode")
                        .data
                        .len()
                    })
                    .sum::<usize>();
                std::hint::black_box(encoded_bytes)
            });
        },
    );

    if cuda_available {
        group.bench_with_input(
            BenchmarkId::new("cuda_resident_strided_cleanup", jobs.len()),
            &(coefficients, jobs),
            |b, (coefficients, jobs)| {
                let context = CudaContext::system_default().expect("CUDA context");
                let coefficient_bytes = coefficients_as_bytes(coefficients);
                let resident_coefficients = context
                    .upload(&coefficient_bytes)
                    .expect("resident strided quantized coefficients");
                let uvlc_table = uvlc_encode_table_bytes();
                let resources = context
                    .upload_htj2k_encode_resources(cuda_encode_tables(&uvlc_table))
                    .expect("CUDA HTJ2K encode resources");
                b.iter(|| {
                    let encoded = context
                        .encode_htj2k_codeblock_regions_resident_with_resources(
                            &resident_coefficients,
                            coefficients.len(),
                            std::hint::black_box(jobs),
                            &resources,
                        )
                        .expect("CUDA resident strided HTJ2K encode");
                    std::hint::black_box(assert_cuda_batch(&encoded))
                });
            },
        );
    }
    group.finish();
}

fn cuda_encode_available(
    pixels: &[u8],
    coefficients: &[i32],
    jobs: &[CudaHtj2kEncodeCodeBlockJob],
) -> bool {
    let samples = J2kLosslessSamples::new(pixels, TILE_DIM, TILE_DIM, 1, 8, false)
        .expect("valid gray8 samples");
    let public_result = encode_j2k_lossless_with_cuda(samples, &cuda_htj2k_options());
    match public_result {
        Ok(encoded) if encoded.backend == BackendKind::Cuda => {}
        Ok(_) if std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("J2K_REQUIRE_CUDA_BENCH is set but CUDA HTJ2K encode was not resident")
        }
        Ok(_) => {
            eprintln!("skipping CUDA HTJ2K encode benches: device encode did not dispatch");
            return false;
        }
        Err(error) if std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("J2K_REQUIRE_CUDA_BENCH is set but CUDA HTJ2K encode failed: {error}")
        }
        Err(error) => {
            eprintln!("skipping CUDA HTJ2K encode benches: {error}");
            return false;
        }
    }

    let context = match CudaContext::system_default() {
        Ok(context) => context,
        Err(error) if std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("J2K_REQUIRE_CUDA_BENCH is set but CUDA context failed: {error}")
        }
        Err(error) => {
            eprintln!("skipping CUDA HTJ2K encode benches: {error}");
            return false;
        }
    };
    let uvlc_table = uvlc_encode_table_bytes();
    let resources = match context.upload_htj2k_encode_resources(cuda_encode_tables(&uvlc_table)) {
        Ok(resources) => resources,
        Err(error) if std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("J2K_REQUIRE_CUDA_BENCH is set but CUDA HTJ2K encode resource upload failed: {error}")
        }
        Err(error) => {
            eprintln!("skipping CUDA HTJ2K encode benches: {error}");
            return false;
        }
    };
    let result =
        context.encode_htj2k_codeblocks_with_resources(coefficients, &jobs[..1], &resources);
    match result {
        Ok(encoded) if encoded.execution().kernel_dispatches() > 0 => true,
        Ok(_) if std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_some() => {
            panic!(
                "J2K_REQUIRE_CUDA_BENCH is set but CUDA HTJ2K code-block encode did not dispatch"
            )
        }
        Ok(_) => {
            eprintln!("skipping CUDA HTJ2K encode benches: code-block kernel did not dispatch");
            false
        }
        Err(error) if std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_some() => {
            panic!("J2K_REQUIRE_CUDA_BENCH is set but CUDA HTJ2K code-block encode failed: {error}")
        }
        Err(error) => {
            eprintln!("skipping CUDA HTJ2K encode benches: {error}");
            false
        }
    }
}

fn cpu_htj2k_options() -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::CpuOnly)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(1))
        .with_validation(J2kEncodeValidation::External)
}

fn cuda_htj2k_options() -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
        .with_max_decomposition_levels(Some(1))
        .with_validation(J2kEncodeValidation::External)
}

fn cuda_encode_tables(uvlc_table: &[u8]) -> CudaHtj2kEncodeTables<'_> {
    CudaHtj2kEncodeTables {
        vlc_table0: ht_vlc_encode_table0(),
        vlc_table1: ht_vlc_encode_table1(),
        uvlc_table,
    }
}

fn uvlc_encode_table_bytes() -> Vec<u8> {
    ht_uvlc_encode_table()
        .iter()
        .flat_map(|entry| {
            [
                entry.pre,
                entry.pre_len,
                entry.suf,
                entry.suf_len,
                entry.ext,
                entry.ext_len,
            ]
        })
        .collect()
}

fn assert_cuda_batch(encoded: &CudaHtj2kEncodedCodeBlocks) -> usize {
    assert_eq!(encoded.execution().kernel_dispatches(), 1);
    assert!(encoded
        .code_blocks()
        .iter()
        .all(|block| block.status().is_ok()));
    encoded
        .code_blocks()
        .iter()
        .map(|block| block.data().len())
        .sum()
}

fn contiguous_block(coefficients: &[i32], job: CudaHtj2kEncodeCodeBlockJob) -> &[i32] {
    let start = usize::try_from(job.coefficient_offset).expect("job offset fits usize");
    let len = area_len(job.width, job.height);
    &coefficients[start..start + len]
}

fn gather_region(coefficients: &[i32], job: CudaHtj2kEncodeCodeBlockRegionJob) -> Vec<i32> {
    let start = usize::try_from(job.coefficient_offset).expect("job offset fits usize");
    let stride = usize::try_from(job.coefficient_stride).expect("job stride fits usize");
    let width = usize::try_from(job.width).expect("job width fits usize");
    let height = usize::try_from(job.height).expect("job height fits usize");
    let mut block = Vec::with_capacity(width * height);
    for y in 0..height {
        let row_start = start + y * stride;
        block.extend_from_slice(&coefficients[row_start..row_start + width]);
    }
    block
}

fn generate_gray_tile(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(area_len(width, height));
    for y in 0..height {
        for x in 0..width {
            let value = (x * 17 + y * 31 + x.wrapping_mul(y) / 11) & 0xff;
            pixels.push(u8::try_from(value).expect("masked sample fits in u8"));
        }
    }
    pixels
}

fn generate_codeblock_coefficients(width: u32, height: u32, batch: usize) -> Vec<i32> {
    let mut coefficients = Vec::with_capacity(area_len(width, height) * batch);
    for block in 0..batch {
        let block = u32::try_from(block).expect("bench block index fits u32");
        for y in 0..height {
            for x in 0..width {
                coefficients.push(patterned_coefficient(x, y, block));
            }
        }
    }
    coefficients
}

fn generate_region_coefficients(width: u32, height: u32) -> Vec<i32> {
    let mut coefficients = Vec::with_capacity(area_len(width, height));
    for y in 0..height {
        for x in 0..width {
            coefficients.push(patterned_coefficient(
                x,
                y,
                x / CODE_BLOCK_DIM + y / CODE_BLOCK_DIM,
            ));
        }
    }
    coefficients
}

fn patterned_coefficient(x: u32, y: u32, block: u32) -> i32 {
    if (x + y + block).is_multiple_of(7) {
        return 0;
    }
    let raw = (x * 13 + y * 17 + block * 19 + (x ^ y)) & 0x1ff;
    (i32::try_from(raw).expect("masked coefficient fits i32") - 256) / 4
}

fn contiguous_jobs(width: u32, height: u32, batch: usize) -> Vec<CudaHtj2kEncodeCodeBlockJob> {
    let block_len = area_len(width, height);
    let mut jobs = Vec::with_capacity(batch);
    for block in 0..batch {
        let offset = block
            .checked_mul(block_len)
            .expect("bench coefficient offset fits usize");
        jobs.push(CudaHtj2kEncodeCodeBlockJob {
            coefficient_offset: u32::try_from(offset).expect("bench offset fits u32"),
            width,
            height,
            total_bitplanes: 8,
            target_coding_passes: 1,
        });
    }
    jobs
}

fn strided_region_jobs(
    block_dim: u32,
    blocks_x: u32,
    blocks_y: u32,
) -> Vec<CudaHtj2kEncodeCodeBlockRegionJob> {
    let stride = block_dim
        .checked_mul(blocks_x)
        .expect("bench stride fits u32");
    let mut jobs = Vec::with_capacity(area_len(blocks_x, blocks_y));
    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let row_offset = by
                .checked_mul(block_dim)
                .and_then(|value| value.checked_mul(stride))
                .expect("bench row offset fits u32");
            let column_offset = bx
                .checked_mul(block_dim)
                .expect("bench column offset fits u32");
            jobs.push(CudaHtj2kEncodeCodeBlockRegionJob {
                coefficient_offset: row_offset + column_offset,
                coefficient_stride: stride,
                width: block_dim,
                height: block_dim,
                total_bitplanes: 8,
                target_coding_passes: 1,
            });
        }
    }
    jobs
}

fn coefficients_as_bytes(coefficients: &[i32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(coefficients));
    for coefficient in coefficients {
        bytes.extend_from_slice(&coefficient.to_ne_bytes());
    }
    bytes
}

fn area_len(width: u32, height: u32) -> usize {
    usize::try_from(width).expect("bench width fits usize")
        * usize::try_from(height).expect("bench height fits usize")
}

criterion_group!(benches, bench_htj2k_encode);
criterion_main!(benches);
