// SPDX-License-Identifier: MIT OR Apache-2.0

use criterion::{Criterion, Throughput};
use j2k::{
    encode_j2k_lossless, EncodeBackendPreference, J2kBlockCodingMode, J2kLosslessEncodeOptions,
    J2kLosslessSamples,
};
use j2k_core::PixelFormat;
use j2k_metal::{
    benchmark_private_buffer_with_bytes, encode_lossless_batch_with_report, MetalBackendSession,
    MetalEncodeInputStaging, MetalLosslessEncodeBatchRequest, MetalLosslessEncodeConfig,
    MetalLosslessEncodeOutcome, MetalLosslessEncodeTile,
};
use j2k_native::{DecodeSettings, Image};
use rayon::prelude::*;

use super::support::{options, run_device_batch, DIMENSION};

const BATCH_SIZES: [usize; 3] = [1, 4, 16];

fn encode_cpu_tile(pixels: &[u8], options: &J2kLosslessEncodeOptions) -> Vec<u8> {
    let samples = J2kLosslessSamples::new(pixels, DIMENSION, DIMENSION, 3, 8, false)
        .expect("valid resident batch CPU samples");
    encode_j2k_lossless(samples, options)
        .expect("resident batch CPU encode")
        .codestream
}

fn encode_cpu_serial(
    pixels: &[u8],
    batch_size: usize,
    options: &J2kLosslessEncodeOptions,
) -> Vec<Vec<u8>> {
    (0..batch_size)
        .map(|_| encode_cpu_tile(pixels, options))
        .collect()
}

fn encode_cpu_parallel(
    pixels: &[u8],
    batch_size: usize,
    options: &J2kLosslessEncodeOptions,
) -> Vec<Vec<u8>> {
    (0..batch_size)
        .into_par_iter()
        .map(|_| encode_cpu_tile(pixels, options))
        .collect()
}

fn run_host_batch(
    session: &MetalBackendSession,
    tiles: &[MetalLosslessEncodeTile<'_>],
    batch_size: usize,
    options: &J2kLosslessEncodeOptions,
) -> Vec<MetalLosslessEncodeOutcome> {
    let outcomes = encode_lossless_batch_with_report(
        MetalLosslessEncodeBatchRequest {
            tiles,
            staging: MetalEncodeInputStaging::AlreadyPaddedContiguous,
            config: MetalLosslessEncodeConfig {
                gpu_encode_inflight_tiles: Some(batch_size),
                gpu_encode_memory_budget_bytes: None,
            },
        },
        options,
        session,
    )
    .expect("resident HTJ2K host-output batch");
    assert_eq!(outcomes.len(), batch_size);
    assert!(outcomes.iter().all(|outcome| {
        outcome.resident.coefficient_prep_used
            && outcome.resident.packetization_used
            && outcome.resident.codestream_assembly_used
    }));
    outcomes
}

fn framed_codestreams(codestreams: impl IntoIterator<Item = Vec<u8>>) -> (String, usize) {
    let mut framed = Vec::new();
    let mut encoded_bytes = 0usize;
    for codestream in codestreams {
        encoded_bytes = encoded_bytes
            .checked_add(codestream.len())
            .expect("resident batch encoded byte total fits usize");
        framed.extend_from_slice(
            &u64::try_from(codestream.len())
                .expect("resident batch codestream length fits u64")
                .to_le_bytes(),
        );
        framed.extend_from_slice(&codestream);
    }
    (
        j2k_test_support::auto_routing_sha256(&framed),
        encoded_bytes,
    )
}

fn verify_decodes_to_source(codestream: &[u8], pixels: &[u8], route: &str) {
    let decoded = Image::new(codestream, &DecodeSettings::default())
        .unwrap_or_else(|error| panic!("{route} codestream parses: {error}"))
        .decode_native()
        .unwrap_or_else(|error| panic!("{route} codestream decodes: {error}"));
    assert_eq!(decoded.data, pixels, "{route} decoded pixels differ");
}

fn verify_cpu_metal_batch(
    session: &MetalBackendSession,
    tiles: &[MetalLosslessEncodeTile<'_>],
    pixels: &[u8],
    batch_size: usize,
    cpu_options: &J2kLosslessEncodeOptions,
    metal_options: &J2kLosslessEncodeOptions,
) -> (String, String, usize, usize, usize) {
    let cpu_serial = encode_cpu_serial(pixels, batch_size, cpu_options);
    let cpu_parallel = encode_cpu_parallel(pixels, batch_size, cpu_options);
    assert_eq!(cpu_parallel, cpu_serial, "parallel CPU output differs");
    verify_decodes_to_source(&cpu_serial[0], pixels, "CPU");

    let device = run_device_batch(session, tiles, metal_options, Some(batch_size));
    assert_eq!(device.stats.effective_inflight_tiles, batch_size);
    assert!(device.stats.max_observed_inflight_tiles <= batch_size);
    if batch_size > 1 {
        assert!(
            device.stats.max_observed_inflight_tiles > 1,
            "resident Metal batch did not overlap tiles"
        );
    }
    let device_codestreams = device
        .outcomes
        .iter()
        .map(|outcome| {
            outcome
                .encoded
                .codestream_bytes()
                .expect("resident batch device codestream is readable")
        })
        .collect::<Vec<_>>();
    verify_decodes_to_source(&device_codestreams[0], pixels, "Metal");

    let host = run_host_batch(session, tiles, batch_size, metal_options);
    for (host_outcome, device_codestream) in host.iter().zip(&device_codestreams) {
        assert_eq!(
            &host_outcome.encoded.codestream, device_codestream,
            "host and device Metal batch routes differ"
        );
    }

    let (cpu_hash, cpu_bytes) = framed_codestreams(cpu_serial);
    let (metal_hash, metal_bytes) = framed_codestreams(device_codestreams);
    (
        cpu_hash,
        metal_hash,
        cpu_bytes,
        metal_bytes,
        device.stats.max_observed_inflight_tiles,
    )
}

fn total_codestream_bytes(codestreams: &[Vec<u8>]) -> usize {
    codestreams.iter().map(Vec::len).sum()
}

pub(crate) fn bench(criterion: &mut Criterion) {
    let session =
        MetalBackendSession::system_default().expect("resident batch benchmark needs Metal");
    let pixels = j2k_test_support::patterned_rgb8(DIMENSION, DIMENSION);
    let input = benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("upload resident batch benchmark input");
    // SAFETY: `input` belongs to `session`, is fully initialized, and remains
    // immutable until all synchronous benchmark submissions finish.
    let tile = unsafe {
        MetalLosslessEncodeTile::from_buffer(
            &input,
            0,
            (DIMENSION, DIMENSION),
            usize::try_from(DIMENSION).expect("dimension fits usize") * 3,
            (DIMENSION, DIMENSION),
            PixelFormat::Rgb8,
        )
    };
    let metal_options = options(J2kBlockCodingMode::HighThroughput);
    let cpu_options = metal_options.with_backend(EncodeBackendPreference::CpuOnly);

    for batch_size in BATCH_SIZES {
        let tiles = std::iter::repeat_n(tile, batch_size).collect::<Vec<_>>();
        let (cpu_hash, metal_hash, cpu_bytes, metal_bytes, max_inflight) = verify_cpu_metal_batch(
            &session,
            &tiles,
            &pixels,
            batch_size,
            &cpu_options,
            &metal_options,
        );
        eprintln!(
            "j2k_metal_resident_batch_probe batch_size={batch_size} size={DIMENSION}x{DIMENSION} cpu_sha256={cpu_hash} metal_sha256={metal_hash} cpu_bytes={cpu_bytes} metal_bytes={metal_bytes} max_inflight={max_inflight}"
        );

        let mut group = criterion.benchmark_group(format!(
            "htj2k-resident-batch/{DIMENSION}x{DIMENSION}/batch-{batch_size}"
        ));
        group.throughput(Throughput::Elements(
            u64::try_from(batch_size).expect("batch size fits u64"),
        ));
        group.bench_function("cpu-serial", |bencher| {
            bencher.iter(|| {
                let codestreams = encode_cpu_serial(&pixels, batch_size, &cpu_options);
                std::hint::black_box(total_codestream_bytes(&codestreams))
            });
        });
        group.bench_function("cpu-parallel", |bencher| {
            bencher.iter(|| {
                let codestreams = encode_cpu_parallel(&pixels, batch_size, &cpu_options);
                std::hint::black_box(total_codestream_bytes(&codestreams))
            });
        });
        group.bench_function("metal-resident", |bencher| {
            bencher.iter(|| {
                let outcomes = run_host_batch(&session, &tiles, batch_size, &metal_options);
                std::hint::black_box(
                    outcomes
                        .iter()
                        .map(|outcome| outcome.encoded.codestream.len())
                        .sum::<usize>(),
                )
            });
        });
        group.finish();
    }
}
