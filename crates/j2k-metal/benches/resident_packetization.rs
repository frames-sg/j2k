// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(target_os = "macos")]
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
#[cfg(target_os = "macos")]
use j2k::{
    EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation, J2kLosslessEncodeOptions,
};
#[cfg(target_os = "macos")]
use j2k_core::{DeviceSubmission, PixelFormat};
#[cfg(target_os = "macos")]
use j2k_metal::{
    benchmark_private_buffer_with_bytes, submit_lossless_batch_to_metal, MetalBackendSession,
    MetalEncodeInputStaging, MetalLosslessBufferEncodeBatchOutcome,
    MetalLosslessEncodeBatchRequest, MetalLosslessEncodeConfig, MetalLosslessEncodeTile,
};
#[cfg(target_os = "macos")]
use j2k_native::{DecodeSettings, Image};

#[cfg(target_os = "macos")]
const DIMENSION: u32 = 512;
#[cfg(target_os = "macos")]
const BATCH_SIZE: usize = 16;

#[cfg(not(target_os = "macos"))]
fn main() {
    assert!(
        std::env::var_os("J2K_REQUIRE_METAL_BENCH").is_none(),
        "J2K Metal resident packetization benchmark requires macOS"
    );
    eprintln!("J2K Metal resident packetization benchmark skipped outside macOS");
}

#[cfg(target_os = "macos")]
fn options(block_coding_mode: J2kBlockCodingMode) -> J2kLosslessEncodeOptions {
    J2kLosslessEncodeOptions::default()
        .with_backend(EncodeBackendPreference::RequireDevice)
        .with_block_coding_mode(block_coding_mode)
        .with_max_decomposition_levels(Some(3))
        .with_validation(J2kEncodeValidation::External)
}

#[cfg(target_os = "macos")]
fn run_batch(
    session: &MetalBackendSession,
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
) -> MetalLosslessBufferEncodeBatchOutcome {
    let outcome = submit_lossless_batch_to_metal(
        MetalLosslessEncodeBatchRequest {
            tiles,
            staging: MetalEncodeInputStaging::AlreadyPaddedContiguous,
            config: MetalLosslessEncodeConfig::default(),
        },
        options,
        session,
    )
    .expect("submit resident packetization benchmark batch")
    .wait()
    .expect("complete resident packetization benchmark batch");
    assert_eq!(outcome.outcomes.len(), BATCH_SIZE);
    assert!(
        outcome
            .outcomes
            .iter()
            .all(|item| item.resident.packetization_used),
        "benchmark must exercise resident packetization"
    );
    outcome
}

#[cfg(target_os = "macos")]
fn probe_exact_output(
    session: &MetalBackendSession,
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &J2kLosslessEncodeOptions,
    pixels: &[u8],
) -> (String, usize) {
    let outcome = run_batch(session, tiles, options);

    let mut framed = Vec::new();
    let mut encoded_bytes = 0usize;
    let mut first = None;
    for item in &outcome.outcomes {
        let codestream = item
            .encoded
            .codestream_bytes()
            .expect("packetization probe codestream is CPU-readable");
        if let Some(expected) = &first {
            assert_eq!(&codestream, expected, "repeated tiles must encode exactly");
        } else {
            let decoded = Image::new(&codestream, &DecodeSettings::default())
                .expect("packetization probe codestream parses")
                .decode_native()
                .expect("packetization probe codestream decodes");
            assert_eq!(decoded.data, pixels);
            first = Some(codestream.clone());
        }
        encoded_bytes = encoded_bytes
            .checked_add(codestream.len())
            .expect("packetization probe encoded-byte count fits usize");
        framed.extend_from_slice(
            &u64::try_from(codestream.len())
                .expect("packetization probe codestream length fits u64")
                .to_le_bytes(),
        );
        framed.extend_from_slice(&codestream);
    }
    (
        j2k_test_support::auto_routing_sha256(&framed),
        encoded_bytes,
    )
}

#[cfg(target_os = "macos")]
fn bench_resident_packetization(criterion: &mut Criterion) {
    let session = MetalBackendSession::system_default()
        .expect("resident packetization benchmark requires a Metal device");
    let pixels = j2k_test_support::patterned_rgb8(DIMENSION, DIMENSION);
    let input = benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("upload resident packetization benchmark input");
    // SAFETY: `input` belongs to `session`, was fully initialized above, and
    // remains immutable until every synchronous benchmark submission returns.
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
    let tiles = vec![tile; BATCH_SIZE];

    let mut group = criterion.benchmark_group("metal_resident_packetization");
    group.throughput(Throughput::Bytes(
        u64::from(DIMENSION) * u64::from(DIMENSION) * 3 * BATCH_SIZE as u64,
    ));
    for (label, block_coding_mode) in [
        ("classic", J2kBlockCodingMode::Classic),
        ("ht", J2kBlockCodingMode::HighThroughput),
    ] {
        let options = options(block_coding_mode);
        let (hash, encoded_bytes) = probe_exact_output(&session, &tiles, &options, &pixels);
        eprintln!(
            "j2k_metal_packetization_probe coding={label} batch_size={BATCH_SIZE} size={DIMENSION}x{DIMENSION} output_sha256={hash} encoded_bytes={encoded_bytes}"
        );
        group.bench_function(label, |bencher| {
            bencher.iter(|| {
                let outcome = run_batch(&session, &tiles, &options);
                std::hint::black_box(
                    outcome
                        .outcomes
                        .iter()
                        .map(|item| item.encoded.byte_len())
                        .sum::<usize>(),
                )
            });
        });
    }
    group.finish();
}

#[cfg(target_os = "macos")]
criterion_group!(benches, bench_resident_packetization);
#[cfg(target_os = "macos")]
criterion_main!(benches);
