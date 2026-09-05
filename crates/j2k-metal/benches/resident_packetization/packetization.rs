// SPDX-License-Identifier: MIT OR Apache-2.0

use criterion::{Criterion, Throughput};
use j2k::J2kBlockCodingMode;
use j2k_core::PixelFormat;
use j2k_metal::{
    benchmark_private_buffer_with_bytes, MetalBackendSession, MetalLosslessEncodeTile,
};
use j2k_native::{DecodeSettings, Image};

use super::support::{options, run_device_batch, DIMENSION};

const PACKETIZATION_BATCH_SIZE: usize = 16;

fn probe_exact_output(
    session: &MetalBackendSession,
    tiles: &[MetalLosslessEncodeTile<'_>],
    options: &j2k::J2kLosslessEncodeOptions,
    pixels: &[u8],
) -> (String, usize) {
    let outcome = run_device_batch(session, tiles, options, None);

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

pub(crate) fn bench(criterion: &mut Criterion) {
    for dimension in [128, DIMENSION, 1024] {
        bench_dimension(criterion, dimension);
    }
}

fn bench_dimension(criterion: &mut Criterion, dimension: u32) {
    let session = MetalBackendSession::system_default()
        .expect("resident packetization benchmark requires a Metal device");
    let pixels = j2k_test_support::patterned_rgb8(dimension, dimension);
    let input = benchmark_private_buffer_with_bytes(&session, &pixels)
        .expect("upload resident packetization benchmark input");
    // SAFETY: `input` belongs to `session`, was fully initialized above, and
    // remains immutable until every synchronous benchmark submission returns.
    let tile = unsafe {
        MetalLosslessEncodeTile::from_buffer(
            &input,
            0,
            (dimension, dimension),
            usize::try_from(dimension).expect("dimension fits usize") * 3,
            (dimension, dimension),
            PixelFormat::Rgb8,
        )
    };
    let tiles = std::iter::repeat_n(tile, PACKETIZATION_BATCH_SIZE).collect::<Vec<_>>();

    let mut group = criterion.benchmark_group(if dimension == DIMENSION {
        "metal_resident_packetization".to_owned()
    } else {
        format!("metal_resident_packetization_{dimension}")
    });
    group.throughput(Throughput::Bytes(
        u64::from(dimension)
            * u64::from(dimension)
            * 3
            * u64::try_from(PACKETIZATION_BATCH_SIZE).expect("batch size fits u64"),
    ));
    for (label, block_coding_mode) in [
        ("classic", J2kBlockCodingMode::Classic),
        ("ht", J2kBlockCodingMode::HighThroughput),
    ] {
        // The Classic 1024x1024 batch-16 scratch allocation exceeds the
        // resident pool's 512 MiB per-allocation limit; HT fits this cell.
        if dimension > DIMENSION && block_coding_mode == J2kBlockCodingMode::Classic {
            continue;
        }
        let options = options(block_coding_mode);
        let (hash, encoded_bytes) = probe_exact_output(&session, &tiles, &options, &pixels);
        eprintln!(
            "j2k_metal_packetization_probe coding={label} batch_size={PACKETIZATION_BATCH_SIZE} size={dimension}x{dimension} output_sha256={hash} encoded_bytes={encoded_bytes}"
        );
        group.bench_function(label, |bencher| {
            bencher.iter(|| {
                let outcome = run_device_batch(&session, &tiles, &options, None);
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
