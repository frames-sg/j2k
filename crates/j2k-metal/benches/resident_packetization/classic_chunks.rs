// SPDX-License-Identifier: MIT OR Apache-2.0

use criterion::{Criterion, SamplingMode, Throughput};
use j2k::J2kBlockCodingMode;
use j2k_core::PixelFormat;
use j2k_metal::{
    benchmark_private_buffer_with_bytes, MetalBackendSession, MetalLosslessEncodeTile,
};
use j2k_native::{DecodeSettings, Image};

use super::support::{options, run_device_batch};

pub(crate) fn bench(criterion: &mut Criterion) {
    let session = MetalBackendSession::system_default().expect("Classic chunk benchmark Metal");
    let options = options(J2kBlockCodingMode::Classic);
    for (width, height) in [(512, 512), (640, 480), (1024, 1024)] {
        let pixels = j2k_test_support::patterned_rgb8(width, height);
        let input = benchmark_private_buffer_with_bytes(&session, &pixels).expect("Classic input");
        // SAFETY: the initialized input is immutable and retained until every
        // synchronous submission below has completed.
        let tile = unsafe {
            MetalLosslessEncodeTile::from_buffer(
                &input,
                0,
                (width, height),
                width as usize * 3,
                (width, height),
                PixelFormat::Rgb8,
            )
        };
        let tiles = vec![tile; 16];
        let mut reference = None;
        for inflight in [8, 4, 16] {
            // The scheduler may reduce the requested width to respect the
            // allocation cap. Record the actual width with the parity probe.
            let probe = run_device_batch(&session, &tiles, &options, Some(inflight));
            let codestreams = probe
                .outcomes
                .iter()
                .map(|item| {
                    item.encoded
                        .codestream_bytes()
                        .expect("Classic chunk readback")
                })
                .collect::<Vec<_>>();
            for codestream in &codestreams {
                let decoded = Image::new(codestream, &DecodeSettings::default())
                    .expect("Classic chunk parse")
                    .decode_native()
                    .expect("Classic chunk decode");
                assert_eq!(decoded.data, pixels);
            }
            if let Some(expected) = &reference {
                assert_eq!(&codestreams, expected);
            } else {
                reference = Some(codestreams.clone());
            }
            let mut framed = Vec::new();
            for codestream in &codestreams {
                framed.extend_from_slice(&(codestream.len() as u64).to_le_bytes());
                framed.extend_from_slice(codestream);
            }
            let id = format!("metal_classic_chunks/{width}x{height}/b16/inflight-{inflight}");
            eprintln!(
                "{id} input_sha256={} output_sha256={} max_inflight={} exact_parity=true",
                j2k_test_support::auto_routing_sha256(&pixels),
                j2k_test_support::auto_routing_sha256(&framed),
                probe.stats.max_observed_inflight_tiles
            );
            let mut group = criterion.benchmark_group(id);
            group.sampling_mode(SamplingMode::Flat);
            group.throughput(Throughput::Bytes(pixels.len() as u64 * 16));
            group.bench_function("resident", |b| {
                b.iter(|| {
                    let outcome = run_device_batch(&session, &tiles, &options, Some(inflight));
                    std::hint::black_box(
                        outcome
                            .outcomes
                            .iter()
                            .map(|item| item.encoded.byte_len())
                            .sum::<usize>(),
                    )
                });
            });
            group.finish();
        }
    }
}
