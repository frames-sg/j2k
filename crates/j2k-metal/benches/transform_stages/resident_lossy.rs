// SPDX-License-Identifier: MIT OR Apache-2.0

use criterion::{Criterion, SamplingMode, Throughput};
use j2k_metal::MetalEncodeStageAccelerator;
use j2k_native::{CpuOnlyJ2kEncodeStageAccelerator, DecodeSettings, EncodeOptions, Image};

pub(super) fn bench(criterion: &mut Criterion) {
    for (width, height, components, batch) in [
        (128, 128, 3, 1),
        (512, 512, 1, 1),
        (512, 512, 3, 1),
        (640, 480, 3, 1),
        (1024, 1024, 3, 1),
        (512, 512, 3, 16),
    ] {
        let pixels = (0..batch)
            .map(|index| {
                let mut pixels = if components == 1 {
                    j2k_test_support::patterned_gray8(width, height)
                } else {
                    j2k_test_support::patterned_rgb8(width, height)
                };
                pixels[0] = pixels[0].wrapping_add(index);
                pixels
            })
            .collect::<Vec<_>>();
        let options = EncodeOptions {
            reversible: false,
            num_decomposition_levels: 3,
            use_mct: components == 3,
            guard_bits: 2,
            use_ht_block_coding: true,
            ..EncodeOptions::default()
        };
        let run = |accelerator: &mut MetalEncodeStageAccelerator| {
            pixels
                .iter()
                .map(|pixels| {
                    j2k_native::encode_with_accelerator(
                        pixels,
                        width,
                        height,
                        components,
                        8,
                        false,
                        &options,
                        accelerator,
                    )
                    .expect("lossy Metal encode")
                })
                .collect::<Vec<_>>()
        };
        let mut accelerator = MetalEncodeStageAccelerator::default();
        let first = run(&mut accelerator);
        let second = run(&mut accelerator);
        assert_eq!(first, second, "deterministic lossy codestreams");
        let mut framed_inputs = Vec::new();
        let mut framed_outputs = Vec::new();
        for (pixels, encoded) in pixels.iter().zip(&first) {
            let expected = j2k_native::encode_with_accelerator(
                pixels,
                width,
                height,
                components,
                8,
                false,
                &options,
                &mut CpuOnlyJ2kEncodeStageAccelerator,
            )
            .expect("scalar lossy oracle");
            assert_eq!(encoded, &expected, "lossy scalar codestream parity");
            let decode = |data: &[u8]| {
                Image::new(data, &DecodeSettings::default())
                    .expect("lossy parse")
                    .decode_native()
                    .expect("lossy decode")
                    .data
            };
            assert_eq!(decode(encoded), decode(&expected));
            framed_inputs.extend_from_slice(&(pixels.len() as u64).to_le_bytes());
            framed_inputs.extend_from_slice(pixels);
            framed_outputs.extend_from_slice(&(encoded.len() as u64).to_le_bytes());
            framed_outputs.extend_from_slice(encoded);
        }
        let id = format!("metal_lossy_resident/{width}x{height}/c{components}/b{batch}");
        eprintln!(
            "{id} input_sha256={} output_sha256={} exact_parity=true",
            j2k_test_support::auto_routing_sha256(&framed_inputs),
            j2k_test_support::auto_routing_sha256(&framed_outputs)
        );
        let mut group = criterion.benchmark_group(id);
        group.sampling_mode(SamplingMode::Flat);
        group.throughput(Throughput::Bytes(
            pixels.iter().map(|p| p.len() as u64).sum(),
        ));
        group.bench_function("encode", |b| {
            b.iter(|| std::hint::black_box(run(&mut accelerator)))
        });
        group.finish();
    }
}
