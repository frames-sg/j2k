// SPDX-License-Identifier: MIT OR Apache-2.0

//! Batch surface outputs reuse a pooled buffer only after every surface that
//! referenced it has been dropped.

use super::*;

/// Four 512x256 tiles: 1.5 MiB of RGB output, above the pooling threshold.
fn distinct_420_jpegs(seed: u8) -> Vec<Arc<[u8]>> {
    distinct_420_jpegs_sized(seed, (512, 256))
}

fn distinct_420_jpegs_sized(seed: u8, (width, height): (u16, u16)) -> Vec<Arc<[u8]>> {
    (0..4_u8)
        .map(|tile| {
            let mut rgb = j2k_test_support::patterned_rgb8(u32::from(width), u32::from(height));
            for (index, sample) in rgb.iter_mut().enumerate() {
                let offset = u8::try_from(index % 251).expect("offset fits u8");
                *sample = sample.wrapping_add(offset.wrapping_mul(seed.wrapping_add(tile)));
            }
            let mut jpeg = Vec::new();
            let mut encoder = jpeg_encoder::Encoder::new(&mut jpeg, 85);
            encoder.set_sampling_factor(jpeg_encoder::SamplingFactor::F_2_2);
            encoder
                .encode(&rgb, width, height, jpeg_encoder::ColorType::Rgb)
                .expect("encode tile");
            Arc::<[u8]>::from(jpeg)
        })
        .collect()
}

fn decode_batch(runtime: &MetalRuntime, jpegs: &[Arc<[u8]>]) -> Vec<Surface> {
    let requests = jpegs
        .iter()
        .map(|jpeg| {
            batch::QueuedRequest::new(
                Arc::clone(jpeg),
                PixelFormat::Rgb8,
                BackendRequest::Metal,
                batch::BatchOp::Full,
                None,
                None,
                Some(Arc::new(
                    j2k_jpeg::adapter::build_fast420_packet(jpeg).expect("fast420 packet"),
                )),
            )
        })
        .collect::<Vec<_>>();
    let packets = batched_fast_packets(&requests)
        .expect("packet lookup")
        .expect("fast packets");
    try_decode_fast_subsampled_full_rgb_batch_to_surfaces::<JpegFast420PacketV1>(
        runtime, &requests, &packets,
    )
    .expect("batch decode")
    .expect("batch stays on Metal")
    .into_iter()
    .map(|surface| surface.expect("tile surface"))
    .collect()
}

fn buffer_address(surface: &Surface) -> usize {
    let (buffer, _) = surface
        .metal_buffer_trusted()
        .expect("batch surface is Metal resident");
    std::ptr::from_ref(buffer).addr()
}

fn assert_matches_cpu(surfaces: &[Surface], jpegs: &[Arc<[u8]>], what: &str) {
    for (index, (surface, jpeg)) in surfaces.iter().zip(jpegs).enumerate() {
        let (expected, _) = CpuDecoder::new(jpeg)
            .expect("CPU decoder")
            .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
            .expect("CPU decode");
        assert!(
            surface.as_bytes().expect("surface bytes").as_ref() == expected.as_slice(),
            "{what} tile {index} differs from the CPU decoder"
        );
    }
}

#[test]
fn live_surfaces_keep_their_output_while_later_batches_decode() {
    if !should_run_metal_runtime() {
        return;
    }
    let runtime = MetalRuntime::new().expect("runtime");
    let (a, b, c) = (
        distinct_420_jpegs(1),
        distinct_420_jpegs(2),
        distinct_420_jpegs(3),
    );

    let first = decode_batch(&runtime, &a);
    let second = decode_batch(&runtime, &b);
    let third = decode_batch(&runtime, &c);
    assert_ne!(buffer_address(&first[0]), buffer_address(&second[0]));
    assert_ne!(buffer_address(&second[0]), buffer_address(&third[0]));
    assert_ne!(buffer_address(&first[0]), buffer_address(&third[0]));
    assert_eq!(runtime.pooled_surface_outputs_for_test(), 2);
    assert_matches_cpu(&first, &a, "first batch");
    assert_matches_cpu(&second, &b, "second batch");
    assert_matches_cpu(&third, &c, "third batch");
}

#[test]
fn dropped_surfaces_release_their_output_for_the_next_batch() {
    if !should_run_metal_runtime() {
        return;
    }
    let runtime = MetalRuntime::new().expect("runtime");
    let (a, b) = (distinct_420_jpegs(4), distinct_420_jpegs(5));

    let first = decode_batch(&runtime, &a);
    let first_buffer = buffer_address(&first[0]);
    assert_matches_cpu(&first, &a, "first batch");
    drop(first);

    let second = decode_batch(&runtime, &b);
    assert_eq!(buffer_address(&second[0]), first_buffer);
    assert_eq!(runtime.pooled_surface_outputs_for_test(), 1);
    assert_matches_cpu(&second, &b, "second batch");

    // One surface of a batch keeps the whole output buffer in use.
    let kept = second.into_iter().next().expect("one surface");
    let third = decode_batch(&runtime, &a);
    assert_ne!(buffer_address(&third[0]), first_buffer);
    assert_matches_cpu(std::slice::from_ref(&kept), &b[..1], "kept surface");
    assert_matches_cpu(&third, &a, "third batch");
}

#[test]
fn small_outputs_are_allocated_per_call() {
    if !should_run_metal_runtime() {
        return;
    }
    let runtime = MetalRuntime::new().expect("runtime");
    let small = distinct_420_jpegs_sized(6, (64, 48));
    let first = decode_batch(&runtime, &small);
    drop(first);
    let second = decode_batch(&runtime, &small);
    assert_eq!(runtime.pooled_surface_outputs_for_test(), 0);
    assert_matches_cpu(&second, &small, "small batch");
}

/// Evidence for the P34 record: input and output SHA-256 of the retained-session
/// `wsi_tile_batch_rgb` workloads over three consecutive 64-tile batches, so
/// later batches write a reused pooled buffer, against the CPU decoder.
#[test]
#[ignore = "P34 evidence; run explicitly with --include-ignored --nocapture"]
fn p34_retained_session_output_hashes() {
    use j2k_core::{DeviceSubmission as _, TileBatchDecodeSubmit as _};
    use sha2::{Digest, Sha256};

    if !should_run_metal_runtime() {
        return;
    }
    let hex = |bytes: &[u8]| -> String { format!("{:x}", Sha256::digest(bytes)) };
    let generated = |sampling| {
        let rgb = j2k_test_support::gpu_bench_rgb8(256, 256);
        let mut jpeg = Vec::new();
        let mut encoder = jpeg_encoder::Encoder::new(&mut jpeg, 90);
        encoder.set_sampling_factor(sampling);
        encoder
            .encode(&rgb, 256, 256, jpeg_encoder::ColorType::Rgb)
            .expect("encode generated benchmark JPEG");
        jpeg
    };
    let inputs: [(&str, Vec<u8>); 6] = [
        (
            "generated/fast420_256x256",
            generated(jpeg_encoder::SamplingFactor::F_2_2),
        ),
        (
            "generated/fast422_256x256",
            generated(jpeg_encoder::SamplingFactor::F_2_1),
        ),
        (
            "generated/fast444_256x256",
            generated(jpeg_encoder::SamplingFactor::F_1_1),
        ),
        ("repo/baseline_420_16x16", BASELINE_420.to_vec()),
        ("repo/baseline_422_16x8", BASELINE_422.to_vec()),
        ("repo/baseline_444_8x8", BASELINE_444.to_vec()),
    ];
    let mut corpus = Vec::new();
    for (name, bytes) in &inputs {
        corpus.extend_from_slice(bytes);
        let (rgb, _) = CpuDecoder::new(bytes)
            .expect("CPU decoder")
            .decode_request(DecodeRequest::full(PixelFormat::Rgb8))
            .expect("CPU decode");
        let expected = hex(&rgb.repeat(64));
        let mut ctx = j2k_jpeg::DecoderContext::default();
        let mut pool = j2k_jpeg::ScratchPool::new();
        let mut session = crate::MetalSession::default();
        for iteration in 0..3 {
            let submissions = (0..64)
                .map(|_| {
                    crate::Codec::submit_tile_to_device(
                        &mut ctx,
                        &mut session,
                        &mut pool,
                        bytes,
                        PixelFormat::Rgb8,
                        BackendRequest::Metal,
                    )
                    .expect("submit")
                })
                .collect::<Vec<_>>();
            let mut output = Vec::new();
            for submission in submissions {
                let surface = submission.wait().expect("surface");
                output.extend_from_slice(surface.as_bytes().expect("surface bytes").as_ref());
            }
            assert_eq!(hex(&output), expected, "{name} iteration {iteration}");
        }
        println!(
            "{name} input_sha256 {} output_sha256 {expected}",
            hex(bytes)
        );
    }
    println!("input_corpus_sha256 {}", hex(&corpus));
}
