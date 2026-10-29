use std::sync::Arc;

use j2k_core::{BackendRequest, PixelFormat};
use j2k_native::{encode_component_planes_53, EncodeComponentPlane, EncodeOptions};

use crate::{J2kDecoder, MetalDecodeRequest, MetalTileBatch};

fn sampled_fixture(width: u32, height: u32, sampling: (u8, u8), ht: bool, seed: u8) -> Arc<[u8]> {
    let factors = [(1, 1), sampling, sampling];
    let data: Vec<Vec<u8>> = factors
        .iter()
        .enumerate()
        .map(|(component, &(sx, sy))| {
            (0..width.div_ceil(u32::from(sx)) * height.div_ceil(u32::from(sy)))
                .map(|i| {
                    u8::try_from(i % 256)
                        .unwrap()
                        .wrapping_mul(13)
                        .wrapping_add(seed)
                        .wrapping_add(u8::try_from(component).unwrap() * 37)
                })
                .collect()
        })
        .collect();
    let planes: Vec<_> = data
        .iter()
        .zip(factors)
        .map(|(data, (x_rsiz, y_rsiz))| EncodeComponentPlane {
            data,
            x_rsiz,
            y_rsiz,
        })
        .collect();
    encode_component_planes_53(
        &planes,
        width,
        height,
        8,
        false,
        &EncodeOptions {
            use_ht_block_coding: ht,
            use_mct: false,
            reversible: true,
            num_decomposition_levels: 2,
            validate_high_throughput_codestream: false,
            ..EncodeOptions::default()
        },
    )
    .unwrap()
    .into()
}

fn compare_batch(inputs: &[Arc<[u8]>], tolerance: u8, format: PixelFormat) -> f64 {
    let expected: Vec<_> = inputs
        .iter()
        .map(|bytes| {
            J2kDecoder::new(bytes)
                .unwrap()
                .decode_request_to_host_surface(MetalDecodeRequest::full(
                    format,
                    BackendRequest::Cpu,
                ))
                .unwrap()
                .as_bytes()
                .unwrap()
                .into_owned()
        })
        .collect();
    let mut batch = MetalTileBatch::with_capacity(inputs.len());
    for input in inputs {
        batch
            .push_shared_tile_request(
                input.clone(),
                MetalDecodeRequest::full(format, BackendRequest::Metal),
            )
            .unwrap();
    }
    crate::engine::reset_metal_command_buffers_for_test();
    let started = std::time::Instant::now();
    let output = batch.decode_all().unwrap();
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(output.len(), inputs.len());
    assert_eq!(
        crate::engine::metal_command_buffers_for_test(),
        1,
        "sampled batch must retain its coefficient graph through a single GPU submission"
    );
    for (surface, expected) in output.iter().zip(expected) {
        let actual = surface.as_bytes().unwrap();
        assert_eq!(actual.len(), expected.len());
        let max_error = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| a.abs_diff(b))
            .max()
            .unwrap();
        assert!(
            max_error <= tolerance,
            "sampled pixel error {max_error} exceeds {tolerance}"
        );
    }
    elapsed_ms
}

#[test]
fn sampled_color_batch_keeps_one_resident_submission() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    for ht in [false, true] {
        for sampling in [(2, 1), (2, 2), (1, 2), (3, 2)] {
            let first = sampled_fixture(33, 31, sampling, ht, 5);
            let second = sampled_fixture(33, 31, sampling, ht, 23);
            for format in [PixelFormat::Rgb8, PixelFormat::Rgba8, PixelFormat::Rgb16] {
                compare_batch(&[first.clone(), second.clone()], 0, format);
                compare_batch(&[first.clone(), first.clone()], 0, format);
            }
        }
    }
}

/// Prepares, submits, and completes `inputs` through the owned
/// `MetalBatchDecoder`, keeping output resident; returns elapsed ms.
#[cfg(not(debug_assertions))]
fn time_owned_batch(inputs: &[Arc<[u8]>], session: &crate::MetalBackendSession) -> f64 {
    let mut decoder = crate::MetalBatchDecoder::with_backend_session_and_options(
        session.clone(),
        j2k::BatchDecodeOptions::default(),
    );
    let started = std::time::Instant::now();
    let prepared = decoder
        .prepare(
            inputs
                .iter()
                .cloned()
                .map(j2k::EncodedImage::full)
                .collect(),
        )
        .unwrap();
    let (_, errors, group_errors) = decoder
        .submit_prepared(&prepared)
        .unwrap()
        .wait()
        .unwrap()
        .into_parts();
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    assert!(
        errors.is_empty() && group_errors.is_empty(),
        "{errors:?} {group_errors:?}"
    );
    elapsed_ms
}

/// Decodes `inputs` one at a time to resident surfaces, waiting for each;
/// returns elapsed ms.
#[cfg(not(debug_assertions))]
fn time_per_image(inputs: &[Arc<[u8]>], session: &crate::MetalBackendSession) -> f64 {
    let started = std::time::Instant::now();
    for input in inputs {
        J2kDecoder::new(input)
            .unwrap()
            .decode_request_to_device_with_session(
                MetalDecodeRequest::full(PixelFormat::Rgb8, BackendRequest::Metal),
                session,
            )
            .unwrap();
    }
    started.elapsed().as_secs_f64() * 1000.0
}

// This local timing harness requires a release build and an external corpus.
// The generated-fixture tests above and the owned-batch sampled group tests
// exercise the same assertions in CI coverage. `J2K_SAMPLED_COUNT` sets the
// tiles per level (default 8).
#[cfg(not(debug_assertions))]
#[test]
#[ignore = "release sampled DICOM decode characterization; requires J2K_SAMPLED_CORPUS"]
fn local_sampled_color_batch_characterization() -> Result<(), &'static str> {
    if cfg!(debug_assertions) {
        return Err("run with --release");
    }
    assert!(j2k_test_support::metal_runtime_gate(module_path!()));
    let directory = std::env::var_os("J2K_SAMPLED_CORPUS").expect("trusted extracted DICOM tiles");
    let count = std::env::var("J2K_SAMPLED_COUNT")
        .ok()
        .map_or(8, |count| count.parse::<usize>().unwrap());
    let session = crate::MetalBackendSession::system_default().unwrap();
    for level in 0..3 {
        let inputs: Vec<Arc<[u8]>> = (0..count)
            .map(|index| {
                std::fs::read(
                    std::path::Path::new(&directory).join(format!("level{level}-tile{index}.j2k")),
                )
                .unwrap()
                .into()
            })
            .collect();
        for sample in 0..8 {
            let ms = compare_batch(&inputs, 2, PixelFormat::Rgb8);
            let owned_ms = time_owned_batch(&inputs, &session);
            let single_ms = time_per_image(&inputs, &session);
            eprintln!(
                "sampled_native level={level} sample={sample} count={count} commands=1 ms={ms:.3} owned_batch_ms={owned_ms:.3} per_image_ms={single_ms:.3}"
            );
        }
    }
    Ok(())
}
