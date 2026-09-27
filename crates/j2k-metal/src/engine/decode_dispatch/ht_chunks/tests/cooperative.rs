// SPDX-License-Identifier: MIT OR Apache-2.0

//! The cooperative (one SIMD group per code block) cleanup decoder must match
//! the CPU oracle and the legacy one-thread-per-block kernel bit for bit.

#[cfg(target_os = "macos")]
use crate::metal_types::prelude::*;

use core::mem::size_of;

use j2k_core::HtGpuJobPassBucket;
use j2k_native::{DecodeSettings, DecoderContext, EncodeOptions, Image};

use super::super::{
    default_metal_ht_chunk_limits, plan_metal_ht_chunks, HtBatchInput, J2kHtCleanupBatchJob,
    MetalHtPipelineKind,
};
use crate::engine::abi::{J2kHtStatus, J2K_HT_STATUS_OK};
use crate::engine::resident_codestream::{
    dispatch_ht_cleanup_batched_in_encoder_with_status_offset, HtCleanupBatchDispatch,
};
use crate::engine::{
    checked_buffer_slice, commit_and_wait_metal, copied_slice_buffer,
    decode_prepared_ht_sub_band_group_on_cpu_profile, new_command_buffer,
    new_compute_command_encoder, prepare_direct_grayscale_plan, with_runtime, zeroed_shared_buffer,
    MetalRuntime,
};

struct Fixture {
    name: &'static str,
    width: u32,
    height: u32,
    bit_depth: u8,
    samples: Vec<u8>,
    options: EncodeOptions,
}

/// Deterministic xorshift noise: high-entropy `MagSgn` streams produce many
/// 0xFF bytes, which exercises bit-unstuffing inside the cooperative window.
fn noise(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state.to_le_bytes()[1]
        })
        .collect()
}

fn gradient(width: u32, height: u32) -> Vec<u8> {
    (0..height)
        .flat_map(|y| (0..width).map(move |x| ((x * 3 + y * 5 + (x * y) / 7) & 0xFF) as u8))
        .collect()
}

fn fixtures() -> Vec<Fixture> {
    let lossless = |levels, cbw, cbh| EncodeOptions {
        reversible: true,
        num_decomposition_levels: levels,
        code_block_width_exp: cbw,
        code_block_height_exp: cbh,
        ..EncodeOptions::default()
    };
    let lossy = |levels| EncodeOptions {
        reversible: false,
        num_decomposition_levels: levels,
        ..EncodeOptions::default()
    };
    vec![
        Fixture {
            name: "gradient_256_64x64",
            width: 256,
            height: 256,
            bit_depth: 8,
            samples: gradient(256, 256),
            options: lossless(3, 4, 4),
        },
        Fixture {
            name: "noise_256_64x64",
            width: 256,
            height: 256,
            bit_depth: 8,
            samples: noise(256 * 256, 0x1234_5678),
            options: lossless(2, 4, 4),
        },
        Fixture {
            name: "noise16_128_64x64",
            width: 128,
            height: 128,
            bit_depth: 16,
            samples: noise(128 * 128 * 2, 0x0bad_cafe),
            options: lossless(2, 4, 4),
        },
        Fixture {
            name: "noise_odd_131x67_32x32",
            width: 131,
            height: 67,
            bit_depth: 8,
            samples: noise(131 * 67, 0x2468_ace0),
            options: lossless(2, 3, 3),
        },
        Fixture {
            name: "noise_512x32_wide_256x16",
            width: 512,
            height: 32,
            bit_depth: 8,
            samples: noise(512 * 32, 0x1357_9bdf),
            options: lossless(1, 6, 2),
        },
        Fixture {
            name: "noise_256x66_wide_128x32",
            width: 256,
            height: 66,
            bit_depth: 8,
            samples: noise(256 * 66, 0x0f0f_0f0f),
            options: lossless(1, 5, 3),
        },
        Fixture {
            name: "noise_lossy97_192",
            width: 192,
            height: 192,
            bit_depth: 8,
            samples: noise(192 * 192, 0x7777_1111),
            options: lossy(3),
        },
        Fixture {
            name: "gradient_lossy97_odd_97x151",
            width: 97,
            height: 151,
            bit_depth: 8,
            samples: gradient(97, 151),
            options: lossy(2),
        },
    ]
}

struct CleanupWorkload {
    name: &'static str,
    coded_data: Vec<u8>,
    jobs: Vec<J2kHtCleanupBatchJob>,
    output_words: usize,
    expected: Vec<f32>,
}

fn cleanup_workloads() -> Vec<CleanupWorkload> {
    let mut workloads = Vec::new();
    for fixture in fixtures() {
        let bytes = j2k_native::encode_htj2k(
            &fixture.samples,
            fixture.width,
            fixture.height,
            1,
            fixture.bit_depth,
            false,
            &fixture.options,
        )
        .unwrap_or_else(|error| panic!("encode {}: {error:?}", fixture.name));
        let image = Image::new(&bytes, &DecodeSettings::default()).expect("fixture image");
        let mut context = DecoderContext::default();
        let direct = image
            .build_direct_grayscale_plan_with_context(&mut context)
            .expect("direct fixture plan");
        let prepared = prepare_direct_grayscale_plan(&direct).expect("prepared fixture plan");
        for group in &prepared.ht_groups {
            let expected = decode_prepared_ht_sub_band_group_on_cpu_profile(group, None)
                .expect("CPU coefficient oracle");
            let input = HtBatchInput {
                source_index: 0,
                payload: group.payload_source.as_ht_payload_source(),
                jobs: &group.jobs,
                output_base: 0,
                execution_owner: &group.execution_owner,
            };
            let plan = plan_metal_ht_chunks(&[input], default_metal_ht_chunk_limits())
                .expect("chunk plan");
            assert_eq!(
                plan.chunk_count(),
                1,
                "{}: fixture fits one chunk",
                fixture.name
            );
            let chunk = plan.pack_chunk(0).expect("packed chunk");
            assert_eq!(
                chunk.bucket,
                HtGpuJobPassBucket::CleanupOnly,
                "{}: native encoder emits cleanup-only blocks",
                fixture.name
            );
            workloads.push(CleanupWorkload {
                name: fixture.name,
                coded_data: chunk.coded_data,
                jobs: chunk.jobs,
                output_words: group.total_coefficients,
                expected,
            });
        }
    }
    workloads
}

#[derive(Clone, Copy)]
enum CleanupRoute {
    /// One thread per block (`j2k_decode_ht_cleanup_batched_cleanup_only`).
    Legacy,
    /// The production cleanup-only dispatcher (cooperative VLC + `MagSgn`).
    Production,
}

/// Decodes `replicas` copies of `workload` (each into its own output span)
/// through `route`, returning the output words and the GPU time in seconds.
fn decode_with(
    runtime: &MetalRuntime,
    route: CleanupRoute,
    workload: &CleanupWorkload,
    replicas: usize,
) -> (Vec<u32>, f64) {
    let mut jobs = Vec::with_capacity(workload.jobs.len() * replicas);
    for replica in 0..replicas {
        let shift = u32::try_from(replica * workload.output_words).expect("replica offset");
        jobs.extend(workload.jobs.iter().map(|job| J2kHtCleanupBatchJob {
            output_offset: job.output_offset + shift,
            ..*job
        }));
    }
    let output_words = workload.output_words * replicas;
    let coded = copied_slice_buffer(&runtime.device, &workload.coded_data).expect("coded buffer");
    let jobs_buffer = copied_slice_buffer(&runtime.device, &jobs).expect("jobs buffer");
    let decoded = zeroed_shared_buffer(&runtime.device, output_words * size_of::<u32>())
        .expect("decoded buffer");
    let status = zeroed_shared_buffer(&runtime.device, jobs.len() * size_of::<J2kHtStatus>())
        .expect("status buffer");
    let kernels = runtime.decode().expect("decode kernels");
    let command_buffer = new_command_buffer(&runtime.queue).expect("command buffer");
    let encoder = new_compute_command_encoder(&command_buffer).expect("encoder");
    match route {
        CleanupRoute::Legacy => {
            encoder.setComputePipelineState(&kernels.ht_cleanup_batched_cleanup_only);
            encoder.set_buffer(0, Some(&coded), 0);
            encoder.set_buffer(1, Some(&decoded), 0);
            encoder.set_buffer(2, Some(&jobs_buffer), 0);
            encoder.set_buffer(3, Some(&kernels.ht_vlc_table0), 0);
            encoder.set_buffer(4, Some(&kernels.ht_vlc_table1), 0);
            encoder.set_buffer(5, Some(&kernels.ht_uvlc_table0), 0);
            encoder.set_buffer(6, Some(&kernels.ht_uvlc_table1), 0);
            encoder.set_buffer(7, Some(&status), 0);
            encoder.dispatchThreads_threadsPerThreadgroup(
                j2k_metal_support::mtl_size(jobs.len() as u64, 1, 1),
                j2k_metal_support::mtl_size(32.min(jobs.len()) as u64, 1, 1),
            );
        }
        CleanupRoute::Production => {
            dispatch_ht_cleanup_batched_in_encoder_with_status_offset(
                kernels,
                &encoder,
                MetalHtPipelineKind::CleanupOnly,
                HtCleanupBatchDispatch {
                    coded_data: &coded,
                    jobs: &jobs_buffer,
                    job_count: jobs.len(),
                    decoded: &decoded,
                    status_buffer: &status,
                    status_offset_bytes: 0,
                },
            )
            .expect("cooperative dispatch");
        }
    }
    encoder.endEncoding();
    commit_and_wait_metal(&command_buffer).expect("cleanup decode");
    let statuses = checked_buffer_slice::<J2kHtStatus>(&status, jobs.len(), "statuses")
        .expect("status readback");
    if let Some(status) = statuses
        .iter()
        .find(|status| status.code != J2K_HT_STATUS_OK)
    {
        panic!(
            "{}: cleanup decode reported code {} detail {}",
            workload.name, status.code, status.detail
        );
    }
    let words = checked_buffer_slice::<u32>(&decoded, output_words, "coefficients")
        .expect("coefficient readback");
    (
        words,
        command_buffer.GPUEndTime() - command_buffer.GPUStartTime(),
    )
}

#[test]
fn cooperative_cleanup_matches_cpu_and_legacy_kernel_bit_for_bit() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }
    let workloads = cleanup_workloads();
    assert!(
        workloads.len() >= fixtures().len(),
        "every fixture must contribute cleanup-only work"
    );
    with_runtime(|runtime| {
        assert_eq!(
            runtime
                .decode()?
                .ht_cleanup_magsgn_batched
                .threadExecutionWidth(),
            32,
            "the cooperative decoder assumes 32-lane SIMD groups"
        );
        for workload in &workloads {
            let expected: Vec<u32> = workload
                .expected
                .iter()
                .map(|value| value.to_bits())
                .collect();
            let (legacy, _) = decode_with(runtime, CleanupRoute::Legacy, workload, 1);
            assert_eq!(legacy, expected, "{}: legacy vs CPU", workload.name);
            // One replica runs one block per SIMD group; larger batches pack
            // 8 and then 32 blocks into each VLC SIMD group.
            for replicas in [1, 12, 96] {
                let (cooperative, _) =
                    decode_with(runtime, CleanupRoute::Production, workload, replicas);
                for (replica, words) in cooperative.chunks(workload.output_words).enumerate() {
                    assert!(
                        words == expected.as_slice(),
                        "{}: cooperative replica {replica} of {replicas} differs from CPU",
                        workload.name
                    );
                }
            }
        }
        Ok(())
    })
    .expect("cooperative cleanup parity");
}
