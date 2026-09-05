// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[path = "../dev_support/graph_programs.rs"]
mod graph_programs;

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[expect(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the benchmark keeps one explicit equivalent-work matrix and converts small sample counts for statistical summaries"
)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use core::{ffi::c_void, ptr::NonNull};
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };

    use j2k::{
        BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples, EncodedImage,
        PreparedBatch,
    };
    use j2k_mpsgraph::{MpsGraphBatchDecoder, MpsGraphProgram, MpsGraphTensorSpec};
    use j2k_test_support::{gpu_bench_rgb8, htj2k_rgb8_fixture};
    use objc2::AnyThread;
    use objc2_foundation::{NSArray, NSDictionary, NSNumber};
    use objc2_metal_performance_shaders::MPSDataType;
    use objc2_metal_performance_shaders_graph::MPSGraphTensorData;

    use graph_programs::{average_cpu, average_program};

    fn summarize(samples: &[f64]) -> (f64, f64, f64) {
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        if samples.len() < 30 {
            return (mean, f64::NAN, f64::NAN);
        }
        let variance = samples
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>()
            / (samples.len() - 1) as f64;
        // The gate starts at 30 samples. Retaining that run's t-critical value
        // for larger custom runs is conservative and keeps this harness small.
        let half_width = 2.045 * (variance / samples.len() as f64).sqrt();
        (mean, mean - half_width, mean + half_width)
    }

    fn read_scores(data: &MPSGraphTensorData, batch: usize) -> Vec<f32> {
        let mut values = vec![0.0_f32; batch];
        // SAFETY: every benchmark graph run is complete and returns one F32
        // score per image. Reading every score lets the preflight compare each
        // output independently with the CPU oracle.
        unsafe {
            data.mpsndarray().readBytes_strideBytes(
                NonNull::new(values.as_mut_ptr().cast::<c_void>()).expect("nonempty batch"),
                core::ptr::null_mut(),
            );
        }
        values
    }

    fn staged_iteration(
        decoder: &mut MpsGraphBatchDecoder,
        program: &MpsGraphProgram,
        prepared: &PreparedBatch,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let decoded = decoder.decode_prepared(prepared)?;
        let (mut groups, errors, group_errors) = decoded.into_parts();
        if !errors.is_empty() || !group_errors.is_empty() || groups.len() != 1 {
            return Err(std::io::Error::other("staged decode did not produce one group").into());
        }
        let group = groups.remove(0);
        let resident = group.resident_batch();
        // SAFETY: codec completion precedes this read, the resident owner is
        // live, and the exact recorded allocation range is requested.
        let host = unsafe {
            j2k_metal_support::checked_buffer_read_vec::<u8>(
                resident.metal_buffer(),
                resident.byte_offset(),
                resident.byte_len(),
            )?
        };
        let upload = j2k_metal_support::checked_shared_buffer_with_bytes(decoder.device(), &host)?;
        let spec = program.input_spec();
        let dimensions = spec.shape().map(NSNumber::new_usize);
        let shape = NSArray::from_retained_slice(&dimensions);
        // SAFETY: the uploaded allocation exactly contains the static RGB8
        // tensor shape and remains live through the blocking graph call.
        let tensor_data = unsafe {
            MPSGraphTensorData::initWithMTLBuffer_shape_dataType(
                MPSGraphTensorData::alloc(),
                &upload,
                &shape,
                MPSDataType::UInt8,
            )
        };
        let feeds = NSDictionary::from_slices(&[program.image_placeholder()], &[&*tensor_data]);
        let targets = NSArray::from_retained_slice(program.targets());
        // SAFETY: all graph inputs and their backing allocations remain live;
        // this API blocks until target execution completes.
        let results = unsafe {
            program
                .graph()
                .runWithMTLCommandQueue_feeds_targetTensors_targetOperations(
                    decoder.command_queue(),
                    &feeds,
                    &targets,
                    None,
                )
        };
        let result = results
            .objectForKey(program.targets()[0].as_ref())
            .ok_or_else(|| std::io::Error::other("staged graph omitted its target"))?;
        Ok(read_scores(&result, spec.shape()[0]))
    }

    fn run_path(
        path: &str,
        decoder: &mut MpsGraphBatchDecoder,
        program: &MpsGraphProgram,
        prepared: &PreparedBatch,
    ) -> Result<(Vec<f32>, Duration), Box<dyn std::error::Error>> {
        let group = &prepared.groups()[0];
        let batch = group.images().len();
        let started = Instant::now();
        match path {
            "staged" => {
                let scores = staged_iteration(decoder, program, prepared)?;
                Ok((scores, started.elapsed()))
            }
            "completed" => {
                let decoded = decoder.decode_prepared(prepared)?;
                let (mut groups, errors, group_errors) = decoded.into_parts();
                if !errors.is_empty() || !group_errors.is_empty() || groups.len() != 1 {
                    return Err(std::io::Error::other(
                        "completed path did not produce one successful group",
                    )
                    .into());
                }
                let output = program
                    .submit_completed(decoder.command_queue(), groups.remove(0))?
                    .wait()?;
                Ok((read_scores(&output.results()[0], batch), started.elapsed()))
            }
            "pipelined" => {
                let output = decoder.run_prepared_group(program, group)?;
                Ok((read_scores(&output.results()[0], batch), started.elapsed()))
            }
            "submit_latency" => {
                let submitted = decoder.submit_prepared_group(program, group)?;
                let submit_elapsed = started.elapsed();
                let output = submitted.wait()?;
                Ok((read_scores(&output.results()[0], batch), submit_elapsed))
            }
            _ => unreachable!("fixed benchmark path matrix"),
        }
    }

    fn validate_scores(
        path: &str,
        actual: &[f32],
        expected: &[f32],
    ) -> Result<(), Box<dyn std::error::Error>> {
        if actual.len() != expected.len() {
            return Err(std::io::Error::other(format!(
                "{path} returned {} scores, expected {}",
                actual.len(),
                expected.len(),
            ))
            .into());
        }
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            if (actual - expected).abs() > 1.0e-5 {
                return Err(std::io::Error::other(format!(
                    "{path} score {index} differs from the CPU oracle: MPSGraph={actual}, CPU={expected}",
                ))
                .into());
            }
        }
        Ok(())
    }

    fn fixture(codec: &str, size: u32) -> Vec<u8> {
        match codec {
            "htj2k" => htj2k_rgb8_fixture(size, size),
            "j2k" => j2k_native::encode(
                &gpu_bench_rgb8(size, size),
                size,
                size,
                3,
                8,
                false,
                &j2k_native::EncodeOptions {
                    reversible: true,
                    num_decomposition_levels: 3,
                    ..j2k_native::EncodeOptions::default()
                },
            )
            .expect("encode classic J2K benchmark fixture"),
            _ => unreachable!("fixed benchmark codec matrix"),
        }
    }

    const PATHS: [&str; 4] = ["staged", "completed", "pipelined", "submit_latency"];

    let iterations = std::env::var("J2K_MPSGRAPH_BENCH_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(if cfg!(debug_assertions) { 1 } else { 30 })
        .max(1);
    println!("codec,size,batch,path,mean_ms,ci95_low_ms,ci95_high_ms,checksum");

    for codec in ["htj2k", "j2k"] {
        for size in [512_u32, 1024] {
            let encoded = Arc::<[u8]>::from(fixture(codec, size));
            for batch in [1_usize, 8, 32] {
                let options = BatchDecodeOptions {
                    layout: BatchLayout::Nhwc,
                    ..BatchDecodeOptions::default()
                };
                let mut decoder = MpsGraphBatchDecoder::system_default(options)?;
                let inputs = (0..batch)
                    .map(|_| EncodedImage::full(encoded.clone()))
                    .collect::<Vec<_>>();
                let prepared = decoder.prepare(inputs)?;
                if prepared.groups().len() != 1 {
                    println!("{codec},{size},{batch},all,unsupported,unsupported,unsupported,0");
                    continue;
                }
                let spec = MpsGraphTensorSpec::from_group_info(
                    prepared.groups()[0].info(),
                    prepared.groups()[0].images().len(),
                )?;
                let program = average_program(spec.shape()[0], spec.shape()[1], spec.shape()[2])?;
                let mut cpu = CpuBatchDecoder::new(options);
                let cpu_decoded = cpu.decode_prepared(&prepared)?;
                if !cpu_decoded.errors().is_empty() || cpu_decoded.groups().len() != 1 {
                    return Err(std::io::Error::other(
                        "CPU benchmark oracle did not produce one successful group",
                    )
                    .into());
                }
                let CpuBatchSamples::U8(cpu_pixels) = cpu_decoded.groups()[0].samples() else {
                    return Err(std::io::Error::other("CPU benchmark oracle was not RGB8").into());
                };
                let expected_scores = average_cpu(
                    cpu_pixels,
                    spec.shape()[0],
                    spec.shape()[1],
                    spec.shape()[2],
                )?;

                for path in PATHS {
                    let (scores, _) = run_path(path, &mut decoder, &program, &prepared)?;
                    validate_scores(path, &scores, &expected_scores)?;
                }

                let mut durations: [Vec<f64>; 4] =
                    core::array::from_fn(|_| Vec::with_capacity(iterations));
                let mut checksums = [0.0_f64; 4];
                for sample_index in 0..iterations {
                    for step in 0..PATHS.len() {
                        let path_index = (sample_index + step) % PATHS.len();
                        let path = PATHS[path_index];
                        let (scores, elapsed) = run_path(path, &mut decoder, &program, &prepared)?;
                        validate_scores(path, &scores, &expected_scores)?;
                        checksums[path_index] += scores
                            .into_iter()
                            .map(std::hint::black_box)
                            .map(f64::from)
                            .sum::<f64>();
                        durations[path_index].push(elapsed.as_secs_f64() * 1_000.0);
                    }
                }

                let mut statistics = [(0.0_f64, 0.0_f64, 0.0_f64); 4];
                for (path_index, path) in PATHS.into_iter().enumerate() {
                    let (mean, low, high) = summarize(&durations[path_index]);
                    println!(
                        "{codec},{size},{batch},{path},{mean:.3},{low:.3},{high:.3},{:.6}",
                        checksums[path_index],
                    );
                    statistics[path_index] = (mean, low, high);
                }

                let staged = statistics[0];
                let direct = statistics[2];
                let qualifies =
                    iterations >= 30 && direct.0 <= staged.0 * 0.90 && direct.2 < staged.1;
                println!(
                "{codec},{size},{batch},speed_claim_qualified,{qualifies},unsupported,unsupported,0"
            );
            }
        }
    }
    Ok(())
}

#[cfg(not(all(target_arch = "aarch64", target_os = "macos")))]
fn main() {
    println!("codec,size,batch,path,mean_ms,ci95_low_ms,ci95_high_ms,checksum");
    for codec in ["htj2k", "j2k"] {
        for size in [512, 1024] {
            for batch in [1, 8, 32] {
                for path in ["staged", "completed", "pipelined", "submit_latency"] {
                    println!("{codec},{size},{batch},{path},unsupported,unsupported,unsupported,0");
                }
            }
        }
    }
}
