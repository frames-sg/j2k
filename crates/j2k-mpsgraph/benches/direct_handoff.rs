// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
#[allow(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::too_many_lines,
    reason = "the benchmark keeps one explicit equivalent-work matrix and converts small sample counts for statistical summaries"
)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use core::{ffi::c_void, ptr::NonNull};
    use std::{sync::Arc, time::Instant};

    use j2k::{BatchDecodeOptions, BatchLayout, EncodedImage};
    use j2k_mpsgraph::{MpsGraphBatchDecoder, MpsGraphProgram, MpsGraphTensorSpec};
    use j2k_test_support::{gpu_bench_rgb8, htj2k_rgb8_fixture};
    use objc2::AnyThread;
    use objc2_foundation::{NSArray, NSDictionary, NSNumber};
    use objc2_metal_performance_shaders::MPSDataType;
    use objc2_metal_performance_shaders_graph::MPSGraphTensorData;

    fn summarize(samples: &[f64]) -> (f64, f64, f64) {
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        if samples.len() < 2 {
            return (mean, mean, mean);
        }
        let variance = samples
            .iter()
            .map(|sample| (sample - mean).powi(2))
            .sum::<f64>()
            / (samples.len() - 1) as f64;
        let half_width = 1.96 * (variance / samples.len() as f64).sqrt();
        (mean, mean - half_width, mean + half_width)
    }

    fn read_score_sum(data: &MPSGraphTensorData, batch: usize) -> f32 {
        let mut values = vec![0.0_f32; batch];
        // SAFETY: every benchmark graph run is complete and returns one F32
        // score for the first image. Reading that final scalar verifies that
        // every compared path executed equivalent graph work.
        unsafe {
            data.mpsndarray().readBytes_strideBytes(
                NonNull::new(values.as_mut_ptr().cast::<c_void>()).expect("nonempty batch"),
                core::ptr::null_mut(),
            );
        }
        values.into_iter().sum()
    }

    fn staged_iteration(
        decoder: &mut MpsGraphBatchDecoder,
        program: &MpsGraphProgram,
        inputs: &[EncodedImage],
    ) -> Result<f32, Box<dyn std::error::Error>> {
        let decoded = decoder.decode(inputs.to_vec())?;
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
        Ok(read_score_sum(&result, spec.shape()[0]))
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

    let iterations = std::env::var("J2K_MPSGRAPH_BENCH_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(if cfg!(debug_assertions) { 1 } else { 5 })
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
                let prepared = decoder.prepare(inputs.clone())?;
                if prepared.groups().len() != 1 {
                    println!("{codec},{size},{batch},all,unsupported,unsupported,unsupported,0");
                    continue;
                }
                let spec = MpsGraphTensorSpec::from_group_info(
                    prepared.groups()[0].info(),
                    prepared.groups()[0].images().len(),
                )?;
                let program = MpsGraphProgram::rgb8_nhwc_reference(
                    spec.shape()[0],
                    spec.shape()[1],
                    spec.shape()[2],
                )?;
                let mut rows = Vec::new();

                for path in ["staged", "completed", "pipelined", "nonblocking"] {
                    let mut durations = Vec::with_capacity(iterations);
                    let mut checksum = 0.0_f64;
                    for _ in 0..iterations {
                        let started = Instant::now();
                        let score = match path {
                            "staged" => staged_iteration(&mut decoder, &program, &inputs)?,
                            "completed" => {
                                let decoded = decoder.decode(inputs.clone())?;
                                let (mut groups, errors, group_errors) = decoded.into_parts();
                                if !errors.is_empty()
                                    || !group_errors.is_empty()
                                    || groups.len() != 1
                                {
                                    return Err(std::io::Error::other(
                                        "completed path did not produce one group",
                                    )
                                    .into());
                                }
                                let output = program
                                    .submit_completed(decoder.command_queue(), groups.remove(0))?
                                    .wait()?;
                                read_score_sum(&output.results()[0], batch)
                            }
                            "pipelined" => {
                                let output =
                                    decoder.run_prepared_group(&program, &prepared.groups()[0])?;
                                read_score_sum(&output.results()[0], batch)
                            }
                            "nonblocking" => {
                                let submitted = decoder
                                    .submit_prepared_group(&program, &prepared.groups()[0])?;
                                let output = submitted.wait()?;
                                read_score_sum(&output.results()[0], batch)
                            }
                            _ => unreachable!(),
                        };
                        checksum += f64::from(std::hint::black_box(score));
                        durations.push(started.elapsed().as_secs_f64() * 1_000.0);
                    }
                    let (mean, low, high) = summarize(&durations);
                    println!(
                        "{codec},{size},{batch},{path},{mean:.3},{low:.3},{high:.3},{checksum:.6}"
                    );
                    rows.push((path, mean, low, high));
                }

                let staged = rows[0];
                let direct = rows[2];
                let qualifies =
                    iterations >= 2 && direct.1 <= staged.1 * 0.90 && direct.3 < staged.2;
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
                for path in ["staged", "completed", "pipelined", "nonblocking"] {
                    println!("{codec},{size},{batch},{path},unsupported,unsupported,unsupported,0");
                }
            }
        }
    }
}
