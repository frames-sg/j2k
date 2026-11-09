// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{hint::black_box, num::NonZeroUsize};

use criterion::Criterion;
use j2k_jpeg::{Decoder, JpegBatchSession, PixelFormat, TileBatchOptions, TileDecodeJob};

use crate::common::{libjpeg_turbo_available, DecodeMode, TurboJpegDecoder};

struct WorkerDecoder(TurboJpegDecoder);

// Without libjpeg-turbo the decoder is a field-less stand-in, which is Send.
#[cfg_attr(
    has_libjpeg_turbo,
    expect(
        clippy::non_send_fields_in_send_ty,
        reason = "the libjpeg-turbo handle moves to one scoped worker at a time, as the SAFETY comment states"
    )
)]
// SAFETY: Each TurboJPEG instance owns independent state. This wrapper transfers
// exclusive access to one scoped worker at a time; no handle is shared between
// concurrent decodes, and all workers join before the handles can be destroyed.
unsafe impl Send for WorkerDecoder {}

#[derive(Clone, Copy)]
struct BatchInput<'a> {
    bytes: &'a [u8],
    width: usize,
    height: usize,
    stride: usize,
    len: usize,
}

impl<'a> BatchInput<'a> {
    fn new(bytes: &'a [u8], channels: usize) -> Self {
        let info = Decoder::inspect(bytes).expect("j2k inspect matched batch");
        let width = info.dimensions.0 as usize;
        let height = info.dimensions.1 as usize;
        let stride = width * channels;
        Self {
            bytes,
            width,
            height,
            stride,
            len: stride * height,
        }
    }
}

/// Comma-separated positive integers from `var`, or from `default` when unset.
fn env_list(var: &str, default: &str) -> Vec<usize> {
    let value = std::env::var(var).unwrap_or_else(|_| String::from(default));
    value
        .split(',')
        .map(|item| {
            item.trim()
                .parse::<usize>()
                .ok()
                .filter(|&n| n > 0)
                .unwrap_or_else(|| panic!("{var}: {item:?} is not a positive integer"))
        })
        .collect()
}

fn report_turbo_build() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "libjpeg_turbo_version\t{}\nlibjpeg_turbo_libdir\t{}",
            option_env!("J2K_LIBJPEG_TURBO_VERSION").unwrap_or("unknown"),
            option_env!("J2K_LIBJPEG_TURBO_LIBDIR").unwrap_or("unknown"),
        );
    });
}

pub(super) fn bench(c: &mut Criterion, name: &str, bytes: &[u8], mode: DecodeMode) {
    let distinct_storage = std::env::var_os("J2K_BENCH_BATCH_DISTINCT_STORAGE").is_some();
    if distinct_storage {
        bench_inputs(c, &format!("{name}_distinct"), &[bytes], mode, true);
    } else {
        bench_inputs(c, name, &[bytes], mode, false);
    }
}

pub(super) fn bench_mixed(c: &mut Criterion, inputs: &[&[u8]], mode: DecodeMode) {
    if inputs.is_empty() {
        return;
    }
    let distinct_storage = std::env::var_os("J2K_BENCH_BATCH_DISTINCT_STORAGE").is_some();
    let storage_name = if distinct_storage {
        "mixed_distinct"
    } else {
        "mixed"
    };
    bench_inputs(
        c,
        &format!("{storage_name}_{}_inputs", inputs.len()),
        inputs,
        mode,
        distinct_storage,
    );
}

fn bench_inputs(
    c: &mut Criterion,
    name: &str,
    inputs: &[&[u8]],
    mode: DecodeMode,
    distinct_storage: bool,
) {
    if !libjpeg_turbo_available() {
        return;
    }
    report_turbo_build();
    let (format, channels, suffix) = match mode {
        DecodeMode::Gray => (PixelFormat::Gray8, 1, "gray"),
        DecodeMode::Rgb => (PixelFormat::Rgb8, 3, "rgb"),
    };
    let inputs = inputs
        .iter()
        .map(|bytes| BatchInput::new(bytes, channels))
        .collect::<Vec<_>>();
    let mut group = c.benchmark_group(format!("decode_batch_{suffix}"));
    for batch_size in env_list("J2K_BENCH_BATCH_SIZES", "16,64") {
        // Limit each codec's retained destinations to 256 MiB when a large
        // single-image corpus is also supplied to this opt-in batch group.
        let selected_inputs = inputs
            .iter()
            .copied()
            .cycle()
            .take(batch_size)
            .collect::<Vec<_>>();
        if selected_inputs
            .iter()
            .try_fold(0usize, |sum, input| sum.checked_add(input.len))
            .is_none_or(|bytes| bytes > 256 * 1024 * 1024)
        {
            continue;
        }
        if distinct_storage
            && selected_inputs
                .iter()
                .try_fold(0usize, |sum, input| sum.checked_add(input.bytes.len()))
                .is_none_or(|bytes| bytes > 256 * 1024 * 1024)
        {
            continue;
        }

        // The opt-in mixed case gives every logical job its own allocation.
        // The output and compressed-input preflights above run before these
        // copies. Keep them immutable and alive across both codec benchmarks,
        // outside every timed iteration.
        let distinct_bytes = distinct_storage.then(|| {
            selected_inputs
                .iter()
                .map(|input| input.bytes.to_vec())
                .collect::<Vec<_>>()
        });
        let batch_inputs = if let Some(distinct_bytes) = &distinct_bytes {
            selected_inputs
                .iter()
                .copied()
                .zip(distinct_bytes)
                .map(|(input, bytes)| BatchInput {
                    bytes: bytes.as_slice(),
                    ..input
                })
                .collect::<Vec<_>>()
        } else {
            selected_inputs
        };
        for workers in env_list("J2K_BENCH_BATCH_WORKERS", "1,2,4") {
            let mut native_outputs = batch_inputs
                .iter()
                .map(|input| vec![0u8; input.len])
                .collect::<Vec<_>>();
            let mut session = JpegBatchSession::new(TileBatchOptions {
                workers: NonZeroUsize::new(workers),
            });
            run_native(&mut session, &mut native_outputs, &batch_inputs, format);
            check_effective_workers(&session, batch_inputs.len(), workers);

            let mut turbo_outputs = batch_inputs
                .iter()
                .map(|input| vec![0u8; input.len])
                .collect::<Vec<_>>();
            // Each handle is set to full-frame, unscaled output once, outside
            // the timed loop; tj3Decompress8 then parses every header itself,
            // so each codec parses each header once per timed decode.
            let mut decoders = (0..workers)
                .map(|_| {
                    let mut decoder = TurboJpegDecoder::new().expect("libjpeg-turbo decoder");
                    decoder
                        .prepare_rgb(batch_inputs[0].bytes)
                        .expect("libjpeg-turbo full-frame setup");
                    WorkerDecoder(decoder)
                })
                .collect::<Vec<_>>();
            run_turbo(&mut decoders, &mut turbo_outputs, &batch_inputs, mode);
            if std::env::var_os("J2K_BENCH_VERIFY_OUTPUTS").is_some() {
                assert_eq!(
                    native_outputs, turbo_outputs,
                    "{name}: matched batch pixels"
                );
            }
            group.bench_function(format!("j2k_workers_{workers}/{batch_size}/{name}"), |b| {
                b.iter(|| run_native(&mut session, &mut native_outputs, &batch_inputs, format));
            });
            group.bench_function(
                format!("libjpeg-turbo_workers_{workers}/{batch_size}/{name}"),
                |b| {
                    b.iter(|| run_turbo(&mut decoders, &mut turbo_outputs, &batch_inputs, mode));
                },
            );
        }
    }
    group.finish();
}

/// Both codecs split a batch into static chunks of `ceil(jobs / workers)`, so
/// the threads actually used can be fewer than requested (24 jobs on 10
/// workers run as 8 chunks of 3). Check j2k uses the same count turbo will.
fn check_effective_workers(session: &JpegBatchSession, jobs: usize, workers: usize) {
    let chunk = jobs.div_ceil(workers);
    let effective_workers = jobs.div_ceil(chunk);
    assert_eq!(
        session.worker_count(),
        effective_workers,
        "batch worker count"
    );
    eprintln!(
        "matched_batch\tjobs={jobs}\trequested_workers={workers}\teffective_workers={effective_workers}"
    );
}

fn run_native(
    session: &mut JpegBatchSession,
    outputs: &mut [Vec<u8>],
    inputs: &[BatchInput<'_>],
    format: PixelFormat,
) {
    let outcomes = {
        let mut jobs = outputs
            .iter_mut()
            .zip(inputs)
            .map(|(out, input)| TileDecodeJob {
                input: input.bytes,
                out: out.as_mut_slice(),
                stride: input.stride,
            })
            .collect::<Vec<_>>();
        session
            .decode_tiles_into(&mut jobs, format)
            .expect("j2k matched batch")
    };
    black_box(outcomes);
    black_box(outputs);
}

fn run_turbo(
    decoders: &mut [WorkerDecoder],
    outputs: &mut [Vec<u8>],
    inputs: &[BatchInput<'_>],
    mode: DecodeMode,
) {
    if decoders.len() == 1 {
        decoders[0].decode_chunk(outputs, inputs, mode);
    } else {
        let worker_count = decoders.len();
        let chunk_size = outputs.len().div_ceil(worker_count);
        // Fixed native worker counts use scoped threads as well, including
        // creation and joining in each timed iteration. One worker runs inline.
        let mut chunks = decoders
            .iter_mut()
            .zip(outputs.chunks_mut(chunk_size))
            .zip(inputs.chunks(chunk_size));
        let ((inline_decoder, inline_outputs), inline_inputs) =
            chunks.next().expect("non-empty TurboJPEG batch");
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(worker_count);
            for ((decoder, chunk), inputs) in chunks {
                handles.push(
                    std::thread::Builder::new()
                        .spawn_scoped(scope, move || {
                            decoder.decode_chunk(chunk, inputs, mode);
                        })
                        .expect("libjpeg-turbo batch worker"),
                );
            }
            inline_decoder.decode_chunk(inline_outputs, inline_inputs, mode);
            for handle in handles {
                handle.join().expect("libjpeg-turbo batch worker");
            }
        });
    }
    black_box(outputs);
}

impl WorkerDecoder {
    fn decode_chunk(
        &mut self,
        outputs: &mut [Vec<u8>],
        inputs: &[BatchInput<'_>],
        mode: DecodeMode,
    ) {
        for (out, input) in outputs.iter_mut().zip(inputs) {
            // Header parsing happens inside tj3Decompress8, once per tile, as
            // j2k parses each header once inside its batch decode. The handles
            // and destination buffers are retained between timed iterations.
            let (width, height) = (input.width, input.height);
            match mode {
                DecodeMode::Gray => {
                    self.0
                        .decode_prepared_gray_into(input.bytes, out, input.stride, width, height)
                }
                DecodeMode::Rgb => {
                    self.0
                        .decode_prepared_rgb_into(input.bytes, out, input.stride, width, height)
                }
            }
            .expect("libjpeg-turbo matched batch");
        }
    }
}
