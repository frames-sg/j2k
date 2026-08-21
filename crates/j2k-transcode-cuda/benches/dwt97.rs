// SPDX-License-Identifier: MIT OR Apache-2.0

#[cfg(feature = "cuda-runtime")]
use std::time::Duration;

#[cfg(feature = "cuda-runtime")]
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
#[cfg(feature = "cuda-runtime")]
use j2k_jpeg::{
    encode_jpeg_baseline, JpegBackend, JpegEncodeOptions, JpegSamples, JpegSubsampling,
};
#[cfg(feature = "cuda-runtime")]
use j2k_native::{DecodeSettings, Image};
#[cfg(feature = "cuda-runtime")]
use j2k_transcode::accelerator::{
    DctGridI16ToHtj2k97CodeBlockJob, DctToWaveletStageAccelerator, Htj2k97CodeBlockOptions,
    IrreversibleQuantizationSubbandScales, J2kSubBandType, PreencodedHtj2k97Component,
};
#[cfg(feature = "cuda-runtime")]
use j2k_transcode::{
    EncodedTranscodeBatch, JpegTileBatchInput, JpegToHtj2kOptions, JpegToHtj2kTranscoder,
};
#[cfg(feature = "cuda-runtime")]
use j2k_transcode_cuda::CudaDctToWaveletStageAccelerator;
#[cfg(feature = "cuda-runtime")]
use sha2::{Digest, Sha256};

#[cfg(feature = "cuda-runtime")]
const STAGED_LABEL: &str = "staged_column_lift_quantize";
#[cfg(feature = "cuda-runtime")]
const BATCH_SIZE: usize = 16;
#[cfg(feature = "cuda-runtime")]
const DIMENSION: usize = 512;

#[cfg(not(feature = "cuda-runtime"))]
fn main() {
    assert!(
        std::env::var_os("J2K_REQUIRE_CUDA_BENCH").is_none(),
        "CUDA DWT97 benchmark requires the cuda-runtime feature and CUDA hardware"
    );
    eprintln!("CUDA DWT97 benchmark skipped without the cuda-runtime feature");
}

#[cfg(feature = "cuda-runtime")]
fn make_blocks(block_cols: usize, block_rows: usize, salt: usize) -> Vec<[i16; 64]> {
    let mut blocks = vec![[0i16; 64]; block_cols * block_rows];
    for (block_index, block) in blocks.iter_mut().enumerate() {
        for (coefficient_index, coefficient) in block.iter_mut().enumerate() {
            let value = (block_index * 7 + coefficient_index * 3 + salt) % 23;
            *coefficient = i16::try_from(value).expect("coefficient fits i16") - 11;
        }
    }
    blocks
}

#[cfg(feature = "cuda-runtime")]
fn update_hash_len(hash: &mut Sha256, value: usize) {
    hash.update(u64::try_from(value).unwrap_or(u64::MAX).to_le_bytes());
}

#[cfg(feature = "cuda-runtime")]
fn finish_sha256(hash: Sha256) -> String {
    format!("{hash:x}", hash = hash.finalize())
}

#[cfg(feature = "cuda-runtime")]
fn output_sha256(components: &[PreencodedHtj2k97Component]) -> String {
    let mut hash = Sha256::new();
    update_hash_len(&mut hash, components.len());
    for component in components {
        hash.update([component.x_rsiz, component.y_rsiz]);
        update_hash_len(&mut hash, component.resolutions.len());
        for resolution in &component.resolutions {
            update_hash_len(&mut hash, resolution.subbands.len());
            for subband in &resolution.subbands {
                let subband_tag = match subband.sub_band_type {
                    J2kSubBandType::LowLow => 0,
                    J2kSubBandType::HighLow => 1,
                    J2kSubBandType::LowHigh => 2,
                    J2kSubBandType::HighHigh => 3,
                };
                hash.update([subband_tag]);
                hash.update(subband.num_cbs_x.to_le_bytes());
                hash.update(subband.num_cbs_y.to_le_bytes());
                hash.update([subband.total_bitplanes]);
                update_hash_len(&mut hash, subband.code_blocks.len());
                for block in &subband.code_blocks {
                    hash.update(block.width.to_le_bytes());
                    hash.update(block.height.to_le_bytes());
                    hash.update(block.encoded.cleanup_length.to_le_bytes());
                    hash.update(block.encoded.refinement_length.to_le_bytes());
                    hash.update([
                        block.encoded.num_coding_passes,
                        block.encoded.num_zero_bitplanes,
                    ]);
                    update_hash_len(&mut hash, block.encoded.data.len());
                    hash.update(&block.encoded.data);
                }
            }
        }
    }
    finish_sha256(hash)
}

#[cfg(feature = "cuda-runtime")]
fn resident_stage_input_sha256(
    jobs: &[DctGridI16ToHtj2k97CodeBlockJob<'_>],
    options: Htj2k97CodeBlockOptions,
) -> String {
    let mut hash = Sha256::new();
    hash.update(b"j2k-cuda-p13-resident-stage-v1\0");
    update_hash_len(&mut hash, jobs.len());
    for job in jobs {
        for value in [
            job.block_cols,
            job.block_rows,
            job.width,
            job.height,
            job.dequantized_blocks.len(),
        ] {
            update_hash_len(&mut hash, value);
        }
        hash.update([job.x_rsiz, job.y_rsiz]);
        for block in job.dequantized_blocks {
            for coefficient in block {
                hash.update(coefficient.to_le_bytes());
            }
        }
    }
    hash.update([
        options.bit_depth,
        options.guard_bits,
        options.code_block_width_exp,
        options.code_block_height_exp,
    ]);
    for scale in [
        options.irreversible_quantization_scale,
        options.irreversible_quantization_subband_scales.low_low,
        options.irreversible_quantization_subband_scales.high_low,
        options.irreversible_quantization_subband_scales.low_high,
        options.irreversible_quantization_subband_scales.high_high,
    ] {
        hash.update(scale.to_bits().to_le_bytes());
    }
    finish_sha256(hash)
}

#[cfg(feature = "cuda-runtime")]
fn encoded_batch_sha256(batch: &EncodedTranscodeBatch) -> String {
    let mut hash = Sha256::new();
    hash.update(b"j2k-cuda-p13-product-output-v1\0");
    update_hash_len(&mut hash, batch.tiles.len());
    for tile in &batch.tiles {
        let codestream = &tile
            .as_ref()
            .expect("P13 product probe tile succeeds")
            .codestream;
        update_hash_len(&mut hash, codestream.len());
        hash.update(codestream);
    }
    finish_sha256(hash)
}

#[cfg(feature = "cuda-runtime")]
fn jpeg_batch_input_sha256(jpeg: &[u8], batch_size: usize) -> String {
    let mut hash = Sha256::new();
    hash.update(b"j2k-cuda-p13-product-input-v1\0");
    update_hash_len(&mut hash, batch_size);
    for _ in 0..batch_size {
        update_hash_len(&mut hash, jpeg.len());
        hash.update(jpeg);
    }
    finish_sha256(hash)
}

#[cfg(feature = "cuda-runtime")]
fn assert_staged_route(timings: j2k_transcode::accelerator::Dwt97BatchStageTimings) {
    assert!(
        timings.column_lift_us > 0,
        "staged route must report its separate column-lift stage"
    );
    assert!(
        timings.quantize_codeblock_us > 0,
        "staged route must report quantization work"
    );
}

#[cfg(feature = "cuda-runtime")]
fn htj2k97_options() -> Htj2k97CodeBlockOptions {
    Htj2k97CodeBlockOptions {
        bit_depth: 8,
        guard_bits: 2,
        code_block_width_exp: 4,
        code_block_height_exp: 4,
        irreversible_quantization_scale: 2.5,
        irreversible_quantization_subband_scales: IrreversibleQuantizationSubbandScales {
            low_low: 0.9,
            high_low: 1.1,
            low_high: 1.2,
            high_high: 1.5,
        },
    }
}

#[cfg(feature = "cuda-runtime")]
fn emit_resident_stage_probe(
    jobs: &[DctGridI16ToHtj2k97CodeBlockJob<'_>],
    options: Htj2k97CodeBlockOptions,
    output: &[PreencodedHtj2k97Component],
    timings: j2k_transcode::accelerator::Dwt97BatchStageTimings,
) {
    let sample_count = jobs.len() * DIMENSION * DIMENSION;
    let common_float_scratch_bytes = sample_count * std::mem::size_of::<f32>() * 2;
    let temporary_float_band_bytes = sample_count * std::mem::size_of::<f32>();
    let temporary_float_band_traffic_bytes = temporary_float_band_bytes * 2;
    let input_upload_bytes = jobs
        .iter()
        .map(|job| job.dequantized_blocks.len() * 64 * std::mem::size_of::<i16>())
        .sum::<usize>();
    let ht_payload_readback_bytes = output
        .iter()
        .flat_map(|component| &component.resolutions)
        .flat_map(|resolution| &resolution.subbands)
        .flat_map(|subband| &subband.code_blocks)
        .map(|block| block.encoded.data.len())
        .sum::<usize>();
    eprintln!(
        "j2k_cuda_dwt97_stages path={STAGED_LABEL} dimension={DIMENSION} batch={BATCH_SIZE} code_block=64x64 input_sha256={} output_sha256={} input_upload_bytes={input_upload_bytes} common_float_scratch_bytes={common_float_scratch_bytes} temporary_float_band_bytes={temporary_float_band_bytes} temporary_float_band_traffic_bytes={temporary_float_band_traffic_bytes} ht_payload_readback_bytes={ht_payload_readback_bytes} pack_upload_us={} idct_row_lift_us={} column_lift_us={} quantize_codeblock_us={} ht_encode_us={} ht_codeblock_dispatches={} readback_us={}",
        resident_stage_input_sha256(jobs, options),
        output_sha256(output),
        timings.pack_upload_us,
        timings.idct_row_lift_us,
        timings.column_lift_us,
        timings.quantize_codeblock_us,
        timings.ht_encode_us,
        timings.ht_codeblock_dispatches,
        timings.readback_us,
    );
}

#[cfg(feature = "cuda-runtime")]
fn duration_from_micros(micros: u128) -> Duration {
    Duration::from_micros(u64::try_from(micros).unwrap_or(u64::MAX))
}

#[cfg(feature = "cuda-runtime")]
fn bench_dwt97(criterion: &mut Criterion) {
    let block_cols = DIMENSION.div_ceil(8);
    let block_rows = DIMENSION.div_ceil(8);
    let blocks = (0..BATCH_SIZE)
        .map(|salt| make_blocks(block_cols, block_rows, salt))
        .collect::<Vec<_>>();
    let jobs = blocks
        .iter()
        .map(|dequantized_blocks| DctGridI16ToHtj2k97CodeBlockJob {
            dequantized_blocks,
            block_cols,
            block_rows,
            width: DIMENSION,
            height: DIMENSION,
            x_rsiz: 1,
            y_rsiz: 1,
        })
        .collect::<Vec<_>>();
    let options = htj2k97_options();
    let mut accelerator = CudaDctToWaveletStageAccelerator::new_explicit_resident_ht_encode();
    let probe = accelerator
        .dct_grid_i16_to_htj2k97_preencoded_batch(&jobs, options)
        .expect("CUDA DWT97 stage probe succeeds")
        .expect("explicit CUDA handles DWT97 stage probe");
    assert_eq!(probe.len(), jobs.len());
    let timings = accelerator
        .last_dwt97_batch_stage_timings()
        .expect("CUDA DWT97 stage timings are available");
    assert_staged_route(timings);
    emit_resident_stage_probe(&jobs, options, &probe, timings);

    let mut group = criterion.benchmark_group("cuda_dwt97_column_quantize");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Elements(
        u64::try_from(jobs.len() * DIMENSION * DIMENSION).unwrap_or(u64::MAX),
    ));
    group.bench_function("resident_preencode_512x512_batch_16", |bencher| {
        bencher.iter(|| {
            let output = accelerator
                .dct_grid_i16_to_htj2k97_preencoded_batch(std::hint::black_box(&jobs), options)
                .expect("CUDA DWT97 end-to-end benchmark succeeds")
                .expect("explicit CUDA handles DWT97 benchmark batch");
            std::hint::black_box(output);
        });
    });
    group.bench_function("column_quantize_512x512_batch_16", |bencher| {
        bencher.iter_custom(|iterations| {
            let mut measured_us = 0u128;
            for _ in 0..iterations {
                let output = accelerator
                    .dct_grid_i16_to_htj2k97_preencoded_batch(std::hint::black_box(&jobs), options)
                    .expect("CUDA DWT97 kernel-time benchmark succeeds")
                    .expect("explicit CUDA handles DWT97 benchmark batch");
                std::hint::black_box(output);
                let timings = accelerator
                    .last_dwt97_batch_stage_timings()
                    .expect("CUDA DWT97 kernel timings remain available");
                measured_us = measured_us
                    .saturating_add(timings.column_lift_us)
                    .saturating_add(timings.quantize_codeblock_us);
            }
            duration_from_micros(measured_us)
        });
    });
    group.finish();
}

#[cfg(feature = "cuda-runtime")]
fn masked_u8(value: usize) -> u8 {
    u8::try_from(value & 0xff).expect("masked fixture value fits u8")
}

#[cfg(feature = "cuda-runtime")]
fn encoded_product_fixture() -> Vec<u8> {
    let mut rgb = Vec::with_capacity(DIMENSION * DIMENSION * 3);
    for y in 0..DIMENSION {
        for x in 0..DIMENSION {
            rgb.push(masked_u8(x * 5 + y * 3 + 17));
            rgb.push(masked_u8(x * 2 + y * 7 + 41));
            rgb.push(masked_u8(x * 11 + y * 13 + 73));
        }
    }
    encode_jpeg_baseline(
        JpegSamples::Rgb8 {
            data: &rgb,
            width: u32::try_from(DIMENSION).expect("fixture width fits u32"),
            height: u32::try_from(DIMENSION).expect("fixture height fits u32"),
        },
        JpegEncodeOptions {
            quality: 90,
            subsampling: JpegSubsampling::Ybr420,
            restart_interval: Some(
                u16::try_from(DIMENSION / 8).expect("fixture restart interval fits u16"),
            ),
            backend: JpegBackend::Cpu,
        },
    )
    .expect("encode P13 product fixture")
    .data
}

#[cfg(feature = "cuda-runtime")]
fn assert_successful_product(batch: &EncodedTranscodeBatch) {
    assert_eq!(batch.report.failed_tiles, 0, "P13 product tiles succeed");
    assert_eq!(
        batch.report.successful_tiles, batch.report.tile_count,
        "P13 product tile count"
    );
    let timing = &batch.report.timings;
    assert!(
        timing.accelerator_work_observed(),
        "P13 product must execute CUDA work"
    );
    assert_eq!(timing.cpu_fallback_jobs, 0, "P13 product has no fallback");
    assert!(
        timing.dwt97_batch_resident_dwt_handoff_count > 0,
        "P13 product must use the resident i16 handoff"
    );
    assert!(
        timing.dwt97_batch_column_lift_us > 0,
        "staged product route reports separate column lift"
    );
    assert!(
        timing.dwt97_batch_quantize_codeblock_us > 0,
        "P13 product reports column/quantization work"
    );
}

#[cfg(feature = "cuda-runtime")]
fn bench_jpeg_to_htj2k_product(criterion: &mut Criterion) {
    let jpeg = encoded_product_fixture();
    let inputs = vec![JpegTileBatchInput { bytes: &jpeg }; BATCH_SIZE];
    let options = JpegToHtj2kOptions::lossy_97();
    let mut probe_transcoder = JpegToHtj2kTranscoder::default();
    let mut probe_accelerator = CudaDctToWaveletStageAccelerator::new_explicit_resident_ht_encode();
    let probe = probe_transcoder
        .transcode_batch_with_accelerator(&inputs, &options, &mut probe_accelerator)
        .expect("P13 product probe succeeds");
    assert_successful_product(&probe);
    let expected_dimensions = (
        u32::try_from(DIMENSION).expect("fixture width fits u32"),
        u32::try_from(DIMENSION).expect("fixture height fits u32"),
    );
    for tile in &probe.tiles {
        let tile = tile.as_ref().expect("P13 product tile succeeds");
        let decoded = Image::new(&tile.codestream, &DecodeSettings::default())
            .expect("P13 product tile codestream parses")
            .decode_native()
            .expect("P13 product tile codestream decodes");
        assert_eq!((decoded.width, decoded.height), expected_dimensions);
    }
    let chroma_dimension = DIMENSION.div_ceil(2);
    let product_float_samples =
        BATCH_SIZE * (DIMENSION * DIMENSION + 2 * chroma_dimension * chroma_dimension);
    let product_temporary_float_band_bytes = product_float_samples * std::mem::size_of::<f32>();
    let product_temporary_float_band_traffic_bytes = product_temporary_float_band_bytes * 2;
    let timing = &probe.report.timings;
    eprintln!(
        "j2k_cuda_dwt97_product path={STAGED_LABEL} dimension={DIMENSION} batch={BATCH_SIZE} sampling=ybr420 input_sha256={} output_sha256={} input_bytes={} output_tiles={} temporary_float_band_bytes={product_temporary_float_band_bytes} temporary_float_band_traffic_bytes={product_temporary_float_band_traffic_bytes} pack_upload_us={} pack_upload_transfers={} pack_upload_bytes={} idct_row_lift_us={} column_lift_us={} quantize_codeblock_us={} readback_us={} readback_transfers={} readback_bytes={} resident_dwt_handoffs={} accelerator_dispatches={}",
        jpeg_batch_input_sha256(&jpeg, BATCH_SIZE),
        encoded_batch_sha256(&probe),
        jpeg.len() * BATCH_SIZE,
        probe.tiles.len(),
        timing.dwt97_batch_pack_upload_us,
        timing.dwt97_batch_pack_upload_transfers,
        timing.dwt97_batch_pack_upload_bytes,
        timing.dwt97_batch_idct_row_lift_us,
        timing.dwt97_batch_column_lift_us,
        timing.dwt97_batch_quantize_codeblock_us,
        timing.dwt97_batch_readback_us,
        timing.dwt97_batch_readback_transfers,
        timing.dwt97_batch_readback_bytes,
        timing.dwt97_batch_resident_dwt_handoff_count,
        timing.accelerator_dispatches,
    );

    let mut group = criterion.benchmark_group("cuda_p13_jpeg_to_htj2k_product");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(3));
    group.measurement_time(Duration::from_secs(5));
    group.throughput(Throughput::Bytes(
        u64::try_from(jpeg.len() * BATCH_SIZE).unwrap_or(u64::MAX),
    ));
    let mut transcoder = JpegToHtj2kTranscoder::default();
    let mut accelerator = CudaDctToWaveletStageAccelerator::new_explicit_resident_ht_encode();
    group.bench_function("srgb_ybr420_512_batch_16", |bencher| {
        bencher.iter(|| {
            let batch = transcoder
                .transcode_batch_with_accelerator(
                    std::hint::black_box(&inputs),
                    &options,
                    &mut accelerator,
                )
                .expect("P13 product benchmark succeeds");
            assert_successful_product(&batch);
            std::hint::black_box(batch);
        });
    });
    group.finish();
}

#[cfg(feature = "cuda-runtime")]
criterion_group!(benches, bench_dwt97, bench_jpeg_to_htj2k_product);
#[cfg(feature = "cuda-runtime")]
criterion_main!(benches);
