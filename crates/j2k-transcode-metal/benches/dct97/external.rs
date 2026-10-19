// SPDX-License-Identifier: MIT OR Apache-2.0

use std::time::{Duration, Instant};

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, Throughput};
use j2k_metal::MetalEncodeStageAccelerator;
use j2k_native::{DecodeSettings, Image, RawBitmap};
use j2k_transcode::accelerator::{
    DctGridToDwt97Job, DctToWaveletStageAccelerator, RayonReversibleDwt53Accelerator,
    TranscodeStageError,
};
use j2k_transcode::dev_support::{dct8x8_blocks_then_dwt97_float_with_scratch, Dct97GridScratch};
use j2k_transcode::{
    BatchTranscodeReport, Dwt97TwoDimensional, EncodedTranscodeBatch, JpegTileBatchInput,
    JpegToHtj2kError, JpegToHtj2kOptions, JpegToHtj2kTranscoder, TranscodeTimingReport,
};
use j2k_transcode_metal::MetalDctToWaveletStageAccelerator;
use j2k_transcode_test_support::corpus_validation::{
    load_external_wsi_fixtures, CorpusValidationOptions,
};
use rayon::prelude::*;

#[derive(Clone, Copy)]
enum Profile {
    Lossless53,
    Lossy97,
}

impl Profile {
    const fn label(self) -> &'static str {
        match self {
            Self::Lossless53 => "lossless53",
            Self::Lossy97 => "lossy97",
        }
    }

    fn options(self) -> JpegToHtj2kOptions {
        match self {
            Self::Lossless53 => JpegToHtj2kOptions::lossless_53(),
            Self::Lossy97 => JpegToHtj2kOptions::lossy_97(),
        }
    }
}

// Mirrors wsi-dicom's production CPU 9/7 batch adapter until it has a shared home.
#[derive(Debug, Default)]
struct RayonDwt97BatchAccelerator;

impl DctToWaveletStageAccelerator for RayonDwt97BatchAccelerator {
    fn supports_dwt97_batch(&self) -> bool {
        true
    }

    fn dct_grid_to_dwt97_batch(
        &mut self,
        jobs: &[DctGridToDwt97Job<'_>],
    ) -> Result<Option<Vec<Dwt97TwoDimensional<f64>>>, TranscodeStageError> {
        jobs.par_iter()
            .map(|job| {
                dct8x8_blocks_then_dwt97_float_with_scratch(
                    job.blocks,
                    job.block_cols,
                    job.block_rows,
                    job.width,
                    job.height,
                    &mut Dct97GridScratch::default(),
                )
                .map_err(|source| {
                    TranscodeStageError::backend("cpu", "9/7 DCT batch transform", source)
                })
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Some)
    }
}

pub(super) fn bench_external(c: &mut Criterion) {
    let corpus_options = CorpusValidationOptions::from_env();
    if corpus_options.external_wsi_roots.is_empty() && !corpus_options.require_external_wsi {
        return;
    }
    let mut fixtures = load_external_wsi_fixtures(&corpus_options)
        .expect("load external JPEG-to-HTJ2K benchmark corpus");
    fixtures.dedup_by(|left, right| left.name == right.name);
    if fixtures.is_empty() {
        eprintln!("skipping external JPEG-to-HTJ2K benchmark: corpus is empty");
        return;
    }
    let inputs: Vec<_> = fixtures
        .iter()
        .map(|fixture| JpegTileBatchInput {
            bytes: &fixture.bytes,
        })
        .collect();
    for profile in [Profile::Lossless53, Profile::Lossy97] {
        bench_profile(c, profile, &inputs);
    }
}

fn bench_profile(c: &mut Criterion, profile: Profile, inputs: &[JpegTileBatchInput<'_>]) {
    let options = profile.options();
    let context = format!("{}_batch_{}", profile.label(), inputs.len());
    let mut group = c.benchmark_group("jpeg_to_htj2k_external_distinct_batch");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(inputs.len() as u64));

    let mut transcoder = JpegToHtj2kTranscoder::default();
    let cpu_reference = register_route(
        &mut group,
        profile,
        &context,
        "cpu_default",
        "cpu",
        "cpu",
        None,
        || transcoder.transcode_batch(inputs, &options),
    );

    register_route(
        &mut group,
        profile,
        &context,
        "cpu_tile_parallel",
        "cpu",
        "cpu",
        Some(&cpu_reference),
        || Ok(transcode_tiles_in_parallel(inputs, &options)),
    );

    match profile {
        Profile::Lossless53 => {
            let mut transcoder = JpegToHtj2kTranscoder::default();
            let mut accelerator = RayonReversibleDwt53Accelerator::default();
            register_route(
                &mut group,
                profile,
                &context,
                "cpu_batched",
                "cpu",
                "cpu",
                Some(&cpu_reference),
                || transcoder.transcode_batch_with_accelerator(inputs, &options, &mut accelerator),
            );
        }
        Profile::Lossy97 => {
            let mut transcoder = JpegToHtj2kTranscoder::default();
            let mut accelerator = RayonDwt97BatchAccelerator;
            register_route(
                &mut group,
                profile,
                &context,
                "cpu_batched",
                "cpu",
                "cpu",
                Some(&cpu_reference),
                || transcoder.transcode_batch_with_accelerator(inputs, &options, &mut accelerator),
            );
        }
    }

    if super::metal_available() {
        let mut transcoder = JpegToHtj2kTranscoder::default();
        let mut transform = MetalDctToWaveletStageAccelerator::new_explicit();
        register_route(
            &mut group,
            profile,
            &context,
            "metal_transform_cpu_encode",
            "metal",
            "cpu",
            Some(&cpu_reference),
            || transcoder.transcode_batch_with_accelerator(inputs, &options, &mut transform),
        );

        let mut transcoder = JpegToHtj2kTranscoder::default();
        let mut transform = MetalDctToWaveletStageAccelerator::new_explicit();
        let mut encode = MetalEncodeStageAccelerator::for_ht_code_block_encode();
        register_route(
            &mut group,
            profile,
            &context,
            "metal_transform_metal_ht",
            "metal",
            "metal_ht_cpu_packetize",
            Some(&cpu_reference),
            || {
                transcoder.transcode_batch_with_accelerators(
                    inputs,
                    &options,
                    &mut transform,
                    &mut encode,
                )
            },
        );
    }
    group.finish();
}

#[expect(
    clippy::too_many_arguments,
    reason = "benchmark registration keeps route labels and the comparison oracle explicit"
)]
fn register_route<F>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    profile: Profile,
    context: &str,
    route: &'static str,
    transform_processor: &str,
    encode_processor: &str,
    reference: Option<&[RawBitmap]>,
    mut run: F,
) -> Vec<RawBitmap>
where
    F: FnMut() -> Result<EncodedTranscodeBatch, JpegToHtj2kError>,
{
    drop(run().unwrap_or_else(|error| panic!("{route} warmup failed: {error}")));
    let start = Instant::now();
    let probe = run().unwrap_or_else(|error| panic!("{route} probe failed: {error}"));
    let wall_us = start.elapsed().as_micros();
    assert_success(route, &probe);
    let decoded = decode_batch(route, &probe);
    if let Some(reference) = reference {
        match profile {
            Profile::Lossless53 => assert_decoded_equal(route, reference, &decoded),
            Profile::Lossy97 => {
                let (max_abs_diff, rmse) = decoded_diff(reference, &decoded);
                eprintln!(
                    "j2k_transcode_external_diff context={context} route={route} max_abs_diff={max_abs_diff} rmse={rmse:.6}"
                );
            }
        }
    }
    emit_diagnostics(
        context,
        route,
        transform_processor,
        encode_processor,
        wall_us,
        &probe,
        &decoded,
    );
    group.bench_function(BenchmarkId::new(context, route), move |b| {
        b.iter(|| {
            let batch = run().unwrap_or_else(|error| panic!("{route} benchmark failed: {error}"));
            assert_success(route, &batch);
            std::hint::black_box(batch)
        });
    });
    decoded
}

fn transcode_tiles_in_parallel(
    inputs: &[JpegTileBatchInput<'_>],
    options: &JpegToHtj2kOptions,
) -> EncodedTranscodeBatch {
    let tiles: Vec<_> = inputs
        .par_iter()
        .map_init(JpegToHtj2kTranscoder::default, |transcoder, input| {
            transcoder.transcode(input.bytes, options)
        })
        .collect();
    let successful_tiles = tiles.iter().filter(|tile| tile.is_ok()).count();
    let mut extract_us = 0u128;
    let mut transform_us = 0u128;
    let mut encode_us = 0u128;
    let mut component_count = 0usize;
    for encoded in tiles.iter().filter_map(|tile| tile.as_ref().ok()) {
        extract_us = extract_us.saturating_add(encoded.report.extract_us);
        transform_us = transform_us.saturating_add(encoded.report.transform_us);
        encode_us = encode_us.saturating_add(encoded.report.encode_us);
        component_count = component_count.saturating_add(encoded.report.component_count);
    }
    EncodedTranscodeBatch {
        tiles,
        report: BatchTranscodeReport {
            tile_count: inputs.len(),
            successful_tiles,
            failed_tiles: inputs.len().saturating_sub(successful_tiles),
            transformed_components: component_count,
            reversible_dwt53_batches: 0,
            reversible_dwt53_batch_jobs: 0,
            extract_us,
            transform_us,
            encode_us,
            timings: TranscodeTimingReport {
                jpeg_dct_extract_us: extract_us,
                dct_to_wavelet_total_us: transform_us,
                dct_to_wavelet_cpu_fallback_us: transform_us,
                htj2k_encode_us: encode_us,
                tile_count: successful_tiles,
                component_count,
                cpu_fallback_jobs: component_count,
                ..TranscodeTimingReport::default()
            },
            coefficient_path: options.coefficient_path,
        },
    }
}

fn assert_success(route: &str, batch: &EncodedTranscodeBatch) {
    assert_eq!(
        batch.report.tile_count,
        batch.tiles.len(),
        "{route} tile count"
    );
    assert_eq!(batch.report.failed_tiles, 0, "{route} failed tiles");
    assert_eq!(
        batch.report.successful_tiles,
        batch.tiles.len(),
        "{route} successes"
    );
}

fn decode_batch(route: &str, batch: &EncodedTranscodeBatch) -> Vec<RawBitmap> {
    batch
        .tiles
        .iter()
        .map(|result| {
            let encoded = result
                .as_ref()
                .expect("successful batch contains encoded tile");
            Image::new(&encoded.codestream, &DecodeSettings::default())
                .unwrap_or_else(|error| panic!("{route} output parse failed: {error}"))
                .decode_native()
                .unwrap_or_else(|error| panic!("{route} output decode failed: {error}"))
        })
        .collect()
}

fn assert_decoded_equal(route: &str, reference: &[RawBitmap], actual: &[RawBitmap]) {
    assert_eq!(reference.len(), actual.len(), "{route} decoded count");
    for (index, (expected, actual)) in reference.iter().zip(actual).enumerate() {
        assert_layout(route, index, expected, actual);
        assert_eq!(expected.data, actual.data, "{route} tile {index} pixels");
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "benchmark reports approximate aggregate RMSE"
)]
fn decoded_diff(reference: &[RawBitmap], actual: &[RawBitmap]) -> (u8, f64) {
    assert_eq!(reference.len(), actual.len(), "lossy decoded count");
    let mut max_abs = 0u8;
    let mut squared_sum = 0f64;
    let mut count = 0usize;
    for (index, (expected, actual)) in reference.iter().zip(actual).enumerate() {
        assert_layout("lossy97", index, expected, actual);
        for (&expected, &actual) in expected.data.iter().zip(&actual.data) {
            let diff = expected.abs_diff(actual);
            max_abs = max_abs.max(diff);
            squared_sum += f64::from(diff).powi(2);
            count += 1;
        }
    }
    (max_abs, (squared_sum / count.max(1) as f64).sqrt())
}

fn assert_layout(route: &str, index: usize, left: &RawBitmap, right: &RawBitmap) {
    assert_eq!(
        (
            left.width,
            left.height,
            left.bit_depth,
            left.signed,
            left.num_components,
            left.bytes_per_sample
        ),
        (
            right.width,
            right.height,
            right.bit_depth,
            right.signed,
            right.num_components,
            right.bytes_per_sample
        ),
        "{route} tile {index} layout",
    );
    assert_eq!(
        left.component_signed, right.component_signed,
        "{route} tile {index} signedness"
    );
    assert_eq!(
        left.data.len(),
        right.data.len(),
        "{route} tile {index} bytes"
    );
}

fn emit_diagnostics(
    context: &str,
    route: &str,
    transform_processor: &str,
    encode_processor: &str,
    wall_us: u128,
    batch: &EncodedTranscodeBatch,
    decoded: &[RawBitmap],
) {
    let report = &batch.report;
    let timings = report.timings;
    let report_aggregate_us = report.extract_us + report.transform_us + report.encode_us;
    let encoded_bytes: usize = batch
        .tiles
        .iter()
        .filter_map(|tile| tile.as_ref().ok())
        .map(|tile| tile.codestream.len())
        .sum();
    let decoded_pixels: u64 = decoded
        .iter()
        .map(|tile| u64::from(tile.width) * u64::from(tile.height))
        .sum();
    eprintln!(
        "j2k_transcode_external_profile context={} route={} transform_processor={} encode_processor={} tiles={} decoded_pixels={} encoded_bytes={} wall_us={} report_aggregate_us={} report_aggregates_may_exceed_wall=true extract_us={} transform_us={} encode_us={} transform_dispatches={} transform_jobs={} cpu_fallback_jobs={} encode_dispatches={} encode_ht_dispatches={} upload_transfers={} upload_bytes={} transform_readback_transfers={} transform_readback_bytes={} ht_status_readback_transfers={} ht_output_readback_transfers={}",
        context, route, transform_processor, encode_processor, report.tile_count, decoded_pixels,
        encoded_bytes, wall_us, report_aggregate_us, report.extract_us, report.transform_us,
        report.encode_us, timings.accelerator_dispatches, timings.accelerator_dispatched_jobs,
        timings.cpu_fallback_jobs, timings.htj2k_encode_accelerator_dispatches,
        timings.htj2k_encode_ht_code_block_dispatches, timings.dwt97_batch_pack_upload_transfers,
        timings.dwt97_batch_pack_upload_bytes, timings.dwt97_batch_readback_transfers,
        timings.dwt97_batch_readback_bytes, timings.dwt97_batch_ht_status_readback_transfers,
        timings.dwt97_batch_ht_output_readback_transfers,
    );
}
