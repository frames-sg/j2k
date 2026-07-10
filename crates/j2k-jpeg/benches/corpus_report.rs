// SPDX-License-Identifier: MIT OR Apache-2.0

#[expect(
    dead_code,
    reason = "forced target audit found 60 comparison-only support items; the compare bench compiles this module with dead_code unsuppressed"
)]
mod common;
#[path = "common/report.rs"]
mod report;

use common::{
    centered_roi,
    classification::{should_compare_full_frame, CorpusInputClass},
    j2k_decode, j2k_decode_region, j2k_decode_region_scaled, j2k_decode_rows, j2k_decode_scaled,
    j2k_decode_tile_batch_region_scaled, j2k_decode_tile_batch_scaled, j2k_inspect,
    jpeg_decoder_decode, jpeg_decoder_decode_batch_region_scaled, jpeg_decoder_decode_batch_scaled,
    jpeg_decoder_decode_region, jpeg_decoder_decode_region_scaled, jpeg_decoder_decode_scaled,
    jpeg_decoder_inspect, load_bench_inputs, scaled_rect, zune_decode,
    zune_decode_batch_region_scaled, zune_decode_batch_scaled, zune_decode_region,
    zune_decode_region_scaled, zune_decode_scaled, zune_inspect, BenchInput, DecodeMode,
};
use j2k_jpeg::{Downscale, Rect};
use report::{
    escape_csv, escape_markdown_table_cell, nanos_as_secs, report_iterations, report_ratio,
    write_reports,
};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

const ROI_SIDE: u32 = 256;
const TILE_BATCH: usize = 64;
const DEFAULT_ITERS: usize = 3;
const TIE_THRESHOLD: f64 = 0.01;

impl DecodeMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Gray => "gray",
            Self::Rgb => "rgb",
        }
    }
}

impl CorpusInputClass {
    fn as_str(self) -> &'static str {
        match self {
            Self::BoundedFullFrame => "bounded_full_frame",
            Self::VeryLarge => "very_large",
        }
    }
}

fn main() {
    let mut inputs = load_bench_inputs();
    if std::env::var_os("J2K_BENCH_INPUTS").is_some() {
        inputs.retain(|input| !input.name.starts_with("repo/"));
    }
    inputs.sort_by(|lhs, rhs| {
        lhs.input_class
            .cmp(&rhs.input_class)
            .then_with(|| lhs.mode.cmp(&rhs.mode))
            .then_with(|| lhs.name.cmp(&rhs.name))
    });
    let iterations = report_iterations(DEFAULT_ITERS);

    let mut rows = Vec::new();
    for input in &inputs {
        rows.extend(run_input(input, iterations));
    }

    let csv = render_csv(&rows);
    let markdown = render_markdown(&rows, iterations);
    let paths = write_reports("target/bench-reports", "corpus-report", &csv, &markdown);

    println!("Wrote {}", paths.csv.display());
    println!("Wrote {}", paths.markdown.display());
    println!();
    println!("{markdown}");
}

#[derive(Clone, Copy)]
enum Operation {
    Inspect,
    DecodeRgb,
    DecodeGray,
    DecodeRowsRgb,
    WsiRegionRgb,
    WsiScaledRgbQ4,
    WsiScaledRgbQ8,
    WsiRegionScaledRgbQ4,
    WsiRegionScaledRgbQ8,
    WsiTileBatchScaledRgbQ4,
    WsiTileBatchRegionScaledRgbQ4,
}

impl Operation {
    fn as_str(self) -> &'static str {
        match self {
            Self::Inspect => "inspect",
            Self::DecodeRgb => "decode_rgb",
            Self::DecodeGray => "decode_gray",
            Self::DecodeRowsRgb => "decode_rows_rgb",
            Self::WsiRegionRgb => "wsi_region_rgb",
            Self::WsiScaledRgbQ4 => "wsi_scaled_rgb_q4",
            Self::WsiScaledRgbQ8 => "wsi_scaled_rgb_q8",
            Self::WsiRegionScaledRgbQ4 => "wsi_region_scaled_rgb_q4",
            Self::WsiRegionScaledRgbQ8 => "wsi_region_scaled_rgb_q8",
            Self::WsiTileBatchScaledRgbQ4 => "wsi_tile_batch_scaled_rgb_q4",
            Self::WsiTileBatchRegionScaledRgbQ4 => "wsi_tile_batch_region_scaled_rgb_q4",
        }
    }
}

#[derive(Clone, Copy)]
enum Library {
    J2K,
    JpegDecoder,
    Zune,
}

#[derive(Clone)]
struct Measurement {
    ns: Option<u128>,
    error: Option<String>,
}

impl Measurement {
    fn success(ns: u128) -> Self {
        Self {
            ns: Some(ns),
            error: None,
        }
    }

    fn skipped(reason: &str) -> Self {
        Self {
            ns: None,
            error: Some(reason.to_string()),
        }
    }

    fn failure(message: String) -> Self {
        Self {
            ns: None,
            error: Some(message),
        }
    }
}

struct ReportRow {
    input_name: String,
    mode: DecodeMode,
    input_class: CorpusInputClass,
    operation: Operation,
    j2k: Measurement,
    jpeg_decoder: Measurement,
    zune: Measurement,
}

fn run_input(input: &BenchInput, iterations: usize) -> Vec<ReportRow> {
    let mut rows = vec![run_compare_row(
        input,
        Operation::Inspect,
        iterations_for(input, Operation::Inspect, iterations),
    )];
    match (input.mode, input.input_class) {
        (DecodeMode::Gray, _) if should_compare_full_frame(input.mode, input.input_class) => {
            rows.push(run_compare_row(
                input,
                Operation::DecodeGray,
                iterations_for(input, Operation::DecodeGray, iterations),
            ));
        }
        (DecodeMode::Rgb, CorpusInputClass::BoundedFullFrame) => {
            rows.push(run_compare_row(
                input,
                Operation::DecodeRgb,
                iterations_for(input, Operation::DecodeRgb, iterations),
            ));
            rows.push(run_compare_row(
                input,
                Operation::WsiRegionRgb,
                iterations_for(input, Operation::WsiRegionRgb, iterations),
            ));
            rows.push(run_compare_row(
                input,
                Operation::WsiScaledRgbQ4,
                iterations_for(input, Operation::WsiScaledRgbQ4, iterations),
            ));
            rows.push(run_compare_row(
                input,
                Operation::WsiScaledRgbQ8,
                iterations_for(input, Operation::WsiScaledRgbQ8, iterations),
            ));
            rows.push(run_compare_row(
                input,
                Operation::WsiRegionScaledRgbQ4,
                iterations_for(input, Operation::WsiRegionScaledRgbQ4, iterations),
            ));
            rows.push(run_compare_row(
                input,
                Operation::WsiRegionScaledRgbQ8,
                iterations_for(input, Operation::WsiRegionScaledRgbQ8, iterations),
            ));
            rows.push(run_compare_row(
                input,
                Operation::WsiTileBatchScaledRgbQ4,
                iterations_for(input, Operation::WsiTileBatchScaledRgbQ4, iterations),
            ));
            rows.push(run_compare_row(
                input,
                Operation::WsiTileBatchRegionScaledRgbQ4,
                iterations_for(input, Operation::WsiTileBatchRegionScaledRgbQ4, iterations),
            ));
        }
        (DecodeMode::Rgb, CorpusInputClass::VeryLarge)
            if should_compare_full_frame(input.mode, input.input_class) =>
        {
            rows.push(run_compare_row(
                input,
                Operation::DecodeRgb,
                iterations_for(input, Operation::DecodeRgb, iterations),
            ));
            rows.push(run_j2k_only_row(
                input,
                Operation::DecodeRowsRgb,
                iterations_for(input, Operation::DecodeRowsRgb, iterations),
                "comparator skipped for very large RGB input; report uses j2k decode_rows",
            ));
        }
        (DecodeMode::Rgb, CorpusInputClass::VeryLarge) => {
            rows.push(run_j2k_only_row(
                input,
                Operation::DecodeRowsRgb,
                iterations_for(input, Operation::DecodeRowsRgb, iterations),
                "comparator skipped for very large RGB input; report uses j2k decode_rows",
            ));
        }
        (DecodeMode::Gray, CorpusInputClass::BoundedFullFrame) => {
            unreachable!("bounded grayscale inputs are always compared full-frame")
        }
        (DecodeMode::Gray, CorpusInputClass::VeryLarge) => {}
    }
    rows
}

fn iterations_for(input: &BenchInput, operation: Operation, default_iters: usize) -> usize {
    if input.input_class == CorpusInputClass::VeryLarge {
        return match operation {
            Operation::Inspect => default_iters,
            Operation::DecodeRowsRgb => default_iters.clamp(1, 2),
            _ => 1,
        };
    }
    default_iters
}

fn inner_loops_for(input: &BenchInput, operation: Operation) -> usize {
    if matches!(
        operation,
        Operation::DecodeRowsRgb
            | Operation::WsiTileBatchScaledRgbQ4
            | Operation::WsiTileBatchRegionScaledRgbQ4
    ) {
        return 1;
    }

    if matches!(
        operation,
        Operation::WsiScaledRgbQ4
            | Operation::WsiScaledRgbQ8
            | Operation::WsiRegionScaledRgbQ4
            | Operation::WsiRegionScaledRgbQ8
    ) {
        let Some(output_bytes) = estimated_output_bytes(input, operation) else {
            return 1;
        };
        return if output_bytes <= 512 * 1024 { 8 } else { 1 };
    }

    let Some(output_bytes) = estimated_output_bytes(input, operation) else {
        return match operation {
            Operation::Inspect => 64,
            _ => 1,
        };
    };

    if output_bytes <= 512 * 1024 {
        64
    } else if output_bytes <= 2 * 1024 * 1024 {
        16
    } else if output_bytes <= 8 * 1024 * 1024 {
        8
    } else if output_bytes <= 64 * 1024 * 1024 {
        2
    } else {
        1
    }
}

fn estimated_output_bytes(input: &BenchInput, operation: Operation) -> Option<usize> {
    let bpp = match operation {
        Operation::DecodeGray => 1usize,
        Operation::DecodeRowsRgb
        | Operation::DecodeRgb
        | Operation::WsiRegionRgb
        | Operation::WsiScaledRgbQ4
        | Operation::WsiScaledRgbQ8
        | Operation::WsiRegionScaledRgbQ4
        | Operation::WsiRegionScaledRgbQ8
        | Operation::WsiTileBatchScaledRgbQ4
        | Operation::WsiTileBatchRegionScaledRgbQ4 => 3usize,
        Operation::Inspect => return None,
    };

    let dims = match operation {
        Operation::DecodeRgb | Operation::DecodeGray | Operation::DecodeRowsRgb => input.dimensions,
        Operation::WsiRegionRgb => rect_dims(centered_roi(input.dimensions, ROI_SIDE)),
        Operation::WsiScaledRgbQ4 | Operation::WsiTileBatchScaledRgbQ4 => rect_dims(scaled_rect(
            Rect::full(input.dimensions),
            Downscale::Quarter,
        )),
        Operation::WsiScaledRgbQ8 => {
            rect_dims(scaled_rect(Rect::full(input.dimensions), Downscale::Eighth))
        }
        Operation::WsiRegionScaledRgbQ4 | Operation::WsiTileBatchRegionScaledRgbQ4 => rect_dims(
            scaled_rect(centered_roi(input.dimensions, ROI_SIDE), Downscale::Quarter),
        ),
        Operation::WsiRegionScaledRgbQ8 => rect_dims(scaled_rect(
            centered_roi(input.dimensions, ROI_SIDE),
            Downscale::Eighth,
        )),
        Operation::Inspect => return None,
    };

    usize::try_from(dims.0)
        .ok()
        .zip(usize::try_from(dims.1).ok())
        .and_then(|(width, height)| width.checked_mul(height))
        .and_then(|pixels| pixels.checked_mul(bpp))
}

fn rect_dims(rect: Rect) -> (u32, u32) {
    (rect.w, rect.h)
}

fn run_compare_row(input: &BenchInput, operation: Operation, iterations: usize) -> ReportRow {
    let (j2k, jpeg_decoder, zune) = run_compare_measurements(operation, input, iterations);
    ReportRow {
        input_name: input.name.clone(),
        mode: input.mode,
        input_class: input.input_class,
        operation,
        j2k,
        jpeg_decoder,
        zune,
    }
}

fn run_j2k_only_row(
    input: &BenchInput,
    operation: Operation,
    iterations: usize,
    skip_reason: &str,
) -> ReportRow {
    ReportRow {
        input_name: input.name.clone(),
        mode: input.mode,
        input_class: input.input_class,
        operation,
        j2k: run_measurement(Library::J2K, operation, input, iterations),
        jpeg_decoder: Measurement::skipped(skip_reason),
        zune: Measurement::skipped(skip_reason),
    }
}

fn run_measurement(
    library: Library,
    operation: Operation,
    input: &BenchInput,
    iterations: usize,
) -> Measurement {
    if !is_supported(library, operation, input) {
        return Measurement::skipped("unsupported for this library/input combination");
    }

    let result = catch_unwind(AssertUnwindSafe(|| {
        let mut samples = Vec::with_capacity(iterations);
        run_operation(library, operation, input);
        for _ in 0..iterations {
            let start = Instant::now();
            run_operation(library, operation, input);
            samples.push(start.elapsed().as_nanos());
        }
        samples.sort_unstable();
        samples[samples.len() / 2]
    }));

    match result {
        Ok(ns) => Measurement::success(ns),
        Err(payload) => Measurement::failure(panic_message(payload.as_ref())),
    }
}

fn is_supported(library: Library, operation: Operation, input: &BenchInput) -> bool {
    match operation {
        Operation::Inspect => true,
        Operation::DecodeRgb
        | Operation::WsiScaledRgbQ8
        | Operation::WsiRegionScaledRgbQ8
        | Operation::WsiTileBatchScaledRgbQ4
        | Operation::WsiTileBatchRegionScaledRgbQ4 => input.mode == DecodeMode::Rgb,
        Operation::DecodeGray => input.mode == DecodeMode::Gray,
        Operation::WsiRegionRgb | Operation::WsiScaledRgbQ4 | Operation::WsiRegionScaledRgbQ4 => {
            input.mode == DecodeMode::Rgb && input.input_class == CorpusInputClass::BoundedFullFrame
        }
        Operation::DecodeRowsRgb => {
            matches!(library, Library::J2K)
                && input.mode == DecodeMode::Rgb
                && input.input_class == CorpusInputClass::VeryLarge
        }
    }
}

fn run_compare_measurements(
    operation: Operation,
    input: &BenchInput,
    iterations: usize,
) -> (Measurement, Measurement, Measurement) {
    let mut j2k = MeasurementState::new(
        Library::J2K,
        is_supported(Library::J2K, operation, input),
        iterations,
    );
    let mut jpeg_decoder = MeasurementState::new(
        Library::JpegDecoder,
        is_supported(Library::JpegDecoder, operation, input),
        iterations,
    );
    let mut zune = MeasurementState::new(
        Library::Zune,
        is_supported(Library::Zune, operation, input),
        iterations,
    );
    let mut states = [&mut j2k, &mut jpeg_decoder, &mut zune];
    let inner_loops = inner_loops_for(input, operation);

    for state in &mut states {
        state.warm(operation, input);
    }

    for iteration in 0..iterations {
        for step in 0..states.len() {
            let idx = (iteration + step) % states.len();
            states[idx].measure(operation, input, inner_loops);
        }
    }

    (j2k.finish(), jpeg_decoder.finish(), zune.finish())
}

fn time_operation(
    library: Library,
    operation: Operation,
    input: &BenchInput,
    inner_loops: usize,
) -> Measurement {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let start = Instant::now();
        for _ in 0..inner_loops {
            run_operation(library, operation, input);
        }
        start.elapsed().as_nanos() / inner_loops as u128
    }));

    match result {
        Ok(ns) => Measurement::success(ns),
        Err(payload) => Measurement::failure(panic_message(payload.as_ref())),
    }
}

fn run_operation(library: Library, operation: Operation, input: &BenchInput) {
    match (library, operation) {
        (Library::J2K, Operation::Inspect) => j2k_inspect(&input.bytes),
        (Library::JpegDecoder, Operation::Inspect) => jpeg_decoder_inspect(&input.bytes),
        (Library::Zune, Operation::Inspect) => zune_inspect(&input.bytes),
        (Library::J2K, Operation::DecodeRgb) => j2k_decode(&input.bytes, DecodeMode::Rgb),
        (Library::JpegDecoder, Operation::DecodeRgb | Operation::DecodeGray) => {
            jpeg_decoder_decode(&input.bytes);
        }
        (Library::Zune, Operation::DecodeRgb) => zune_decode(&input.bytes, DecodeMode::Rgb),
        (Library::J2K, Operation::DecodeGray) => j2k_decode(&input.bytes, DecodeMode::Gray),
        (Library::Zune, Operation::DecodeGray) => zune_decode(&input.bytes, DecodeMode::Gray),
        (Library::J2K, Operation::DecodeRowsRgb) => j2k_decode_rows(&input.bytes),
        (Library::J2K, Operation::WsiRegionRgb) => j2k_decode_region(&input.bytes, ROI_SIDE),
        (Library::JpegDecoder, Operation::WsiRegionRgb) => {
            jpeg_decoder_decode_region(&input.bytes, ROI_SIDE);
        }
        (Library::Zune, Operation::WsiRegionRgb) => zune_decode_region(&input.bytes, ROI_SIDE),
        (Library::J2K, Operation::WsiScaledRgbQ4) => {
            j2k_decode_scaled(&input.bytes, Downscale::Quarter);
        }
        (Library::JpegDecoder, Operation::WsiScaledRgbQ4) => {
            jpeg_decoder_decode_scaled(&input.bytes, Downscale::Quarter);
        }
        (Library::Zune, Operation::WsiScaledRgbQ4) => {
            zune_decode_scaled(&input.bytes, Downscale::Quarter);
        }
        (Library::J2K, Operation::WsiScaledRgbQ8) => {
            j2k_decode_scaled(&input.bytes, Downscale::Eighth);
        }
        (Library::JpegDecoder, Operation::WsiScaledRgbQ8) => {
            jpeg_decoder_decode_scaled(&input.bytes, Downscale::Eighth);
        }
        (Library::Zune, Operation::WsiScaledRgbQ8) => {
            zune_decode_scaled(&input.bytes, Downscale::Eighth);
        }
        (Library::J2K, Operation::WsiRegionScaledRgbQ4) => {
            j2k_decode_region_scaled(&input.bytes, ROI_SIDE, Downscale::Quarter);
        }
        (Library::JpegDecoder, Operation::WsiRegionScaledRgbQ4) => {
            jpeg_decoder_decode_region_scaled(&input.bytes, ROI_SIDE, Downscale::Quarter);
        }
        (Library::Zune, Operation::WsiRegionScaledRgbQ4) => {
            zune_decode_region_scaled(&input.bytes, ROI_SIDE, Downscale::Quarter);
        }
        (Library::J2K, Operation::WsiRegionScaledRgbQ8) => {
            j2k_decode_region_scaled(&input.bytes, ROI_SIDE, Downscale::Eighth);
        }
        (Library::JpegDecoder, Operation::WsiRegionScaledRgbQ8) => {
            jpeg_decoder_decode_region_scaled(&input.bytes, ROI_SIDE, Downscale::Eighth);
        }
        (Library::Zune, Operation::WsiRegionScaledRgbQ8) => {
            zune_decode_region_scaled(&input.bytes, ROI_SIDE, Downscale::Eighth);
        }
        (Library::J2K, Operation::WsiTileBatchScaledRgbQ4) => {
            j2k_decode_tile_batch_scaled(&input.bytes, TILE_BATCH, Downscale::Quarter);
        }
        (Library::JpegDecoder, Operation::WsiTileBatchScaledRgbQ4) => {
            jpeg_decoder_decode_batch_scaled(&input.bytes, TILE_BATCH, Downscale::Quarter);
        }
        (Library::Zune, Operation::WsiTileBatchScaledRgbQ4) => {
            zune_decode_batch_scaled(&input.bytes, TILE_BATCH, Downscale::Quarter);
        }
        (Library::J2K, Operation::WsiTileBatchRegionScaledRgbQ4) => {
            j2k_decode_tile_batch_region_scaled(
                &input.bytes,
                TILE_BATCH,
                ROI_SIDE,
                Downscale::Quarter,
            );
        }
        (Library::JpegDecoder, Operation::WsiTileBatchRegionScaledRgbQ4) => {
            jpeg_decoder_decode_batch_region_scaled(
                &input.bytes,
                TILE_BATCH,
                ROI_SIDE,
                Downscale::Quarter,
            );
        }
        (Library::Zune, Operation::WsiTileBatchRegionScaledRgbQ4) => {
            zune_decode_batch_region_scaled(&input.bytes, TILE_BATCH, ROI_SIDE, Downscale::Quarter);
        }
        _ => unreachable!("unsupported operation dispatched after validation"),
    }
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    if let Some(message) = payload.downcast_ref::<&'static str>() {
        return (*message).to_string();
    }
    "panic without string payload".to_string()
}

fn render_csv(rows: &[ReportRow]) -> String {
    let mut csv = String::from(
        "input,mode,class,operation, j2k_ns,jpeg_decoder_ns,zune_ns, j2k_error,jpeg_decoder_error,zune_error,fastest\n",
    );
    for row in rows {
        let fastest = fastest_label(row).unwrap_or("n/a");
        writeln!(
            csv,
            "\"{}\",{},{},{},{},{},{},\"{}\",\"{}\",\"{}\",{}",
            escape_csv(&row.input_name),
            row.mode.as_str(),
            row.input_class.as_str(),
            row.operation.as_str(),
            render_ns(&row.j2k),
            render_ns(&row.jpeg_decoder),
            render_ns(&row.zune),
            escape_csv(row.j2k.error.as_deref().unwrap_or("")),
            escape_csv(row.jpeg_decoder.error.as_deref().unwrap_or("")),
            escape_csv(row.zune.error.as_deref().unwrap_or("")),
            fastest
        )
        .expect("writing CSV to a String cannot fail");
    }
    csv
}

fn render_markdown(rows: &[ReportRow], iterations: usize) -> String {
    let mut summary = BTreeMap::<&'static str, Summary>::new();
    for row in rows {
        summary
            .entry(row.operation.as_str())
            .or_default()
            .accumulate(row);
    }

    let mut md = String::new();
    md.push_str("# J2K JPEG corpus report\n\n");
    writeln!(
        md,
        "- inputs: {}",
        rows.iter()
            .map(|row| &row.input_name)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    )
    .expect("writing Markdown to a String cannot fail");
    writeln!(md, "- rows: {}", rows.len()).expect("writing Markdown to a String cannot fail");
    writeln!(md, "- iterations per measurement: {iterations}")
        .expect("writing Markdown to a String cannot fail");
    writeln!(md, "- tie threshold: {:.0}%\n", TIE_THRESHOLD * 100.0)
        .expect("writing Markdown to a String cannot fail");
    md.push_str("## Summary by operation\n\n");
    md.push_str("| operation | j2k fastest | vs jpeg wins | vs jpeg losses | vs zune wins | vs zune losses | failures |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|\n");
    for (operation, stats) in &summary {
        writeln!(
            md,
            "| {} | {} | {} | {} | {} | {} | {} |",
            operation,
            stats.j2k_fastest,
            stats.vs_jpeg_wins,
            stats.vs_jpeg_losses,
            stats.vs_zune_wins,
            stats.vs_zune_losses,
            stats.failures,
        )
        .expect("writing Markdown to a String cannot fail");
    }
    md.push_str("\n## Rows where j2k is not fastest\n\n");
    md.push_str("| input | operation | j2k | jpeg-decoder | zune-jpeg | fastest |\n");
    md.push_str("|---|---|---:|---:|---:|---|\n");
    let mut any_slower = false;
    for row in rows {
        let fastest = fastest_label(row);
        if fastest == Some("j2k") || fastest.is_none() {
            continue;
        }
        any_slower = true;
        writeln!(
            md,
            "| {} | {} | {} | {} | {} | {} |",
            escape_markdown_table_cell(&row.input_name),
            escape_markdown_table_cell(row.operation.as_str()),
            escape_markdown_table_cell(&format_measurement(&row.j2k)),
            escape_markdown_table_cell(&format_measurement(&row.jpeg_decoder)),
            escape_markdown_table_cell(&format_measurement(&row.zune)),
            escape_markdown_table_cell(fastest.unwrap_or("n/a")),
        )
        .expect("writing Markdown to a String cannot fail");
    }
    if !any_slower {
        md.push_str("| none | — | — | — | — | — |\n");
    }

    md.push_str("\n## Failures / skips\n\n");
    md.push_str("| input | operation | j2k | jpeg-decoder | zune-jpeg |\n");
    md.push_str("|---|---|---|---|---|\n");
    let mut any_failures = false;
    for row in rows {
        if row.j2k.error.is_none() && row.jpeg_decoder.error.is_none() && row.zune.error.is_none() {
            continue;
        }
        any_failures = true;
        writeln!(
            md,
            "| {} | {} | {} | {} | {} |",
            escape_markdown_table_cell(&row.input_name),
            escape_markdown_table_cell(row.operation.as_str()),
            escape_markdown_table_cell(row.j2k.error.as_deref().unwrap_or("ok")),
            escape_markdown_table_cell(row.jpeg_decoder.error.as_deref().unwrap_or("ok")),
            escape_markdown_table_cell(row.zune.error.as_deref().unwrap_or("ok")),
        )
        .expect("writing Markdown to a String cannot fail");
    }
    if !any_failures {
        md.push_str("| none | — | — | — | — |\n");
    }

    md
}

#[derive(Default)]
struct Summary {
    j2k_fastest: usize,
    vs_jpeg_wins: usize,
    vs_jpeg_losses: usize,
    vs_zune_wins: usize,
    vs_zune_losses: usize,
    failures: usize,
}

struct MeasurementState {
    library: Library,
    samples: Vec<u128>,
    error: Option<String>,
    supported: bool,
}

impl MeasurementState {
    fn new(library: Library, supported: bool, iterations: usize) -> Self {
        Self {
            library,
            samples: Vec::with_capacity(iterations),
            error: None,
            supported,
        }
    }

    fn warm(&mut self, operation: Operation, input: &BenchInput) {
        if !self.supported || self.error.is_some() {
            return;
        }
        let measurement = time_operation(self.library, operation, input, 1);
        if let Some(message) = measurement.error {
            self.error = Some(message);
        }
    }

    fn measure(&mut self, operation: Operation, input: &BenchInput, inner_loops: usize) {
        if !self.supported || self.error.is_some() {
            return;
        }
        match time_operation(self.library, operation, input, inner_loops) {
            Measurement {
                ns: Some(ns),
                error: None,
            } => self.samples.push(ns),
            Measurement {
                error: Some(message),
                ..
            } => self.error = Some(message),
            Measurement {
                ns: None,
                error: None,
            } => self.error = Some("measurement without result".to_string()),
        }
    }

    fn finish(mut self) -> Measurement {
        if !self.supported {
            return Measurement::skipped("unsupported for this library/input combination");
        }
        if let Some(message) = self.error {
            return Measurement::failure(message);
        }
        self.samples.sort_unstable();
        Measurement::success(self.samples[self.samples.len() / 2])
    }
}

impl Summary {
    fn accumulate(&mut self, row: &ReportRow) {
        if fastest_label(row) == Some("j2k") {
            self.j2k_fastest += 1;
        }
        match compare_measurements(&row.j2k, &row.jpeg_decoder) {
            Some(Ordering::Less) => self.vs_jpeg_wins += 1,
            Some(Ordering::Greater) => self.vs_jpeg_losses += 1,
            _ => {}
        }
        match compare_measurements(&row.j2k, &row.zune) {
            Some(Ordering::Less) => self.vs_zune_wins += 1,
            Some(Ordering::Greater) => self.vs_zune_losses += 1,
            _ => {}
        }
        if row.j2k.error.is_some() || row.jpeg_decoder.error.is_some() || row.zune.error.is_some() {
            self.failures += 1;
        }
    }
}

fn fastest_label(row: &ReportRow) -> Option<&'static str> {
    let mut best_ns: Option<u128> = None;
    for measurement in [&row.j2k, &row.jpeg_decoder, &row.zune] {
        let Some(ns) = measurement.ns else {
            continue;
        };
        best_ns = Some(best_ns.map_or(ns, |best| best.min(ns)));
    }
    let best_ns = best_ns?;
    if row.j2k.ns.is_some_and(|ns| {
        let max_ns = ns.max(best_ns);
        max_ns > 0 && report_ratio(ns.abs_diff(best_ns), max_ns) <= TIE_THRESHOLD
    }) {
        return Some("j2k");
    }
    if row.jpeg_decoder.ns.is_some_and(|ns| ns == best_ns) {
        return Some("jpeg-decoder");
    }
    if row.zune.ns.is_some_and(|ns| ns == best_ns) {
        return Some("zune-jpeg");
    }
    None
}

fn compare_measurements(lhs: &Measurement, rhs: &Measurement) -> Option<Ordering> {
    let (Some(lhs_ns), Some(rhs_ns)) = (lhs.ns, rhs.ns) else {
        return None;
    };
    let max_ns = lhs_ns.max(rhs_ns);
    if max_ns > 0 && report_ratio(lhs_ns.abs_diff(rhs_ns), max_ns) <= TIE_THRESHOLD {
        return Some(Ordering::Equal);
    }
    Some(lhs_ns.cmp(&rhs_ns))
}

fn format_measurement(measurement: &Measurement) -> String {
    if let Some(ns) = measurement.ns {
        format_ns(ns)
    } else {
        measurement
            .error
            .clone()
            .unwrap_or_else(|| "n/a".to_string())
    }
}

fn render_ns(measurement: &Measurement) -> String {
    measurement.ns.map_or_else(String::new, |ns| ns.to_string())
}

fn format_ns(ns: u128) -> String {
    if ns >= 1_000_000 {
        format!("{:.3} ms", nanos_as_secs(ns) * 1_000.0)
    } else if ns >= 1_000 {
        format!("{:.3} µs", nanos_as_secs(ns) * 1_000_000.0)
    } else {
        format!("{ns} ns")
    }
}
