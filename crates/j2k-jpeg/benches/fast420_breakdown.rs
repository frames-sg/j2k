// SPDX-License-Identifier: MIT OR Apache-2.0

#[expect(
    dead_code,
    reason = "forced target audit found 59 comparison-only support items; the compare bench compiles this module with dead_code unsuppressed"
)]
mod common;
#[path = "common/report.rs"]
mod report;

use common::{
    libjpeg_turbo_available, libjpeg_turbo_decode_batch, load_bench_inputs, NullSink,
    TurboJpegDecoder,
};
use j2k_jpeg::bench_support::{bench_profile_fast420_tile_batch, BenchFast420Profile};
use j2k_jpeg::{Decoder, DecoderContext, JpegView, ScratchPool};
use report::{
    escape_csv, escape_markdown_table_cell, nanos_as_secs, report_iterations, report_ratio,
    write_reports,
};
use std::fmt::Write as _;
use std::time::Instant;

const TILE_BATCH: usize = 64;
const DEFAULT_ITERS: usize = 3;

struct BreakdownRow {
    input_name: String,
    j2k_ns: u128,
    turbo_ns: Option<u128>,
    profile: BenchFast420Profile,
}

fn main() {
    let mut inputs = load_bench_inputs();
    if std::env::var_os("J2K_BENCH_INPUTS").is_some() {
        inputs.retain(|input| !input.name.starts_with("repo/"));
    }

    let iterations = report_iterations(DEFAULT_ITERS);

    let mut turbo = if libjpeg_turbo_available() {
        Some(TurboJpegDecoder::new().expect("create libjpeg-turbo decoder"))
    } else {
        None
    };

    let mut rows = Vec::new();
    for input in &inputs {
        let Some(profile) = bench_profile_fast420_tile_batch(&input.bytes, TILE_BATCH)
            .unwrap_or_else(|err| panic!("profile {}: {err}", input.name))
        else {
            continue;
        };

        let j2k_ns = median_ns(iterations, || {
            j2k_decode_tile_batch_sequential(&input.bytes, TILE_BATCH);
        });
        let turbo_ns = turbo.as_mut().map(|decoder| {
            median_ns(iterations, || {
                libjpeg_turbo_decode_batch(decoder, &input.bytes, TILE_BATCH);
            })
        });

        rows.push(BreakdownRow {
            input_name: input.name.clone(),
            j2k_ns,
            turbo_ns,
            profile,
        });
    }

    let csv = render_csv(&rows);
    let markdown = render_markdown(&rows, iterations);
    let paths = write_reports("target/bench-reports", "fast420-breakdown", &csv, &markdown);

    println!("Wrote {}", paths.csv.display());
    println!("Wrote {}", paths.markdown.display());
    println!();
    println!("{markdown}");
}

fn render_csv(rows: &[BreakdownRow]) -> String {
    let mut csv = String::from(
        "input, j2k_ns,turbo_ns,turbo_speedup,profile_total_ns,parse_plan_ns,mcu_decode_ns,rgb_emit_ns,finish_ns,total_blocks,dc_only_blocks,bottom_half_zero_blocks,general_blocks\n",
    );
    for row in rows {
        let counts = row.profile.block_activity_counts();
        writeln!(
            csv,
            "\"{}\",{},{},{},{},{},{},{},{},{},{},{},{}",
            escape_csv(&row.input_name),
            row.j2k_ns,
            row.turbo_ns.map_or_else(String::new, |ns| ns.to_string()),
            row.turbo_ns
                .map_or_else(String::new, |turbo| ratio(row.j2k_ns, turbo)),
            row.profile.total_ns(),
            row.profile.parse_plan_ns(),
            row.profile.mcu_decode_ns(),
            row.profile.rgb_emit_ns(),
            row.profile.finish_ns(),
            counts.total_blocks(),
            counts.dc_only_blocks(),
            counts.bottom_half_zero_blocks(),
            counts.general_blocks(),
        )
        .expect("writing CSV to a String cannot fail");
    }
    csv
}

fn render_markdown(rows: &[BreakdownRow], iterations: usize) -> String {
    let mut md = String::new();
    md.push_str("# Fast 4:2:0 Breakdown\n\n");
    writeln!(
        md,
        "Batch size: {TILE_BATCH} tiles. Median iterations: {iterations}.\n"
    )
    .expect("writing Markdown to a String cannot fail");
    md.push_str("## Summary\n\n");
    md.push_str("| metric | value |\n");
    md.push_str("|---|---:|\n");
    writeln!(md, "| profiled inputs | {} |", rows.len())
        .expect("writing Markdown to a String cannot fail");
    if let Some(speedup) = mean_turbo_speedup(rows) {
        writeln!(md, "| mean turbo speedup | {speedup:.2}x |")
            .expect("writing Markdown to a String cannot fail");
    } else {
        md.push_str("| mean turbo speedup | n/a |\n");
    }
    let (parse, mcu, rgb, finish, total) = aggregate_stage_ns(rows);
    writeln!(
        md,
        "| profile parse/plan | {} ({:.1}%) |",
        format_ms(parse),
        pct(parse, total)
    )
    .expect("writing Markdown to a String cannot fail");
    writeln!(
        md,
        "| profile MCU decode | {} ({:.1}%) |",
        format_ms(mcu),
        pct(mcu, total)
    )
    .expect("writing Markdown to a String cannot fail");
    writeln!(
        md,
        "| profile RGB emit | {} ({:.1}%) |",
        format_ms(rgb),
        pct(rgb, total)
    )
    .expect("writing Markdown to a String cannot fail");
    writeln!(
        md,
        "| profile finish | {} ({:.1}%) |",
        format_ms(finish),
        pct(finish, total)
    )
    .expect("writing Markdown to a String cannot fail");
    md.push('\n');

    md.push_str("## Inputs\n\n");
    md.push_str("| input | j2k | turbo | turbo speedup | parse/plan | MCU decode | RGB emit | blocks dc/bhz/general |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|---:|\n");
    for row in rows {
        let counts = row.profile.block_activity_counts();
        writeln!(
            md,
            "| {} | {} | {} | {} | {} ({:.1}%) | {} ({:.1}%) | {} ({:.1}%) | {}/{}/{} |",
            escape_markdown_table_cell(&row.input_name),
            format_ms(row.j2k_ns),
            row.turbo_ns.map_or_else(|| "n/a".to_string(), format_ms),
            row.turbo_ns.map_or_else(
                || "n/a".to_string(),
                |turbo| format!("{}x", ratio(row.j2k_ns, turbo))
            ),
            format_ms(row.profile.parse_plan_ns()),
            pct(row.profile.parse_plan_ns(), row.profile.total_ns()),
            format_ms(row.profile.mcu_decode_ns()),
            pct(row.profile.mcu_decode_ns(), row.profile.total_ns()),
            format_ms(row.profile.rgb_emit_ns()),
            pct(row.profile.rgb_emit_ns(), row.profile.total_ns()),
            counts.dc_only_blocks(),
            counts.bottom_half_zero_blocks(),
            counts.general_blocks(),
        )
        .expect("writing Markdown to a String cannot fail");
    }
    md
}

fn aggregate_stage_ns(rows: &[BreakdownRow]) -> (u128, u128, u128, u128, u128) {
    rows.iter().fold((0, 0, 0, 0, 0), |acc, row| {
        (
            acc.0 + row.profile.parse_plan_ns(),
            acc.1 + row.profile.mcu_decode_ns(),
            acc.2 + row.profile.rgb_emit_ns(),
            acc.3 + row.profile.finish_ns(),
            acc.4 + row.profile.total_ns(),
        )
    })
}

fn j2k_decode_tile_batch_sequential(bytes: &[u8], batch_size: usize) {
    let mut ctx = DecoderContext::new();
    let mut pool = ScratchPool::new();
    let mut sink = NullSink;
    for _ in 0..batch_size {
        let view = JpegView::parse(bytes).expect("j2k parse tile");
        let decoder = Decoder::from_view_in_context(view, &mut ctx).expect("j2k prepare tile");
        decoder
            .decode_rows_with_scratch(&mut pool, &mut sink)
            .expect("j2k decode tile");
    }
}

fn median_ns(iterations: usize, mut f: impl FnMut()) -> u128 {
    f();
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        samples.push(start.elapsed().as_nanos());
    }
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn format_ms(ns: u128) -> String {
    format!("{:.3} ms", nanos_as_secs(ns) * 1_000.0)
}

fn mean_turbo_speedup(rows: &[BreakdownRow]) -> Option<f64> {
    let mut count = 0usize;
    let mut total = 0.0f64;
    for row in rows {
        if let Some(turbo) = row.turbo_ns {
            total += report_ratio(row.j2k_ns, turbo);
            count += 1;
        }
    }
    (count > 0)
        .then(|| total / f64::from(u32::try_from(count).expect("profiled input count fits in u32")))
}

fn ratio(lhs: u128, rhs: u128) -> String {
    format!("{:.3}", report_ratio(lhs, rhs))
}

fn pct(part: u128, total: u128) -> f64 {
    if total == 0 {
        0.0
    } else {
        report_ratio(part, total) * 100.0
    }
}
