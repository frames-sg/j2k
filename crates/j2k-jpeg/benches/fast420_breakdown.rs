// SPDX-License-Identifier: Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::format_push_string,
    clippy::manual_clamp
)]

mod common;

use common::{
    j2k_decode_tile_batch_sequential, libjpeg_turbo_available, libjpeg_turbo_decode_batch,
    load_bench_inputs,
    report::{format_ms, median_ns, report_iterations, write_reports},
    TurboJpegDecoder,
};
use j2k_jpeg::bench_support::{bench_profile_fast420_tile_batch, BenchFast420Profile};

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
        csv.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            row.input_name,
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
        ));
    }
    csv
}

fn render_markdown(rows: &[BreakdownRow], iterations: usize) -> String {
    let mut md = String::new();
    md.push_str("# Fast 4:2:0 Breakdown\n\n");
    md.push_str(&format!(
        "Batch size: {TILE_BATCH} tiles. Median iterations: {iterations}.\n\n"
    ));
    md.push_str("## Summary\n\n");
    md.push_str("| metric | value |\n");
    md.push_str("|---|---:|\n");
    md.push_str(&format!("| profiled inputs | {} |\n", rows.len()));
    if let Some(speedup) = mean_turbo_speedup(rows) {
        md.push_str(&format!("| mean turbo speedup | {speedup:.2}x |\n"));
    } else {
        md.push_str("| mean turbo speedup | n/a |\n");
    }
    let (parse, mcu, rgb, finish, total) = aggregate_stage_ns(rows);
    md.push_str(&format!(
        "| profile parse/plan | {} ({:.1}%) |\n",
        format_ms(parse),
        pct(parse, total)
    ));
    md.push_str(&format!(
        "| profile MCU decode | {} ({:.1}%) |\n",
        format_ms(mcu),
        pct(mcu, total)
    ));
    md.push_str(&format!(
        "| profile RGB emit | {} ({:.1}%) |\n",
        format_ms(rgb),
        pct(rgb, total)
    ));
    md.push_str(&format!(
        "| profile finish | {} ({:.1}%) |\n",
        format_ms(finish),
        pct(finish, total)
    ));
    md.push('\n');

    md.push_str("## Inputs\n\n");
    md.push_str("| input | j2k | turbo | turbo speedup | parse/plan | MCU decode | RGB emit | blocks dc/bhz/general |\n");
    md.push_str("|---|---:|---:|---:|---:|---:|---:|---:|\n");
    for row in rows {
        let counts = row.profile.block_activity_counts();
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} ({:.1}%) | {} ({:.1}%) | {} ({:.1}%) | {}/{}/{} |\n",
            row.input_name,
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
        ));
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

fn mean_turbo_speedup(rows: &[BreakdownRow]) -> Option<f64> {
    let mut count = 0usize;
    let mut total = 0.0f64;
    for row in rows {
        if let Some(turbo) = row.turbo_ns {
            total += row.j2k_ns as f64 / turbo as f64;
            count += 1;
        }
    }
    (count > 0).then(|| total / count as f64)
}

fn ratio(lhs: u128, rhs: u128) -> String {
    format!("{:.3}", lhs as f64 / rhs as f64)
}

fn pct(part: u128, total: u128) -> f64 {
    if total == 0 {
        0.0
    } else {
        part as f64 * 100.0 / total as f64
    }
}
