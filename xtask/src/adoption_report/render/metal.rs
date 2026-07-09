//! Metal-specific adoption benchmark summaries.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::markdown::{markdown_header, markdown_row};

use super::{metric_label, numeric_field, value_field};

#[derive(Default)]
struct MetalAutoGroup {
    rows: usize,
    cpu_ms_total: f64,
    auto_ms_total: f64,
}

#[derive(Default)]
struct MetalResidentGroup {
    rows: usize,
    cpu_ms_total: f64,
    hybrid_ms_total: f64,
    hybrid_rows: usize,
    resident_host_ms_total: f64,
    resident_buffer_ms_total: f64,
    host_readback_ms_total: f64,
}

#[derive(Default)]
struct MetalDecodeGroup {
    rows: usize,
    cpu_ms_total: f64,
    resident_ms_total: f64,
    readback_ms_total: f64,
}

#[derive(Default)]
struct MetalTranscodeGroup {
    rows: usize,
    total_ms_total: f64,
    transfer_bytes_total: u64,
    dct_handoffs_total: u64,
    dwt_handoffs_total: u64,
    dispatches_total: u64,
    tiles_total: u64,
}

pub(super) fn metal_decode_summary(out: &mut String, metal: &Value) {
    let Some(rows) = metal.get("benches").and_then(Value::as_array) else {
        out.push_str("\nNo Metal decode benchmark rows recorded.\n");
        return;
    };
    let mut groups =
        BTreeMap::<(String, String, String, String, String, String), MetalDecodeGroup>::new();
    for row in rows {
        let Some(cpu_ms) = numeric_field(row, "cpu_ms") else {
            continue;
        };
        let Some(resident_ms) = numeric_field(row, "metal_resident_ms") else {
            continue;
        };
        let Some(readback_ms) = numeric_field(row, "metal_readback_ms") else {
            continue;
        };
        let key = (
            metal_decode_source_category(row),
            value_field(row, "codec"),
            value_field(row, "container"),
            value_field(row, "operation"),
            value_field(row, "fmt"),
            value_field(row, "size"),
        );
        let group = groups.entry(key).or_default();
        group.rows += 1;
        group.cpu_ms_total += cpu_ms;
        group.resident_ms_total += resident_ms;
        group.readback_ms_total += readback_ms;
    }
    if groups.is_empty() {
        out.push_str("\nNo measured Metal decode benchmark rows recorded.\n");
        return;
    }

    out.push_str("\nMetal decode row summary:\n\n");
    let columns = [
        "source",
        "codec",
        "container",
        "operation",
        "fmt",
        "size",
        "rows",
        "cpu_ms_avg",
        "metal_resident_ms_avg",
        "metal_readback_ms_avg",
        "readback_vs_cpu",
        "winner",
    ];
    markdown_header(out, &columns);
    for ((source, codec, container, operation, fmt, size), group) in groups {
        let rows = group.rows as f64;
        let cpu_avg = group.cpu_ms_total / rows;
        let resident_avg = group.resident_ms_total / rows;
        let readback_avg = group.readback_ms_total / rows;
        let ratio = readback_avg / cpu_avg;
        let winner = if readback_avg < cpu_avg {
            "metal-readback"
        } else if cpu_avg < readback_avg {
            "cpu"
        } else {
            "tie"
        };
        markdown_row(
            out,
            [
                source,
                codec,
                container,
                operation,
                fmt,
                size,
                group.rows.to_string(),
                format!("{cpu_avg:.3}"),
                format!("{resident_avg:.3}"),
                format!("{readback_avg:.3}"),
                format!("{ratio:.3}x"),
                winner.to_string(),
            ],
        );
    }
}

fn metal_decode_source_category(row: &Value) -> String {
    let source = value_field(row, "source");
    if source.starts_with("external:") {
        "external".to_string()
    } else {
        source
    }
}

pub(super) fn metal_auto_summary(out: &mut String, metal: &Value) {
    let Some(rows) = metal.get("auto_benches").and_then(Value::as_array) else {
        out.push_str("\nNo Metal auto benchmark rows recorded.\n");
        return;
    };
    let mut groups = BTreeMap::<(String, String, String, String), MetalAutoGroup>::new();
    for row in rows {
        let Some(cpu_ms) = numeric_field(row, "cpu_ms") else {
            continue;
        };
        let Some(auto_ms) = numeric_field(row, "auto_ms") else {
            continue;
        };
        let key = (
            value_field(row, "mode"),
            value_field(row, "codec"),
            value_field(row, "components"),
            value_field(row, "size"),
        );
        let group = groups.entry(key).or_default();
        group.rows += 1;
        group.cpu_ms_total += cpu_ms;
        group.auto_ms_total += auto_ms;
    }
    if groups.is_empty() {
        out.push_str("\nNo measured Metal auto benchmark rows recorded.\n");
        return;
    }

    out.push_str("\nMetal auto external row summary:\n\n");
    let columns = [
        "mode",
        "codec",
        "components",
        "size",
        "rows",
        "cpu_ms_avg",
        "metal_auto_ms_avg",
        "metal_vs_cpu",
        "winner",
    ];
    markdown_header(out, &columns);
    for ((mode, codec, components, size), group) in groups {
        let cpu_avg = group.cpu_ms_total / group.rows as f64;
        let auto_avg = group.auto_ms_total / group.rows as f64;
        let ratio = auto_avg / cpu_avg;
        let winner = if auto_avg < cpu_avg {
            "metal-auto"
        } else if cpu_avg < auto_avg {
            "cpu"
        } else {
            "tie"
        };
        markdown_row(
            out,
            [
                mode,
                codec,
                components,
                size,
                group.rows.to_string(),
                format!("{cpu_avg:.3}"),
                format!("{auto_avg:.3}"),
                format!("{ratio:.3}x"),
                winner.to_string(),
            ],
        );
    }
}

pub(super) fn metal_resident_summary(out: &mut String, metal: &Value) {
    let Some(rows) = metal.get("resident_benches").and_then(Value::as_array) else {
        out.push_str("\nNo Metal resident benchmark rows recorded.\n");
        return;
    };
    let mut groups =
        BTreeMap::<(String, String, String, String, String), MetalResidentGroup>::new();
    for row in rows {
        if row.get("packetization_used").and_then(Value::as_bool) != Some(true)
            || row.get("codestream_assembly_used").and_then(Value::as_bool) != Some(true)
        {
            continue;
        }
        let Some(cpu_ms) = numeric_field(row, "cpu_ms") else {
            continue;
        };
        let Some(resident_host_ms) = numeric_field(row, "resident_host_ms") else {
            continue;
        };
        let Some(resident_buffer_ms) = numeric_field(row, "resident_buffer_ms") else {
            continue;
        };
        let key = (
            value_field(row, "mode"),
            value_field(row, "codec"),
            value_field(row, "components"),
            value_field(row, "size"),
            value_field(row, "batch_size"),
        );
        let group = groups.entry(key).or_default();
        group.rows += 1;
        group.cpu_ms_total += cpu_ms;
        group.resident_host_ms_total += resident_host_ms;
        group.resident_buffer_ms_total += resident_buffer_ms;
        if let Some(hybrid_ms) = numeric_field(row, "hybrid_cpu_packet_ms") {
            group.hybrid_rows += 1;
            group.hybrid_ms_total += hybrid_ms;
        }
        if let Some(host_readback_ms) = numeric_field(row, "host_readback_ms") {
            group.host_readback_ms_total += host_readback_ms;
        }
    }
    if groups.is_empty() {
        out.push_str("\nNo verified Metal resident packetization rows recorded.\n");
        return;
    }

    out.push_str("\nMetal resident packetization summary:\n\n");
    let columns = [
        "mode",
        "codec",
        "components",
        "size",
        "batch_size",
        "rows",
        "cpu_ms_avg",
        "hybrid_cpu_packet_ms_avg",
        "resident_host_ms_avg",
        "resident_buffer_ms_avg",
        "host_readback_ms_avg",
        "resident_host_vs_cpu",
        "winner",
    ];
    markdown_header(out, &columns);
    for ((mode, codec, components, size, batch_size), group) in groups {
        let rows = group.rows as f64;
        let cpu_avg = group.cpu_ms_total / rows;
        let resident_host_avg = group.resident_host_ms_total / rows;
        let resident_buffer_avg = group.resident_buffer_ms_total / rows;
        let host_readback_avg = group.host_readback_ms_total / rows;
        let hybrid_avg =
            (group.hybrid_rows > 0).then(|| group.hybrid_ms_total / group.hybrid_rows as f64);
        let ratio = resident_host_avg / cpu_avg;
        let winner = if resident_host_avg < cpu_avg {
            "resident-host"
        } else if cpu_avg < resident_host_avg {
            "cpu"
        } else {
            "tie"
        };
        markdown_row(
            out,
            [
                mode,
                codec,
                components,
                size,
                batch_size,
                group.rows.to_string(),
                format!("{cpu_avg:.3}"),
                metric_label(hybrid_avg),
                format!("{resident_host_avg:.3}"),
                format!("{resident_buffer_avg:.3}"),
                format!("{host_readback_avg:.3}"),
                format!("{ratio:.3}x"),
                winner.to_string(),
            ],
        );
    }
}

pub(super) fn metal_transcode_summary(out: &mut String, metal: &Value) {
    let Some(rows) = metal.get("profiles").and_then(Value::as_array) else {
        out.push_str("\nNo Metal transcode profile rows recorded.\n");
        return;
    };
    let mut groups = BTreeMap::<(String, String, String, String), MetalTranscodeGroup>::new();
    for row in rows {
        let Some(total_us) = numeric_field(row, "total_us") else {
            continue;
        };
        let key = (
            value_field(row, "context"),
            value_field(row, "request"),
            value_field(row, "transform_processor"),
            value_field(row, "pipeline"),
        );
        let group = groups.entry(key).or_default();
        group.rows += 1;
        group.total_ms_total += total_us / 1000.0;
        group.transfer_bytes_total += integer_field(row, "host_to_device_transfer_bytes")
            .unwrap_or(0)
            + integer_field(row, "device_to_host_transfer_bytes").unwrap_or(0);
        group.dct_handoffs_total +=
            integer_field(row, "dwt97_batch_resident_dct_handoff_count").unwrap_or(0);
        group.dwt_handoffs_total +=
            integer_field(row, "dwt97_batch_resident_dwt_handoff_count").unwrap_or(0);
        group.dispatches_total += integer_field(row, "accelerator_dispatches").unwrap_or(0);
        group.tiles_total += integer_field(row, "successful_tiles").unwrap_or(0);
    }
    if groups.is_empty() {
        out.push_str("\nNo measured Metal transcode profile rows recorded.\n");
        return;
    }

    out.push_str("\nMetal transcode profile summary:\n\n");
    let columns = [
        "context",
        "request",
        "transform_processor",
        "pipeline",
        "rows",
        "total_ms_avg",
        "successful_tiles",
        "dct_handoffs",
        "dwt_handoffs",
        "accelerator_dispatches",
        "transfer_bytes",
    ];
    markdown_header(out, &columns);
    for ((context, request, transform_processor, pipeline), group) in groups {
        markdown_row(
            out,
            [
                context,
                request,
                transform_processor,
                pipeline,
                group.rows.to_string(),
                format!("{:.3}", group.total_ms_total / group.rows as f64),
                group.tiles_total.to_string(),
                group.dct_handoffs_total.to_string(),
                group.dwt_handoffs_total.to_string(),
                group.dispatches_total.to_string(),
                group.transfer_bytes_total.to_string(),
            ],
        );
    }
}

fn integer_field(row: &Value, key: &str) -> Option<u64> {
    match row.get(key)? {
        Value::Number(number) => number.as_u64(),
        Value::String(value) => value.parse::<u64>().ok(),
        _ => None,
    }
}
