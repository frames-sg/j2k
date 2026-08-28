// SPDX-License-Identifier: MIT OR Apache-2.0

//! Opt-in HTJ2K encoder interoperability mode within the encode comparator.

use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use j2k::{wrap_j2k_codestream, J2kFileWrapOptions};

use super::sample_stats;

mod codec;
use codec::{
    decode_with_j2k, decode_with_openjph, discover_openjph_tools, encode_with_j2k,
    encode_with_openjph, validate_ht_profile,
};
mod metrics;
use metrics::{max_sample_delta, psnr};
mod source;
use source::generated_source;
mod types;
use types::{
    matrix_cells, EncodedSample, MatrixCell, MatrixContext, Producer, Profile, SourceImage,
};

const WIDTH: u32 = 128;
const HEIGHT: u32 = 96;
const QFACTOR: u8 = 90;
const DEFAULT_REPEATS: usize = 3;

pub(super) fn run() -> Result<(), String> {
    let (compress, expand) = discover_openjph_tools()?;
    let context = MatrixContext {
        compress,
        expand,
        work_dir: matrix_work_dir()?,
        repeats: matrix_repeats()?,
    };

    println!("openjph_compress_bin\t{}", context.compress.display());
    println!("openjph_expand_bin\t{}", context.expand.display());
    println!("encode_repeats\t{}", context.repeats);
    println!(
        "producer\tformat\tprofile\tcontainer\tcomponents\tbit_depth\tencode_median_us\tencoded_bytes\tpsnr_db\tmax_cross_decoder_delta\tlossless_source_exact"
    );

    for cell in matrix_cells() {
        run_cell(&context, cell)?;
    }
    println!("matrix_complete\ttrue");
    Ok(())
}

fn run_cell(context: &MatrixContext, cell: MatrixCell) -> Result<(), String> {
    let source = generated_source(cell.format);
    let source_path = context.work_dir.join(format!(
        "{}.{}",
        cell.format.label(),
        cell.format.pnm_extension()
    ));
    fs::write(&source_path, &source.pnm_bytes)
        .map_err(|error| format!("write {}: {error}", source_path.display()))?;
    for producer in [Producer::J2k, Producer::OpenJph] {
        run_producer(context, cell, &source, &source_path, producer)?;
    }
    Ok(())
}

fn run_producer(
    context: &MatrixContext,
    cell: MatrixCell,
    source: &SourceImage,
    source_path: &Path,
    producer: Producer,
) -> Result<(), String> {
    let encoded = measure_encode(context, producer, cell.profile, source, source_path)?;
    validate_ht_profile(&encoded.codestream, cell.profile)?;
    let jph = wrap_j2k_codestream(&encoded.codestream, J2kFileWrapOptions::jph())
        .map_err(|error| format!("wrap JPH: {error}"))?;
    for (container, bytes) in [
        ("j2c", encoded.codestream.as_slice()),
        ("jph", jph.as_slice()),
    ] {
        run_container_row(&ContainerRow {
            context,
            cell,
            source,
            producer,
            encoded: &encoded,
            container,
            bytes,
        })?;
    }
    Ok(())
}

struct ContainerRow<'a> {
    context: &'a MatrixContext,
    cell: MatrixCell,
    source: &'a SourceImage,
    producer: Producer,
    encoded: &'a EncodedSample,
    container: &'a str,
    bytes: &'a [u8],
}

fn run_container_row(row: &ContainerRow<'_>) -> Result<(), String> {
    let input_path = row.context.work_dir.join(format!(
        "{}_{}_{}.{}",
        row.producer.label(),
        row.source.format.label(),
        row.cell.profile.label(),
        row.container
    ));
    fs::write(&input_path, row.bytes)
        .map_err(|error| format!("write {}: {error}", input_path.display()))?;
    let j2k_decoded = decode_with_j2k(row.bytes, row.source)?;
    let openjph_decoded = decode_with_openjph(
        &row.context.expand,
        &input_path,
        &row.context.work_dir,
        row.producer,
        row.cell,
        row.container,
    )?;
    let cross_delta = max_sample_delta(
        &j2k_decoded,
        &openjph_decoded,
        row.source.format.bit_depth(),
    )?;
    let source_exact =
        j2k_decoded == row.source.pixels_le && openjph_decoded == row.source.pixels_le;
    validate_parity(
        row.cell,
        row.source,
        row.producer,
        row.container,
        cross_delta,
        source_exact,
    )?;
    let psnr_db = psnr(
        &row.source.pixels_le,
        &j2k_decoded,
        row.source.format.bit_depth(),
    )?;
    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{}\t{}\t{}",
        row.producer.label(),
        row.source.format.label(),
        row.cell.profile.label(),
        row.container,
        row.source.format.components(),
        row.source.format.bit_depth(),
        row.encoded.median_us,
        row.bytes.len(),
        if psnr_db.is_infinite() {
            "inf".to_string()
        } else {
            format!("{psnr_db:.3}")
        },
        cross_delta,
        source_exact
    );
    Ok(())
}

fn validate_parity(
    cell: MatrixCell,
    source: &SourceImage,
    producer: Producer,
    container: &str,
    cross_delta: u32,
    source_exact: bool,
) -> Result<(), String> {
    if cell.profile == Profile::Lossless && !source_exact {
        return Err(format!(
            "{} {} {container} did not round-trip losslessly through both decoders",
            producer.label(),
            source.format.label()
        ));
    }
    if cell.profile == Profile::Qfactor90 && cross_delta > 1 {
        return Err(format!(
            "{} {} {container} cross-decoder delta {cross_delta} exceeds one sample value",
            producer.label(),
            source.format.label()
        ));
    }
    Ok(())
}

fn measure_encode(
    context: &MatrixContext,
    producer: Producer,
    profile: Profile,
    source: &SourceImage,
    source_path: &Path,
) -> Result<EncodedSample, String> {
    let mut samples_us = Vec::with_capacity(context.repeats);
    let mut codestream = Vec::new();
    for index in 0..context.repeats {
        let start = Instant::now();
        codestream = match producer {
            Producer::J2k => encode_with_j2k(source, profile)?,
            Producer::OpenJph => encode_with_openjph(
                &context.compress,
                source_path,
                &context.work_dir,
                source,
                profile,
                index,
            )?,
        };
        samples_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);
    }
    Ok(EncodedSample {
        codestream,
        median_us: sample_stats(&samples_us)?.median,
    })
}

fn matrix_repeats() -> Result<usize, String> {
    let repeats = std::env::var("J2K_OPENJPH_MATRIX_REPEATS")
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|error| format!("invalid J2K_OPENJPH_MATRIX_REPEATS: {error}"))
        })
        .transpose()?
        .unwrap_or(DEFAULT_REPEATS);
    if repeats == 0 {
        return Err("J2K_OPENJPH_MATRIX_REPEATS must be positive".to_string());
    }
    Ok(repeats)
}

fn matrix_work_dir() -> Result<PathBuf, String> {
    let path = PathBuf::from("target")
        .join("j2k-openjph-encode-matrix")
        .join(std::process::id().to_string());
    fs::create_dir_all(&path).map_err(|error| format!("create {}: {error}", path.display()))?;
    Ok(path)
}
