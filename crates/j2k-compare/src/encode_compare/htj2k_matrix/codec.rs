// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

use j2k::{
    encode_j2k_lossless, encode_j2k_lossy, EncodeBackendPreference, J2kBlockCodingMode, J2kDecoder,
    J2kEncodeValidation, J2kLosslessEncodeOptions, J2kLosslessSamples, J2kLossyEncodeOptions,
    J2kLossySamples,
};
use j2k_core::PixelFormat;

use super::super::tools::path_lookup;
use super::{
    source::read_pnm_as_le,
    types::{MatrixCell, Producer, Profile, SampleFormat, SourceImage},
    HEIGHT, QFACTOR, WIDTH,
};

pub(super) fn discover_openjph_tools() -> Result<(PathBuf, PathBuf), String> {
    Ok((
        required_tool(
            "J2K_OPENJPH_COMPRESS_BIN",
            "ojph_compress",
            "target/reference/openjph-0.31.0/build-reference/src/apps/ojph_compress/ojph_compress",
        )?,
        required_tool(
            "J2K_OPENJPH_EXPAND_BIN",
            "ojph_expand",
            "target/reference/openjph-0.31.0/build-reference/src/apps/ojph_expand/ojph_expand",
        )?,
    ))
}

fn required_tool(env_name: &str, program: &str, local_fallback: &str) -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os(env_name).map(PathBuf::from) {
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "{env_name} does not identify a file: {}",
            path.display()
        ));
    }
    let local_fallback = PathBuf::from(local_fallback);
    if local_fallback.is_file() {
        return Ok(local_fallback);
    }
    path_lookup(program)
        .filter(|path| path.is_file())
        .ok_or_else(|| {
            format!(
            "{program} is unavailable; set {env_name} or run scripts/prepare-openjph-reference.sh"
        )
        })
}

pub(super) fn encode_with_j2k(source: &SourceImage, profile: Profile) -> Result<Vec<u8>, String> {
    match profile {
        Profile::Lossless => {
            let samples = J2kLosslessSamples::new(
                &source.pixels_le,
                WIDTH,
                HEIGHT,
                source.format.components(),
                source.format.bit_depth(),
                false,
            )
            .map_err(|error| error.to_string())?;
            let options = J2kLosslessEncodeOptions::default()
                .with_backend(EncodeBackendPreference::CpuOnly)
                .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
                .with_max_decomposition_levels(Some(1))
                .with_validation(J2kEncodeValidation::External);
            encode_j2k_lossless(samples, &options)
                .map(|encoded| encoded.codestream)
                .map_err(|error| error.to_string())
        }
        Profile::Qfactor90 => {
            let samples = J2kLossySamples::new(
                &source.pixels_le,
                WIDTH,
                HEIGHT,
                source.format.components(),
                source.format.bit_depth(),
                false,
            )
            .map_err(|error| error.to_string())?;
            let options = J2kLossyEncodeOptions::default()
                .with_backend(EncodeBackendPreference::CpuOnly)
                .with_block_coding_mode(J2kBlockCodingMode::HighThroughput)
                .with_max_decomposition_levels(Some(1))
                .with_qfactor(Some(QFACTOR))
                .with_validation(J2kEncodeValidation::External);
            encode_j2k_lossy(samples, &options)
                .map(|encoded| encoded.codestream)
                .map_err(|error| error.to_string())
        }
    }
}

pub(super) fn encode_with_openjph(
    compress: &Path,
    source_path: &Path,
    work_dir: &Path,
    source: &SourceImage,
    profile: Profile,
    index: usize,
) -> Result<Vec<u8>, String> {
    let output = work_dir.join(format!(
        "openjph_{}_{}_{}.j2c",
        source.format.label(),
        profile.label(),
        index
    ));
    let mut command = Command::new(compress);
    command
        .arg("-i")
        .arg(source_path)
        .arg("-o")
        .arg(&output)
        .arg("-num_decomps")
        .arg("1")
        .arg("-block_size")
        .arg("{64,64}")
        .arg("-prog_order")
        .arg("LRCP")
        .arg("-colour_trans")
        .arg(if source.format == SampleFormat::Rgb8 {
            "true"
        } else {
            "false"
        });
    match profile {
        Profile::Lossless => {
            command.arg("-reversible").arg("true");
        }
        Profile::Qfactor90 => {
            command
                .arg("-reversible")
                .arg("false")
                .arg("-qfactor")
                .arg(QFACTOR.to_string());
        }
    }
    let result = command
        .output()
        .map_err(|error| format!("start ojph_compress: {error}"))?;
    if !result.status.success() {
        return Err(format!(
            "ojph_compress exited with {}: {}",
            result.status,
            String::from_utf8_lossy(&result.stderr).trim()
        ));
    }
    fs::read(&output).map_err(|error| format!("read {}: {error}", output.display()))
}

pub(super) fn validate_ht_profile(codestream: &[u8], profile: Profile) -> Result<(), String> {
    let header = j2k_native::inspect_j2k_codestream_header(codestream)
        .map_err(|error| format!("inspect encoded HTJ2K profile: {error}"))?;
    if !header.high_throughput {
        return Err("encoder matrix output did not use HT block coding".to_string());
    }
    let expect_reversible = profile == Profile::Lossless;
    if header.reversible != expect_reversible {
        return Err(format!(
            "encoder matrix reversible={} but expected {expect_reversible}",
            header.reversible
        ));
    }
    if header.resolution_levels != 2 {
        return Err(format!(
            "encoder matrix resolution levels {} != 2",
            header.resolution_levels
        ));
    }
    Ok(())
}

pub(super) fn decode_with_j2k(bytes: &[u8], source: &SourceImage) -> Result<Vec<u8>, String> {
    let format = match source.format {
        SampleFormat::Gray8 => PixelFormat::Gray8,
        SampleFormat::Rgb8 => PixelFormat::Rgb8,
        SampleFormat::Gray16 => PixelFormat::Gray16,
    };
    let stride = usize::try_from(WIDTH).map_err(|_| "width exceeds usize".to_string())?
        * format.bytes_per_pixel();
    let height = usize::try_from(HEIGHT).map_err(|_| "height exceeds usize".to_string())?;
    let mut output = vec![0_u8; stride * height];
    J2kDecoder::new(bytes)
        .map_err(|error| error.to_string())?
        .decode_into(&mut output, stride, format)
        .map_err(|error| error.to_string())?;
    Ok(output)
}

pub(super) fn decode_with_openjph(
    expand: &Path,
    input: &Path,
    work_dir: &Path,
    producer: Producer,
    cell: MatrixCell,
    container: &str,
) -> Result<Vec<u8>, String> {
    let output = work_dir.join(format!(
        "decoded_{}_{}_{}_{}.{}",
        producer.label(),
        cell.format.label(),
        cell.profile.label(),
        container,
        cell.format.pnm_extension()
    ));
    let result = Command::new(expand)
        .arg("-i")
        .arg(input)
        .arg("-o")
        .arg(&output)
        .output()
        .map_err(|error| format!("start ojph_expand: {error}"))?;
    if !result.status.success() {
        return Err(format!(
            "ojph_expand exited with {}: {}",
            result.status,
            String::from_utf8_lossy(&result.stderr).trim()
        ));
    }
    read_pnm_as_le(&output, cell.format)
}
