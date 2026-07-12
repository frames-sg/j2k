// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicU64, Ordering},
        OnceLock,
    },
};

use j2k_core::{Downscale, PixelFormat};

use super::{Container, FixtureCase};

static OPENJPH_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
static KAKADU_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);
static OPENJPH_EXPAND_PROGRAM: OnceLock<Option<PathBuf>> = OnceLock::new();
static KAKADU_EXPAND_PROGRAM: OnceLock<Option<PathBuf>> = OnceLock::new();

pub(super) fn decode_openjph_once(case: &FixtureCase, input: &[u8]) -> Result<Vec<u8>, String> {
    let Some(program) = openjph_expand_program() else {
        return Err("ojph_expand is unavailable".to_string());
    };
    let temp_dir = openjph_temp_dir()?;
    let token = OPENJPH_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let input_path = temp_dir.join(format!(
        "{}_{}_input.{}",
        std::process::id(),
        token,
        openjph_input_extension(case.container)
    ));
    let output_path = temp_dir.join(format!(
        "{}_{}_output.{}",
        std::process::id(),
        token,
        openjph_output_extension(case.format)
    ));
    let result = (|| {
        fs::write(&input_path, input).map_err(|error| {
            format!(
                "write OpenJPH staged input {}: {error}",
                input_path.display()
            )
        })?;
        let mut command = Command::new(program);
        command
            .arg("-i")
            .arg(&input_path)
            .arg("-o")
            .arg(&output_path);
        if case.operation.scale() != Downscale::None {
            let reduce = reduce_factor(case.operation.scale())?;
            command.arg("-skip_res").arg(format!("{reduce},{reduce}"));
        }
        let output = command
            .output()
            .map_err(|error| format!("start ojph_expand: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "ojph_expand exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        read_cli_pnm_output("OpenJPH", &output_path, case.format)
    })();
    cleanup_cli_temp(&input_path, result.is_ok())?;
    cleanup_cli_temp(&output_path, result.is_ok())?;
    result
}

pub(super) fn decode_kakadu_once(case: &FixtureCase, input: &[u8]) -> Result<Vec<u8>, String> {
    let Some(program) = kakadu_expand_program() else {
        return Err("kdu_expand is unavailable".to_string());
    };
    let temp_dir = kakadu_temp_dir()?;
    let token = KAKADU_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let input_path = temp_dir.join(format!(
        "{}_{}_input.{}",
        std::process::id(),
        token,
        openjph_input_extension(case.container)
    ));
    let output_path = temp_dir.join(format!(
        "{}_{}_output.{}",
        std::process::id(),
        token,
        openjph_output_extension(case.format)
    ));
    let result = (|| {
        fs::write(&input_path, input).map_err(|error| {
            format!(
                "write Kakadu staged input {}: {error}",
                input_path.display()
            )
        })?;
        let mut command = Command::new(program);
        command
            .arg("-i")
            .arg(&input_path)
            .arg("-o")
            .arg(&output_path);
        if case.operation.scale() != Downscale::None {
            command
                .arg("-reduce")
                .arg(reduce_factor(case.operation.scale())?.to_string());
        }
        let output = command
            .output()
            .map_err(|error| format!("start kdu_expand: {error}"))?;
        if !output.status.success() {
            return Err(format!(
                "kdu_expand exited with {}: {}",
                output.status,
                String::from_utf8_lossy(&output.stderr).trim()
            ));
        }
        read_cli_pnm_output("Kakadu", &output_path, case.format)
    })();
    cleanup_cli_temp(&input_path, result.is_ok())?;
    cleanup_cli_temp(&output_path, result.is_ok())?;
    result
}

pub(super) fn openjph_is_available() -> bool {
    openjph_expand_program().is_some()
}

pub(super) fn openjph_command_label() -> String {
    openjph_expand_program().map_or_else(
        || "not found".to_string(),
        |program| program.display().to_string(),
    )
}

pub(super) fn openjph_version_label() -> &'static str {
    if openjph_is_available() {
        "available-version-not-reported-by-ojph_expand"
    } else {
        "unavailable"
    }
}

pub(super) fn kakadu_is_available() -> bool {
    kakadu_expand_program().is_some()
}

pub(super) fn kakadu_command_label() -> String {
    kakadu_expand_program().map_or_else(
        || "not found".to_string(),
        |program| program.display().to_string(),
    )
}

pub(super) fn kakadu_version_label() -> &'static str {
    if kakadu_is_available() {
        "available-version-not-reported-by-kdu_expand"
    } else {
        "unavailable"
    }
}

fn openjph_temp_dir() -> Result<PathBuf, String> {
    let dir = std::env::current_dir()
        .map_err(|error| format!("current_dir: {error}"))?
        .join("target")
        .join("j2k-openjph-expand");
    fs::create_dir_all(&dir).map_err(|error| format!("create {}: {error}", dir.display()))?;
    Ok(dir)
}

fn kakadu_temp_dir() -> Result<PathBuf, String> {
    let dir = std::env::current_dir()
        .map_err(|error| format!("current_dir: {error}"))?
        .join("target")
        .join("j2k-kakadu-expand");
    fs::create_dir_all(&dir).map_err(|error| format!("create {}: {error}", dir.display()))?;
    Ok(dir)
}

fn cleanup_cli_temp(path: &Path, fail_on_cleanup_error: bool) -> Result<(), String> {
    if !path.exists() {
        return Ok(());
    }
    match fs::remove_file(path) {
        Err(error) if fail_on_cleanup_error => {
            Err(format!("remove temp file {}: {error}", path.display()))
        }
        Ok(()) | Err(_) => Ok(()),
    }
}

fn openjph_input_extension(container: Container) -> &'static str {
    match container {
        Container::RawCodestream => "j2c",
        Container::Jp2 => "jp2",
        Container::Jph => "jph",
        Container::Jhc => "jhc",
    }
}

fn openjph_output_extension(format: PixelFormat) -> &'static str {
    match format {
        PixelFormat::Gray8 => "pgm",
        PixelFormat::Rgb8 => "ppm",
        _ => "pnm",
    }
}

fn read_cli_pnm_output(
    tool_label: &str,
    path: &Path,
    format: PixelFormat,
) -> Result<Vec<u8>, String> {
    let image = image::ImageReader::open(path)
        .map_err(|error| format!("open {tool_label} output {}: {error}", path.display()))?
        .with_guessed_format()
        .map_err(|error| format!("guess {tool_label} output {}: {error}", path.display()))?
        .decode()
        .map_err(|error| format!("decode {tool_label} output {}: {error}", path.display()))?;
    match format {
        PixelFormat::Gray8 => Ok(image.into_luma8().into_raw()),
        PixelFormat::Rgb8 => Ok(image.into_rgb8().into_raw()),
        other => Err(format!(
            "{tool_label} output format {other:?} is unsupported"
        )),
    }
}

fn openjph_expand_program() -> Option<&'static PathBuf> {
    OPENJPH_EXPAND_PROGRAM
        .get_or_init(discover_openjph_expand_program)
        .as_ref()
}

fn discover_openjph_expand_program() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("J2K_OPENJPH_EXPAND_BIN").map(PathBuf::from) {
        return command_is_runnable(&path).then_some(path);
    }
    [
        PathBuf::from("/opt/homebrew/bin/ojph_expand"),
        PathBuf::from("/usr/local/bin/ojph_expand"),
        PathBuf::from("ojph_expand"),
    ]
    .into_iter()
    .find(|candidate| command_is_runnable(candidate))
}

fn command_is_runnable(program: &Path) -> bool {
    Command::new(program).output().is_ok()
}

fn kakadu_expand_program() -> Option<&'static PathBuf> {
    KAKADU_EXPAND_PROGRAM
        .get_or_init(discover_kakadu_expand_program)
        .as_ref()
}

fn discover_kakadu_expand_program() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("J2K_KDU_EXPAND_BIN").map(PathBuf::from) {
        return command_is_runnable(&path).then_some(path);
    }
    [
        PathBuf::from("/opt/homebrew/bin/kdu_expand"),
        PathBuf::from("/usr/local/bin/kdu_expand"),
        PathBuf::from("kdu_expand"),
    ]
    .into_iter()
    .find(|candidate| command_is_runnable(candidate))
}

pub(super) fn reduce_factor(scale: Downscale) -> Result<u32, String> {
    match scale {
        Downscale::None => Ok(0),
        Downscale::Half => Ok(1),
        Downscale::Quarter => Ok(2),
        Downscale::Eighth => Ok(3),
        _ => Err(format!(
            "unsupported downscale for external comparator: {scale:?}"
        )),
    }
}

#[cfg(test)]
mod tests;
