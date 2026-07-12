// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    common, include_kakadu_encoder, Command, EncoderKind, EncoderTool, ImageCase, Path, PathBuf,
    Stdio,
};

pub(super) fn all_encoder_tools() -> Result<Vec<EncoderTool>, String> {
    let current = std::env::current_exe().map_err(|error| format!("current_exe: {error}"))?;
    let openjpeg_program = discover_command(
        "J2K_OPENJPEG_COMPRESS_BIN",
        "opj_compress",
        &[
            "/opt/homebrew/bin/opj_compress",
            "/usr/local/bin/opj_compress",
        ],
    );
    let grok_program = discover_command(
        "J2K_GROK_COMPRESS_BIN",
        "grk_compress",
        &[
            "/opt/homebrew/bin/grk_compress",
            "/usr/local/bin/grk_compress",
        ],
    );
    let kakadu_program = discover_command(
        "J2K_KDU_COMPRESS_BIN",
        "kdu_compress",
        &[
            "/opt/homebrew/bin/kdu_compress",
            "/usr/local/bin/kdu_compress",
        ],
    );
    let mut tools = vec![
        EncoderTool {
            kind: EncoderKind::J2k,
            program: current,
            available: true,
        },
        EncoderTool {
            kind: EncoderKind::OpenJpeg,
            program: openjpeg_program
                .clone()
                .unwrap_or_else(|| PathBuf::from("opj_compress")),
            available: openjpeg_program.is_some(),
        },
        EncoderTool {
            kind: EncoderKind::Grok,
            program: grok_program
                .clone()
                .unwrap_or_else(|| PathBuf::from("grk_compress")),
            available: grok_program.is_some(),
        },
    ];
    if include_kakadu_encoder() {
        tools.push(EncoderTool {
            kind: EncoderKind::Kakadu,
            program: kakadu_program
                .clone()
                .unwrap_or_else(|| PathBuf::from("kdu_compress")),
            available: kakadu_program.is_some(),
        });
    }
    Ok(tools)
}

pub(super) fn selected_encoder_tools(
    all_tools: &[EncoderTool],
) -> Result<Vec<EncoderTool>, String> {
    let Some(selected) = selected_encoder_kinds()? else {
        return Ok(all_tools.to_vec());
    };
    Ok(selected
        .into_iter()
        .filter_map(|kind| all_tools.iter().find(|tool| tool.kind == kind).cloned())
        .collect())
}

pub(super) fn selected_encoder_kinds() -> Result<Option<Vec<EncoderKind>>, String> {
    let Some(value) = std::env::var("J2K_ENCODE_COMPARE_ENCODERS").ok() else {
        return Ok(None);
    };
    let mut kinds = Vec::new();
    for raw in value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let kind = match raw.to_ascii_lowercase().as_str() {
            "j2k" => EncoderKind::J2k,
            "openjpeg" | "opj" => EncoderKind::OpenJpeg,
            "grok" | "grk" => EncoderKind::Grok,
            "kakadu" | "kdu" => EncoderKind::Kakadu,
            other => {
                return Err(format!(
                    "J2K_ENCODE_COMPARE_ENCODERS has unknown encoder {other:?}; expected j2k, openjpeg, grok, or kakadu"
                ));
            }
        };
        if !kinds.contains(&kind) {
            kinds.push(kind);
        }
    }
    if kinds.is_empty() {
        return Err("J2K_ENCODE_COMPARE_ENCODERS must include at least one encoder".to_string());
    }
    Ok(Some(kinds))
}

pub(super) fn discover_command(
    env_name: &str,
    program: &str,
    fallbacks: &[&str],
) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(env_name)
        .map(PathBuf::from)
        .filter(|path| path.exists())
    {
        return Some(path);
    }
    if let Some(path) = path_lookup(program) {
        return Some(path);
    }
    fallbacks
        .iter()
        .map(PathBuf::from)
        .find(|path| path.exists())
}

pub(super) fn path_lookup(program: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(program);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

pub(super) fn run_encoder_once(
    case: &ImageCase,
    tool: &EncoderTool,
    work_dir: &Path,
    suffix: &str,
) -> Result<PathBuf, String> {
    let output = work_dir.join(format!(
        "{}_{}_{}.jp2",
        tool.kind.label(),
        case.name,
        suffix
    ));
    let mut command = Command::new(&tool.program);
    match tool.kind {
        EncoderKind::J2k => {
            command
                .arg("--encode-one")
                .arg("--input")
                .arg(&case.pnm_path)
                .arg("--output")
                .arg(&output);
        }
        EncoderKind::OpenJpeg => {
            command
                .arg("-i")
                .arg(&case.pnm_path)
                .arg("-o")
                .arg(&output)
                .arg("-n")
                .arg("3")
                .arg("-b")
                .arg("64,64")
                .arg("-p")
                .arg("LRCP")
                .arg("-threads")
                .arg("1")
                .env("OPJ_NUM_THREADS", "1");
        }
        EncoderKind::Grok => {
            command
                .arg("-i")
                .arg(&case.pnm_path)
                .arg("-o")
                .arg(&output)
                .arg("-n")
                .arg("3")
                .arg("-b")
                .arg("64,64")
                .arg("-p")
                .arg("LRCP")
                .arg("-H")
                .arg("1");
        }
        EncoderKind::Kakadu => {
            command
                .arg("-i")
                .arg(&case.pnm_path)
                .arg("-o")
                .arg(&output)
                .arg("Creversible=yes")
                .arg("Clevels=2")
                .arg("Cblk={64,64}")
                .arg("Corder=LRCP")
                .arg("-rate")
                .arg("-");
        }
    }
    let status = command
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|error| format!("start {}: {error}", tool.kind.label()))?;
    if !status.success() {
        return Err(format!(
            "{} encoder exited with {status} for {}",
            tool.kind.label(),
            case.name
        ));
    }
    Ok(output)
}

pub(super) fn command_template(encoder: EncoderKind) -> &'static str {
    match encoder {
        EncoderKind::J2k => {
            "jp2k_encode_compare --encode-one --input INPUT.pnm --output OUTPUT.jp2"
        }
        EncoderKind::OpenJpeg => {
            "OPJ_NUM_THREADS=1 opj_compress -i INPUT.pnm -o OUTPUT.jp2 -n 3 -b 64,64 -p LRCP -threads 1"
        }
        EncoderKind::Grok => {
            "grk_compress -i INPUT.pnm -o OUTPUT.jp2 -n 3 -b 64,64 -p LRCP -H 1"
        }
        EncoderKind::Kakadu => {
            "kdu_compress -i INPUT.pnm -o OUTPUT.jp2 Creversible=yes Clevels=2 Cblk={64,64} Corder=LRCP -rate -"
        }
    }
}

pub(super) fn samples_label(samples: &[f64]) -> String {
    samples
        .iter()
        .map(|value| format!("{value:.3}"))
        .collect::<Vec<_>>()
        .join(",")
}

pub(super) fn dimensions_label(width: u32, height: u32) -> String {
    common::dimensions_label(width, height)
}

pub(super) fn tool_available(tools: &[EncoderTool], kind: EncoderKind) -> bool {
    tools.iter().any(|tool| tool.kind == kind && tool.available)
}

pub(super) fn tool_command(tools: &[EncoderTool], kind: EncoderKind) -> String {
    tools.iter().find(|tool| tool.kind == kind).map_or_else(
        || "not found".to_string(),
        |tool| tool.program.display().to_string(),
    )
}

pub(super) fn tool_version(tools: &[EncoderTool], kind: EncoderKind) -> String {
    let Some(tool) = tools.iter().find(|tool| tool.kind == kind) else {
        return "not found".to_string();
    };
    if !tool.available {
        return "unavailable".to_string();
    }
    command_version_label(tool).unwrap_or_else(|error| format!("unavailable:{error}"))
}

pub(super) fn tool_version_available(tools: &[EncoderTool], kind: EncoderKind) -> bool {
    let Some(tool) = tools.iter().find(|tool| tool.kind == kind) else {
        return false;
    };
    tool.available && command_version_label(tool).is_ok()
}

pub(super) fn command_version_label(tool: &EncoderTool) -> Result<String, String> {
    let arg_sets: &[&[&str]] = match tool.kind {
        EncoderKind::J2k => return Ok(env!("CARGO_PKG_VERSION").to_string()),
        EncoderKind::OpenJpeg => &[&["-h"]],
        EncoderKind::Grok => &[&["--help"], &["-h"]],
        EncoderKind::Kakadu => &[&["-usage"], &["-h"]],
    };
    for args in arg_sets {
        let output = Command::new(&tool.program)
            .args(*args)
            .output()
            .map_err(|error| format!("{}:{error}", tool.kind.label()))?;
        let mut text = String::new();
        text.push_str(&String::from_utf8_lossy(&output.stdout));
        text.push_str(&String::from_utf8_lossy(&output.stderr));
        if let Some(line) = extract_version_line(tool.kind, &text) {
            return Ok(line);
        }
    }
    if tool.kind == EncoderKind::Kakadu {
        Ok("available-version-not-reported-by-kdu_compress".to_string())
    } else {
        Err("version-line-not-found".to_string())
    }
}

pub(super) fn extract_version_line(kind: EncoderKind, text: &str) -> Option<String> {
    version_line_by_priority(kind, text, true)
        .or_else(|| version_line_by_priority(kind, text, false))
}

pub(super) fn version_line_by_priority(
    kind: EncoderKind,
    text: &str,
    prefer_compiled: bool,
) -> Option<String> {
    text.lines().find_map(|line| {
        let lower = line.to_ascii_lowercase();
        let compiled_match = match kind {
            EncoderKind::J2k => false,
            EncoderKind::OpenJpeg => lower.contains("compiled against openjp2"),
            EncoderKind::Grok => lower.contains("compiled against libgrok"),
            EncoderKind::Kakadu => lower.contains("kakadu"),
        };
        let fallback_match = match kind {
            EncoderKind::J2k => false,
            EncoderKind::OpenJpeg => lower.contains("openjpeg"),
            EncoderKind::Grok => lower.contains("grok"),
            EncoderKind::Kakadu => lower.contains("kdu_compress") || lower.contains("kakadu"),
        };
        let matches_priority = if prefer_compiled {
            compiled_match
        } else {
            fallback_match
        };
        matches_priority.then(|| line.split_whitespace().collect::<Vec<_>>().join(" "))
    })
}

pub(super) fn selected_encoders_label(tools: &[EncoderTool]) -> String {
    tools
        .iter()
        .map(|tool| tool.kind.label())
        .collect::<Vec<_>>()
        .join(",")
}

#[cfg(all(test, unix))]
mod tests;
