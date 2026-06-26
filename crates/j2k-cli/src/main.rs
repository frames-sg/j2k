// SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::{self, Read};
use std::path::Path;
use std::process::ExitCode;

const INSPECT_READ_LIMIT: usize = 64 * 1024 * 1024;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let subcommand = args.next();
    match subcommand.as_deref() {
        Some("inspect") => {
            let Some(path) = args.next() else {
                eprintln!("usage: j2k inspect <file>");
                return ExitCode::from(2);
            };
            inspect(Path::new(&path))
        }
        Some("transcode") => match parse_transcode_args(args.collect()) {
            Ok(args) => transcode(&args),
            Err(message) => {
                eprintln!("{message}");
                eprintln!("{TRANSCODE_USAGE}");
                ExitCode::from(2)
            }
        },
        Some("--help" | "-h" | "help") | None => {
            eprintln!("j2k {}", env!("CARGO_PKG_VERSION"));
            eprintln!("Usage:");
            eprintln!("  j2k inspect <file>                                      Parse JPEG or JPEG 2000 headers");
            eprintln!("  j2k transcode <input.jpg> <output.j2k> --htj2k --lossless-53");
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("unknown subcommand: {other}");
            ExitCode::from(2)
        }
    }
}

const TRANSCODE_USAGE: &str = "usage: j2k transcode <input.jpg> <output.j2k> --htj2k --lossless-53";

#[derive(Debug, Clone, PartialEq, Eq)]
struct TranscodeArgs {
    input: String,
    output: String,
}

fn parse_transcode_args(raw_args: Vec<String>) -> Result<TranscodeArgs, String> {
    let mut input = None;
    let mut output = None;
    let mut htj2k = false;
    let mut lossless_53 = false;

    for arg in raw_args {
        match arg.as_str() {
            "--htj2k" => htj2k = true,
            "--lossless-53" => lossless_53 = true,
            flag if flag.starts_with('-') => {
                return Err(format!("unsupported transcode option: {flag}"));
            }
            path if input.is_none() => input = Some(path.to_string()),
            path if output.is_none() => output = Some(path.to_string()),
            extra => return Err(format!("unexpected transcode argument: {extra}")),
        }
    }

    if !htj2k {
        return Err("transcode requires --htj2k".to_string());
    }
    if !lossless_53 {
        return Err("transcode requires --lossless-53".to_string());
    }

    let input = input.ok_or_else(|| "missing input JPEG path".to_string())?;
    let output = output.ok_or_else(|| "missing output J2K path".to_string())?;
    Ok(TranscodeArgs { input, output })
}

fn transcode(args: &TranscodeArgs) -> ExitCode {
    match transcode_jpeg_to_htj2k(Path::new(&args.input), Path::new(&args.output)) {
        Ok(summary) => {
            println!("{summary}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(1)
        }
    }
}

fn transcode_jpeg_to_htj2k(input: &Path, output: &Path) -> Result<String, String> {
    let bytes =
        std::fs::read(input).map_err(|e| format!("error reading {}: {e}", input.display()))?;
    let encoded =
        transcode_jpeg_to_htj2k_bytes(&bytes).map_err(|e| format!("error transcoding: {e}"))?;
    std::fs::write(output, &encoded.codestream)
        .map_err(|e| format!("error writing {}: {e}", output.display()))?;
    Ok(format_transcode_summary(&encoded))
}

fn transcode_jpeg_to_htj2k_bytes(
    bytes: &[u8],
) -> Result<j2k_transcode::EncodedTranscode, j2k_transcode::JpegToHtj2kError> {
    j2k_transcode::jpeg_to_htj2k(bytes, &j2k_transcode::JpegToHtj2kOptions::lossless_53())
}

fn format_transcode_summary(encoded: &j2k_transcode::EncodedTranscode) -> String {
    let report = &encoded.report;
    let timings = report.timings;
    format!(
        "transcoded {}x{} comps={} bytes={} path={} transform_dispatches={} cpu_fallback_jobs={} extract_us={} transform_us={} encode_us={}",
        report.width,
        report.height,
        report.component_count,
        encoded.codestream.len(),
        report.path,
        timings.accelerator_dispatches,
        timings.cpu_fallback_jobs,
        report.extract_us,
        report.transform_us,
        report.encode_us,
    )
}

fn inspect(path: &Path) -> ExitCode {
    let input = match read_inspect_input(path) {
        Ok(input) => input,
        Err(e) => {
            eprintln!("error reading {}: {e}", path.display());
            return ExitCode::from(1);
        }
    };
    match inspect_bytes(&input.bytes) {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("{message}");
            if input.truncated {
                eprintln!(
                    "note: inspect read only the first {INSPECT_READ_LIMIT} bytes to avoid unbounded memory use"
                );
            }
            ExitCode::from(1)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct InspectInput {
    bytes: Vec<u8>,
    truncated: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InspectFormat {
    Jpeg,
    J2k,
}

fn detect_inspect_format(bytes: &[u8]) -> InspectFormat {
    if bytes.starts_with(&[0, 0, 0, 12, b'j', b'P', b' ', b' ']) || bytes.starts_with(&[0xFF, 0x4F])
    {
        InspectFormat::J2k
    } else {
        InspectFormat::Jpeg
    }
}

fn inspect_bytes(bytes: &[u8]) -> Result<String, String> {
    match detect_inspect_format(bytes) {
        InspectFormat::Jpeg => match j2k_jpeg::Decoder::inspect(bytes) {
            Ok(info) => Ok(format!(
                "{}×{} {:?} {:?} bit={} samp={:?} mcu={}x{} units={}x{} rst={:?} scans={}",
                info.dimensions.0,
                info.dimensions.1,
                info.sof_kind,
                info.color_space,
                info.bit_depth,
                info.sampling.components(),
                info.mcu_geometry.width,
                info.mcu_geometry.height,
                info.mcu_geometry.columns,
                info.mcu_geometry.rows,
                info.restart_interval,
                info.scan_count,
            )),
            Err(e) => {
                let mut message = format!("error: {e}");
                if e.is_unsupported() {
                    message.push_str(
                        "\nhint: this file is not supported by j2k; try jpeg-decoder or openjpeg",
                    );
                }
                Err(message)
            }
        },
        InspectFormat::J2k => match j2k::J2kDecoder::inspect(bytes) {
            Ok(info) => Ok(format!(
                "{}×{} {:?} bit={} comps={} levels={} tiles={:?}",
                info.dimensions.0,
                info.dimensions.1,
                info.colorspace,
                info.bit_depth,
                info.components,
                info.resolution_levels,
                info.tile_layout,
            )),
            Err(e) => Err(format!("error: {e}")),
        },
    }
}

fn read_inspect_input(path: &Path) -> io::Result<InspectInput> {
    read_inspect_input_with_limit(path, INSPECT_READ_LIMIT)
}

fn read_inspect_input_with_limit(path: &Path, limit: usize) -> io::Result<InspectInput> {
    let mut file = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    let mut limited = file.by_ref().take((limit + 1) as u64);
    limited.read_to_end(&mut buf)?;
    let truncated = buf.len() > limit;
    if truncated {
        buf.truncate(limit);
    }
    Ok(InspectInput {
        bytes: buf,
        truncated,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        detect_inspect_format, format_transcode_summary, inspect_bytes, parse_transcode_args,
        read_inspect_input_with_limit, transcode_jpeg_to_htj2k, transcode_jpeg_to_htj2k_bytes,
        InspectFormat, TranscodeArgs,
    };
    use j2k_test_support::{minimal_j2k_codestream, minimal_jp2, JPEG_GRAYSCALE_8X8};

    #[test]
    fn detects_j2k_codestream_magic() {
        assert_eq!(
            detect_inspect_format(&minimal_j2k_codestream()),
            InspectFormat::J2k
        );
    }

    #[test]
    fn detects_jp2_magic() {
        assert_eq!(detect_inspect_format(&minimal_jp2()), InspectFormat::J2k);
    }

    #[test]
    fn inspect_read_is_bounded() {
        let path =
            std::env::temp_dir().join(format!("j2k-cli-inspect-bounded-{}", std::process::id()));
        std::fs::write(&path, [0xFF; 12]).expect("write bounded-read fixture");

        let input = read_inspect_input_with_limit(&path, 8).expect("read bounded inspect input");
        let _ = std::fs::remove_file(&path);

        assert!(input.truncated);
        assert_eq!(input.bytes.len(), 8);
    }

    #[test]
    fn inspect_bytes_dispatches_to_j2k() {
        let line = inspect_bytes(&minimal_jp2()).expect("jp2 inspect");
        assert!(line.contains("128×64"));
        assert!(line.contains("levels=6"));
    }

    #[test]
    fn parses_supported_transcode_command() {
        let args = parse_transcode_args(vec![
            "input.jpg".to_string(),
            "output.j2k".to_string(),
            "--htj2k".to_string(),
            "--lossless-53".to_string(),
        ])
        .expect("parse transcode command");
        assert_eq!(
            args,
            TranscodeArgs {
                input: "input.jpg".to_string(),
                output: "output.j2k".to_string()
            }
        );
    }

    #[test]
    fn rejects_unsupported_transcode_option() {
        let err = parse_transcode_args(vec![
            "input.jpg".to_string(),
            "output.j2k".to_string(),
            "--htj2k".to_string(),
            "--lossy-97".to_string(),
        ])
        .expect_err("unsupported option should fail");
        assert!(err.contains("unsupported transcode option"));
    }

    #[test]
    fn transcodes_fixture_to_nonempty_htj2k_codestream() {
        let encoded =
            transcode_jpeg_to_htj2k_bytes(JPEG_GRAYSCALE_8X8).expect("transcode JPEG fixture");
        assert!(!encoded.codestream.is_empty());
        assert!(encoded.codestream.starts_with(&[0xff, 0x4f]));
        let summary = format_transcode_summary(&encoded);
        assert!(summary.contains("transcoded 8x8"));
        assert!(summary.contains("bytes="));
    }

    #[test]
    fn transcode_file_writes_output_codestream() {
        let base = std::env::temp_dir().join(format!("j2k-cli-transcode-{}", std::process::id()));
        std::fs::create_dir_all(&base).expect("create temp dir");
        let input = base.join("input.jpg");
        let output = base.join("output.j2k");
        std::fs::write(&input, JPEG_GRAYSCALE_8X8).expect("write JPEG fixture");

        let summary = transcode_jpeg_to_htj2k(&input, &output).expect("transcode file");
        let written = std::fs::read(&output).expect("read output J2K");

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&output);
        let _ = std::fs::remove_dir(&base);

        assert!(summary.contains("transcoded 8x8"));
        assert!(written.starts_with(&[0xff, 0x4f]));
    }
}
