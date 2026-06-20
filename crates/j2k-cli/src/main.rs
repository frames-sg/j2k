// SPDX-License-Identifier: Apache-2.0

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
        Some("--help" | "-h" | "help") | None => {
            eprintln!("j2k {}", env!("CARGO_PKG_VERSION"));
            eprintln!("Usage:");
            eprintln!("  j2k inspect <file>    Parse JPEG or JPEG 2000 headers and print Info");
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("unknown subcommand: {other}");
            ExitCode::from(2)
        }
    }
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
        detect_inspect_format, inspect_bytes, read_inspect_input_with_limit, InspectFormat,
    };
    use j2k_test_support::{minimal_j2k_codestream, minimal_jp2};

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
}
