// SPDX-License-Identifier: Apache-2.0

use std::io::{self, Read};
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let subcommand = args.next();
    match subcommand.as_deref() {
        Some("inspect") => {
            let path = match args.next() {
                Some(p) => p,
                None => {
                    eprintln!("usage: signinum inspect <file>");
                    return ExitCode::from(2);
                }
            };
            inspect(Path::new(&path))
        }
        Some("--help") | Some("-h") | Some("help") | None => {
            eprintln!("signinum {}", env!("CARGO_PKG_VERSION"));
            eprintln!("Usage:");
            eprintln!(
                "  signinum inspect <file>    Parse JPEG or JPEG 2000 headers and print Info"
            );
            ExitCode::SUCCESS
        }
        Some(other) => {
            eprintln!("unknown subcommand: {other}");
            ExitCode::from(2)
        }
    }
}

fn inspect(path: &Path) -> ExitCode {
    let bytes = match read_file(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error reading {}: {e}", path.display());
            return ExitCode::from(1);
        }
    };
    match inspect_bytes(&bytes) {
        Ok(line) => {
            println!("{line}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(1)
        }
    }
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
        InspectFormat::Jpeg => match signinum_jpeg::Decoder::inspect(bytes) {
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
                        "\nhint: this file is not supported by signinum; try jpeg-decoder or openjpeg",
                    );
                }
                Err(message)
            }
        },
        InspectFormat::J2k => match signinum_j2k::J2kDecoder::inspect(bytes) {
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

fn read_file(path: &Path) -> io::Result<Vec<u8>> {
    let mut file = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::{detect_inspect_format, inspect_bytes, InspectFormat};
    use signinum_test_support::{minimal_j2k_codestream, minimal_jp2};

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
    fn inspect_bytes_dispatches_to_j2k() {
        let line = inspect_bytes(&minimal_jp2()).expect("jp2 inspect");
        assert!(line.contains("128×64"));
        assert!(line.contains("levels=6"));
    }
}
