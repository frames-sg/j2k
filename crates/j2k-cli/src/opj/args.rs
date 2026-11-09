// SPDX-License-Identifier: MIT OR Apache-2.0

//! `opj_decompress` command-line parsing.
//!
//! Options keep `opj_decompress` 2.5's spellings and meanings. Options j2k cannot
//! honour exactly are rejected with the reason instead of being ignored, and
//! malformed values are errors where `opj_decompress` would fall back to a
//! default.

use std::path::{Path, PathBuf};

/// What one `opj_decompress` command line asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Command {
    /// Print the option summary.
    Help,
    /// Decode one input file to one output file.
    Decode(DecodeArgs),
}

/// A single-image decode request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecodeArgs {
    pub(crate) input: PathBuf,
    pub(crate) output: PathBuf,
    pub(crate) format: OutputFormat,
    /// Highest resolution levels to discard (`-r`).
    pub(crate) reduce: u8,
    /// Decode area on the reference grid (`-d`); `None` decodes the image.
    pub(crate) area: Option<DecodeArea>,
    /// Decode threads (`-threads`); `None` follows `OPJ_NUM_THREADS`.
    pub(crate) threads: Option<Threads>,
    pub(crate) quiet: bool,
    pub(crate) force_rgb: bool,
}

/// Output file format, chosen by the `-o` extension as `opj_decompress` does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OutputFormat {
    Bmp,
    Png,
    Tiff,
    /// PNM; `gray` for a `.pgm` name, which keeps only the first component.
    Pnm {
        gray: bool,
    },
}

/// `-d x0,y0,x1,y1`: a rectangle on the full-resolution reference grid, so
/// the image's own top-left sample is at its origin (`XOsiz`, `YOsiz`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct DecodeArea {
    pub(crate) x0: u32,
    pub(crate) y0: u32,
    pub(crate) x1: u32,
    pub(crate) y1: u32,
}

/// `-threads N|ALL_CPUS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Threads {
    Count(u32),
    AllCpus,
}

/// Parse `opj_decompress` options (without the program name).
pub(crate) fn parse(args: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let mut parsed = Parsed::default();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-quiet" => parsed.quiet = true,
            "-force-rgb" => parsed.force_rgb = true,
            "-threads" => parsed.threads = Some(parse_threads(&value(&mut args, &arg)?)?),
            "-upsample" => return Err(unsupported(&arg, UPSAMPLE)),
            "-split-pnm" => {
                return Err(unsupported(&arg, "per-component PNM files are not written"))
            }
            "-allow-partial" => return Err(unsupported(&arg, PARTIAL)),
            "-ImgDir" | "-OutFor" => return Err(unsupported(&arg, DIRECTORY)),
            "-h" => return Ok(Command::Help),
            _ => parse_short(&arg, &mut args, &mut parsed)?,
        }
    }
    parsed.finish().map(Command::Decode)
}

const UPSAMPLE: &str = "components with different sampling are always replicated onto the \
                        image grid, and uniformly subsampled images are not decoded";
const PARTIAL: &str = "j2k does not decode truncated codestreams";
const DIRECTORY: &str = "directory batches are not decoded; pass one -i and one -o";

#[derive(Default)]
struct Parsed {
    input: Option<PathBuf>,
    output: Option<(PathBuf, OutputFormat)>,
    reduce: u8,
    area: Option<DecodeArea>,
    threads: Option<Threads>,
    quiet: bool,
    force_rgb: bool,
}

impl Parsed {
    fn finish(self) -> Result<DecodeArgs, String> {
        let (Some(input), Some((output, format))) = (self.input, self.output) else {
            return Err(
                "[ERROR] Required parameters are missing: -i <input> and -o <output>".into(),
            );
        };
        Ok(DecodeArgs {
            input,
            output,
            format,
            reduce: self.reduce,
            area: self.area.filter(|area| *area != DecodeArea::default()),
            threads: self.threads,
            quiet: self.quiet,
            force_rgb: self.force_rgb,
        })
    }
}

/// One-letter options, with the value attached (`-r2`) or following (`-r 2`).
fn parse_short(
    arg: &str,
    args: &mut impl Iterator<Item = String>,
    parsed: &mut Parsed,
) -> Result<(), String> {
    let mut chars = arg.chars();
    let (Some('-'), Some(letter)) = (chars.next(), chars.next()) else {
        return Err(format!("unexpected argument {arg:?}; options start with -"));
    };
    if !"iorlxdtpc".contains(letter) {
        return Err(format!("unknown option {arg}"));
    }
    let attached = chars.as_str();
    let option = format!("-{letter}");
    let value = if attached.is_empty() {
        value(args, &option)?
    } else {
        attached.to_owned()
    };
    match letter {
        'i' => parsed.input = Some(PathBuf::from(value)),
        'o' => {
            let path = PathBuf::from(value);
            let format = output_format(&path)?;
            parsed.output = Some((path, format));
        }
        'r' => {
            parsed.reduce = parse_number(&option, &value)?
                .try_into()
                .map_err(|_| format!("-r {value}: reduce factor is out of range"))?;
        }
        'l' => {
            if parse_number(&option, &value)? != 0 {
                return Err(unsupported(
                    &format!("-l {value}"),
                    "every quality layer is decoded; only -l 0 (all layers) is accepted",
                ));
            }
        }
        'd' => parsed.area = Some(parse_area(&value)?),
        't' => {
            return Err(unsupported(
                "-t",
                "codestream tiles are not decoded by index; pass the tile's area to -d",
            ))
        }
        'c' => {
            return Err(unsupported(
                "-c",
                "components are decoded together with their colour transforms",
            ))
        }
        'p' => return Err(unsupported("-p", "output precision is not forced")),
        _ => return Err(unsupported("-x", "index files are not written")),
    }
    Ok(())
}

fn value(args: &mut impl Iterator<Item = String>, option: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("option {option} requires a value"))
}

fn unsupported(option: &str, reason: &str) -> String {
    format!("[ERROR] option {option} is not supported by j2k: {reason}")
}

fn parse_number(option: &str, value: &str) -> Result<u32, String> {
    value
        .parse()
        .map_err(|_| format!("{option} {value}: expected a non-negative integer"))
}

fn parse_threads(value: &str) -> Result<Threads, String> {
    if value == "ALL_CPUS" {
        return Ok(Threads::AllCpus);
    }
    value
        .parse()
        .map(Threads::Count)
        .map_err(|_| format!("-threads {value}: expected a thread count or ALL_CPUS"))
}

fn parse_area(value: &str) -> Result<DecodeArea, String> {
    let fields = value
        .split(',')
        .map(|field| field.trim().parse::<u32>())
        .collect::<Result<Vec<_>, _>>();
    match fields.as_deref() {
        Ok(&[x0, y0, x1, y1]) => Ok(DecodeArea { x0, y0, x1, y1 }),
        _ => Err(format!(
            "-d {value}: expected four non-negative integers x0,y0,x1,y1"
        )),
    }
}

/// The output format `opj_decompress` picks from the name's extension.
fn output_format(path: &Path) -> Result<OutputFormat, String> {
    let extension = path
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    match extension.as_str() {
        "bmp" => Ok(OutputFormat::Bmp),
        "png" => Ok(OutputFormat::Png),
        "tif" | "tiff" => Ok(OutputFormat::Tiff),
        "pgm" => Ok(OutputFormat::Pnm { gray: true }),
        "pnm" | "ppm" => Ok(OutputFormat::Pnm { gray: false }),
        "pgx" | "raw" | "yuv" | "rawl" | "tga" => Err(unsupported(
            &format!("-o {}", path.display()),
            "write .bmp, .png, .tif/.tiff, .pgm, .ppm or .pnm",
        )),
        _ => Err(format!(
            "[ERROR] Unknown output format image {} [j2k writes *.bmp, *.png, *.tif(f), *.pgm, *.ppm or *.pnm]",
            path.display()
        )),
    }
}

/// Decode threads: `-threads`, else `OPJ_NUM_THREADS` (`ALL_CPUS` or a
/// count of at most twice the processors), else one, as `opj_decompress`
/// counts them.
pub(crate) fn thread_count(threads: Option<Threads>, env: Option<&str>, cpus: usize) -> usize {
    let requested = match threads {
        Some(Threads::Count(count)) if count > 0 => usize::try_from(count).unwrap_or(usize::MAX),
        Some(Threads::AllCpus) => cpus,
        _ => match env {
            Some("ALL_CPUS") => cpus,
            Some(count) => count
                .trim()
                .parse::<usize>()
                .unwrap_or(0)
                .min(cpus.saturating_mul(2)),
            None => 1,
        },
    };
    requested.max(1)
}

#[cfg(test)]
mod tests {
    use super::{parse, thread_count, Command, DecodeArea, DecodeArgs, OutputFormat, Threads};
    use std::path::PathBuf;

    fn args(line: &str) -> Vec<String> {
        line.split_whitespace().map(String::from).collect()
    }

    fn decode(line: &str) -> DecodeArgs {
        match parse(args(line)).expect("parse") {
            Command::Decode(args) => args,
            Command::Help => panic!("unexpected help"),
        }
    }

    #[test]
    fn parses_the_cantaloupe_command_line() {
        let parsed = decode("-quiet -i /src/image -d 512,256,1536,768 -r 2 -o /tmp/out.bmp");
        assert_eq!(
            parsed,
            DecodeArgs {
                input: PathBuf::from("/src/image"),
                output: PathBuf::from("/tmp/out.bmp"),
                format: OutputFormat::Bmp,
                reduce: 2,
                area: Some(DecodeArea {
                    x0: 512,
                    y0: 256,
                    x1: 1536,
                    y1: 768
                }),
                threads: None,
                quiet: true,
                force_rgb: false,
            }
        );
    }

    #[test]
    fn accepts_attached_values_and_keeps_the_last_repeat() {
        let parsed = decode("-iimage.jp2 -r1 -r3 -o out.PGM -threads ALL_CPUS -l 0");
        assert_eq!(parsed.input, PathBuf::from("image.jp2"));
        assert_eq!(parsed.reduce, 3);
        assert_eq!(parsed.format, OutputFormat::Pnm { gray: true });
        assert_eq!(parsed.threads, Some(Threads::AllCpus));
    }

    #[test]
    fn output_formats_follow_the_extension() {
        for (name, format) in [
            ("a.bmp", OutputFormat::Bmp),
            ("a.png", OutputFormat::Png),
            ("a.tif", OutputFormat::Tiff),
            ("a.TIFF", OutputFormat::Tiff),
            ("a.ppm", OutputFormat::Pnm { gray: false }),
            ("a.pnm", OutputFormat::Pnm { gray: false }),
            ("a.pgm", OutputFormat::Pnm { gray: true }),
        ] {
            assert_eq!(
                decode(&format!("-i a.jp2 -o {name}")).format,
                format,
                "{name}"
            );
        }
    }

    #[test]
    fn an_all_zero_area_is_the_whole_image() {
        assert_eq!(decode("-i a.jp2 -o a.png -d 0,0,0,0").area, None);
    }

    #[test]
    fn rejects_what_j2k_cannot_honour() {
        for (line, needle) in [
            ("-i a.jp2 -o a.png -l 2", "-l 2"),
            ("-i a.jp2 -o a.png -t 0", "-t"),
            ("-i a.jp2 -o a.png -c 0,1", "-c"),
            ("-i a.jp2 -o a.png -p 8", "-p"),
            ("-i a.jp2 -o a.png -x a.idx", "-x"),
            ("-i a.jp2 -o a.png -upsample", "-upsample"),
            ("-i a.jp2 -o a.png -split-pnm", "-split-pnm"),
            ("-i a.jp2 -o a.png -allow-partial", "-allow-partial"),
            ("-ImgDir dir -OutFor PNG", "-ImgDir"),
            ("-i a.jp2 -o a.raw", ".raw"),
            ("-i a.jp2 -o a.tga", ".tga"),
        ] {
            let error = parse(args(line)).expect_err(line);
            assert!(error.contains("not supported by j2k"), "{line}: {error}");
            assert!(error.contains(needle), "{line}: {error}");
        }
    }

    #[test]
    fn rejects_malformed_and_missing_arguments() {
        for (line, needle) in [
            ("-i a.jp2", "Required parameters are missing"),
            ("-o a.png", "Required parameters are missing"),
            ("-i a.jp2 -o a.png -r", "requires a value"),
            ("-i a.jp2 -o a.png -r two", "non-negative integer"),
            ("-i a.jp2 -o a.png -r 256", "out of range"),
            ("-i a.jp2 -o a.png -d 1,2,3", "four non-negative integers"),
            (
                "-i a.jp2 -o a.png -d 1,2,3,-4",
                "four non-negative integers",
            ),
            ("-i a.jp2 -o a.png -threads many", "thread count"),
            ("-i a.jp2 -o a.jpg", "Unknown output format"),
            ("-i a.jp2 -o a.png -z", "unknown option -z"),
            ("-i a.jp2 -o a.png --quiet", "unknown option --quiet"),
            ("-i a.jp2 -o a.png stray", "unexpected argument"),
        ] {
            let error = parse(args(line)).expect_err(line);
            assert!(error.contains(needle), "{line}: {error}");
        }
    }

    #[test]
    fn help_wins() {
        assert_eq!(parse(args("-i a.jp2 -h")).expect("help"), Command::Help);
    }

    #[test]
    fn thread_counts_follow_opj_decompress() {
        assert_eq!(thread_count(None, None, 8), 1);
        assert_eq!(
            thread_count(Some(Threads::Count(3)), Some("ALL_CPUS"), 8),
            3
        );
        assert_eq!(thread_count(Some(Threads::AllCpus), None, 8), 8);
        assert_eq!(thread_count(Some(Threads::Count(0)), Some("4"), 8), 4);
        assert_eq!(thread_count(None, Some("ALL_CPUS"), 8), 8);
        assert_eq!(thread_count(None, Some("100"), 8), 16);
        assert_eq!(thread_count(None, Some("junk"), 8), 1);
    }
}
