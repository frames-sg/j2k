// SPDX-License-Identifier: MIT OR Apache-2.0

//! An `opj_decompress`-compatible decoder: `j2k decode <options>`, or the
//! binary invoked as `opj_decompress` (for example through a symlink), so
//! image servers configured to run `opj_decompress` run j2k unchanged.
//!
//! The output image is the only thing written to the `-o` path; notes,
//! warnings and errors go to stderr, never stdout, so an output path linked
//! to `/dev/stdout` carries only the image.

mod args;
mod decode;
mod write;

use std::{io, path::Path, process::ExitCode};

use args::{Command, DecodeArgs};

/// Which command contract an invocation follows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Flavor {
    /// Invoked as `opj_decompress`: that tool's exit status (1 for every
    /// failure, including bad arguments) and help on stdout.
    OpjDecompress,
    /// `j2k decode`: j2k's exit status (2 for bad arguments, 1 for runtime
    /// failures) and help on stderr.
    J2kDecode,
}

/// The program name that selects the `opj_decompress` interface.
pub(crate) const OPJ_DECOMPRESS: &str = "opj_decompress";

/// Run one `opj_decompress` command line, without the program name.
pub(crate) fn run(flavor: Flavor, arguments: impl IntoIterator<Item = String>) -> ExitCode {
    let command = match args::parse(arguments) {
        Ok(command) => command,
        Err(message) => {
            eprintln!("{message}");
            eprintln!("Help: {} -h", flavor.program());
            return flavor.argument_error();
        }
    };
    let args = match command {
        Command::Help => return help(flavor),
        Command::Decode(args) => args,
    };
    let mut notes = Vec::new();
    let result = decode_to_file(&args, &mut notes);
    if !args.quiet {
        for note in &notes {
            eprintln!("{note}");
        }
    }
    match result {
        Ok(()) => {
            if !args.quiet {
                eprintln!("[INFO] Generated Outfile {}", args.output.display());
            }
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("ERROR -> j2k: {message}");
            ExitCode::FAILURE
        }
    }
}

impl Flavor {
    const fn program(self) -> &'static str {
        match self {
            Self::OpjDecompress => OPJ_DECOMPRESS,
            Self::J2kDecode => "j2k decode",
        }
    }

    fn argument_error(self) -> ExitCode {
        match self {
            Self::OpjDecompress => ExitCode::FAILURE,
            Self::J2kDecode => ExitCode::from(2),
        }
    }
}

fn help(flavor: Flavor) -> ExitCode {
    let text = help_text(flavor);
    match flavor {
        // opj_decompress prints its help to stdout and exits with 1.
        Flavor::OpjDecompress => {
            print!("{text}");
            ExitCode::FAILURE
        }
        Flavor::J2kDecode => {
            eprint!("{text}");
            ExitCode::SUCCESS
        }
    }
}

/// The option summary. It names no "vX.Y.Z" version: Cantaloupe reads
/// `opj_decompress -h` for one and stops passing `-quiet` below 2.2.
fn help_text(flavor: Flavor) -> String {
    format!(
        "This is the j2k {version} decoder with the opj_decompress interface.\n\
         It decodes JPEG 2000 and HTJ2K codestreams (JP2, JPH, J2K, J2C) to image files.\n\
         \n\
         Usage: {program} -i <input> -o <output> [options]\n\
         \n\
         \x20 -i <file>         JP2/JPH file or raw codestream, recognised by its signature\n\
         \x20 -o <file>         output: .bmp, .png, .tif/.tiff, .pgm, .ppm or .pnm\n\
         \x20 -r <n>            discard the n highest resolution levels\n\
         \x20 -d <x0,y0,x1,y1>  decode area on the full-resolution reference grid\n\
         \x20 -threads <n|ALL_CPUS>  decode threads (default: OPJ_NUM_THREADS, else 1)\n\
         \x20 -force-rgb        write gray images as RGB\n\
         \x20 -l 0              decode all quality layers (the only -l value accepted)\n\
         \x20 -quiet            no notes or warnings on stderr\n\
         \n\
         Not supported, and rejected: -t, -c, -p, -x, -l above 0, -upsample, -split-pnm,\n\
         -allow-partial, -ImgDir, -OutFor and .pgx/.raw/.yuv/.rawl/.tga output.\n",
        version = env!("CARGO_PKG_VERSION"),
        program = flavor.program(),
    )
}

fn decode_to_file(args: &DecodeArgs, notes: &mut Vec<String>) -> Result<(), String> {
    let bytes = read_input(&args.input)?;
    let threads = args::thread_count(
        args.threads,
        std::env::var("OPJ_NUM_THREADS").ok().as_deref(),
        std::thread::available_parallelism().map_or(1, usize::from),
    );
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map_err(|error| format!("could not start {threads} decode threads: {error}"))?;
    let image = pool.install(|| decode::decode(&bytes, args.area, args.reduce, notes))?;
    drop(bytes);
    let image = if args.force_rgb {
        write::force_rgb(image)
    } else {
        image
    };
    if matches!(args.format, args::OutputFormat::Pnm { gray: true }) && image.channels > 1 {
        notes.push("warning: only the first component is written to a PGM file".into());
    }
    let encoded = write::encode(&image, args.format)?;
    write_output(&args.output, &encoded)
}

fn read_input(path: &Path) -> Result<Vec<u8>, String> {
    if path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("jpt"))
    {
        return Err(format!(
            "{}: JPT (JPIP) streams are not supported",
            path.display()
        ));
    }
    let bytes = std::fs::read(path)
        .map_err(|error| format!("infile cannot be read: {}: {error}", path.display()))?;
    // Cantaloupe reports a source-format error when stderr says this.
    j2k::extract_j2k_codestream_payload(&bytes)
        .map_err(|_| format!("Unknown input file format: {}", path.display()))?;
    Ok(bytes)
}

/// Write the finished image in one pass, so a path linked to `/dev/stdout`
/// or a pipe works; a regular file left partly written is removed, as
/// `opj_decompress` removes a failed output.
fn write_output(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let result = std::fs::write(path, bytes);
    if let Err(error) = result {
        if std::fs::symlink_metadata(path).is_ok_and(|metadata| metadata.is_file()) {
            remove_partial(path);
        }
        return Err(format!("Outfile {} not generated: {error}", path.display()));
    }
    Ok(())
}

fn remove_partial(path: &Path) {
    if let Err(error) = std::fs::remove_file(path) {
        if error.kind() != io::ErrorKind::NotFound {
            eprintln!(
                "warning: could not remove the partial output {}: {error}",
                path.display()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{help_text, Flavor};

    #[test]
    fn help_names_no_openjpeg_style_version() {
        // Cantaloupe matches `[ ]v\d+.\d+.\d+` in `opj_decompress -h` output.
        for flavor in [Flavor::OpjDecompress, Flavor::J2kDecode] {
            let text = help_text(flavor);
            let bytes = text.as_bytes();
            let versioned = bytes
                .windows(3)
                .any(|window| window[0] == b' ' && window[1] == b'v' && window[2].is_ascii_digit());
            assert!(!versioned, "{text}");
        }
    }
}
