use std::{
    any::Any,
    fs,
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
};

use j2k::{J2kDecoder, PixelFormat};
use j2k_compare::{grok, openjpeg};
use j2k_test_support::fnv1a64_hex;

use crate::adoption_corpus::{
    canonical_label, codec_from_bytes, collect_decode_fixture_paths, container_from_path_and_bytes,
    corpus_category, corpus_name, manifest_row, validate_tsv_field,
};

#[derive(Debug)]
struct AdoptionCurateOptions {
    out_dir: PathBuf,
    fixture_dirs: String,
    corpus_name: Option<String>,
    corpus_category: Option<String>,
    license_status: String,
    encode_command: String,
    max_files: Option<usize>,
}

pub(crate) fn adoption_curate(args: impl Iterator<Item = String>) -> Result<(), String> {
    let args = args.collect::<Vec<_>>();
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        println!("{}", help_text());
        return Ok(());
    }
    let options = AdoptionCurateOptions::parse(args.into_iter())?;
    let paths = collect_decode_fixture_paths(&options.fixture_dirs)?;
    if paths.is_empty() {
        return Err("--fixtures did not contain supported J2K/JP2/JPH files".to_string());
    }
    let fixture_out_dir = options.out_dir.join("fixtures");
    fs::create_dir_all(&fixture_out_dir)
        .map_err(|err| format!("create {}: {err}", fixture_out_dir.display()))?;

    let mut manifest =
        "path\tcorpus_category\tcorpus_name\tlicense_status\tencode_command\tinput_fnv1a64\tsource_fnv1a64\tcodec\tcontainer\n".to_string();
    let mut skipped = "path\treason\n".to_string();
    let mut accepted = 0_usize;
    let mut skipped_count = 0_usize;

    for (index, (root, path)) in paths.iter().enumerate() {
        if options
            .max_files
            .is_some_and(|max_files| accepted >= max_files)
        {
            skipped.push_str(&manifest_row(&[
                path.display().to_string(),
                "max-files-limit".to_string(),
            ])?);
            skipped_count += 1;
            continue;
        }
        match curate_one(&options, &fixture_out_dir, index, root, path) {
            Ok(row) => {
                manifest.push_str(&row);
                accepted += 1;
            }
            Err(reason) => {
                skipped.push_str(&manifest_row(&[
                    path.display().to_string(),
                    sanitize_skip_reason(&reason),
                ])?);
                skipped_count += 1;
            }
        }
    }

    if accepted == 0 {
        return Err("no supported fixtures were accepted".to_string());
    }

    let manifest_path = options.out_dir.join("fixtures.tsv");
    fs::write(&manifest_path, manifest)
        .map_err(|err| format!("write {}: {err}", manifest_path.display()))?;
    let skipped_path = options.out_dir.join("skipped.tsv");
    fs::write(&skipped_path, skipped)
        .map_err(|err| format!("write {}: {err}", skipped_path.display()))?;
    write_readme(&options, accepted, skipped_count)?;
    eprintln!(
        "curated {accepted} fixture files into {}; skipped {skipped_count}",
        options.out_dir.display()
    );
    Ok(())
}

fn curate_one(
    options: &AdoptionCurateOptions,
    fixture_out_dir: &Path,
    index: usize,
    root: &Path,
    path: &Path,
) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|err| format!("read: {err}"))?;
    let info = catch_curate_unwind(|| J2kDecoder::inspect(&bytes))
        .map_err(|payload| format!("inspect-panic: {payload}"))?
        .map_err(|err| format!("inspect: {err}"))?;
    if !matches!(info.components, 1 | 3) {
        return Err(format!("unsupported-components-{}", info.components));
    }
    if info.bit_depth != 8 {
        return Err(format!("unsupported-bit-depth-{}", info.bit_depth));
    }
    let codec = codec_from_bytes(&bytes);
    if codec == "unknown" {
        return Err("unknown-codec".to_string());
    }
    let baseline = validate_full_decode(&bytes, info.dimensions, info.components)?;
    validate_external_comparators(&bytes, info.components, &baseline)?;
    let extension = path
        .extension()
        .and_then(|value| value.to_str())
        .unwrap_or("j2k");
    let staged = fixture_out_dir.join(format!(
        "{index:06}_{}.{}",
        sanitize_id(
            path.file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("fixture")
        ),
        sanitize_id(extension)
    ));
    fs::copy(path, &staged).map_err(|err| format!("copy to {}: {err}", staged.display()))?;
    let input_hash = fnv1a64_hex(&bytes);
    let fields = [
        canonical_label(&staged)?,
        corpus_category(path, options.corpus_category.as_deref()),
        corpus_name(root, options.corpus_name.as_deref()),
        options.license_status.clone(),
        options.encode_command.clone(),
        input_hash.clone(),
        input_hash,
        codec.to_string(),
        container_from_path_and_bytes(path, &bytes).to_string(),
    ];
    manifest_row(&fields)
}

fn validate_full_decode(
    bytes: &[u8],
    dimensions: (u32, u32),
    components: u8,
) -> Result<Vec<u8>, String> {
    let format = match components {
        1 => PixelFormat::Gray8,
        3 => PixelFormat::Rgb8,
        _ => return Err(format!("unsupported-components-{components}")),
    };
    let width = usize::try_from(dimensions.0).map_err(|_| "width-overflows-usize".to_string())?;
    let height = usize::try_from(dimensions.1).map_err(|_| "height-overflows-usize".to_string())?;
    let stride = width
        .checked_mul(format.bytes_per_pixel())
        .ok_or_else(|| "decode-stride-overflow".to_string())?;
    let len = stride
        .checked_mul(height)
        .ok_or_else(|| "decode-buffer-overflow".to_string())?;
    let baseline = catch_curate_unwind(|| {
        let mut decoder = J2kDecoder::new(bytes).map_err(|err| format!("decode-open: {err}"))?;
        let mut out = vec![0_u8; len];
        decoder
            .decode_into(&mut out, stride, format)
            .map_err(|err| format!("decode-full: {err}"))?;
        Ok::<Vec<u8>, String>(out)
    })
    .map_err(|payload| format!("decode-panic: {payload}"))??;
    Ok(baseline)
}

fn validate_external_comparators(
    bytes: &[u8],
    components: u8,
    baseline: &[u8],
) -> Result<(), String> {
    let openjpeg_output = match components {
        1 => openjpeg::decode_gray(bytes),
        3 => openjpeg::decode_rgb(bytes),
        _ => return Err(format!("unsupported-components-{components}")),
    }
    .map_err(|err| format!("openjpeg-full: {err}"))?;
    if openjpeg_output != baseline {
        return Err(format!(
            "openjpeg-full-mismatch:{}-vs-{}",
            openjpeg_output.len(),
            baseline.len()
        ));
    }

    if grok::is_available() {
        let grok_output = match components {
            1 => grok::decode_gray(bytes),
            3 => grok::decode_rgb(bytes),
            _ => return Err(format!("unsupported-components-{components}")),
        }
        .map_err(|err| format!("grok-full: {err}"))?;
        if grok_output != baseline {
            return Err(format!(
                "grok-full-mismatch:{}-vs-{}",
                grok_output.len(),
                baseline.len()
            ));
        }
    }
    Ok(())
}

fn catch_curate_unwind<T>(operation: impl FnOnce() -> T) -> Result<T, String> {
    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(AssertUnwindSafe(operation));
    panic::set_hook(hook);
    result.map_err(|payload| panic_message(payload.as_ref()))
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

fn sanitize_skip_reason(reason: &str) -> String {
    let out = reason
        .chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .collect::<String>();
    let trimmed = out.trim();
    if trimmed.is_empty() {
        "unspecified-skip-reason".to_string()
    } else {
        trimmed.to_string()
    }
}

fn write_readme(
    options: &AdoptionCurateOptions,
    accepted: usize,
    skipped: usize,
) -> Result<(), String> {
    let text = format!(
        "# Curated J2K Adoption Fixtures\n\n\
Generated by `cargo xtask adoption-curate`.\n\n\
- accepted fixtures: {accepted}\n\
- skipped fixtures: {skipped}\n\
- source path-list: `{}`\n\
- manifest: `{}`\n\
- skipped report: `{}`\n\n\
`fixtures.tsv` records `source_fnv1a64`; for curated native compressed files this \
equals `input_fnv1a64` because the compressed file is the source fixture artifact.\n\n\
Use the `fixtures/` directory and `fixtures.tsv` with `cargo xtask adoption-benchmark`.\n",
        options.fixture_dirs,
        options.out_dir.join("fixtures.tsv").display(),
        options.out_dir.join("skipped.tsv").display()
    );
    let path = options.out_dir.join("README.md");
    fs::write(&path, text).map_err(|err| format!("write {}: {err}", path.display()))
}

impl AdoptionCurateOptions {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut options = Self {
            out_dir: PathBuf::from("target/j2k-adoption-curated"),
            fixture_dirs: String::new(),
            corpus_name: None,
            corpus_category: None,
            license_status: String::new(),
            encode_command: String::new(),
            max_files: None,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out-dir" => {
                    options.out_dir = PathBuf::from(
                        args.next()
                            .ok_or_else(|| "--out-dir requires a value".to_string())?,
                    );
                }
                "--fixtures" | "--decode-fixtures" => {
                    options.fixture_dirs = args
                        .next()
                        .ok_or_else(|| format!("{arg} requires a platform path-list value"))?;
                }
                "--corpus-name" => {
                    options.corpus_name = Some(
                        args.next()
                            .ok_or_else(|| "--corpus-name requires a value".to_string())?,
                    );
                }
                "--corpus-category" => {
                    options.corpus_category = Some(
                        args.next()
                            .ok_or_else(|| "--corpus-category requires a value".to_string())?,
                    );
                }
                "--license-status" => {
                    options.license_status = args
                        .next()
                        .ok_or_else(|| "--license-status requires a value".to_string())?;
                }
                "--encode-command" | "--decode-encode-command" => {
                    options.encode_command = args
                        .next()
                        .ok_or_else(|| format!("{arg} requires a value"))?;
                }
                "--max-files" => {
                    options.max_files = Some(parse_positive_usize(
                        &args
                            .next()
                            .ok_or_else(|| "--max-files requires a value".to_string())?,
                        "--max-files",
                    )?);
                }
                "--help" | "-h" => unreachable!("help handled before option parsing"),
                other => {
                    return Err(format!(
                        "unknown adoption-curate argument `{other}`\n{}",
                        help_text()
                    ));
                }
            }
        }
        if options.fixture_dirs.is_empty() {
            return Err("--fixtures is required".to_string());
        }
        if options.license_status.is_empty() {
            return Err("--license-status is required".to_string());
        }
        if options.encode_command.is_empty() {
            return Err("--encode-command is required".to_string());
        }
        validate_tsv_field(&options.license_status)?;
        validate_tsv_field(&options.encode_command)?;
        if let Some(value) = &options.corpus_name {
            validate_tsv_field(value)?;
        }
        if let Some(value) = &options.corpus_category {
            validate_tsv_field(value)?;
        }
        Ok(options)
    }
}

fn parse_positive_usize(value: &str, label: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|err| format!("{label} must be a positive integer: {err}"))?;
    if parsed == 0 {
        return Err(format!("{label} must be greater than zero"));
    }
    Ok(parsed)
}

fn sanitize_id(value: &str) -> String {
    let mut out = String::new();
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    let trimmed = out.trim_matches('_');
    if trimmed.is_empty() {
        "fixture".to_string()
    } else {
        trimmed.to_string()
    }
}

fn help_text() -> String {
    "usage: cargo xtask adoption-curate --fixtures PATHS --license-status STATUS --encode-command CMD [--corpus-name NAME] [--corpus-category CATEGORY] [--max-files N] [--out-dir DIR]".to_string()
}

#[cfg(test)]
mod tests {
    use super::adoption_curate;
    use j2k::{
        encode_j2k_lossless, EncodeBackendPreference, J2kBlockCodingMode, J2kEncodeValidation,
        J2kLosslessEncodeOptions, J2kLosslessSamples, J2kProgressionOrder, ReversibleTransform,
    };
    use j2k_test_support::patterned_gray8;

    #[test]
    fn curates_supported_decode_fixtures_and_records_skips() {
        let root = std::env::current_dir()
            .expect("current dir")
            .join("target")
            .join("j2k-adoption-curate-test")
            .join(std::process::id().to_string());
        let input = root.join("input");
        let out = root.join("out");
        std::fs::create_dir_all(&input).expect("create input");
        std::fs::write(input.join("valid.j2k"), classic_gray_fixture()).expect("write valid");
        std::fs::write(input.join("invalid.jp2"), b"not a jpeg 2000 file").expect("write invalid");

        adoption_curate(
            [
                "--fixtures",
                input.to_str().expect("utf8 input"),
                "--license-status",
                "permissive-test-fixture",
                "--encode-command",
                "test-source",
                "--corpus-name",
                "curate-test",
                "--corpus-category",
                "interop",
                "--out-dir",
                out.to_str().expect("utf8 out"),
            ]
            .map(str::to_string)
            .into_iter(),
        )
        .expect("curate");

        let manifest = std::fs::read_to_string(out.join("fixtures.tsv")).expect("manifest");
        let skipped = std::fs::read_to_string(out.join("skipped.tsv")).expect("skipped");
        assert!(manifest.contains("\tinterop\tcurate-test\tpermissive-test-fixture\ttest-source\t"));
        assert!(manifest.contains("\tj2k\traw-codestream\n"));
        assert!(manifest.starts_with(
            "path\tcorpus_category\tcorpus_name\tlicense_status\tencode_command\tinput_fnv1a64\tsource_fnv1a64\tcodec\tcontainer\n"
        ));
        assert!(skipped.contains("invalid.jp2"));
        assert!(skipped.contains("inspect:"));
    }

    fn classic_gray_fixture() -> Vec<u8> {
        let width = 128;
        let height = 128;
        let pixels = patterned_gray8(width, height);
        let samples =
            J2kLosslessSamples::new(&pixels, width, height, 1, 8, false).expect("samples");
        let options = J2kLosslessEncodeOptions::new(
            EncodeBackendPreference::CpuOnly,
            J2kBlockCodingMode::Classic,
            J2kProgressionOrder::Lrcp,
            Some(2),
            ReversibleTransform::Rct53,
            J2kEncodeValidation::CpuRoundTrip,
        );
        encode_j2k_lossless(samples, &options)
            .expect("encode fixture")
            .codestream
    }
}
