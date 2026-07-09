use std::{fs, path::PathBuf};

use j2k_test_support::fnv1a64_hex;

use crate::adoption_corpus::{
    canonical_label, codec_from_bytes, collect_decode_fixture_paths, collect_encode_source_paths,
    container_from_path_and_bytes, corpus_category, corpus_name, manifest_row,
    source_image_pixel_hash, validate_tsv_field,
};

#[derive(Debug)]
struct AdoptionManifestOptions {
    out_dir: PathBuf,
    decode_dirs: Option<String>,
    encode_dirs: Option<String>,
    corpus_name: Option<String>,
    corpus_category: Option<String>,
    license_status: String,
    decode_encode_command: Option<String>,
    encode_source_command: Option<String>,
}

pub(crate) fn adoption_manifest(args: impl Iterator<Item = String>) -> Result<(), String> {
    let args = args.collect::<Vec<_>>();
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        println!("{}", help_text());
        return Ok(());
    }
    let options = AdoptionManifestOptions::parse(args.into_iter())?;
    fs::create_dir_all(&options.out_dir)
        .map_err(|err| format!("failed to create {}: {err}", options.out_dir.display()))?;
    if let Some(path_list) = &options.decode_dirs {
        let paths = collect_decode_fixture_paths(path_list)?;
        if paths.is_empty() {
            return Err("--decode-fixtures did not contain supported J2K fixtures".to_string());
        }
        write_decode_manifest(&options, &paths)?;
    }
    if let Some(path_list) = &options.encode_dirs {
        let paths = collect_encode_source_paths(path_list)?;
        if paths.is_empty() {
            return Err("--encode-fixtures did not contain supported source images".to_string());
        }
        write_encode_manifest(&options, &paths)?;
    }
    eprintln!(
        "wrote adoption benchmark manifests under {}",
        options.out_dir.display()
    );
    Ok(())
}

fn write_decode_manifest(
    options: &AdoptionManifestOptions,
    paths: &[(PathBuf, PathBuf)],
) -> Result<(), String> {
    let command = options
        .decode_encode_command
        .as_deref()
        .ok_or_else(|| "--decode-encode-command is required with --decode-fixtures".to_string())?;
    let mut out =
        "path\tcorpus_category\tcorpus_name\tlicense_status\tencode_command\tinput_fnv1a64\tcodec\tcontainer\n".to_string();
    for (root, path) in paths {
        let bytes = fs::read(path).map_err(|err| format!("read {}: {err}", path.display()))?;
        let fields = [
            canonical_label(path)?,
            corpus_category(path, options.corpus_category.as_deref()),
            corpus_name(root, options.corpus_name.as_deref()),
            options.license_status.clone(),
            command.to_string(),
            fnv1a64_hex(&bytes),
            codec_from_bytes(&bytes).to_string(),
            container_from_path_and_bytes(path, &bytes).to_string(),
        ];
        out.push_str(&manifest_row(&fields)?);
    }
    let manifest = options.out_dir.join("fixtures.tsv");
    fs::write(&manifest, out)
        .map_err(|err| format!("failed to write {}: {err}", manifest.display()))
}

fn write_encode_manifest(
    options: &AdoptionManifestOptions,
    paths: &[(PathBuf, PathBuf)],
) -> Result<(), String> {
    let command = options
        .encode_source_command
        .as_deref()
        .ok_or_else(|| "--encode-source-command is required with --encode-fixtures".to_string())?;
    let mut out =
        "path\tcorpus_category\tcorpus_name\tlicense_status\tsource_command\tinput_fnv1a64\n"
            .to_string();
    for (root, path) in paths {
        let fields = [
            canonical_label(path)?,
            corpus_category(path, options.corpus_category.as_deref()),
            corpus_name(root, options.corpus_name.as_deref()),
            options.license_status.clone(),
            command.to_string(),
            source_image_pixel_hash(path)?,
        ];
        out.push_str(&manifest_row(&fields)?);
    }
    let manifest = options.out_dir.join("encode-fixtures.tsv");
    fs::write(&manifest, out)
        .map_err(|err| format!("failed to write {}: {err}", manifest.display()))
}

impl AdoptionManifestOptions {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut options = Self {
            out_dir: PathBuf::from("target/j2k-adoption-manifests"),
            decode_dirs: None,
            encode_dirs: None,
            corpus_name: None,
            corpus_category: None,
            license_status: String::new(),
            decode_encode_command: None,
            encode_source_command: None,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out-dir" => {
                    options.out_dir = PathBuf::from(
                        args.next()
                            .ok_or_else(|| "--out-dir requires a value".to_string())?,
                    );
                }
                "--decode-fixtures" | "--fixtures" => {
                    options.decode_dirs = Some(
                        args.next()
                            .ok_or_else(|| format!("{arg} requires a platform path-list value"))?,
                    );
                }
                "--encode-fixtures" | "--encode-inputs" => {
                    options.encode_dirs = Some(
                        args.next()
                            .ok_or_else(|| format!("{arg} requires a platform path-list value"))?,
                    );
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
                "--decode-encode-command" => {
                    options.decode_encode_command =
                        Some(args.next().ok_or_else(|| {
                            "--decode-encode-command requires a value".to_string()
                        })?);
                }
                "--encode-source-command" => {
                    options.encode_source_command =
                        Some(args.next().ok_or_else(|| {
                            "--encode-source-command requires a value".to_string()
                        })?);
                }
                "--help" | "-h" => unreachable!("help handled before option parsing"),
                other => {
                    return Err(format!(
                        "unknown adoption-manifest argument `{other}`\n{}",
                        help_text()
                    ));
                }
            }
        }
        if options.decode_dirs.is_none() && options.encode_dirs.is_none() {
            return Err("--decode-fixtures or --encode-fixtures is required".to_string());
        }
        if options.license_status.is_empty() {
            return Err("--license-status is required".to_string());
        }
        validate_tsv_field(&options.license_status)?;
        if let Some(value) = &options.corpus_name {
            validate_tsv_field(value)?;
        }
        if let Some(value) = &options.corpus_category {
            validate_tsv_field(value)?;
        }
        if options.decode_dirs.is_some() && options.decode_encode_command.is_none() {
            return Err("--decode-encode-command is required with --decode-fixtures".to_string());
        }
        if options.encode_dirs.is_some() && options.encode_source_command.is_none() {
            return Err("--encode-source-command is required with --encode-fixtures".to_string());
        }
        Ok(options)
    }
}

fn help_text() -> String {
    "usage: cargo xtask adoption-manifest --license-status STATUS [--decode-fixtures PATHS --decode-encode-command CMD] [--encode-fixtures PATHS --encode-source-command CMD] [--corpus-name NAME] [--corpus-category CATEGORY] [--out-dir DIR]".to_string()
}

#[cfg(test)]
mod tests {
    use super::{adoption_manifest, AdoptionManifestOptions};

    #[test]
    fn requires_license_status() {
        let error = AdoptionManifestOptions::parse(
            [
                "--encode-fixtures",
                "images",
                "--encode-source-command",
                "source",
            ]
            .map(str::to_string)
            .into_iter(),
        )
        .expect_err("license is required");

        assert!(error.contains("--license-status is required"));
    }

    #[test]
    fn jpylyzer_paths_are_parser_robustness_not_interop() {
        assert_eq!(
            j2k_compare::common::infer_corpus_category(std::path::Path::new(
                "vendor/jpylyzer/invalid.jp2",
            )),
            "parser-robustness"
        );
    }

    #[test]
    fn writes_decode_and_encode_manifests() {
        let root = std::env::current_dir()
            .expect("current dir")
            .join("target")
            .join("j2k-adoption-manifest-test")
            .join(std::process::id().to_string());
        let decode_dir = root.join("openjpeg-data");
        let encode_dir = root.join("kodak");
        let out_dir = root.join("out");
        std::fs::create_dir_all(&decode_dir).expect("create decode dir");
        std::fs::create_dir_all(&encode_dir).expect("create encode dir");
        std::fs::write(decode_dir.join("fixture.jp2"), b"not-a-real-jp2").expect("fixture");
        std::fs::write(
            encode_dir.join("kodim01.pgm"),
            b"P5\n2 2\n255\n\x00\x01\x02\x03",
        )
        .expect("image");

        adoption_manifest(
            [
                "--decode-fixtures",
                decode_dir.to_str().expect("utf8 path"),
                "--decode-encode-command",
                "source-native",
                "--encode-fixtures",
                encode_dir.to_str().expect("utf8 path"),
                "--encode-source-command",
                "source-png",
                "--license-status",
                "cc0",
                "--out-dir",
                out_dir.to_str().expect("utf8 path"),
            ]
            .map(str::to_string)
            .into_iter(),
        )
        .expect("write manifests");

        let decode_manifest =
            std::fs::read_to_string(out_dir.join("fixtures.tsv")).expect("decode manifest");
        let encode_manifest =
            std::fs::read_to_string(out_dir.join("encode-fixtures.tsv")).expect("encode manifest");
        assert!(decode_manifest.starts_with(
            "path\tcorpus_category\tcorpus_name\tlicense_status\tencode_command\tinput_fnv1a64\tcodec\tcontainer\n"
        ));
        assert!(encode_manifest.starts_with(
            "path\tcorpus_category\tcorpus_name\tlicense_status\tsource_command\tinput_fnv1a64\n"
        ));
        assert!(decode_manifest.contains("\tinterop\topenjpeg-data\tcc0\tsource-native\t"));
        assert!(encode_manifest.contains("\tnatural-image\tkodak\tcc0\tsource-png\t"));
    }
}
