use std::{
    fs,
    path::{Path, PathBuf},
};

use j2k::{
    encode_j2k_lossless, wrap_j2k_codestream, EncodeBackendPreference, J2kBlockCodingMode,
    J2kEncodeValidation, J2kFileWrapOptions, J2kLosslessEncodeOptions, J2kLosslessSamples,
    J2kProgressionOrder, ReversibleTransform,
};
use j2k_test_support::{fnv1a64_hex, pnm_bytes};

use crate::adoption_corpus::{
    canonical_label, collect_encode_source_paths, corpus_category, corpus_name, load_source_image,
    manifest_row, validate_tsv_field, SourceImage,
};

const MIN_PROFILE_DIMENSION: u32 = 128;

#[derive(Debug)]
struct AdoptionMaterializeOptions {
    out_dir: PathBuf,
    source_dirs: String,
    corpus_name: Option<String>,
    corpus_category: Option<String>,
    license_status: String,
    source_command: String,
    profiles: Vec<MaterializeProfile>,
}

#[derive(Debug)]
struct MaterializedSource {
    root: PathBuf,
    source: PathBuf,
    staged_pnm: PathBuf,
    pixels: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaterializeProfile {
    Classic,
    Htj2k,
}

#[derive(Debug, Clone, Copy)]
enum DecodeContainer {
    RawCodestream,
    Jp2,
    Jph,
}

pub(crate) fn adoption_materialize(args: impl Iterator<Item = String>) -> Result<(), String> {
    let args = args.collect::<Vec<_>>();
    if args
        .iter()
        .any(|arg| matches!(arg.as_str(), "--help" | "-h"))
    {
        println!("{}", help_text());
        return Ok(());
    }
    let options = AdoptionMaterializeOptions::parse(args.into_iter())?;
    let sources = collect_encode_source_paths(&options.source_dirs)?;
    if sources.is_empty() {
        return Err("--encode-fixtures did not contain supported source images".to_string());
    }

    let staged_dir = options.out_dir.join("staged-pnm");
    let decode_dir = options.out_dir.join("decode-fixtures");
    fs::create_dir_all(&staged_dir)
        .map_err(|err| format!("failed to create {}: {err}", staged_dir.display()))?;
    for profile in &options.profiles {
        fs::create_dir_all(decode_dir.join(profile.dir_name())).map_err(|err| {
            format!(
                "failed to create {}: {err}",
                decode_dir.join(profile.dir_name()).display()
            )
        })?;
    }

    let mut materialized = Vec::with_capacity(sources.len());
    let mut decode_rows =
        "path\tcorpus_category\tcorpus_name\tlicense_status\tencode_command\tinput_fnv1a64\tsource_fnv1a64\tcodec\tcontainer\n".to_string();
    for (index, (root, source)) in sources.iter().enumerate() {
        let source_image = load_source_image(source)?;
        validate_source_shape(source, &source_image)?;
        let source_hash = fnv1a64_hex(&source_image.pixels);
        let id = materialized_id(index, source, &source_hash);
        let staged_pnm = staged_dir.join(format!(
            "{id}.{}",
            if source_image.channels == 1 {
                "pgm"
            } else {
                "ppm"
            }
        ));
        let staged_bytes = pnm_bytes(
            &source_image.pixels,
            source_image.width,
            source_image.height,
            usize::from(source_image.channels),
        )
        .map_err(|err| format!("stage {} as PNM: {err}", source.display()))?;
        fs::write(&staged_pnm, staged_bytes)
            .map_err(|err| format!("write staged PNM {}: {err}", staged_pnm.display()))?;

        for profile in &options.profiles {
            let codestream = encode_source_codestream(&source_image, *profile)?;
            write_decode_fixture_variant(
                &options,
                &mut decode_rows,
                &decode_dir,
                &id,
                source,
                root,
                &source_hash,
                *profile,
                DecodeContainer::RawCodestream,
                &codestream,
            )?;
            write_decode_fixture_variant(
                &options,
                &mut decode_rows,
                &decode_dir,
                &id,
                source,
                root,
                &source_hash,
                *profile,
                profile.file_container(),
                &codestream,
            )?;
        }

        materialized.push(MaterializedSource {
            root: root.clone(),
            source: source.clone(),
            staged_pnm,
            pixels: source_image.pixels,
        });
    }

    let encode_rows = encode_manifest_rows(&options, &materialized)?;
    let fixture_manifest = options.out_dir.join("fixtures.tsv");
    let encode_manifest = options.out_dir.join("encode-fixtures.tsv");
    fs::write(&fixture_manifest, decode_rows)
        .map_err(|err| format!("write {}: {err}", fixture_manifest.display()))?;
    fs::write(&encode_manifest, encode_rows)
        .map_err(|err| format!("write {}: {err}", encode_manifest.display()))?;
    write_readme(&options)?;
    eprintln!(
        "materialized {} source images into {}",
        materialized.len(),
        options.out_dir.display()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn write_decode_fixture_variant(
    options: &AdoptionMaterializeOptions,
    decode_rows: &mut String,
    decode_dir: &Path,
    id: &str,
    source: &Path,
    root: &Path,
    source_hash: &str,
    profile: MaterializeProfile,
    container: DecodeContainer,
    codestream: &[u8],
) -> Result<(), String> {
    let bytes = match container {
        DecodeContainer::RawCodestream => codestream.to_vec(),
        DecodeContainer::Jp2 => wrap_j2k_codestream(codestream, J2kFileWrapOptions::jp2())
            .map_err(|err| format!("wrap classic fixture as JP2: {err}"))?,
        DecodeContainer::Jph => wrap_j2k_codestream(codestream, J2kFileWrapOptions::jph())
            .map_err(|err| format!("wrap HTJ2K fixture as JPH: {err}"))?,
    };
    let fixture_path = decode_dir
        .join(profile.dir_name())
        .join(format!("{id}.{}", container.extension()));
    fs::write(&fixture_path, &bytes)
        .map_err(|err| format!("write fixture {}: {err}", fixture_path.display()))?;
    let fields = [
        canonical_label(&fixture_path)?,
        corpus_category(source, options.corpus_category.as_deref()),
        corpus_name(root, options.corpus_name.as_deref()),
        options.license_status.clone(),
        encode_command_label(&profile, container, source)?,
        fnv1a64_hex(&bytes),
        source_hash.to_string(),
        profile.codec().to_string(),
        container.manifest_label().to_string(),
    ];
    decode_rows.push_str(&manifest_row(&fields)?);
    Ok(())
}

fn encode_manifest_rows(
    options: &AdoptionMaterializeOptions,
    materialized: &[MaterializedSource],
) -> Result<String, String> {
    let mut out =
        "path\tcorpus_category\tcorpus_name\tlicense_status\tsource_command\tinput_fnv1a64\n"
            .to_string();
    for item in materialized {
        let fields = [
            canonical_label(&item.staged_pnm)?,
            corpus_category(&item.source, options.corpus_category.as_deref()),
            corpus_name(&item.root, options.corpus_name.as_deref()),
            options.license_status.clone(),
            source_command_label(options, &item.source)?,
            fnv1a64_hex(&item.pixels),
        ];
        out.push_str(&manifest_row(&fields)?);
    }
    Ok(out)
}

fn encode_source_codestream(
    image: &SourceImage,
    profile: MaterializeProfile,
) -> Result<Vec<u8>, String> {
    let samples = J2kLosslessSamples::new(
        &image.pixels,
        image.width,
        image.height,
        u16::from(image.channels),
        8,
        false,
    )
    .map_err(|err| format!("prepare lossless samples: {err}"))?;
    let options = J2kLosslessEncodeOptions::new(
        EncodeBackendPreference::CpuOnly,
        profile.block_coding_mode(),
        J2kProgressionOrder::Lrcp,
        Some(2),
        ReversibleTransform::Rct53,
        J2kEncodeValidation::CpuRoundTrip,
    );
    encode_j2k_lossless(samples, &options)
        .map(|encoded| encoded.codestream)
        .map_err(|err| format!("encode {} fixture: {err}", profile.label()))
}

fn validate_source_shape(path: &Path, image: &SourceImage) -> Result<(), String> {
    if image.width < MIN_PROFILE_DIMENSION || image.height < MIN_PROFILE_DIMENSION {
        return Err(format!(
            "{} is {}x{}; adoption materialization requires at least {MIN_PROFILE_DIMENSION}x{MIN_PROFILE_DIMENSION} so downstream encode-profile validation can enforce 3 resolution levels",
            path.display(),
            image.width,
            image.height
        ));
    }
    if !matches!(image.channels, 1 | 3) {
        return Err(format!(
            "{} has unsupported channel count {}; expected 1 or 3",
            path.display(),
            image.channels
        ));
    }
    Ok(())
}

fn materialized_id(index: usize, path: &Path, source_hash: &str) -> String {
    let stem = path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("source");
    format!(
        "{index:06}_{}_{}",
        sanitize_id(stem),
        &source_hash[..8.min(source_hash.len())]
    )
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
        "source".to_string()
    } else {
        trimmed.to_string()
    }
}

fn encode_command_label(
    profile: &MaterializeProfile,
    container: DecodeContainer,
    source: &Path,
) -> Result<String, String> {
    Ok(format!(
        "cargo-xtask-adoption-materialize;profile={};codec={};container={};source={}",
        profile.label(),
        profile.codec(),
        container.manifest_label(),
        canonical_label(source)?
    ))
}

fn source_command_label(
    options: &AdoptionMaterializeOptions,
    source: &Path,
) -> Result<String, String> {
    Ok(format!(
        "{};staged-pnm-from={}",
        options.source_command,
        canonical_label(source)?
    ))
}

fn write_readme(options: &AdoptionMaterializeOptions) -> Result<(), String> {
    let text = format!(
        "# Materialized J2K Adoption Fixtures\n\n\
Generated by `cargo xtask adoption-materialize`.\n\n\
Decode fixtures include raw codestream variants plus JP2 wrappers for classic \
J2K and JPH wrappers for HTJ2K. Rows in `fixtures.tsv` carry `source_fnv1a64`, \
so those variants remain tied to the \
same source image for publication diversity gates.\n\n\
Use these paths for the full adoption benchmark:\n\n\
```bash\n\
cargo xtask adoption-benchmark \\\n  --fixtures \"{}\" \\\n  --manifest \"{}\" \\\n  --encode-fixtures \"{}\" \\\n  --encode-manifest \"{}\" \\\n  --out-dir target/j2k-adoption-benchmark/full\n\
```\n\n\
Add `--require-cuda` and/or `--require-metal` on hardware runners. Do not publish \
until the adoption benchmark summary reports clean publication gates.\n",
        options.out_dir.join("decode-fixtures").display(),
        options.out_dir.join("fixtures.tsv").display(),
        options.out_dir.join("staged-pnm").display(),
        options.out_dir.join("encode-fixtures.tsv").display()
    );
    let path = options.out_dir.join("README.md");
    fs::write(&path, text).map_err(|err| format!("write {}: {err}", path.display()))
}

impl MaterializeProfile {
    const fn dir_name(self) -> &'static str {
        match self {
            Self::Classic => "classic",
            Self::Htj2k => "htj2k",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Classic => "classic-lossless-raw-lrcp-rct53-3resolutions",
            Self::Htj2k => "htj2k-lossless-raw-lrcp-rct53-3resolutions",
        }
    }

    const fn codec(self) -> &'static str {
        match self {
            Self::Classic => "j2k",
            Self::Htj2k => "htj2k",
        }
    }

    const fn block_coding_mode(self) -> J2kBlockCodingMode {
        match self {
            Self::Classic => J2kBlockCodingMode::Classic,
            Self::Htj2k => J2kBlockCodingMode::HighThroughput,
        }
    }

    const fn file_container(self) -> DecodeContainer {
        match self {
            Self::Classic => DecodeContainer::Jp2,
            Self::Htj2k => DecodeContainer::Jph,
        }
    }
}

impl DecodeContainer {
    const fn extension(self) -> &'static str {
        match self {
            Self::RawCodestream => "j2k",
            Self::Jp2 => "jp2",
            Self::Jph => "jph",
        }
    }

    const fn manifest_label(self) -> &'static str {
        match self {
            Self::RawCodestream => "raw-codestream",
            Self::Jp2 => "jp2",
            Self::Jph => "jph",
        }
    }
}

impl AdoptionMaterializeOptions {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut out_dir = PathBuf::from("target/j2k-adoption-materialized");
        let mut source_dirs = None;
        let mut corpus_name = None;
        let mut corpus_category = None;
        let mut license_status = String::new();
        let mut source_command = None;
        let mut profiles = vec![MaterializeProfile::Classic, MaterializeProfile::Htj2k];
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out-dir" => {
                    out_dir = PathBuf::from(
                        args.next()
                            .ok_or_else(|| "--out-dir requires a value".to_string())?,
                    );
                }
                "--encode-fixtures" | "--encode-inputs" | "--source-images" => {
                    source_dirs = Some(
                        args.next()
                            .ok_or_else(|| format!("{arg} requires a platform path-list value"))?,
                    );
                }
                "--corpus-name" => {
                    corpus_name = Some(
                        args.next()
                            .ok_or_else(|| "--corpus-name requires a value".to_string())?,
                    );
                }
                "--corpus-category" => {
                    corpus_category = Some(
                        args.next()
                            .ok_or_else(|| "--corpus-category requires a value".to_string())?,
                    );
                }
                "--license-status" => {
                    license_status = args
                        .next()
                        .ok_or_else(|| "--license-status requires a value".to_string())?;
                }
                "--source-command" | "--encode-source-command" => {
                    source_command = Some(
                        args.next()
                            .ok_or_else(|| format!("{arg} requires a value"))?,
                    );
                }
                "--profiles" => {
                    profiles = parse_profiles(&args.next().ok_or_else(|| {
                        "--profiles requires a comma-separated value".to_string()
                    })?)?;
                }
                "--help" | "-h" => unreachable!("help handled before option parsing"),
                other => {
                    return Err(format!(
                        "unknown adoption-materialize argument `{other}`\n{}",
                        help_text()
                    ));
                }
            }
        }
        let source_dirs = source_dirs.ok_or_else(|| "--encode-fixtures is required".to_string())?;
        if license_status.is_empty() {
            return Err("--license-status is required".to_string());
        }
        let source_command =
            source_command.ok_or_else(|| "--source-command is required".to_string())?;
        validate_tsv_field(&license_status)?;
        validate_tsv_field(&source_command)?;
        if let Some(value) = &corpus_name {
            validate_tsv_field(value)?;
        }
        if let Some(value) = &corpus_category {
            validate_tsv_field(value)?;
        }
        if profiles.is_empty() {
            return Err("--profiles must include at least one profile".to_string());
        }
        Ok(Self {
            out_dir,
            source_dirs,
            corpus_name,
            corpus_category,
            license_status,
            source_command,
            profiles,
        })
    }
}

fn parse_profiles(value: &str) -> Result<Vec<MaterializeProfile>, String> {
    let mut profiles = Vec::new();
    for raw in value
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
    {
        let profile = match raw.to_ascii_lowercase().as_str() {
            "all" => {
                for profile in [MaterializeProfile::Classic, MaterializeProfile::Htj2k] {
                    if !profiles.contains(&profile) {
                        profiles.push(profile);
                    }
                }
                continue;
            }
            "classic" | "j2k" => MaterializeProfile::Classic,
            "htj2k" | "ht" => MaterializeProfile::Htj2k,
            other => {
                return Err(format!(
                    "unknown materialize profile {other:?}; expected classic, htj2k, or all"
                ));
            }
        };
        if !profiles.contains(&profile) {
            profiles.push(profile);
        }
    }
    Ok(profiles)
}

fn help_text() -> String {
    "usage: cargo xtask adoption-materialize --encode-fixtures PATHS --license-status STATUS --source-command CMD [--profiles classic,htj2k] [--corpus-name NAME] [--corpus-category CATEGORY] [--out-dir DIR]".to_string()
}

#[cfg(test)]
mod tests {
    use super::{adoption_materialize, parse_profiles, MaterializeProfile};
    use j2k_test_support::pnm_bytes;

    #[test]
    fn parses_profiles_without_duplicates() {
        assert_eq!(
            parse_profiles("classic,htj2k,all").expect("profiles"),
            vec![MaterializeProfile::Classic, MaterializeProfile::Htj2k]
        );
    }

    #[test]
    fn materializes_source_images_into_decode_and_encode_manifests() {
        let root = std::env::current_dir()
            .expect("current dir")
            .join("target")
            .join("j2k-adoption-materialize-test")
            .join(std::process::id().to_string());
        let source_dir = root.join("kodak");
        let out_dir = root.join("out");
        std::fs::create_dir_all(&source_dir).expect("create source dir");
        let pixels = (0..128 * 128)
            .map(|index| u8::try_from(index % 251).expect("fits"))
            .collect::<Vec<_>>();
        std::fs::write(
            source_dir.join("kodim01.pgm"),
            pnm_bytes(&pixels, 128, 128, 1).expect("pnm"),
        )
        .expect("write source");

        adoption_materialize(
            [
                "--encode-fixtures",
                source_dir.to_str().expect("utf8 path"),
                "--source-command",
                "test-pgm",
                "--license-status",
                "cc0",
                "--out-dir",
                out_dir.to_str().expect("utf8 path"),
            ]
            .map(str::to_string)
            .into_iter(),
        )
        .expect("materialize");

        let fixture_manifest =
            std::fs::read_to_string(out_dir.join("fixtures.tsv")).expect("fixture manifest");
        let encode_manifest =
            std::fs::read_to_string(out_dir.join("encode-fixtures.tsv")).expect("encode manifest");
        assert!(fixture_manifest.starts_with(
            "path\tcorpus_category\tcorpus_name\tlicense_status\tencode_command\tinput_fnv1a64\tsource_fnv1a64\tcodec\tcontainer\n"
        ));
        assert!(fixture_manifest.contains("\tnatural-image\tkodak\tcc0\t"));
        assert!(fixture_manifest.contains("\tj2k\traw-codestream\n"));
        assert!(fixture_manifest.contains("\tj2k\tjp2\n"));
        assert!(fixture_manifest.contains("\thtj2k\traw-codestream\n"));
        assert!(fixture_manifest.contains("\thtj2k\tjph\n"));
        assert!(encode_manifest.contains("\tnatural-image\tkodak\tcc0\ttest-pgm;staged-pnm-from="));
        assert_eq!(fixture_manifest.matches("\n/").count(), 4);
        assert!(
            std::fs::read_dir(out_dir.join("decode-fixtures").join("classic"))
                .expect("classic dir")
                .next()
                .is_some()
        );
        assert!(
            std::fs::read_dir(out_dir.join("decode-fixtures").join("htj2k"))
                .expect("ht dir")
                .next()
                .is_some()
        );
    }
}
