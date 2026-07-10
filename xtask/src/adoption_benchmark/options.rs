use std::path::PathBuf;

#[derive(Debug, Clone)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "these fields preserve independent command-line switches and their require variants"
)]
pub(crate) struct AdoptionBenchmarkOptions {
    pub(super) out_dir: PathBuf,
    pub(super) input_dirs: Option<String>,
    pub(super) manifest: Option<PathBuf>,
    pub(super) encode_input_dirs: Option<String>,
    pub(super) encode_manifest: Option<PathBuf>,
    pub(super) cuda_decode_batch_sizes: Option<String>,
    pub(super) include_generated: bool,
    pub(super) quick: bool,
    pub(super) cuda: bool,
    pub(super) metal: bool,
    pub(super) openjph: bool,
    pub(super) kakadu: bool,
    pub(super) require_cuda: bool,
    pub(super) require_metal: bool,
    pub(super) require_openjph: bool,
    pub(super) require_kakadu: bool,
    pub(super) finalize_existing: bool,
}

impl AdoptionBenchmarkOptions {
    #[expect(
        clippy::too_many_lines,
        reason = "keeping option recognition and cross-option validation together makes the CLI contract auditable"
    )]
    pub(super) fn parse(mut args: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut options = Self {
            out_dir: PathBuf::from("target/j2k-adoption-benchmark"),
            input_dirs: None,
            manifest: None,
            encode_input_dirs: None,
            encode_manifest: None,
            cuda_decode_batch_sizes: None,
            include_generated: false,
            quick: false,
            cuda: false,
            metal: false,
            openjph: false,
            kakadu: false,
            require_cuda: false,
            require_metal: false,
            require_openjph: false,
            require_kakadu: false,
            finalize_existing: false,
        };
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--out-dir" => {
                    options.out_dir = PathBuf::from(
                        args.next()
                            .ok_or_else(|| "--out-dir requires a value".to_string())?,
                    );
                }
                "--fixtures" | "--input-dirs" => {
                    options.input_dirs = Some(
                        args.next()
                            .ok_or_else(|| format!("{arg} requires a platform path-list value"))?,
                    );
                }
                "--manifest" => {
                    options.manifest = Some(PathBuf::from(
                        args.next()
                            .ok_or_else(|| "--manifest requires a value".to_string())?,
                    ));
                }
                "--encode-fixtures" | "--encode-input-dirs" => {
                    options.encode_input_dirs = Some(
                        args.next()
                            .ok_or_else(|| format!("{arg} requires a platform path-list value"))?,
                    );
                }
                "--encode-manifest" => {
                    options.encode_manifest =
                        Some(PathBuf::from(args.next().ok_or_else(|| {
                            "--encode-manifest requires a value".to_string()
                        })?));
                }
                "--cuda-decode-batch-sizes" => {
                    options.cuda_decode_batch_sizes = Some(parse_batch_size_list(
                        &args.next().ok_or_else(|| {
                            "--cuda-decode-batch-sizes requires a comma-separated value".to_string()
                        })?,
                        "--cuda-decode-batch-sizes",
                    )?);
                }
                "--include-generated" => options.include_generated = true,
                "--quick" => options.quick = true,
                "--cuda" => options.cuda = true,
                "--metal" => options.metal = true,
                "--openjph" => options.openjph = true,
                "--kakadu" => options.kakadu = true,
                "--require-cuda" => {
                    options.cuda = true;
                    options.require_cuda = true;
                }
                "--require-metal" => {
                    options.metal = true;
                    options.require_metal = true;
                }
                "--require-openjph" => {
                    options.openjph = true;
                    options.require_openjph = true;
                }
                "--require-kakadu" => {
                    options.kakadu = true;
                    options.require_kakadu = true;
                }
                "--finalize-existing" => {
                    options.finalize_existing = true;
                }
                "--help" | "-h" => unreachable!("help handled before option parsing"),
                other => {
                    return Err(format!(
                        "unknown adoption-benchmark argument `{other}`\n{}",
                        help_text()
                    ))
                }
            }
        }
        if options.manifest.is_some() && options.input_dirs.is_none() {
            return Err("--manifest requires --fixtures/--input-dirs".to_string());
        }
        if options.encode_manifest.is_some() && options.encode_input_dirs.is_none() {
            return Err(
                "--encode-manifest requires --encode-fixtures/--encode-input-dirs".to_string(),
            );
        }
        if !options.include_generated
            && (options.input_dirs.is_none() || options.encode_input_dirs.is_none())
        {
            return Err(
                "external-only benchmark requires --fixtures and --encode-fixtures, or pass --include-generated for smoke runs"
                    .to_string(),
            );
        }
        Ok(options)
    }
}

pub(super) fn help_text() -> String {
    "usage: cargo xtask adoption-benchmark [--fixtures PATHS --manifest FILE] [--encode-fixtures PATHS --encode-manifest FILE] [--include-generated] [--quick] [--cuda|--require-cuda] [--cuda-decode-batch-sizes LIST] [--metal|--require-metal] [--openjph|--require-openjph] [--kakadu|--require-kakadu] [--finalize-existing] [--out-dir DIR]".to_string()
}

pub(super) fn parse_batch_size_list(value: &str, label: &str) -> Result<String, String> {
    let mut sizes = Vec::new();
    for raw in value.split(',') {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let size = raw
            .parse::<usize>()
            .map_err(|error| format!("{label} has invalid batch size {raw:?}: {error}"))?;
        if size == 0 {
            return Err(format!("{label} entries must be greater than zero"));
        }
        if !sizes.contains(&size) {
            sizes.push(size);
        }
    }
    if sizes.is_empty() {
        return Err(format!("{label} must include at least one batch size"));
    }
    Ok(sizes
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(","))
}
