//! Clean consumer validation for the packaged `j2k-ml` source archive.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use flate2::read::GzDecoder;

use crate::process::{cargo, run_command_owned, CommandContext};

use super::{append_patch_config_args, PackageGateStep};

pub(super) const CORE_PACKAGE_CONSUMERS: [&str; 4] =
    ["j2k", "j2k-cuda", "j2k-metal", "j2k-mpsgraph"];

pub(super) fn j2k_ml_consumer_checks(target_os: &str) -> &'static [&'static str] {
    match target_os {
        "linux" => &["cpu", "cuda", "cpu,cuda"],
        "macos" => &["cpu", "metal", "cpu,metal"],
        _ => &["cpu"],
    }
}

fn toml_string(value: &str) -> Result<String, String> {
    serde_json::to_string(value).map_err(|error| format!("failed to quote TOML string: {error}"))
}

pub(super) fn j2k_ml_consumer_manifest(
    step: &PackageGateStep,
    packaged_crate_path: &str,
) -> Result<String, String> {
    let mut manifest = format!(
        "[package]\nname = \"j2k-ml-package-consumer\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n\
         [features]\ndefault = []\ncpu = [\"j2k-ml/cpu\", \"dep:burn-flex\"]\ncuda = [\"j2k-ml/cuda\"]\nmetal = [\"j2k-ml/metal\"]\n\n\
         [dependencies]\nburn-flex = {{ version = \"0.21.0\", default-features = false, features = [\"std\"], optional = true }}\n\
         j2k = \"={version}\"\nj2k-ml = {{ version = \"={version}\", default-features = false }}\n\n\
         [patch.crates-io]\nj2k-ml = {{ path = {packaged_crate_path} }}\n",
        version = step.version,
        packaged_crate_path = toml_string(packaged_crate_path)?,
    );
    for (dependency, path) in &step.patches {
        writeln!(
            &mut manifest,
            "{dependency} = {{ path = {} }}",
            toml_string(path)?
        )
        .unwrap();
    }
    Ok(manifest)
}

pub(super) fn package_consumer_manifest(
    step: &PackageGateStep,
    packaged_crates: &BTreeMap<String, PathBuf>,
) -> Result<String, String> {
    let current = packaged_crates.get(&step.package).ok_or_else(|| {
        format!(
            "clean consumer is missing packaged source for `{}`",
            step.package
        )
    })?;
    let features = if step.package == "j2k-cuda" {
        "[features]\ndefault = []\ncuda-runtime = [\"j2k-cuda/cuda-runtime\"]\n\n"
    } else {
        ""
    };
    let mut manifest = format!(
        "[package]\nname = \"{}-package-consumer\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n\
         {}\
         [dependencies]\n{} = \"={}\"\n\n\
         [patch.crates-io]\n{} = {{ path = {} }}\n",
        step.package,
        features,
        step.package,
        step.version,
        step.package,
        toml_string(&current.to_string_lossy())?,
    );
    for (dependency, _) in &step.patches {
        let path = packaged_crates.get(dependency).ok_or_else(|| {
            format!("clean consumer is missing packaged source for `{dependency}`")
        })?;
        writeln!(
            &mut manifest,
            "{dependency} = {{ path = {} }}",
            toml_string(&path.to_string_lossy())?
        )
        .unwrap();
    }
    Ok(manifest)
}

pub(super) fn package_consumer_source(package: &str) -> Result<&'static str, String> {
    match package {
        "j2k" => Ok("fn main() { let _ = j2k::J2kDecoder::new(&[]); }\n"),
        "j2k-cuda" => Ok("fn main() { let _ = j2k_cuda::J2kDecoder::new(&[]); }\n"),
        "j2k-metal" => Ok("fn main() { let _ = j2k_metal::J2kDecoder::new(&[]); }\n"),
        "j2k-mpsgraph" => Ok(
            "fn main() { let _ = j2k_mpsgraph::MpsGraphBatchDecoder::system_default(Default::default()); }\n",
        ),
        _ => Err(format!(
            "no clean package consumer source is defined for `{package}`"
        )),
    }
}

pub(super) const CONSUMER_SOURCE: &str = r#"use j2k::BatchDecodeOptions;

#[cfg(feature = "cpu")]
fn check_cpu_api() {
    let _decoder = j2k_ml::CpuBurnDecoder::<burn_flex::Flex>::new(
        burn_flex::FlexDevice,
        BatchDecodeOptions::default(),
    );
}

#[cfg(feature = "cuda")]
fn check_cuda_api() {
    let _decoder = j2k_ml::CudaUploadBurnDecoder::new(
        Default::default(),
        BatchDecodeOptions::default(),
    );
    let _compat_decoder: j2k_ml::CudaBurnDecoder = j2k_ml::CudaBurnDecoder::new(
        Default::default(),
        BatchDecodeOptions::default(),
    );
}

#[cfg(feature = "metal")]
fn check_metal_api() {
    let _decoder = j2k_ml::MetalUploadBurnDecoder::system_default(BatchDecodeOptions::default());
    let _compat_decoder: Result<j2k_ml::MetalBurnDecoder, _> =
        j2k_ml::MetalBurnDecoder::system_default(BatchDecodeOptions::default());
}

fn main() {
    #[cfg(feature = "cpu")]
    check_cpu_api();
    #[cfg(feature = "cuda")]
    check_cuda_api();
    #[cfg(feature = "metal")]
    check_metal_api();
}
"#;

fn fresh_consumer_dir() -> Result<PathBuf, String> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock precedes Unix epoch: {error}"))?
        .as_nanos();
    let path = env::temp_dir().join(format!(
        "j2k-ml-package-consumer-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(path.join("src")).map_err(|error| {
        format!(
            "failed to create clean j2k-ml consumer at {}: {error}",
            path.display()
        )
    })?;
    Ok(path)
}

pub(super) fn package_archive_path(
    metadata: &serde_json::Value,
    step: &PackageGateStep,
) -> Result<PathBuf, String> {
    let target_directory = metadata
        .get("target_directory")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "cargo metadata has no target_directory".to_string())?;
    Ok(Path::new(target_directory)
        .join("package")
        .join(format!("{}-{}.crate", step.package, step.version)))
}

pub(super) fn extract_packaged_crate(
    archive_path: &Path,
    destination: &Path,
    step: &PackageGateStep,
) -> Result<PathBuf, String> {
    fs::create_dir_all(destination).map_err(|error| {
        format!(
            "failed to create package extraction directory {}: {error}",
            destination.display()
        )
    })?;
    let archive_file = File::open(archive_path).map_err(|error| {
        format!(
            "failed to open packaged source {}: {error}",
            archive_path.display()
        )
    })?;
    let mut archive = tar::Archive::new(GzDecoder::new(archive_file));
    archive.unpack(destination).map_err(|error| {
        format!(
            "failed to extract packaged source {}: {error}",
            archive_path.display()
        )
    })?;
    let crate_path = destination.join(format!("{}-{}", step.package, step.version));
    if !crate_path.join("Cargo.toml").is_file() {
        return Err(format!(
            "packaged source {} did not contain {}/Cargo.toml",
            archive_path.display(),
            crate_path.display()
        ));
    }
    Ok(crate_path)
}

pub(super) fn run_j2k_ml_consumer_gate(
    step: &PackageGateStep,
    archive_path: &Path,
) -> Result<(), String> {
    let consumer = fresh_consumer_dir()?;
    let result = (|| {
        let packaged_crate =
            extract_packaged_crate(archive_path, &consumer.join("packaged"), step)?;
        fs::write(
            consumer.join("Cargo.toml"),
            j2k_ml_consumer_manifest(step, &packaged_crate.to_string_lossy())?,
        )
        .map_err(|error| format!("failed to write clean consumer manifest: {error}"))?;
        fs::write(consumer.join("src/main.rs"), CONSUMER_SOURCE)
            .map_err(|error| format!("failed to write clean consumer source: {error}"))?;

        let target_dir = consumer.join("target");
        for features in j2k_ml_consumer_checks(env::consts::OS) {
            run_command_owned(
                cargo(),
                &[
                    "check".to_string(),
                    "--no-default-features".to_string(),
                    "--features".to_string(),
                    (*features).to_string(),
                ],
                CommandContext::new()
                    .current_dir(&consumer)
                    .target_dir(&target_dir),
            )?;
        }
        let combined = j2k_ml_consumer_checks(env::consts::OS)
            .last()
            .copied()
            .unwrap_or("cpu");
        let mut example_args = vec![
            "check".to_string(),
            "--examples".to_string(),
            "--no-default-features".to_string(),
            "--features".to_string(),
            combined.to_string(),
        ];
        append_patch_config_args(&mut example_args, step)?;
        run_command_owned(
            cargo(),
            &example_args,
            CommandContext::new()
                .current_dir(&packaged_crate)
                .target_dir(&target_dir),
        )?;
        let mut doc_args = vec![
            "doc".to_string(),
            "--no-deps".to_string(),
            "--no-default-features".to_string(),
            "--features".to_string(),
            combined.to_string(),
        ];
        append_patch_config_args(&mut doc_args, step)?;
        run_command_owned(
            cargo(),
            &doc_args,
            CommandContext::new()
                .current_dir(&packaged_crate)
                .target_dir(&target_dir),
        )
    })();
    let cleanup = fs::remove_dir_all(&consumer).map_err(|error| {
        format!(
            "failed to remove clean j2k-ml consumer {}: {error}",
            consumer.display()
        )
    });
    result.and(cleanup)
}

pub(super) fn run_core_package_consumer_gates(
    metadata: &serde_json::Value,
    plan: &[PackageGateStep],
    consumers: &[&str],
    cuda_runtime: bool,
) -> Result<(), String> {
    let consumer = fresh_core_consumer_dir()?;
    let result = (|| {
        let targets = consumers
            .iter()
            .map(|package| {
                plan.iter()
                    .find(|step| step.package == *package)
                    .ok_or_else(|| format!("package gate plan omitted `{package}`"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let required = targets
            .iter()
            .flat_map(|step| {
                std::iter::once(step.package.as_str()).chain(
                    step.patches
                        .iter()
                        .map(|(dependency, _)| dependency.as_str()),
                )
            })
            .collect::<BTreeSet<_>>();
        let mut packaged_crates = BTreeMap::new();
        for package in required {
            let step = plan
                .iter()
                .find(|step| step.package == package)
                .ok_or_else(|| format!("package gate plan omitted dependency `{package}`"))?;
            let extracted = extract_packaged_crate(
                &package_archive_path(metadata, step)?,
                &consumer.join("packaged"),
                step,
            )?;
            packaged_crates.insert(package.to_string(), extracted);
        }

        let target_dir = consumer.join("target");
        for step in targets {
            let project = consumer.join(&step.package);
            fs::create_dir_all(project.join("src")).map_err(|error| {
                format!(
                    "failed to create clean {} consumer at {}: {error}",
                    step.package,
                    project.display()
                )
            })?;
            fs::write(
                project.join("Cargo.toml"),
                package_consumer_manifest(step, &packaged_crates)?,
            )
            .map_err(|error| {
                format!(
                    "failed to write clean {} consumer manifest: {error}",
                    step.package
                )
            })?;
            fs::write(
                project.join("src/main.rs"),
                package_consumer_source(&step.package)?,
            )
            .map_err(|error| {
                format!(
                    "failed to write clean {} consumer source: {error}",
                    step.package
                )
            })?;
            let mut args = vec!["check".to_string()];
            if cuda_runtime && step.package == "j2k-cuda" {
                args.extend(["--features".to_string(), "cuda-runtime".to_string()]);
            }
            run_command_owned(
                cargo(),
                &args,
                CommandContext::new()
                    .current_dir(&project)
                    .target_dir(&target_dir),
            )?;
        }
        Ok(())
    })();
    let cleanup = fs::remove_dir_all(&consumer).map_err(|error| {
        format!(
            "failed to remove clean core/GPU consumers {}: {error}",
            consumer.display()
        )
    });
    result.and(cleanup)
}

fn fresh_core_consumer_dir() -> Result<PathBuf, String> {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| format!("system clock precedes Unix epoch: {error}"))?
        .as_nanos();
    let path = env::temp_dir().join(format!(
        "j2k-core-gpu-package-consumers-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir_all(&path).map_err(|error| {
        format!(
            "failed to create clean core/GPU consumer root at {}: {error}",
            path.display()
        )
    })?;
    Ok(path)
}
