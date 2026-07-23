//! Clean consumer validation for the packaged `j2k-ml` source archive.

use std::env;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use flate2::read::GzDecoder;

use crate::process::{cargo, run_command_owned, CommandContext};

use super::{append_patch_config_args, PackageGateStep};

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

const CONSUMER_SOURCE: &str = r#"use j2k::BatchDecodeOptions;

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
}

#[cfg(feature = "metal")]
fn check_metal_api() {
    let _decoder = j2k_ml::MetalUploadBurnDecoder::system_default(BatchDecodeOptions::default());
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
