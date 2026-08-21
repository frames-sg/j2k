// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared CUDA-Oxide project staging and PTX packaging for internal engine crates.

use std::{
    env,
    error::Error,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

mod staging;

#[cfg(test)]
use staging::copy_file_as;
use staging::{copy_project, stage_cuda_oxide_dependency_config, stage_cuda_oxide_shared_prelude};

const REQUIRE_CUDA_OXIDE_BUILD_ENV: &str = "J2K_REQUIRE_CUDA_OXIDE_BUILD";
const CODEC_MATH_MANIFEST_DIR_ENV: &str = "DEP_J2K_CODEC_MATH_MANIFEST_DIR";
const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");
/// One feature-gated CUDA-Oxide project packaged by a codec engine.
#[derive(Clone, Copy, Debug)]
pub struct CudaOxideProject {
    /// Cargo feature environment variable that enables this project.
    pub feature_env: &'static str,
    /// Source directory relative to the calling crate manifest.
    pub source_dir: &'static str,
    /// NUL-terminated PTX file written into the calling crate's `OUT_DIR`.
    pub output_name: &'static str,
    /// PTX artifact emitted by `cargo oxide build`.
    pub artifact_name: &'static str,
    /// Human-readable build diagnostic name.
    pub display_name: &'static str,
    /// Rust sources beyond the standard staged project files.
    pub extra_sources: &'static [&'static str],
    /// Rust cfg emitted when the real CUDA-Oxide build succeeds.
    pub built_cfg: &'static str,
}

/// Stage and compile the enabled CUDA-Oxide projects for one crate.
///
/// `aggregate_cfg`, when present, is emitted whenever any listed project
/// feature is enabled, including hosts that use placeholder PTX.
///
/// # Errors
///
/// Returns an error when Cargo omits required build metadata, the shared
/// codec-math source cannot be located, project staging fails, a required
/// CUDA-Oxide build cannot run, or its PTX artifact cannot be packaged.
pub fn build_cuda_oxide_projects(
    projects: &[CudaOxideProject],
    aggregate_cfg: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let out_dir = PathBuf::from(
        env::var_os("OUT_DIR").ok_or_else(|| io::Error::other("Cargo did not provide OUT_DIR"))?,
    );
    let host = env::var("HOST")?;
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR")
            .ok_or_else(|| io::Error::other("Cargo did not provide CARGO_MANIFEST_DIR"))?,
    );
    let codec_math_crate_path = codec_math_crate_path(&manifest_dir)?;
    emit_metadata(projects, aggregate_cfg, &codec_math_crate_path);
    stage_cuda_oxide_shared_prelude(&out_dir)?;

    let require_cuda_oxide = env::var_os(REQUIRE_CUDA_OXIDE_BUILD_ENV).is_some();
    for project in projects
        .iter()
        .filter(|project| env::var_os(project.feature_env).is_some())
    {
        if compile_project(
            &manifest_dir,
            &out_dir,
            &host,
            &codec_math_crate_path,
            *project,
            require_cuda_oxide,
        )? {
            println!("cargo:rustc-cfg={}", project.built_cfg);
        }
    }
    Ok(())
}

fn codec_math_crate_path(manifest_dir: &Path) -> Result<PathBuf, io::Error> {
    if let Some(path) = env::var_os(CODEC_MATH_MANIFEST_DIR_ENV) {
        let path = PathBuf::from(path);
        if path.join("Cargo.toml").is_file() {
            return Ok(path);
        }
        return Err(io::Error::other(format!(
            "{CODEC_MATH_MANIFEST_DIR_ENV} does not identify a j2k-codec-math crate: {}",
            path.display()
        )));
    }

    fallback_codec_math_crate_path(manifest_dir, PACKAGE_VERSION)
}

fn fallback_codec_math_crate_path(
    manifest_dir: &Path,
    version: &str,
) -> Result<PathBuf, io::Error> {
    sibling_crate_path(manifest_dir, "j2k-codec-math", version).map_err(|_| {
        let parent = manifest_dir
        .parent()
        .map_or_else(|| Path::new("<missing parent>"), |path| path);
        io::Error::other(format!(
            "Cargo did not provide {CODEC_MATH_MANIFEST_DIR_ENV} and no sibling j2k-codec-math crate exists at {} or {}",
            parent.join("j2k-codec-math").display(),
            parent.join(format!("j2k-codec-math-{version}")).display()
        ))
    })
}

fn j2k_types_crate_path(codec_math_crate_path: &Path, version: &str) -> io::Result<PathBuf> {
    sibling_crate_path(codec_math_crate_path, "j2k-types", version).map_err(|_| {
        let parent = codec_math_crate_path
            .parent()
            .map_or_else(|| Path::new("<missing parent>"), |path| path);
        io::Error::other(format!(
            "no sibling j2k-types crate exists at {} or {}",
            parent.join("j2k-types").display(),
            parent.join(format!("j2k-types-{version}")).display()
        ))
    })
}

fn sibling_crate_path(manifest_dir: &Path, crate_name: &str, version: &str) -> io::Result<PathBuf> {
    let parent = manifest_dir
        .parent()
        .ok_or_else(|| io::Error::other("CUDA crate manifest has no crates parent"))?;
    for candidate in [
        parent.join(crate_name),
        parent.join(format!("{crate_name}-{version}")),
    ] {
        if candidate.join("Cargo.toml").is_file() {
            return Ok(candidate);
        }
    }
    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!("no sibling {crate_name} crate"),
    ))
}

fn emit_metadata(
    projects: &[CudaOxideProject],
    aggregate_cfg: Option<&str>,
    codec_math_crate_path: &Path,
) {
    for relative in [
        "src/lib.rs",
        "src/classic.rs",
        "src/dwt.rs",
        "src/jpeg.rs",
        "src/mct.rs",
    ] {
        println!(
            "cargo:rerun-if-changed={}",
            codec_math_crate_path.join(relative).display()
        );
    }
    for project in projects {
        for relative in [
            "Cargo.toml.in",
            "rust-toolchain.toml",
            "src/main.rs",
            "simt/Cargo.toml.in",
            "simt/src/main.rs",
        ]
        .into_iter()
        .chain(project.extra_sources.iter().copied())
        {
            println!("cargo:rerun-if-changed={}/{}", project.source_dir, relative);
        }
        println!("cargo:rustc-check-cfg=cfg({})", project.built_cfg);
    }
    println!("cargo:rerun-if-env-changed=J2K_CUDA_OXIDE_ARCH");
    println!("cargo:rerun-if-env-changed={REQUIRE_CUDA_OXIDE_BUILD_ENV}");
    if let Some(cfg) = aggregate_cfg {
        println!("cargo:rustc-check-cfg=cfg({cfg})");
        if projects
            .iter()
            .any(|project| env::var_os(project.feature_env).is_some())
        {
            println!("cargo:rustc-cfg={cfg}");
        }
    }
}

fn compile_project(
    manifest_dir: &Path,
    out_dir: &Path,
    host: &str,
    codec_math_crate_path: &Path,
    project: CudaOxideProject,
    require_cuda_oxide: bool,
) -> io::Result<bool> {
    let output = out_dir.join(project.output_name);
    if !host.contains("linux") {
        return skip_project(
            &output,
            require_cuda_oxide,
            project.display_name,
            &format!(
                "{} requires a Linux host; current HOST={host}",
                project.display_name
            ),
        );
    }

    let source_dir = manifest_dir.join(project.source_dir);
    let project_dir = out_dir.join(project.output_name.trim_end_matches(".ptx"));
    stage_project(
        &source_dir,
        &project_dir,
        codec_math_crate_path,
        project.extra_sources,
    )?;

    let arch = env::var("J2K_CUDA_OXIDE_ARCH").unwrap_or_else(|_| "sm_80".to_string());
    println!(
        "cargo:warning=building {} with `cargo oxide build --arch {arch}`",
        project.display_name
    );
    let status = cargo_oxide_build_status(&project_dir, &arch);
    let status = match status {
        Ok(status) => status,
        Err(error) => {
            return skip_project(
                &output,
                require_cuda_oxide,
                project.display_name,
                &format!("failed to invoke cargo oxide build: {error}"),
            );
        }
    };
    if !status.success() {
        return skip_project(
            &output,
            require_cuda_oxide,
            project.display_name,
            &format!("{} build failed with status {status}", project.display_name),
        );
    }

    let generated = project_dir.join("ptx").join(project.artifact_name);
    let mut bytes = fs::read(&generated).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "{} build did not produce {}: {error}",
                project.display_name,
                generated.display()
            ),
        )
    })?;
    if bytes.last().copied() != Some(0) {
        bytes.push(0);
    }
    fs::write(&output, bytes).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "failed to write {} PTX to {}: {error}",
                project.display_name,
                output.display()
            ),
        )
    })?;
    Ok(true)
}

fn stage_project(
    source_dir: &Path,
    project_dir: &Path,
    codec_math_crate_path: &Path,
    extra_sources: &[&str],
) -> io::Result<()> {
    let j2k_types_crate_path = j2k_types_crate_path(codec_math_crate_path, PACKAGE_VERSION)?;
    copy_project(
        source_dir,
        project_dir,
        codec_math_crate_path,
        extra_sources,
    )?;
    stage_cuda_oxide_dependency_config(project_dir, &j2k_types_crate_path)
}

fn cargo_oxide_build_status(
    project_dir: &Path,
    arch: &str,
) -> io::Result<std::process::ExitStatus> {
    Command::new("cargo")
        .args(["oxide", "build", "--arch"])
        .arg(arch)
        .env_remove("RUSTUP_TOOLCHAIN")
        .env_remove("RUSTC")
        .env_remove("RUSTC_WRAPPER")
        .env_remove("RUSTC_WORKSPACE_WRAPPER")
        .env_remove("RUSTDOC")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .current_dir(project_dir)
        .status()
}

fn skip_project(
    output: &Path,
    required: bool,
    display_name: &str,
    message: &str,
) -> io::Result<bool> {
    if required {
        return Err(io::Error::other(message.to_string()));
    }
    println!("cargo:warning=skipping {display_name} build: {message}");
    fs::write(output, b".version 7.0\n.target sm_52\n.address_size 64\n\0").map_err(|error| {
        io::Error::new(
            error.kind(),
            format!("write placeholder {display_name} PTX: {error}"),
        )
    })?;
    Ok(false)
}

#[cfg(test)]
mod tests;
