use std::{fs, io, path::Path};

const SIMT_PRELUDE: &[u8] = include_bytes!("cuda_oxide_simt_prelude.rs");

pub(super) fn copy_project(
    source_dir: &Path,
    project_dir: &Path,
    codec_math_crate_path: &Path,
    extra_sources: &[&str],
) -> io::Result<()> {
    copy_file_as(
        source_dir,
        project_dir,
        Path::new("Cargo.toml.in"),
        Path::new("Cargo.toml"),
        codec_math_crate_path,
    )?;
    copy_file(
        source_dir,
        project_dir,
        Path::new("rust-toolchain.toml"),
        codec_math_crate_path,
    )?;
    copy_file(
        source_dir,
        project_dir,
        Path::new("src/main.rs"),
        codec_math_crate_path,
    )?;
    copy_file_as(
        source_dir,
        project_dir,
        Path::new("simt/Cargo.toml.in"),
        Path::new("simt/Cargo.toml"),
        codec_math_crate_path,
    )?;
    copy_file(
        source_dir,
        project_dir,
        Path::new("simt/src/main.rs"),
        codec_math_crate_path,
    )?;
    for relative in extra_sources {
        copy_file(
            source_dir,
            project_dir,
            Path::new(relative),
            codec_math_crate_path,
        )?;
    }
    Ok(())
}

fn copy_file(
    source_dir: &Path,
    project_dir: &Path,
    relative: &Path,
    codec_math_crate_path: &Path,
) -> io::Result<()> {
    copy_file_as(
        source_dir,
        project_dir,
        relative,
        relative,
        codec_math_crate_path,
    )
}

pub(super) fn copy_file_as(
    source_dir: &Path,
    project_dir: &Path,
    source_relative: &Path,
    dest_relative: &Path,
    codec_math_crate_path: &Path,
) -> io::Result<()> {
    let source = source_dir.join(source_relative);
    let dest = project_dir.join(dest_relative);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "failed to create cuda-oxide project dir {}: {error}",
                    parent.display()
                ),
            )
        })?;
    }
    if source_relative.extension().and_then(|value| value.to_str()) == Some("in") {
        let source_text = fs::read_to_string(&source).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "failed to read cuda-oxide project template {}: {error}",
                    source.display()
                ),
            )
        })?;
        let rendered = source_text.replace(
            "__J2K_CODEC_MATH_PATH__",
            &codec_math_crate_path.to_string_lossy(),
        );
        fs::write(&dest, rendered).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "failed to render cuda-oxide project template {} to {}: {error}",
                    source.display(),
                    dest.display()
                ),
            )
        })?;
    } else {
        fs::copy(&source, &dest).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "failed to stage cuda-oxide project file {} to {}: {error}",
                    source.display(),
                    dest.display()
                ),
            )
        })?;
    }
    Ok(())
}

pub(super) fn stage_cuda_oxide_shared_prelude(out_dir: &Path) -> io::Result<()> {
    let dest = out_dir.join("cuda_oxide_simt_prelude.rs");
    fs::write(&dest, SIMT_PRELUDE).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "failed to stage CUDA-Oxide SIMT prelude to {}: {error}",
                dest.display()
            ),
        )
    })
}

pub(super) fn stage_cuda_oxide_dependency_config(
    project_dir: &Path,
    j2k_types_crate_path: &Path,
) -> io::Result<()> {
    let cargo_dir = project_dir.join(".cargo");
    fs::create_dir_all(&cargo_dir).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "failed to create staged CUDA-Oxide Cargo config directory {}: {error}",
                cargo_dir.display()
            ),
        )
    })?;
    let config = format!(
        "[patch.crates-io]\nj2k-types = {{ path = {:?} }}\n",
        j2k_types_crate_path.to_string_lossy()
    );
    let destination = cargo_dir.join("config.toml");
    fs::write(&destination, config).map_err(|error| {
        io::Error::new(
            error.kind(),
            format!(
                "failed to stage CUDA-Oxide Cargo config {}: {error}",
                destination.display()
            ),
        )
    })
}
