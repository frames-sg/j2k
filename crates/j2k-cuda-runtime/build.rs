use std::env;
use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

const REQUIRE_CUDA_OXIDE_BUILD_ENV: &str = "J2K_REQUIRE_CUDA_OXIDE_BUILD";
const CODEC_MATH_MANIFEST_DIR_ENV: &str = "DEP_J2K_CODEC_MATH_MANIFEST_DIR";
const CUDA_OXIDE_FEATURE_ENV_VARS: &[&str] = &[
    "CARGO_FEATURE_CUDA_OXIDE_COPY_U8",
    "CARGO_FEATURE_CUDA_OXIDE_J2K_ENCODE",
    "CARGO_FEATURE_CUDA_OXIDE_J2K_DECODE_STORE",
    "CARGO_FEATURE_CUDA_OXIDE_J2K_CLASSIC_DECODE",
    "CARGO_FEATURE_CUDA_OXIDE_J2K_DEQUANTIZE",
    "CARGO_FEATURE_CUDA_OXIDE_J2K_IDWT",
    "CARGO_FEATURE_CUDA_OXIDE_J2K_ML",
    "CARGO_FEATURE_CUDA_OXIDE_HTJ2K_DECODE",
    "CARGO_FEATURE_CUDA_OXIDE_HTJ2K_ENCODE",
    "CARGO_FEATURE_CUDA_OXIDE_TRANSCODE",
    "CARGO_FEATURE_CUDA_OXIDE_JPEG_DECODE",
    "CARGO_FEATURE_CUDA_OXIDE_JPEG_ENCODE",
];
const CUDA_OXIDE_J2K_ENCODE_EXTRA_SOURCES: &[&str] = &[
    "simt/src/abi.rs",
    "simt/src/constants.rs",
    "simt/src/dwt53.rs",
    "simt/src/dwt97.rs",
    "simt/src/exports.rs",
    "simt/src/helpers.rs",
    "simt/src/packet_writer.rs",
    "simt/src/packetization.rs",
    "simt/src/quantization.rs",
    "simt/src/tag_tree.rs",
];
const CUDA_OXIDE_J2K_DECODE_STORE_EXTRA_SOURCES: &[&str] = &[
    "simt/src/abi.rs",
    "simt/src/color.rs",
    "simt/src/exports.rs",
    "simt/src/layout.rs",
    "simt/src/memory.rs",
    "simt/src/native_color.rs",
    "simt/src/sample.rs",
    "simt/src/transform.rs",
];
const CUDA_OXIDE_TRANSCODE_EXTRA_SOURCES: &[&str] = &[
    "simt/src/abi.rs",
    "simt/src/constants.rs",
    "simt/src/dwt97.rs",
    "simt/src/exports.rs",
    "simt/src/helpers.rs",
    "simt/src/quantization.rs",
    "simt/src/reversible53.rs",
];
const CUDA_OXIDE_JPEG_DECODE_EXTRA_SOURCES: &[&str] = &["simt/src/component_planes.rs"];

struct BuildContext<'a> {
    out_dir: &'a Path,
    host: &'a str,
    codec_math_crate_path: &'a Path,
}

fn main() -> Result<(), Box<dyn Error>> {
    let out_dir = PathBuf::from(
        env::var_os("OUT_DIR").ok_or_else(|| io::Error::other("Cargo did not provide OUT_DIR"))?,
    );
    let host = env::var("HOST")?;
    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR")
            .ok_or_else(|| io::Error::other("Cargo did not provide CARGO_MANIFEST_DIR"))?,
    );
    let codec_math_crate_path = codec_math_crate_path(&manifest_dir)?;
    emit_build_script_metadata(&codec_math_crate_path);
    compile_cuda_oxide_feature_projects(&BuildContext {
        out_dir: &out_dir,
        host: &host,
        codec_math_crate_path: &codec_math_crate_path,
    });
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

    let workspace_path = manifest_dir
        .parent()
        .ok_or_else(|| {
            io::Error::other("j2k-cuda-runtime manifest directory has no crates parent")
        })?
        .join("j2k-codec-math");
    if workspace_path.join("Cargo.toml").is_file() {
        return Ok(workspace_path);
    }
    Err(io::Error::other(format!(
        "Cargo did not provide {CODEC_MATH_MANIFEST_DIR_ENV} and no workspace j2k-codec-math crate exists at {}",
        workspace_path.display()
    )))
}

fn emit_build_script_metadata(codec_math_crate_path: &Path) {
    emit_codec_math_rerun_inputs(codec_math_crate_path);
    println!("cargo:rerun-if-changed=src/cuda_oxide_simt_prelude.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_copy_u8/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_copy_u8/rust-toolchain.toml");
    println!("cargo:rerun-if-changed=src/cuda_oxide_copy_u8/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_copy_u8/simt/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_copy_u8/simt/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_encode/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_encode/rust-toolchain.toml");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_encode/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_encode/simt/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_encode/simt/src/main.rs");
    for relative in CUDA_OXIDE_J2K_ENCODE_EXTRA_SOURCES {
        println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_encode/{relative}");
    }
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_decode_store/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_decode_store/rust-toolchain.toml");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_decode_store/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_decode_store/simt/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_decode_store/simt/src/main.rs");
    for relative in CUDA_OXIDE_J2K_DECODE_STORE_EXTRA_SOURCES {
        println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_decode_store/{relative}");
    }
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_classic_decode/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_classic_decode/rust-toolchain.toml");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_classic_decode/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_classic_decode/simt/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_classic_decode/simt/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_dequantize/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_dequantize/rust-toolchain.toml");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_dequantize/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_dequantize/simt/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_dequantize/simt/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_idwt/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_idwt/rust-toolchain.toml");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_idwt/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_idwt/simt/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_idwt/simt/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_ml/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_ml/rust-toolchain.toml");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_ml/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_ml/simt/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_ml/simt/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_htj2k_decode/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_htj2k_decode/rust-toolchain.toml");
    println!("cargo:rerun-if-changed=src/cuda_oxide_htj2k_decode/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_htj2k_decode/simt/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_htj2k_decode/simt/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_htj2k_encode/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_htj2k_encode/rust-toolchain.toml");
    println!("cargo:rerun-if-changed=src/cuda_oxide_htj2k_encode/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_htj2k_encode/simt/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_htj2k_encode/simt/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_transcode/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_transcode/rust-toolchain.toml");
    println!("cargo:rerun-if-changed=src/cuda_oxide_transcode/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_transcode/simt/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_transcode/simt/src/main.rs");
    for relative in CUDA_OXIDE_TRANSCODE_EXTRA_SOURCES {
        println!("cargo:rerun-if-changed=src/cuda_oxide_transcode/{relative}");
    }
    println!("cargo:rerun-if-changed=src/cuda_oxide_jpeg_decode/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_jpeg_decode/rust-toolchain.toml");
    println!("cargo:rerun-if-changed=src/cuda_oxide_jpeg_decode/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_jpeg_decode/simt/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_jpeg_decode/simt/src/main.rs");
    for relative in CUDA_OXIDE_JPEG_DECODE_EXTRA_SOURCES {
        println!("cargo:rerun-if-changed=src/cuda_oxide_jpeg_decode/{relative}");
    }
    println!("cargo:rerun-if-changed=src/cuda_oxide_jpeg_encode/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_jpeg_encode/rust-toolchain.toml");
    println!("cargo:rerun-if-changed=src/cuda_oxide_jpeg_encode/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_jpeg_encode/simt/Cargo.toml.in");
    println!("cargo:rerun-if-changed=src/cuda_oxide_jpeg_encode/simt/src/main.rs");
    println!("cargo:rerun-if-env-changed=J2K_CUDA_OXIDE_ARCH");
    println!("cargo:rerun-if-env-changed={REQUIRE_CUDA_OXIDE_BUILD_ENV}");
    println!("cargo:rustc-check-cfg=cfg(j2k_cuda_oxide_copy_u8_built)");
    println!("cargo:rustc-check-cfg=cfg(j2k_cuda_oxide_j2k_encode_built)");
    println!("cargo:rustc-check-cfg=cfg(j2k_cuda_oxide_j2k_decode_store_built)");
    println!("cargo:rustc-check-cfg=cfg(j2k_cuda_oxide_j2k_classic_decode_built)");
    println!("cargo:rustc-check-cfg=cfg(j2k_cuda_oxide_j2k_dequantize_built)");
    println!("cargo:rustc-check-cfg=cfg(j2k_cuda_oxide_j2k_idwt_built)");
    println!("cargo:rustc-check-cfg=cfg(j2k_cuda_oxide_j2k_ml_built)");
    println!("cargo:rustc-check-cfg=cfg(j2k_cuda_oxide_htj2k_decode_built)");
    println!("cargo:rustc-check-cfg=cfg(j2k_cuda_oxide_htj2k_encode_built)");
    println!("cargo:rustc-check-cfg=cfg(j2k_cuda_oxide_transcode_built)");
    println!("cargo:rustc-check-cfg=cfg(j2k_cuda_oxide_jpeg_decode_built)");
    println!("cargo:rustc-check-cfg=cfg(j2k_cuda_oxide_jpeg_encode_built)");
    println!("cargo:rustc-check-cfg=cfg(j2k_cuda_oxide_enabled)");
    if CUDA_OXIDE_FEATURE_ENV_VARS
        .iter()
        .any(|feature| env::var_os(feature).is_some())
    {
        println!("cargo:rustc-cfg=j2k_cuda_oxide_enabled");
    }
}

fn emit_codec_math_rerun_inputs(codec_math_crate_path: &Path) {
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
}

fn compile_cuda_oxide_feature_projects(context: &BuildContext<'_>) {
    let require_all_cuda_oxide = env::var_os(REQUIRE_CUDA_OXIDE_BUILD_ENV).is_some();
    stage_cuda_oxide_shared_prelude(context.out_dir);
    if env::var_os("CARGO_FEATURE_CUDA_OXIDE_COPY_U8").is_some()
        && compile_cuda_oxide_copy_u8(context, require_all_cuda_oxide)
    {
        println!("cargo:rustc-cfg=j2k_cuda_oxide_copy_u8_built");
    }

    if env::var_os("CARGO_FEATURE_CUDA_OXIDE_J2K_ENCODE").is_some()
        && compile_cuda_oxide_j2k_encode(context, require_all_cuda_oxide)
    {
        println!("cargo:rustc-cfg=j2k_cuda_oxide_j2k_encode_built");
    }

    if env::var_os("CARGO_FEATURE_CUDA_OXIDE_J2K_DECODE_STORE").is_some()
        && compile_cuda_oxide_j2k_decode_store(context, require_all_cuda_oxide)
    {
        println!("cargo:rustc-cfg=j2k_cuda_oxide_j2k_decode_store_built");
    }

    if env::var_os("CARGO_FEATURE_CUDA_OXIDE_J2K_CLASSIC_DECODE").is_some()
        && compile_cuda_oxide_j2k_classic_decode(context, require_all_cuda_oxide)
    {
        println!("cargo:rustc-cfg=j2k_cuda_oxide_j2k_classic_decode_built");
    }

    if env::var_os("CARGO_FEATURE_CUDA_OXIDE_J2K_DEQUANTIZE").is_some()
        && compile_cuda_oxide_j2k_dequantize(context, require_all_cuda_oxide)
    {
        println!("cargo:rustc-cfg=j2k_cuda_oxide_j2k_dequantize_built");
    }

    if env::var_os("CARGO_FEATURE_CUDA_OXIDE_J2K_IDWT").is_some()
        && compile_cuda_oxide_j2k_idwt(context, require_all_cuda_oxide)
    {
        println!("cargo:rustc-cfg=j2k_cuda_oxide_j2k_idwt_built");
    }

    if env::var_os("CARGO_FEATURE_CUDA_OXIDE_J2K_ML").is_some()
        && compile_cuda_oxide_j2k_ml(context, require_all_cuda_oxide)
    {
        println!("cargo:rustc-cfg=j2k_cuda_oxide_j2k_ml_built");
    }

    if env::var_os("CARGO_FEATURE_CUDA_OXIDE_HTJ2K_DECODE").is_some()
        && compile_cuda_oxide_htj2k_decode(context, require_all_cuda_oxide)
    {
        println!("cargo:rustc-cfg=j2k_cuda_oxide_htj2k_decode_built");
    }

    if env::var_os("CARGO_FEATURE_CUDA_OXIDE_HTJ2K_ENCODE").is_some()
        && compile_cuda_oxide_htj2k_encode(context, require_all_cuda_oxide)
    {
        println!("cargo:rustc-cfg=j2k_cuda_oxide_htj2k_encode_built");
    }

    if env::var_os("CARGO_FEATURE_CUDA_OXIDE_TRANSCODE").is_some()
        && compile_cuda_oxide_transcode(context, require_all_cuda_oxide)
    {
        println!("cargo:rustc-cfg=j2k_cuda_oxide_transcode_built");
    }

    if env::var_os("CARGO_FEATURE_CUDA_OXIDE_JPEG_DECODE").is_some()
        && compile_cuda_oxide_jpeg_decode(context, require_all_cuda_oxide)
    {
        println!("cargo:rustc-cfg=j2k_cuda_oxide_jpeg_decode_built");
    }

    if env::var_os("CARGO_FEATURE_CUDA_OXIDE_JPEG_ENCODE").is_some()
        && compile_cuda_oxide_jpeg_encode(context, require_all_cuda_oxide)
    {
        println!("cargo:rustc-cfg=j2k_cuda_oxide_jpeg_encode_built");
    }
}

fn compile_cuda_oxide_copy_u8(context: &BuildContext<'_>, require_cuda_oxide: bool) -> bool {
    compile_cuda_oxide_project(
        context,
        CudaOxideProject {
            source_dir: Path::new("src/cuda_oxide_copy_u8"),
            output_name: "cuda_oxide_copy_u8.ptx",
            artifact_name: "j2k_cuda_oxide_copy_u8.ptx",
            display_name: "cuda-oxide CopyU8",
        },
        require_cuda_oxide,
    )
}

fn compile_cuda_oxide_j2k_encode(context: &BuildContext<'_>, require_cuda_oxide: bool) -> bool {
    compile_cuda_oxide_project(
        context,
        CudaOxideProject {
            source_dir: Path::new("src/cuda_oxide_j2k_encode"),
            output_name: "cuda_oxide_j2k_encode.ptx",
            artifact_name: "j2k_cuda_oxide_j2k_encode.ptx",
            display_name: "cuda-oxide J2K encode",
        },
        require_cuda_oxide,
    )
}

fn compile_cuda_oxide_j2k_decode_store(
    context: &BuildContext<'_>,
    require_cuda_oxide: bool,
) -> bool {
    compile_cuda_oxide_project(
        context,
        CudaOxideProject {
            source_dir: Path::new("src/cuda_oxide_j2k_decode_store"),
            output_name: "cuda_oxide_j2k_decode_store.ptx",
            artifact_name: "j2k_cuda_oxide_j2k_decode_store.ptx",
            display_name: "cuda-oxide J2K decode store",
        },
        require_cuda_oxide,
    )
}

fn compile_cuda_oxide_j2k_dequantize(context: &BuildContext<'_>, require_cuda_oxide: bool) -> bool {
    compile_cuda_oxide_project(
        context,
        CudaOxideProject {
            source_dir: Path::new("src/cuda_oxide_j2k_dequantize"),
            output_name: "cuda_oxide_j2k_dequantize.ptx",
            artifact_name: "j2k_cuda_oxide_j2k_dequantize.ptx",
            display_name: "cuda-oxide J2K dequantize",
        },
        require_cuda_oxide,
    )
}

fn compile_cuda_oxide_j2k_idwt(context: &BuildContext<'_>, require_cuda_oxide: bool) -> bool {
    compile_cuda_oxide_project(
        context,
        CudaOxideProject {
            source_dir: Path::new("src/cuda_oxide_j2k_idwt"),
            output_name: "cuda_oxide_j2k_idwt.ptx",
            artifact_name: "j2k_cuda_oxide_j2k_idwt.ptx",
            display_name: "cuda-oxide J2K IDWT",
        },
        require_cuda_oxide,
    )
}

fn compile_cuda_oxide_j2k_classic_decode(
    context: &BuildContext<'_>,
    require_cuda_oxide: bool,
) -> bool {
    compile_cuda_oxide_project(
        context,
        CudaOxideProject {
            source_dir: Path::new("src/cuda_oxide_j2k_classic_decode"),
            output_name: "cuda_oxide_j2k_classic_decode.ptx",
            artifact_name: "j2k_cuda_oxide_j2k_classic_decode.ptx",
            display_name: "cuda-oxide classic J2K decode",
        },
        require_cuda_oxide,
    )
}

fn compile_cuda_oxide_j2k_ml(context: &BuildContext<'_>, require_cuda_oxide: bool) -> bool {
    compile_cuda_oxide_project(
        context,
        CudaOxideProject {
            source_dir: Path::new("src/cuda_oxide_j2k_ml"),
            output_name: "cuda_oxide_j2k_ml.ptx",
            artifact_name: "j2k_cuda_oxide_j2k_ml.ptx",
            display_name: "cuda-oxide j2k-ml",
        },
        require_cuda_oxide,
    )
}

fn compile_cuda_oxide_htj2k_decode(context: &BuildContext<'_>, require_cuda_oxide: bool) -> bool {
    compile_cuda_oxide_project(
        context,
        CudaOxideProject {
            source_dir: Path::new("src/cuda_oxide_htj2k_decode"),
            output_name: "cuda_oxide_htj2k_decode.ptx",
            artifact_name: "j2k_cuda_oxide_htj2k_decode.ptx",
            display_name: "cuda-oxide HTJ2K decode",
        },
        require_cuda_oxide,
    )
}

fn compile_cuda_oxide_htj2k_encode(context: &BuildContext<'_>, require_cuda_oxide: bool) -> bool {
    compile_cuda_oxide_project(
        context,
        CudaOxideProject {
            source_dir: Path::new("src/cuda_oxide_htj2k_encode"),
            output_name: "cuda_oxide_htj2k_encode.ptx",
            artifact_name: "j2k_cuda_oxide_htj2k_encode.ptx",
            display_name: "cuda-oxide HTJ2K encode",
        },
        require_cuda_oxide,
    )
}

fn compile_cuda_oxide_transcode(context: &BuildContext<'_>, require_cuda_oxide: bool) -> bool {
    compile_cuda_oxide_project(
        context,
        CudaOxideProject {
            source_dir: Path::new("src/cuda_oxide_transcode"),
            output_name: "cuda_oxide_transcode.ptx",
            artifact_name: "j2k_cuda_oxide_transcode.ptx",
            display_name: "cuda-oxide transcode",
        },
        require_cuda_oxide,
    )
}

fn compile_cuda_oxide_jpeg_decode(context: &BuildContext<'_>, require_cuda_oxide: bool) -> bool {
    compile_cuda_oxide_project(
        context,
        CudaOxideProject {
            source_dir: Path::new("src/cuda_oxide_jpeg_decode"),
            output_name: "cuda_oxide_jpeg_decode.ptx",
            artifact_name: "j2k_cuda_oxide_jpeg_decode.ptx",
            display_name: "cuda-oxide JPEG decode",
        },
        require_cuda_oxide,
    )
}

fn compile_cuda_oxide_jpeg_encode(context: &BuildContext<'_>, require_cuda_oxide: bool) -> bool {
    compile_cuda_oxide_project(
        context,
        CudaOxideProject {
            source_dir: Path::new("src/cuda_oxide_jpeg_encode"),
            output_name: "cuda_oxide_jpeg_encode.ptx",
            artifact_name: "j2k_cuda_oxide_jpeg_encode.ptx",
            display_name: "cuda-oxide JPEG encode",
        },
        require_cuda_oxide,
    )
}

#[derive(Clone, Copy)]
struct CudaOxideProject<'a> {
    source_dir: &'a Path,
    output_name: &'a str,
    artifact_name: &'a str,
    display_name: &'a str,
}

fn compile_cuda_oxide_project(
    context: &BuildContext<'_>,
    project: CudaOxideProject<'_>,
    require_cuda_oxide: bool,
) -> bool {
    let output = context.out_dir.join(project.output_name);
    if !context.host.contains("linux") {
        return skip_cuda_oxide_project(
            &output,
            require_cuda_oxide,
            project.display_name,
            &format!(
                "{} requires a Linux host; current HOST={}",
                project.display_name, context.host
            ),
        );
    }

    let project_dir = context
        .out_dir
        .join(project.output_name.trim_end_matches(".ptx"));
    copy_cuda_oxide_project(
        project.source_dir,
        &project_dir,
        context.codec_math_crate_path,
    );

    let arch = env::var("J2K_CUDA_OXIDE_ARCH").unwrap_or_else(|_| "sm_80".to_string());
    println!(
        "cargo:warning=building {} with `cargo oxide build --arch {arch}`",
        project.display_name
    );
    // Use the rustup cargo proxy so the staged rust-toolchain.toml selects
    // cuda-oxide's pinned nightly instead of the outer workspace toolchain.
    let status = Command::new("cargo")
        .args(["oxide", "build", "--arch"])
        .arg(&arch)
        .env_remove("RUSTUP_TOOLCHAIN")
        .env_remove("RUSTC")
        .env_remove("RUSTC_WRAPPER")
        .env_remove("RUSTC_WORKSPACE_WRAPPER")
        .env_remove("RUSTDOC")
        .env_remove("RUSTFLAGS")
        .env_remove("CARGO_ENCODED_RUSTFLAGS")
        .current_dir(&project_dir)
        .status();
    let status = match status {
        Ok(status) => status,
        Err(error) => {
            return skip_cuda_oxide_project(
                &output,
                require_cuda_oxide,
                project.display_name,
                &format!("failed to invoke cargo oxide build: {error}"),
            );
        }
    };
    if !status.success() {
        return skip_cuda_oxide_project(
            &output,
            require_cuda_oxide,
            project.display_name,
            &format!("{} build failed with status {status}", project.display_name),
        );
    }

    let generated = project_dir.join("ptx").join(project.artifact_name);
    let mut bytes = fs::read(&generated).unwrap_or_else(|error| {
        panic!(
            "{} build did not produce {}: {error}",
            project.display_name,
            generated.display()
        )
    });
    if bytes.last().copied() != Some(0) {
        bytes.push(0);
    }
    fs::write(&output, bytes).unwrap_or_else(|error| {
        panic!(
            "failed to write {} PTX to {}: {error}",
            project.display_name,
            output.display()
        )
    });
    true
}

fn skip_cuda_oxide_project(
    output: &Path,
    required: bool,
    display_name: &str,
    message: &str,
) -> bool {
    assert!(!required, "{message}");
    println!("cargo:warning=skipping {display_name} build: {message}");
    fs::write(output, b".version 7.0\n.target sm_52\n.address_size 64\n\0")
        .unwrap_or_else(|error| panic!("write placeholder {display_name} PTX: {error}"));
    false
}

fn copy_cuda_oxide_project(source_dir: &Path, project_dir: &Path, codec_math_crate_path: &Path) {
    copy_cuda_oxide_file_as(
        source_dir,
        project_dir,
        Path::new("Cargo.toml.in"),
        Path::new("Cargo.toml"),
        codec_math_crate_path,
    );
    copy_cuda_oxide_file(
        source_dir,
        project_dir,
        Path::new("rust-toolchain.toml"),
        codec_math_crate_path,
    );
    copy_cuda_oxide_file(
        source_dir,
        project_dir,
        Path::new("src/main.rs"),
        codec_math_crate_path,
    );
    copy_cuda_oxide_file_as(
        source_dir,
        project_dir,
        Path::new("simt/Cargo.toml.in"),
        Path::new("simt/Cargo.toml"),
        codec_math_crate_path,
    );
    copy_cuda_oxide_file(
        source_dir,
        project_dir,
        Path::new("simt/src/main.rs"),
        codec_math_crate_path,
    );
    for relative in cuda_oxide_extra_sources(source_dir) {
        copy_cuda_oxide_file(
            source_dir,
            project_dir,
            Path::new(relative),
            codec_math_crate_path,
        );
    }
}

fn cuda_oxide_extra_sources(source_dir: &Path) -> &'static [&'static str] {
    if source_dir == Path::new("src/cuda_oxide_j2k_encode") {
        CUDA_OXIDE_J2K_ENCODE_EXTRA_SOURCES
    } else if source_dir == Path::new("src/cuda_oxide_j2k_decode_store") {
        CUDA_OXIDE_J2K_DECODE_STORE_EXTRA_SOURCES
    } else if source_dir == Path::new("src/cuda_oxide_transcode") {
        CUDA_OXIDE_TRANSCODE_EXTRA_SOURCES
    } else if source_dir == Path::new("src/cuda_oxide_jpeg_decode") {
        CUDA_OXIDE_JPEG_DECODE_EXTRA_SOURCES
    } else {
        &[]
    }
}

fn stage_cuda_oxide_shared_prelude(out_dir: &Path) {
    let source = Path::new("src/cuda_oxide_simt_prelude.rs");
    let dest = out_dir.join("cuda_oxide_simt_prelude.rs");
    fs::copy(source, &dest).unwrap_or_else(|error| {
        panic!(
            "failed to stage CUDA Oxide SIMT prelude {} to {}: {error}",
            source.display(),
            dest.display()
        )
    });
}

fn copy_cuda_oxide_file(
    source_dir: &Path,
    project_dir: &Path,
    relative: &Path,
    codec_math_crate_path: &Path,
) {
    copy_cuda_oxide_file_as(
        source_dir,
        project_dir,
        relative,
        relative,
        codec_math_crate_path,
    );
}

fn copy_cuda_oxide_file_as(
    source_dir: &Path,
    project_dir: &Path,
    source_relative: &Path,
    dest_relative: &Path,
    codec_math_crate_path: &Path,
) {
    let source = source_dir.join(source_relative);
    let dest = project_dir.join(dest_relative);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|error| {
            panic!(
                "failed to create cuda-oxide project dir {}: {error}",
                parent.display()
            )
        });
    }
    if source_relative
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension == "in")
    {
        let source_text = fs::read_to_string(&source).unwrap_or_else(|error| {
            panic!(
                "failed to read cuda-oxide project template {}: {error}",
                source.display()
            )
        });
        let rendered = source_text.replace(
            "__J2K_CODEC_MATH_PATH__",
            &codec_math_crate_path.to_string_lossy(),
        );
        fs::write(&dest, rendered).unwrap_or_else(|error| {
            panic!(
                "failed to render cuda-oxide project template {} to {}: {error}",
                source.display(),
                dest.display()
            )
        });
    } else {
        fs::copy(&source, &dest).unwrap_or_else(|error| {
            panic!(
                "failed to stage cuda-oxide project file {} to {}: {error}",
                source.display(),
                dest.display()
            )
        });
    }
}
