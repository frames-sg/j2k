use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/j2k_encode_kernels.cu");
    println!("cargo:rerun-if-changed=src/j2k_encode_kernels.ptx");
    println!("cargo:rerun-if-changed=src/htj2k_decode_kernels.cu");
    println!("cargo:rerun-if-changed=src/htj2k_decode_kernels.ptx");
    println!("cargo:rerun-if-changed=src/htj2k_encode_kernels.cu");
    println!("cargo:rerun-if-changed=src/htj2k_encode_kernels.ptx");
    println!("cargo:rerun-if-changed=src/jpeg_decode_kernels.cu");
    println!("cargo:rerun-if-changed=src/transcode_kernels.cu");
    println!("cargo:rerun-if-changed=src/cuda_oxide_copy_u8/Cargo.toml");
    println!("cargo:rerun-if-changed=src/cuda_oxide_copy_u8/rust-toolchain.toml");
    println!("cargo:rerun-if-changed=src/cuda_oxide_copy_u8/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_copy_u8/simt/Cargo.toml");
    println!("cargo:rerun-if-changed=src/cuda_oxide_copy_u8/simt/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_encode/Cargo.toml");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_encode/rust-toolchain.toml");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_encode/src/main.rs");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_encode/simt/Cargo.toml");
    println!("cargo:rerun-if-changed=src/cuda_oxide_j2k_encode/simt/src/main.rs");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=J2K_CUDA_OXIDE_ARCH");
    println!("cargo:rerun-if-env-changed=J2K_REQUIRE_CUDA_OXIDE_COPY_U8");
    println!("cargo:rerun-if-env-changed=J2K_REQUIRE_CUDA_OXIDE_J2K_ENCODE");
    println!("cargo:rerun-if-env-changed=J2K_REQUIRE_CUDA_HTJ2K_STRICT");
    println!("cargo:rerun-if-env-changed=J2K_REQUIRE_CUDA_KERNEL_BUILD");
    println!("cargo:rustc-check-cfg=cfg( j2k_cuda_j2k_encode_ptx_built)");
    println!("cargo:rustc-check-cfg=cfg( j2k_cuda_htj2k_encode_ptx_built)");
    println!("cargo:rustc-check-cfg=cfg( j2k_cuda_jpeg_decode_ptx_built)");
    println!("cargo:rustc-check-cfg=cfg( j2k_cuda_transcode_ptx_built)");
    println!("cargo:rustc-check-cfg=cfg(j2k_cuda_oxide_copy_u8_built)");
    println!("cargo:rustc-check-cfg=cfg(j2k_cuda_oxide_j2k_encode_built)");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let require_kernel_build = env::var_os("J2K_REQUIRE_CUDA_HTJ2K_STRICT").is_some()
        || env::var_os("J2K_REQUIRE_CUDA_KERNEL_BUILD").is_some();
    let nvcc = configured_nvcc(require_kernel_build);

    let j2k_encode_ptx = out_dir.join("j2k_encode_kernels.ptx");
    let j2k_encode_compiled = compile_or_copy_ptx(
        nvcc.as_deref(),
        Path::new("src/j2k_encode_kernels.cu"),
        Path::new("src/j2k_encode_kernels.ptx"),
        &j2k_encode_ptx,
        require_kernel_build,
    );
    if j2k_encode_compiled {
        println!("cargo:rustc-cfg= j2k_cuda_j2k_encode_ptx_built");
    }
    let htj2k_decode_ptx = out_dir.join("htj2k_decode_kernels.ptx");
    compile_or_copy_ptx(
        nvcc.as_deref(),
        Path::new("src/htj2k_decode_kernels.cu"),
        Path::new("src/htj2k_decode_kernels.ptx"),
        &htj2k_decode_ptx,
        require_kernel_build,
    );
    let htj2k_encode_ptx = out_dir.join("htj2k_encode_kernels.ptx");
    let htj2k_encode_compiled = compile_or_copy_ptx(
        nvcc.as_deref(),
        Path::new("src/htj2k_encode_kernels.cu"),
        Path::new("src/htj2k_encode_kernels.ptx"),
        &htj2k_encode_ptx,
        require_kernel_build,
    );
    if htj2k_encode_compiled {
        println!("cargo:rustc-cfg= j2k_cuda_htj2k_encode_ptx_built");
    }

    let jpeg_decode_ptx = out_dir.join("jpeg_decode_kernels.ptx");
    let jpeg_decode_compiled = compile_optional_ptx(
        nvcc.as_deref(),
        Path::new("src/jpeg_decode_kernels.cu"),
        &jpeg_decode_ptx,
        require_kernel_build,
    );
    if jpeg_decode_compiled {
        println!("cargo:rustc-cfg= j2k_cuda_jpeg_decode_ptx_built");
    }

    // Transcode (DCT->wavelet) kernels are new: there is no checked-in PTX
    // fallback, so this is an OPTIONAL compile. On nvcc success it sets a cfg
    // that gates the Rust dispatch; on a non-nvcc host it is skipped (the
    // dispatch is cfg'd out). The runner sets the strict env, which requires
    // nvcc to succeed.
    let transcode_ptx = out_dir.join("transcode_kernels.ptx");
    let transcode_compiled = compile_optional_ptx(
        nvcc.as_deref(),
        Path::new("src/transcode_kernels.cu"),
        &transcode_ptx,
        require_kernel_build,
    );
    if transcode_compiled {
        println!("cargo:rustc-cfg= j2k_cuda_transcode_ptx_built");
    }

    if env::var_os("CARGO_FEATURE_CUDA_OXIDE_COPY_U8").is_some() {
        let require_cuda_oxide = env::var_os("J2K_REQUIRE_CUDA_OXIDE_COPY_U8").is_some();
        if compile_cuda_oxide_copy_u8(&out_dir, require_cuda_oxide) {
            println!("cargo:rustc-cfg=j2k_cuda_oxide_copy_u8_built");
        }
    }

    if env::var_os("CARGO_FEATURE_CUDA_OXIDE_J2K_ENCODE").is_some() {
        let require_cuda_oxide = env::var_os("J2K_REQUIRE_CUDA_OXIDE_J2K_ENCODE").is_some();
        if compile_cuda_oxide_j2k_encode(&out_dir, require_cuda_oxide) {
            println!("cargo:rustc-cfg=j2k_cuda_oxide_j2k_encode_built");
        }
    }
}

fn compile_cuda_oxide_copy_u8(out_dir: &Path, require_cuda_oxide: bool) -> bool {
    compile_cuda_oxide_project(
        out_dir,
        CudaOxideProject {
            source_dir: Path::new("src/cuda_oxide_copy_u8"),
            output_name: "cuda_oxide_copy_u8.ptx",
            artifact_name: "j2k_cuda_oxide_copy_u8.ptx",
            display_name: "cuda-oxide CopyU8",
        },
        require_cuda_oxide,
    )
}

fn compile_cuda_oxide_j2k_encode(out_dir: &Path, require_cuda_oxide: bool) -> bool {
    compile_cuda_oxide_project(
        out_dir,
        CudaOxideProject {
            source_dir: Path::new("src/cuda_oxide_j2k_encode"),
            output_name: "cuda_oxide_j2k_encode.ptx",
            artifact_name: "j2k_cuda_oxide_j2k_encode.ptx",
            display_name: "cuda-oxide J2K encode",
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
    out_dir: &Path,
    project: CudaOxideProject<'_>,
    require_cuda_oxide: bool,
) -> bool {
    let output = out_dir.join(project.output_name);
    let host = env::var("HOST").expect("HOST is set by cargo");
    if !host.contains("linux") {
        return skip_cuda_oxide_project(
            &output,
            require_cuda_oxide,
            project.display_name,
            &format!(
                "{} requires a Linux host; current HOST={host}",
                project.display_name
            ),
        );
    }

    let project_dir = out_dir.join(project.output_name.trim_end_matches(".ptx"));
    copy_cuda_oxide_project(project.source_dir, &project_dir);

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

fn copy_cuda_oxide_project(source_dir: &Path, project_dir: &Path) {
    copy_cuda_oxide_file(source_dir, project_dir, Path::new("Cargo.toml"));
    copy_cuda_oxide_file(source_dir, project_dir, Path::new("rust-toolchain.toml"));
    copy_cuda_oxide_file(source_dir, project_dir, Path::new("src/main.rs"));
    copy_cuda_oxide_file(source_dir, project_dir, Path::new("simt/Cargo.toml"));
    copy_cuda_oxide_file(source_dir, project_dir, Path::new("simt/src/main.rs"));
}

fn copy_cuda_oxide_file(source_dir: &Path, project_dir: &Path, relative: &Path) {
    let source = source_dir.join(relative);
    let dest = project_dir.join(relative);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|error| {
            panic!(
                "failed to create cuda-oxide project dir {}: {error}",
                parent.display()
            )
        });
    }
    fs::copy(&source, &dest).unwrap_or_else(|error| {
        panic!(
            "failed to stage cuda-oxide project file {} to {}: {error}",
            source.display(),
            dest.display()
        )
    });
}

/// Compile a CUDA kernel to PTX with nvcc only (no checked-in fallback).
///
/// Returns whether nvcc produced PTX. When `require_kernel_build` is set (the
/// runner), nvcc failure is a hard error; otherwise a non-nvcc host simply
/// skips the kernel and the Rust dispatch is cfg-gated off.
fn compile_optional_ptx(
    nvcc: Option<&std::ffi::OsStr>,
    source: &Path,
    ptx: &Path,
    require_kernel_build: bool,
) -> bool {
    let compiled = nvcc.is_some_and(|nvcc| {
        Command::new(nvcc)
            .args(["--ptx", "-O3", "--std=c++14", "--fmad=false"])
            .arg(source)
            .arg("-o")
            .arg(ptx)
            .status()
            .is_ok_and(|status| status.success())
    });

    if compiled {
        let mut bytes = fs::read(ptx).expect("read generated CUDA transcode PTX");
        if bytes.last().copied() != Some(0) {
            bytes.push(0);
            fs::write(ptx, bytes).expect("NUL-terminate generated CUDA transcode PTX");
        }
        true
    } else {
        assert!(
            !require_kernel_build,
            "strict CUDA kernel build required, but nvcc failed for {}",
            source.display()
        );
        // No checked-in fallback exists for this kernel. Write a placeholder
        // empty PTX module so `include_bytes!(OUT_DIR/transcode_kernels.ptx)`
        // always resolves on non-nvcc hosts and the Rust dispatch type-checks.
        // It is never loaded at runtime: the dispatch first checks the
        // `j2k_cuda_transcode_ptx_built` cfg and returns a typed error.
        fs::write(ptx, b".version 7.0\n.target sm_52\n.address_size 64\n\0")
            .expect("write placeholder transcode PTX");
        false
    }
}

fn compile_or_copy_ptx(
    nvcc: Option<&std::ffi::OsStr>,
    source: &Path,
    fallback: &Path,
    ptx: &Path,
    require_kernel_build: bool,
) -> bool {
    // --fmad=false: native (Rust/LLVM) does not contract a*b+c into a single-rounding
    // FMA; nvcc does by default. Disabling it keeps the f32 DWT/RCT lossless path
    // bit-identical to the native reference (byte-parity requirement).
    let compiled = nvcc.is_some_and(|nvcc| {
        Command::new(nvcc)
            .args(["--ptx", "-O3", "--std=c++14", "--fmad=false"])
            .arg(source)
            .arg("-o")
            .arg(ptx)
            .status()
            .is_ok_and(|status| status.success())
    });

    if compiled {
        let mut bytes = fs::read(ptx).expect("read generated CUDA PTX");
        if bytes.last().copied() != Some(0) {
            bytes.push(0);
            fs::write(ptx, bytes).expect("NUL-terminate generated CUDA PTX");
        }
        true
    } else {
        assert!(
            !require_kernel_build,
            "strict CUDA kernel build required, but nvcc failed for {}",
            source.display()
        );
        let mut bytes = fs::read(fallback).expect("read fallback CUDA PTX");
        if bytes.last().copied() != Some(0) {
            bytes.push(0);
        }
        fs::write(ptx, bytes).expect("write fallback CUDA PTX");
        false
    }
}

fn configured_nvcc(strict: bool) -> Option<std::ffi::OsString> {
    let nvcc = env::var_os("NVCC");
    if strict {
        let nvcc = nvcc.expect("strict CUDA kernel build requires absolute NVCC");
        assert!(
            Path::new(&nvcc).is_absolute(),
            "strict CUDA kernel build requires absolute NVCC, got {}",
            Path::new(&nvcc).display()
        );
        Some(nvcc)
    } else {
        nvcc
    }
}
