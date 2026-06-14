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
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=SIGNINUM_REQUIRE_CUDA_HTJ2K_STRICT");
    println!("cargo:rerun-if-env-changed=SIGNINUM_REQUIRE_CUDA_KERNEL_BUILD");
    println!("cargo:rustc-check-cfg=cfg(signinum_cuda_j2k_encode_ptx_built)");
    println!("cargo:rustc-check-cfg=cfg(signinum_cuda_htj2k_encode_ptx_built)");
    println!("cargo:rustc-check-cfg=cfg(signinum_cuda_jpeg_decode_ptx_built)");
    println!("cargo:rustc-check-cfg=cfg(signinum_cuda_transcode_ptx_built)");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let require_kernel_build = env::var_os("SIGNINUM_REQUIRE_CUDA_HTJ2K_STRICT").is_some()
        || env::var_os("SIGNINUM_REQUIRE_CUDA_KERNEL_BUILD").is_some();
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
        println!("cargo:rustc-cfg=signinum_cuda_j2k_encode_ptx_built");
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
        println!("cargo:rustc-cfg=signinum_cuda_htj2k_encode_ptx_built");
    }

    let jpeg_decode_ptx = out_dir.join("jpeg_decode_kernels.ptx");
    let jpeg_decode_compiled = compile_optional_ptx(
        nvcc.as_deref(),
        Path::new("src/jpeg_decode_kernels.cu"),
        &jpeg_decode_ptx,
        require_kernel_build,
    );
    if jpeg_decode_compiled {
        println!("cargo:rustc-cfg=signinum_cuda_jpeg_decode_ptx_built");
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
        println!("cargo:rustc-cfg=signinum_cuda_transcode_ptx_built");
    }
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
        // `signinum_cuda_transcode_ptx_built` cfg and returns a typed error.
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
