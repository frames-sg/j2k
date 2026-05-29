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
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=SIGNINUM_REQUIRE_CUDA_HTJ2K_STRICT");
    println!("cargo:rerun-if-env-changed=SIGNINUM_REQUIRE_CUDA_KERNEL_BUILD");

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let nvcc = env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());
    let require_kernel_build = env::var_os("SIGNINUM_REQUIRE_CUDA_HTJ2K_STRICT").is_some()
        || env::var_os("SIGNINUM_REQUIRE_CUDA_KERNEL_BUILD").is_some();

    let j2k_encode_ptx = out_dir.join("j2k_encode_kernels.ptx");
    compile_or_copy_ptx(
        &nvcc,
        Path::new("src/j2k_encode_kernels.cu"),
        Path::new("src/j2k_encode_kernels.ptx"),
        &j2k_encode_ptx,
        require_kernel_build,
    );
    let htj2k_decode_ptx = out_dir.join("htj2k_decode_kernels.ptx");
    compile_or_copy_ptx(
        &nvcc,
        Path::new("src/htj2k_decode_kernels.cu"),
        Path::new("src/htj2k_decode_kernels.ptx"),
        &htj2k_decode_ptx,
        require_kernel_build,
    );
    let htj2k_encode_ptx = out_dir.join("htj2k_encode_kernels.ptx");
    compile_or_copy_ptx(
        &nvcc,
        Path::new("src/htj2k_encode_kernels.cu"),
        Path::new("src/htj2k_encode_kernels.ptx"),
        &htj2k_encode_ptx,
        require_kernel_build,
    );
}

fn compile_or_copy_ptx(
    nvcc: &std::ffi::OsStr,
    source: &Path,
    fallback: &Path,
    ptx: &Path,
    require_kernel_build: bool,
) {
    let compiled = Command::new(nvcc)
        .args(["--ptx", "-O3", "--std=c++14"])
        .arg(source)
        .arg("-o")
        .arg(ptx)
        .status()
        .is_ok_and(|status| status.success());

    if compiled {
        let mut bytes = fs::read(ptx).expect("read generated CUDA PTX");
        if bytes.last().copied() != Some(0) {
            bytes.push(0);
            fs::write(ptx, bytes).expect("NUL-terminate generated CUDA PTX");
        }
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
    }
}
