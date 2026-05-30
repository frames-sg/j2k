use std::env;
use std::path::PathBuf;
use std::process::Command;

// Compiles the C++ nvJPEG/nvJPEG2000 baseline (`cuda/nv_baseline.cu`) into a
// static library and links it plus the NVIDIA codec libraries, but ONLY when the
// `nvjpeg2000` feature is enabled AND nvcc is available. On any other host this
// is a no-op so the workspace still builds (the Rust side is cfg-gated on
// `nvbaseline_built`).
//
// Library locations can be overridden for non-standard installs:
//   CUDA_LIB_DIR        (default: /usr/local/cuda/targets/x86_64-linux/lib)
//   NVJPEG2K_LIB_DIR    (default: CUDA_LIB_DIR)
//   NVJPEG2K_INCLUDE_DIR (passed to nvcc as -I if set)
// Set SIGNINUM_REQUIRE_NV_BASELINE_BUILD=1 to make an nvcc failure fatal.
fn main() {
    println!("cargo:rerun-if-changed=cuda/nv_baseline.cu");
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_LIB_DIR");
    println!("cargo:rerun-if-env-changed=NVJPEG2K_LIB_DIR");
    println!("cargo:rerun-if-env-changed=NVJPEG2K_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=SIGNINUM_REQUIRE_NV_BASELINE_BUILD");
    println!("cargo:rustc-check-cfg=cfg(nvbaseline_built)");

    if env::var_os("CARGO_FEATURE_NVJPEG2000").is_none() {
        return;
    }

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    let nvcc = env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());
    let strict = env::var_os("SIGNINUM_REQUIRE_NV_BASELINE_BUILD").is_some();

    let object = out_dir.join("nv_baseline.o");
    let archive = out_dir.join("libnvbaseline.a");

    let mut compile = Command::new(&nvcc);
    compile
        .args(["-c", "-O3", "--std=c++14", "-Xcompiler", "-fPIC"])
        .arg("cuda/nv_baseline.cu")
        .arg("-o")
        .arg(&object);
    if let Some(include_dir) = env::var_os("NVJPEG2K_INCLUDE_DIR") {
        compile.arg("-I").arg(include_dir);
    }

    let compiled = compile.status().is_ok_and(|status| status.success());
    if !compiled {
        assert!(
            !strict,
            "SIGNINUM_REQUIRE_NV_BASELINE_BUILD set, but nvcc failed to compile cuda/nv_baseline.cu"
        );
        // No nvcc / NVIDIA headers: leave `nvbaseline_built` unset; the binary
        // prints rebuild instructions instead of linking against absent libs.
        return;
    }

    let archived = Command::new("ar")
        .arg("rcs")
        .arg(&archive)
        .arg(&object)
        .status()
        .is_ok_and(|status| status.success());
    assert!(archived, "ar failed to archive nv_baseline.o");

    let cuda_lib_dir = env::var("CUDA_LIB_DIR")
        .unwrap_or_else(|_| "/usr/local/cuda/targets/x86_64-linux/lib".to_string());
    let nvjpeg2k_lib_dir = env::var("NVJPEG2K_LIB_DIR").unwrap_or_else(|_| cuda_lib_dir.clone());

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-search=native={cuda_lib_dir}");
    println!("cargo:rustc-link-search=native={nvjpeg2k_lib_dir}");
    println!("cargo:rustc-link-lib=static=nvbaseline");
    println!("cargo:rustc-link-lib=dylib=nvjpeg2k");
    println!("cargo:rustc-link-lib=dylib=nvjpeg");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-cfg=nvbaseline_built");
}
