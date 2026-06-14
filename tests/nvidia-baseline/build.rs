use std::env;
use std::ffi::OsStr;
use std::fs;
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
    let strict = env::var_os("SIGNINUM_REQUIRE_NV_BASELINE_BUILD").is_some();
    let nvcc = configured_nvcc(strict);

    let object = out_dir.join("nv_baseline.o");
    let archive = out_dir.join("libnvbaseline.a");

    let include_dir = env::var_os("NVJPEG2K_INCLUDE_DIR");
    let stream_parse_define = match probe_nvjpeg2k_decode_api(
        nvcc.as_deref(),
        include_dir.as_deref(),
        &out_dir,
    ) {
        Ok(define) => define,
        Err(message) => {
            assert!(
                !strict,
                "SIGNINUM_REQUIRE_NV_BASELINE_BUILD set, but nvJPEG2000 decode APIs are unavailable: {message}"
            );
            return;
        }
    };

    let Some(nvcc) = nvcc.as_deref() else {
        return;
    };

    let mut compile = Command::new(nvcc);
    compile
        .args(["-c", "-O3", "--std=c++14", "-Xcompiler", "-fPIC"])
        .arg("cuda/nv_baseline.cu")
        .arg("-o")
        .arg(&object);
    if let Some(define) = stream_parse_define {
        compile.arg(define);
    }
    if let Some(include_dir) = include_dir {
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

fn probe_nvjpeg2k_decode_api(
    nvcc: Option<&OsStr>,
    include_dir: Option<&OsStr>,
    out_dir: &std::path::Path,
) -> Result<Option<&'static str>, String> {
    let Some(nvcc) = nvcc else {
        return Err("NVCC not configured".to_string());
    };
    let current = r"
#include <cuda_runtime.h>
#include <nvjpeg2k.h>
int main() {
    nvjpeg2kHandle_t handle = nullptr;
    nvjpeg2kDecodeState_t decode_state = nullptr;
    nvjpeg2kDecodeParams_t decode_params = nullptr;
    nvjpeg2kStream_t stream = nullptr;
    unsigned char data[4] = {0, 0, 0, 0};
    nvjpeg2kImage_t image;
    nvjpeg2kCreateSimple(&handle);
    nvjpeg2kDecodeStateCreate(handle, &decode_state);
    nvjpeg2kDecodeParamsCreate(&decode_params);
    nvjpeg2kStreamCreate(&stream);
    nvjpeg2kDecodeParamsSetOutputFormat(decode_params, NVJPEG2K_FORMAT_INTERLEAVED);
    nvjpeg2kDecodeParamsSetRGBOutput(decode_params, 1);
    nvjpeg2kStreamParse(handle, data, sizeof(data), 0, 0, stream);
    nvjpeg2kDecodeImage(handle, decode_state, stream, decode_params, &image, 0);
    return 0;
}
";
    if compile_probe(
        nvcc,
        include_dir,
        out_dir,
        "nvjpeg2k_decode_current.cu",
        current,
    ) {
        return Ok(None);
    }

    let legacy = current.replace(
        "nvjpeg2kStreamParse(handle, data, sizeof(data), 0, 0, stream);",
        "nvjpeg2kStreamParse(handle, data, sizeof(data), 0, 0, &stream);",
    );
    if compile_probe(
        nvcc,
        include_dir,
        out_dir,
        "nvjpeg2k_decode_legacy.cu",
        &legacy,
    ) {
        return Ok(Some("-DNVB_STREAM_PARSE_USES_OUT_POINTER=1"));
    }

    Err("neither current nor legacy nvjpeg2kStreamParse decode probes compiled".to_string())
}

fn compile_probe(
    nvcc: &OsStr,
    include_dir: Option<&OsStr>,
    out_dir: &std::path::Path,
    name: &str,
    source: &str,
) -> bool {
    let source_path = out_dir.join(name);
    let object_path = out_dir.join(format!("{name}.o"));
    if fs::write(&source_path, source).is_err() {
        return false;
    }
    let mut command = Command::new(nvcc);
    command
        .args(["-c", "--std=c++14"])
        .arg(&source_path)
        .arg("-o")
        .arg(&object_path);
    if let Some(include_dir) = include_dir {
        command.arg("-I").arg(include_dir);
    }
    command.status().is_ok_and(|status| status.success())
}

fn configured_nvcc(strict: bool) -> Option<std::ffi::OsString> {
    let nvcc = env::var_os("NVCC");
    if strict {
        let nvcc = nvcc.expect("strict NVIDIA baseline build requires absolute NVCC");
        assert!(
            std::path::Path::new(&nvcc).is_absolute(),
            "strict NVIDIA baseline build requires absolute NVCC, got {}",
            std::path::Path::new(&nvcc).display()
        );
        Some(nvcc)
    } else {
        nvcc
    }
}
