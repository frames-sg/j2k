// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::super::super::{assert_pattern_checks, repo_root, rust_sources, PatternCheck};

#[test]
fn cuda_gpu_abi_byte_views_require_compile_time_no_padding_proofs() {
    let root = repo_root();
    let core = fs::read_to_string(root.join("crates/j2k-core/src/accelerator.rs"))
        .expect("read shared GPU ABI contract");
    let bytes = fs::read_to_string(root.join("crates/j2k-cuda-runtime/src/bytes.rs"))
        .expect("read CUDA byte-view facade");
    let abi = fs::read_to_string(root.join("crates/j2k-cuda-runtime/src/bytes/abi.rs"))
        .expect("read CUDA GPU ABI proofs");
    let abi_tests = fs::read_to_string(root.join("crates/j2k-cuda-runtime/src/bytes/abi/tests.rs"))
        .expect("read CUDA GPU ABI proof tests");
    let reversible53 =
        fs::read_to_string(root.join("crates/j2k-cuda-runtime/src/transcode/reversible53.rs"))
            .expect("read CUDA reversible transcode upload");

    assert!(
        abi.lines().count() < 500,
        "CUDA GPU ABI proof ledger must remain focused"
    );
    assert!(
        abi_tests.lines().count() < 100,
        "CUDA GPU ABI proof tests must remain focused"
    );
    assert_pattern_checks(&[
        PatternCheck::new("shared safe GPU byte-view contract", &core).required(&[
            "object representation contains no internal or tail padding",
            "compile-time field-offset/end proof",
            "size-only tests and comments are insufficient",
            "including explicit reserved fields",
        ]),
        PatternCheck::new("CUDA byte-view facade", &bytes)
            .required(&["mod abi;"])
            .forbidden(&["unsafe impl GpuAbi", "impl_cuda_gpu_abi!"]),
        PatternCheck::new("CUDA compile-time no-padding proof macro", &abi).required(&[
            "macro_rules! prove_cuda_gpu_abi_layout",
            "macro_rules! impl_cuda_gpu_abi",
            "fn assert_field_types(value: &$ty)",
            "let _: [(); size_of::<$ty>()] = [(); $offset];",
            "let _: [(); offset_of!($ty, $field)] = [(); $offset];",
            "$offset + size_of::<$field_ty>();",
            "prove_cuda_gpu_abi_layout!(",
            "unsafe impl GpuAbi for $ty",
            "CudaJpegEntropyCheckpoint {",
            "CudaHtj2kCleanupMultiKernelJob {",
            "CudaHtj2kDequantizeKernelJob {",
            "CudaJ2kIdwtMultiKernelJob {",
            "CudaJ2kStoreRgb8MctBatchJob {",
            "reserved_tail: u32",
            "mod tests;",
        ]),
        PatternCheck::new("CUDA no-padding proof tests", &abi_tests).required(&[
            "explicit_tail_fields_preserve_cuda_host_abi_sizes_and_offsets",
            "explicit_cuda_tail_fields_are_part_of_safe_byte_views",
        ]),
        PatternCheck::new("CUDA primitive slice byte views", &reversible53)
            .required(&["i16_slice_as_bytes(dequantized_blocks)"])
            .forbidden(&["std::slice::from_raw_parts("]),
    ]);

    let mut direct_impl_paths = Vec::new();
    for path in rust_sources(&root.join("crates/j2k-cuda-runtime/src")) {
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        if source.contains("unsafe impl GpuAbi") {
            direct_impl_paths.push(
                path.strip_prefix(root)
                    .expect("CUDA source under repository root")
                    .to_path_buf(),
            );
        }
    }
    assert_eq!(
        direct_impl_paths,
        [std::path::PathBuf::from(
            "crates/j2k-cuda-runtime/src/bytes/abi.rs"
        )],
        "CUDA GpuAbi implementations must use the compile-time proof macro"
    );
}

#[test]
fn cuda_host_device_mirrors_explicitly_occupy_existing_tail_padding() {
    let root = repo_root();
    let read = |relative: &str| {
        fs::read_to_string(root.join(relative))
            .unwrap_or_else(|error| panic!("read {relative}: {error}"))
    };
    let host_jpeg = read("crates/j2k-cuda-runtime/src/jpeg/types.rs");
    let device_jpeg = read("crates/j2k-cuda-runtime/src/cuda_oxide_jpeg_decode/simt/src/main.rs");
    let host_ht = read("crates/j2k-cuda-runtime/src/htj2k_decode/types.rs");
    let device_ht = read("crates/j2k-cuda-runtime/src/cuda_oxide_htj2k_decode/simt/src/main.rs");
    let device_dequant =
        read("crates/j2k-cuda-runtime/src/cuda_oxide_j2k_dequantize/simt/src/main.rs");
    let host_j2k = read("crates/j2k-cuda-runtime/src/j2k_decode/types.rs");
    let device_idwt = read("crates/j2k-cuda-runtime/src/cuda_oxide_j2k_idwt/simt/src/main.rs");
    let device_store =
        read("crates/j2k-cuda-runtime/src/cuda_oxide_j2k_decode_store/simt/src/abi.rs");

    for (name, source, minimum_tail_fields) in [
        ("CUDA JPEG host ABI", host_jpeg.as_str(), 1usize),
        ("CUDA JPEG device ABI", device_jpeg.as_str(), 1),
        ("CUDA HT host ABI", host_ht.as_str(), 2),
        ("CUDA HT decode device ABI", device_ht.as_str(), 1),
        ("CUDA dequantize device ABI", device_dequant.as_str(), 2),
        ("CUDA J2K host ABI", host_j2k.as_str(), 2),
        ("CUDA IDWT device ABI", device_idwt.as_str(), 1),
        ("CUDA store device ABI", device_store.as_str(), 1),
    ] {
        assert!(
            source.matches("reserved_tail: u32").count() >= minimum_tail_fields,
            "{name} must explicitly occupy every audited tail-padding slot"
        );
    }
}
