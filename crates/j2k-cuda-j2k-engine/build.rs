use std::error::Error;

use j2k_cuda_build_support::{build_cuda_oxide_projects, CudaOxideProject};

const J2K_DECODE_STORE_EXTRA_SOURCES: &[&str] = &[
    "simt/src/abi.rs",
    "simt/src/color.rs",
    "simt/src/exports.rs",
    "simt/src/layout.rs",
    "simt/src/memory.rs",
    "simt/src/native_color.rs",
    "simt/src/sample.rs",
    "simt/src/transform.rs",
];
const HTJ2K_ENCODE_EXTRA_SOURCES: &[&str] = &["simt/src/analysis.rs"];
const J2K_ENCODE_EXTRA_SOURCES: &[&str] = &[
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

const PROJECTS: &[CudaOxideProject] = &[
    CudaOxideProject {
        feature_env: "CARGO_FEATURE_CUDA_OXIDE_HTJ2K_ENCODE",
        source_dir: "src/cuda_oxide_htj2k_encode",
        output_name: "cuda_oxide_htj2k_encode.ptx",
        artifact_name: "j2k_cuda_oxide_htj2k_encode.ptx",
        display_name: "cuda-oxide HTJ2K encode",
        extra_sources: HTJ2K_ENCODE_EXTRA_SOURCES,
        built_cfg: "j2k_cuda_oxide_htj2k_encode_built",
    },
    CudaOxideProject {
        feature_env: "CARGO_FEATURE_CUDA_OXIDE_J2K_ENCODE",
        source_dir: "src/cuda_oxide_j2k_encode",
        output_name: "cuda_oxide_j2k_encode.ptx",
        artifact_name: "j2k_cuda_oxide_j2k_encode.ptx",
        display_name: "cuda-oxide J2K encode",
        extra_sources: J2K_ENCODE_EXTRA_SOURCES,
        built_cfg: "j2k_cuda_oxide_j2k_encode_built",
    },
    CudaOxideProject {
        feature_env: "CARGO_FEATURE_CUDA_OXIDE_J2K_ML",
        source_dir: "src/cuda_oxide_j2k_ml",
        output_name: "cuda_oxide_j2k_ml.ptx",
        artifact_name: "j2k_cuda_oxide_j2k_ml.ptx",
        display_name: "cuda-oxide j2k-ml",
        extra_sources: &[],
        built_cfg: "j2k_cuda_oxide_j2k_ml_built",
    },
    CudaOxideProject {
        feature_env: "CARGO_FEATURE_CUDA_OXIDE_J2K_CLASSIC_DECODE",
        source_dir: "src/cuda_oxide_j2k_classic_decode",
        output_name: "cuda_oxide_j2k_classic_decode.ptx",
        artifact_name: "j2k_cuda_oxide_j2k_classic_decode.ptx",
        display_name: "cuda-oxide classic J2K decode",
        extra_sources: &[],
        built_cfg: "j2k_cuda_oxide_j2k_classic_decode_built",
    },
    CudaOxideProject {
        feature_env: "CARGO_FEATURE_CUDA_OXIDE_HTJ2K_DECODE",
        source_dir: "src/cuda_oxide_htj2k_decode",
        output_name: "cuda_oxide_htj2k_decode.ptx",
        artifact_name: "j2k_cuda_oxide_htj2k_decode.ptx",
        display_name: "cuda-oxide HTJ2K decode",
        extra_sources: &[],
        built_cfg: "j2k_cuda_oxide_htj2k_decode_built",
    },
    CudaOxideProject {
        feature_env: "CARGO_FEATURE_CUDA_OXIDE_J2K_DEQUANTIZE",
        source_dir: "src/cuda_oxide_j2k_dequantize",
        output_name: "cuda_oxide_j2k_dequantize.ptx",
        artifact_name: "j2k_cuda_oxide_j2k_dequantize.ptx",
        display_name: "cuda-oxide J2K dequantize",
        extra_sources: &[],
        built_cfg: "j2k_cuda_oxide_j2k_dequantize_built",
    },
    CudaOxideProject {
        feature_env: "CARGO_FEATURE_CUDA_OXIDE_J2K_DECODE_STORE",
        source_dir: "src/cuda_oxide_j2k_decode_store",
        output_name: "cuda_oxide_j2k_decode_store.ptx",
        artifact_name: "j2k_cuda_oxide_j2k_decode_store.ptx",
        display_name: "cuda-oxide J2K decode store",
        extra_sources: J2K_DECODE_STORE_EXTRA_SOURCES,
        built_cfg: "j2k_cuda_oxide_j2k_decode_store_built",
    },
    CudaOxideProject {
        feature_env: "CARGO_FEATURE_CUDA_OXIDE_J2K_IDWT",
        source_dir: "src/cuda_oxide_j2k_idwt",
        output_name: "cuda_oxide_j2k_idwt.ptx",
        artifact_name: "j2k_cuda_oxide_j2k_idwt.ptx",
        display_name: "cuda-oxide J2K IDWT",
        extra_sources: &[],
        built_cfg: "j2k_cuda_oxide_j2k_idwt_built",
    },
];

fn main() -> Result<(), Box<dyn Error>> {
    build_cuda_oxide_projects(PROJECTS, None)
}
