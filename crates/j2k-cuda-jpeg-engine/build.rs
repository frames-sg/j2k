use std::error::Error;

use j2k_cuda_build_support::{build_cuda_oxide_projects, CudaOxideProject};

const PROJECTS: &[CudaOxideProject] = &[
    CudaOxideProject {
        feature_env: "CARGO_FEATURE_CUDA_OXIDE_JPEG_DECODE",
        source_dir: "src/cuda_oxide_jpeg_decode",
        output_name: "cuda_oxide_jpeg_decode.ptx",
        artifact_name: "j2k_cuda_oxide_jpeg_decode.ptx",
        display_name: "cuda-oxide JPEG decode",
        extra_sources: &["simt/src/component_planes.rs"],
        built_cfg: "j2k_cuda_oxide_jpeg_decode_built",
    },
    CudaOxideProject {
        feature_env: "CARGO_FEATURE_CUDA_OXIDE_JPEG_ENCODE",
        source_dir: "src/cuda_oxide_jpeg_encode",
        output_name: "cuda_oxide_jpeg_encode.ptx",
        artifact_name: "j2k_cuda_oxide_jpeg_encode.ptx",
        display_name: "cuda-oxide JPEG encode",
        extra_sources: &[],
        built_cfg: "j2k_cuda_oxide_jpeg_encode_built",
    },
];

fn main() -> Result<(), Box<dyn Error>> {
    build_cuda_oxide_projects(PROJECTS, None)
}
