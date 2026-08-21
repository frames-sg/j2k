use std::error::Error;

use j2k_cuda_build_support::{build_cuda_oxide_projects, CudaOxideProject};

const TRANSCODE_EXTRA_SOURCES: &[&str] = &[
    "simt/src/constants.rs",
    "simt/src/dwt97.rs",
    "simt/src/exports.rs",
    "simt/src/helpers.rs",
    "simt/src/quantization.rs",
    "simt/src/reversible53.rs",
];

const PROJECTS: &[CudaOxideProject] = &[CudaOxideProject {
    feature_env: "CARGO_FEATURE_CUDA_OXIDE_TRANSCODE",
    source_dir: "src/cuda_oxide_transcode",
    output_name: "cuda_oxide_transcode.ptx",
    artifact_name: "j2k_cuda_oxide_transcode.ptx",
    display_name: "cuda-oxide transcode",
    extra_sources: TRANSCODE_EXTRA_SOURCES,
    built_cfg: "j2k_cuda_oxide_transcode_built",
}];

fn main() -> Result<(), Box<dyn Error>> {
    build_cuda_oxide_projects(PROJECTS, None)
}
