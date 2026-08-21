use std::error::Error;

use j2k_cuda_build_support::{build_cuda_oxide_projects, CudaOxideProject};

const PROJECTS: &[CudaOxideProject] = &[CudaOxideProject {
    feature_env: "CARGO_FEATURE_CUDA_OXIDE_COPY_U8",
    source_dir: "src/cuda_oxide_copy_u8",
    output_name: "cuda_oxide_copy_u8.ptx",
    artifact_name: "j2k_cuda_oxide_copy_u8.ptx",
    display_name: "cuda-oxide CopyU8",
    extra_sources: &[],
    built_cfg: "j2k_cuda_oxide_copy_u8_built",
}];

fn main() -> Result<(), Box<dyn Error>> {
    build_cuda_oxide_projects(PROJECTS, Some("j2k_cuda_oxide_enabled"))
}
