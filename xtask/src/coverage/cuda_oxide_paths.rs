// SPDX-License-Identifier: MIT OR Apache-2.0

use std::path::Path;

pub(super) const CUDA_OXIDE_PACKAGE_ROOTS: &[&str] = &[
    "crates/j2k-cuda-runtime",
    "crates/j2k-cuda-j2k-engine",
    "crates/j2k-cuda-jpeg-engine",
    "crates/j2k-cuda-transcode-engine",
];

pub(super) fn is_cuda_oxide_source(path: &str) -> bool {
    cuda_oxide_relative_path(path).is_some()
}

pub(super) fn is_cuda_oxide_device_rust(path: &str) -> bool {
    is_cuda_oxide_source(path)
        && path.contains("/simt/src/")
        && Path::new(path)
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("rs"))
}

pub(super) fn is_cuda_oxide_host_scaffold(path: &str) -> bool {
    let Some(relative) = cuda_oxide_relative_path(path) else {
        return false;
    };
    let mut components = relative.split('/');
    components.next().is_some()
        && components.next() == Some("src")
        && components.next() == Some("main.rs")
        && components.next().is_none()
}

fn cuda_oxide_relative_path(path: &str) -> Option<&str> {
    CUDA_OXIDE_PACKAGE_ROOTS.iter().find_map(|root| {
        path.strip_prefix(root)
            .and_then(|relative| relative.strip_prefix("/src/cuda_oxide_"))
    })
}
