// SPDX-License-Identifier: MIT OR Apache-2.0

const CONTEXT: &str = include_str!("../context.rs");
const MODULES: &[(&str, &str, usize)] = &[
    (
        "context/band_transfer.rs",
        include_str!("band_transfer.rs"),
        75,
    ),
    ("context/compact.rs", include_str!("compact.rs"), 150),
    ("context/creation.rs", include_str!("creation.rs"), 100),
    ("context/device.rs", include_str!("device.rs"), 80),
    ("context/inner.rs", include_str!("inner.rs"), 100),
    (
        "context/kernel_cache.rs",
        include_str!("kernel_cache.rs"),
        250,
    ),
    (
        "context/kernel_dispatch.rs",
        include_str!("kernel_dispatch.rs"),
        425,
    ),
    ("context/lifecycle.rs", include_str!("lifecycle.rs"), 175),
    ("context/operations.rs", include_str!("operations.rs"), 100),
    ("context/pointer.rs", include_str!("pointer.rs"), 50),
    (
        "context/pointer/stream_ordered.rs",
        include_str!("pointer/stream_ordered.rs"),
        150,
    ),
    ("context/pinned_host.rs", include_str!("pinned_host.rs"), 75),
    (
        "context/test_kernels.rs",
        include_str!("test_kernels.rs"),
        180,
    ),
];

#[test]
fn cuda_context_uses_focused_real_modules() {
    let include_macro = ["include", "!("].concat();
    let wildcard_import = ["use super::", "*"].concat();
    assert!(
        CONTEXT.lines().count() < 150,
        "context.rs must remain a focused module shell"
    );
    for module in [
        "mod band_transfer;",
        "mod compact;",
        "mod creation;",
        "mod device;",
        "mod inner;",
        "mod kernel_cache;",
        "mod kernel_dispatch;",
        "mod lifecycle;",
        "mod operations;",
        "mod pointer;",
        "mod pinned_host;",
    ] {
        assert!(CONTEXT.contains(module), "context.rs must contain {module}");
    }
    assert!(!CONTEXT.contains(&include_macro));

    for (path, source, max_lines) in MODULES {
        assert!(
            source.lines().count() < *max_lines,
            "{path} must stay below its focused-module line-count ratchet"
        );
        assert!(
            !source.contains(&include_macro),
            "{path} must be a real module"
        );
        assert!(
            !source.contains(&wildcard_import),
            "{path} must use explicit imports"
        );
    }
}
