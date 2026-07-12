// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use super::*;

const METAL_ADAPTER_ROOTS: &[&str] = &[
    "crates/j2k-metal",
    "crates/j2k-jpeg-metal",
    "crates/j2k-transcode-metal",
];

const DIRECT_RESOURCE_CONSTRUCTORS: &[&str] = &[
    ".new_buffer(",
    ".new_buffer_with_data(",
    ".new_buffer_with_bytes_no_copy(",
    ".new_texture(",
    "TextureDescriptor::new(",
    ".new_command_queue(",
    ".new_command_buffer(",
    ".new_compute_command_encoder(",
    ".new_blit_command_encoder(",
];

const RAW_CONSTRUCTION_BOUNDARIES: &[(&str, &[&str])] = &[
    (
        "Sel::register(\"newBufferWithLength:options:\")",
        &["crates/j2k-metal-support/src/allocation.rs"],
    ),
    (
        "Sel::register(\"newBufferWithBytes:length:options:\")",
        &["crates/j2k-metal-support/src/allocation.rs"],
    ),
    (
        "Sel::register(\"newTextureWithDescriptor:\")",
        &["crates/j2k-metal-support/src/allocation.rs"],
    ),
    (
        "Sel::register(\"newCommandQueue\")",
        &["crates/j2k-metal-support/src/runtime.rs"],
    ),
    (
        "Sel::register(\"commandBuffer\")",
        &["crates/j2k-metal-support/src/runtime.rs"],
    ),
    (
        "Sel::register(\"computeCommandEncoder\")",
        &["crates/j2k-metal-support/src/runtime.rs"],
    ),
    (
        "Sel::register(\"blitCommandEncoder\")",
        &["crates/j2k-metal-support/src/runtime.rs"],
    ),
    (
        "Sel::register(\"newLibraryWithSource:options:error:\")",
        &["crates/j2k-metal-support/src/pipeline.rs"],
    ),
    (
        "Sel::register(\"newComputePipelineStateWithFunction:error:\")",
        &["crates/j2k-metal-support/src/pipeline.rs"],
    ),
    (
        "Sel::register(\"new\")",
        &[
            "crates/j2k-metal-support/src/allocation.rs",
            "crates/j2k-metal-support/src/pipeline.rs",
        ],
    ),
    (
        "Sel::register(\"alloc\")",
        &["crates/j2k-metal-support/src/pipeline.rs"],
    ),
    (
        "Sel::register(\"initWithBytes:length:encoding:\")",
        &["crates/j2k-metal-support/src/pipeline.rs"],
    ),
];

#[test]
fn metal_adapters_use_only_checked_resource_construction() {
    let root = repo_root();
    assert_rust_source_scan_checks(
        root,
        &[
            RustSourceScanCheck::new("Metal adapter resource construction", METAL_ADAPTER_ROOTS)
                .forbidden(DIRECT_RESOURCE_CONSTRUCTORS),
        ],
    );

    assert_rust_source_scan_checks(
        root,
        &[RustSourceScanCheck::new(
            "Metal adapter foreign-handle construction",
            METAL_ADAPTER_ROOTS,
        )
        .forbidden(&["::from_ptr("])],
    );
}

#[test]
fn raw_metal_resource_construction_stays_in_focused_support_modules() {
    let root = repo_root();
    let scan_roots = [
        "crates/j2k-metal-support",
        "crates/j2k-metal",
        "crates/j2k-jpeg-metal",
        "crates/j2k-transcode-metal",
    ];
    let mut sources = Vec::new();
    for relative_root in scan_roots {
        for path in rust_sources(&root.join(relative_root)) {
            let relative = path
                .strip_prefix(root)
                .expect("Metal source under repository root")
                .to_string_lossy()
                .replace('\\', "/");
            let source = fs::read_to_string(&path)
                .unwrap_or_else(|error| panic!("read {relative}: {error}"));
            sources.push((relative, source));
        }
    }

    for (selector, allowed_owners) in RAW_CONSTRUCTION_BOUNDARIES {
        let mut matches = sources
            .iter()
            .filter_map(|(relative, source)| source.contains(selector).then_some(relative.as_str()))
            .collect::<Vec<_>>();
        matches.sort_unstable();
        assert_eq!(
            matches, *allowed_owners,
            "raw Metal selector `{selector}` must stay in its reviewed support owner"
        );
    }

    for (relative, source) in &sources {
        if source.contains("::from_ptr(") {
            assert!(
                matches!(
                    relative.as_str(),
                    "crates/j2k-metal-support/src/allocation.rs"
                        | "crates/j2k-metal-support/src/pipeline.rs"
                        | "crates/j2k-metal-support/src/runtime.rs"
                ),
                "foreign Metal handles may only be formed after nil checks in focused support modules; found in {relative}"
            );
        }
        assert!(
            !source.contains("newBufferWithBytesNoCopy")
                && !source.contains("new_buffer_with_bytes_no_copy")
                && !source.contains("checked_shared_buffer_with_mut_slice_no_copy"),
            "borrow-erasing Metal no-copy constructors must remain absent; found in {relative}"
        );
    }
}

#[test]
fn metal_support_modules_stay_reviewably_focused() {
    let root = repo_root();
    let source_root = root.join("crates/j2k-metal-support/src");
    for path in rust_sources(&source_root) {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("UTF-8 Metal support source name");
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        let line_count = source.lines().count();
        let cap = match name {
            "lib.rs" => 100,
            "tests.rs" => 600,
            _ => 425,
        };
        assert!(
            line_count <= cap,
            "Metal support {name} has {line_count} lines, exceeding its focused-owner cap of {cap}"
        );
    }
}

#[test]
fn checked_texture_preflights_geometry_and_planned_bytes_before_allocation() {
    let root = repo_root();
    let allocation = fs::read_to_string(root.join("crates/j2k-metal-support/src/allocation.rs"))
        .expect("read Metal support allocation owner");
    let errors = fs::read_to_string(root.join("crates/j2k-metal-support/src/error.rs"))
        .expect("read Metal support errors");

    assert_pattern_checks(&[
        PatternCheck::new("checked Metal texture allocation plan", &allocation).required(&[
            "fn checked_texture_descriptor_geometry(",
            "descriptor.mipmap_level_count() == 0",
            "descriptor.sample_count() == 0",
            "fn checked_texture_planned_bytes(",
            "device.heap_texture_size_and_align(descriptor).size",
            "let dimensions = checked_texture_allocation_plan(device, descriptor)?;",
            "Sel::register(\"newTextureWithDescriptor:\")",
        ]),
        PatternCheck::new("typed Metal texture preflight errors", &errors)
            .required(&["TextureDescriptorInvalid {", "TextureAllocationTooLarge {"]),
    ]);
    assert!(
        allocation.find("checked_texture_allocation_plan(device, descriptor)?")
            < allocation.find("Sel::register(\"newTextureWithDescriptor:\")"),
        "texture allocation preflight must precede Objective-C allocation dispatch"
    );
}

#[test]
fn checked_command_resources_are_owned_across_autorelease_boundaries() {
    let root = repo_root();
    let runtime = fs::read_to_string(root.join("crates/j2k-metal-support/src/runtime.rs"))
        .expect("read Metal support runtime owner");
    let tests = fs::read_to_string(root.join("crates/j2k-metal-support/src/tests.rs"))
        .expect("read Metal support tests");

    assert_pattern_checks(&[
        PatternCheck::new("owned checked Metal command resources", &runtime)
            .required(&[
                "checked_command_buffer_from_autoreleased_ptr(",
                "Result<CommandBuffer, MetalSupportError>",
                "CommandBufferRef::from_ptr(raw) }.to_owned()",
                "checked_compute_encoder_from_autoreleased_ptr(",
                "Result<ComputeCommandEncoder, MetalSupportError>",
                "ComputeCommandEncoderRef::from_ptr(raw) }.to_owned()",
                "checked_blit_encoder_from_autoreleased_ptr(",
                "Result<BlitCommandEncoder, MetalSupportError>",
                "BlitCommandEncoderRef::from_ptr(raw) }.to_owned()",
            ])
            .forbidden(&[
                "Result<&CommandBufferRef, MetalSupportError>",
                "Result<&ComputeCommandEncoderRef, MetalSupportError>",
                "Result<&BlitCommandEncoderRef, MetalSupportError>",
            ]),
        PatternCheck::new("Metal autorelease ownership regression", &tests).required(&[
            "fn checked_command_resources_survive_their_creation_autorelease_pool()",
            "metal::objc::rc::autoreleasepool",
            "checked_command_buffer(&queue)",
            "checked_compute_command_encoder(&command_buffer)",
            "checked_blit_command_encoder(&command_buffer)",
            "commit_and_wait(&command_buffer)",
        ]),
    ]);
}

#[test]
fn j2k_metal_slice_uploads_use_one_owned_copy_boundary() {
    let root = repo_root();
    let compute_root = root.join("crates/j2k-metal/src/compute");
    let buffers = fs::read_to_string(compute_root.join("direct_buffers.rs"))
        .expect("read J2K Metal direct buffer owner");

    assert_eq!(
        buffers.matches("fn copied_slice_buffer").count(),
        1,
        "J2K Metal slice upload construction must stay single-sourced"
    );
    assert_pattern_checks(&[
        PatternCheck::new("owned J2K Metal slice upload", &buffers).required(&[
            "/// Copy caller-owned GPU ABI input into Metal-owned shared storage.",
            "fn copied_slice_buffer<T: GpuAbi>",
            "new_shared_buffer_with_slice(device, data)",
        ]),
    ]);

    for path in rust_sources(&compute_root) {
        let source = fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("read {}: {error}", path.display()));
        assert!(
            !source.contains("borrow_slice_buffer")
                && !source.contains("owned_slice_buffer")
                && !source.contains("retained_cpu_coefficients"),
            "obsolete no-copy lifetime scaffolding must remain absent from {}",
            path.display()
        );
    }
}
