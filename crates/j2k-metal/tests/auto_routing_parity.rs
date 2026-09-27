// SPDX-License-Identifier: MIT OR Apache-2.0

//! Route-parity pre-flight for an Auto-routing corpus. `benches/auto_routing`
//! stops at the first decode whose Metal route differs from the CPU route;
//! this check decodes every case and operation and reports every mismatch, so
//! a corpus can be vetted before the long timing run.
//!
//! ```sh
//! J2K_AUTO_ROUTING_MANIFEST=/path/manifest.json J2K_AUTO_ROUTING_ROOT=/path/corpus \
//!   cargo test --profile gpu-quick -p j2k-metal --test auto_routing_parity -- \
//!   --include-ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use std::path::Path;

use j2k_core::{BackendRequest, Downscale, PixelFormat, Rect};
use j2k_metal::{J2kDecoder, MetalBackendSession, MetalDecodeRequest};
use j2k_test_support::{
    load_auto_routing_manifest, AutoRoutingPixelFormat, AutoRoutingWorkloadKind,
};

fn decode(bytes: &[u8], request: MetalDecodeRequest, session: &MetalBackendSession) -> Vec<u8> {
    J2kDecoder::new(bytes)
        .expect("parity decoder")
        .decode_request_to_device_with_session(request, session)
        .expect("parity decode")
        .as_bytes()
        .expect("parity readback")
        .into_owned()
}

/// Operations the routing bench times, as `(label, request)` builders.
fn requests(
    fmt: PixelFormat,
    (width, height): (u32, u32),
) -> [(&'static str, impl Fn(BackendRequest) -> MetalDecodeRequest); 3] {
    let roi = Rect {
        x: width.saturating_sub((width / 2).max(1)) / 2,
        y: height.saturating_sub((height / 2).max(1)) / 2,
        w: (width / 2).max(1),
        h: (height / 2).max(1),
    };
    [
        (
            "full",
            Box::new(move |backend| MetalDecodeRequest::full(fmt, backend))
                as Box<dyn Fn(BackendRequest) -> MetalDecodeRequest>,
        ),
        (
            "roi",
            Box::new(move |backend| MetalDecodeRequest::region(fmt, roi, backend)),
        ),
        (
            "half",
            Box::new(move |backend| MetalDecodeRequest::scaled(fmt, Downscale::Half, backend)),
        ),
    ]
}

#[test]
#[ignore = "corpus pre-flight; set J2K_AUTO_ROUTING_MANIFEST and J2K_AUTO_ROUTING_ROOT"]
fn auto_routing_corpus_routes_are_bit_identical() {
    let manifest = std::env::var_os("J2K_AUTO_ROUTING_MANIFEST")
        .expect("J2K_AUTO_ROUTING_MANIFEST must name the corpus manifest");
    let root = std::env::var_os("J2K_AUTO_ROUTING_ROOT")
        .expect("J2K_AUTO_ROUTING_ROOT must name the corpus root");
    let set = load_auto_routing_manifest(Path::new(&manifest), Path::new(&root))
        .expect("load Auto-routing manifest");
    let session = MetalBackendSession::system_default().expect("Metal session");
    let mut mismatches = Vec::new();
    for workload in set
        .workloads
        .iter()
        .filter(|workload| workload.kind == AutoRoutingWorkloadKind::Decode)
    {
        let fmt = match workload.pixel_format {
            AutoRoutingPixelFormat::Gray8 => PixelFormat::Gray8,
            AutoRoutingPixelFormat::Rgb8 => PixelFormat::Rgb8,
        };
        let dimensions = j2k::J2kDecoder::inspect(&workload.bytes)
            .expect("inspect parity case")
            .dimensions;
        for (label, request) in requests(fmt, dimensions) {
            let cpu = decode(&workload.bytes, request(BackendRequest::Cpu), &session);
            let metal = decode(&workload.bytes, request(BackendRequest::Metal), &session);
            if cpu == metal {
                continue;
            }
            let differing = cpu.iter().zip(&metal).filter(|(a, b)| a != b).count();
            let max_delta = cpu
                .iter()
                .zip(&metal)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap_or(0);
            mismatches.push(format!(
                "{} {label}: {differing} of {} bytes differ, max delta {max_delta}, lengths {} / {}",
                workload.id,
                cpu.len(),
                cpu.len(),
                metal.len()
            ));
        }
    }
    for line in &mismatches {
        println!("{line}");
    }
    assert!(
        mismatches.is_empty(),
        "{} route mismatches",
        mismatches.len()
    );
}
