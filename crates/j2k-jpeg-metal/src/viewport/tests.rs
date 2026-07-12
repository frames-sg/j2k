// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

mod budget;

const BASELINE_420: &[u8] = include_bytes!("../../fixtures/jpeg/baseline_420_16x16.jpg");
#[cfg(target_os = "macos")]
const BASELINE_422: &[u8] = include_bytes!("../../fixtures/jpeg/baseline_422_16x8.jpg");
const BASELINE_420_RESTART: &[u8] =
    include_bytes!("../../fixtures/jpeg/baseline_420_restart_32x16.jpg");

fn sparse_workload_from(workload: &ViewportWorkload) -> ViewportWorkload {
    ViewportWorkload {
        scale: workload.scale,
        viewport_dims: workload.viewport_dims,
        tiles: vec![
            *workload.tiles.first().expect("viewport tile"),
            *workload.tiles.last().expect("viewport tile"),
        ],
    }
}

#[test]
fn auto_strategy_keeps_large_contiguous_nonrestart_workloads_on_cpu_contiguous() {
    let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
    let workload = suggest_viewport_workload((2_048, 1_024)).expect("contiguous workload");

    assert_eq!(
        choose_viewport_surface_strategy_for_decoder(&decoder, &workload, BackendRequest::Auto)
            .expect("strategy"),
        ViewportSurfaceStrategy::CpuContiguous
    );
}

#[test]
fn auto_strategy_prefers_hybrid_for_restart_coded_contiguous_workloads() {
    let decoder = CpuDecoder::new(BASELINE_420_RESTART).expect("decoder");
    let workload = ViewportWorkload {
        scale: Downscale::None,
        viewport_dims: (32, 16),
        tiles: vec![ViewportTile {
            source_roi: Rect {
                x: 0,
                y: 0,
                w: 32,
                h: 16,
            },
            dest: Rect {
                x: 0,
                y: 0,
                w: 32,
                h: 16,
            },
        }],
    };

    assert_eq!(
        choose_viewport_surface_strategy_for_decoder(&decoder, &workload, BackendRequest::Auto)
            .expect("strategy"),
        if cfg!(target_os = "macos") {
            ViewportSurfaceStrategy::HybridContiguous
        } else {
            ViewportSurfaceStrategy::CpuContiguous
        }
    );
}

#[cfg(target_os = "macos")]
#[test]
fn direct_viewport_plan_reuses_one_packet_owner_for_execution() {
    let decoder = CpuDecoder::new(BASELINE_420_RESTART).expect("decoder");
    let workload = ViewportWorkload {
        scale: Downscale::None,
        viewport_dims: (32, 16),
        tiles: vec![ViewportTile {
            source_roi: Rect {
                x: 0,
                y: 0,
                w: 32,
                h: 16,
            },
            dest: Rect {
                x: 0,
                y: 0,
                w: 32,
                h: 16,
            },
        }],
    };

    let plan = resolve_viewport_surface_plan(&decoder, &workload, BackendRequest::Auto)
        .expect("resolved viewport plan");
    assert_eq!(plan.strategy, ViewportSurfaceStrategy::HybridContiguous);
    let packet = plan.fast_packet.as_ref().expect("direct packet");
    let execution_packets = JpegFastPackets::from_shared(Some(packet));

    assert_eq!(
        std::ptr::from_ref(execution_packets.fast420.expect("execution fast420 packet")),
        std::ptr::from_ref(packet.fast420().expect("planned fast420 packet")),
    );
}

#[cfg(target_os = "macos")]
#[test]
fn viewport_direct_packet_detection_includes_fast422() {
    let decoder = CpuDecoder::new(BASELINE_422).expect("decoder");

    assert!(has_direct_viewport_packet(&decoder).expect("fast-packet selection"));
}

#[test]
fn auto_strategy_keeps_large_sparse_nonrestart_workloads_on_cpu_composite() {
    let decoder = CpuDecoder::new(BASELINE_420).expect("decoder");
    let contiguous = suggest_viewport_workload((2_048, 1_024)).expect("contiguous workload");
    let workload = sparse_workload_from(&contiguous);

    assert_eq!(
        choose_viewport_surface_strategy_for_decoder(&decoder, &workload, BackendRequest::Auto)
            .expect("strategy"),
        ViewportSurfaceStrategy::CpuComposite
    );
}

#[test]
fn auto_strategy_keeps_restart_coded_sparse_workloads_on_cpu_composite() {
    let decoder = CpuDecoder::new(BASELINE_420_RESTART).expect("decoder");
    let contiguous = suggest_viewport_workload((8_192, 2_048)).expect("contiguous workload");
    let workload = sparse_workload_from(&contiguous);

    assert_eq!(
        choose_viewport_surface_strategy_for_decoder(&decoder, &workload, BackendRequest::Auto)
            .expect("strategy"),
        ViewportSurfaceStrategy::CpuComposite
    );
}
