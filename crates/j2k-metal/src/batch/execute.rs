// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::BackendKind;

use crate::{profile, Surface};

use super::heuristics::{
    can_decode_requests_as_repeated_full_color_batch,
    can_decode_requests_as_repeated_full_grayscale_batch, profile_route_label, BatchRoute,
    GroupedRequests,
};
use super::{
    decode_cpu_host_batch, decode_distinct_full_color_batch, decode_distinct_full_grayscale_batch,
    decode_distinct_region_scaled_direct_batch,
    decode_distinct_region_scaled_direct_batch_prechecked, decode_individual,
    decode_repeated_full_color, decode_repeated_full_grayscale,
    decode_repeated_region_scaled_direct_batch_prechecked, QueuedRequest, SessionState,
};

fn complete_cpu_host_fallback(session: &mut SessionState, requests: Vec<QueuedRequest>) {
    if requests.len() > 1 {
        if let Some(Ok(surfaces)) = decode_cpu_host_batch(&requests) {
            if complete_batch_surfaces(session, &requests, surfaces) {
                return;
            }
        }
    }
    for request in requests {
        session.submissions = session.submissions.saturating_add(1);
        session.completed[request.output_slot] = Some(decode_individual(&request));
    }
}

fn complete_batch_surfaces(
    session: &mut SessionState,
    requests: &[QueuedRequest],
    surfaces: Vec<Surface>,
) -> bool {
    if surfaces.len() != requests.len() {
        return false;
    }
    session.submissions = session.submissions.saturating_add(1);
    for (request, surface) in requests.iter().zip(surfaces) {
        session.completed[request.output_slot] = Some(Ok(surface));
    }
    true
}

pub(super) fn process_batch(session: &mut SessionState, grouped: GroupedRequests) {
    let GroupedRequests { route, requests } = grouped;
    let profile_enabled = profile::metal_profile_stages_enabled();
    let started = profile::profile_now(profile_enabled);
    let request_count = requests.len();
    let slots = if profile_enabled {
        requests
            .iter()
            .map(|request| request.output_slot)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let backend = profile_backend_label(&requests);
    let fmt = profile_format_label(&requests);

    process_batch_inner(session, route, requests);

    if profile_enabled {
        profile::emit_metal_batch_profile_row(
            "decode",
            &profile::MetalBatchProfileRow {
                slice: "decode_batch",
                stage: "execute",
                pipeline: "metal_cpu_hybrid",
                processor: "hybrid",
                route: profile_route_label(route),
                backend: &backend,
                fmt: &fmt,
                request_count,
                output_count: profile_completed_output_count(session, &slots),
                elapsed_us: profile::elapsed_us(started),
                outcome: profile_completed_outcome(session, &slots),
            },
        );
    }
}

fn process_batch_inner(
    session: &mut SessionState,
    route: BatchRoute,
    requests: Vec<QueuedRequest>,
) {
    if route == BatchRoute::AutoRegionScaledDirectCpu {
        complete_cpu_host_fallback(session, requests);
        return;
    }

    if matches!(
        route,
        BatchRoute::AutoRegionScaledDirectMetal | BatchRoute::AutoRepeatedRegionScaledDirectMetal
    ) && requests.len() > 1
    {
        let decoded = if route == BatchRoute::AutoRepeatedRegionScaledDirectMetal {
            decode_repeated_region_scaled_direct_batch_prechecked(&requests)
        } else {
            decode_distinct_region_scaled_direct_batch_prechecked(&requests)
        };
        if let Some(Ok(surfaces)) = decoded {
            if complete_batch_surfaces(session, &requests, surfaces) {
                return;
            }
        }
        complete_cpu_host_fallback(session, requests);
        return;
    }

    if can_decode_requests_as_repeated_full_grayscale_batch(&requests) {
        if let Some(Ok(surfaces)) = decode_repeated_full_grayscale(&requests[0], requests.len()) {
            if complete_batch_surfaces(session, &requests, surfaces) {
                return;
            }
        }
    }

    if can_decode_requests_as_repeated_full_color_batch(&requests) {
        if let Some(Ok(surfaces)) = decode_repeated_full_color(&requests[0], requests.len()) {
            if complete_batch_surfaces(session, &requests, surfaces) {
                return;
            }
        }
    }

    if requests.len() > 1 {
        if let Some(Ok(surfaces)) = decode_distinct_full_grayscale_batch(&requests) {
            if complete_batch_surfaces(session, &requests, surfaces) {
                return;
            }
        }
    }

    if requests.len() > 1 {
        if let Some(Ok(surfaces)) = decode_distinct_full_color_batch(&requests) {
            if complete_batch_surfaces(session, &requests, surfaces) {
                return;
            }
        }
    }

    if requests.len() > 1 {
        if let Some(Ok(surfaces)) = decode_distinct_region_scaled_direct_batch(&requests) {
            if complete_batch_surfaces(session, &requests, surfaces) {
                return;
            }
        }
    }

    if requests.len() > 1 {
        if let Some(Ok(surfaces)) = decode_cpu_host_batch(&requests) {
            if complete_batch_surfaces(session, &requests, surfaces) {
                return;
            }
        }
    }

    for request in requests {
        session.submissions = session.submissions.saturating_add(1);
        session.completed[request.output_slot] = Some(decode_individual(&request));
    }
}

fn profile_backend_label(requests: &[QueuedRequest]) -> String {
    let Some(first) = requests.first() else {
        return "none".to_string();
    };
    if requests
        .iter()
        .all(|request| request.backend == first.backend)
    {
        format!("{:?}", first.backend)
    } else {
        "mixed".to_string()
    }
}

fn profile_format_label(requests: &[QueuedRequest]) -> String {
    let Some(first) = requests.first() else {
        return "none".to_string();
    };
    if requests.iter().all(|request| request.fmt == first.fmt) {
        format!("{:?}", first.fmt)
    } else {
        "mixed".to_string()
    }
}

fn profile_completed_output_count(session: &SessionState, slots: &[usize]) -> usize {
    slots
        .iter()
        .filter(|&&slot| session.completed.get(slot).is_some_and(Option::is_some))
        .count()
}

fn profile_completed_outcome(session: &SessionState, slots: &[usize]) -> &'static str {
    let mut ok_count = 0usize;
    let mut err_count = 0usize;
    let mut cpu_count = 0usize;
    let mut metal_count = 0usize;
    let mut pending_count = 0usize;

    for &slot in slots {
        match session.completed.get(slot).and_then(Option::as_ref) {
            Some(Ok(surface)) => {
                ok_count = ok_count.saturating_add(1);
                match surface.backend {
                    BackendKind::Cpu => cpu_count = cpu_count.saturating_add(1),
                    BackendKind::Metal => metal_count = metal_count.saturating_add(1),
                    BackendKind::Cuda => {}
                }
            }
            Some(Err(_)) => err_count = err_count.saturating_add(1),
            None => pending_count = pending_count.saturating_add(1),
        }
    }

    if pending_count > 0 {
        "pending"
    } else if err_count > 0 && ok_count == 0 {
        "error"
    } else if err_count > 0 {
        "mixed_error"
    } else if metal_count == ok_count && ok_count > 0 {
        "metal_surface"
    } else if cpu_count == ok_count && ok_count > 0 {
        "cpu_surface"
    } else if ok_count > 0 {
        "mixed"
    } else {
        "none"
    }
}
