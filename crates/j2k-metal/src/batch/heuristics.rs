// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k_core::{BackendRequest, BatchInfrastructureError, PixelFormat};

use super::{BatchOp, QueuedRequest};

const AUTO_REGION_SCALED_DIRECT_BATCH64_MIN_DIM: u32 = 512;
const AUTO_REGION_SCALED_DIRECT_BATCH64_MIN_COUNT: usize = 64;
pub(super) const AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_DIM: u32 = 512;
pub(super) const AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_COUNT: usize = 2;
pub(super) const AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_DIM: u32 = 1024;
pub(super) const AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_COUNT: usize = 16;
const REGION_SCALED_DIRECT_FORMATS: [PixelFormat; 5] = [
    PixelFormat::Gray8,
    PixelFormat::Gray16,
    PixelFormat::Rgb8,
    PixelFormat::Rgba8,
    PixelFormat::Rgb16,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum BatchRoute {
    Generic,
    AutoRegionScaledDirectCpu,
    AutoRegionScaledDirectMetal,
    AutoRepeatedRegionScaledDirectMetal,
}

pub(super) fn profile_route_label(route: BatchRoute) -> &'static str {
    match route {
        BatchRoute::Generic => "generic",
        BatchRoute::AutoRegionScaledDirectCpu => "auto_region_scaled_direct_cpu",
        BatchRoute::AutoRegionScaledDirectMetal => "auto_region_scaled_direct_metal",
        BatchRoute::AutoRepeatedRegionScaledDirectMetal => {
            "auto_repeated_region_scaled_direct_metal"
        }
    }
}

pub(super) struct GroupedRequests {
    pub(super) route: BatchRoute,
    pub(super) requests: Vec<QueuedRequest>,
}

impl GroupedRequests {
    fn generic(requests: Vec<QueuedRequest>) -> Self {
        Self {
            route: BatchRoute::Generic,
            requests,
        }
    }
}

pub(super) fn group_metal_requests(
    queued: Vec<QueuedRequest>,
) -> Result<Vec<GroupedRequests>, BatchInfrastructureError> {
    let request_count = queued.len();
    let budget = crate::batch_allocation::BatchMetadataBudget::new("J2K Metal request grouping");
    budget.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<QueuedRequest>(queued.capacity()),
        crate::batch_allocation::BatchMetadataRequest::of::<QueuedRequest>(queued.len()),
        crate::batch_allocation::BatchMetadataRequest::of::<usize>(queued.len()),
        crate::batch_allocation::BatchMetadataRequest::of::<GroupedRequests>(queued.len()),
        crate::batch_allocation::BatchMetadataRequest::of::<GroupedRequests>(queued.len()),
    ])?;
    let repeated = group_repeated_full_metal_requests(queued)?;
    let grayscale = coalesce_distinct_full_grayscale_metal_requests(repeated)?;
    let color = coalesce_distinct_full_color_metal_requests(grayscale)?;
    let region_scaled = coalesce_distinct_region_scaled_direct_metal_requests(color)?;
    let grouped = coalesce_cpu_host_batches(region_scaled)?;
    let mut actual =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal grouped request ownership");
    actual.account_capacity::<usize>(request_count)?;
    actual.account_capacity::<GroupedRequests>(grouped.capacity())?;
    for group in &grouped {
        actual.account_capacity::<QueuedRequest>(group.requests.capacity())?;
    }
    Ok(grouped)
}

fn group_repeated_full_metal_requests(
    queued: Vec<QueuedRequest>,
) -> Result<Vec<GroupedRequests>, BatchInfrastructureError> {
    let mut batches: Vec<GroupedRequests> = Vec::new();
    for request in queued {
        if let Some(batch) = batches.iter_mut().find(|batch| {
            batch.route == BatchRoute::Generic
                && can_decode_as_repeated_full_metal_batch(&batch.requests[0], &request)
        }) {
            crate::batch_allocation::try_reserve_for_push(
                &mut batch.requests,
                "J2K Metal repeated request group",
            )?;
            batch.requests.push(request);
        } else {
            push_group(
                &mut batches,
                GroupedRequests::generic(singleton_request(request)?),
            )?;
        }
    }
    Ok(batches)
}

fn coalesce_distinct_full_grayscale_metal_requests(
    repeated_batches: Vec<GroupedRequests>,
) -> Result<Vec<GroupedRequests>, BatchInfrastructureError> {
    let mut batches = Vec::new();
    let mut gray8 = Vec::new();
    let mut gray16 = Vec::new();

    for batch in repeated_batches {
        if batch.route == BatchRoute::Generic
            && batch.requests.len() == 1
            && is_distinct_full_grayscale_metal_candidate(&batch.requests[0])
        {
            let request = batch
                .requests
                .into_iter()
                .next()
                .expect("single-entry batch has request");
            match request.fmt {
                PixelFormat::Gray8 => push_request(&mut gray8, request)?,
                PixelFormat::Gray16 => push_request(&mut gray16, request)?,
                _ => push_group(
                    &mut batches,
                    GroupedRequests::generic(singleton_request(request)?),
                )?,
            }
        } else {
            push_group(&mut batches, batch)?;
        }
    }

    push_coalesced_or_single(&mut batches, gray8)?;
    push_coalesced_or_single(&mut batches, gray16)?;
    Ok(batches)
}

fn coalesce_distinct_region_scaled_direct_metal_requests(
    repeated_batches: Vec<GroupedRequests>,
) -> Result<Vec<GroupedRequests>, BatchInfrastructureError> {
    let mut batches = Vec::new();
    let mut metal_by_format: [Vec<QueuedRequest>; REGION_SCALED_DIRECT_FORMATS.len()] =
        std::array::from_fn(|_| Vec::new());
    let mut auto_by_format: [Vec<QueuedRequest>; REGION_SCALED_DIRECT_FORMATS.len()] =
        std::array::from_fn(|_| Vec::new());

    for batch in repeated_batches {
        if batch.route == BatchRoute::Generic
            && batch.requests.len() == 1
            && is_region_scaled_direct_batch_candidate(&batch.requests[0])
        {
            let request = batch
                .requests
                .into_iter()
                .next()
                .expect("single-entry batch has request");
            let Some(format_idx) = region_scaled_direct_format_index(request.fmt) else {
                push_group(
                    &mut batches,
                    GroupedRequests::generic(singleton_request(request)?),
                )?;
                continue;
            };
            match request.backend {
                BackendRequest::Metal => {
                    push_request(&mut metal_by_format[format_idx], request)?;
                }
                BackendRequest::Auto => {
                    push_request(&mut auto_by_format[format_idx], request)?;
                }
                _ => push_group(
                    &mut batches,
                    GroupedRequests::generic(singleton_request(request)?),
                )?,
            }
        } else {
            push_group(&mut batches, batch)?;
        }
    }

    for requests in metal_by_format {
        push_coalesced_or_single(&mut batches, requests)?;
    }
    for requests in auto_by_format {
        push_auto_region_scaled_direct_batches(&mut batches, requests)?;
    }
    Ok(batches)
}

fn push_coalesced_or_single(
    batches: &mut Vec<GroupedRequests>,
    requests: Vec<QueuedRequest>,
) -> Result<(), BatchInfrastructureError> {
    push_coalesced_or_single_with_route(batches, requests, BatchRoute::Generic)
}

fn push_coalesced_or_single_with_route(
    batches: &mut Vec<GroupedRequests>,
    requests: Vec<QueuedRequest>,
    route: BatchRoute,
) -> Result<(), BatchInfrastructureError> {
    if requests.is_empty() {
        return Ok(());
    }
    if requests.len() == 1 {
        for request in requests {
            push_group(
                batches,
                GroupedRequests {
                    route,
                    requests: singleton_request(request)?,
                },
            )?;
        }
    } else {
        push_group(batches, GroupedRequests { route, requests })?;
    }
    Ok(())
}

fn push_auto_region_scaled_direct_batches(
    batches: &mut Vec<GroupedRequests>,
    requests: Vec<QueuedRequest>,
) -> Result<(), BatchInfrastructureError> {
    let Some(classification) = auto_region_scaled_direct_metal_classification(&requests) else {
        push_coalesced_or_single_with_route(
            batches,
            requests,
            BatchRoute::AutoRegionScaledDirectCpu,
        )?;
        return Ok(());
    };

    let mut metal_requests = Vec::new();
    let mut cpu_requests = Vec::new();
    for request in requests {
        if request
            .max_image_dim()
            .is_some_and(|max_dim| max_dim >= classification.min_dim)
        {
            push_request(&mut metal_requests, request)?;
        } else {
            push_request(&mut cpu_requests, request)?;
        }
    }
    push_coalesced_or_single_with_route(batches, metal_requests, classification.route)?;
    push_coalesced_or_single_with_route(
        batches,
        cpu_requests,
        BatchRoute::AutoRegionScaledDirectCpu,
    )?;
    Ok(())
}

#[expect(
    clippy::similar_names,
    reason = "pixel-format buckets intentionally use parallel rgb8/rgba8/rgb16 names"
)]
fn coalesce_distinct_full_color_metal_requests(
    repeated_batches: Vec<GroupedRequests>,
) -> Result<Vec<GroupedRequests>, BatchInfrastructureError> {
    let mut batches = Vec::new();
    let mut rgb8 = Vec::new();
    let mut rgba8 = Vec::new();
    let mut rgb16 = Vec::new();

    for batch in repeated_batches {
        if batch.route == BatchRoute::Generic
            && batch.requests.len() == 1
            && is_distinct_full_color_metal_candidate(&batch.requests[0])
        {
            let request = batch
                .requests
                .into_iter()
                .next()
                .expect("single-entry batch has request");
            match request.fmt {
                PixelFormat::Rgb8 => push_request(&mut rgb8, request)?,
                PixelFormat::Rgba8 => push_request(&mut rgba8, request)?,
                PixelFormat::Rgb16 => push_request(&mut rgb16, request)?,
                _ => push_group(
                    &mut batches,
                    GroupedRequests::generic(singleton_request(request)?),
                )?,
            }
        } else {
            push_group(&mut batches, batch)?;
        }
    }

    push_coalesced_or_single(&mut batches, rgb8)?;
    push_coalesced_or_single(&mut batches, rgba8)?;
    push_coalesced_or_single(&mut batches, rgb16)?;
    Ok(batches)
}

fn coalesce_cpu_host_batches(
    batches: Vec<GroupedRequests>,
) -> Result<Vec<GroupedRequests>, BatchInfrastructureError> {
    let mut coalesced: Vec<GroupedRequests> = Vec::new();
    let mut cpu_groups: Vec<Vec<QueuedRequest>> = Vec::new();
    for batch in batches {
        if batch.route == BatchRoute::Generic
            && batch.requests.len() == 1
            && is_cpu_host_batch_candidate(&batch.requests[0])
        {
            let request = batch
                .requests
                .into_iter()
                .next()
                .expect("single-entry batch has request");
            if let Some(existing) = cpu_groups
                .iter_mut()
                .find(|existing| can_coalesce_cpu_host_batch(&existing[0], &request))
            {
                push_request(existing, request)?;
            } else {
                crate::batch_allocation::try_reserve_for_push(
                    &mut cpu_groups,
                    "J2K Metal CPU request groups",
                )?;
                cpu_groups.push(singleton_request(request)?);
            }
        } else {
            push_group(&mut coalesced, batch)?;
        }
    }
    for requests in cpu_groups {
        push_group(&mut coalesced, GroupedRequests::generic(requests))?;
    }
    Ok(coalesced)
}

fn singleton_request(
    request: QueuedRequest,
) -> Result<Vec<QueuedRequest>, BatchInfrastructureError> {
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal singleton request group");
    let mut requests = budget.try_vec(1, "J2K Metal singleton request")?;
    requests.push(request);
    Ok(requests)
}

fn push_request(
    requests: &mut Vec<QueuedRequest>,
    request: QueuedRequest,
) -> Result<(), BatchInfrastructureError> {
    crate::batch_allocation::try_reserve_for_push(requests, "J2K Metal grouped requests")?;
    requests.push(request);
    Ok(())
}

fn push_group(
    batches: &mut Vec<GroupedRequests>,
    batch: GroupedRequests,
) -> Result<(), BatchInfrastructureError> {
    crate::batch_allocation::try_reserve_for_push(batches, "J2K Metal request groups")?;
    batches.push(batch);
    Ok(())
}

fn is_cpu_host_batch_candidate(request: &QueuedRequest) -> bool {
    matches!(request.op, BatchOp::Full | BatchOp::RegionScaled { .. })
        && matches!(request.backend, BackendRequest::Cpu | BackendRequest::Auto)
}

fn can_coalesce_cpu_host_batch(first: &QueuedRequest, next: &QueuedRequest) -> bool {
    is_cpu_host_batch_candidate(first)
        && is_cpu_host_batch_candidate(next)
        && first.fmt == next.fmt
        && matches!(
            (&first.op, &next.op),
            (BatchOp::Full, BatchOp::Full)
                | (BatchOp::RegionScaled { .. }, BatchOp::RegionScaled { .. })
        )
}

fn can_decode_as_repeated_full_grayscale_batch(
    first: &QueuedRequest,
    next: &QueuedRequest,
) -> bool {
    is_repeated_full_grayscale_candidate(first)
        && is_repeated_full_grayscale_candidate(next)
        && first.fmt == next.fmt
        && first.backend == next.backend
        && same_input_bytes(first, next)
}

fn can_decode_as_repeated_full_color_batch(first: &QueuedRequest, next: &QueuedRequest) -> bool {
    is_repeated_full_color_candidate(first)
        && is_repeated_full_color_candidate(next)
        && first.fmt == next.fmt
        && first.backend == next.backend
        && same_input_bytes(first, next)
}

pub(super) fn same_input_bytes(first: &QueuedRequest, next: &QueuedRequest) -> bool {
    if Arc::ptr_eq(&first.input, &next.input) {
        return true;
    }
    if first.input.len() != next.input.len() {
        return false;
    }
    if first.input_fingerprint() != next.input_fingerprint() {
        return false;
    }
    first.input.as_ref() == next.input.as_ref()
}

fn can_decode_as_repeated_full_metal_batch(first: &QueuedRequest, next: &QueuedRequest) -> bool {
    can_decode_as_repeated_full_grayscale_batch(first, next)
        || can_decode_as_repeated_full_color_batch(first, next)
}

pub(super) fn is_repeated_full_grayscale_candidate(request: &QueuedRequest) -> bool {
    matches!(request.op, BatchOp::Full)
        && matches!(request.fmt, PixelFormat::Gray8 | PixelFormat::Gray16)
        && matches!(
            request.backend,
            BackendRequest::Auto | BackendRequest::Metal
        )
}

pub(super) fn is_repeated_full_color_candidate(request: &QueuedRequest) -> bool {
    matches!(request.op, BatchOp::Full)
        && matches!(
            request.fmt,
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
        )
        && request.backend == BackendRequest::Metal
}

pub(super) fn is_distinct_full_grayscale_metal_candidate(request: &QueuedRequest) -> bool {
    matches!(request.op, BatchOp::Full)
        && matches!(request.fmt, PixelFormat::Gray8 | PixelFormat::Gray16)
        && request.backend == BackendRequest::Metal
}

pub(super) fn is_distinct_full_color_metal_candidate(request: &QueuedRequest) -> bool {
    matches!(request.op, BatchOp::Full)
        && matches!(
            request.fmt,
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
        )
        && request.backend == BackendRequest::Metal
}

pub(super) fn is_region_scaled_direct_batch_candidate(request: &QueuedRequest) -> bool {
    matches!(request.op, BatchOp::RegionScaled { .. })
        && region_scaled_direct_format_index(request.fmt).is_some()
        && matches!(
            request.backend,
            BackendRequest::Auto | BackendRequest::Metal
        )
}

fn region_scaled_direct_format_index(fmt: PixelFormat) -> Option<usize> {
    REGION_SCALED_DIRECT_FORMATS
        .iter()
        .position(|candidate| *candidate == fmt)
}

pub(super) fn should_auto_use_metal_for_region_scaled_direct_batch(
    requests: &[QueuedRequest],
) -> bool {
    auto_region_scaled_direct_metal_min_dim(requests).is_some()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AutoRegionScaledDirectMetalClassification {
    min_dim: u32,
    route: BatchRoute,
}

pub(super) fn auto_region_scaled_direct_metal_min_dim(requests: &[QueuedRequest]) -> Option<u32> {
    auto_region_scaled_direct_metal_classification(requests)
        .map(|classification| classification.min_dim)
}

fn auto_region_scaled_direct_metal_classification(
    requests: &[QueuedRequest],
) -> Option<AutoRegionScaledDirectMetalClassification> {
    let first = requests.first()?;
    let is_repeated_rgb = matches!(
        first.fmt,
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
    ) && can_decode_requests_as_repeated_region_scaled_batch(requests);
    if matches!(
        first.fmt,
        PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
    ) {
        if !is_repeated_rgb {
            return None;
        }
        let repeated_rgb_eligible = requests
            .iter()
            .filter(|request| {
                request.max_image_dim().is_some_and(|max_dim| {
                    max_dim >= AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_DIM
                })
            })
            .count();
        if repeated_rgb_eligible >= AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_COUNT {
            return Some(AutoRegionScaledDirectMetalClassification {
                min_dim: AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_DIM,
                route: BatchRoute::AutoRepeatedRegionScaledDirectMetal,
            });
        }
    }

    let mut count_512_class = 0usize;
    let mut count_1024_class = 0usize;
    for request in requests {
        let Some(max_dim) = request.max_image_dim() else {
            continue;
        };
        if max_dim >= AUTO_REGION_SCALED_DIRECT_BATCH64_MIN_DIM {
            count_512_class += 1;
        }
        if max_dim >= AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_DIM {
            count_1024_class += 1;
        }
    }

    if count_512_class >= AUTO_REGION_SCALED_DIRECT_BATCH64_MIN_COUNT {
        Some(AutoRegionScaledDirectMetalClassification {
            min_dim: AUTO_REGION_SCALED_DIRECT_BATCH64_MIN_DIM,
            route: BatchRoute::AutoRegionScaledDirectMetal,
        })
    } else if count_1024_class >= AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_COUNT {
        Some(AutoRegionScaledDirectMetalClassification {
            min_dim: AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_DIM,
            route: BatchRoute::AutoRegionScaledDirectMetal,
        })
    } else {
        None
    }
}

pub(super) fn can_decode_requests_as_repeated_region_scaled_batch(
    requests: &[QueuedRequest],
) -> bool {
    let Some((first, rest)) = requests.split_first() else {
        return false;
    };
    !rest.is_empty()
        && rest.iter().all(|request| {
            is_region_scaled_direct_batch_candidate(first)
                && is_region_scaled_direct_batch_candidate(request)
                && first.fmt == request.fmt
                && first.backend == request.backend
                && same_input_bytes(first, request)
                && matches!(
                    (first.op, request.op),
                    (
                        BatchOp::RegionScaled {
                            roi: first_roi,
                            scale: first_scale
                        },
                        BatchOp::RegionScaled {
                            roi: request_roi,
                            scale: request_scale
                        }
                    ) if first_roi == request_roi && first_scale == request_scale
                )
        })
}

pub(super) fn can_decode_requests_as_repeated_full_grayscale_batch(
    requests: &[QueuedRequest],
) -> bool {
    let Some((first, rest)) = requests.split_first() else {
        return false;
    };
    !rest.is_empty()
        && rest
            .iter()
            .all(|request| can_decode_as_repeated_full_grayscale_batch(first, request))
}

pub(super) fn can_decode_requests_as_repeated_full_color_batch(requests: &[QueuedRequest]) -> bool {
    let Some((first, rest)) = requests.split_first() else {
        return false;
    };
    !rest.is_empty()
        && rest
            .iter()
            .all(|request| can_decode_as_repeated_full_color_batch(first, request))
}
