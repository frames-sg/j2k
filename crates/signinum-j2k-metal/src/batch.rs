// SPDX-License-Identifier: Apache-2.0

use std::{
    cell::OnceCell,
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::{Arc, Mutex},
};

use signinum_core::{BackendKind, BackendRequest, DeviceSubmission, Downscale, PixelFormat, Rect};
use signinum_j2k::{
    decode_tiles_into, decode_tiles_region_scaled_into, TileBatchOptions, TileDecodeJob,
    TileRegionScaledDecodeJob,
};

use crate::{profile, Error, J2kDecoder, MetalSession, Storage, Surface, SurfaceResidency};

const AUTO_REGION_SCALED_DIRECT_BATCH64_MIN_DIM: u32 = 512;
const AUTO_REGION_SCALED_DIRECT_BATCH64_MIN_COUNT: usize = 64;
const AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_DIM: u32 = 512;
const AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_COUNT: usize = 2;
const AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_DIM: u32 = 1024;
const AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_COUNT: usize = 16;
const REGION_SCALED_DIRECT_FORMATS: [PixelFormat; 5] = [
    PixelFormat::Gray8,
    PixelFormat::Gray16,
    PixelFormat::Rgb8,
    PixelFormat::Rgba8,
    PixelFormat::Rgb16,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchOp {
    Full,
    Region(Rect),
    Scaled(Downscale),
    RegionScaled { roi: Rect, scale: Downscale },
}

#[derive(Clone)]
struct QueuedRequest {
    input: Arc<[u8]>,
    fmt: PixelFormat,
    backend: BackendRequest,
    op: BatchOp,
    output_slot: usize,
    max_image_dim: OnceCell<Option<u32>>,
    input_fingerprint: OnceCell<u64>,
}

impl QueuedRequest {
    fn max_image_dim(&self) -> Option<u32> {
        *self.max_image_dim.get_or_init(|| {
            let decoder = J2kDecoder::new(self.input.as_ref()).ok()?;
            let dims = decoder.inner.info().dimensions;
            Some(dims.0.max(dims.1))
        })
    }

    fn input_fingerprint(&self) -> u64 {
        *self.input_fingerprint.get_or_init(|| {
            let mut hasher = DefaultHasher::new();
            self.input.len().hash(&mut hasher);
            if !self.input.is_empty() {
                let len = self.input.len();
                for offset in [0, len / 4, len / 2, len.saturating_sub(8)] {
                    let end = offset.saturating_add(8).min(len);
                    self.input[offset..end].hash(&mut hasher);
                }
            }
            hasher.finish()
        })
    }

    #[cfg(test)]
    fn max_image_dim_cache_filled_for_test(&self) -> bool {
        self.max_image_dim.get().is_some()
    }

    #[cfg(test)]
    fn input_fingerprint_cache_filled_for_test(&self) -> bool {
        self.input_fingerprint.get().is_some()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BatchRoute {
    Generic,
    AutoRegionScaledDirectCpu,
    AutoRegionScaledDirectMetal,
    AutoRepeatedRegionScaledDirectMetal,
}

fn profile_route_label(route: BatchRoute) -> &'static str {
    match route {
        BatchRoute::Generic => "generic",
        BatchRoute::AutoRegionScaledDirectCpu => "auto_region_scaled_direct_cpu",
        BatchRoute::AutoRegionScaledDirectMetal => "auto_region_scaled_direct_metal",
        BatchRoute::AutoRepeatedRegionScaledDirectMetal => {
            "auto_repeated_region_scaled_direct_metal"
        }
    }
}

struct GroupedRequests {
    route: BatchRoute,
    requests: Vec<QueuedRequest>,
}

impl GroupedRequests {
    fn generic(requests: Vec<QueuedRequest>) -> Self {
        Self {
            route: BatchRoute::Generic,
            requests,
        }
    }
}

#[doc(hidden)]
pub struct BenchmarkGroupedRequests {
    pub batch_count: usize,
    pub max_batch_len: usize,
}

#[doc(hidden)]
pub fn benchmark_group_region_scaled_requests(
    inputs: &[Arc<[u8]>],
    fmt: PixelFormat,
    backend: BackendRequest,
    roi: Rect,
    scale: Downscale,
) -> BenchmarkGroupedRequests {
    let queued = inputs
        .iter()
        .enumerate()
        .map(|(output_slot, input)| QueuedRequest {
            input: input.clone(),
            fmt,
            backend,
            op: BatchOp::RegionScaled { roi, scale },
            output_slot,
            max_image_dim: OnceCell::new(),
            input_fingerprint: OnceCell::new(),
        })
        .collect::<Vec<_>>();
    let batches = group_metal_requests(queued);
    BenchmarkGroupedRequests {
        batch_count: batches.len(),
        max_batch_len: batches
            .iter()
            .map(|batch| batch.requests.len())
            .max()
            .unwrap_or(0),
    }
}

#[derive(Default)]
pub(crate) struct SessionState {
    pub(crate) submissions: u64,
    queued: Vec<QueuedRequest>,
    completed: Vec<Option<Result<Surface, Error>>>,
}

#[derive(Clone, Default)]
pub(crate) struct SharedSession(pub(crate) Arc<Mutex<SessionState>>);

pub struct MetalSubmission {
    session: SharedSession,
    slot: usize,
}

impl DeviceSubmission for MetalSubmission {
    type Output = Surface;
    type Error = Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        let mut session = self.session.0.lock().expect("J2K Metal session");
        flush_if_needed(&mut session);
        take_surface(&mut session, self.slot)
    }
}

pub(crate) fn queue_tile_request(
    session: &mut MetalSession,
    input: &[u8],
    fmt: PixelFormat,
    backend: BackendRequest,
    op: BatchOp,
) -> MetalSubmission {
    queue_tile_request_shared(session, Arc::<[u8]>::from(input), fmt, backend, op)
}

pub(crate) fn queue_tile_request_shared(
    session: &mut MetalSession,
    input: Arc<[u8]>,
    fmt: PixelFormat,
    backend: BackendRequest,
    op: BatchOp,
) -> MetalSubmission {
    let mut state = session.shared.0.lock().expect("J2K Metal session");
    let slot = state.completed.len();
    state.completed.push(None);
    state.queued.push(QueuedRequest {
        input,
        fmt,
        backend,
        op,
        output_slot: slot,
        max_image_dim: OnceCell::new(),
        input_fingerprint: OnceCell::new(),
    });
    MetalSubmission {
        session: session.shared.clone(),
        slot,
    }
}

fn flush_if_needed(session: &mut SessionState) {
    if session.queued.is_empty() {
        return;
    }

    let profile_enabled = profile::metal_profile_stages_enabled();
    let queued = std::mem::take(&mut session.queued);
    let request_count = queued.len();
    let group_started = profile::profile_now(profile_enabled);
    let batches = group_metal_requests(queued);
    if profile_enabled {
        profile::emit_metal_batch_profile_row(
            "decode",
            &profile::MetalBatchProfileRow {
                slice: "decode_batch",
                stage: "group",
                pipeline: "metal_cpu_hybrid",
                processor: "scheduler",
                route: "all",
                backend: "mixed",
                fmt: "mixed",
                request_count,
                output_count: batches.len(),
                elapsed_us: profile::elapsed_us(group_started),
                outcome: "grouped",
            },
        );
    }

    for batch in batches {
        process_batch(session, batch);
    }
}

fn group_metal_requests(queued: Vec<QueuedRequest>) -> Vec<GroupedRequests> {
    coalesce_cpu_host_batches(coalesce_distinct_region_scaled_direct_metal_requests(
        coalesce_distinct_full_color_metal_requests(
            coalesce_distinct_full_grayscale_metal_requests(group_repeated_full_metal_requests(
                queued,
            )),
        ),
    ))
}

fn group_repeated_full_metal_requests(queued: Vec<QueuedRequest>) -> Vec<GroupedRequests> {
    let mut batches: Vec<GroupedRequests> = Vec::new();
    for request in queued {
        if let Some(batch) = batches.iter_mut().find(|batch| {
            batch.route == BatchRoute::Generic
                && can_decode_as_repeated_full_metal_batch(&batch.requests[0], &request)
        }) {
            batch.requests.push(request);
        } else {
            batches.push(GroupedRequests::generic(vec![request]));
        }
    }
    batches
}

fn coalesce_distinct_full_grayscale_metal_requests(
    repeated_batches: Vec<GroupedRequests>,
) -> Vec<GroupedRequests> {
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
                PixelFormat::Gray8 => gray8.push(request),
                PixelFormat::Gray16 => gray16.push(request),
                _ => unreachable!("candidate pixel format is restricted above"),
            }
        } else {
            batches.push(batch);
        }
    }

    push_coalesced_or_single(&mut batches, gray8);
    push_coalesced_or_single(&mut batches, gray16);
    batches
}

fn coalesce_distinct_region_scaled_direct_metal_requests(
    repeated_batches: Vec<GroupedRequests>,
) -> Vec<GroupedRequests> {
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
            let format_idx = region_scaled_direct_format_index(request.fmt)
                .expect("candidate pixel format is restricted above");
            match request.backend {
                BackendRequest::Metal => metal_by_format[format_idx].push(request),
                BackendRequest::Auto => auto_by_format[format_idx].push(request),
                _ => unreachable!("candidate backend is restricted above"),
            }
        } else {
            batches.push(batch);
        }
    }

    for requests in metal_by_format {
        push_coalesced_or_single(&mut batches, requests);
    }
    for requests in auto_by_format {
        push_auto_region_scaled_direct_batches(&mut batches, requests);
    }
    batches
}

fn push_coalesced_or_single(batches: &mut Vec<GroupedRequests>, requests: Vec<QueuedRequest>) {
    push_coalesced_or_single_with_route(batches, requests, BatchRoute::Generic);
}

fn push_coalesced_or_single_with_route(
    batches: &mut Vec<GroupedRequests>,
    requests: Vec<QueuedRequest>,
    route: BatchRoute,
) {
    if requests.is_empty() {
        return;
    }
    if requests.len() == 1 {
        batches.extend(requests.into_iter().map(|request| GroupedRequests {
            route,
            requests: vec![request],
        }));
    } else {
        batches.push(GroupedRequests { route, requests });
    }
}

fn push_auto_region_scaled_direct_batches(
    batches: &mut Vec<GroupedRequests>,
    requests: Vec<QueuedRequest>,
) {
    let Some(classification) = auto_region_scaled_direct_metal_classification(&requests) else {
        push_coalesced_or_single_with_route(
            batches,
            requests,
            BatchRoute::AutoRegionScaledDirectCpu,
        );
        return;
    };

    let mut metal_requests = Vec::new();
    let mut cpu_requests = Vec::new();
    for request in requests {
        if request
            .max_image_dim()
            .is_some_and(|max_dim| max_dim >= classification.min_dim)
        {
            metal_requests.push(request);
        } else {
            cpu_requests.push(request);
        }
    }
    push_coalesced_or_single_with_route(batches, metal_requests, classification.route);
    push_coalesced_or_single_with_route(
        batches,
        cpu_requests,
        BatchRoute::AutoRegionScaledDirectCpu,
    );
}

#[allow(clippy::similar_names)]
fn coalesce_distinct_full_color_metal_requests(
    repeated_batches: Vec<GroupedRequests>,
) -> Vec<GroupedRequests> {
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
                PixelFormat::Rgb8 => rgb8.push(request),
                PixelFormat::Rgba8 => rgba8.push(request),
                PixelFormat::Rgb16 => rgb16.push(request),
                _ => unreachable!("candidate pixel format is restricted above"),
            }
        } else {
            batches.push(batch);
        }
    }

    push_coalesced_or_single(&mut batches, rgb8);
    push_coalesced_or_single(&mut batches, rgba8);
    push_coalesced_or_single(&mut batches, rgb16);
    batches
}

fn coalesce_cpu_host_batches(batches: Vec<GroupedRequests>) -> Vec<GroupedRequests> {
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
                existing.push(request);
            } else {
                cpu_groups.push(vec![request]);
            }
        } else {
            coalesced.push(batch);
        }
    }
    coalesced.extend(cpu_groups.into_iter().map(GroupedRequests::generic));
    coalesced
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

fn same_input_bytes(first: &QueuedRequest, next: &QueuedRequest) -> bool {
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

fn is_repeated_full_grayscale_candidate(request: &QueuedRequest) -> bool {
    matches!(request.op, BatchOp::Full)
        && matches!(request.fmt, PixelFormat::Gray8 | PixelFormat::Gray16)
        && matches!(
            request.backend,
            BackendRequest::Auto | BackendRequest::Metal
        )
}

fn is_repeated_full_color_candidate(request: &QueuedRequest) -> bool {
    matches!(request.op, BatchOp::Full)
        && matches!(
            request.fmt,
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
        )
        && request.backend == BackendRequest::Metal
}

fn is_distinct_full_grayscale_metal_candidate(request: &QueuedRequest) -> bool {
    matches!(request.op, BatchOp::Full)
        && matches!(request.fmt, PixelFormat::Gray8 | PixelFormat::Gray16)
        && request.backend == BackendRequest::Metal
}

fn is_distinct_full_color_metal_candidate(request: &QueuedRequest) -> bool {
    matches!(request.op, BatchOp::Full)
        && matches!(
            request.fmt,
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16
        )
        && request.backend == BackendRequest::Metal
}

fn is_region_scaled_direct_batch_candidate(request: &QueuedRequest) -> bool {
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

fn should_auto_use_metal_for_region_scaled_direct_batch(requests: &[QueuedRequest]) -> bool {
    auto_region_scaled_direct_metal_min_dim(requests).is_some()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AutoRegionScaledDirectMetalClassification {
    min_dim: u32,
    route: BatchRoute,
}

fn auto_region_scaled_direct_metal_min_dim(requests: &[QueuedRequest]) -> Option<u32> {
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

fn can_decode_requests_as_repeated_region_scaled_batch(requests: &[QueuedRequest]) -> bool {
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

fn can_decode_requests_as_repeated_full_grayscale_batch(requests: &[QueuedRequest]) -> bool {
    let Some((first, rest)) = requests.split_first() else {
        return false;
    };
    !rest.is_empty()
        && rest
            .iter()
            .all(|request| can_decode_as_repeated_full_grayscale_batch(first, request))
}

fn can_decode_requests_as_repeated_full_color_batch(requests: &[QueuedRequest]) -> bool {
    let Some((first, rest)) = requests.split_first() else {
        return false;
    };
    !rest.is_empty()
        && rest
            .iter()
            .all(|request| can_decode_as_repeated_full_color_batch(first, request))
}

fn complete_cpu_host_fallback(session: &mut SessionState, requests: Vec<QueuedRequest>) {
    if requests.len() > 1 {
        if let Some(Ok(surfaces)) = decode_cpu_host_batch(&requests) {
            if surfaces.len() == requests.len() {
                session.submissions = session.submissions.saturating_add(1);
                for (request, surface) in requests.into_iter().zip(surfaces) {
                    session.completed[request.output_slot] = Some(Ok(surface));
                }
                return;
            }
        }
    }
    for request in requests {
        session.submissions = session.submissions.saturating_add(1);
        session.completed[request.output_slot] = Some(decode_individual(&request));
    }
}

fn process_batch(session: &mut SessionState, grouped: GroupedRequests) {
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
            if surfaces.len() == requests.len() {
                session.submissions = session.submissions.saturating_add(1);
                for (request, surface) in requests.into_iter().zip(surfaces) {
                    session.completed[request.output_slot] = Some(Ok(surface));
                }
                return;
            }
        }
        complete_cpu_host_fallback(session, requests);
        return;
    }

    if can_decode_requests_as_repeated_full_grayscale_batch(&requests) {
        if let Some(Ok(surfaces)) = decode_repeated_full_grayscale(&requests[0], requests.len()) {
            if surfaces.len() == requests.len() {
                session.submissions = session.submissions.saturating_add(1);
                for (request, surface) in requests.into_iter().zip(surfaces) {
                    session.completed[request.output_slot] = Some(Ok(surface));
                }
                return;
            }
        }
    }

    if can_decode_requests_as_repeated_full_color_batch(&requests) {
        if let Some(Ok(surfaces)) = decode_repeated_full_color(&requests[0], requests.len()) {
            if surfaces.len() == requests.len() {
                session.submissions = session.submissions.saturating_add(1);
                for (request, surface) in requests.into_iter().zip(surfaces) {
                    session.completed[request.output_slot] = Some(Ok(surface));
                }
                return;
            }
        }
    }

    if requests.len() > 1 {
        if let Some(Ok(surfaces)) = decode_distinct_full_grayscale_batch(&requests) {
            if surfaces.len() == requests.len() {
                session.submissions = session.submissions.saturating_add(1);
                for (request, surface) in requests.into_iter().zip(surfaces) {
                    session.completed[request.output_slot] = Some(Ok(surface));
                }
                return;
            }
        }
    }

    if requests.len() > 1 {
        if let Some(result) = decode_distinct_full_color_batch(&requests) {
            match result {
                Ok(surfaces) if surfaces.len() == requests.len() => {
                    session.submissions = session.submissions.saturating_add(1);
                    for (request, surface) in requests.into_iter().zip(surfaces) {
                        session.completed[request.output_slot] = Some(Ok(surface));
                    }
                    return;
                }
                Ok(_) | Err(_) => {}
            }
        }
    }

    if requests.len() > 1 {
        if let Some(Ok(surfaces)) = decode_distinct_region_scaled_direct_batch(&requests) {
            if surfaces.len() == requests.len() {
                session.submissions = session.submissions.saturating_add(1);
                for (request, surface) in requests.into_iter().zip(surfaces) {
                    session.completed[request.output_slot] = Some(Ok(surface));
                }
                return;
            }
        }
    }

    if requests.len() > 1 {
        if let Some(Ok(surfaces)) = decode_cpu_host_batch(&requests) {
            if surfaces.len() == requests.len() {
                session.submissions = session.submissions.saturating_add(1);
                for (request, surface) in requests.into_iter().zip(surfaces) {
                    session.completed[request.output_slot] = Some(Ok(surface));
                }
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

fn decode_cpu_host_batch(requests: &[QueuedRequest]) -> Option<Result<Vec<Surface>, Error>> {
    decode_cpu_full_batch(requests).or_else(|| decode_cpu_region_scaled_batch(requests))
}

fn decode_cpu_full_batch(requests: &[QueuedRequest]) -> Option<Result<Vec<Surface>, Error>> {
    let first = requests.first()?;
    if requests.len() <= 1
        || !requests
            .iter()
            .all(|request| is_cpu_host_full_batch_candidate(request) && request.fmt == first.fmt)
    {
        return None;
    }

    Some(decode_cpu_full_batch_inner(requests, first.fmt))
}

fn is_cpu_host_full_batch_candidate(request: &QueuedRequest) -> bool {
    matches!(request.op, BatchOp::Full)
        && matches!(request.backend, BackendRequest::Cpu | BackendRequest::Auto)
}

fn decode_cpu_full_batch_inner(
    requests: &[QueuedRequest],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    let mut dims = Vec::with_capacity(requests.len());
    let mut outputs = Vec::with_capacity(requests.len());
    for request in requests {
        let decoder = J2kDecoder::new(request.input.as_ref())?;
        let tile_dims = decoder.inner.info().dimensions;
        let stride = tile_dims.0 as usize * fmt.bytes_per_pixel();
        dims.push(tile_dims);
        outputs.push(vec![0_u8; stride * tile_dims.1 as usize]);
    }

    {
        let mut jobs = requests
            .iter()
            .zip(dims.iter())
            .zip(outputs.iter_mut())
            .map(|((request, dims), out)| TileDecodeJob {
                input: request.input.as_ref(),
                out: out.as_mut_slice(),
                stride: dims.0 as usize * fmt.bytes_per_pixel(),
            })
            .collect::<Vec<_>>();
        decode_tiles_into(&mut jobs, fmt, TileBatchOptions::default())
            .map_err(|err| Error::Decode(err.source))?;
    }

    Ok(outputs
        .into_iter()
        .zip(dims)
        .map(|(bytes, dimensions)| host_surface(bytes, dimensions, fmt))
        .collect())
}

fn decode_cpu_region_scaled_batch(
    requests: &[QueuedRequest],
) -> Option<Result<Vec<Surface>, Error>> {
    let first = requests.first()?;
    if requests.len() <= 1
        || !requests.iter().all(|request| {
            is_cpu_host_region_scaled_batch_candidate(request) && request.fmt == first.fmt
        })
    {
        return None;
    }

    Some(decode_cpu_region_scaled_batch_inner(requests, first.fmt))
}

fn is_cpu_host_region_scaled_batch_candidate(request: &QueuedRequest) -> bool {
    matches!(request.op, BatchOp::RegionScaled { .. })
        && matches!(request.backend, BackendRequest::Cpu | BackendRequest::Auto)
}

fn decode_cpu_region_scaled_batch_inner(
    requests: &[QueuedRequest],
    fmt: PixelFormat,
) -> Result<Vec<Surface>, Error> {
    let mut dims = Vec::with_capacity(requests.len());
    let mut outputs = Vec::with_capacity(requests.len());
    for request in requests {
        let BatchOp::RegionScaled { roi, scale } = request.op else {
            unreachable!("candidate op is restricted above");
        };
        let dimensions = roi.scaled_covering(scale);
        let stride = dimensions.w as usize * fmt.bytes_per_pixel();
        dims.push((dimensions.w, dimensions.h));
        outputs.push(vec![0_u8; stride * dimensions.h as usize]);
    }

    {
        let mut jobs = requests
            .iter()
            .zip(outputs.iter_mut())
            .map(|(request, out)| {
                let BatchOp::RegionScaled { roi, scale } = request.op else {
                    unreachable!("candidate op is restricted above");
                };
                let dimensions = roi.scaled_covering(scale);
                TileRegionScaledDecodeJob {
                    input: request.input.as_ref(),
                    out: out.as_mut_slice(),
                    stride: dimensions.w as usize * fmt.bytes_per_pixel(),
                    roi,
                    scale,
                }
            })
            .collect::<Vec<_>>();
        decode_tiles_region_scaled_into(&mut jobs, fmt, TileBatchOptions::default())
            .map_err(|err| Error::Decode(err.source))?;
    }

    Ok(outputs
        .into_iter()
        .zip(dims)
        .map(|(bytes, dimensions)| host_surface(bytes, dimensions, fmt))
        .collect())
}

fn host_surface(bytes: Vec<u8>, dimensions: (u32, u32), fmt: PixelFormat) -> Surface {
    Surface {
        backend: BackendKind::Cpu,
        residency: SurfaceResidency::Host,
        dimensions,
        fmt,
        pitch_bytes: dimensions.0 as usize * fmt.bytes_per_pixel(),
        byte_offset: 0,
        storage: Storage::Host(bytes),
    }
}

fn decode_repeated_full_grayscale(
    request: &QueuedRequest,
    count: usize,
) -> Option<Result<Vec<Surface>, Error>> {
    if !is_repeated_full_grayscale_candidate(request) || count <= 1 {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        let result =
            J2kDecoder::new(request.input.as_ref()).and_then(|mut decoder| match request.backend {
                BackendRequest::Auto => {
                    decoder.decode_repeated_grayscale_auto_to_device(request.fmt, count)
                }
                BackendRequest::Metal => {
                    decoder.decode_repeated_grayscale_direct_to_device(request.fmt, count)
                }
                _ => unreachable!("candidate backend is restricted above"),
            });
        Some(result)
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

fn decode_repeated_full_color(
    request: &QueuedRequest,
    count: usize,
) -> Option<Result<Vec<Surface>, Error>> {
    if !is_repeated_full_color_candidate(request) || count <= 1 {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        let result = J2kDecoder::new(request.input.as_ref()).and_then(|mut decoder| {
            decoder.decode_repeated_color_direct_to_device(request.fmt, count)
        });
        Some(result)
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

fn decode_distinct_full_grayscale_batch(
    requests: &[QueuedRequest],
) -> Option<Result<Vec<Surface>, Error>> {
    let first = requests.first()?;
    if requests.len() <= 1
        || !requests.iter().all(|request| {
            is_distinct_full_grayscale_metal_candidate(request) && request.fmt == first.fmt
        })
    {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        let inputs = requests
            .iter()
            .map(|request| request.input.clone())
            .collect::<Vec<_>>();
        Some(crate::decode_full_grayscale_batch_direct_to_device(
            &inputs, first.fmt,
        ))
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

fn decode_distinct_full_color_batch(
    requests: &[QueuedRequest],
) -> Option<Result<Vec<Surface>, Error>> {
    let first = requests.first()?;
    if requests.len() <= 1
        || !requests.iter().all(|request| {
            is_distinct_full_color_metal_candidate(request) && request.fmt == first.fmt
        })
    {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        let inputs = requests
            .iter()
            .map(|request| request.input.clone())
            .collect::<Vec<_>>();
        Some(crate::decode_full_color_batch_direct_to_device(
            &inputs, first.fmt,
        ))
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

fn decode_distinct_region_scaled_direct_batch(
    requests: &[QueuedRequest],
) -> Option<Result<Vec<Surface>, Error>> {
    decode_distinct_region_scaled_direct_batch_inner(requests, false)
}

fn decode_repeated_region_scaled_direct_batch_prechecked(
    requests: &[QueuedRequest],
) -> Option<Result<Vec<Surface>, Error>> {
    let first = requests.first()?;
    if requests.len() <= 1 {
        return None;
    }
    let BatchOp::RegionScaled { roi, scale } = first.op else {
        return None;
    };

    #[cfg(target_os = "macos")]
    {
        let result = match first.fmt {
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 => {
                crate::hybrid::decode_repeated_region_scaled_color_batch_direct_to_device(
                    first.input.as_ref(),
                    roi,
                    scale,
                    first.fmt,
                    requests.len(),
                )
            }
            _ => return None,
        };
        Some(result)
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

fn decode_distinct_region_scaled_direct_batch_prechecked(
    requests: &[QueuedRequest],
) -> Option<Result<Vec<Surface>, Error>> {
    decode_distinct_region_scaled_direct_batch_inner(requests, true)
}

fn decode_distinct_region_scaled_direct_batch_inner(
    requests: &[QueuedRequest],
    auto_metal_prechecked: bool,
) -> Option<Result<Vec<Surface>, Error>> {
    let first = requests.first()?;
    if requests.len() <= 1
        || !requests.iter().all(|request| {
            is_region_scaled_direct_batch_candidate(request)
                && request.fmt == first.fmt
                && request.backend == first.backend
        })
    {
        return None;
    }
    if first.backend == BackendRequest::Auto
        && !auto_metal_prechecked
        && !should_auto_use_metal_for_region_scaled_direct_batch(requests)
    {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        let request_specs = requests
            .iter()
            .map(|request| match request.op {
                BatchOp::RegionScaled { roi, scale } => (request.input.clone(), roi, scale),
                _ => unreachable!("candidate op is restricted above"),
            })
            .collect::<Vec<_>>();
        let result = match first.fmt {
            PixelFormat::Gray8 | PixelFormat::Gray16 => {
                crate::hybrid::decode_region_scaled_grayscale_batch_direct_to_device(
                    &request_specs,
                    first.fmt,
                )
            }
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 => {
                crate::hybrid::decode_region_scaled_color_batch_direct_to_device(
                    &request_specs,
                    first.fmt,
                )
            }
            _ => unreachable!("candidate pixel format is restricted above"),
        };
        Some(result)
    }

    #[cfg(not(target_os = "macos"))]
    {
        None
    }
}

fn decode_individual(request: &QueuedRequest) -> Result<Surface, Error> {
    let mut decoder = J2kDecoder::new(request.input.as_ref())?;
    match request.op {
        BatchOp::Full => decoder.decode_to_surface_impl(request.fmt, request.backend),
        BatchOp::Region(roi) => {
            decoder.decode_region_to_surface_impl(request.fmt, roi, request.backend)
        }
        BatchOp::Scaled(scale) => {
            decoder.decode_scaled_to_surface_impl(request.fmt, scale, request.backend)
        }
        BatchOp::RegionScaled { roi, scale } => {
            decoder.decode_region_scaled_to_surface_impl(request.fmt, roi, scale, request.backend)
        }
    }
}

fn take_surface(session: &mut SessionState, slot: usize) -> Result<Surface, Error> {
    session
        .completed
        .get_mut(slot)
        .and_then(Option::take)
        .ok_or_else(|| Error::MetalKernel {
            message: format!("missing queued J2K Metal surface for slot {slot}"),
        })?
}

#[cfg(test)]
mod tests {
    use super::*;

    fn auto_rgb_region_scaled_request(input: Arc<[u8]>) -> QueuedRequest {
        QueuedRequest {
            input,
            fmt: PixelFormat::Rgb8,
            backend: BackendRequest::Auto,
            op: BatchOp::RegionScaled {
                roi: Rect {
                    x: 128,
                    y: 128,
                    w: 512,
                    h: 256,
                },
                scale: Downscale::Quarter,
            },
            output_slot: 0,
            max_image_dim: OnceCell::new(),
            input_fingerprint: OnceCell::new(),
        }
    }

    fn auto_rgb_region_scaled_request_with_max_dim(
        input: Arc<[u8]>,
        max_image_dim: u32,
    ) -> QueuedRequest {
        let request = auto_rgb_region_scaled_request(input);
        request.max_image_dim.set(Some(max_image_dim)).ok();
        request
    }

    #[test]
    fn auto_region_scaled_rgb_threshold_requires_repeated_inputs() {
        let requests = (0..AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_COUNT)
            .map(|idx| auto_rgb_region_scaled_request(Arc::from([idx as u8])))
            .collect::<Vec<_>>();

        assert!(!can_decode_requests_as_repeated_region_scaled_batch(
            &requests
        ));
        assert_eq!(
            auto_region_scaled_direct_metal_min_dim(&requests),
            None,
            "distinct RGB ROI+scaled Auto batches must stay CPU until hybrid wins for distinct inputs"
        );

        let shared = Arc::<[u8]>::from([1_u8]);
        let repeated = (0..AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_COUNT)
            .map(|_| auto_rgb_region_scaled_request(shared.clone()))
            .collect::<Vec<_>>();
        assert!(can_decode_requests_as_repeated_region_scaled_batch(
            &repeated
        ));
    }

    #[test]
    fn auto_region_scaled_repeated_rgb_uses_measured_batch_two_metal_threshold() {
        let shared = Arc::<[u8]>::from([1_u8]);
        let repeated = (0..2)
            .map(|_| auto_rgb_region_scaled_request_with_max_dim(shared.clone(), 512))
            .collect::<Vec<_>>();

        assert_eq!(
            auto_region_scaled_direct_metal_min_dim(&repeated),
            Some(512),
            "measured repeated RGB ROI+scaled batches should route to Metal from batch 2 at 512px"
        );

        let single = vec![auto_rgb_region_scaled_request_with_max_dim(shared, 512)];
        assert_eq!(auto_region_scaled_direct_metal_min_dim(&single), None);
    }

    #[test]
    fn queued_request_caches_image_dimension_probe() {
        let request = auto_rgb_region_scaled_request(Arc::from([0_u8]));

        assert!(!request.max_image_dim_cache_filled_for_test());
        assert_eq!(request.max_image_dim(), None);
        assert!(request.max_image_dim_cache_filled_for_test());
        assert_eq!(request.max_image_dim(), None);
    }

    #[test]
    fn repeated_input_check_uses_pointer_identity_before_fingerprint() {
        let shared = Arc::<[u8]>::from([1_u8, 2, 3, 4]);
        let first = auto_rgb_region_scaled_request(shared.clone());
        let next = auto_rgb_region_scaled_request(shared);

        assert!(same_input_bytes(&first, &next));
        assert!(!first.input_fingerprint_cache_filled_for_test());
        assert!(!next.input_fingerprint_cache_filled_for_test());
    }

    #[test]
    fn auto_region_scaled_grouping_preserves_repeated_rgb_metal_decision() {
        let shared = Arc::<[u8]>::from([1_u8, 2, 3, 4]);
        let requests = (0..AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_COUNT)
            .map(|_| {
                auto_rgb_region_scaled_request_with_max_dim(
                    shared.clone(),
                    AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_DIM,
                )
            })
            .collect::<Vec<_>>();

        let grouped = group_metal_requests(requests);

        assert_eq!(grouped.len(), 1);
        assert_eq!(
            grouped[0].route,
            BatchRoute::AutoRepeatedRegionScaledDirectMetal
        );
        assert_eq!(
            grouped[0].requests.len(),
            AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_COUNT
        );
        assert!(
            grouped[0]
                .requests
                .iter()
                .all(|request| !request.input_fingerprint_cache_filled_for_test()),
            "shared repeated inputs should be classified by Arc identity without fingerprinting"
        );
    }

    #[test]
    fn auto_region_scaled_distinct_rgb_grouping_preserves_cpu_decision() {
        let requests = (0..AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_COUNT)
            .map(|idx| {
                auto_rgb_region_scaled_request_with_max_dim(
                    Arc::from([idx as u8]),
                    AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_DIM,
                )
            })
            .collect::<Vec<_>>();

        let grouped = group_metal_requests(requests);

        assert_eq!(grouped.len(), 1);
        assert_eq!(grouped[0].route, BatchRoute::AutoRegionScaledDirectCpu);
        assert_eq!(
            grouped[0].requests.len(),
            AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_COUNT
        );
    }

    #[test]
    fn profile_route_labels_are_stable_for_decode_batch_slices() {
        assert_eq!(profile_route_label(BatchRoute::Generic), "generic");
        assert_eq!(
            profile_route_label(BatchRoute::AutoRegionScaledDirectCpu),
            "auto_region_scaled_direct_cpu"
        );
        assert_eq!(
            profile_route_label(BatchRoute::AutoRegionScaledDirectMetal),
            "auto_region_scaled_direct_metal"
        );
        assert_eq!(
            profile_route_label(BatchRoute::AutoRepeatedRegionScaledDirectMetal),
            "auto_repeated_region_scaled_direct_metal"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn auto_region_scaled_prechecked_error_does_not_retry_generic_direct_path() {
        let _guard = crate::hybrid::region_scaled_color_plan_test_lock_for_test();
        crate::hybrid::reset_region_scaled_color_plan_builds_for_test();
        let shared = Arc::<[u8]>::from([1_u8, 2, 3, 4]);
        let requests = (0..AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_COUNT)
            .map(|slot| {
                let mut request = auto_rgb_region_scaled_request_with_max_dim(
                    shared.clone(),
                    AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_DIM,
                );
                request.output_slot = slot;
                request
            })
            .collect::<Vec<_>>();
        let mut session = SessionState {
            submissions: 0,
            queued: Vec::new(),
            completed: (0..requests.len()).map(|_| None).collect(),
        };

        process_batch(
            &mut session,
            GroupedRequests {
                route: BatchRoute::AutoRepeatedRegionScaledDirectMetal,
                requests,
            },
        );

        assert_eq!(
            crate::hybrid::region_scaled_color_plan_builds_for_test(),
            1,
            "failed prechecked Auto Metal routing should fall back to CPU without retrying generic direct Metal"
        );
        assert!(
            session
                .completed
                .iter()
                .all(|result| matches!(result, Some(Err(_)))),
            "invalid inputs should be surfaced on every fallback request"
        );
    }
}
