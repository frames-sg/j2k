// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    cell::OnceCell,
    collections::hash_map::DefaultHasher,
    hash::{Hash, Hasher},
    sync::{Arc, Mutex, MutexGuard},
};

use j2k_core::{BackendRequest, DeviceSubmission, Downscale, PixelFormat, Rect};

use crate::{profile, Error, J2kDecoder, MetalSession, Surface};

mod cpu;
mod execute;
mod heuristics;
use self::cpu::decode_cpu_host_batch;
use self::execute::process_batch;
#[cfg(test)]
use self::heuristics::{
    auto_region_scaled_direct_metal_min_dim, can_decode_requests_as_repeated_region_scaled_batch,
    profile_route_label, same_input_bytes, BatchRoute, GroupedRequests,
    AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_COUNT, AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_DIM,
    AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_COUNT,
    AUTO_REGION_SCALED_DIRECT_REPEATED_RGB_MIN_DIM,
};
use self::heuristics::{
    group_metal_requests, is_distinct_full_color_metal_candidate,
    is_distinct_full_grayscale_metal_candidate, is_region_scaled_direct_batch_candidate,
    is_repeated_full_color_candidate, is_repeated_full_grayscale_candidate,
    should_auto_use_metal_for_region_scaled_direct_batch,
};

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

fn batch_scheduler_invariant(message: &'static str) -> Error {
    Error::MetalKernel {
        message: format!("internal J2K Metal batch scheduler error: {message}"),
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
) -> Result<BenchmarkGroupedRequests, Error> {
    let mut budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal benchmark grouping requests");
    let mut queued = budget.try_vec(inputs.len(), "J2K Metal benchmark queued requests")?;
    queued.extend(
        inputs
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
            }),
    );
    let batches = group_metal_requests(queued)?;
    Ok(BenchmarkGroupedRequests {
        batch_count: batches.len(),
        max_batch_len: batches
            .iter()
            .map(|batch| batch.requests.len())
            .max()
            .unwrap_or(0),
    })
}

#[derive(Default)]
pub(crate) struct SessionState {
    pub(crate) submissions: u64,
    queued: Vec<QueuedRequest>,
    completed: Vec<Option<Result<Surface, Error>>>,
}

#[derive(Clone, Default)]
pub(crate) struct SharedSession(pub(crate) Arc<Mutex<SessionState>>);

impl SharedSession {
    pub(crate) fn lock(&self) -> Result<MutexGuard<'_, SessionState>, Error> {
        self.0.lock().map_err(|_| Error::MetalStatePoisoned {
            state: "J2K Metal session",
        })
    }
}

pub struct MetalSubmission {
    session: SharedSession,
    slot: usize,
}

#[doc(hidden)]
impl DeviceSubmission for MetalSubmission {
    type Output = Surface;
    type Error = Error;

    fn wait(self) -> Result<Self::Output, Self::Error> {
        let mut session = self.session.lock()?;
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
) -> Result<MetalSubmission, Error> {
    queue_tile_request_shared(session, Arc::<[u8]>::from(input), fmt, backend, op)
}

pub(crate) fn queue_tile_request_shared(
    session: &mut MetalSession,
    input: Arc<[u8]>,
    fmt: PixelFormat,
    backend: BackendRequest,
    op: BatchOp,
) -> Result<MetalSubmission, Error> {
    queue_tile_request_shared_with_retained(session, input, fmt, backend, op, 0)
}

pub(crate) fn queue_tile_request_shared_with_retained(
    session: &mut MetalSession,
    input: Arc<[u8]>,
    fmt: PixelFormat,
    backend: BackendRequest,
    op: BatchOp,
    retained_submission_capacity: usize,
) -> Result<MetalSubmission, Error> {
    let mut state = session.shared.lock()?;
    crate::batch_allocation::try_reserve_for_push(
        &mut state.completed,
        "J2K Metal queued completion slots",
    )?;
    crate::batch_allocation::try_reserve_for_push(&mut state.queued, "J2K Metal queued requests")?;
    let aggregate =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal queued request state");
    aggregate.preflight(&[
        crate::batch_allocation::BatchMetadataRequest::of::<MetalSubmission>(
            retained_submission_capacity,
        ),
        crate::batch_allocation::BatchMetadataRequest::of::<QueuedRequest>(state.queued.capacity()),
        crate::batch_allocation::BatchMetadataRequest::of::<Option<Result<Surface, Error>>>(
            state.completed.capacity(),
        ),
    ])?;
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
    Ok(MetalSubmission {
        session: session.shared.clone(),
        slot,
    })
}

fn flush_if_needed(session: &mut SessionState) {
    if session.queued.is_empty() {
        return;
    }

    let profile_enabled = profile::metal_profile_stages_enabled();
    let queued = std::mem::take(&mut session.queued);
    let request_count = queued.len();
    let mut slot_budget =
        crate::batch_allocation::BatchMetadataBudget::new("J2K Metal grouping recovery slots");
    let mut output_slots =
        match slot_budget.try_vec(queued.len(), "J2K Metal grouping recovery output slots") {
            Ok(slots) => slots,
            Err(error) => {
                for request in queued {
                    session.completed[request.output_slot] = Some(Err(error.into()));
                }
                return;
            }
        };
    output_slots.extend(queued.iter().map(|request| request.output_slot));
    let group_started = profile::profile_now(profile_enabled);
    let batches = match group_metal_requests(queued) {
        Ok(batches) => batches,
        Err(error) => {
            for output_slot in output_slots {
                session.completed[output_slot] = Some(Err(error.into()));
            }
            return;
        }
    };
    drop(output_slots);
    if profile_enabled {
        profile::emit_metal_batch_profile_row(
            "decode",
            &profile::MetalBatchProfileRow {
                slice: "decode_batch",
                stage: "group",
                pipeline: "metal_cpu_hybrid",
                processor: "scheduler",
                route: "all",
                backend: profile::MetalBatchProfileValue::Mixed,
                fmt: profile::MetalBatchProfileValue::Mixed,
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
                _ => Err(batch_scheduler_invariant(
                    "repeated grayscale batch contains an unsupported backend",
                )),
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
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal distinct grayscale batch inputs",
        );
        let mut inputs =
            match budget.try_vec(requests.len(), "J2K Metal distinct grayscale input handles") {
                Ok(inputs) => inputs,
                Err(error) => return Some(Err(error.into())),
            };
        inputs.extend(requests.iter().map(|request| request.input.clone()));
        Some(crate::decoder::decode_full_grayscale_batch_direct_to_device(&inputs, first.fmt))
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
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal distinct color batch inputs",
        );
        let mut inputs =
            match budget.try_vec(requests.len(), "J2K Metal distinct color input handles") {
                Ok(inputs) => inputs,
                Err(error) => return Some(Err(error.into())),
            };
        inputs.extend(requests.iter().map(|request| request.input.clone()));
        Some(crate::decoder::decode_full_color_batch_direct_to_device(
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
    if !matches!(first.op, BatchOp::RegionScaled { .. }) {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        let BatchOp::RegionScaled { roi, scale } = first.op else {
            return Some(Err(batch_scheduler_invariant(
                "repeated direct batch is missing its region-scaled operation",
            )));
        };
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
        let mut budget = crate::batch_allocation::BatchMetadataBudget::new(
            "J2K Metal direct batch request specifications",
        );
        let mut request_specs = match budget.try_vec(
            requests.len(),
            "J2K Metal direct batch request specifications",
        ) {
            Ok(specs) => specs,
            Err(error) => return Some(Err(error.into())),
        };
        for request in requests {
            match request.op {
                BatchOp::RegionScaled { roi, scale } => {
                    request_specs.push((request.input.clone(), roi, scale));
                }
                _ => {
                    return Some(Err(batch_scheduler_invariant(
                        "direct region-scaled batch contains a non-region-scaled request",
                    )));
                }
            }
        }
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
            _ => Err(batch_scheduler_invariant(
                "direct region-scaled batch contains an unsupported pixel format",
            )),
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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "bounded test fixture index fits in u8"
    )]
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

        let grouped = group_metal_requests(requests).expect("group requests");

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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "bounded test fixture index fits in u8"
    )]
    fn auto_region_scaled_distinct_rgb_grouping_preserves_cpu_decision() {
        let requests = (0..AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_COUNT)
            .map(|idx| {
                auto_rgb_region_scaled_request_with_max_dim(
                    Arc::from([idx as u8]),
                    AUTO_REGION_SCALED_DIRECT_BATCH16_MIN_DIM,
                )
            })
            .collect::<Vec<_>>();

        let grouped = group_metal_requests(requests).expect("group requests");

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
