// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_core::{BackendKind, BackendRequest, PixelFormat};

use crate::{profile, MetalBackendSession, Surface};

use super::heuristics::{
    can_decode_requests_as_repeated_full_color_batch,
    can_decode_requests_as_repeated_full_grayscale_batch, profile_route_label, BatchRoute,
    GroupedRequests,
};
use super::{
    decode_cpu_host_batch, decode_distinct_full_color_batch, decode_distinct_full_grayscale_batch,
    decode_distinct_region_scaled_direct_batch, decode_individual, decode_repeated_full_color,
    decode_repeated_full_grayscale, QueuedRequest, SessionState,
};

struct PendingBatchProfile {
    output_slots: Vec<usize>,
    backend: profile::MetalBatchProfileValue<BackendRequest>,
    fmt: profile::MetalBatchProfileValue<PixelFormat>,
}

impl PendingBatchProfile {
    fn new(requests: &[QueuedRequest]) -> Option<Self> {
        let mut budget =
            crate::batch_allocation::BatchMetadataBudget::new("J2K Metal batch profile context");
        let mut output_slots =
            match budget.try_vec(requests.len(), "J2K Metal batch profile output slots") {
                Ok(output_slots) => output_slots,
                Err(error) => {
                    j2k_profile::emit_profile_error("metal_batch_profile_context", &error);
                    return None;
                }
            };
        output_slots.extend(requests.iter().map(|request| request.output_slot));
        Some(Self {
            output_slots,
            backend: uniform_profile_value(requests.iter().map(|request| request.backend)),
            fmt: uniform_profile_value(requests.iter().map(|request| request.fmt)),
        })
    }
}

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
        session.completed[request.output_slot] = Some(decode_individual(&request, None));
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

pub(super) fn complete_device_failure(
    session: &mut SessionState,
    requests: &[QueuedRequest],
    source: &crate::Error,
) {
    session.submissions = session.submissions.saturating_add(1);
    for request in requests {
        session.completed[request.output_slot] = Some(Err(source.clone()));
    }
}

fn complete_device_result(
    session: &mut SessionState,
    requests: &[QueuedRequest],
    decoded: Option<Result<Vec<Surface>, crate::Error>>,
) -> bool {
    match decoded {
        None
        | Some(Err(crate::Error::MetalDirectFallback {
            reason: crate::MetalDirectFallbackReason::UnsupportedPlan,
            ..
        })) => false,
        Some(Ok(surfaces)) => {
            if !complete_batch_surfaces(session, requests, surfaces) {
                complete_device_failure(
                    session,
                    requests,
                    &crate::Error::MetalStateInvariant {
                        state: "J2K Metal batch output",
                        reason: "surface count does not match request count",
                    },
                );
            }
            true
        }
        Some(Err(error)) => {
            complete_device_failure(session, requests, &error);
            true
        }
    }
}

pub(super) fn process_batch(
    session: &mut SessionState,
    grouped: GroupedRequests,
    backend: Option<&MetalBackendSession>,
) {
    let GroupedRequests { route, requests } = grouped;
    let profile_enabled = profile::metal_profile_stages_enabled();
    let started = profile::profile_now(profile_enabled);
    let request_count = requests.len();
    let pending_profile = if profile_enabled {
        PendingBatchProfile::new(&requests)
    } else {
        None
    };

    process_batch_inner(session, route, requests, backend);

    if let Some(pending) = pending_profile {
        profile::emit_metal_batch_profile_row(
            "decode",
            &profile::MetalBatchProfileRow {
                slice: "decode_batch",
                stage: "execute",
                pipeline: "metal_cpu_hybrid",
                processor: "hybrid",
                route: profile_route_label(route),
                backend: pending.backend,
                fmt: pending.fmt,
                request_count,
                output_count: profile_completed_output_count(session, &pending.output_slots),
                elapsed_us: profile::elapsed_us(started),
                outcome: profile_completed_outcome(session, &pending.output_slots),
            },
        );
    }
}

fn process_batch_inner(
    session: &mut SessionState,
    route: BatchRoute,
    requests: Vec<QueuedRequest>,
    backend: Option<&MetalBackendSession>,
) {
    if route == BatchRoute::AutoRegionScaledDirectCpu {
        complete_cpu_host_fallback(session, requests);
        return;
    }

    // The grouped Metal routes below do not check precision; each request's
    // own route keeps components wider than Metal decodes exactly on the CPU
    // for Auto and rejects them for explicit Metal.
    if requests
        .iter()
        .any(|request| crate::routing::exceeds_metal_decode_precision(&request.input))
    {
        for request in requests {
            session.submissions = session.submissions.saturating_add(1);
            session.completed[request.output_slot] = Some(decode_individual(&request, backend));
        }
        return;
    }

    if can_decode_requests_as_repeated_full_grayscale_batch(&requests) {
        let decoded = decode_repeated_full_grayscale(&requests[0], requests.len(), backend);
        if complete_device_result(session, &requests, decoded) {
            return;
        }
    }

    if can_decode_requests_as_repeated_full_color_batch(&requests) {
        let decoded = decode_repeated_full_color(&requests[0], requests.len(), backend);
        if complete_device_result(session, &requests, decoded) {
            return;
        }
    }

    if requests.len() > 1 {
        let decoded = decode_distinct_full_grayscale_batch(&requests, backend);
        if complete_device_result(session, &requests, decoded) {
            return;
        }
    }

    if requests.len() > 1 {
        let decoded = decode_distinct_full_color_batch(&requests, backend);
        if complete_device_result(session, &requests, decoded) {
            return;
        }
    }

    if requests.len() > 1 {
        let decoded = decode_distinct_region_scaled_direct_batch(&requests, backend);
        if complete_device_result(session, &requests, decoded) {
            return;
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
        session.completed[request.output_slot] = Some(decode_individual(&request, backend));
    }
}

fn uniform_profile_value<T: Copy + Eq>(
    values: impl IntoIterator<Item = T>,
) -> profile::MetalBatchProfileValue<T> {
    let mut values = values.into_iter();
    let Some(first) = values.next() else {
        return profile::MetalBatchProfileValue::None;
    };
    if values.all(|value| value == first) {
        profile::MetalBatchProfileValue::Uniform(first)
    } else {
        profile::MetalBatchProfileValue::Mixed
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn requests(distinct: bool) -> Vec<QueuedRequest> {
        let shared = Arc::<[u8]>::from([1_u8]);
        (0..2)
            .map(|slot| {
                QueuedRequest::new(
                    if distinct {
                        Arc::from([u8::try_from(slot).unwrap()])
                    } else {
                        shared.clone()
                    },
                    PixelFormat::Gray8,
                    BackendRequest::Metal,
                    super::super::BatchOp::Full,
                    slot,
                )
            })
            .collect()
    }

    #[test]
    fn batch_completion_only_retries_pre_execution_capability_rejections() {
        for decoded in [
            None,
            Some(Err(crate::Error::MetalDirectFallback {
                message: "unsupported prepared plan".into(),
                reason: crate::MetalDirectFallbackReason::UnsupportedPlan,
            })),
        ] {
            let requests = requests(true);
            let mut session = SessionState::default();
            for _ in &requests {
                session
                    .completed
                    .try_reserve_slot("test batch outputs")
                    .unwrap();
                session.completed.allocate().unwrap();
            }
            assert!(!complete_device_result(&mut session, &requests, decoded,));
            assert_eq!(session.submissions, 0);
            assert!(session.completed.iter().all(Option::is_none));
        }
    }

    #[test]
    fn selected_batch_failures_and_wrong_output_counts_are_terminal() {
        use j2k_core::CodecError;
        let flags = |error: &crate::Error| {
            (
                error.is_truncated(),
                error.is_not_implemented(),
                error.is_unsupported(),
                error.is_buffer_error(),
                error.session_is_unusable(),
            )
        };
        let native =
            j2k_native::DecodeError::Format(j2k_native::FormatError::TooShort { need: 9, have: 3 });
        let count_error = crate::Error::MetalStateInvariant {
            state: "J2K Metal batch output",
            reason: "surface count does not match request count",
        };
        for distinct in [false, true] {
            for decoded in [
                Err(crate::error::native_decode_error(native)),
                Err(crate::Error::Decode(j2k::J2kError::NotImplemented(
                    j2k_core::NotImplemented {
                        what: "test feature",
                    },
                ))),
                Err(crate::Error::Buffer(
                    j2k_core::BufferError::OutputTooSmall {
                        required: 9,
                        have: 3,
                    },
                )),
                Err(crate::Error::UnsupportedMetalRequest {
                    reason: "unsupported image contract",
                }),
                Err(crate::Error::MetalKernel {
                    message: "synthetic dispatch failure".into(),
                }),
                Err(crate::Error::MetalDirectFallback {
                    message: "GPU rejected runtime input".into(),
                    reason: crate::MetalDirectFallbackReason::UnsupportedRuntimeInput,
                }),
                Err(crate::Error::MetalSupport {
                    message: "command creation failed".into(),
                    source: j2k_metal_support::MetalSupportError::CommandBufferUnavailable,
                }),
                Err(crate::Error::MetalRuntime {
                    message: "runtime creation failed".into(),
                }),
                Ok(Vec::new()),
            ] {
                let expected = decoded.as_ref().err().unwrap_or(&count_error);
                let expected_variant = core::mem::discriminant(expected);
                let expected_flags = flags(expected);
                let expected_message = expected.to_string();
                let requests = requests(distinct);
                let mut session = SessionState::default();
                for _ in &requests {
                    session
                        .completed
                        .try_reserve_slot("test batch outputs")
                        .unwrap();
                    session.completed.allocate().unwrap();
                }
                assert!(complete_device_result(
                    &mut session,
                    &requests,
                    Some(decoded)
                ));
                assert_eq!(session.submissions, 1);
                for result in session.completed.iter() {
                    let Some(Err(error)) = result else {
                        panic!("a selected failure must complete every request");
                    };
                    assert_eq!(core::mem::discriminant(error), expected_variant);
                    assert_eq!(flags(error), expected_flags);
                    assert_eq!(error.to_string(), expected_message);
                    match error {
                        crate::Error::NativeDecode { .. } => {
                            let opaque =
                                core::error::Error::source(error).expect("opaque native source");
                            assert!(opaque.is::<crate::NativeBackendError>());
                            assert_eq!(
                                opaque.source().and_then(
                                    |source| source.downcast_ref::<j2k_native::DecodeError>()
                                ),
                                Some(&native)
                            );
                        }
                        crate::Error::MetalSupport { .. } => assert_eq!(
                            core::error::Error::source(error)
                                .and_then(|source| source
                                    .downcast_ref::<j2k_metal_support::MetalSupportError>()),
                            Some(&j2k_metal_support::MetalSupportError::CommandBufferUnavailable),
                        ),
                        _ => {}
                    }
                }
            }
        }
    }

    #[test]
    fn uniform_profile_values_distinguish_empty_uniform_and_mixed_inputs() {
        assert_eq!(
            uniform_profile_value(std::iter::empty::<u8>()),
            profile::MetalBatchProfileValue::None
        );
        assert_eq!(
            uniform_profile_value([7_u8, 7]),
            profile::MetalBatchProfileValue::Uniform(7)
        );
        assert_eq!(
            uniform_profile_value([7_u8, 8]),
            profile::MetalBatchProfileValue::Mixed
        );
    }
}
