// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k_core::{BackendRequest, PixelFormat};

use crate::{Error, J2kDecoder, MetalBackendSession, MetalDecodeRequest, Surface};

use super::heuristics::{
    is_distinct_full_color_metal_candidate, is_distinct_full_grayscale_metal_candidate,
    is_region_scaled_direct_batch_candidate, is_repeated_full_color_candidate,
    is_repeated_full_grayscale_candidate, should_auto_use_metal_for_region_scaled_direct_batch,
};
use super::request::{batch_scheduler_invariant, BatchOp, QueuedRequest};

pub(super) fn decode_repeated_full_grayscale(
    request: &QueuedRequest,
    count: usize,
    backend: Option<&MetalBackendSession>,
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
                BackendRequest::Metal => decoder.decode_repeated_grayscale_direct_to_device_routed(
                    request.fmt,
                    count,
                    backend,
                ),
                _ => Err(batch_scheduler_invariant(
                    "repeated grayscale batch contains an unsupported backend",
                )),
            });
        Some(result)
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = backend;
        None
    }
}

pub(super) fn decode_repeated_full_color(
    request: &QueuedRequest,
    count: usize,
    backend: Option<&MetalBackendSession>,
) -> Option<Result<Vec<Surface>, Error>> {
    if !is_repeated_full_color_candidate(request) || count <= 1 {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        Some(
            J2kDecoder::new(request.input.as_ref()).and_then(|mut decoder| {
                decoder.decode_repeated_color_direct_to_device_routed(request.fmt, count, backend)
            }),
        )
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = backend;
        None
    }
}

pub(super) fn decode_distinct_full_grayscale_batch(
    requests: &[QueuedRequest],
    backend: Option<&MetalBackendSession>,
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
        let inputs = match collect_inputs(requests, "J2K Metal distinct grayscale input handles") {
            Ok(inputs) => inputs,
            Err(error) => return Some(Err(error)),
        };
        Some(
            crate::decoder::decode_full_grayscale_batch_direct_to_device_routed(
                &inputs, first.fmt, backend,
            ),
        )
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = backend;
        None
    }
}

pub(super) fn decode_distinct_full_color_batch(
    requests: &[QueuedRequest],
    backend: Option<&MetalBackendSession>,
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
        let inputs = match collect_inputs(requests, "J2K Metal distinct color input handles") {
            Ok(inputs) => inputs,
            Err(error) => return Some(Err(error)),
        };
        Some(
            crate::decoder::decode_full_color_batch_direct_to_device_routed(
                &inputs, first.fmt, backend,
            ),
        )
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = backend;
        None
    }
}

#[cfg(target_os = "macos")]
fn collect_inputs(requests: &[QueuedRequest], what: &'static str) -> Result<Vec<Arc<[u8]>>, Error> {
    let mut budget = crate::batch_allocation::BatchMetadataBudget::new(what);
    let mut inputs = budget.try_vec(requests.len(), what)?;
    inputs.extend(requests.iter().map(|request| request.input.clone()));
    Ok(inputs)
}

pub(super) fn decode_distinct_region_scaled_direct_batch(
    requests: &[QueuedRequest],
    backend: Option<&MetalBackendSession>,
) -> Option<Result<Vec<Surface>, Error>> {
    decode_distinct_region_scaled_direct_batch_inner(requests, false, backend)
}

pub(super) fn decode_repeated_region_scaled_direct_batch_prechecked(
    requests: &[QueuedRequest],
    backend: Option<&MetalBackendSession>,
) -> Option<Result<Vec<Surface>, Error>> {
    let first = requests.first()?;
    if requests.len() <= 1 || !matches!(first.op, BatchOp::RegionScaled { .. }) {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        let BatchOp::RegionScaled { roi, scale } = first.op else {
            return Some(Err(batch_scheduler_invariant(
                "repeated direct batch is missing its region-scaled operation",
            )));
        };
        match first.fmt {
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 => Some(
                crate::hybrid::decode_repeated_region_scaled_color_batch_direct_to_device_routed(
                    first.input.as_ref(),
                    roi,
                    scale,
                    first.fmt,
                    requests.len(),
                    backend,
                ),
            ),
            _ => None,
        }
    }

    #[cfg(not(target_os = "macos"))]
    {
        let _ = backend;
        None
    }
}

pub(super) fn decode_distinct_region_scaled_direct_batch_prechecked(
    requests: &[QueuedRequest],
    backend: Option<&MetalBackendSession>,
) -> Option<Result<Vec<Surface>, Error>> {
    decode_distinct_region_scaled_direct_batch_inner(requests, true, backend)
}

fn decode_distinct_region_scaled_direct_batch_inner(
    requests: &[QueuedRequest],
    auto_metal_prechecked: bool,
    backend: Option<&MetalBackendSession>,
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
            let BatchOp::RegionScaled { roi, scale } = request.op else {
                return Some(Err(batch_scheduler_invariant(
                    "direct region-scaled batch contains a non-region-scaled request",
                )));
            };
            request_specs.push((request.input.clone(), roi, scale));
        }
        let result = match first.fmt {
            PixelFormat::Gray8 | PixelFormat::Gray16 => {
                crate::hybrid::decode_region_scaled_grayscale_batch_direct_to_device_routed(
                    &request_specs,
                    first.fmt,
                    backend,
                )
            }
            PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Rgb16 => {
                crate::hybrid::decode_region_scaled_color_batch_direct_to_device_routed(
                    &request_specs,
                    first.fmt,
                    backend,
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
        let _ = backend;
        None
    }
}

pub(super) fn decode_individual(
    request: &QueuedRequest,
    backend: Option<&MetalBackendSession>,
) -> Result<Surface, Error> {
    let mut decoder = J2kDecoder::new(request.input.as_ref())?;
    if let Some(backend) = backend {
        return decoder.decode_request_to_device_with_session(
            MetalDecodeRequest {
                fmt: request.fmt,
                op: request.op.into(),
                backend: request.backend,
            },
            backend,
        );
    }
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

impl From<BatchOp> for crate::MetalDecodeOp {
    fn from(value: BatchOp) -> Self {
        match value {
            BatchOp::Full => Self::Full,
            BatchOp::Region(roi) => Self::Region(roi),
            BatchOp::Scaled(scale) => Self::Scaled(scale),
            BatchOp::RegionScaled { roi, scale } => Self::RegionScaled { roi, scale },
        }
    }
}
