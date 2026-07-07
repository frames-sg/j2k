// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    decode_tiles_into, decode_tiles_region_scaled_into, TileBatchOptions, TileDecodeJob,
    TileRegionScaledDecodeJob,
};
use j2k_core::{
    checked_surface_len, BackendKind, BackendRequest, PixelFormat,
    DEFAULT_MAX_HOST_ALLOCATION_BYTES,
};

use crate::{Error, J2kDecoder, Storage, Surface, SurfaceResidency};

use super::{batch_scheduler_invariant, BatchOp, QueuedRequest};

pub(super) fn decode_cpu_host_batch(
    requests: &[QueuedRequest],
) -> Option<Result<Vec<Surface>, Error>> {
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
    let mut allocations = Vec::with_capacity(requests.len());
    for request in requests {
        let decoder = J2kDecoder::new(request.input.as_ref())?;
        let tile_dims = decoder.inner.info().dimensions;
        let allocation = checked_cpu_batch_surface(tile_dims, fmt)?;
        dims.push(tile_dims);
        allocations.push(allocation);
    }
    let mut outputs = allocations
        .iter()
        .map(|(_, len)| vec![0_u8; *len])
        .collect::<Vec<_>>();

    {
        let mut jobs = requests
            .iter()
            .zip(dims.iter())
            .zip(allocations.iter())
            .zip(outputs.iter_mut())
            .map(|(((request, _dims), (stride, _len)), out)| TileDecodeJob {
                input: request.input.as_ref(),
                out: out.as_mut_slice(),
                stride: *stride,
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
    let mut allocations = Vec::with_capacity(requests.len());
    for request in requests {
        let BatchOp::RegionScaled { roi, scale } = request.op else {
            return Err(batch_scheduler_invariant(
                "CPU region-scaled batch contains a non-region-scaled request",
            ));
        };
        let dimensions = roi.scaled_covering(scale);
        let dims_tuple = (dimensions.w, dimensions.h);
        let allocation = checked_cpu_batch_surface(dims_tuple, fmt)?;
        dims.push(dims_tuple);
        allocations.push(allocation);
    }
    let mut outputs = allocations
        .iter()
        .map(|(_, len)| vec![0_u8; *len])
        .collect::<Vec<_>>();

    {
        let mut jobs = Vec::with_capacity(requests.len());
        for ((request, (stride, _len)), out) in requests
            .iter()
            .zip(allocations.iter())
            .zip(outputs.iter_mut())
        {
            let BatchOp::RegionScaled { roi, scale } = request.op else {
                return Err(batch_scheduler_invariant(
                    "CPU region-scaled job creation received a non-region-scaled request",
                ));
            };
            jobs.push(TileRegionScaledDecodeJob {
                input: request.input.as_ref(),
                out: out.as_mut_slice(),
                stride: *stride,
                roi,
                scale,
            });
        }
        decode_tiles_region_scaled_into(&mut jobs, fmt, TileBatchOptions::default())
            .map_err(|err| Error::Decode(err.source))?;
    }

    Ok(outputs
        .into_iter()
        .zip(dims)
        .map(|(bytes, dimensions)| host_surface(bytes, dimensions, fmt))
        .collect())
}

fn checked_cpu_batch_surface(dims: (u32, u32), fmt: PixelFormat) -> Result<(usize, usize), Error> {
    checked_surface_len(
        dims,
        fmt.bytes_per_pixel(),
        DEFAULT_MAX_HOST_ALLOCATION_BYTES,
        "j2k Metal CPU batch fallback surface",
    )
    .map_err(Error::from)
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
