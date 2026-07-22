// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{J2kEncodeStageError, J2kHtj2kTileEncodeJob, J2kResidentHtj2kTileEncodeJob};
use j2k_cuda_runtime::{
    CudaContext, CudaDeviceBuffer, CudaDwt53LevelShape, CudaHtj2kEncodeResources,
    CudaJ2kQuantizeJob, CudaJ2kQuantizeSubbandRegionJob, CudaJ2kResidentComponents,
    CudaJ2kStridedInterleavedPixels,
};

use crate::allocation::HostPhaseBudget;
use crate::encode::stage_error::{
    adapter_error, arithmetic_overflow, internal_invariant, runtime_error, CudaStageResult,
};

use super::super::{
    cuda_component_count_u8, cuda_encode_format, time_cuda_stage, CudaEncodeStageTimings,
    CudaLosslessEncodeTile,
};
use super::code_blocks::{cuda_ht_region_jobs, encoded_ht_code_blocks_from_cuda};
use super::host_budget::account_encoded_resolution_owners;
use super::htj2k_allocation_error;
use super::ordering::cuda_order_component_resolution_packets;
use super::tile_packets::cuda_packetize_tile_body;
use super::types::{
    CudaEncodedHtj2kResolution, CudaEncodedHtj2kSubband, CudaEncodedHtj2kTile,
    CudaHtj2kEncodeRuntime, CudaHtj2kTileEncodeStats, CudaTileSubbandKind, CudaTileSubbandRegion,
};
use super::validation::{resident_job_from_host, validate_cuda_htj2k_tile_job};

#[cfg(feature = "cuda-runtime")]
pub(in crate::encode) fn cuda_encode_htj2k_tile_body(
    context: &CudaContext,
    encode_resources: &CudaHtj2kEncodeResources,
    job: J2kHtj2kTileEncodeJob<'_>,
    collect_profile: bool,
) -> CudaStageResult<Option<CudaEncodedHtj2kTile>> {
    let resident_job = resident_job_from_host(job)?;
    validate_cuda_htj2k_tile_job(resident_job)?;
    let num_components = cuda_component_count_u8(
        resident_job.input.num_components(),
        "CUDA HTJ2K tile encode supports at most 255 components",
    )?;
    let num_pixels = (resident_job.input.width() as usize)
        .checked_mul(resident_job.input.height() as usize)
        .ok_or_else(|| arithmetic_overflow("CUDA HTJ2K tile pixel count"))?;
    let (components, deinterleave_us) = time_cuda_stage(
        "j2k.htj2k.encode.tile.deinterleave",
        context,
        collect_profile,
        || {
            context.j2k_deinterleave_to_f32_resident(
                job.pixels,
                num_pixels,
                num_components,
                resident_job.input.bit_depth(),
                resident_job.input.signed(),
            )
        },
    )
    .map_err(|error| runtime_error("deinterleave CUDA HTJ2K host tile", error))?;
    cuda_encode_htj2k_resident_components_body(
        context,
        encode_resources,
        resident_job,
        components,
        deinterleave_us,
        collect_profile,
    )
}

#[cfg(feature = "cuda-runtime")]
pub(in crate::encode) fn cuda_encode_htj2k_device_tile_body(
    context: &CudaContext,
    encode_resources: &CudaHtj2kEncodeResources,
    tile: CudaLosslessEncodeTile<'_>,
    job: J2kResidentHtj2kTileEncodeJob<'_>,
    collect_profile: bool,
) -> CudaStageResult<Option<CudaEncodedHtj2kTile>> {
    validate_cuda_htj2k_tile_job(job)?;
    let num_components = cuda_component_count_u8(
        job.input.num_components(),
        "CUDA HTJ2K tile encode supports at most 255 components",
    )?;
    let format = cuda_encode_format(tile.format).map_err(|error| match error {
        crate::Error::UnsupportedCudaRequest { reason } => J2kEncodeStageError::unsupported(reason),
        source => adapter_error("validate CUDA HTJ2K tile format", source),
    })?;
    if job.input.width() != tile.output_width || job.input.height() != tile.output_height {
        return Err(J2kEncodeStageError::invalid_request(
            "CUDA HTJ2K tile encode job dimensions do not match CUDA tile",
        ));
    }
    if tile.width != tile.output_width || tile.height != tile.output_height {
        return Err(J2kEncodeStageError::unsupported(
            "CUDA HTJ2K tile encode does not support input padding",
        ));
    }
    if job.input.num_components() != u16::from(format.components)
        || job.input.bit_depth() != format.bit_depth
        || job.input.signed()
    {
        return Err(J2kEncodeStageError::invalid_request(
            "CUDA HTJ2K tile encode job sample format does not match CUDA tile",
        ));
    }
    let (components, deinterleave_us) = time_cuda_stage(
        "j2k.htj2k.encode.tile.device_deinterleave",
        context,
        collect_profile,
        || {
            context.j2k_deinterleave_strided_to_f32_resident(CudaJ2kStridedInterleavedPixels {
                buffer: tile.buffer,
                byte_offset: tile.byte_offset,
                width: tile.width,
                height: tile.height,
                pitch_bytes: tile.pitch_bytes,
                num_components,
                bit_depth: job.input.bit_depth(),
                signed: job.input.signed(),
            })
        },
    )
    .map_err(|error| runtime_error("deinterleave CUDA HTJ2K device tile", error))?;
    cuda_encode_htj2k_resident_components_body(
        context,
        encode_resources,
        job,
        components,
        deinterleave_us,
        collect_profile,
    )
}

#[cfg(feature = "cuda-runtime")]
#[expect(
    clippy::too_many_lines,
    reason = "resident HTJ2K encoding keeps CUDA stage order, fallbacks, and profiling atomic"
)]
fn cuda_encode_htj2k_resident_components_body(
    context: &CudaContext,
    encode_resources: &CudaHtj2kEncodeResources,
    job: J2kResidentHtj2kTileEncodeJob<'_>,
    mut components: CudaJ2kResidentComponents,
    deinterleave_us: u128,
    collect_profile: bool,
) -> CudaStageResult<Option<CudaEncodedHtj2kTile>> {
    let mut stats = CudaHtj2kTileEncodeStats {
        collect_profile,
        deinterleave_dispatches: components.execution().kernel_dispatches(),
        timings: CudaEncodeStageTimings {
            deinterleave_us,
            ..CudaEncodeStageTimings::default()
        },
        ..CudaHtj2kTileEncodeStats::default()
    };
    let pool = context.buffer_pool();
    let runtime = CudaHtj2kEncodeRuntime {
        context,
        resources: encode_resources,
        pool: &pool,
    };

    if job.use_mct {
        let (execution, mct_us) = if job.reversible {
            time_cuda_stage(
                "j2k.htj2k.encode.tile.rct",
                context,
                collect_profile,
                || context.j2k_forward_rct_resident(&mut components),
            )
            .map_err(|error| runtime_error("apply CUDA HTJ2K tile RCT", error))?
        } else {
            time_cuda_stage(
                "j2k.htj2k.encode.tile.ict",
                context,
                collect_profile,
                || context.j2k_forward_ict_resident(&mut components),
            )
            .map_err(|error| runtime_error("apply CUDA HTJ2K tile ICT", error))?
        };
        stats.timings.mct_us = stats.timings.mct_us.saturating_add(mct_us);
        if job.reversible {
            stats.forward_rct_dispatches = execution.kernel_dispatches();
        } else {
            stats.forward_ict_dispatches = execution.kernel_dispatches();
        }
    }

    let mut component_host_budget = HostPhaseBudget::new("j2k CUDA HTJ2K component packet graph");
    let mut component_resolution_packets = component_host_budget
        .try_vec_with_capacity(usize::from(job.input.num_components()))
        .map_err(htj2k_allocation_error)?;
    if job.num_decomposition_levels == 0 {
        for component in 0..job.input.num_components() {
            let y0 = u32::from(component)
                .checked_mul(job.input.height())
                .ok_or_else(|| arithmetic_overflow("CUDA HTJ2K tile component offset"))?;
            let subband = cuda_encode_tile_subband_region(
                runtime,
                components.buffer(),
                CudaTileSubbandRegion {
                    x0: 0,
                    y0,
                    width: job.input.width(),
                    height: job.input.height(),
                    stride: job.input.width(),
                },
                job.quantization_steps[0],
                job,
                CudaTileSubbandKind::LowLow,
                &mut stats,
            )?;
            // These inner vectors have codec-fixed cardinalities: one
            // resolution containing one LL subband. The image-derived outer
            // component collection is reserved fallibly above.
            let packets = vec![CudaEncodedHtj2kResolution {
                subbands: vec![subband],
            }];
            account_encoded_resolution_owners(
                &mut component_host_budget,
                &packets,
                packets.capacity(),
            )?;
            component_resolution_packets.push(packets);
        }
    } else {
        for component in 0..job.input.num_components() {
            let component_u8 =
                cuda_component_count_u8(component, "CUDA HTJ2K tile component index exceeds 255")?;
            let packets = if job.reversible {
                let (dwt, dwt_us) = time_cuda_stage(
                    "j2k.htj2k.encode.tile.dwt53",
                    context,
                    collect_profile,
                    || {
                        context.j2k_forward_dwt53_resident_component(
                            &components,
                            component_u8,
                            job.input.width(),
                            job.input.height(),
                            job.num_decomposition_levels,
                        )
                    },
                )
                .map_err(|error| runtime_error("apply CUDA HTJ2K tile DWT 5/3", error))?;
                stats.forward_dwt53_dispatches = stats
                    .forward_dwt53_dispatches
                    .saturating_add(dwt.execution().kernel_dispatches());
                stats.timings.dwt_us = stats.timings.dwt_us.saturating_add(dwt_us);
                cuda_encode_dwt_component_packets(
                    runtime,
                    job,
                    dwt.buffer(),
                    dwt.levels(),
                    dwt.ll_dimensions(),
                    &mut stats,
                )?
            } else {
                let (dwt, dwt_us) = time_cuda_stage(
                    "j2k.htj2k.encode.tile.dwt97",
                    context,
                    collect_profile,
                    || {
                        context.j2k_forward_dwt97_resident_component(
                            &components,
                            component_u8,
                            job.input.width(),
                            job.input.height(),
                            job.num_decomposition_levels,
                        )
                    },
                )
                .map_err(|error| runtime_error("apply CUDA HTJ2K tile DWT 9/7", error))?;
                stats.forward_dwt97_dispatches = stats
                    .forward_dwt97_dispatches
                    .saturating_add(dwt.execution().kernel_dispatches());
                stats.timings.dwt_us = stats.timings.dwt_us.saturating_add(dwt_us);
                cuda_encode_dwt_component_packets(
                    runtime,
                    job,
                    dwt.buffer(),
                    dwt.levels(),
                    dwt.ll_dimensions(),
                    &mut stats,
                )?
            };
            account_encoded_resolution_owners(
                &mut component_host_budget,
                &packets,
                packets.capacity(),
            )?;
            component_resolution_packets.push(packets);
        }
    }

    let resolution_packets = cuda_order_component_resolution_packets(
        component_resolution_packets,
        job.input.num_components(),
    )?;
    let (tile_data, packetization_dispatches, packetize_us) = cuda_packetize_tile_body(
        context,
        job,
        &resolution_packets,
        resolution_packets.capacity(),
        stats.ht_code_block_jobs,
    )?;
    stats.timings.packetize_us = stats.timings.packetize_us.saturating_add(packetize_us);
    Ok(Some(CudaEncodedHtj2kTile {
        tile_data,
        deinterleave_dispatches: stats.deinterleave_dispatches,
        forward_rct_dispatches: stats.forward_rct_dispatches,
        forward_ict_dispatches: stats.forward_ict_dispatches,
        forward_dwt53_dispatches: stats.forward_dwt53_dispatches,
        forward_dwt97_dispatches: stats.forward_dwt97_dispatches,
        quantize_jobs: stats.quantize_jobs,
        quantize_dispatches: stats.quantize_dispatches,
        ht_code_block_dispatches: stats.ht_code_block_dispatches,
        ht_code_block_jobs: stats.ht_code_block_jobs,
        packetization_dispatches,
        timings: stats.timings,
    }))
}

#[cfg(feature = "cuda-runtime")]
fn cuda_encode_dwt_component_packets(
    runtime: CudaHtj2kEncodeRuntime<'_>,
    job: J2kResidentHtj2kTileEncodeJob<'_>,
    transformed: &CudaDeviceBuffer,
    levels: &[CudaDwt53LevelShape],
    ll_dimensions: (u32, u32),
    stats: &mut CudaHtj2kTileEncodeStats,
) -> CudaStageResult<Vec<CudaEncodedHtj2kResolution>> {
    if levels.len() != usize::from(job.num_decomposition_levels) {
        return Err(internal_invariant(
            "CUDA HTJ2K tile DWT level count mismatch",
        ));
    }
    let (ll_width, ll_height) = ll_dimensions;
    let full_width = levels.first().map_or(ll_width, |level| level.width);
    // Deliberately codec-bounded: validated decomposition levels fit the J2K
    // level ceiling, and each resolution below owns exactly one or three
    // subband descriptors. Image/code-block-sized vectors are fallible.
    let mut packets = Vec::with_capacity(levels.len().saturating_add(1));

    let ll_subband = cuda_encode_tile_subband_region(
        runtime,
        transformed,
        CudaTileSubbandRegion {
            x0: 0,
            y0: 0,
            width: ll_width,
            height: ll_height,
            stride: full_width,
        },
        job.quantization_steps[0],
        job,
        CudaTileSubbandKind::LowLow,
        stats,
    )?;
    packets.push(CudaEncodedHtj2kResolution {
        subbands: vec![ll_subband],
    });

    for (level_idx, level) in levels.iter().rev().enumerate() {
        let step_base = 1usize
            .checked_add(level_idx.saturating_mul(3))
            .ok_or_else(|| arithmetic_overflow("CUDA HTJ2K tile quantization step index"))?;
        let hl = cuda_encode_tile_subband_region(
            runtime,
            transformed,
            CudaTileSubbandRegion {
                x0: level.low_width,
                y0: 0,
                width: level.high_width,
                height: level.low_height,
                stride: full_width,
            },
            job.quantization_steps[step_base],
            job,
            CudaTileSubbandKind::HighLow,
            stats,
        )?;
        let lh = cuda_encode_tile_subband_region(
            runtime,
            transformed,
            CudaTileSubbandRegion {
                x0: 0,
                y0: level.low_height,
                width: level.low_width,
                height: level.high_height,
                stride: full_width,
            },
            job.quantization_steps[step_base + 1],
            job,
            CudaTileSubbandKind::LowHigh,
            stats,
        )?;
        let hh = cuda_encode_tile_subband_region(
            runtime,
            transformed,
            CudaTileSubbandRegion {
                x0: level.low_width,
                y0: level.low_height,
                width: level.high_width,
                height: level.high_height,
                stride: full_width,
            },
            job.quantization_steps[step_base + 2],
            job,
            CudaTileSubbandKind::HighHigh,
            stats,
        )?;
        packets.push(CudaEncodedHtj2kResolution {
            subbands: vec![hl, lh, hh],
        });
    }

    Ok(packets)
}

#[cfg(feature = "cuda-runtime")]
fn cuda_encode_tile_subband_region(
    runtime: CudaHtj2kEncodeRuntime<'_>,
    source: &CudaDeviceBuffer,
    region: CudaTileSubbandRegion,
    quantization_step: (u16, u16),
    job: J2kResidentHtj2kTileEncodeJob<'_>,
    subband_kind: CudaTileSubbandKind,
    stats: &mut CudaHtj2kTileEncodeStats,
) -> CudaStageResult<CudaEncodedHtj2kSubband> {
    if region.width == 0 || region.height == 0 {
        return Ok(CudaEncodedHtj2kSubband {
            code_blocks: Vec::new(),
            num_cbs_x: 0,
            num_cbs_y: 0,
        });
    }

    let (step_exponent, step_mantissa) = quantization_step;
    let step_exponent_u8 = u8::try_from(step_exponent).map_err(|_| {
        J2kEncodeStageError::invalid_request("CUDA HTJ2K tile quantization exponent exceeds u8")
    })?;
    let total_bitplanes = job
        .guard_bits
        .saturating_add(step_exponent_u8)
        .saturating_sub(1);
    let (quantized, quantize_us) = time_cuda_stage(
        "j2k.htj2k.encode.tile.quantize",
        runtime.context,
        stats.collect_profile,
        || {
            runtime.context.j2k_quantize_subband_region_resident(
                source,
                CudaJ2kQuantizeSubbandRegionJob {
                    x0: region.x0,
                    y0: region.y0,
                    width: region.width,
                    height: region.height,
                    stride: region.stride,
                    quantization: CudaJ2kQuantizeJob {
                        step_exponent,
                        step_mantissa,
                        range_bits: cuda_tile_subband_range_bits(
                            job.input.bit_depth(),
                            subband_kind,
                        ),
                        reversible: job.reversible,
                    },
                },
            )
        },
    )
    .map_err(|error| runtime_error("quantize CUDA HTJ2K tile subband", error))?;
    stats.quantize_jobs = stats.quantize_jobs.saturating_add(1);
    stats.quantize_dispatches = stats
        .quantize_dispatches
        .saturating_add(quantized.execution().kernel_dispatches());
    stats.timings.quantize_us = stats.timings.quantize_us.saturating_add(quantize_us);

    let region_jobs = cuda_ht_region_jobs(
        region.width,
        region.height,
        job.code_block_width,
        job.code_block_height,
        total_bitplanes,
    )?;
    stats.ht_code_block_jobs = stats.ht_code_block_jobs.saturating_add(region_jobs.len());
    let mut host_budget = HostPhaseBudget::new("j2k CUDA HTJ2K tile region jobs");
    host_budget
        .account_vec(&region_jobs)
        .map_err(htj2k_allocation_error)?;
    let encoded = runtime
        .context
        .encode_htj2k_codeblock_regions_resident_with_resources_and_pool_and_live_host_bytes(
            quantized.buffer(),
            quantized.coefficient_count(),
            &region_jobs,
            runtime.resources,
            runtime.pool,
            host_budget.live_bytes(),
        )
        .map_err(|error| runtime_error("encode CUDA HTJ2K tile code blocks", error))?;
    stats.ht_code_block_dispatches = stats
        .ht_code_block_dispatches
        .saturating_add(encoded.execution().kernel_dispatches());
    stats.timings.ht_encode_us = stats
        .timings
        .ht_encode_us
        .saturating_add(encoded.stage_timings().ht_encode_us);

    Ok(CudaEncodedHtj2kSubband {
        code_blocks: encoded_ht_code_blocks_from_cuda(encoded)?,
        num_cbs_x: region.width.div_ceil(job.code_block_width),
        num_cbs_y: region.height.div_ceil(job.code_block_height),
    })
}

#[cfg(feature = "cuda-runtime")]
fn cuda_tile_subband_range_bits(bit_depth: u8, subband_kind: CudaTileSubbandKind) -> u8 {
    let log_gain = match subband_kind {
        CudaTileSubbandKind::LowLow => 0,
        CudaTileSubbandKind::HighLow | CudaTileSubbandKind::LowHigh => 1,
        CudaTileSubbandKind::HighHigh => 2,
    };
    bit_depth.saturating_add(log_gain)
}
