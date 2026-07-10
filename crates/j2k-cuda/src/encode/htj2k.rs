// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k::{
    EncodedHtJ2kCodeBlock, J2kHtCodeBlockEncodeJob, J2kHtSubbandEncodeJob, J2kHtj2kTileEncodeJob,
    J2kPacketizationBlockCodingMode, J2kPacketizationCodeBlock, J2kPacketizationEncodeJob,
    J2kPacketizationPacketDescriptor, J2kPacketizationResolution, J2kPacketizationSubband,
};
use j2k_cuda_runtime::{
    CudaBufferPool, CudaContext, CudaDeviceBuffer, CudaDwt53LevelShape,
    CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeCodeBlockRegionJob, CudaHtj2kEncodeResources,
    CudaHtj2kEncodeTables, CudaJ2kQuantizeJob, CudaJ2kQuantizeSubbandRegionJob,
    CudaJ2kResidentComponents, CudaJ2kStridedInterleavedPixels,
};

use super::packetization::{
    cuda_packetization_blocks, cuda_packetization_packets, cuda_packetization_subbands,
    cuda_packetization_tag_nodes, cuda_packetization_tag_states,
    flatten_cuda_htj2k_packetization_job,
};
use super::{
    cuda_component_count_u8, cuda_encode_format, time_cuda_stage, CudaEncodeStageTimings,
    CudaLosslessEncodeTile,
};

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_encode_ht_code_block(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    job: J2kHtCodeBlockEncodeJob<'_>,
) -> core::result::Result<j2k_cuda_runtime::CudaHtj2kEncodedCodeBlocks, &'static str> {
    let coefficient_len = (job.width as usize)
        .checked_mul(job.height as usize)
        .ok_or("CUDA HTJ2K code-block encode job is too large")?;
    if coefficient_len != job.coefficients.len() {
        return Err("CUDA HTJ2K code-block encode job has invalid coefficient length");
    }
    let cuda_jobs = [CudaHtj2kEncodeCodeBlockJob {
        coefficient_offset: 0,
        width: job.width,
        height: job.height,
        total_bitplanes: job.total_bitplanes,
        target_coding_passes: job.target_coding_passes,
    }];
    context
        .encode_htj2k_codeblocks_with_resources(job.coefficients, &cuda_jobs, resources)
        .map_err(|_| "CUDA HTJ2K code-block encode kernel failed")
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_encode_ht_code_blocks(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    jobs: &[J2kHtCodeBlockEncodeJob<'_>],
) -> core::result::Result<j2k_cuda_runtime::CudaHtj2kEncodedCodeBlocks, &'static str> {
    let total_coefficients = jobs.iter().try_fold(0usize, |acc, job| {
        let coefficient_len = (job.width as usize)
            .checked_mul(job.height as usize)
            .ok_or("CUDA HTJ2K code-block batch is too large")?;
        if coefficient_len != job.coefficients.len() {
            return Err("CUDA HTJ2K code-block encode job has invalid coefficient length");
        }
        acc.checked_add(coefficient_len)
            .ok_or("CUDA HTJ2K code-block batch is too large")
    })?;
    let mut coefficients = Vec::with_capacity(total_coefficients);
    let mut cuda_jobs = Vec::with_capacity(jobs.len());
    for job in jobs {
        let coefficient_offset = u32::try_from(coefficients.len())
            .map_err(|_| "CUDA HTJ2K code-block batch is too large")?;
        coefficients.extend_from_slice(job.coefficients);
        cuda_jobs.push(CudaHtj2kEncodeCodeBlockJob {
            coefficient_offset,
            width: job.width,
            height: job.height,
            total_bitplanes: job.total_bitplanes,
            target_coding_passes: job.target_coding_passes,
        });
    }

    context
        .encode_htj2k_codeblocks_with_resources(&coefficients, &cuda_jobs, resources)
        .map_err(|_| "CUDA HTJ2K code-block batch encode kernel failed")
}

#[cfg(feature = "cuda-runtime")]
pub(super) struct CudaEncodedHtj2kTile {
    pub(super) tile_data: Vec<u8>,
    pub(super) deinterleave_dispatches: usize,
    pub(super) forward_rct_dispatches: usize,
    pub(super) forward_ict_dispatches: usize,
    pub(super) forward_dwt53_dispatches: usize,
    pub(super) forward_dwt97_dispatches: usize,
    pub(super) quantize_jobs: usize,
    pub(super) quantize_dispatches: usize,
    pub(super) ht_code_block_dispatches: usize,
    pub(super) ht_code_block_jobs: usize,
    pub(super) packetization_dispatches: usize,
    pub(super) timings: CudaEncodeStageTimings,
}

#[cfg(feature = "cuda-runtime")]
#[derive(Default)]
struct CudaHtj2kTileEncodeStats {
    collect_profile: bool,
    deinterleave_dispatches: usize,
    forward_rct_dispatches: usize,
    forward_ict_dispatches: usize,
    forward_dwt53_dispatches: usize,
    forward_dwt97_dispatches: usize,
    quantize_jobs: usize,
    quantize_dispatches: usize,
    ht_code_block_dispatches: usize,
    ht_code_block_jobs: usize,
    timings: CudaEncodeStageTimings,
}

#[cfg(feature = "cuda-runtime")]
struct CudaEncodedHtj2kResolution {
    subbands: Vec<CudaEncodedHtj2kSubband>,
}

#[cfg(feature = "cuda-runtime")]
struct CudaEncodedHtj2kSubband {
    code_blocks: Vec<EncodedHtJ2kCodeBlock>,
    num_cbs_x: u32,
    num_cbs_y: u32,
}

#[cfg(feature = "cuda-runtime")]
#[derive(Clone, Copy)]
struct CudaTileSubbandRegion {
    x0: u32,
    y0: u32,
    width: u32,
    height: u32,
    stride: u32,
}

#[cfg(feature = "cuda-runtime")]
#[derive(Clone, Copy)]
enum CudaTileSubbandKind {
    LowLow,
    HighLow,
    LowHigh,
    HighHigh,
}

#[cfg(feature = "cuda-runtime")]
#[derive(Clone, Copy)]
struct CudaHtj2kEncodeRuntime<'a> {
    context: &'a CudaContext,
    resources: &'a CudaHtj2kEncodeResources,
    pool: &'a CudaBufferPool,
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_encode_htj2k_tile_body(
    context: &CudaContext,
    encode_resources: &CudaHtj2kEncodeResources,
    job: J2kHtj2kTileEncodeJob<'_>,
    collect_profile: bool,
) -> core::result::Result<Option<CudaEncodedHtj2kTile>, &'static str> {
    validate_cuda_htj2k_tile_job(job)?;
    let num_components = cuda_component_count_u8(
        job.num_components,
        "CUDA HTJ2K tile encode supports at most 255 components",
    )?;
    let num_pixels = (job.width as usize)
        .checked_mul(job.height as usize)
        .ok_or("CUDA HTJ2K tile dimensions are too large")?;
    let (components, deinterleave_us) = time_cuda_stage(
        "j2k.htj2k.encode.tile.deinterleave",
        context,
        collect_profile,
        || {
            context.j2k_deinterleave_to_f32_resident(
                job.pixels,
                num_pixels,
                num_components,
                job.bit_depth,
                job.signed,
            )
        },
    )
    .map_err(|_| "CUDA HTJ2K tile deinterleave failed")?;
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
fn validate_cuda_htj2k_tile_job(
    job: J2kHtj2kTileEncodeJob<'_>,
) -> core::result::Result<(), &'static str> {
    let _ = cuda_component_count_u8(
        job.num_components,
        "CUDA HTJ2K tile encode supports at most 255 components",
    )?;
    if job
        .component_sampling
        .iter()
        .any(|&sampling| sampling != (1, 1))
    {
        return Err("CUDA HTJ2K tile encode does not support component subsampling != (1, 1)");
    }
    // Native treats `use_mct = options.use_mct && num_components >= 3`, applying the
    // color transform to component planes 0,1,2 and passing any 4th plane through
    // unchanged. The resident path mirrors this: RCT/ICT runs on the first three
    // planes (see `j2k_forward_rct_resident`/`j2k_forward_ict_resident`), and every
    // component — including the passthrough 4th — still flows through the per-component
    // DWT → quantize → HT code-block → packetization loop below.
    //
    // Only `{1, 3, 4}` component counts are in scope. Reject any other count with a
    // typed hard error rather than `Ok(None)` (a silent CPU fallback is forbidden for
    // in-scope inputs).
    if !matches!(job.num_components, 1 | 3 | 4) {
        return Err("CUDA HTJ2K tile encode supports 1, 3, or 4 components");
    }
    if job.use_mct && job.num_components < 3 {
        return Err("CUDA HTJ2K tile encode requires at least three components for MCT");
    }
    if job.code_block_width == 0 || job.code_block_height == 0 {
        return Err("CUDA HTJ2K tile encode job has invalid code-block dimensions");
    }
    let expected_quantization_steps = 1usize
        .checked_add(usize::from(job.num_decomposition_levels).saturating_mul(3))
        .ok_or("CUDA HTJ2K tile quantization step count overflow")?;
    if job.quantization_steps.len() != expected_quantization_steps {
        return Err("CUDA HTJ2K tile quantization step count mismatch");
    }
    Ok(())
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_encode_htj2k_device_tile_body(
    context: &CudaContext,
    encode_resources: &CudaHtj2kEncodeResources,
    tile: CudaLosslessEncodeTile<'_>,
    job: J2kHtj2kTileEncodeJob<'_>,
    collect_profile: bool,
) -> core::result::Result<Option<CudaEncodedHtj2kTile>, &'static str> {
    validate_cuda_htj2k_tile_job(job)?;
    let num_components = cuda_component_count_u8(
        job.num_components,
        "CUDA HTJ2K tile encode supports at most 255 components",
    )?;
    let format = cuda_encode_format(tile.format).map_err(|_| "CUDA HTJ2K tile format failed")?;
    if job.width != tile.output_width || job.height != tile.output_height {
        return Err("CUDA HTJ2K tile encode job dimensions do not match CUDA tile");
    }
    if tile.width != tile.output_width || tile.height != tile.output_height {
        return Err("CUDA HTJ2K tile encode does not support input padding");
    }
    if job.num_components != u16::from(format.components)
        || job.bit_depth != format.bit_depth
        || job.signed
    {
        return Err("CUDA HTJ2K tile encode job sample format does not match CUDA tile");
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
                bit_depth: job.bit_depth,
                signed: job.signed,
            })
        },
    )
    .map_err(|_| "CUDA HTJ2K tile device deinterleave failed")?;
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
    job: J2kHtj2kTileEncodeJob<'_>,
    mut components: CudaJ2kResidentComponents,
    deinterleave_us: u128,
    collect_profile: bool,
) -> core::result::Result<Option<CudaEncodedHtj2kTile>, &'static str> {
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
            .map_err(|_| "CUDA HTJ2K tile RCT failed")?
        } else {
            time_cuda_stage(
                "j2k.htj2k.encode.tile.ict",
                context,
                collect_profile,
                || context.j2k_forward_ict_resident(&mut components),
            )
            .map_err(|_| "CUDA HTJ2K tile ICT failed")?
        };
        stats.timings.mct_us = stats.timings.mct_us.saturating_add(mct_us);
        if job.reversible {
            stats.forward_rct_dispatches = execution.kernel_dispatches();
        } else {
            stats.forward_ict_dispatches = execution.kernel_dispatches();
        }
    }

    let mut component_resolution_packets = Vec::with_capacity(usize::from(job.num_components));
    if job.num_decomposition_levels == 0 {
        for component in 0..job.num_components {
            let y0 = u32::from(component)
                .checked_mul(job.height)
                .ok_or("CUDA HTJ2K tile component offset overflow")?;
            let subband = cuda_encode_tile_subband_region(
                runtime,
                components.buffer(),
                CudaTileSubbandRegion {
                    x0: 0,
                    y0,
                    width: job.width,
                    height: job.height,
                    stride: job.width,
                },
                job.quantization_steps[0],
                job,
                CudaTileSubbandKind::LowLow,
                &mut stats,
            )?;
            component_resolution_packets.push(vec![CudaEncodedHtj2kResolution {
                subbands: vec![subband],
            }]);
        }
    } else {
        for component in 0..job.num_components {
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
                            job.width,
                            job.height,
                            job.num_decomposition_levels,
                        )
                    },
                )
                .map_err(|_| "CUDA HTJ2K tile DWT 5/3 failed")?;
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
                            job.width,
                            job.height,
                            job.num_decomposition_levels,
                        )
                    },
                )
                .map_err(|_| "CUDA HTJ2K tile DWT 9/7 failed")?;
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
            component_resolution_packets.push(packets);
        }
    }

    let resolution_packets =
        cuda_order_component_resolution_packets(component_resolution_packets, job.num_components)?;
    let (tile_data, packetization_dispatches, packetize_us) =
        cuda_packetize_tile_body(context, job, &resolution_packets, stats.ht_code_block_jobs)?;
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
    job: J2kHtj2kTileEncodeJob<'_>,
    transformed: &CudaDeviceBuffer,
    levels: &[CudaDwt53LevelShape],
    ll_dimensions: (u32, u32),
    stats: &mut CudaHtj2kTileEncodeStats,
) -> core::result::Result<Vec<CudaEncodedHtj2kResolution>, &'static str> {
    if levels.len() != usize::from(job.num_decomposition_levels) {
        return Err("CUDA HTJ2K tile DWT level count mismatch");
    }
    let (ll_width, ll_height) = ll_dimensions;
    let full_width = levels.first().map_or(ll_width, |level| level.width);
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
            .ok_or("CUDA HTJ2K tile quantization step index overflow")?;
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
    job: J2kHtj2kTileEncodeJob<'_>,
    subband_kind: CudaTileSubbandKind,
    stats: &mut CudaHtj2kTileEncodeStats,
) -> core::result::Result<CudaEncodedHtj2kSubband, &'static str> {
    if region.width == 0 || region.height == 0 {
        return Ok(CudaEncodedHtj2kSubband {
            code_blocks: Vec::new(),
            num_cbs_x: 0,
            num_cbs_y: 0,
        });
    }

    let (step_exponent, step_mantissa) = quantization_step;
    let step_exponent_u8 = u8::try_from(step_exponent)
        .map_err(|_| "CUDA HTJ2K tile quantization exponent exceeds u8")?;
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
                        range_bits: cuda_tile_subband_range_bits(job.bit_depth, subband_kind),
                        reversible: job.reversible,
                    },
                },
            )
        },
    )
    .map_err(|_| "CUDA HTJ2K tile quantize failed")?;
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
    let encoded = runtime
        .context
        .encode_htj2k_codeblock_regions_resident_with_resources_and_pool(
            quantized.buffer(),
            quantized.coefficient_count(),
            &region_jobs,
            runtime.resources,
            runtime.pool,
        )
        .map_err(|_| "CUDA HTJ2K tile code-block encode failed")?;
    stats.ht_code_block_dispatches = stats
        .ht_code_block_dispatches
        .saturating_add(encoded.execution().kernel_dispatches());
    stats.timings.ht_encode_us = stats
        .timings
        .ht_encode_us
        .saturating_add(encoded.stage_timings().ht_encode_us);

    Ok(CudaEncodedHtj2kSubband {
        code_blocks: encoded_ht_code_blocks_from_cuda(&encoded),
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

#[cfg(feature = "cuda-runtime")]
fn cuda_order_component_resolution_packets(
    component_resolution_packets: Vec<Vec<CudaEncodedHtj2kResolution>>,
    num_components: u16,
) -> core::result::Result<Vec<CudaEncodedHtj2kResolution>, &'static str> {
    if component_resolution_packets.len() != usize::from(num_components) {
        return Err("CUDA HTJ2K tile component packet count mismatch");
    }
    let resolution_count = component_resolution_packets
        .first()
        .map_or(0usize, Vec::len);
    let mut component_iters: Vec<_> = component_resolution_packets
        .into_iter()
        .map(Vec::into_iter)
        .collect();
    let mut resolution_packets =
        Vec::with_capacity(resolution_count.saturating_mul(component_iters.len()));

    for _resolution in 0..resolution_count {
        for component in &mut component_iters {
            resolution_packets.push(
                component
                    .next()
                    .ok_or("CUDA HTJ2K tile component resolution count mismatch")?,
            );
        }
    }
    if component_iters
        .iter_mut()
        .any(|component| component.next().is_some())
    {
        return Err("CUDA HTJ2K tile component resolution count mismatch");
    }

    Ok(resolution_packets)
}

#[cfg(feature = "cuda-runtime")]
fn cuda_ht_region_jobs(
    width: u32,
    height: u32,
    code_block_width: u32,
    code_block_height: u32,
    total_bitplanes: u8,
) -> core::result::Result<Vec<CudaHtj2kEncodeCodeBlockRegionJob>, &'static str> {
    if code_block_width == 0 || code_block_height == 0 {
        return Err("CUDA HTJ2K encode job has invalid code-block dimensions");
    }
    if width == 0 || height == 0 {
        return Ok(Vec::new());
    }

    let num_cbs_x = width.div_ceil(code_block_width);
    let num_cbs_y = height.div_ceil(code_block_height);
    let count = (num_cbs_x as usize)
        .checked_mul(num_cbs_y as usize)
        .ok_or("CUDA HTJ2K code-block count overflow")?;
    let mut cuda_jobs = Vec::with_capacity(count);
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let x0 = cbx
                .checked_mul(code_block_width)
                .ok_or("CUDA HTJ2K code-block x offset overflow")?;
            let y0 = cby
                .checked_mul(code_block_height)
                .ok_or("CUDA HTJ2K code-block y offset overflow")?;
            let block_width = (x0 + code_block_width).min(width) - x0;
            let block_height = (y0 + code_block_height).min(height) - y0;
            let offset = (y0 as usize)
                .checked_mul(width as usize)
                .and_then(|row| row.checked_add(x0 as usize))
                .ok_or("CUDA HTJ2K code-block offset overflow")?;
            cuda_jobs.push(CudaHtj2kEncodeCodeBlockRegionJob {
                coefficient_offset: u32::try_from(offset)
                    .map_err(|_| "CUDA HTJ2K code-block offset exceeds u32")?,
                coefficient_stride: width,
                width: block_width,
                height: block_height,
                total_bitplanes,
                target_coding_passes: 1,
            });
        }
    }
    Ok(cuda_jobs)
}

#[cfg(feature = "cuda-runtime")]
fn cuda_packetize_tile_body(
    context: &CudaContext,
    job: J2kHtj2kTileEncodeJob<'_>,
    resolution_packets: &[CudaEncodedHtj2kResolution],
    code_block_count: usize,
) -> core::result::Result<(Vec<u8>, usize, u128), &'static str> {
    let packet_descriptors =
        cuda_tile_packet_descriptors(resolution_packets.len(), 1, job.num_components)?;
    let resolutions: Vec<J2kPacketizationResolution<'_>> = resolution_packets
        .iter()
        .map(|resolution| J2kPacketizationResolution {
            subbands: resolution
                .subbands
                .iter()
                .map(|subband| {
                    let code_blocks = subband
                        .code_blocks
                        .iter()
                        .map(|block| J2kPacketizationCodeBlock {
                            data: block.data.as_slice(),
                            ht_cleanup_length: block.cleanup_length,
                            ht_refinement_length: block.refinement_length,
                            num_coding_passes: block.num_coding_passes,
                            num_zero_bitplanes: block.num_zero_bitplanes,
                            previously_included: false,
                            l_block: 3,
                            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                        })
                        .collect();
                    J2kPacketizationSubband {
                        code_blocks,
                        num_cbs_x: subband.num_cbs_x,
                        num_cbs_y: subband.num_cbs_y,
                    }
                })
                .collect(),
        })
        .collect();

    let packetization_job = J2kPacketizationEncodeJob {
        resolution_count: u32::try_from(resolutions.len())
            .map_err(|_| "CUDA HTJ2K tile resolution count exceeds u32")?,
        num_layers: 1,
        num_components: job.num_components,
        code_block_count: u32::try_from(code_block_count)
            .map_err(|_| "CUDA HTJ2K tile code-block count exceeds u32")?,
        progression_order: job.progression_order,
        packet_descriptors: &packet_descriptors,
        resolutions: &resolutions,
    };
    let plan = flatten_cuda_htj2k_packetization_job(packetization_job)?;
    let packets = cuda_packetization_packets(&plan);
    let subbands = cuda_packetization_subbands(&plan);
    let blocks = cuda_packetization_blocks(&plan);
    let tag_states = cuda_packetization_tag_states(&plan);
    let tag_nodes = cuda_packetization_tag_nodes(&plan);
    let packetized = context
        .packetize_htj2k_cleanup_packets_with_tag_state(
            &plan.payload,
            &packets,
            &subbands,
            &blocks,
            &tag_states,
            &tag_nodes,
        )
        .map_err(|_| "CUDA HTJ2K tile packetization failed")?;
    Ok((
        packetized.data().to_vec(),
        packetized.execution().kernel_dispatches(),
        packetized.stage_timings().packetize_us,
    ))
}

#[cfg(feature = "cuda-runtime")]
fn cuda_tile_packet_descriptors(
    packet_count: usize,
    num_layers: u8,
    num_components: u16,
) -> core::result::Result<Vec<J2kPacketizationPacketDescriptor>, &'static str> {
    if num_layers != 1 {
        return Err("CUDA HTJ2K tile encode currently prepares one packet layer");
    }
    let component_count = usize::from(num_components).max(1);
    (0..packet_count)
        .map(|packet_index| {
            Ok(J2kPacketizationPacketDescriptor {
                packet_index: u32::try_from(packet_index)
                    .map_err(|_| "CUDA HTJ2K tile packet index exceeds u32")?,
                state_index: u32::try_from(packet_index)
                    .map_err(|_| "CUDA HTJ2K tile packet state index exceeds u32")?,
                layer: 0,
                resolution: u32::try_from(packet_index / component_count)
                    .map_err(|_| "CUDA HTJ2K tile packet resolution exceeds u32")?,
                component: u16::try_from(packet_index % component_count)
                    .map_err(|_| "CUDA HTJ2K tile packet component exceeds u16")?,
                precinct: 0,
            })
        })
        .collect()
}

#[cfg(feature = "cuda-runtime")]
pub(super) struct CudaEncodedHtSubband {
    pub(super) quantize_dispatches: usize,
    pub(super) encode: j2k_cuda_runtime::CudaHtj2kEncodedCodeBlocks,
    pub(super) timings: CudaEncodeStageTimings,
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_encode_ht_subband(
    context: &CudaContext,
    encode_resources: &CudaHtj2kEncodeResources,
    job: J2kHtSubbandEncodeJob<'_>,
    collect_profile: bool,
) -> core::result::Result<CudaEncodedHtSubband, &'static str> {
    let expected_len = (job.width as usize)
        .checked_mul(job.height as usize)
        .ok_or("CUDA HTJ2K subband encode dimensions are too large")?;
    if expected_len != job.coefficients.len() {
        return Err("CUDA HTJ2K subband encode job has invalid coefficient length");
    }
    if job.code_block_width == 0 || job.code_block_height == 0 {
        return Err("CUDA HTJ2K subband encode job has invalid code-block dimensions");
    }

    let sample_buffer = context
        .upload_f32_pinned(job.coefficients)
        .map_err(|_| "CUDA HTJ2K subband upload failed")?;
    let (quantized, quantize_us) = time_cuda_stage(
        "j2k.htj2k.encode.subband.quantize",
        context,
        collect_profile,
        || {
            context.j2k_quantize_subband_resident(
                &sample_buffer,
                job.coefficients.len(),
                CudaJ2kQuantizeJob {
                    step_exponent: job.step_exponent,
                    step_mantissa: job.step_mantissa,
                    range_bits: job.range_bits,
                    reversible: job.reversible,
                },
            )
        },
    )
    .map_err(|_| "CUDA quantize subband encode kernel failed")?;
    let cuda_jobs = cuda_ht_subband_region_jobs(job)?;
    let pool = context.buffer_pool();
    let encoded = context
        .encode_htj2k_codeblock_regions_resident_with_resources_and_pool(
            quantized.buffer(),
            quantized.coefficient_count(),
            &cuda_jobs,
            encode_resources,
            &pool,
        )
        .map_err(|_| "CUDA HTJ2K resident subband encode kernel failed")?;

    Ok(CudaEncodedHtSubband {
        quantize_dispatches: quantized.execution().kernel_dispatches(),
        timings: CudaEncodeStageTimings {
            quantize_us,
            ht_encode_us: encoded.stage_timings().ht_encode_us,
            ..CudaEncodeStageTimings::default()
        },
        encode: encoded,
    })
}

#[cfg(feature = "cuda-runtime")]
fn cuda_ht_subband_region_jobs(
    job: J2kHtSubbandEncodeJob<'_>,
) -> core::result::Result<Vec<CudaHtj2kEncodeCodeBlockRegionJob>, &'static str> {
    cuda_ht_region_jobs(
        job.width,
        job.height,
        job.code_block_width,
        job.code_block_height,
        job.total_bitplanes,
    )
}

#[cfg(feature = "cuda-runtime")]
fn encoded_ht_code_block_from_cuda(
    encoded: &j2k_cuda_runtime::CudaHtj2kEncodedCodeBlock,
) -> EncodedHtJ2kCodeBlock {
    EncodedHtJ2kCodeBlock {
        data: encoded.data().to_vec(),
        cleanup_length: encoded.cleanup_length(),
        refinement_length: encoded.refinement_length(),
        num_coding_passes: encoded.num_coding_passes(),
        num_zero_bitplanes: encoded.num_zero_bitplanes(),
    }
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn encoded_ht_code_blocks_from_cuda(
    encoded: &j2k_cuda_runtime::CudaHtj2kEncodedCodeBlocks,
) -> Vec<EncodedHtJ2kCodeBlock> {
    encoded
        .code_blocks()
        .iter()
        .map(encoded_ht_code_block_from_cuda)
        .collect()
}

#[cfg(feature = "cuda-runtime")]
pub(super) fn cuda_htj2k_encode_tables() -> CudaHtj2kEncodeTables<'static> {
    CudaHtj2kEncodeTables {
        vlc_table0: j2k_native::ht_vlc_encode_table0(),
        vlc_table1: j2k_native::ht_vlc_encode_table1(),
        uvlc_table: j2k_native::ht_uvlc_encode_table_bytes(),
    }
}
