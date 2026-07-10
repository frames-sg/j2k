// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    htj2k97_subband_total_bitplanes, CudaBufferPool, CudaContext, CudaHtj2k97DeviceCodeblockBands,
    CudaHtj2kCompactEncodedCodeBlock, CudaHtj2kEncodeCodeBlockJob, CudaHtj2kEncodeResidentTarget,
    CudaHtj2kEncodeResources, CudaHtj2kEncodeStageTimings, CudaHtj2kEncodeTables,
    CudaHtj2kEncodedCodeBlock, CudaPooledDeviceBuffer, CudaTranscodeError,
    DctGridI16ToHtj2k97CodeBlockJob, DctGridToHtj2k97CodeBlockJob, EncodedHtJ2kCodeBlock,
    Htj2k97CodeBlockOptions, J2kSubBandType, PreencodedHtj2k97CodeBlock,
    PreencodedHtj2k97CompactCodeBlock, PreencodedHtj2k97CompactComponent,
    PreencodedHtj2k97CompactResolution, PreencodedHtj2k97CompactSubband,
    PreencodedHtj2k97Component, PreencodedHtj2k97Resolution, PreencodedHtj2k97Subband,
    ResidentCompactPreencodedGroups, ResidentPreencodedGroups,
};

pub(super) type ResidentSubbands = (
    Vec<PreencodedHtj2k97Subband>,
    Vec<PreencodedHtj2k97Subband>,
    Vec<PreencodedHtj2k97Subband>,
    Vec<PreencodedHtj2k97Subband>,
    CudaHtj2kEncodeStageTimings,
    usize,
);

pub(super) type CompactResidentSubbands = (
    Vec<u8>,
    Vec<PreencodedHtj2k97CompactSubband>,
    Vec<PreencodedHtj2k97CompactSubband>,
    Vec<PreencodedHtj2k97CompactSubband>,
    Vec<PreencodedHtj2k97CompactSubband>,
    CudaHtj2kEncodeStageTimings,
    usize,
);

pub(super) struct ResidentDeviceGroup<'a, J> {
    pub(super) group_index: usize,
    pub(super) bands: CudaHtj2k97DeviceCodeblockBands,
    pub(super) jobs: &'a [J],
}

pub(super) struct ResidentSubbandEncodePlan<'a> {
    pub(super) coefficients: &'a j2k_cuda_runtime::CudaDeviceBuffer,
    pub(super) coefficient_count: usize,
    pub(super) jobs: Vec<CudaHtj2kEncodeCodeBlockJob>,
    pub(super) shapes: Vec<(u32, u32)>,
    pub(super) sub_band_type: J2kSubBandType,
    pub(super) num_cbs_x: usize,
    pub(super) num_cbs_y: usize,
    pub(super) total_bitplanes: u8,
}

pub(super) struct ResidentSubbandGroupPlans<'a, J> {
    pub(super) group_index: usize,
    pub(super) jobs: &'a [J],
    pub(super) ll: ResidentSubbandEncodePlan<'a>,
    pub(super) hl: ResidentSubbandEncodePlan<'a>,
    pub(super) lh: ResidentSubbandEncodePlan<'a>,
    pub(super) hh: ResidentSubbandEncodePlan<'a>,
}

impl<'a, J> ResidentSubbandGroupPlans<'a, J> {
    fn plans(&self) -> [&ResidentSubbandEncodePlan<'a>; 4] {
        [&self.ll, &self.hl, &self.lh, &self.hh]
    }
}

pub(super) trait Htj2k97ComponentJob {
    fn x_rsiz(&self) -> u8;
    fn y_rsiz(&self) -> u8;
}

impl Htj2k97ComponentJob for DctGridToHtj2k97CodeBlockJob<'_> {
    fn x_rsiz(&self) -> u8 {
        self.x_rsiz
    }

    fn y_rsiz(&self) -> u8 {
        self.y_rsiz
    }
}

impl Htj2k97ComponentJob for DctGridI16ToHtj2k97CodeBlockJob<'_> {
    fn x_rsiz(&self) -> u8 {
        self.x_rsiz
    }

    fn y_rsiz(&self) -> u8 {
        self.y_rsiz
    }
}

#[allow(clippy::similar_names)]
pub(super) fn assemble_preencoded_components<J: Htj2k97ComponentJob>(
    jobs: &[J],
    ll_subbands: Vec<PreencodedHtj2k97Subband>,
    hl_subbands: Vec<PreencodedHtj2k97Subband>,
    lh_subbands: Vec<PreencodedHtj2k97Subband>,
    hh_subbands: Vec<PreencodedHtj2k97Subband>,
) -> Result<Vec<PreencodedHtj2k97Component>, CudaTranscodeError> {
    if ll_subbands.len() != jobs.len()
        || hl_subbands.len() != jobs.len()
        || lh_subbands.len() != jobs.len()
        || hh_subbands.len() != jobs.len()
    {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident HTJ2K component assembly count mismatch",
        ));
    }

    let components = jobs
        .iter()
        .zip(ll_subbands)
        .zip(hl_subbands)
        .zip(lh_subbands)
        .zip(hh_subbands)
        .map(|((((job, ll), hl), lh), hh)| PreencodedHtj2k97Component {
            x_rsiz: job.x_rsiz(),
            y_rsiz: job.y_rsiz(),
            resolutions: vec![
                PreencodedHtj2k97Resolution { subbands: vec![ll] },
                PreencodedHtj2k97Resolution {
                    subbands: vec![hl, lh, hh],
                },
            ],
        })
        .collect();

    Ok(components)
}

#[allow(clippy::similar_names)]
pub(super) fn assemble_compact_preencoded_components<J: Htj2k97ComponentJob>(
    jobs: &[J],
    ll_subbands: Vec<PreencodedHtj2k97CompactSubband>,
    hl_subbands: Vec<PreencodedHtj2k97CompactSubband>,
    lh_subbands: Vec<PreencodedHtj2k97CompactSubband>,
    hh_subbands: Vec<PreencodedHtj2k97CompactSubband>,
) -> Result<Vec<PreencodedHtj2k97CompactComponent>, CudaTranscodeError> {
    if ll_subbands.len() != jobs.len()
        || hl_subbands.len() != jobs.len()
        || lh_subbands.len() != jobs.len()
        || hh_subbands.len() != jobs.len()
    {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident HTJ2K compact component assembly count mismatch",
        ));
    }

    let components = jobs
        .iter()
        .zip(ll_subbands)
        .zip(hl_subbands)
        .zip(lh_subbands)
        .zip(hh_subbands)
        .map(
            |((((job, ll), hl), lh), hh)| PreencodedHtj2k97CompactComponent {
                x_rsiz: job.x_rsiz(),
                y_rsiz: job.y_rsiz(),
                resolutions: vec![
                    PreencodedHtj2k97CompactResolution { subbands: vec![ll] },
                    PreencodedHtj2k97CompactResolution {
                        subbands: vec![hl, lh, hh],
                    },
                ],
            },
        )
        .collect();

    Ok(components)
}

pub(super) fn encode_resident_subbands(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    pool: &CudaBufferPool,
    bands: &CudaHtj2k97DeviceCodeblockBands,
    item_count: usize,
    options: Htj2k97CodeBlockOptions,
) -> Result<ResidentSubbands, CudaTranscodeError> {
    let plans = [
        resident_subband_encode_plan(
            &bands.ll,
            item_count,
            bands.low_width,
            bands.low_height,
            J2kSubBandType::LowLow,
            options,
        )?,
        resident_subband_encode_plan(
            &bands.hl,
            item_count,
            bands.high_width,
            bands.low_height,
            J2kSubBandType::HighLow,
            options,
        )?,
        resident_subband_encode_plan(
            &bands.lh,
            item_count,
            bands.low_width,
            bands.high_height,
            J2kSubBandType::LowHigh,
            options,
        )?,
        resident_subband_encode_plan(
            &bands.hh,
            item_count,
            bands.high_width,
            bands.high_height,
            J2kSubBandType::HighHigh,
            options,
        )?,
    ];
    let targets: Vec<_> = plans
        .iter()
        .filter(|plan| !plan.jobs.is_empty())
        .map(|plan| CudaHtj2kEncodeResidentTarget {
            coefficients: plan.coefficients,
            coefficient_count: plan.coefficient_count,
            jobs: &plan.jobs,
        })
        .collect();
    let encoded = context
        .encode_htj2k_codeblocks_multi_resident_with_resources_and_pool(&targets, resources, pool)
        .map_err(|_| CudaTranscodeError::Kernel("CUDA resident multi-input HTJ2K encode failed"))?;
    let expected_blocks = plans.iter().map(|plan| plan.jobs.len()).sum::<usize>();
    if encoded.code_blocks().len() != expected_blocks {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident multi-input HTJ2K encode returned wrong block count",
        ));
    }
    let ht_timings = encoded.stage_timings();
    let dispatches = encoded.execution().kernel_dispatches();
    let mut encoded_blocks = encoded.into_code_blocks().into_iter();

    let ll = split_resident_subband_blocks(&plans[0], item_count, &mut encoded_blocks)?;
    let hl = split_resident_subband_blocks(&plans[1], item_count, &mut encoded_blocks)?;
    let lh = split_resident_subband_blocks(&plans[2], item_count, &mut encoded_blocks)?;
    let hh = split_resident_subband_blocks(&plans[3], item_count, &mut encoded_blocks)?;
    if encoded_blocks.next().is_some() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident multi-input HTJ2K output count mismatch",
        ));
    }

    Ok((ll, hl, lh, hh, ht_timings, dispatches))
}

pub(super) fn encode_resident_compact_subbands(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    pool: &CudaBufferPool,
    bands: &CudaHtj2k97DeviceCodeblockBands,
    item_count: usize,
    options: Htj2k97CodeBlockOptions,
) -> Result<CompactResidentSubbands, CudaTranscodeError> {
    let plans = [
        resident_subband_encode_plan(
            &bands.ll,
            item_count,
            bands.low_width,
            bands.low_height,
            J2kSubBandType::LowLow,
            options,
        )?,
        resident_subband_encode_plan(
            &bands.hl,
            item_count,
            bands.high_width,
            bands.low_height,
            J2kSubBandType::HighLow,
            options,
        )?,
        resident_subband_encode_plan(
            &bands.lh,
            item_count,
            bands.low_width,
            bands.high_height,
            J2kSubBandType::LowHigh,
            options,
        )?,
        resident_subband_encode_plan(
            &bands.hh,
            item_count,
            bands.high_width,
            bands.high_height,
            J2kSubBandType::HighHigh,
            options,
        )?,
    ];
    let targets: Vec<_> = plans
        .iter()
        .filter(|plan| !plan.jobs.is_empty())
        .map(|plan| CudaHtj2kEncodeResidentTarget {
            coefficients: plan.coefficients,
            coefficient_count: plan.coefficient_count,
            jobs: &plan.jobs,
        })
        .collect();
    let encoded = context
        .encode_htj2k_codeblocks_multi_resident_compact_with_resources_and_pool(
            &targets, resources, pool,
        )
        .map_err(|_| {
            CudaTranscodeError::Kernel("CUDA resident compact multi-input HTJ2K encode failed")
        })?;
    let expected_blocks = plans.iter().map(|plan| plan.jobs.len()).sum::<usize>();
    if encoded.code_blocks().len() != expected_blocks {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident compact multi-input HTJ2K encode returned wrong block count",
        ));
    }
    let ht_timings = encoded.stage_timings();
    let dispatches = encoded.execution().kernel_dispatches();
    let (payload, encoded_blocks) = encoded.into_payload_and_code_blocks();
    let mut encoded_blocks = encoded_blocks.into_iter();

    let ll = split_resident_compact_subband_blocks(&plans[0], item_count, &mut encoded_blocks)?;
    let hl = split_resident_compact_subband_blocks(&plans[1], item_count, &mut encoded_blocks)?;
    let lh = split_resident_compact_subband_blocks(&plans[2], item_count, &mut encoded_blocks)?;
    let hh = split_resident_compact_subband_blocks(&plans[3], item_count, &mut encoded_blocks)?;
    if encoded_blocks.next().is_some() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident compact multi-input HTJ2K output count mismatch",
        ));
    }

    Ok((payload, ll, hl, lh, hh, ht_timings, dispatches))
}

#[allow(clippy::similar_names)]
pub(super) fn device_band_groups_to_preencoded_components<J: Htj2k97ComponentJob>(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    pool: &CudaBufferPool,
    groups: &[ResidentDeviceGroup<'_, J>],
    options: Htj2k97CodeBlockOptions,
) -> Result<ResidentPreencodedGroups, CudaTranscodeError> {
    let group_plans = groups
        .iter()
        .map(|group| {
            if group.bands.item_count != group.jobs.len() {
                return Err(CudaTranscodeError::Kernel(
                    "CUDA grouped resident 9/7 band item count mismatch",
                ));
            }
            Ok(ResidentSubbandGroupPlans {
                group_index: group.group_index,
                jobs: group.jobs,
                ll: resident_subband_encode_plan(
                    &group.bands.ll,
                    group.bands.item_count,
                    group.bands.low_width,
                    group.bands.low_height,
                    J2kSubBandType::LowLow,
                    options,
                )?,
                hl: resident_subband_encode_plan(
                    &group.bands.hl,
                    group.bands.item_count,
                    group.bands.high_width,
                    group.bands.low_height,
                    J2kSubBandType::HighLow,
                    options,
                )?,
                lh: resident_subband_encode_plan(
                    &group.bands.lh,
                    group.bands.item_count,
                    group.bands.low_width,
                    group.bands.high_height,
                    J2kSubBandType::LowHigh,
                    options,
                )?,
                hh: resident_subband_encode_plan(
                    &group.bands.hh,
                    group.bands.item_count,
                    group.bands.high_width,
                    group.bands.high_height,
                    J2kSubBandType::HighHigh,
                    options,
                )?,
            })
        })
        .collect::<Result<Vec<_>, CudaTranscodeError>>()?;

    let targets = group_plans
        .iter()
        .flat_map(ResidentSubbandGroupPlans::plans)
        .filter(|plan| !plan.jobs.is_empty())
        .map(|plan| CudaHtj2kEncodeResidentTarget {
            coefficients: plan.coefficients,
            coefficient_count: plan.coefficient_count,
            jobs: &plan.jobs,
        })
        .collect::<Vec<_>>();
    let encoded = context
        .encode_htj2k_codeblocks_multi_resident_with_resources_and_pool(&targets, resources, pool)
        .map_err(|_| {
            CudaTranscodeError::Kernel("CUDA grouped resident multi-input HTJ2K encode failed")
        })?;
    let expected_blocks = group_plans
        .iter()
        .flat_map(ResidentSubbandGroupPlans::plans)
        .map(|plan| plan.jobs.len())
        .sum::<usize>();
    if encoded.code_blocks().len() != expected_blocks {
        return Err(CudaTranscodeError::Kernel(
            "CUDA grouped resident multi-input HTJ2K encode returned wrong block count",
        ));
    }
    let ht_timings = encoded.stage_timings();
    let dispatches = encoded.execution().kernel_dispatches();
    let mut encoded_blocks = encoded.into_code_blocks().into_iter();
    let mut outputs = Vec::with_capacity(group_plans.len());

    for group in &group_plans {
        let item_count = group.jobs.len();
        let ll = split_resident_subband_blocks(&group.ll, item_count, &mut encoded_blocks)?;
        let hl = split_resident_subband_blocks(&group.hl, item_count, &mut encoded_blocks)?;
        let lh = split_resident_subband_blocks(&group.lh, item_count, &mut encoded_blocks)?;
        let hh = split_resident_subband_blocks(&group.hh, item_count, &mut encoded_blocks)?;
        let components = assemble_preencoded_components(group.jobs, ll, hl, lh, hh)?;
        outputs.push((group.group_index, components));
    }
    if encoded_blocks.next().is_some() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA grouped resident multi-input HTJ2K output count mismatch",
        ));
    }

    Ok((outputs, ht_timings, dispatches))
}

#[allow(clippy::similar_names)]
pub(super) fn device_band_groups_to_compact_preencoded_components<J: Htj2k97ComponentJob>(
    context: &CudaContext,
    resources: &CudaHtj2kEncodeResources,
    pool: &CudaBufferPool,
    groups: &[ResidentDeviceGroup<'_, J>],
    options: Htj2k97CodeBlockOptions,
) -> Result<ResidentCompactPreencodedGroups, CudaTranscodeError> {
    let group_plans = groups
        .iter()
        .map(|group| {
            if group.bands.item_count != group.jobs.len() {
                return Err(CudaTranscodeError::Kernel(
                    "CUDA grouped resident 9/7 band item count mismatch",
                ));
            }
            Ok(ResidentSubbandGroupPlans {
                group_index: group.group_index,
                jobs: group.jobs,
                ll: resident_subband_encode_plan(
                    &group.bands.ll,
                    group.bands.item_count,
                    group.bands.low_width,
                    group.bands.low_height,
                    J2kSubBandType::LowLow,
                    options,
                )?,
                hl: resident_subband_encode_plan(
                    &group.bands.hl,
                    group.bands.item_count,
                    group.bands.high_width,
                    group.bands.low_height,
                    J2kSubBandType::HighLow,
                    options,
                )?,
                lh: resident_subband_encode_plan(
                    &group.bands.lh,
                    group.bands.item_count,
                    group.bands.low_width,
                    group.bands.high_height,
                    J2kSubBandType::LowHigh,
                    options,
                )?,
                hh: resident_subband_encode_plan(
                    &group.bands.hh,
                    group.bands.item_count,
                    group.bands.high_width,
                    group.bands.high_height,
                    J2kSubBandType::HighHigh,
                    options,
                )?,
            })
        })
        .collect::<Result<Vec<_>, CudaTranscodeError>>()?;

    let targets = group_plans
        .iter()
        .flat_map(ResidentSubbandGroupPlans::plans)
        .filter(|plan| !plan.jobs.is_empty())
        .map(|plan| CudaHtj2kEncodeResidentTarget {
            coefficients: plan.coefficients,
            coefficient_count: plan.coefficient_count,
            jobs: &plan.jobs,
        })
        .collect::<Vec<_>>();
    let encoded = context
        .encode_htj2k_codeblocks_multi_resident_compact_with_resources_and_pool(
            &targets, resources, pool,
        )
        .map_err(|_| {
            CudaTranscodeError::Kernel(
                "CUDA grouped resident compact multi-input HTJ2K encode failed",
            )
        })?;
    let expected_blocks = group_plans
        .iter()
        .flat_map(ResidentSubbandGroupPlans::plans)
        .map(|plan| plan.jobs.len())
        .sum::<usize>();
    if encoded.code_blocks().len() != expected_blocks {
        return Err(CudaTranscodeError::Kernel(
            "CUDA grouped resident compact multi-input HTJ2K encode returned wrong block count",
        ));
    }
    let ht_timings = encoded.stage_timings();
    let dispatches = encoded.execution().kernel_dispatches();
    let (payload, encoded_blocks) = encoded.into_payload_and_code_blocks();
    let mut encoded_blocks = encoded_blocks.into_iter();
    let mut outputs = Vec::with_capacity(group_plans.len());

    for group in &group_plans {
        let item_count = group.jobs.len();
        let ll = split_resident_compact_subband_blocks(&group.ll, item_count, &mut encoded_blocks)?;
        let hl = split_resident_compact_subband_blocks(&group.hl, item_count, &mut encoded_blocks)?;
        let lh = split_resident_compact_subband_blocks(&group.lh, item_count, &mut encoded_blocks)?;
        let hh = split_resident_compact_subband_blocks(&group.hh, item_count, &mut encoded_blocks)?;
        let components = assemble_compact_preencoded_components(group.jobs, ll, hl, lh, hh)?;
        outputs.push((group.group_index, components));
    }
    if encoded_blocks.next().is_some() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA grouped resident compact multi-input HTJ2K output count mismatch",
        ));
    }

    Ok((payload, outputs, ht_timings, dispatches))
}

pub(super) fn resident_subband_encode_plan(
    coefficients: &CudaPooledDeviceBuffer,
    item_count: usize,
    width: usize,
    height: usize,
    sub_band_type: J2kSubBandType,
    options: Htj2k97CodeBlockOptions,
) -> Result<ResidentSubbandEncodePlan<'_>, CudaTranscodeError> {
    let coefficient_buffer = coefficients
        .as_device_buffer()
        .ok_or(CudaTranscodeError::Kernel(
            "CUDA resident 9/7 pooled band checkout missing",
        ))?;
    let cb_width = htj2k97_code_block_dim(options.code_block_width_exp)?;
    let cb_height = htj2k97_code_block_dim(options.code_block_height_exp)?;
    let num_cbs_x = if width == 0 {
        0
    } else {
        width.div_ceil(cb_width)
    };
    let num_cbs_y = if height == 0 {
        0
    } else {
        height.div_ceil(cb_height)
    };
    let total_bitplanes = htj2k97_subband_total_bitplanes(options, sub_band_type);
    if width == 0 || height == 0 {
        return Ok(ResidentSubbandEncodePlan {
            coefficients: coefficient_buffer,
            coefficient_count: 0,
            jobs: Vec::new(),
            shapes: Vec::new(),
            sub_band_type,
            num_cbs_x: 0,
            num_cbs_y: 0,
            total_bitplanes,
        });
    }

    let item_stride = width.checked_mul(height).ok_or(CudaTranscodeError::Kernel(
        "CUDA resident 9/7 band dimensions overflow",
    ))?;
    let coefficient_count =
        item_stride
            .checked_mul(item_count)
            .ok_or(CudaTranscodeError::Kernel(
                "CUDA resident 9/7 band item count overflow",
            ))?;
    let mut encode_jobs = Vec::with_capacity(item_count * num_cbs_x * num_cbs_y);
    let mut shapes = Vec::with_capacity(item_count * num_cbs_x * num_cbs_y);
    for item in 0..item_count {
        let item_offset = item
            .checked_mul(item_stride)
            .ok_or(CudaTranscodeError::Kernel(
                "CUDA resident 9/7 band item offset overflow",
            ))?;
        for cby in 0..num_cbs_y {
            for cbx in 0..num_cbs_x {
                let block_width = (width - cbx * cb_width).min(cb_width);
                let block_height = (height - cby * cb_height).min(cb_height);
                let block_offset = cby
                    .checked_mul(cb_height)
                    .and_then(|value| value.checked_mul(width))
                    .and_then(|value| {
                        value.checked_add(cbx.checked_mul(cb_width)?.checked_mul(block_height)?)
                    })
                    .and_then(|value| value.checked_add(item_offset))
                    .ok_or(CudaTranscodeError::Kernel(
                        "CUDA resident 9/7 code-block offset overflow",
                    ))?;
                encode_jobs.push(CudaHtj2kEncodeCodeBlockJob {
                    coefficient_offset: to_u32(block_offset)?,
                    width: to_u32(block_width)?,
                    height: to_u32(block_height)?,
                    total_bitplanes,
                    target_coding_passes: 1,
                });
                shapes.push((to_u32(block_width)?, to_u32(block_height)?));
            }
        }
    }

    Ok(ResidentSubbandEncodePlan {
        coefficients: coefficient_buffer,
        coefficient_count,
        jobs: encode_jobs,
        shapes,
        sub_band_type,
        num_cbs_x,
        num_cbs_y,
        total_bitplanes,
    })
}

pub(super) fn split_resident_subband_blocks(
    plan: &ResidentSubbandEncodePlan<'_>,
    item_count: usize,
    encoded_blocks: &mut impl Iterator<Item = CudaHtj2kEncodedCodeBlock>,
) -> Result<Vec<PreencodedHtj2k97Subband>, CudaTranscodeError> {
    let blocks_per_item =
        plan.num_cbs_x
            .checked_mul(plan.num_cbs_y)
            .ok_or(CudaTranscodeError::Kernel(
                "CUDA resident HTJ2K code-block count overflow",
            ))?;
    let mut shape_index = 0usize;
    let mut subbands = Vec::with_capacity(item_count);
    for _ in 0..item_count {
        let mut code_blocks = Vec::with_capacity(blocks_per_item);
        for _ in 0..blocks_per_item {
            let (width, height) =
                *plan
                    .shapes
                    .get(shape_index)
                    .ok_or(CudaTranscodeError::Kernel(
                        "CUDA resident HTJ2K shape count mismatch",
                    ))?;
            shape_index = shape_index.saturating_add(1);
            let encoded = encoded_blocks.next().ok_or(CudaTranscodeError::Kernel(
                "CUDA resident HTJ2K output count mismatch",
            ))?;
            let (data, cleanup_length, refinement_length, num_coding_passes, num_zero_bitplanes) =
                encoded.into_parts();
            code_blocks.push(PreencodedHtj2k97CodeBlock {
                width,
                height,
                encoded: EncodedHtJ2kCodeBlock {
                    data,
                    cleanup_length,
                    refinement_length,
                    num_coding_passes,
                    num_zero_bitplanes,
                },
            });
        }
        subbands.push(PreencodedHtj2k97Subband {
            sub_band_type: plan.sub_band_type,
            num_cbs_x: to_u32(plan.num_cbs_x)?,
            num_cbs_y: to_u32(plan.num_cbs_y)?,
            total_bitplanes: plan.total_bitplanes,
            code_blocks,
        });
    }
    if shape_index != plan.shapes.len() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident HTJ2K shape count mismatch",
        ));
    }
    Ok(subbands)
}

pub(super) fn split_resident_compact_subband_blocks(
    plan: &ResidentSubbandEncodePlan<'_>,
    item_count: usize,
    encoded_blocks: &mut impl Iterator<Item = CudaHtj2kCompactEncodedCodeBlock>,
) -> Result<Vec<PreencodedHtj2k97CompactSubband>, CudaTranscodeError> {
    let blocks_per_item =
        plan.num_cbs_x
            .checked_mul(plan.num_cbs_y)
            .ok_or(CudaTranscodeError::Kernel(
                "CUDA resident HTJ2K compact code-block count overflow",
            ))?;
    let mut shape_index = 0usize;
    let mut subbands = Vec::with_capacity(item_count);
    for _ in 0..item_count {
        let mut code_blocks = Vec::with_capacity(blocks_per_item);
        for _ in 0..blocks_per_item {
            let (width, height) =
                *plan
                    .shapes
                    .get(shape_index)
                    .ok_or(CudaTranscodeError::Kernel(
                        "CUDA resident HTJ2K compact shape count mismatch",
                    ))?;
            shape_index = shape_index.saturating_add(1);
            let encoded = encoded_blocks.next().ok_or(CudaTranscodeError::Kernel(
                "CUDA resident HTJ2K compact output count mismatch",
            ))?;
            let (
                payload_range,
                cleanup_length,
                refinement_length,
                num_coding_passes,
                num_zero_bitplanes,
            ) = encoded.into_parts();
            code_blocks.push(PreencodedHtj2k97CompactCodeBlock {
                width,
                height,
                payload_range,
                cleanup_length,
                refinement_length,
                num_coding_passes,
                num_zero_bitplanes,
            });
        }
        subbands.push(PreencodedHtj2k97CompactSubband {
            sub_band_type: plan.sub_band_type,
            num_cbs_x: to_u32(plan.num_cbs_x)?,
            num_cbs_y: to_u32(plan.num_cbs_y)?,
            total_bitplanes: plan.total_bitplanes,
            code_blocks,
        });
    }
    if shape_index != plan.shapes.len() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident HTJ2K compact shape count mismatch",
        ));
    }
    Ok(subbands)
}

pub(super) fn to_u32(value: usize) -> Result<u32, CudaTranscodeError> {
    u32::try_from(value).map_err(|_| CudaTranscodeError::Kernel("CUDA value exceeds u32"))
}

pub(super) fn cuda_htj2k_encode_tables() -> CudaHtj2kEncodeTables<'static> {
    CudaHtj2kEncodeTables {
        vlc_table0: j2k_native::ht_vlc_encode_table0(),
        vlc_table1: j2k_native::ht_vlc_encode_table1(),
        uvlc_table: j2k_native::ht_uvlc_encode_table_bytes(),
    }
}

pub(super) fn validate_band_len(
    band: &[i32],
    item_count: usize,
    item_size: usize,
) -> Result<(), CudaTranscodeError> {
    let expected = item_count
        .checked_mul(item_size)
        .ok_or(CudaTranscodeError::Kernel(
            "CUDA 9/7 code-block band length overflow",
        ))?;
    if band.len() != expected {
        return Err(CudaTranscodeError::Kernel(
            "CUDA 9/7 code-block band output length mismatch",
        ));
    }
    Ok(())
}

pub(super) fn validate_htj2k97_codeblock_options(
    options: Htj2k97CodeBlockOptions,
) -> Result<(usize, usize), CudaTranscodeError> {
    j2k_transcode::validate_htj2k97_codeblock_options(options)
        .map_err(CudaTranscodeError::UnsupportedJob)
}

pub(super) fn htj2k97_code_block_dim(exp_minus_two: u8) -> Result<usize, CudaTranscodeError> {
    1usize
        .checked_shl(u32::from(exp_minus_two) + 2)
        .ok_or(CudaTranscodeError::UnsupportedJob(
            "CUDA 9/7 code-block exponent is too large",
        ))
}
