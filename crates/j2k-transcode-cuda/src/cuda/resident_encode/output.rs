// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    checked_element_product, checked_host_byte_sum, checked_host_bytes,
    CudaHtj2kCompactEncodedCodeBlock, CudaHtj2kEncodedCodeBlock, CudaTranscodeError,
    DctGridI16ToHtj2k97CodeBlockJob, DctGridToHtj2k97CodeBlockJob, EncodedHtJ2kCodeBlock,
    PreencodedHtj2k97CodeBlock, PreencodedHtj2k97CompactCodeBlock,
    PreencodedHtj2k97CompactComponent, PreencodedHtj2k97CompactResolution,
    PreencodedHtj2k97CompactSubband, PreencodedHtj2k97Component, PreencodedHtj2k97Resolution,
    PreencodedHtj2k97Subband,
};
use super::planning::{
    reserve_component_assembly_budget, ResidentMetadataBudget, ResidentSubbandEncodePlan,
};
use super::to_u32;

pub(in crate::cuda) trait Htj2k97ComponentJob {
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

pub(in crate::cuda) fn assemble_preencoded_components<J: Htj2k97ComponentJob>(
    jobs: &[J],
    base_resolution_bands: Vec<PreencodedHtj2k97Subband>,
    horizontal_detail_bands: Vec<PreencodedHtj2k97Subband>,
    vertical_detail_bands: Vec<PreencodedHtj2k97Subband>,
    diagonal_detail_bands: Vec<PreencodedHtj2k97Subband>,
) -> Result<Vec<PreencodedHtj2k97Component>, CudaTranscodeError> {
    let mut budget = ResidentMetadataBudget::new("CUDA resident aggregate metadata");
    account_preencoded_subband_sources(
        &mut budget,
        [
            &base_resolution_bands,
            &horizontal_detail_bands,
            &vertical_detail_bands,
            &diagonal_detail_bands,
        ],
    )?;
    assemble_preencoded_components_with_budget(
        jobs,
        base_resolution_bands,
        horizontal_detail_bands,
        vertical_detail_bands,
        diagonal_detail_bands,
        &mut budget,
    )
}

pub(super) fn assemble_preencoded_components_with_budget<J: Htj2k97ComponentJob>(
    jobs: &[J],
    base_resolution_bands: Vec<PreencodedHtj2k97Subband>,
    horizontal_detail_bands: Vec<PreencodedHtj2k97Subband>,
    vertical_detail_bands: Vec<PreencodedHtj2k97Subband>,
    diagonal_detail_bands: Vec<PreencodedHtj2k97Subband>,
    budget: &mut ResidentMetadataBudget,
) -> Result<Vec<PreencodedHtj2k97Component>, CudaTranscodeError> {
    if base_resolution_bands.len() != jobs.len()
        || horizontal_detail_bands.len() != jobs.len()
        || vertical_detail_bands.len() != jobs.len()
        || diagonal_detail_bands.len() != jobs.len()
    {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident HTJ2K component assembly count mismatch",
        ));
    }
    reserve_component_assembly_budget::<
        PreencodedHtj2k97Component,
        PreencodedHtj2k97Resolution,
        PreencodedHtj2k97Subband,
    >(
        budget,
        jobs.len(),
        "CUDA resident component assembly metadata",
    )?;

    let mut components =
        budget.try_vec_with_capacity(jobs.len(), "CUDA resident preencoded components")?;
    for ((((job, base), horizontal), vertical), diagonal) in jobs
        .iter()
        .zip(base_resolution_bands)
        .zip(horizontal_detail_bands)
        .zip(vertical_detail_bands)
        .zip(diagonal_detail_bands)
    {
        let low_resolution = PreencodedHtj2k97Resolution {
            subbands: budget.try_vec_from_array([base], "CUDA resident LL resolution subbands")?,
        };
        let high_resolution = PreencodedHtj2k97Resolution {
            subbands: budget.try_vec_from_array(
                [horizontal, vertical, diagonal],
                "CUDA resident high-frequency resolution subbands",
            )?,
        };
        components.push(PreencodedHtj2k97Component {
            x_rsiz: job.x_rsiz(),
            y_rsiz: job.y_rsiz(),
            resolutions: budget.try_vec_from_array(
                [low_resolution, high_resolution],
                "CUDA resident component resolutions",
            )?,
        });
    }

    Ok(components)
}

pub(in crate::cuda) fn assemble_compact_preencoded_components<J: Htj2k97ComponentJob>(
    jobs: &[J],
    payload: &Vec<u8>,
    base_resolution_bands: Vec<PreencodedHtj2k97CompactSubband>,
    horizontal_detail_bands: Vec<PreencodedHtj2k97CompactSubband>,
    vertical_detail_bands: Vec<PreencodedHtj2k97CompactSubband>,
    diagonal_detail_bands: Vec<PreencodedHtj2k97CompactSubband>,
) -> Result<Vec<PreencodedHtj2k97CompactComponent>, CudaTranscodeError> {
    let mut budget = ResidentMetadataBudget::new("CUDA compact resident aggregate metadata");
    budget.account_vec(payload)?;
    account_compact_subband_sources(
        &mut budget,
        [
            &base_resolution_bands,
            &horizontal_detail_bands,
            &vertical_detail_bands,
            &diagonal_detail_bands,
        ],
    )?;
    assemble_compact_preencoded_components_with_budget(
        jobs,
        base_resolution_bands,
        horizontal_detail_bands,
        vertical_detail_bands,
        diagonal_detail_bands,
        &mut budget,
    )
}

pub(super) fn assemble_compact_preencoded_components_with_budget<J: Htj2k97ComponentJob>(
    jobs: &[J],
    base_resolution_bands: Vec<PreencodedHtj2k97CompactSubband>,
    horizontal_detail_bands: Vec<PreencodedHtj2k97CompactSubband>,
    vertical_detail_bands: Vec<PreencodedHtj2k97CompactSubband>,
    diagonal_detail_bands: Vec<PreencodedHtj2k97CompactSubband>,
    budget: &mut ResidentMetadataBudget,
) -> Result<Vec<PreencodedHtj2k97CompactComponent>, CudaTranscodeError> {
    if base_resolution_bands.len() != jobs.len()
        || horizontal_detail_bands.len() != jobs.len()
        || vertical_detail_bands.len() != jobs.len()
        || diagonal_detail_bands.len() != jobs.len()
    {
        return Err(CudaTranscodeError::Kernel(
            "CUDA resident HTJ2K compact component assembly count mismatch",
        ));
    }
    reserve_component_assembly_budget::<
        PreencodedHtj2k97CompactComponent,
        PreencodedHtj2k97CompactResolution,
        PreencodedHtj2k97CompactSubband,
    >(
        budget,
        jobs.len(),
        "CUDA compact resident component assembly metadata",
    )?;

    let mut components =
        budget.try_vec_with_capacity(jobs.len(), "CUDA resident compact preencoded components")?;
    for ((((job, base), horizontal), vertical), diagonal) in jobs
        .iter()
        .zip(base_resolution_bands)
        .zip(horizontal_detail_bands)
        .zip(vertical_detail_bands)
        .zip(diagonal_detail_bands)
    {
        let low_resolution = PreencodedHtj2k97CompactResolution {
            subbands: budget
                .try_vec_from_array([base], "CUDA compact resident LL resolution subbands")?,
        };
        let high_resolution = PreencodedHtj2k97CompactResolution {
            subbands: budget.try_vec_from_array(
                [horizontal, vertical, diagonal],
                "CUDA compact resident high-frequency resolution subbands",
            )?,
        };
        components.push(PreencodedHtj2k97CompactComponent {
            x_rsiz: job.x_rsiz(),
            y_rsiz: job.y_rsiz(),
            resolutions: budget.try_vec_from_array(
                [low_resolution, high_resolution],
                "CUDA compact resident component resolutions",
            )?,
        });
    }

    Ok(components)
}

pub(super) fn split_resident_subband_blocks(
    plan: &ResidentSubbandEncodePlan<'_>,
    item_count: usize,
    encoded_blocks: &mut impl Iterator<Item = CudaHtj2kEncodedCodeBlock>,
    budget: &mut ResidentMetadataBudget,
) -> Result<Vec<PreencodedHtj2k97Subband>, CudaTranscodeError> {
    let blocks_per_item =
        plan.num_cbs_x
            .checked_mul(plan.num_cbs_y)
            .ok_or(CudaTranscodeError::Kernel(
                "CUDA resident HTJ2K code-block count overflow",
            ))?;
    preflight_split_metadata::<PreencodedHtj2k97Subband, PreencodedHtj2k97CodeBlock>(
        budget,
        item_count,
        blocks_per_item,
        "CUDA resident HTJ2K code-block metadata",
    )?;
    let mut shape_index = 0usize;
    let mut subbands = budget.try_vec_with_capacity(item_count, "CUDA resident HTJ2K subbands")?;
    for _ in 0..item_count {
        let mut code_blocks = budget
            .try_vec_with_capacity(blocks_per_item, "CUDA resident HTJ2K code-block metadata")?;
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
            budget.account_vec(&data)?;
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
    budget: &mut ResidentMetadataBudget,
) -> Result<Vec<PreencodedHtj2k97CompactSubband>, CudaTranscodeError> {
    let blocks_per_item =
        plan.num_cbs_x
            .checked_mul(plan.num_cbs_y)
            .ok_or(CudaTranscodeError::Kernel(
                "CUDA resident HTJ2K compact code-block count overflow",
            ))?;
    preflight_split_metadata::<PreencodedHtj2k97CompactSubband, PreencodedHtj2k97CompactCodeBlock>(
        budget,
        item_count,
        blocks_per_item,
        "CUDA compact resident HTJ2K code-block metadata",
    )?;
    let mut shape_index = 0usize;
    let mut subbands =
        budget.try_vec_with_capacity(item_count, "CUDA compact resident HTJ2K subbands")?;
    for _ in 0..item_count {
        let mut code_blocks = budget.try_vec_with_capacity(
            blocks_per_item,
            "CUDA compact resident HTJ2K code-block metadata",
        )?;
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

fn account_preencoded_subband_sources(
    budget: &mut ResidentMetadataBudget,
    sources: [&Vec<PreencodedHtj2k97Subband>; 4],
) -> Result<(), CudaTranscodeError> {
    for subbands in sources {
        budget.account_vec(subbands)?;
        for subband in subbands {
            budget.account_vec(&subband.code_blocks)?;
            for block in &subband.code_blocks {
                budget.account_vec(&block.encoded.data)?;
            }
        }
    }
    Ok(())
}

fn preflight_split_metadata<Subband, CodeBlock>(
    budget: &ResidentMetadataBudget,
    item_count: usize,
    blocks_per_item: usize,
    what: &'static str,
) -> Result<(), CudaTranscodeError> {
    let block_count = checked_element_product(&[item_count, blocks_per_item], what)?;
    let additional = checked_host_byte_sum(
        &[
            checked_host_bytes::<Subband>(item_count, what)?,
            checked_host_bytes::<CodeBlock>(block_count, what)?,
        ],
        what,
    )?;
    budget.preflight_bytes(additional)
}

fn account_compact_subband_sources(
    budget: &mut ResidentMetadataBudget,
    sources: [&Vec<PreencodedHtj2k97CompactSubband>; 4],
) -> Result<(), CudaTranscodeError> {
    for subbands in sources {
        budget.account_vec(subbands)?;
        for subband in subbands {
            budget.account_vec(&subband.code_blocks)?;
        }
    }
    Ok(())
}
