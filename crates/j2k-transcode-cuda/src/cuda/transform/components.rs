// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::{
    checked_element_product, checked_host_byte_sum, checked_host_bytes, htj2k97_code_block_dim,
    htj2k97_subband_total_bitplanes, to_u32, validate_band_len, CudaHtj2k97CodeblockBands,
    CudaTranscodeError, DctGridToHtj2k97CodeBlockJob, HostPhaseBudget, Htj2k97CodeBlockOptions,
    J2kSubBandType, PrequantizedHtj2k97CodeBlock, PrequantizedHtj2k97Component,
    PrequantizedHtj2k97Resolution, PrequantizedHtj2k97Subband,
};

fn code_block_count(
    width: usize,
    height: usize,
    options: Htj2k97CodeBlockOptions,
) -> Result<usize, CudaTranscodeError> {
    let cb_width = htj2k97_code_block_dim(options.code_block_width_exp)?;
    let cb_height = htj2k97_code_block_dim(options.code_block_height_exp)?;
    checked_element_product(
        &[width.div_ceil(cb_width), height.div_ceil(cb_height)],
        "CUDA 9/7 output code-block count",
    )
}

fn preflight_component_allocation_budget(
    budget: &HostPhaseBudget,
    bands: &CudaHtj2k97CodeblockBands,
    item_count: usize,
    options: Htj2k97CodeBlockOptions,
) -> Result<(), CudaTranscodeError> {
    let destination_coefficient_bytes = checked_host_byte_sum(
        &[
            checked_host_bytes::<i32>(bands.ll.len(), "CUDA 9/7 LL coefficient copies")?,
            checked_host_bytes::<i32>(bands.hl.len(), "CUDA 9/7 HL coefficient copies")?,
            checked_host_bytes::<i32>(bands.lh.len(), "CUDA 9/7 LH coefficient copies")?,
            checked_host_bytes::<i32>(bands.hh.len(), "CUDA 9/7 HH coefficient copies")?,
        ],
        "CUDA 9/7 destination coefficient copies",
    )?;
    let resolution_count =
        checked_element_product(&[item_count, 2], "CUDA 9/7 component resolution metadata")?;
    let subband_count =
        checked_element_product(&[item_count, 4], "CUDA 9/7 component subband metadata")?;
    let block_counts = [
        code_block_count(bands.low_width, bands.low_height, options)?,
        code_block_count(bands.high_width, bands.low_height, options)?,
        code_block_count(bands.low_width, bands.high_height, options)?,
        code_block_count(bands.high_width, bands.high_height, options)?,
    ];
    let code_block_bytes = checked_host_byte_sum(
        &[
            checked_host_bytes::<PrequantizedHtj2k97CodeBlock>(
                checked_element_product(
                    &[item_count, block_counts[0]],
                    "CUDA 9/7 LL code-block metadata",
                )?,
                "CUDA 9/7 LL code-block metadata",
            )?,
            checked_host_bytes::<PrequantizedHtj2k97CodeBlock>(
                checked_element_product(
                    &[item_count, block_counts[1]],
                    "CUDA 9/7 HL code-block metadata",
                )?,
                "CUDA 9/7 HL code-block metadata",
            )?,
            checked_host_bytes::<PrequantizedHtj2k97CodeBlock>(
                checked_element_product(
                    &[item_count, block_counts[2]],
                    "CUDA 9/7 LH code-block metadata",
                )?,
                "CUDA 9/7 LH code-block metadata",
            )?,
            checked_host_bytes::<PrequantizedHtj2k97CodeBlock>(
                checked_element_product(
                    &[item_count, block_counts[3]],
                    "CUDA 9/7 HH code-block metadata",
                )?,
                "CUDA 9/7 HH code-block metadata",
            )?,
        ],
        "CUDA 9/7 aggregate code-block metadata",
    )?;
    let additional = checked_host_byte_sum(
        &[
            destination_coefficient_bytes,
            checked_host_bytes::<PrequantizedHtj2k97Component>(
                item_count,
                "CUDA 9/7 component metadata",
            )?,
            checked_host_bytes::<PrequantizedHtj2k97Resolution>(
                resolution_count,
                "CUDA 9/7 resolution metadata",
            )?,
            checked_host_bytes::<PrequantizedHtj2k97Subband>(
                subband_count,
                "CUDA 9/7 subband metadata",
            )?,
            code_block_bytes,
        ],
        "CUDA 9/7 component assembly workspace",
    )?;
    budget.preflight_bytes(additional)
}

/// Reslice one subband's code-block-major `i32` buffer (one item) into a
/// prequantized HTJ2K subband, mirroring the shared code-block oracle layout
/// (outer code-block row, inner code-block column, each block row-major).
fn subband_from_codeblock_slice(
    data: &[i32],
    width: usize,
    height: usize,
    sub_band_type: J2kSubBandType,
    options: Htj2k97CodeBlockOptions,
    budget: &mut HostPhaseBudget,
) -> Result<PrequantizedHtj2k97Subband, CudaTranscodeError> {
    let cb_width = htj2k97_code_block_dim(options.code_block_width_exp)?;
    let cb_height = htj2k97_code_block_dim(options.code_block_height_exp)?;
    let num_cbs_x = width.div_ceil(cb_width);
    let num_cbs_y = height.div_ceil(cb_height);
    let code_block_count = checked_element_product(
        &[num_cbs_x, num_cbs_y],
        "CUDA 9/7 output code-block metadata",
    )?;
    let mut code_blocks =
        budget.try_vec_with_capacity(code_block_count, "CUDA 9/7 output code-block metadata")?;
    let mut offset = 0usize;
    for cby in 0..num_cbs_y {
        for cbx in 0..num_cbs_x {
            let block_width = (width - cbx * cb_width).min(cb_width);
            let block_height = (height - cby * cb_height).min(cb_height);
            let len = block_width * block_height;
            let end = offset.checked_add(len).ok_or(CudaTranscodeError::Kernel(
                "CUDA 9/7 code-block band length overflow",
            ))?;
            if end > data.len() {
                return Err(CudaTranscodeError::Kernel(
                    "CUDA 9/7 code-block band output is shorter than expected",
                ));
            }
            code_blocks.push(PrequantizedHtj2k97CodeBlock {
                coefficients: budget.try_vec_from_slice(
                    &data[offset..end],
                    "CUDA 9/7 prequantized code-block coefficients",
                )?,
                width: to_u32(block_width)?,
                height: to_u32(block_height)?,
            });
            offset = end;
        }
    }
    if offset != data.len() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA 9/7 code-block band output has trailing data",
        ));
    }
    Ok(PrequantizedHtj2k97Subband {
        sub_band_type,
        num_cbs_x: to_u32(num_cbs_x)?,
        num_cbs_y: to_u32(num_cbs_y)?,
        total_bitplanes: htj2k97_subband_total_bitplanes(options, sub_band_type),
        code_blocks,
    })
}

fn item_band_slice(
    band: &[i32],
    item: usize,
    item_size: usize,
) -> Result<&[i32], CudaTranscodeError> {
    let start = item
        .checked_mul(item_size)
        .ok_or(CudaTranscodeError::Kernel(
            "CUDA 9/7 code-block band item offset overflow",
        ))?;
    let end = start
        .checked_add(item_size)
        .ok_or(CudaTranscodeError::Kernel(
            "CUDA 9/7 code-block band item range overflow",
        ))?;
    band.get(start..end).ok_or(CudaTranscodeError::Kernel(
        "CUDA 9/7 code-block band item range mismatch",
    ))
}

fn component_from_subbands(
    job: &DctGridToHtj2k97CodeBlockJob<'_>,
    ll: PrequantizedHtj2k97Subband,
    hl: PrequantizedHtj2k97Subband,
    lh: PrequantizedHtj2k97Subband,
    hh: PrequantizedHtj2k97Subband,
    budget: &mut HostPhaseBudget,
) -> Result<PrequantizedHtj2k97Component, CudaTranscodeError> {
    let low_resolution = PrequantizedHtj2k97Resolution {
        subbands: budget.try_vec_from_array([ll], "CUDA 9/7 LL resolution subbands")?,
    };
    let high_resolution = PrequantizedHtj2k97Resolution {
        subbands: budget
            .try_vec_from_array([hl, lh, hh], "CUDA 9/7 high-frequency resolution subbands")?,
    };
    Ok(PrequantizedHtj2k97Component {
        x_rsiz: job.x_rsiz,
        y_rsiz: job.y_rsiz,
        resolutions: budget.try_vec_from_array(
            [low_resolution, high_resolution],
            "CUDA 9/7 component resolutions",
        )?,
    })
}

/// Reslice the per-item code-block bands into prequantized HTJ2K components,
/// one per job (resolution nesting `[[LL], [HL, LH, HH]]`).
#[expect(
    clippy::similar_names,
    reason = "LL, HL, LH, and HH are standard wavelet subband names"
)]
pub(super) fn codeblock_bands_to_components(
    bands: &CudaHtj2k97CodeblockBands,
    jobs: &[DctGridToHtj2k97CodeBlockJob<'_>],
    options: Htj2k97CodeBlockOptions,
) -> Result<Vec<PrequantizedHtj2k97Component>, CudaTranscodeError> {
    if bands.item_count != jobs.len() {
        return Err(CudaTranscodeError::Kernel(
            "CUDA 9/7 code-block band item count mismatch",
        ));
    }
    let ll_size = checked_element_product(
        &[bands.low_width, bands.low_height],
        "CUDA 9/7 LL band geometry",
    )?;
    let hl_size = checked_element_product(
        &[bands.high_width, bands.low_height],
        "CUDA 9/7 HL band geometry",
    )?;
    let lh_size = checked_element_product(
        &[bands.low_width, bands.high_height],
        "CUDA 9/7 LH band geometry",
    )?;
    let hh_size = checked_element_product(
        &[bands.high_width, bands.high_height],
        "CUDA 9/7 HH band geometry",
    )?;
    validate_band_len(&bands.ll, bands.item_count, ll_size)?;
    validate_band_len(&bands.hl, bands.item_count, hl_size)?;
    validate_band_len(&bands.lh, bands.item_count, lh_size)?;
    validate_band_len(&bands.hh, bands.item_count, hh_size)?;
    let mut budget = HostPhaseBudget::new("CUDA 9/7 component assembly workspace");
    account_codeblock_bands(&mut budget, bands)?;
    preflight_component_allocation_budget(&budget, bands, jobs.len(), options)?;
    let mut components =
        budget.try_vec_with_capacity(jobs.len(), "CUDA 9/7 prequantized components")?;
    for (item, job) in jobs.iter().enumerate() {
        let ll = subband_from_codeblock_slice(
            item_band_slice(&bands.ll, item, ll_size)?,
            bands.low_width,
            bands.low_height,
            J2kSubBandType::LowLow,
            options,
            &mut budget,
        )?;
        let hl = subband_from_codeblock_slice(
            item_band_slice(&bands.hl, item, hl_size)?,
            bands.high_width,
            bands.low_height,
            J2kSubBandType::HighLow,
            options,
            &mut budget,
        )?;
        let lh = subband_from_codeblock_slice(
            item_band_slice(&bands.lh, item, lh_size)?,
            bands.low_width,
            bands.high_height,
            J2kSubBandType::LowHigh,
            options,
            &mut budget,
        )?;
        let hh = subband_from_codeblock_slice(
            item_band_slice(&bands.hh, item, hh_size)?,
            bands.high_width,
            bands.high_height,
            J2kSubBandType::HighHigh,
            options,
            &mut budget,
        )?;
        components.push(component_from_subbands(job, ll, hl, lh, hh, &mut budget)?);
    }
    Ok(components)
}

fn account_codeblock_bands(
    budget: &mut HostPhaseBudget,
    bands: &CudaHtj2k97CodeblockBands,
) -> Result<(), CudaTranscodeError> {
    budget.account_vec(&bands.ll)?;
    budget.account_vec(&bands.hl)?;
    budget.account_vec(&bands.lh)?;
    budget.account_vec(&bands.hh)?;
    Ok(())
}
