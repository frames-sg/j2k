// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    validate_band_len, EncodedHtJ2kCodeBlock, J2kForwardDwt53Output, J2kForwardDwt97Output,
    J2kSubBandType, PrecomputedHtj2k53Component, PrecomputedHtj2k53Image,
    PrecomputedHtj2k97Component, PrecomputedHtj2k97Image, PreencodedHtj2k97CompactCodeBlock,
    PreencodedHtj2k97CompactComponent, PreencodedHtj2k97CompactImage,
    PreencodedHtj2k97CompactResolution, PreencodedHtj2k97Component, PreencodedHtj2k97Image,
    PreencodedHtj2k97Resolution, PrequantizedHtj2k97Component, PrequantizedHtj2k97Image,
    PrequantizedHtj2k97Resolution, QuantStepSize,
};

pub(in crate::j2c::encode) fn validate_precomputed_dwt_geometry(
    image: &PrecomputedHtj2k53Image,
) -> Result<(), &'static str> {
    for component in &image.components {
        let component_width = image.width.div_ceil(u32::from(component.x_rsiz));
        let component_height = image.height.div_ceil(u32::from(component.y_rsiz));
        validate_precomputed_component_dwt_geometry(
            &component.dwt,
            component_width,
            component_height,
        )?;
    }

    Ok(())
}

pub(in crate::j2c::encode) fn validate_precomputed_dwt97_geometry(
    image: &PrecomputedHtj2k97Image,
) -> Result<(), &'static str> {
    for component in &image.components {
        let component_width = image.width.div_ceil(u32::from(component.x_rsiz));
        let component_height = image.height.div_ceil(u32::from(component.y_rsiz));
        validate_precomputed_component_dwt_geometry(
            &component.dwt,
            component_width,
            component_height,
        )?;
    }

    Ok(())
}

pub(in crate::j2c::encode) fn validate_precomputed_component_dwt_geometry(
    dwt: &impl PrecomputedDwtGeometryView,
    component_width: u32,
    component_height: u32,
) -> Result<(), &'static str> {
    if let Some(highest_level) = dwt.last_level_geometry() {
        if highest_level.width != component_width || highest_level.height != component_height {
            return Err("precomputed DWT component dimensions mismatch");
        }
    }

    let mut expected_width = component_width;
    let mut expected_height = component_height;
    for level_index in (0..dwt.level_count()).rev() {
        let level = dwt.level_geometry(level_index);
        let low_width = expected_width.div_ceil(2);
        let low_height = expected_height.div_ceil(2);
        let high_width = expected_width / 2;
        let high_height = expected_height / 2;

        if level.width != expected_width
            || level.height != expected_height
            || level.low_width != low_width
            || level.low_height != low_height
            || level.high_width != high_width
            || level.high_height != high_height
        {
            return Err("precomputed DWT recursive geometry mismatch");
        }
        validate_band_len(level.hl_len, high_width, low_height)?;
        validate_band_len(level.lh_len, low_width, high_height)?;
        validate_band_len(level.hh_len, high_width, high_height)?;

        expected_width = low_width;
        expected_height = low_height;
    }

    if dwt.ll_width() != expected_width || dwt.ll_height() != expected_height {
        return Err("precomputed DWT component dimensions mismatch");
    }
    validate_band_len(dwt.ll_len(), expected_width, expected_height)
}

#[derive(Debug, Clone, Copy)]
pub(in crate::j2c::encode) struct PrecomputedDwtLevelGeometry {
    width: u32,
    height: u32,
    low_width: u32,
    low_height: u32,
    high_width: u32,
    high_height: u32,
    hl_len: usize,
    lh_len: usize,
    hh_len: usize,
}

pub(in crate::j2c::encode) trait PrecomputedDwtGeometryView {
    fn ll_len(&self) -> usize;
    fn ll_width(&self) -> u32;
    fn ll_height(&self) -> u32;
    fn level_count(&self) -> usize;
    fn level_geometry(&self, index: usize) -> PrecomputedDwtLevelGeometry;

    fn last_level_geometry(&self) -> Option<PrecomputedDwtLevelGeometry> {
        self.level_count()
            .checked_sub(1)
            .map(|index| self.level_geometry(index))
    }
}

impl PrecomputedDwtGeometryView for J2kForwardDwt53Output {
    fn ll_len(&self) -> usize {
        self.ll.len()
    }

    fn ll_width(&self) -> u32 {
        self.ll_width
    }

    fn ll_height(&self) -> u32 {
        self.ll_height
    }

    fn level_count(&self) -> usize {
        self.levels.len()
    }

    fn level_geometry(&self, index: usize) -> PrecomputedDwtLevelGeometry {
        let level = &self.levels[index];
        PrecomputedDwtLevelGeometry {
            width: level.width,
            height: level.height,
            low_width: level.low_width,
            low_height: level.low_height,
            high_width: level.high_width,
            high_height: level.high_height,
            hl_len: level.hl.len(),
            lh_len: level.lh.len(),
            hh_len: level.hh.len(),
        }
    }
}

impl PrecomputedDwtGeometryView for J2kForwardDwt97Output {
    fn ll_len(&self) -> usize {
        self.ll.len()
    }

    fn ll_width(&self) -> u32 {
        self.ll_width
    }

    fn ll_height(&self) -> u32 {
        self.ll_height
    }

    fn level_count(&self) -> usize {
        self.levels.len()
    }

    fn level_geometry(&self, index: usize) -> PrecomputedDwtLevelGeometry {
        let level = &self.levels[index];
        PrecomputedDwtLevelGeometry {
            width: level.width,
            height: level.height,
            low_width: level.low_width,
            low_height: level.low_height,
            high_width: level.high_width,
            high_height: level.high_height,
            hl_len: level.hl.len(),
            lh_len: level.lh.len(),
            hh_len: level.hh.len(),
        }
    }
}

pub(in crate::j2c::encode) fn uniform_level_count<T>(
    components: &[T],
    len_of: impl Fn(&T) -> usize,
    first_to_levels: impl Fn(usize) -> Result<usize, &'static str>,
    mismatch: &'static str,
) -> Result<u8, &'static str> {
    let first_len = len_of(components.first().ok_or("unsupported component count")?);
    let levels = first_to_levels(first_len)?;
    if components
        .iter()
        .any(|component| len_of(component) != first_len)
    {
        return Err(mismatch);
    }
    u8::try_from(levels).map_err(|_| "decomposition level count exceeds u8")
}

pub(in crate::j2c::encode) fn dwt_levels_only(levels: usize) -> Result<usize, &'static str> {
    Ok(levels)
}

pub(in crate::j2c::encode) fn precomputed_level_count(
    components: &[PrecomputedHtj2k53Component],
) -> Result<u8, &'static str> {
    uniform_level_count(
        components,
        |component| component.dwt.levels.len(),
        dwt_levels_only,
        "precomputed components must use the same decomposition level count",
    )
}

pub(in crate::j2c::encode) fn precomputed_97_level_count(
    components: &[PrecomputedHtj2k97Component],
) -> Result<u8, &'static str> {
    uniform_level_count(
        components,
        |component| component.dwt.levels.len(),
        dwt_levels_only,
        "precomputed components must use the same decomposition level count",
    )
}

pub(in crate::j2c::encode) fn prequantized_97_level_count(
    components: &[PrequantizedHtj2k97Component],
) -> Result<u8, &'static str> {
    uniform_level_count(
        components,
        |component| component.resolutions.len(),
        |len| {
            len.checked_sub(1)
                .ok_or("prequantized components must contain at least one decomposition level")
        },
        "prequantized components must use the same decomposition level count",
    )
}

pub(in crate::j2c::encode) fn preencoded_97_level_count(
    components: &[PreencodedHtj2k97Component],
) -> Result<u8, &'static str> {
    uniform_level_count(
        components,
        |component| component.resolutions.len(),
        |len| {
            len.checked_sub(1)
                .ok_or("preencoded components must contain at least one decomposition level")
        },
        "preencoded components must use the same decomposition level count",
    )
}

pub(in crate::j2c::encode) fn preencoded_compact_97_level_count(
    components: &[PreencodedHtj2k97CompactComponent],
) -> Result<u8, &'static str> {
    uniform_level_count(
        components,
        |component| component.resolutions.len(),
        |len| {
            len.checked_sub(1)
                .ok_or("preencoded components must contain at least one decomposition level")
        },
        "preencoded components must use the same decomposition level count",
    )
}

pub(in crate::j2c::encode) fn validate_prequantized_htj2k97_image(
    image: &PrequantizedHtj2k97Image,
    guard_bits: u8,
    step_sizes: &[QuantStepSize],
) -> Result<(), &'static str> {
    for component in &image.components {
        if component.resolutions.is_empty() {
            return Err("prequantized components must contain at least one resolution");
        }
        validate_prequantized_resolution(
            &component.resolutions[0],
            &[J2kSubBandType::LowLow],
            guard_bits,
            &step_sizes[0..1],
        )?;
        for (level_index, resolution) in component.resolutions.iter().enumerate().skip(1) {
            let step_base = 1 + (level_index - 1) * 3;
            validate_prequantized_resolution(
                resolution,
                &[
                    J2kSubBandType::HighLow,
                    J2kSubBandType::LowHigh,
                    J2kSubBandType::HighHigh,
                ],
                guard_bits,
                &step_sizes[step_base..step_base + 3],
            )?;
        }
    }

    Ok(())
}

pub(in crate::j2c::encode) fn validate_preencoded_htj2k97_image(
    image: &PreencodedHtj2k97Image,
    guard_bits: u8,
    step_sizes: &[QuantStepSize],
) -> Result<(), &'static str> {
    for component in &image.components {
        if component.resolutions.is_empty() {
            return Err("preencoded components must contain at least one resolution");
        }
        validate_preencoded_resolution(
            &component.resolutions[0],
            &[J2kSubBandType::LowLow],
            guard_bits,
            &step_sizes[0..1],
        )?;
        for (level_index, resolution) in component.resolutions.iter().enumerate().skip(1) {
            let step_base = 1 + (level_index - 1) * 3;
            validate_preencoded_resolution(
                resolution,
                &[
                    J2kSubBandType::HighLow,
                    J2kSubBandType::LowHigh,
                    J2kSubBandType::HighHigh,
                ],
                guard_bits,
                &step_sizes[step_base..step_base + 3],
            )?;
        }
    }

    Ok(())
}

pub(in crate::j2c::encode) fn validate_preencoded_compact_htj2k97_image(
    image: &PreencodedHtj2k97CompactImage,
    guard_bits: u8,
    step_sizes: &[QuantStepSize],
) -> Result<(), &'static str> {
    for component in &image.components {
        if component.resolutions.is_empty() {
            return Err("preencoded components must contain at least one resolution");
        }
        validate_preencoded_compact_resolution(
            &component.resolutions[0],
            &[J2kSubBandType::LowLow],
            guard_bits,
            &step_sizes[0..1],
            image.payload.len(),
        )?;
        for (level_index, resolution) in component.resolutions.iter().enumerate().skip(1) {
            let step_base = 1 + (level_index - 1) * 3;
            validate_preencoded_compact_resolution(
                resolution,
                &[
                    J2kSubBandType::HighLow,
                    J2kSubBandType::LowHigh,
                    J2kSubBandType::HighHigh,
                ],
                guard_bits,
                &step_sizes[step_base..step_base + 3],
                image.payload.len(),
            )?;
        }
    }

    Ok(())
}

pub(in crate::j2c::encode) fn validate_prequantized_resolution(
    resolution: &PrequantizedHtj2k97Resolution,
    expected_subbands: &[J2kSubBandType],
    guard_bits: u8,
    step_sizes: &[QuantStepSize],
) -> Result<(), &'static str> {
    if resolution.subbands.len() != expected_subbands.len() {
        return Err("prequantized resolution subband count mismatch");
    }
    for ((subband, expected_subband), step_size) in resolution
        .subbands
        .iter()
        .zip(expected_subbands)
        .zip(step_sizes)
    {
        if subband.sub_band_type != *expected_subband {
            return Err("prequantized resolution subband order mismatch");
        }
        let expected_blocks = subband
            .num_cbs_x
            .checked_mul(subband.num_cbs_y)
            .ok_or("prequantized code-block count overflow")?;
        if expected_blocks == 0 {
            if subband.total_bitplanes != 0 || !subband.code_blocks.is_empty() {
                return Err("empty prequantized subbands must not contain code-block data");
            }
            continue;
        }
        debug_assert!(step_size.exponent <= u16::from(u8::MAX));
        let expected_total_bitplanes = guard_bits
            .saturating_add(step_size.exponent as u8)
            .saturating_sub(1);
        if subband.total_bitplanes != expected_total_bitplanes {
            return Err("prequantized subband bitplane count mismatch");
        }
        if usize::try_from(expected_blocks).map_err(|_| "prequantized code-block count overflow")?
            != subband.code_blocks.len()
        {
            return Err("prequantized code-block count mismatch");
        }
        for block in &subband.code_blocks {
            if block.width == 0 || block.height == 0 {
                return Err("prequantized code-block dimensions must be non-zero");
            }
            validate_band_len(block.coefficients.len(), block.width, block.height)?;
        }
    }

    Ok(())
}

pub(in crate::j2c::encode) fn validate_preencoded_resolution(
    resolution: &PreencodedHtj2k97Resolution,
    expected_subbands: &[J2kSubBandType],
    guard_bits: u8,
    step_sizes: &[QuantStepSize],
) -> Result<(), &'static str> {
    if resolution.subbands.len() != expected_subbands.len() {
        return Err("preencoded resolution subband count mismatch");
    }
    for ((subband, expected_subband), step_size) in resolution
        .subbands
        .iter()
        .zip(expected_subbands)
        .zip(step_sizes)
    {
        if subband.sub_band_type != *expected_subband {
            return Err("preencoded resolution subband order mismatch");
        }
        let expected_blocks = subband
            .num_cbs_x
            .checked_mul(subband.num_cbs_y)
            .ok_or("preencoded code-block count overflow")?;
        if expected_blocks == 0 {
            if subband.total_bitplanes != 0 || !subband.code_blocks.is_empty() {
                return Err("empty preencoded subbands must not contain code-block data");
            }
            continue;
        }
        debug_assert!(step_size.exponent <= u16::from(u8::MAX));
        let expected_total_bitplanes = guard_bits
            .saturating_add(step_size.exponent as u8)
            .saturating_sub(1);
        if subband.total_bitplanes != expected_total_bitplanes {
            return Err("preencoded subband bitplane count mismatch");
        }
        if usize::try_from(expected_blocks).map_err(|_| "preencoded code-block count overflow")?
            != subband.code_blocks.len()
        {
            return Err("preencoded code-block count mismatch");
        }
        for block in &subband.code_blocks {
            if block.width == 0 || block.height == 0 {
                return Err("preencoded code-block dimensions must be non-zero");
            }
            validate_preencoded_code_block_payload(&block.encoded, subband.total_bitplanes)?;
        }
    }

    Ok(())
}

pub(in crate::j2c::encode) fn validate_preencoded_compact_resolution(
    resolution: &PreencodedHtj2k97CompactResolution,
    expected_subbands: &[J2kSubBandType],
    guard_bits: u8,
    step_sizes: &[QuantStepSize],
    payload_len: usize,
) -> Result<(), &'static str> {
    if resolution.subbands.len() != expected_subbands.len() {
        return Err("preencoded resolution subband count mismatch");
    }
    for ((subband, expected_subband), step_size) in resolution
        .subbands
        .iter()
        .zip(expected_subbands)
        .zip(step_sizes)
    {
        if subband.sub_band_type != *expected_subband {
            return Err("preencoded resolution subband order mismatch");
        }
        let expected_blocks = subband
            .num_cbs_x
            .checked_mul(subband.num_cbs_y)
            .ok_or("preencoded code-block count overflow")?;
        if expected_blocks == 0 {
            if subband.total_bitplanes != 0 || !subband.code_blocks.is_empty() {
                return Err("empty preencoded subbands must not contain code-block data");
            }
            continue;
        }
        debug_assert!(step_size.exponent <= u16::from(u8::MAX));
        let expected_total_bitplanes = guard_bits
            .saturating_add(step_size.exponent as u8)
            .saturating_sub(1);
        if subband.total_bitplanes != expected_total_bitplanes {
            return Err("preencoded subband bitplane count mismatch");
        }
        if usize::try_from(expected_blocks).map_err(|_| "preencoded code-block count overflow")?
            != subband.code_blocks.len()
        {
            return Err("preencoded code-block count mismatch");
        }
        for block in &subband.code_blocks {
            if block.width == 0 || block.height == 0 {
                return Err("preencoded code-block dimensions must be non-zero");
            }
            validate_preencoded_compact_code_block_payload(
                block,
                payload_len,
                subband.total_bitplanes,
            )?;
        }
    }

    Ok(())
}

pub(in crate::j2c::encode) fn validate_preencoded_code_block_payload(
    block: &EncodedHtJ2kCodeBlock,
    total_bitplanes: u8,
) -> Result<(), &'static str> {
    let data_len = u32::try_from(block.data.len()).map_err(|_| "HTJ2K payload too large")?;
    if block.num_coding_passes == 0 {
        if data_len != 0 || block.cleanup_length != 0 || block.refinement_length != 0 {
            return Err("empty HTJ2K code-block payload metadata mismatch");
        }
        if block.num_zero_bitplanes != total_bitplanes {
            return Err("empty HTJ2K code-block zero-bitplane count mismatch");
        }
        return Ok(());
    }
    if block.num_coding_passes > 164 {
        return Err("HTJ2K code-block coding pass count out of range");
    }
    if block.num_zero_bitplanes >= total_bitplanes {
        return Err("HTJ2K code-block zero-bitplane count out of range");
    }
    let segment_len = block
        .cleanup_length
        .checked_add(block.refinement_length)
        .ok_or("HTJ2K payload segment length overflow")?;
    if segment_len != data_len {
        return Err("HTJ2K payload segment length mismatch");
    }
    Ok(())
}

pub(in crate::j2c::encode) fn validate_preencoded_compact_code_block_payload(
    block: &PreencodedHtj2k97CompactCodeBlock,
    payload_len: usize,
    total_bitplanes: u8,
) -> Result<(), &'static str> {
    if block.payload_range.start > block.payload_range.end || block.payload_range.end > payload_len
    {
        return Err("HTJ2K payload range out of bounds");
    }
    let data_len = u32::try_from(block.payload_range.end - block.payload_range.start)
        .map_err(|_| "HTJ2K payload too large")?;
    if block.num_coding_passes == 0 {
        if data_len != 0 || block.cleanup_length != 0 || block.refinement_length != 0 {
            return Err("empty HTJ2K code-block payload metadata mismatch");
        }
        if block.num_zero_bitplanes != total_bitplanes {
            return Err("empty HTJ2K code-block zero-bitplane count mismatch");
        }
        return Ok(());
    }
    if block.num_coding_passes > 164 {
        return Err("HTJ2K code-block coding pass count out of range");
    }
    if block.num_zero_bitplanes >= total_bitplanes {
        return Err("HTJ2K code-block zero-bitplane count out of range");
    }
    let segment_len = block
        .cleanup_length
        .checked_add(block.refinement_length)
        .ok_or("HTJ2K payload segment length overflow")?;
    if segment_len != data_len {
        return Err("HTJ2K payload segment length mismatch");
    }
    Ok(())
}
