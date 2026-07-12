// SPDX-License-Identifier: MIT OR Apache-2.0

//! Actual-capacity accounting for retained and temporary tile metadata.

use alloc::vec::Vec;
use core::mem::size_of;

use super::{PacketLengthMetadata, Tile, TilePart};
use crate::error::{Result, ValidationError};
#[cfg(test)]
use crate::j2c::codestream::allocation::retained_header_bytes;
use crate::j2c::codestream::{
    CodingStyleComponent, CodingStyleParameters, ComponentInfo, Header, ProgressionChange,
    QuantizationInfo, StepSize,
};
use crate::{try_reserve_decode_elements, DEFAULT_MAX_DECODE_BYTES};

#[derive(Debug)]
pub(super) struct TileMetadataBudget {
    retained_image_bytes: usize,
    retained_bytes: usize,
    cap: usize,
    accounting_valid: bool,
}

impl TileMetadataBudget {
    pub(super) fn for_image(main_header: &Header<'_>, retained_image_bytes: usize) -> Result<Self> {
        let planned_tile_bytes = minimum_inherited_tile_bytes(main_header)?;
        let planned_live_bytes =
            checked_add_metadata_bytes(retained_image_bytes, planned_tile_bytes)?;
        validate_metadata_byte_cap(planned_live_bytes, DEFAULT_MAX_DECODE_BYTES)?;
        Self::with_cap(retained_image_bytes, DEFAULT_MAX_DECODE_BYTES)
    }

    #[cfg(test)]
    pub(super) fn for_header(main_header: &Header<'_>) -> Result<Self> {
        Self::for_image(main_header, retained_header_bytes(main_header)?)
    }

    fn with_cap(retained_image_bytes: usize, cap: usize) -> Result<Self> {
        validate_metadata_byte_cap(retained_image_bytes, cap)?;
        Ok(Self {
            retained_image_bytes,
            retained_bytes: retained_image_bytes,
            cap,
            accounting_valid: true,
        })
    }

    pub(super) fn remaining_bytes(&self) -> usize {
        self.cap.saturating_sub(self.retained_bytes)
    }

    pub(super) fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    pub(super) fn transaction(&mut self) -> TileMetadataTransaction<'_> {
        TileMetadataTransaction {
            budget: self,
            temporary_bytes: 0,
        }
    }

    pub(super) fn try_reserve_retained<T>(
        &mut self,
        values: &mut Vec<T>,
        target_len: usize,
    ) -> Result<()> {
        self.try_reserve_accounted_with(values, target_len, try_reserve_decode_elements)
    }

    fn try_reserve_accounted_with<T>(
        &mut self,
        values: &mut Vec<T>,
        target_len: usize,
        reserve: impl FnOnce(&mut Vec<T>, usize) -> Result<()>,
    ) -> Result<()> {
        self.ensure_accounting_valid()?;
        if target_len <= values.capacity() {
            return Ok(());
        }

        let live_before = self.retained_bytes;
        let old_bytes = checked_vector_bytes::<T>(values.capacity())?;
        let planned_bytes = checked_vector_bytes::<T>(target_len)?;
        validate_transient_peak(live_before, planned_bytes, self.cap)?;

        let reserve_result = reserve(values, target_len);
        let actual_bytes = checked_vector_bytes::<T>(values.capacity())?;
        self.retained_bytes = checked_replacement_bytes(live_before, old_bytes, actual_bytes)?;

        reserve_result?;
        validate_transient_peak(live_before, actual_bytes, self.cap)
    }

    fn account_existing_capacity<T>(&mut self, capacity: usize) -> Result<usize> {
        self.ensure_accounting_valid()?;
        let bytes = checked_vector_bytes::<T>(capacity)?;
        let retained_bytes = checked_add_metadata_bytes(self.retained_bytes, bytes)?;
        validate_metadata_byte_cap(retained_bytes, self.cap)?;
        self.retained_bytes = retained_bytes;
        Ok(bytes)
    }

    fn ensure_accounting_valid(&self) -> Result<()> {
        if !self.accounting_valid {
            return Err(ValidationError::ImageTooLarge.into());
        }
        Ok(())
    }

    pub(super) fn validate_owner_graph(&self, tiles: &Vec<Tile<'_>>) -> Result<()> {
        self.ensure_accounting_valid()?;
        let actual = checked_add_metadata_bytes(
            self.retained_image_bytes,
            tile_owner_allocation_bytes(tiles)?,
        )?;
        validate_metadata_byte_cap(actual, self.cap)?;
        if actual != self.retained_bytes {
            return Err(ValidationError::ImageTooLarge.into());
        }
        Ok(())
    }
}

pub(super) struct TileMetadataTransaction<'a> {
    budget: &'a mut TileMetadataBudget,
    temporary_bytes: usize,
}

impl TileMetadataTransaction<'_> {
    pub(super) fn remaining_bytes(&self) -> usize {
        self.budget.remaining_bytes()
    }

    pub(super) fn try_reserve_retained<T>(
        &mut self,
        values: &mut Vec<T>,
        target_len: usize,
    ) -> Result<()> {
        self.budget.try_reserve_retained(values, target_len)
    }

    pub(super) fn try_reserve_temporary<T>(
        &mut self,
        values: &mut Vec<T>,
        target_len: usize,
    ) -> Result<()> {
        let old_bytes = checked_vector_bytes::<T>(values.capacity())?;
        let reserve_result = self.budget.try_reserve_retained(values, target_len);
        let actual_bytes = checked_vector_bytes::<T>(values.capacity())?;
        self.temporary_bytes =
            checked_replacement_bytes(self.temporary_bytes, old_bytes, actual_bytes)?;
        reserve_result
    }

    pub(super) fn track_temporary_vec<T>(&mut self, values: &Vec<T>) -> Result<()> {
        let bytes = self
            .budget
            .account_existing_capacity::<T>(values.capacity())?;
        self.temporary_bytes = checked_add_metadata_bytes(self.temporary_bytes, bytes)?;
        Ok(())
    }

    pub(super) fn try_copy_temporary<T: Copy>(&mut self, source: &[T]) -> Result<Vec<T>> {
        let mut destination = Vec::new();
        self.try_reserve_temporary(&mut destination, source.len())?;
        destination.extend_from_slice(source);
        Ok(destination)
    }

    pub(super) fn replace_coding_parameters(
        &mut self,
        destination: &mut CodingStyleParameters,
        replacement: CodingStyleParameters,
    ) -> Result<()> {
        self.replace_owner::<(u8, u8), _>(destination, replacement, |parameters| {
            parameters.precinct_exponents.capacity()
        })
    }

    pub(super) fn replace_quantization(
        &mut self,
        destination: &mut QuantizationInfo,
        replacement: QuantizationInfo,
    ) -> Result<()> {
        self.replace_owner::<StepSize, _>(destination, replacement, |quantization| {
            quantization.step_sizes.capacity()
        })
    }

    pub(super) fn append_temporary<T>(
        &mut self,
        destination: &mut Vec<T>,
        mut source: Vec<T>,
    ) -> Result<()> {
        let target_len = destination
            .len()
            .checked_add(source.len())
            .ok_or(ValidationError::ImageTooLarge)?;
        self.try_reserve_retained(destination, target_len)?;
        let source_bytes = checked_vector_bytes::<T>(source.capacity())?;
        let (temporary_bytes, retained_bytes) = self.checked_release(source_bytes)?;
        destination.append(&mut source);
        drop(source);
        self.temporary_bytes = temporary_bytes;
        self.budget.retained_bytes = retained_bytes;
        Ok(())
    }

    pub(super) fn retain_temporary_vec<T>(&mut self, values: &Vec<T>) -> Result<()> {
        let bytes = checked_vector_bytes::<T>(values.capacity())?;
        self.temporary_bytes = self
            .temporary_bytes
            .checked_sub(bytes)
            .ok_or(ValidationError::ImageTooLarge)?;
        Ok(())
    }

    pub(super) fn release_temporary_capacity<T>(&mut self, capacity: usize) -> Result<()> {
        let bytes = checked_vector_bytes::<T>(capacity)?;
        let (temporary_bytes, retained_bytes) = self.checked_release(bytes)?;
        self.temporary_bytes = temporary_bytes;
        self.budget.retained_bytes = retained_bytes;
        Ok(())
    }

    fn replace_owner<T, U>(
        &mut self,
        destination: &mut U,
        replacement: U,
        capacity: impl Fn(&U) -> usize,
    ) -> Result<()> {
        let old_bytes = checked_vector_bytes::<T>(capacity(destination))?;
        let replacement_bytes = checked_vector_bytes::<T>(capacity(&replacement))?;
        let temporary_bytes = self
            .temporary_bytes
            .checked_sub(replacement_bytes)
            .ok_or(ValidationError::ImageTooLarge)?;
        let retained_bytes = self
            .budget
            .retained_bytes
            .checked_sub(old_bytes)
            .ok_or(ValidationError::ImageTooLarge)?;
        *destination = replacement;
        self.temporary_bytes = temporary_bytes;
        self.budget.retained_bytes = retained_bytes;
        Ok(())
    }

    fn checked_release(&self, bytes: usize) -> Result<(usize, usize)> {
        let temporary_bytes = self
            .temporary_bytes
            .checked_sub(bytes)
            .ok_or(ValidationError::ImageTooLarge)?;
        let retained_bytes = self
            .budget
            .retained_bytes
            .checked_sub(bytes)
            .ok_or(ValidationError::ImageTooLarge)?;
        Ok((temporary_bytes, retained_bytes))
    }
}

impl Drop for TileMetadataTransaction<'_> {
    fn drop(&mut self) {
        if self.temporary_bytes == 0 {
            return;
        }
        if let Some(retained_bytes) = self.budget.retained_bytes.checked_sub(self.temporary_bytes) {
            self.budget.retained_bytes = retained_bytes;
        } else {
            self.budget.accounting_valid = false;
        }
    }
}

fn validate_metadata_byte_cap(bytes: usize, cap: usize) -> Result<()> {
    if bytes > cap {
        return Err(ValidationError::ImageTooLarge.into());
    }
    Ok(())
}

fn checked_vector_bytes<T>(len: usize) -> Result<usize> {
    size_of::<T>()
        .checked_mul(len)
        .ok_or(ValidationError::ImageTooLarge.into())
}

fn checked_add_metadata_bytes(total: usize, additional: usize) -> Result<usize> {
    total
        .checked_add(additional)
        .ok_or(ValidationError::ImageTooLarge.into())
}

fn checked_replacement_bytes(total: usize, old: usize, replacement: usize) -> Result<usize> {
    total
        .checked_sub(old)
        .and_then(|bytes| bytes.checked_add(replacement))
        .ok_or(ValidationError::ImageTooLarge.into())
}

fn validate_transient_peak(live_bytes: usize, replacement_bytes: usize, cap: usize) -> Result<()> {
    let peak = checked_add_metadata_bytes(live_bytes, replacement_bytes)?;
    validate_metadata_byte_cap(peak, cap)
}

fn include_capacity<T>(bytes: &mut usize, capacity: usize) -> Result<()> {
    *bytes = checked_add_metadata_bytes(*bytes, checked_vector_bytes::<T>(capacity)?)?;
    Ok(())
}

fn packet_length_capacity(metadata: &PacketLengthMetadata) -> usize {
    metadata.lengths.capacity()
}

fn minimum_inherited_tile_bytes(main_header: &Header<'_>) -> Result<usize> {
    let num_tiles = usize::try_from(main_header.size_data.num_tiles())
        .map_err(|_| ValidationError::ImageTooLarge)?;
    let mut per_tile_bytes =
        checked_vector_bytes::<ComponentInfo>(main_header.component_infos.len())?;
    for component in &main_header.component_infos {
        include_capacity::<(u8, u8)>(
            &mut per_tile_bytes,
            component.coding_style.parameters.precinct_exponents.len(),
        )?;
        include_capacity::<StepSize>(
            &mut per_tile_bytes,
            component.quantization_info.step_sizes.len(),
        )?;
    }
    include_capacity::<ProgressionChange>(
        &mut per_tile_bytes,
        main_header.progression_changes.len(),
    )?;
    let nested_tile_bytes = per_tile_bytes
        .checked_mul(num_tiles)
        .ok_or(ValidationError::ImageTooLarge)?;
    checked_add_metadata_bytes(
        checked_vector_bytes::<Tile<'_>>(num_tiles)?,
        nested_tile_bytes,
    )
}

fn tile_owner_allocation_bytes(tiles: &Vec<Tile<'_>>) -> Result<usize> {
    let mut bytes = checked_vector_bytes::<Tile<'_>>(tiles.capacity())?;
    for tile in tiles {
        include_capacity::<ComponentInfo>(&mut bytes, tile.component_infos.capacity())?;
        for component in &tile.component_infos {
            include_capacity::<(u8, u8)>(
                &mut bytes,
                component
                    .coding_style
                    .parameters
                    .precinct_exponents
                    .capacity(),
            )?;
            include_capacity::<StepSize>(
                &mut bytes,
                component.quantization_info.step_sizes.capacity(),
            )?;
        }
        include_capacity::<ProgressionChange>(&mut bytes, tile.progression_changes.capacity())?;
        include_capacity::<TilePart<'_>>(&mut bytes, tile.tile_parts.capacity())?;
        for tile_part in &tile.tile_parts {
            match tile_part {
                TilePart::Merged(part) => {
                    include_capacity::<u32>(
                        &mut bytes,
                        packet_length_capacity(&part.packet_lengths),
                    )?;
                }
                TilePart::Separated(part) => {
                    include_capacity::<crate::reader::BitReader<'_>>(
                        &mut bytes,
                        part.headers.capacity(),
                    )?;
                    include_capacity::<u32>(
                        &mut bytes,
                        packet_length_capacity(&part.packet_lengths),
                    )?;
                }
            }
        }
    }
    Ok(bytes)
}

pub(super) fn inherit_tile_metadata(
    tile: &mut Tile<'_>,
    header: &Header<'_>,
    budget: &mut TileMetadataBudget,
) -> Result<()> {
    budget.try_reserve_retained(&mut tile.component_infos, header.component_infos.len())?;
    for source in &header.component_infos {
        tile.component_infos.push(ComponentInfo {
            size_info: source.size_info,
            coding_style: CodingStyleComponent {
                flags: source.coding_style.flags,
                parameters: CodingStyleParameters {
                    num_decomposition_levels: source
                        .coding_style
                        .parameters
                        .num_decomposition_levels,
                    num_resolution_levels: source.coding_style.parameters.num_resolution_levels,
                    code_block_width: source.coding_style.parameters.code_block_width,
                    code_block_height: source.coding_style.parameters.code_block_height,
                    code_block_style: source.coding_style.parameters.code_block_style,
                    transformation: source.coding_style.parameters.transformation,
                    precinct_exponents: Vec::new(),
                },
            },
            quantization_info: QuantizationInfo {
                quantization_style: source.quantization_info.quantization_style,
                guard_bits: source.quantization_info.guard_bits,
                step_sizes: Vec::new(),
            },
            roi_shift: source.roi_shift,
        });
        let destination = tile
            .component_infos
            .last_mut()
            .ok_or(ValidationError::InvalidComponentMetadata)?;
        budget.try_reserve_retained(
            &mut destination.coding_style.parameters.precinct_exponents,
            source.coding_style.parameters.precinct_exponents.len(),
        )?;
        destination
            .coding_style
            .parameters
            .precinct_exponents
            .extend_from_slice(&source.coding_style.parameters.precinct_exponents);
        budget.try_reserve_retained(
            &mut destination.quantization_info.step_sizes,
            source.quantization_info.step_sizes.len(),
        )?;
        destination
            .quantization_info
            .step_sizes
            .extend_from_slice(&source.quantization_info.step_sizes);
    }

    budget.try_reserve_retained(
        &mut tile.progression_changes,
        header.progression_changes.len(),
    )?;
    tile.progression_changes
        .extend(header.progression_changes.iter().cloned());
    Ok(())
}

pub(super) fn try_clone_coding_parameters(
    source: &CodingStyleParameters,
    transaction: &mut TileMetadataTransaction<'_>,
) -> Result<CodingStyleParameters> {
    Ok(CodingStyleParameters {
        num_decomposition_levels: source.num_decomposition_levels,
        num_resolution_levels: source.num_resolution_levels,
        code_block_width: source.code_block_width,
        code_block_height: source.code_block_height,
        code_block_style: source.code_block_style,
        transformation: source.transformation,
        precinct_exponents: transaction.try_copy_temporary(&source.precinct_exponents)?,
    })
}

pub(super) fn try_clone_quantization_info(
    source: &QuantizationInfo,
    transaction: &mut TileMetadataTransaction<'_>,
) -> Result<QuantizationInfo> {
    Ok(QuantizationInfo {
        quantization_style: source.quantization_style,
        guard_bits: source.guard_bits,
        step_sizes: transaction.try_copy_temporary(&source.step_sizes)?,
    })
}

#[cfg(test)]
mod accounting_tests;
#[cfg(test)]
mod tests {
    use alloc::vec;

    use super::*;
    use crate::error::DecodeError;
    use crate::j2c::codestream::{
        CodeBlockStyle, CodingStyleComponent, CodingStyleDefault, CodingStyleFlags,
        ComponentSizeInfo, ProgressionOrder, QuantizationStyle, SizeData, WaveletTransform,
    };

    fn component_info(num_decomposition_levels: u8, step_size_count: usize) -> ComponentInfo {
        let num_resolution_levels = num_decomposition_levels + 1;
        ComponentInfo {
            size_info: ComponentSizeInfo {
                precision: 8,
                signed: false,
                horizontal_resolution: 1,
                vertical_resolution: 1,
            },
            coding_style: CodingStyleComponent {
                flags: CodingStyleFlags::default(),
                parameters: CodingStyleParameters {
                    num_decomposition_levels,
                    num_resolution_levels,
                    code_block_width: 6,
                    code_block_height: 6,
                    code_block_style: CodeBlockStyle::default(),
                    transformation: WaveletTransform::Reversible53,
                    precinct_exponents: vec![(15, 15); usize::from(num_resolution_levels)],
                },
            },
            quantization_info: QuantizationInfo {
                quantization_style: QuantizationStyle::NoQuantization,
                guard_bits: 2,
                step_sizes: vec![
                    StepSize {
                        mantissa: 0,
                        exponent: 8,
                    };
                    step_size_count
                ],
            },
            roi_shift: 0,
        }
    }

    fn copy_coding_style(source: &CodingStyleComponent) -> CodingStyleComponent {
        CodingStyleComponent {
            flags: source.flags,
            parameters: CodingStyleParameters {
                num_decomposition_levels: source.parameters.num_decomposition_levels,
                num_resolution_levels: source.parameters.num_resolution_levels,
                code_block_width: source.parameters.code_block_width,
                code_block_height: source.parameters.code_block_height,
                code_block_style: source.parameters.code_block_style,
                transformation: source.parameters.transformation,
                precinct_exponents: Vec::from(source.parameters.precinct_exponents.as_slice()),
            },
        }
    }

    fn copy_component(source: &ComponentInfo) -> ComponentInfo {
        ComponentInfo {
            size_info: source.size_info,
            coding_style: copy_coding_style(&source.coding_style),
            quantization_info: QuantizationInfo {
                quantization_style: source.quantization_info.quantization_style,
                guard_bits: source.quantization_info.guard_bits,
                step_sizes: Vec::from(source.quantization_info.step_sizes.as_slice()),
            },
            roi_shift: source.roi_shift,
        }
    }

    fn header(
        reference_grid_width: u32,
        reference_grid_height: u32,
        component_count: usize,
        component: &ComponentInfo,
    ) -> Header<'static> {
        Header {
            size_data: SizeData {
                reference_grid_width,
                reference_grid_height,
                image_area_x_offset: 0,
                image_area_y_offset: 0,
                tile_width: 1,
                tile_height: 1,
                tile_x_offset: 0,
                tile_y_offset: 0,
                component_sizes: vec![component.size_info; component_count],
                x_shrink_factor: 1,
                y_shrink_factor: 1,
                x_resolution_shrink_factor: 1,
                y_resolution_shrink_factor: 1,
            },
            global_coding_style: CodingStyleDefault {
                progression_order: ProgressionOrder::LayerResolutionComponentPosition,
                num_layers: 1,
                mct: false,
                component_parameters: copy_coding_style(&component.coding_style),
            },
            component_infos: (0..component_count)
                .map(|_| copy_component(component))
                .collect(),
            progression_changes: Vec::new(),
            plm_packet_lengths: Vec::new(),
            ppm_packets: Vec::new(),
            skipped_resolution_levels: 0,
            strict: false,
        }
    }

    #[test]
    fn logical_inheritance_preflight_rejects_deep_tile_amplification() {
        let component = component_info(32, 97);
        let header = header(32_768, 2, 16, &component);

        assert!(
            minimum_inherited_tile_bytes(&header).expect("logical tile graph")
                > DEFAULT_MAX_DECODE_BYTES
        );
        assert_eq!(
            TileMetadataBudget::for_header(&header).expect_err("deep tile graph must reject"),
            DecodeError::Validation(ValidationError::ImageTooLarge)
        );
    }

    #[test]
    fn inherited_tile_graph_matches_allocator_reported_capacities() {
        let component = component_info(2, 7);
        let header = header(1, 1, 2, &component);
        let baseline = retained_header_bytes(&header).expect("header capacity");
        let mut budget = TileMetadataBudget::for_header(&header).expect("tile budget");
        let mut tiles = Vec::new();

        budget
            .try_reserve_retained(&mut tiles, 1)
            .expect("outer tile owner");
        tiles.push(Tile::new(0, &header));
        inherit_tile_metadata(&mut tiles[0], &header, &mut budget)
            .expect("inherited component owners");

        let actual = baseline + tile_owner_allocation_bytes(&tiles).expect("tile owner bytes");
        assert_eq!(budget.retained_bytes(), actual);
        budget
            .validate_owner_graph(&tiles)
            .expect("ledger equals owner graph");
    }
}
