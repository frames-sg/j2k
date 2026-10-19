// SPDX-License-Identifier: MIT OR Apache-2.0

//! Order packet contributions by their position on the reference grid.

use super::{
    EncodeParams, EncodeProgressionOrder, J2kPacketizationPacketDescriptor,
    NativeEncodePipelineError, NativeEncodePipelineResult,
};

pub(super) fn sort_packet_descriptors_for_tile(
    descriptors: &mut [J2kPacketizationPacketDescriptor],
    params: &EncodeParams,
    tile_size: (u32, u32),
) -> NativeEncodePipelineResult<()> {
    match params.progression_order {
        EncodeProgressionOrder::Lrcp => descriptors.sort_unstable_by_key(|packet| {
            (
                packet.layer,
                packet.resolution,
                packet.component,
                packet.precinct,
            )
        }),
        EncodeProgressionOrder::Rlcp => descriptors.sort_unstable_by_key(|packet| {
            (
                packet.resolution,
                packet.layer,
                packet.component,
                packet.precinct,
            )
        }),
        order => {
            let geometry = PacketGeometry::try_new(params, tile_size)?;
            // Validate before sorting, so the key calculation is infallible and
            // needs no separately allocated array of positions or packet owners.
            for packet in descriptors.iter() {
                geometry.validate_packet(packet)?;
            }
            descriptors.sort_unstable_by_key(|packet| {
                let (x, y) = geometry.position(packet);
                let component = u64::from(packet.component);
                let resolution = u64::from(packet.resolution);
                let layer = u64::from(packet.layer);
                let position_key = match order {
                    EncodeProgressionOrder::Lrcp => (layer, resolution, component, 0, 0),
                    EncodeProgressionOrder::Rlcp => (resolution, layer, component, 0, 0),
                    EncodeProgressionOrder::Rpcl => (resolution, y, x, component, layer),
                    EncodeProgressionOrder::Pcrl => (y, x, component, resolution, layer),
                    EncodeProgressionOrder::Cprl => (component, y, x, resolution, layer),
                };
                (position_key, packet.precinct, packet.packet_index)
            });
        }
    }
    Ok(())
}

struct PacketGeometry<'a> {
    params: &'a EncodeParams,
    tile_width: u64,
    tile_height: u64,
}

impl<'a> PacketGeometry<'a> {
    fn try_new(
        params: &'a EncodeParams,
        tile_size: (u32, u32),
    ) -> NativeEncodePipelineResult<Self> {
        if tile_size.0 == 0
            || tile_size.1 == 0
            || params.num_decomposition_levels > 32
            || (!params.precinct_exponents.is_empty()
                && params.precinct_exponents.len()
                    != usize::from(params.num_decomposition_levels) + 1)
            || params
                .precinct_exponents
                .iter()
                .any(|&(x, y)| x > 15 || y > 15)
            || (!params.component_sampling.is_empty()
                && params.component_sampling.len() != usize::from(params.num_components))
            || params
                .component_sampling
                .iter()
                .any(|&(x, y)| x == 0 || y == 0)
        {
            return Err(NativeEncodePipelineError::internal_invariant(
                "invalid packet progression geometry",
            ));
        }
        Ok(Self {
            params,
            tile_width: u64::from(tile_size.0),
            tile_height: u64::from(tile_size.1),
        })
    }

    fn validate_packet(
        &self,
        packet: &J2kPacketizationPacketDescriptor,
    ) -> NativeEncodePipelineResult<()> {
        if packet.component >= self.params.num_components
            || packet.resolution > u32::from(self.params.num_decomposition_levels)
            || packet.layer >= self.params.num_layers
        {
            return Err(NativeEncodePipelineError::internal_invariant(
                "packet descriptor exceeds progression bounds",
            ));
        }
        let (step_x, step_y) = self.packet_grid_steps(packet);
        let (sampling_x, sampling_y) = self.component_sampling(packet);
        let columns = self.tile_width.div_ceil(sampling_x).div_ceil(step_x);
        let rows = self.tile_height.div_ceil(sampling_y).div_ceil(step_y);
        if packet.precinct >= columns * rows {
            return Err(NativeEncodePipelineError::internal_invariant(
                "packet descriptor exceeds precinct grid",
            ));
        }
        Ok(())
    }

    // Precinct steps in a tile-component resolution, as the splitter uses
    // them. At most 32 levels and 15 precinct bits keep both within u64.
    fn packet_grid_steps(&self, packet: &J2kPacketizationPacketDescriptor) -> (u64, u64) {
        let (precinct_x, precinct_y) = self
            .params
            .precinct_exponents
            .get(packet.resolution as usize)
            .copied()
            .unwrap_or((15, 15));
        let level_shift = u32::from(self.params.num_decomposition_levels) - packet.resolution;
        (
            1_u64 << (u32::from(precinct_x) + level_shift),
            1_u64 << (u32::from(precinct_y) + level_shift),
        )
    }

    // The splitter partitions each component's own tile-component extent,
    // ceil(tile / sampling), so precinct indices decode against that grid.
    fn component_sampling(&self, packet: &J2kPacketizationPacketDescriptor) -> (u64, u64) {
        let (sampling_x, sampling_y) = self
            .params
            .component_sampling
            .get(usize::from(packet.component))
            .copied()
            .unwrap_or((1, 1));
        (u64::from(sampling_x), u64::from(sampling_y))
    }

    fn position(&self, packet: &J2kPacketizationPacketDescriptor) -> (u64, u64) {
        let (step_x, step_y) = self.packet_grid_steps(packet);
        let (sampling_x, sampling_y) = self.component_sampling(packet);
        let columns = self.tile_width.div_ceil(sampling_x).div_ceil(step_x);
        // Sampling maps a component's packet position onto the common reference
        // grid. Validated indices bound each result by a u32 extent times u8.
        (
            (packet.precinct % columns) * step_x * sampling_x,
            (packet.precinct / columns) * step_y * sampling_y,
        )
    }
}
