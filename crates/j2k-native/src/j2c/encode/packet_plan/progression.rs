// SPDX-License-Identifier: MIT OR Apache-2.0

//! Order packet contributions by their position on the reference grid.

use super::{
    EncodeParams, EncodeProgressionOrder, J2kPacketizationPacketDescriptor,
    NativeEncodePipelineError, NativeEncodePipelineResult,
};

pub(super) fn sort_packet_descriptors_for_tile(
    descriptors: &mut [J2kPacketizationPacketDescriptor],
    params: &EncodeParams,
    tile_bounds: j2k_types::J2kRect,
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
            let geometry = PacketGeometry::try_new(params, tile_bounds)?;
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
    tile_bounds: j2k_types::J2kRect,
}

impl<'a> PacketGeometry<'a> {
    fn try_new(
        params: &'a EncodeParams,
        tile_bounds: j2k_types::J2kRect,
    ) -> NativeEncodePipelineResult<Self> {
        if tile_bounds.x0 >= tile_bounds.x1
            || tile_bounds.y0 >= tile_bounds.y1
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
            tile_bounds,
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
        let grid = self.packet_grid(packet);
        if packet.precinct >= grid.columns * grid.rows {
            return Err(NativeEncodePipelineError::internal_invariant(
                "packet descriptor exceeds precinct grid",
            ));
        }
        Ok(())
    }

    fn packet_grid(&self, packet: &J2kPacketizationPacketDescriptor) -> PacketGrid {
        let (precinct_x, precinct_y) = self
            .params
            .precinct_exponents
            .get(packet.resolution as usize)
            .copied()
            .unwrap_or((15, 15));
        let (sampling_x, sampling_y) = self
            .params
            .component_sampling
            .get(usize::from(packet.component))
            .copied()
            .unwrap_or((1, 1));
        let (sampling_x, sampling_y) = (u64::from(sampling_x), u64::from(sampling_y));
        let scale = 1_u64 << (u32::from(self.params.num_decomposition_levels) - packet.resolution);
        let (span_x, span_y) = (1_u64 << precinct_x, 1_u64 << precinct_y);
        // T.800 B-12/B-14: round each endpoint in component and then
        // resolution coordinates before finding the zero-anchored precincts.
        let x0 = u64::from(self.tile_bounds.x0)
            .div_ceil(sampling_x)
            .div_ceil(scale);
        let y0 = u64::from(self.tile_bounds.y0)
            .div_ceil(sampling_y)
            .div_ceil(scale);
        let x1 = u64::from(self.tile_bounds.x1)
            .div_ceil(sampling_x)
            .div_ceil(scale);
        let y1 = u64::from(self.tile_bounds.y1)
            .div_ceil(sampling_y)
            .div_ceil(scale);
        PacketGrid {
            first_column: x0 / span_x,
            first_row: y0 / span_y,
            columns: if x0 == x1 {
                0
            } else {
                x1.div_ceil(span_x) - x0 / span_x
            },
            rows: if y0 == y1 {
                0
            } else {
                y1.div_ceil(span_y) - y0 / span_y
            },
            step_x: span_x * scale * sampling_x,
            step_y: span_y * scale * sampling_y,
        }
    }

    fn position(&self, packet: &J2kPacketizationPacketDescriptor) -> (u64, u64) {
        let grid = self.packet_grid(packet);
        // B.12: the first precinct may start inside its grid cell. Its
        // progression position is clipped to the tile's reference-grid origin.
        (
            ((grid.first_column + packet.precinct % grid.columns) * grid.step_x)
                .max(u64::from(self.tile_bounds.x0)),
            ((grid.first_row + packet.precinct / grid.columns) * grid.step_y)
                .max(u64::from(self.tile_bounds.y0)),
        )
    }
}

struct PacketGrid {
    first_column: u64,
    first_row: u64,
    columns: u64,
    rows: u64,
    step_x: u64,
    step_y: u64,
}
