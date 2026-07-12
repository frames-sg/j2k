// SPDX-License-Identifier: MIT OR Apache-2.0

//! One-tile extraction, packetization, and ownership transition.

use super::super::allocation::{checked_add_bytes, checked_element_bytes};
use super::super::single_tile::encode_single_tile_packets_impl;
use super::super::tile_parts::{
    consume_packetized_tile_into_tile_parts, encoded_tile_parts_retained_bytes, EncodedTilePart,
};
use super::super::{
    packet_encode, EncodeRoiRegion, J2kEncodeStageAccelerator, NativeEncodePipelineResult, Vec,
};
use super::input::{extract_interleaved_tile, interleaved_tile_output_len, roi_regions_for_tile};
use super::plan::MultiTileLoopPlan;
use super::{append_encoded_tile_parts, MultiTileEncodeRequest};

mod grid;
pub(super) use grid::{TileGrid, TilePosition};

struct PreparedTileInput {
    pixels: Vec<u8>,
    roi_regions: Vec<EncodeRoiRegion>,
    retained_bytes: usize,
}

struct TileIteration<'iteration, 'request, 'input> {
    request: &'iteration MultiTileEncodeRequest<'request, 'input>,
    loop_plan: &'iteration MultiTileLoopPlan,
    position: TilePosition,
    loop_retained_bytes: usize,
    accumulated_bytes: usize,
}

impl<'iteration, 'request, 'input> TileIteration<'iteration, 'request, 'input> {
    fn try_new(
        request: &'iteration MultiTileEncodeRequest<'request, 'input>,
        loop_plan: &'iteration MultiTileLoopPlan,
        position: TilePosition,
        loop_retained_bytes: usize,
        tile_bodies: &Vec<EncodedTilePart>,
    ) -> NativeEncodePipelineResult<Self> {
        let accumulated_bytes =
            encoded_tile_parts_retained_bytes(tile_bodies, tile_bodies.capacity())?;
        Ok(Self {
            request,
            loop_plan,
            position,
            loop_retained_bytes,
            accumulated_bytes,
        })
    }

    fn parent_phase_bytes(
        &self,
        added_bytes: usize,
        context: &'static str,
    ) -> NativeEncodePipelineResult<usize> {
        Ok(checked_add_bytes(
            self.loop_retained_bytes,
            checked_add_bytes(self.accumulated_bytes, added_bytes, context)?,
            context,
        )?)
    }

    fn prepare_input(&self) -> NativeEncodePipelineResult<PreparedTileInput> {
        let requested_pixel_bytes = interleaved_tile_output_len(
            self.position.width,
            self.position.height,
            self.request.num_components,
            self.request.bit_depth,
        )?;
        let requested_roi_bytes = checked_element_bytes::<EncodeRoiRegion>(
            self.request.roi_regions.len(),
            "multi-tile ROI scratch",
        )?;
        let requested_input_bytes = checked_add_bytes(
            requested_pixel_bytes,
            requested_roi_bytes,
            "multi-tile input scratch",
        )?;
        self.request.session.checked_phase(
            self.parent_phase_bytes(requested_input_bytes, "multi-tile input phase")?,
            "multi-tile input phase",
        )?;

        let pixels = extract_interleaved_tile(
            self.request.pixels,
            self.request.width,
            self.position.origin_x,
            self.position.origin_y,
            self.position.width,
            self.position.height,
            self.request.num_components,
            self.request.bit_depth,
        )?;
        let pixel_and_requested_roi = checked_add_bytes(
            pixels.capacity(),
            requested_roi_bytes,
            "multi-tile input scratch",
        )?;
        self.request.session.checked_phase(
            self.parent_phase_bytes(pixel_and_requested_roi, "multi-tile input phase")?,
            "multi-tile input phase",
        )?;

        let roi_regions = roi_regions_for_tile(
            self.request.roi_regions,
            self.position.origin_x,
            self.position.origin_y,
            self.position.width,
            self.position.height,
        )?;
        let retained_bytes = checked_add_bytes(
            pixels.capacity(),
            checked_element_bytes::<EncodeRoiRegion>(
                roi_regions.capacity(),
                "multi-tile ROI scratch",
            )?,
            "multi-tile input scratch",
        )?;
        self.request.session.checked_phase(
            self.parent_phase_bytes(retained_bytes, "multi-tile input phase")?,
            "multi-tile input phase",
        )?;
        Ok(PreparedTileInput {
            pixels,
            roi_regions,
            retained_bytes,
        })
    }

    fn packetize(
        &self,
        tile_bodies: &Vec<EncodedTilePart>,
        input: &PreparedTileInput,
        accelerator: &mut impl J2kEncodeStageAccelerator,
    ) -> NativeEncodePipelineResult<packet_encode::PacketizedTileData> {
        let packetized = {
            let child_phase_owners = (
                self.loop_plan.child_options(),
                tile_bodies,
                &input.pixels,
                &input.roi_regions,
            );
            let child_session = self.request.session.checked_child_session(
                &child_phase_owners,
                self.parent_phase_bytes(input.retained_bytes, "retained multi-tile parent owners")?,
                "retained multi-tile parent owners",
            )?;
            encode_single_tile_packets_impl(
                &input.pixels,
                self.position.width,
                self.position.height,
                self.request.num_components,
                self.request.bit_depth,
                self.request.signed,
                self.loop_plan.child_options(),
                self.request.block_coding_mode,
                &input.roi_regions,
                self.request.component_sample_info,
                &child_session,
                accelerator,
            )?
        };
        let packetized_bytes = packet_encode::packetized_tile_retained_bytes(&packetized)?;
        let input_and_packetized = checked_add_bytes(
            input.retained_bytes,
            packetized_bytes,
            "multi-tile direct packet output",
        )?;
        self.request.session.checked_phase(
            self.parent_phase_bytes(input_and_packetized, "multi-tile direct packet output")?,
            "multi-tile direct packet output",
        )?;
        Ok(packetized)
    }

    fn commit(
        self,
        tile_bodies: &mut Vec<EncodedTilePart>,
        input: PreparedTileInput,
        packetized: packet_encode::PacketizedTileData,
    ) -> NativeEncodePipelineResult<()> {
        drop(input.pixels);
        drop(input.roi_regions);
        let iteration_base = checked_add_bytes(
            self.loop_retained_bytes,
            self.accumulated_bytes,
            "multi-tile retained owners",
        )?;
        let new_parts = consume_packetized_tile_into_tile_parts(
            self.position.index,
            packetized,
            self.request.options.tile_part_packet_limit,
            iteration_base,
            self.request.session,
        )?;
        append_encoded_tile_parts(
            tile_bodies,
            new_parts,
            self.loop_retained_bytes,
            0,
            self.request.session,
        )
    }
}

pub(super) fn encode_tile(
    request: &MultiTileEncodeRequest<'_, '_>,
    loop_plan: &MultiTileLoopPlan,
    loop_retained_bytes: usize,
    tile_bodies: &mut Vec<EncodedTilePart>,
    position: TilePosition,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> NativeEncodePipelineResult<()> {
    let iteration = TileIteration::try_new(
        request,
        loop_plan,
        position,
        loop_retained_bytes,
        tile_bodies,
    )?;
    let input = iteration.prepare_input()?;
    let packetized = iteration.packetize(tile_bodies, &input, accelerator)?;
    iteration.commit(tile_bodies, input, packetized)
}
