// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained HTJ2K and classic grayscale plan construction.

use super::{
    append_classic_payload_records, append_decode_elements, bail, build_direct_grayscale_tile_plan,
    grayscale_plan_rects, output_region_rect, payload_record_span, referenced_output_region,
    strip_classic_payload_owners, strip_grayscale_payload_owners, tile, tile_intersects_output,
    validate_grayscale_tile, BitReader, ClassicPayloadCollector, DecoderContext, DecodingError,
    Header, J2kReferencedClassicPlan, J2kReferencedHtj2kPlan, J2kReferencedTileGeometry,
    J2kReferencedTilePlan, Result, ValidationError, Vec,
};

pub(crate) fn build_referenced_htj2k_grayscale_plan<'a>(
    data: &'a [u8],
    payload_range_owner: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
) -> Result<J2kReferencedHtj2kPlan> {
    ctx.release_reusable_allocations();
    let result = (|| {
        let mut reader = BitReader::new(data);
        let parsed_tiles = tile::parse(&mut reader, header, retained_image_bytes)?;
        let output_region = referenced_output_region(header, ctx);
        let output_rect = output_region_rect(output_region);
        let mut payloads = Vec::new();
        let mut tile_plans = Vec::new();
        crate::try_reserve_decode_elements(&mut tile_plans, parsed_tiles.len())?;
        let mut next_band_id = 0;

        for tile in parsed_tiles.iter() {
            validate_grayscale_tile(tile)?;
            if !tile_intersects_output(tile, header, output_region)? {
                continue;
            }
            let first_record = payloads.len();
            let mut tile_payloads = Vec::new();
            let mut geometry = build_direct_grayscale_tile_plan(
                data,
                payload_range_owner,
                tile,
                header,
                parsed_tiles.structural_workspace_bytes(),
                ctx,
                &mut next_band_id,
                Some(output_region),
                Some(output_region),
                Some(&mut tile_payloads),
                None,
            )?;
            let job_count = strip_grayscale_payload_owners(&mut geometry)?;
            if job_count != tile_payloads.len() {
                bail!(DecodingError::CodeBlockDecodeFailure);
            }
            append_decode_elements(&mut payloads, &mut tile_payloads)?;
            let payload_records = payload_record_span(first_record, job_count)?;
            let (decoded_rect, destination_rect) = grayscale_plan_rects(&geometry, output_rect)?;
            tile_plans.push(J2kReferencedTilePlan::new(
                usize::try_from(tile.idx).map_err(|_| ValidationError::ImageTooLarge)?,
                decoded_rect,
                destination_rect,
                payload_records,
                super::J2kWaveletTransform::from(tile.component_infos[0].wavelet_transform()),
                J2kReferencedTileGeometry::Grayscale(geometry),
            ));
        }

        Ok(J2kReferencedHtj2kPlan::grayscale(
            tile_plans,
            (
                header.size_data.image_width(),
                header.size_data.image_height(),
            ),
            output_rect,
            payloads,
        ))
    })();
    ctx.release_reusable_allocations();
    result
}

pub(crate) fn build_referenced_classic_grayscale_plan<'a>(
    data: &'a [u8],
    payload_range_owner: &'a [u8],
    header: &Header<'a>,
    retained_image_bytes: usize,
    ctx: &mut DecoderContext<'a>,
) -> Result<J2kReferencedClassicPlan> {
    ctx.release_reusable_allocations();
    let result = (|| {
        let mut reader = BitReader::new(data);
        let parsed_tiles = tile::parse(&mut reader, header, retained_image_bytes)?;
        let output_region = referenced_output_region(header, ctx);
        let output_rect = output_region_rect(output_region);
        let mut payloads = Vec::new();
        let mut ranges = Vec::new();
        let mut tile_plans = Vec::new();
        crate::try_reserve_decode_elements(&mut tile_plans, parsed_tiles.len())?;
        let mut next_band_id = 0;

        for tile in parsed_tiles.iter() {
            validate_grayscale_tile(tile)?;
            if !tile_intersects_output(tile, header, output_region)? {
                continue;
            }
            let first_record = payloads.len();
            let mut tile_payloads = Vec::new();
            let mut tile_ranges = Vec::new();
            let mut collector = ClassicPayloadCollector {
                payloads: &mut tile_payloads,
                ranges: &mut tile_ranges,
            };
            let mut geometry = build_direct_grayscale_tile_plan(
                data,
                payload_range_owner,
                tile,
                header,
                parsed_tiles.structural_workspace_bytes(),
                ctx,
                &mut next_band_id,
                Some(output_region),
                Some(output_region),
                None,
                Some(&mut collector),
            )?;
            let job_count = strip_classic_payload_owners(&mut geometry)?;
            if job_count != tile_payloads.len() {
                bail!(DecodingError::CodeBlockDecodeFailure);
            }
            append_classic_payload_records(
                &mut payloads,
                &mut ranges,
                &mut tile_payloads,
                &mut tile_ranges,
            )?;
            let payload_records = payload_record_span(first_record, job_count)?;
            let (decoded_rect, destination_rect) = grayscale_plan_rects(&geometry, output_rect)?;
            tile_plans.push(J2kReferencedTilePlan::new(
                usize::try_from(tile.idx).map_err(|_| ValidationError::ImageTooLarge)?,
                decoded_rect,
                destination_rect,
                payload_records,
                super::J2kWaveletTransform::from(tile.component_infos[0].wavelet_transform()),
                J2kReferencedTileGeometry::Grayscale(geometry),
            ));
        }

        Ok(J2kReferencedClassicPlan::grayscale(
            tile_plans,
            (
                header.size_data.image_width(),
                header.size_data.image_height(),
            ),
            output_rect,
            payloads,
            ranges,
        ))
    })();
    ctx.release_reusable_allocations();
    result
}
