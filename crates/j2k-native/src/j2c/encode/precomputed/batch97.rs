// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    encode_prepared_resolution_packets, packet_encode, prepare_precomputed_htj2k97_image_for_batch,
    scalar_packet_descriptors, write_single_tile_packetized_codestream, EncodeOptions,
    J2kEncodeStageAccelerator, PrecomputedHtj2k97Image, PreparedPrecomputedHtj2k97Image, Vec,
};

#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

/// Encode multiple precomputed irreversible 9/7 wavelet images while sharing
/// one HT code-block batch across all prepared tiles.
#[doc(hidden)]
pub fn encode_precomputed_htj2k_97_batch_with_accelerator(
    images: &[PrecomputedHtj2k97Image],
    options: &EncodeOptions,
    accelerator: &mut impl J2kEncodeStageAccelerator,
) -> Result<Vec<Vec<u8>>, &'static str> {
    if images.is_empty() {
        return Ok(Vec::new());
    }
    if options.num_layers != 1 {
        return Err("batch precomputed 9/7 encode currently supports one quality layer");
    }

    let mut prepared_images = prepare_precomputed_htj2k97_images_for_batch(images, options)?;
    let mut all_packets = Vec::new();
    for prepared in &mut prepared_images {
        prepared.packet_count = prepared.prepared_packets.len();
        all_packets.append(&mut prepared.prepared_packets);
    }

    let mut encoded_packets =
        encode_prepared_resolution_packets(all_packets, accelerator)?.into_iter();
    let mut codestreams = Vec::with_capacity(prepared_images.len());
    for prepared in prepared_images {
        let mut resolution_packets = Vec::with_capacity(prepared.packet_count);
        for _ in 0..prepared.packet_count {
            resolution_packets.push(
                encoded_packets
                    .next()
                    .ok_or("encoded packet count mismatch")?,
            );
        }
        let scalar_packet_descriptors = scalar_packet_descriptors(&prepared.packet_descriptors);
        let packetized_tile =
            packet_encode::form_tile_bitstream_with_descriptors_lengths_and_markers(
                &mut resolution_packets,
                &scalar_packet_descriptors,
                packet_encode::PacketMarkerOptions {
                    write_sop: prepared.params.write_sop,
                    write_eph: prepared.params.write_eph,
                    separate_packet_headers: prepared.params.write_ppm || prepared.params.write_ppt,
                },
            )?;
        codestreams.push(write_single_tile_packetized_codestream(
            &prepared.params,
            &packetized_tile,
            &prepared.quant_params,
            options.tile_part_packet_limit,
        )?);
    }
    if encoded_packets.next().is_some() {
        return Err("encoded packet count mismatch");
    }

    Ok(codestreams)
}

#[cfg(feature = "parallel")]
fn prepare_precomputed_htj2k97_images_for_batch(
    images: &[PrecomputedHtj2k97Image],
    options: &EncodeOptions,
) -> Result<Vec<PreparedPrecomputedHtj2k97Image>, &'static str> {
    images
        .par_iter()
        .map(|image| prepare_precomputed_htj2k97_image_for_batch(image, options))
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn prepare_precomputed_htj2k97_images_for_batch(
    images: &[PrecomputedHtj2k97Image],
    options: &EncodeOptions,
) -> Result<Vec<PreparedPrecomputedHtj2k97Image>, &'static str> {
    images
        .iter()
        .map(|image| prepare_precomputed_htj2k97_image_for_batch(image, options))
        .collect()
}
