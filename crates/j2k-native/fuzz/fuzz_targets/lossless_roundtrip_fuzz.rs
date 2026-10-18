#![no_main]

//! Lossless encode then decode: every image the encoder accepts with the
//! reversible transform must decode to exactly the input samples, for classic
//! and HTJ2K block coding. The first ten bytes choose the image and the coding
//! options, and the rest supply the samples.

use j2k_native::{
    encode, encode_htj2k, DecodeSettings, EncodeOptions, EncodeProgressionOrder, Image,
};
use libfuzzer_sys::fuzz_target;

fn effective_decomposition_levels(
    width: u32,
    height: u32,
    tile_size: Option<(u32, u32)>,
    requested: u8,
) -> u8 {
    let (level_width, level_height) = match tile_size {
        Some((tile_width, tile_height)) if tile_width < width || tile_height < height => {
            let edge_width = if width.is_multiple_of(tile_width) {
                tile_width
            } else {
                width % tile_width
            };
            let edge_height = if height.is_multiple_of(tile_height) {
                tile_height
            } else {
                height % tile_height
            };
            (edge_width, edge_height)
        }
        _ => (width, height),
    };
    let maximum_levels = u8::try_from(level_width.min(level_height).ilog2())
        .expect("a u32 dimension has at most 31 decomposition levels");
    requested.min(maximum_levels)
}

fn custom_precinct_exponents(
    levels: u8,
    code_block_width_exp: u8,
    code_block_height_exp: u8,
) -> Vec<(u8, u8)> {
    (0..=levels)
        .map(|resolution| {
            // Above resolution zero, B-17 and B-18 use PPx - 1 and PPy - 1.
            let detail_adjustment = u8::from(resolution > 0);
            (
                code_block_width_exp + 2 + detail_adjustment,
                code_block_height_exp + 2 + detail_adjustment,
            )
        })
        .collect()
}

fuzz_target!(|data: &[u8]| {
    let Some((head, rest)) = data.split_first_chunk::<10>() else {
        return;
    };
    let width = 1 + u32::from(head[0] % 40);
    let height = 1 + u32::from(head[1] % 40);
    let components: u16 = if head[2] & 1 == 0 { 1 } else { 3 };
    let high_throughput = head[2] & 2 != 0;
    let bit_depth = 1 + head[3] % 16;
    let requested_levels = head[4] % 6;
    let code_block_width_exp = head[5] % 5;
    let code_block_height_exp = (head[5] / 5) % 5;
    let tile_size =
        (head[8] != 0).then(|| (1 + u32::from(head[8] % 48), 1 + u32::from(head[9] % 48)));
    let effective_levels =
        effective_decomposition_levels(width, height, tile_size, requested_levels);
    let precinct_exponents = if head[2] & 64 != 0 {
        custom_precinct_exponents(
            effective_levels,
            code_block_width_exp,
            code_block_height_exp,
        )
    } else {
        Vec::new()
    };
    let options = EncodeOptions {
        reversible: true,
        validate_high_throughput_codestream: false,
        num_decomposition_levels: requested_levels,
        code_block_width_exp,
        code_block_height_exp,
        progression_order: match head[6] % 5 {
            0 => EncodeProgressionOrder::Lrcp,
            1 => EncodeProgressionOrder::Rlcp,
            2 => EncodeProgressionOrder::Rpcl,
            3 => EncodeProgressionOrder::Pcrl,
            _ => EncodeProgressionOrder::Cprl,
        },
        num_layers: 1 + (head[6] / 5) % 3,
        use_mct: components == 3 && head[2] & 4 != 0,
        write_sop: head[7] & 1 != 0,
        write_eph: head[7] & 2 != 0,
        write_plt: head[7] & 4 != 0,
        write_tlm: head[7] & 8 != 0,
        write_plm: head[2] & 8 != 0,
        write_ppm: head[2] & 16 != 0,
        write_ppt: head[2] & 32 != 0,
        tile_part_packet_limit: (head[7] & 16 != 0).then(|| 1 + u16::from(head[7] >> 5)),
        tile_size,
        precinct_exponents,
        ..EncodeOptions::default()
    };

    let samples = (width * height) as usize * usize::from(components);
    let max = (1_u32 << bit_depth) - 1;
    let byte = |i: usize| rest.get(i % rest.len().max(1)).copied().unwrap_or(0);
    let value = |i: usize| (u32::from(byte(2 * i)) | u32::from(byte(2 * i + 1)) << 8) & max;
    let pixels: Vec<u8> = if bit_depth <= 8 {
        (0..samples).map(|i| value(i) as u8).collect()
    } else {
        (0..samples)
            .flat_map(|i| (value(i) as u16).to_le_bytes())
            .collect()
    };

    let encoded = if high_throughput {
        encode_htj2k(
            &pixels, width, height, components, bit_depth, false, &options,
        )
    } else {
        encode(
            &pixels, width, height, components, bit_depth, false, &options,
        )
    };
    let Ok(encoded) = encoded else {
        return;
    };
    let image = Image::new(&encoded, &DecodeSettings::strict())
        .unwrap_or_else(|error| panic!("encoder output does not parse: {error}"));
    let decoded = image
        .decode_native()
        .unwrap_or_else(|error| panic!("encoder output does not decode: {error}"));
    assert!(
        decoded.data == pixels,
        "lossless round trip changed the samples"
    );
});
