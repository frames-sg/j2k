#![no_main]

//! Lossless encode then decode: every image the encoder accepts with the
//! reversible transform must decode to exactly the input samples, for classic
//! and HTJ2K block coding. The first ten bytes choose the image and the coding
//! options, and the rest supply the samples. Precinct sizes stay at their
//! default until #130 is fixed.

use j2k_native::{
    encode, encode_htj2k, DecodeSettings, EncodeOptions, EncodeProgressionOrder, Image,
};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Some((head, rest)) = data.split_first_chunk::<10>() else {
        return;
    };
    let width = 1 + u32::from(head[0] % 40);
    let height = 1 + u32::from(head[1] % 40);
    let components: u16 = if head[2] & 1 == 0 { 1 } else { 3 };
    let high_throughput = head[2] & 2 != 0;
    let bit_depth = 1 + head[3] % 16;
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: head[4] % 6,
        code_block_width_exp: head[5] % 5,
        code_block_height_exp: (head[5] / 5) % 5,
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
        tile_size: (head[8] != 0)
            .then(|| (1 + u32::from(head[8] % 48), 1 + u32::from(head[9] % 48))),
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
