// SPDX-License-Identifier: MIT OR Apache-2.0

use j2k_native::{
    encode, execute_direct_color_plan_rgb8_into, DecodeSettings, DecoderContext, EncodeOptions,
    Image, J2kDirectCpuScratch, J2kRect,
};

/// The direct colour plan adds back each component's level shift, 2^37 for
/// unsigned 38-bit samples, and scales samples deeper than 16 bits to 8 bits
/// over the component's range, as the full-image decode does.
#[test]
fn direct_color_plan_matches_full_decode_for_38_bit_samples() {
    let (width, height) = (16_u32, 16_u32);
    let pixels = (0..u64::from(width * height))
        .flat_map(|index| {
            let (x, y) = (index % u64::from(width), index / u64::from(width));
            [x << 33, y << 33, (x + y) << 32]
        })
        .flat_map(|sample| (sample + (1 << 36)).to_le_bytes()[..5].to_vec())
        .collect::<Vec<_>>();
    let codestream = encode(
        &pixels,
        width,
        height,
        3,
        38,
        false,
        &EncodeOptions {
            reversible: false,
            num_decomposition_levels: 1,
            ..EncodeOptions::default()
        },
    )
    .expect("encode irreversible RGB 38-bit fixture");
    let image = Image::new(&codestream, &DecodeSettings::default()).expect("parse");
    let full = image.decode().expect("full-image RGB8 decode");

    let plan = image
        .build_direct_color_plan_with_context(&mut DecoderContext::default())
        .expect("direct colour plan");
    let mut direct = vec![0_u8; full.len()];
    execute_direct_color_plan_rgb8_into(
        &plan,
        J2kRect {
            x0: 0,
            y0: 0,
            x1: width,
            y1: height,
        },
        &mut J2kDirectCpuScratch::new(),
        &mut direct,
        width as usize * 3,
    )
    .expect("execute direct colour plan");

    assert!(
        full.iter().any(|&sample| sample != 0 && sample != 255),
        "the fixture must exercise the scaling, not only saturation"
    );
    assert_eq!(direct, full);
}
