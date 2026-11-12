// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// The classic irreversible Tier-1 reconstruction holds magnitudes doubled;
/// its ROI rule must still read the undoubled magnitude (T.800 H.2), as the
/// CPU decoder does.
///
/// The fixture moves `shift` of the encoder's seven guard bits into an RGN
/// marker, so every decoded integer stays as encoded and each coefficient is
/// a region one exactly when it reaches `2^shift`. At the smallest shift whose
/// CPU decode matches the decode without the marker, every coefficient is
/// background and the largest lies in `[2^(shift-1), 2^shift)`, where a rule
/// reading the doubled magnitude takes it for a region one.
#[test]
fn irreversible_classic_roi_background_below_the_shift_matches_cpu_on_metal() {
    if !should_run_metal_runtime() {
        return;
    }

    let (width, height) = (64_u32, 64_u32);
    let pixels = j2k_test_support::gradient_u8(width, height, 1);
    let codestream = encode(
        &pixels,
        width,
        height,
        1,
        8,
        false,
        &EncodeOptions {
            reversible: false,
            guard_bits: 7,
            num_decomposition_levels: 1,
            // Coarse steps keep every coefficient below 2^7, so a shift the
            // guard bits can pay for leaves them all in the background.
            irreversible_quantization_scale: 1024.0,
            ..EncodeOptions::default()
        },
    )
    .expect("encode irreversible classic fixture");
    let without_rgn = cpu_gray8(&codestream, width, height);

    let (shift, bytes) = (1_u8..=7)
        .map(|shift| (shift, guard_bits_moved_into_rgn(&codestream, shift)))
        .find(|(_, bytes)| cpu_gray8(bytes, width, height) == without_rgn)
        .expect("some shift leaves every coefficient in the background");
    assert!(
        shift > 1,
        "a one-bit shift must already reach coefficients, or the fixture tests nothing"
    );

    let options = {
        let mut options = BatchDecodeOptions::default();
        options.layout = BatchLayout::Nhwc;
        options
    };
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("Metal decoder");
    let batch = decoder
        .decode_batch(vec![EncodedImage::full(Arc::from(bytes))])
        .expect("Metal decode");
    assert!(batch.errors().is_empty(), "{:?}", batch.errors());
    assert!(
        batch.group_errors().is_empty(),
        "{:?}",
        batch.group_errors()
    );
    let group = &batch.groups()[0];
    assert!(
        group.dispatch_report().tier1 > 0,
        "the fixture must execute Metal Tier-1"
    );
    assert_eq!(
        group.surfaces()[0]
            .as_bytes()
            .expect("decoded surface bytes"),
        without_rgn,
        "shift {shift}"
    );
}

fn cpu_gray8(codestream: &[u8], width: u32, height: u32) -> Vec<u8> {
    let mut decoder = J2kDecoder::new(codestream).expect("host decoder");
    let mut pixels = vec![0_u8; (width * height) as usize];
    decoder
        .decode_into(&mut pixels, width as usize, PixelFormat::Gray8)
        .expect("host decode");
    pixels
}

/// Insert a main-header RGN marker of `shift` for component 0 and lower the
/// QCD guard bits by as much, which keeps every code-block's bitplane count.
fn guard_bits_moved_into_rgn(codestream: &[u8], shift: u8) -> Vec<u8> {
    let qcd = codestream
        .windows(2)
        .position(|window| window == [0xFF, 0x5C])
        .expect("QCD marker");
    let sqcd = codestream[qcd + 4];
    let guard_bits = sqcd >> 5;
    assert!(
        guard_bits >= shift,
        "guard bits {guard_bits} < shift {shift}"
    );
    let mut bytes = codestream.to_vec();
    bytes[qcd + 4] = ((guard_bits - shift) << 5) | (sqcd & 0x1F);
    let lqcd = usize::from(u16::from_be_bytes([bytes[qcd + 2], bytes[qcd + 3]]));
    let rgn = [0xFF, 0x5E, 0x00, 0x05, 0x00, 0x00, shift];
    bytes.splice(qcd + 2 + lqcd..qcd + 2 + lqcd, rgn);
    bytes
}
