// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[test]
fn classic_multitile_rgb8_decodes_exactly_on_metal() {
    if !should_run_metal_runtime() {
        return;
    }
    let (encoded, source_pixels) = fixture_classic_multitile_rgb8();
    let dimensions = (19_u32, 13_u32);
    let encoded = Arc::<[u8]>::from(encoded);
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut host_decoder = J2kDecoder::new(encoded.as_ref()).expect("classic CPU decoder");
    let mut expected = vec![0_u8; source_pixels.len()];
    host_decoder
        .decode_into(&mut expected, dimensions.0 as usize * 3, PixelFormat::Rgb8)
        .expect("CPU multi-tile classic RGB8 oracle");
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(vec![EncodedImage::full(encoded)])
        .expect("prepare odd-edge multi-tile classic RGB8 fixture");
    assert!(prepared.errors().is_empty());
    assert_eq!(prepared.groups().len(), 1);
    assert_eq!(
        prepared.groups()[0].images()[0].preparation_depth(),
        PreparationDepth::ClassicOffsetPlan
    );

    let image_len = expected.len();
    let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
        decoder.backend_session().device(),
        image_len + 4,
    )
    .expect("multi-tile classic RGB8 destination");
    let layout = MetalImageLayout::new_batch(
        4,
        dimensions,
        dimensions.0 as usize * 3,
        PixelFormat::Rgb8,
        1,
        image_len,
    )
    .expect("multi-tile classic RGB8 destination layout");
    // SAFETY: the fresh allocation is exclusively offered to the codec call.
    let destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(buffer.clone(), layout)
            .expect("multi-tile classic RGB8 destination guard")
    };
    decoder
        .submit_prepared_group_into(&prepared.groups()[0], destination)
        .expect("submit multi-tile classic RGB8 fixture")
        .wait()
        .expect("complete multi-tile classic RGB8 fixture");
    // SAFETY: completion released the exclusive destination owner.
    let actual = unsafe {
        j2k_metal_support::checked_buffer_read_vec::<u8>(&buffer, 4, image_len)
            .expect("read multi-tile classic RGB8 destination")
    };
    assert_eq!(actual, expected);
}
