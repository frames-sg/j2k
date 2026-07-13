// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{ImageDecode, J2kDecoder, J2kScratchPool, PixelFormat, Rect};
use j2k_native::{encode, EncodeOptions};

#[test]
fn image_decode_trait_forwards_parse_construction_and_scratch_decode() {
    let expected = [7_u8, 13, 19, 23];
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let codestream = encode(&expected, 2, 2, 1, 8, false, &options).expect("encode fixture");

    let inspected =
        <J2kDecoder<'_> as ImageDecode<'_>>::inspect(&codestream).expect("trait inspect delegates");
    assert_eq!(inspected.dimensions, (2, 2));
    assert_eq!(inspected.components, 1);

    let view =
        <J2kDecoder<'_> as ImageDecode<'_>>::parse(&codestream).expect("trait parse delegates");
    assert_eq!(view.info(), &inspected);
    assert!(core::ptr::eq(view.bytes().as_ptr(), codestream.as_ptr()));

    let mut decoder =
        <J2kDecoder<'_> as ImageDecode<'_>>::from_view(view).expect("trait from_view delegates");
    let mut pool = J2kScratchPool::new();
    let mut output = [0xA5; 6];
    let outcome = <J2kDecoder<'_> as ImageDecode<'_>>::decode_into_with_scratch(
        &mut decoder,
        &mut pool,
        &mut output,
        3,
        PixelFormat::Gray8,
    )
    .expect("trait scratch decode delegates");

    assert_eq!(output, [7, 13, 0xA5, 19, 23, 0xA5]);
    assert_eq!(outcome.decoded, Rect::full((2, 2)));
}
