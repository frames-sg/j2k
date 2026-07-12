// SPDX-License-Identifier: MIT OR Apache-2.0

#[test]
fn packet_builders_reuse_their_single_parsed_view() {
    let source = [
        include_str!("../build.rs"),
        include_str!("../build/gray.rs"),
    ]
    .concat();
    assert_eq!(source.matches("JpegView::parse(bytes)").count(), 2);
    assert_eq!(source.matches("Decoder::from_view(view)").count(), 1);
    assert!(!source.contains("Decoder::new("));
    assert!(!source.contains("parse_header("));
}
