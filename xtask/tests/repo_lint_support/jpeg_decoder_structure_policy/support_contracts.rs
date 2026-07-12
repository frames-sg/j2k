// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{assert_pattern_checks, PatternCheck};

pub(super) fn assert_decoder_support_contracts(
    decoder: &str,
    warning_ownership: &str,
    scratch: &str,
    core_traits: &str,
    sink_writer: &str,
    bench_support: &str,
) {
    assert_pattern_checks(&[
        PatternCheck::new("decoder/warning_ownership.rs helpers", warning_ownership).required(&[
            "pub(super) fn merged_warnings",
            "pub(super) fn try_clone_warnings",
        ]),
        PatternCheck::new("decoder.rs warning-owner wiring", decoder)
            .required(&["use self::warning_ownership::{merged_warnings, try_clone_warnings};"])
            .forbidden(&[
                "pub(super) fn merged_warnings",
                "pub(super) fn try_clone_warnings",
            ]),
    ]);
    let scratch_patterns = [
        "pub(super) fn compute_decode_scratch_bytes",
        "pub(super) fn compute_lossless_scratch_bytes",
        "pub(super) fn compute_extended12_planes_scratch_bytes",
        "pub(super) fn checked_scratch_len",
        "pub(super) fn checked_usize_product",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("decoder/scratch.rs helpers", scratch).required(&scratch_patterns),
        PatternCheck::new("decoder.rs scratch helper exclusion", decoder)
            .forbidden(&scratch_patterns),
    ]);
    let core_trait_patterns = [
        "impl ImageCodec for Decoder<'_>",
        "impl TileBatchDecode for JpegCodec",
        "pub(super) struct CroppedWriter",
        "impl<W: ComponentRowWriter + ?Sized> OutputWriter for &mut W",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("decoder/core_traits.rs trait adapters", core_traits)
            .required(&core_trait_patterns)
            .forbidden(&["ComponentWriterAdapter"]),
        PatternCheck::new("decoder.rs core trait adapter exclusion", decoder)
            .forbidden(&core_trait_patterns),
        PatternCheck::new("decoder.rs component writer adapter exclusion", decoder)
            .forbidden(&["ComponentWriterAdapter"]),
        PatternCheck::new("decoder.rs sink writer re-export", decoder)
            .required(&["pub(crate) use self::sink_writer::SinkWriter;"]),
    ]);
    let sink_writer_patterns = [
        "pub(crate) struct SinkWriter",
        "pub(crate) fn into_rows",
        "impl<S> InterleavedRgbWriter for SinkWriter<'_, S>",
        "impl<S> OutputWriter for SinkWriter<'_, S>",
    ];
    assert_pattern_checks(&[
        PatternCheck::new("decoder/sink_writer.rs helpers", sink_writer)
            .required(&sink_writer_patterns),
        PatternCheck::new("decoder.rs sink writer helper exclusion", decoder)
            .forbidden(&sink_writer_patterns),
        PatternCheck::new("bench profile shared sink writer reuse", bench_support)
            .required(&[
                "struct BlackBoxRowSink",
                "impl RowSink<u8> for BlackBoxRowSink",
                "SinkWriter::new(&mut sink, rows, dec.backend)",
            ])
            .forbidden(&[
                "struct BenchProfileSinkWriter",
                "impl InterleavedRgbWriter for BenchProfileSinkWriter",
                "impl OutputWriter for BenchProfileSinkWriter",
            ]),
    ]);
}
