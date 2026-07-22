// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{
    assert_file_pattern_checks, assert_pattern_checks, repo_root, FilePatternCheck, PatternCheck,
};

#[test]
fn metal_resident_retry_uses_typed_error_classification() {
    let root = repo_root();
    let resident_estimate =
        fs::read_to_string(root.join("crates/j2k-metal/src/encode/resident_estimate.rs"))
            .expect("read resident estimate");
    let metal_error = fs::read_to_string(root.join("crates/j2k-metal/src/error.rs"))
        .expect("read j2k-metal error source");
    let tier1_encode =
        fs::read_to_string(root.join("crates/j2k-metal/src/compute/tier1_encode.rs"))
            .expect("read j2k-metal tier1 encode source");
    let classification_sources = [
        resident_estimate.as_str(),
        metal_error.as_str(),
        tier1_encode.as_str(),
    ]
    .join("\n");

    assert_pattern_checks(&[
        PatternCheck::new("Metal resident retry decision source", &resident_estimate)
            .forbidden(&[".contains("]),
        PatternCheck::new(
            "typed Metal retry classification sources",
            &classification_sources,
        )
        .required(&[
            "MetalKernelRetryable",
            "encode_status_retry_class",
            "ResidentClassicBatch",
            "ResidentHtBatch",
            "is_conservative_retry_candidate",
        ]),
    ]);
}

#[test]
fn gpu_adapter_error_classification_uses_shared_core_impl() {
    let root = repo_root();
    let core_error =
        fs::read_to_string(root.join("crates/j2k-core/src/error.rs")).expect("read core error");
    assert_pattern_checks(&[
        PatternCheck::new("j2k-core adapter error classifier", &core_error).required(&[
            "pub enum AdapterErrorKind",
            "pub trait AdapterErrorParts",
            "adapter_error_is_unsupported",
            "adapter_error_is_buffer_error",
        ]),
    ]);
    let jpeg_metal_lib = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/lib.rs"))
        .expect("read JPEG Metal lib module");
    let jpeg_metal_decode_request =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/decode_request.rs"))
            .expect("read JPEG Metal decode request module");
    let jpeg_metal_decoder = fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/decoder.rs"))
        .expect("read JPEG Metal decoder module");
    let jpeg_metal_codec_batch =
        fs::read_to_string(root.join("crates/j2k-jpeg-metal/src/codec_batch.rs"))
            .expect("read JPEG Metal codec batch module");
    assert!(
        jpeg_metal_lib.lines().count() < 932,
        "j2k-jpeg-metal lib.rs must keep focused public paths re-exported under the post-request-type-split line ratchet"
    );
    assert_pattern_checks(&[
        PatternCheck::new("j2k-jpeg-metal public module shell", &jpeg_metal_lib)
            .required(&[
                "mod error;",
                "pub use error::Error;",
                "mod decode_request;",
                "pub use decode_request::{MetalDecodeOp, MetalDecodeRequest};",
                "mod decoder;",
                "pub use decoder::Decoder;",
                "mod codec_batch;",
                "pub use codec_batch::{",
            ])
            .forbidden(&[
                "pub enum MetalDecodeOp",
                "pub struct MetalDecodeRequest",
                "pub enum Rgb8MetalBatchOp",
                "pub struct Decoder<'a>",
                "impl Codec {",
            ]),
        PatternCheck::new(
            "j2k-jpeg-metal decode request module",
            &jpeg_metal_decode_request,
        )
        .required(&["pub enum MetalDecodeOp", "pub struct MetalDecodeRequest"]),
        PatternCheck::new("j2k-jpeg-metal decoder module", &jpeg_metal_decoder)
            .required(&["pub struct Decoder<'a>", "impl<'a> Decoder<'a>"]),
        PatternCheck::new("j2k-jpeg-metal codec batch module", &jpeg_metal_codec_batch).required(
            &[
                "impl Codec",
                "pub enum Rgb8MetalBatchOp",
                "pub fn inspect_rgb8_decoder_batch_metal_output(",
            ],
        ),
    ]);

    let adapter_classifier_patterns = [
        "impl AdapterErrorParts for Error",
        "adapter_error_is_truncated(self)",
        "adapter_error_is_not_implemented(self)",
        "adapter_error_is_unsupported(self)",
        "adapter_error_is_buffer_error(self)",
    ];
    for relative in [
        "crates/j2k-cuda/src/error.rs",
        "crates/j2k-metal/src/error.rs",
        "crates/j2k-jpeg-cuda/src/error.rs",
        "crates/j2k-jpeg-metal/src/error.rs",
    ] {
        let source = fs::read_to_string(root.join(relative))
            .unwrap_or_else(|err| panic!("read {relative}: {err}"));
        assert_pattern_checks(&[
            PatternCheck::new(relative, &source).required(&adapter_classifier_patterns)
        ]);
    }
}

#[test]
fn gpu_decoder_cpu_host_facades_use_core_blanket_impl() {
    let root = repo_root();
    assert_file_pattern_checks(
        root,
        &[
            FilePatternCheck::new("crates/j2k-core/src/traits.rs")
                .named("j2k-core CPU-backed ImageDecode blanket impl")
                .required(&[
                    "pub trait CpuBackedImageDecode<'a>",
                    "impl<'a, T> ImageDecode<'a> for T",
                    "T: CpuBackedImageDecode<'a>",
                ]),
            FilePatternCheck::new("crates/j2k-cuda/src/decoder/api.rs")
                .required(&["impl<'a> CpuBackedImageDecode<'a>"])
                .forbidden(&["impl<'a> ImageDecode<'a>"]),
            FilePatternCheck::new("crates/j2k-metal/src/decoder/adapters.rs")
                .required(&["impl<'a> CpuBackedImageDecode<'a>"])
                .forbidden(&["impl<'a> ImageDecode<'a>"]),
            FilePatternCheck::new("crates/j2k-jpeg-cuda/src/decoder.rs")
                .required(&["impl<'a> CpuBackedImageDecode<'a>"])
                .forbidden(&["impl<'a> ImageDecode<'a>"]),
            FilePatternCheck::new("crates/j2k-jpeg-metal/src/decoder.rs")
                .required(&["impl<'a> CpuBackedImageDecode<'a>"])
                .forbidden(&["impl<'a> ImageDecode<'a>"]),
        ],
    );
}
