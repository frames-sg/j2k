// SPDX-License-Identifier: Apache-2.0

use signinum::{
    j2k::{
        encode_j2k_lossless, J2kAdaptiveBackendRequest, J2kAdaptiveOperation,
        J2kAdaptiveRoutePlanner, J2kLosslessEncodeOptions, J2kLosslessSamples,
    },
    tilecodec::UncompressedCodec,
    BackendKind, BackendRequest, CompressedPayloadKind, CompressedTransferSyntax,
    PassthroughCandidate, PassthroughRequirements, TileDecompress,
};

#[test]
fn facade_default_features_are_cpu_portable() {
    let manifest = std::fs::read_to_string(env!("CARGO_MANIFEST_DIR").to_owned() + "/Cargo.toml")
        .expect("read facade manifest");

    assert!(
        manifest.contains("default = []"),
        "signinum facade defaults must be portable on Linux, macOS, and Windows; GPU adapters should be opt-in"
    );
    assert!(
        !manifest.contains("default = [\"metal\"]"),
        "Metal must not be enabled by default for the facade"
    );
}

#[test]
fn facade_prelude_exports_common_user_types() {
    use signinum::prelude::{
        BackendRequest as PreludeBackendRequest, DeflateCodec as PreludeDeflateCodec,
        J2kDecoder as PreludeJ2kDecoder, J2kLosslessEncodeOptions as PreludeJ2kOptions,
        JpegDecoder as PreludeJpegDecoder, LzwCodec as PreludeLzwCodec,
        PixelFormat as PreludePixelFormat, TileDecompress as PreludeTileDecompress,
        UncompressedCodec as PreludeUncompressedCodec, ZstdCodec as PreludeZstdCodec,
    };

    fn assert_tile_decompress<T: PreludeTileDecompress>() {}

    assert_eq!(
        PreludeBackendRequest::default(),
        PreludeBackendRequest::Auto
    );
    assert_eq!(PreludePixelFormat::Rgb8.bytes_per_pixel(), 3);
    let _options = PreludeJ2kOptions::default();
    let _ = std::any::type_name::<PreludeJpegDecoder>();
    let _ = std::any::type_name::<PreludeJ2kDecoder>();
    let _ = std::any::type_name::<PreludeDeflateCodec>();
    let _ = std::any::type_name::<PreludeLzwCodec>();
    let _ = std::any::type_name::<PreludeUncompressedCodec>();
    let _ = std::any::type_name::<PreludeZstdCodec>();
    assert_tile_decompress::<PreludeDeflateCodec>();
    assert_tile_decompress::<PreludeLzwCodec>();
    assert_tile_decompress::<PreludeUncompressedCodec>();
    assert_tile_decompress::<PreludeZstdCodec>();
}

#[test]
fn facade_runtime_backend_default_is_auto() {
    assert_eq!(BackendRequest::default(), BackendRequest::Auto);
    assert_eq!(
        J2kLosslessEncodeOptions::default().backend,
        signinum::EncodeBackendPreference::ACCELERATED
    );
    assert_eq!(
        signinum::EncodeBackendPreference::CPU_ONLY,
        signinum::EncodeBackendPreference::CpuOnly
    );
    assert_eq!(
        signinum::EncodeBackendPreference::STRICT_DEVICE,
        signinum::EncodeBackendPreference::RequireDevice
    );
}

#[test]
fn facade_exports_adaptive_j2k_route_types() {
    let _planner = J2kAdaptiveRoutePlanner::detected();
    assert_eq!(
        J2kAdaptiveBackendRequest::from_backend_request(BackendRequest::Auto),
        J2kAdaptiveBackendRequest::Accelerated
    );
    assert_eq!(J2kAdaptiveOperation::Encode, J2kAdaptiveOperation::Encode);
}

#[test]
fn facade_auto_j2k_lossless_encode_keeps_ungated_small_workloads_on_cpu() {
    let pixels: Vec<u8> = (0..4 * 4 * 3)
        .map(|value| u8::try_from((value * 11) & 0xFF).expect("masked sample fits"))
        .collect();
    let samples = J2kLosslessSamples::new(&pixels, 4, 4, 3, 8, false).expect("valid samples");

    let encoded =
        encode_j2k_lossless(samples, &J2kLosslessEncodeOptions::default()).expect("encode");

    assert_eq!(encoded.backend, BackendKind::Cpu);
    assert!(encoded.codestream.starts_with(&[0xFF, 0x4F]));
}

#[test]
fn facade_exports_tilecodec_contracts() {
    let input = [1, 2, 3, 4];
    let mut output = [0; 4];
    let mut pool = <UncompressedCodec as TileDecompress>::Pool::default();

    let written = UncompressedCodec::decompress_into(&mut pool, &input, &mut output)
        .expect("uncompressed tile copy");

    assert_eq!(written, input.len());
    assert_eq!(output, input);
}

#[test]
fn facade_exports_passthrough_contracts() {
    let info = signinum::core::Info {
        dimensions: (1, 1),
        components: 1,
        colorspace: signinum::core::Colorspace::SGray,
        bit_depth: 8,
        tile_layout: None,
        coded_unit_layout: None,
        restart_interval: None,
        resolution_levels: 1,
    };
    let bytes = [0xff, 0x4f, 0xff, 0xd9];
    let candidate = PassthroughCandidate::new(
        &bytes,
        CompressedTransferSyntax::Jpeg2000Lossless,
        CompressedPayloadKind::Jpeg2000Codestream,
        info,
    );
    let requirements = PassthroughRequirements::new(
        CompressedTransferSyntax::Jpeg2000Lossless,
        CompressedPayloadKind::Jpeg2000Codestream,
    );

    assert_eq!(
        candidate
            .copy_bytes_if_eligible(&requirements)
            .expect("facade passthrough bytes"),
        bytes
    );
}
