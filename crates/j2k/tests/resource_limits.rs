// SPDX-License-Identifier: MIT OR Apache-2.0

//! Decodes that exceed a resource limit fail with `J2kError::ResourceLimit`
//! on every route, apart from invalid input and unsupported features.

use std::sync::Arc;

use j2k::{
    decode_tiles_region_into, prepare_batch, BatchDecodeOptions, BatchItemError, CodecError,
    CpuBatchDecoder, DecodeRequest, EncodedImage, J2kDecoder, J2kError, PixelFormat, Rect,
    TileBatchError, TileBatchOptions, TileRegionDecodeJob,
};
use j2k_native::{encode, EncodeOptions};

/// A small gray codestream whose SIZ claims one `dimension`-square tile.
/// Headers parse; a decode is rejected by its resource limits before it
/// allocates for, or reads the packets of, the claimed tile.
fn large_siz_codestream(dimension: u32) -> Vec<u8> {
    let pixels = (0..=u8::MAX).collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 1,
        ..EncodeOptions::default()
    };
    let mut codestream = encode(&pixels, 16, 16, 1, 8, false, &options).expect("encode");
    let siz = codestream
        .windows(2)
        .position(|marker| marker == [0xff, 0x51])
        .expect("SIZ marker");
    for field in [6, 10, 22, 26] {
        codestream[siz + field..siz + field + 4].copy_from_slice(&dimension.to_be_bytes());
    }
    codestream
}

/// The resource limit `error` reports, after checking it is no other class.
fn resource_limit(error: &J2kError) -> (Option<usize>, Option<usize>) {
    let J2kError::ResourceLimit { requested, cap, .. } = error else {
        panic!("expected a resource limit, got {error:?}");
    };
    assert!(!error.is_truncated());
    assert!(!error.is_unsupported());
    assert!(!error.is_not_implemented());
    assert!(!error.is_buffer_error());
    if let (Some(requested), Some(cap)) = (requested, cap) {
        assert!(requested > cap, "requested {requested} within cap {cap}");
    }
    (*requested, *cap)
}

const TINY_ROI: Rect = Rect {
    x: 0,
    y: 0,
    w: 4,
    h: 4,
};

#[test]
fn decoder_component_planes_over_the_cap_report_requested_and_cap_bytes() {
    let codestream = large_siz_codestream(65_537);
    let mut decoder = J2kDecoder::new(&codestream).expect("large SIZ parses");

    let error = decoder
        .decode_native_components()
        .expect_err("65537x65537 planes exceed the decode cap");

    let (requested, cap) = resource_limit(&error);
    assert!(requested.is_some() && cap.is_some(), "{error:?}");
}

#[test]
fn decode_into_over_the_cap_reports_requested_and_cap_bytes() {
    // 11600x11600 gray needs 538 MB of f32 planes, over the 512 MiB decode
    // cap. The decode is rejected before it writes the caller's buffer.
    let dimension = 11_600_u32;
    let codestream = large_siz_codestream(dimension);
    let mut decoder = J2kDecoder::new(&codestream).expect("large SIZ parses");
    let stride = dimension as usize;
    let mut out = vec![0_u8; stride * stride];

    let error = decoder
        .decode_into(&mut out, stride, PixelFormat::Gray8)
        .expect_err("the planes exceed the decode cap");

    let (requested, cap) = resource_limit(&error);
    assert!(requested.is_some() && cap.is_some(), "{error:?}");
}

#[test]
fn decoder_region_over_the_metadata_limit_is_a_resource_limit() {
    let codestream = large_siz_codestream(1 << 24);
    let mut decoder = J2kDecoder::new(&codestream).expect("large SIZ parses");
    let mut pool = j2k::J2kScratchPool::new();
    let stride = TINY_ROI.w as usize;
    let mut out = vec![0_u8; stride * TINY_ROI.h as usize];

    let error = decoder
        .decode_region_into(&mut pool, &mut out, stride, PixelFormat::Gray8, TINY_ROI)
        .expect_err("a 2^24-square tile's code-block metadata exceeds the cap");

    resource_limit(&error);
}

#[test]
fn tile_batch_errors_keep_the_resource_limit_with_the_tile_index() {
    let codestream = large_siz_codestream(1 << 24);
    let stride = TINY_ROI.w as usize;
    let mut out = vec![0_u8; stride * TINY_ROI.h as usize];
    let mut jobs = [TileRegionDecodeJob {
        input: &codestream,
        out: &mut out,
        stride,
        roi: TINY_ROI,
    }];

    let error =
        decode_tiles_region_into(&mut jobs, PixelFormat::Gray8, TileBatchOptions::new(None))
            .expect_err("the tile exceeds the decode cap");

    let TileBatchError::Tile(tile) = &error else {
        panic!("expected a tile failure, got {error:?}");
    };
    assert_eq!(tile.index, 0);
    resource_limit(&tile.source);
}

#[test]
fn owned_batch_item_errors_keep_the_resource_limit() {
    let codestream = Arc::<[u8]>::from(large_siz_codestream(1 << 24));
    let input = EncodedImage::new(codestream, DecodeRequest::Region { roi: TINY_ROI });
    let options = BatchDecodeOptions::default();
    let prepared = prepare_batch(vec![input], options).expect("prepare the batch");
    let errors = if prepared.errors().is_empty() {
        let mut decoder = CpuBatchDecoder::new(options);
        decoder
            .decode_prepared(&prepared)
            .expect("item failures stay per item")
            .errors()
            .to_vec()
    } else {
        prepared.errors().to_vec()
    };

    assert_eq!(errors.len(), 1, "{errors:?}");
    assert_eq!(errors[0].index, 0);
    let BatchItemError::Codec { source, .. } = &errors[0].source else {
        panic!("expected a codec failure, got {:?}", errors[0].source);
    };
    resource_limit(source);
}
