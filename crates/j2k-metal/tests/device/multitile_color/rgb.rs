// SPDX-License-Identifier: MIT OR Apache-2.0

use super::batch_inputs::independent_multitile_inputs;
use super::*;

fn assert_multitile_rgb8_batch_request(
    decoder: &mut MetalBatchDecoder,
    encoded: &[u8],
    options: BatchDecodeOptions,
    request: DecodeRequest,
) {
    let inputs = independent_multitile_inputs(encoded, request);
    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu
        .decode(inputs.clone())
        .unwrap_or_else(|error| panic!("CPU multi-tile RGB8 {request:?} oracle: {error}"));
    assert!(expected.errors().is_empty(), "CPU request {request:?}");
    let CpuBatchSamples::U8(expected) = expected.groups()[0].samples() else {
        panic!("OpenJPH RGB8 must use U8 batch storage")
    };

    let prepared = decoder
        .prepare(inputs)
        .unwrap_or_else(|error| panic!("prepare multi-tile RGB8 {request:?}: {error}"));
    assert!(prepared.errors().is_empty(), "Metal request {request:?}");
    assert_eq!(prepared.groups().len(), 1, "Metal request {request:?}");
    let group = &prepared.groups()[0];
    assert_eq!(group.images().len(), 2, "Metal request {request:?}");
    assert!(group
        .images()
        .iter()
        .all(|image| image.preparation_depth() == PreparationDepth::Htj2kOffsetPlan));
    let (width, height) = group.info().dimensions;
    let samples_per_image = width as usize * height as usize * 3;
    let output_len = samples_per_image * group.images().len();
    assert_eq!(expected.len(), output_len, "CPU request {request:?}");
    let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
        decoder.backend_session().device(),
        output_len + 4,
    )
    .expect("multi-tile RGB8 batch destination");
    let layout = MetalImageLayout::new_batch(
        4,
        (width, height),
        width as usize * 3,
        PixelFormat::Rgb8,
        group.images().len(),
        samples_per_image,
    )
    .expect("multi-tile RGB8 batch destination layout");
    // SAFETY: the fresh allocation is exclusively offered to the codec call.
    let destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(buffer.clone(), layout)
            .expect("multi-tile RGB8 batch destination guard")
    };
    decoder
        .submit_prepared_group_into(group, destination)
        .unwrap_or_else(|error| panic!("submit multi-tile RGB8 {request:?}: {error}"))
        .wait()
        .unwrap_or_else(|error| panic!("complete multi-tile RGB8 {request:?}: {error}"));

    // SAFETY: completion released the exclusive destination owner.
    let actual = unsafe {
        j2k_metal_support::checked_buffer_read_vec::<u8>(&buffer, 4, output_len)
            .expect("read multi-tile RGB8 batch destination")
    };
    assert_eq!(
        actual.as_slice(),
        expected.as_slice(),
        "request {request:?}"
    );
}
#[test]
fn independent_openjph_multitile_rgb_decodes_exactly_on_metal() {
    if !should_run_metal_runtime() {
        return;
    }
    let fixture = j2k_test_support::openjph_batch_fixtures()
        .iter()
        .find(|fixture| fixture.name == "openjph-rgb-u8-53-raw")
        .expect("checked-in OpenJPH RGB8 fixture");
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder
        .prepare(vec![EncodedImage::full(Arc::from(fixture.encoded))])
        .expect("prepare independent OpenJPH RGB fixture");
    assert!(prepared.errors().is_empty());
    assert_eq!(prepared.groups().len(), 1);
    assert_eq!(
        prepared.groups()[0].images()[0].preparation_depth(),
        PreparationDepth::Htj2kOffsetPlan
    );

    let image_len = fixture.width as usize * fixture.height as usize * 3;
    let buffer = j2k_metal_support::checked_shared_buffer_for_len::<u8>(
        decoder.backend_session().device(),
        image_len + 4,
    )
    .expect("independent RGB destination");
    let layout = MetalImageLayout::new_batch(
        4,
        (fixture.width, fixture.height),
        fixture.width as usize * 3,
        PixelFormat::Rgb8,
        1,
        image_len,
    )
    .expect("independent RGB destination layout");
    // SAFETY: the fresh allocation is exclusively offered to the codec call.
    let destination = unsafe {
        MetalImageDestination::from_exclusive_buffer(buffer.clone(), layout)
            .expect("independent RGB destination guard")
    };
    decoder
        .submit_prepared_group_into(&prepared.groups()[0], destination)
        .expect("submit independent multi-tile RGB fixture")
        .wait()
        .expect("complete independent multi-tile RGB fixture");
    // SAFETY: completion released the exclusive destination owner.
    let actual = unsafe {
        j2k_metal_support::checked_buffer_read_vec::<u8>(&buffer, 4, image_len)
            .expect("read independent multi-tile RGB destination")
    };
    assert_eq!(actual.as_slice(), fixture.oracle);
}

#[test]
fn independent_openjph_multitile_rgb_batch_matches_cpu_for_all_requests() {
    if !should_run_metal_runtime() {
        return;
    }
    let fixture = j2k_test_support::openjph_batch_fixtures()
        .iter()
        .find(|fixture| fixture.name == "openjph-rgb-u8-53-raw")
        .expect("checked-in OpenJPH multi-tile RGB8 fixture");
    let roi = Rect {
        x: 8,
        y: 5,
        w: 8,
        h: 6,
    };
    let requests = [
        DecodeRequest::Full,
        DecodeRequest::Region { roi },
        DecodeRequest::Reduced {
            scale: Downscale::Half,
        },
        DecodeRequest::RegionReduced {
            roi,
            scale: Downscale::Half,
        },
    ];
    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");

    for request in requests {
        assert_multitile_rgb8_batch_request(&mut decoder, fixture.encoded, options, request);
    }
}
