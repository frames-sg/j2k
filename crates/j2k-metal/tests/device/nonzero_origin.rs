// SPDX-License-Identifier: MIT OR Apache-2.0

//! frames-sg/j2k#333: when a resolution is one sample tall at an odd row, the
//! reversible 5/3 inverse transform divides that lone sample by two with
//! floor division on Metal, as the CPU does since #332. Multiplying it by 0.5
//! left half values that rounded up.

use super::*;
use j2k_test_support::{NONZERO_ORIGIN_GRAY8_53X2, NONZERO_ORIGIN_GRAY8_53X2_TILE0};

const WIDTH: u32 = 53;
const HEIGHT: u32 = 2;

fn cpu_decode() -> Vec<u8> {
    let mut decoder = J2kDecoder::new(NONZERO_ORIGIN_GRAY8_53X2).expect("parse the seed");
    let mut samples = vec![0; (WIDTH * HEIGHT) as usize];
    decoder
        .decode_into(&mut samples, WIDTH as usize, PixelFormat::Gray8)
        .expect("CPU decode");
    assert_tile0("CPU", &samples);
    samples
}

fn assert_tile0(label: &str, samples: &[u8]) {
    assert_eq!(samples.len(), (WIDTH * HEIGHT) as usize, "{label} size");
    for (y, expected) in NONZERO_ORIGIN_GRAY8_53X2_TILE0.iter().enumerate() {
        let row = &samples[y * WIDTH as usize..][..expected.len()];
        assert_eq!(row, expected, "{label} tile 0 row {y}");
    }
}

fn assert_metal_matches_cpu(label: &str, surface: &Surface, cpu: &[u8]) {
    assert_eq!(surface.backend_kind(), BackendKind::Metal, "{label}");
    assert_eq!(surface.dimensions(), (WIDTH, HEIGHT), "{label}");
    let samples = surface.as_bytes().expect("surface bytes");
    assert_tile0(label, &samples);
    assert_eq!(samples.as_ref(), cpu, "{label} differs from the CPU decode");
}

#[test]
fn explicit_metal_decode_of_lone_odd_row_matches_cpu() {
    if !should_run_metal_runtime() {
        return;
    }
    let cpu = cpu_decode();

    let surface = J2kDecoder::new(NONZERO_ORIGIN_GRAY8_53X2)
        .expect("parse the seed")
        .decode_to_device(PixelFormat::Gray8, BackendRequest::Metal)
        .expect("explicit Metal decode");
    assert_metal_matches_cpu("single", &surface, &cpu);
}

#[test]
fn metal_batch_decode_of_lone_odd_row_matches_cpu() {
    if !should_run_metal_runtime() {
        return;
    }
    let cpu = cpu_decode();

    let options = {
        let mut options = BatchDecodeOptions::default();
        options.layout = BatchLayout::Nhwc;
        options
    };
    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("Metal batch decoder");
    let seed: Arc<[u8]> = Arc::from(NONZERO_ORIGIN_GRAY8_53X2);
    let batch = decoder
        .decode_batch(vec![EncodedImage::full(Arc::clone(&seed)); 4])
        .expect("Metal batch decode");
    assert!(batch.errors().is_empty(), "{:?}", batch.errors());
    assert!(
        batch.group_errors().is_empty(),
        "{:?}",
        batch.group_errors()
    );
    let surfaces = batch.groups()[0].surfaces();
    assert_eq!(surfaces.len(), 4);
    for (index, surface) in surfaces.iter().enumerate() {
        assert_metal_matches_cpu(&format!("batch image {index}"), surface, &cpu);
    }
}
