// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{BatchDecodeOptions, BatchLayout, CpuBatchDecoder, CpuBatchSamples, EncodedImage};
use j2k_metal::MetalBatchDecoder;
use j2k_native::{encode_htj2k, EncodeOptions};

fn signed_rgb_fixture(seed: i16, use_mct: bool) -> Arc<[u8]> {
    const WIDTH: u32 = 8;
    const HEIGHT: u32 = 8;
    let mut samples = Vec::with_capacity(WIDTH as usize * HEIGHT as usize * 3);
    for index in 0..WIDTH * HEIGHT {
        let base = i16::try_from(index).expect("fixture index") * 11 + seed - 350;
        samples.extend_from_slice(&[base + 5, base - 3, base + 1]);
    }
    let bytes = samples
        .iter()
        .flat_map(|sample| sample.to_le_bytes())
        .collect::<Vec<_>>();
    let options = EncodeOptions {
        reversible: true,
        num_decomposition_levels: 2,
        use_mct,
        ..EncodeOptions::default()
    };
    Arc::from(
        encode_htj2k(&bytes, WIDTH, HEIGHT, 3, 12, true, &options)
            .expect("encode signed RGB HTJ2K fixture"),
    )
}

#[test]
fn mixed_signed_rgb_mct_modes_preserve_each_images_samples() {
    if !j2k_test_support::metal_runtime_gate(module_path!()) {
        return;
    }

    let options = BatchDecodeOptions {
        layout: BatchLayout::Nhwc,
        ..BatchDecodeOptions::default()
    };
    let inputs = vec![
        EncodedImage::full(signed_rgb_fixture(0, true)),
        EncodedImage::full(signed_rgb_fixture(97, false)),
    ];
    let mut cpu = CpuBatchDecoder::new(options);
    let expected = cpu.decode(inputs.clone()).expect("CPU signed RGB oracle");
    assert_eq!(
        expected.groups().len(),
        2,
        "MCT and non-MCT inputs must not share one homogeneous color group"
    );

    let mut decoder =
        MetalBatchDecoder::system_default_with_options(options).expect("persistent Metal decoder");
    let prepared = decoder.prepare(inputs).expect("prepare mixed-MCT group");
    assert!(prepared.errors().is_empty());
    assert_eq!(
        prepared.groups().len(),
        2,
        "prepared grouping must keep incompatible color transforms separate"
    );
    let result = decoder
        .decode_prepared(&prepared)
        .expect("decode separated mixed-MCT groups");
    assert_eq!(result.groups().len(), expected.groups().len());
    for (actual_group, expected_group) in result.groups().iter().zip(expected.groups()) {
        let CpuBatchSamples::I16(expected_samples) = expected_group.samples() else {
            panic!("signed RGB group must use I16 storage")
        };
        assert_eq!(
            actual_group.source_indices(),
            expected_group.source_indices()
        );
        let resident = actual_group
            .resident_batch()
            .expect("completed group has resident Metal storage");
        // SAFETY: completed codec output is read without retaining or submitting a writer.
        let actual = unsafe {
            j2k_metal_support::checked_buffer_read_vec::<i16>(
                resident.metal_buffer(),
                resident.byte_offset(),
                expected_samples.len(),
            )
            .expect("read separated mixed-MCT resident output")
        };
        assert_eq!(actual.as_slice(), expected_samples.as_slice());
    }
}
