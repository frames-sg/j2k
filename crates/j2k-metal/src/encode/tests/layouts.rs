// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

#[cfg(target_os = "macos")]
#[test]
#[expect(
    clippy::too_many_lines,
    reason = "public layout matrix is one behavior-focused regression"
)]
fn metal_encode_deinterleave_public_layouts_match_native_reference() {
    #[derive(Clone, Copy)]
    struct Case {
        name: &'static str,
        num_components: u8,
        bit_depth: u8,
        signed: bool,
    }

    #[expect(
        clippy::cast_possible_truncation,
        reason = "bounded public-layout fixture indices fit target sample widths"
    )]
    fn case_pixels(case: Case, num_pixels: usize) -> Vec<u8> {
        let sample_count = num_pixels * usize::from(case.num_components);
        if case.bit_depth <= 8 {
            return (0..sample_count)
                .map(|idx| ((idx * 37 + 11) & 0xff) as u8)
                .collect();
        }

        let mut pixels = Vec::with_capacity(sample_count * 2);
        for idx in 0..sample_count {
            let sample = ((idx * 1031 + 0x7123) & 0xffff) as u16;
            pixels.extend_from_slice(&sample.to_le_bytes());
        }
        pixels
    }

    if !should_run_metal_runtime() {
        return;
    }

    let cases = [
        Case {
            name: "Gray8 unsigned",
            num_components: 1,
            bit_depth: 8,
            signed: false,
        },
        Case {
            name: "two-component 8-bit unsigned",
            num_components: 2,
            bit_depth: 8,
            signed: false,
        },
        Case {
            name: "RGB8 unsigned",
            num_components: 3,
            bit_depth: 8,
            signed: false,
        },
        Case {
            name: "RGBA8 unsigned",
            num_components: 4,
            bit_depth: 8,
            signed: false,
        },
        Case {
            name: "RGB8 signed",
            num_components: 3,
            bit_depth: 8,
            signed: true,
        },
        Case {
            name: "Gray16 signed",
            num_components: 1,
            bit_depth: 16,
            signed: true,
        },
        Case {
            name: "RGB16 unsigned",
            num_components: 3,
            bit_depth: 16,
            signed: false,
        },
        Case {
            name: "RGBA12 unsigned",
            num_components: 4,
            bit_depth: 12,
            signed: false,
        },
    ];

    for case in cases {
        let pixels = case_pixels(case, 5);
        J2kLosslessSamples::new(
            &pixels,
            5,
            1,
            u16::from(case.num_components),
            case.bit_depth,
            case.signed,
        )
        .unwrap_or_else(|err| panic!("lossless public layout rejected for {}: {err}", case.name));
        J2kLossySamples::new(
            &pixels,
            5,
            1,
            u16::from(case.num_components),
            case.bit_depth,
            case.signed,
        )
        .unwrap_or_else(|err| panic!("lossy public layout rejected for {}: {err}", case.name));

        let job = J2kDeinterleaveToF32Job {
            pixels: &pixels,
            num_pixels: 5,
            num_components: u16::from(case.num_components),
            bit_depth: case.bit_depth,
            signed: case.signed,
        };
        let expected = try_deinterleave_reference(
            job.pixels,
            job.num_pixels,
            job.num_components,
            job.bit_depth,
            job.signed,
        )
        .expect("valid native deinterleave reference input");
        let mut accelerator = MetalEncodeStageAccelerator::default();

        let actual = accelerator
            .encode_deinterleave(job)
            .unwrap_or_else(|err| {
                panic!("Metal deinterleave stage failed for {}: {err}", case.name)
            })
            .unwrap_or_else(|| panic!("Metal deinterleave did not dispatch for {}", case.name));

        assert_eq!(actual, expected, "{}", case.name);
        assert_eq!(accelerator.deinterleave_attempts(), 1, "{}", case.name);
        assert_eq!(accelerator.deinterleave_dispatches(), 1, "{}", case.name);
        assert_eq!(
            accelerator.dispatch_report().deinterleave,
            1,
            "{}",
            case.name
        );
    }
}
