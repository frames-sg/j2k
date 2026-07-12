// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{upsample_h2v1_sample_at, upsample_h2v2_rows_at};

#[test]
fn extended12_decode_modules_stay_focused_and_fragment_free() {
    const ROOT: &str = include_str!("../extended12.rs");
    const PLANES: &str = include_str!("planes.rs");
    const PLANE_ALLOCATION: &str = include_str!("planes/allocation.rs");
    const PROGRESSIVE: &str = include_str!("progressive.rs");
    const PROGRESSIVE_444: &str = include_str!("progressive/color444.rs");
    const PROGRESSIVE_SUBSAMPLED: &str = include_str!("progressive/subsampled.rs");
    const PROGRESSIVE_FOUR: &str = include_str!("progressive/four_component.rs");
    const RGBA: &str = include_str!("rgba.rs");
    const SAMPLING: &str = include_str!("sampling.rs");
    const SEQUENTIAL: &str = include_str!("sequential.rs");
    const SEQUENTIAL_444: &str = include_str!("sequential/color444.rs");
    const SEQUENTIAL_SUBSAMPLED: &str = include_str!("sequential/subsampled.rs");
    const SEQUENTIAL_FOUR: &str = include_str!("sequential/four_component.rs");
    const STATE: &str = include_str!("state.rs");
    const UPSAMPLE: &str = include_str!("upsample.rs");
    const WRITERS: &str = include_str!("writers.rs");

    let modules = [
        ("extended12.rs", ROOT, 40usize),
        ("extended12/planes.rs", PLANES, 400),
        ("extended12/planes/allocation.rs", PLANE_ALLOCATION, 320),
        ("extended12/progressive.rs", PROGRESSIVE, 210),
        ("extended12/progressive/color444.rs", PROGRESSIVE_444, 120),
        (
            "extended12/progressive/subsampled.rs",
            PROGRESSIVE_SUBSAMPLED,
            100,
        ),
        (
            "extended12/progressive/four_component.rs",
            PROGRESSIVE_FOUR,
            80,
        ),
        ("extended12/rgba.rs", RGBA, 70),
        ("extended12/sampling.rs", SAMPLING, 280),
        ("extended12/sequential.rs", SEQUENTIAL, 260),
        ("extended12/sequential/color444.rs", SEQUENTIAL_444, 120),
        (
            "extended12/sequential/subsampled.rs",
            SEQUENTIAL_SUBSAMPLED,
            100,
        ),
        (
            "extended12/sequential/four_component.rs",
            SEQUENTIAL_FOUR,
            180,
        ),
        ("extended12/state.rs", STATE, 70),
        ("extended12/upsample.rs", UPSAMPLE, 170),
        ("extended12/writers.rs", WRITERS, 380),
    ];

    for (path, source, max_lines) in modules {
        let line_count = source.lines().count();
        assert!(
            line_count <= max_lines,
            "{path} grew to {line_count} lines; split it before exceeding {max_lines}"
        );
        assert!(
            !source.contains("include!(") && !source.contains("#[path"),
            "{path} must remain a real Rust module, not a textual source fragment"
        );
    }

    for declaration in [
        "mod planes;",
        "mod progressive;",
        "mod rgba;",
        "mod sampling;",
        "mod sequential;",
        "mod state;",
        "mod upsample;",
        "mod writers;",
    ] {
        assert!(
            ROOT.contains(declaration),
            "extended12 facade lost required module boundary {declaration}"
        );
    }
    for route in [PROGRESSIVE, SEQUENTIAL] {
        for declaration in ["mod color444;", "mod four_component;", "mod subsampled;"] {
            assert!(
                route.contains(declaration),
                "extended12 route lost required path boundary {declaration}"
            );
        }
    }
    assert!(
        PLANES.contains("mod allocation;"),
        "extended12 plane orchestration lost its allocation boundary"
    );
}

#[test]
fn extended12_diagnostic_literals_survive_module_moves_exactly_once_per_path() {
    const PROGRESSIVE_SUBSAMPLED: &str = include_str!("progressive/subsampled.rs");
    const SEQUENTIAL_SUBSAMPLED: &str = include_str!("sequential/subsampled.rs");
    const WRITERS: &str = include_str!("writers.rs");

    assert_eq!(
        PROGRESSIVE_SUBSAMPLED
            .matches("4:4:4 path is handled directly")
            .count(),
        1
    );
    assert_eq!(
        SEQUENTIAL_SUBSAMPLED
            .matches("4:4:4 path is handled directly")
            .count(),
        1
    );
    assert_eq!(
        WRITERS
            .matches("12-bit four-component path only accepts CMYK/YCCK")
            .count(),
        1
    );
    assert_eq!(
        WRITERS
            .matches("12-bit four-component plane path only accepts CMYK/YCCK")
            .count(),
        1
    );
}

#[test]
fn extended12_generic_upsampling_keeps_edge_and_rounding_goldens() {
    let row_u8 = [0u8, 1, 255];
    let actual_u8 = (0..row_u8.len() * 2)
        .map(|x| upsample_h2v1_sample_at(&row_u8, x))
        .collect::<Vec<_>>();
    assert_eq!(actual_u8, [0, 0, 1, 65, 192, 255]);

    let row_u16 = [0u16, 1, 4095];
    let actual_u16 = (0..row_u16.len() * 2)
        .map(|x| upsample_h2v1_sample_at(&row_u16, x))
        .collect::<Vec<_>>();
    assert_eq!(actual_u16, [0, 0, 1, 1025, 3072, 4095]);

    let current = [100u16, 400, 900];
    let near = [300u16, 200, 700];
    let actual_h2v2 = (0..5)
        .map(|x| upsample_h2v2_rows_at(&current, &near, 5, x))
        .collect::<Vec<_>>();
    assert_eq!(actual_h2v2, [150, 200, 300, 475, 850]);
}
