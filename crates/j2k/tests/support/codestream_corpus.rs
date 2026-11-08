// SPDX-License-Identifier: MIT OR Apache-2.0

//! A small corpus of JPEG 2000 codestreams for decode-path parity tests.

use j2k_native::{encode, EncodeOptions};

fn rgb_fixture(
    width: u32,
    height: u32,
    decomposition_levels: u8,
    tile_size: Option<(u32, u32)>,
) -> Vec<u8> {
    let pixels = (0..height)
        .flat_map(|y| {
            (0..width).flat_map(move |x| {
                [x % 251, y % 241, (x + y) % 233]
                    .map(|value| u8::try_from(value).expect("fixture sample fits u8"))
            })
        })
        .collect::<Vec<_>>();
    encode(
        &pixels,
        width,
        height,
        3,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: decomposition_levels,
            tile_size,
            ..EncodeOptions::default()
        },
    )
    .expect("encode RGB fixture")
}

/// Codestreams covering odd image origins, tiles, empty coarse levels,
/// progression orders, HTJ2K, sampled components and 12/16-bit precision.
pub(super) fn parity_corpus() -> Vec<(String, Vec<u8>)> {
    let fixtures =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../j2k-test-support/fixtures");
    let mut corpus = Vec::new();
    for directory in [
        "openjpeg_origin",
        "openjpeg_progression",
        "empty_coarse_levels",
        "htj2k",
    ] {
        let mut entries = std::fs::read_dir(fixtures.join(directory))
            .expect("fixture directory")
            .map(|entry| entry.expect("fixture entry").path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "j2k" || extension == "j2c")
            })
            .collect::<Vec<_>>();
        entries.sort();
        for path in entries {
            corpus.push((
                format!(
                    "{directory}/{}",
                    path.file_name().expect("name").to_string_lossy()
                ),
                std::fs::read(&path).expect("fixture bytes"),
            ));
        }
    }
    corpus.push(("rgb 61x47 5 levels".into(), rgb_fixture(61, 47, 5, None)));
    corpus.push((
        "rgb 61x47 tiled 3 levels".into(),
        rgb_fixture(61, 47, 3, Some((24, 20))),
    ));
    let gray16 = (0_u32..53 * 37)
        .flat_map(|index| {
            u16::try_from((index * 2_731) % 65_536)
                .expect("u16")
                .to_le_bytes()
        })
        .collect::<Vec<_>>();
    for (name, signed, use_ht_block_coding) in
        [("gray16 HT", false, true), ("gray16 signed", true, false)]
    {
        corpus.push((
            name.into(),
            encode(
                &gray16,
                53,
                37,
                1,
                16,
                signed,
                &EncodeOptions {
                    reversible: true,
                    num_decomposition_levels: 4,
                    use_ht_block_coding,
                    ..EncodeOptions::default()
                },
            )
            .expect("encode 16-bit gray"),
        ));
    }
    let rgba = (0_u32..45 * 33 * 4)
        .map(|index| u8::try_from((index * 13 + index / 97) % 256).expect("u8"))
        .collect::<Vec<_>>();
    corpus.push((
        "rgba8".into(),
        encode(
            &rgba,
            45,
            33,
            4,
            8,
            false,
            &EncodeOptions {
                reversible: true,
                num_decomposition_levels: 3,
                ..EncodeOptions::default()
            },
        )
        .expect("encode RGBA"),
    ));
    let luma = (0_u32..48 * 40)
        .map(|index| u8::try_from((index * 7) % 256).expect("u8"))
        .collect::<Vec<_>>();
    let chroma = (0_u32..24 * 20)
        .map(|index| u8::try_from((index * 11 + 3) % 256).expect("u8"))
        .collect::<Vec<_>>();
    let planes = [
        j2k_native::EncodeComponentPlane {
            data: &luma,
            x_rsiz: 1,
            y_rsiz: 1,
        },
        j2k_native::EncodeComponentPlane {
            data: &chroma,
            x_rsiz: 2,
            y_rsiz: 2,
        },
        j2k_native::EncodeComponentPlane {
            data: &chroma,
            x_rsiz: 2,
            y_rsiz: 2,
        },
    ];
    corpus.push((
        "sampled 4:2:0".into(),
        j2k_native::encode_component_planes_53(
            &planes,
            48,
            40,
            8,
            false,
            &EncodeOptions {
                num_decomposition_levels: 3,
                use_mct: false,
                ..EncodeOptions::default()
            },
        )
        .expect("encode sampled planes"),
    ));
    corpus
}
