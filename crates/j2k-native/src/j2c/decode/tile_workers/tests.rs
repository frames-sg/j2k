// SPDX-License-Identifier: MIT OR Apache-2.0

use alloc::vec::Vec;

use crate::j2c::decode::WorkspaceBytes;
use crate::{
    encode, CpuDecodeParallelism, DecodeError, DecodeSettings, DecoderContext, EncodeOptions, Image,
};

const AUTO: CpuDecodeParallelism = CpuDecodeParallelism::Auto;
const SERIAL: CpuDecodeParallelism = CpuDecodeParallelism::Serial;

/// A noisy 8-bit image of `components` planes in tiles of `tile`, with the
/// MCT when there are at least three.
fn encode_tiled(
    (width, height): (u32, u32),
    tile: (u32, u32),
    components: u16,
    ht: bool,
    reversible: bool,
) -> Vec<u8> {
    let mut state = 0x9e37_79b9_u32;
    let mut pixels = Vec::new();
    for y in 0..height {
        for x in 0..width {
            for component in 0..u32::from(components) {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                pixels.push((x * 3 + y * 5 + component * 41 + (state & 31)).to_le_bytes()[0]);
            }
        }
    }
    let options = EncodeOptions {
        reversible,
        use_ht_block_coding: ht,
        num_decomposition_levels: 3,
        tile_size: Some(tile),
        use_mct: components >= 3,
        ..EncodeOptions::default()
    };
    encode(&pixels, width, height, components, 8, false, &options).expect("encode")
}

#[derive(Debug, PartialEq)]
struct Decoded {
    planes: Result<Vec<Vec<u32>>, DecodeError>,
    bitmap: Result<Vec<u8>, DecodeError>,
}

/// Decode twice with one context, the second time into reused planes, and
/// return the first result with the workspace peak of its plane decode.
fn decode(
    bytes: &[u8],
    settings: &DecodeSettings,
    reduction: u8,
    policy: CpuDecodeParallelism,
    threads: usize,
) -> (Decoded, WorkspaceBytes) {
    let image = match Image::new_with_reduction(bytes, settings, reduction) {
        Ok(image) => image,
        Err(error) => {
            let decoded = Decoded {
                planes: Err(error),
                bitmap: Err(error),
            };
            return (decoded, WorkspaceBytes::default());
        }
    };
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("test pool");
    pool.install(|| {
        let mut context = DecoderContext::default();
        context.set_cpu_decode_parallelism(policy);
        let first = decode_once(&image, &mut context);
        assert_eq!(
            decode_once(&image, &mut context).0,
            first.0,
            "reused context"
        );
        first
    })
}

fn decode_once<'a>(
    image: &Image<'a>,
    context: &mut DecoderContext<'a>,
) -> (Decoded, WorkspaceBytes) {
    let planes = image
        .decode_components_with_context(context)
        .map(|decoded| {
            let planes = decoded.planes().iter();
            planes
                .map(|plane| {
                    plane
                        .samples()
                        .iter()
                        .map(|sample| sample.to_bits())
                        .collect()
                })
                .collect()
        });
    let peak = context.tile_decode_context.debug_counters.workspace_peak;
    let bitmap = image.decode_with_context(context).map(|bitmap| bitmap.data);
    (Decoded { planes, bitmap }, peak.bytes())
}

#[test]
fn concurrent_tiles_match_the_serial_decode() {
    // (size, tile, components, HT, reversible, reductions)
    let fixtures = [
        // An even grid.
        ((256, 192), (64, 64), 3, true, true, 0),
        // Partial edge tiles.
        ((200, 150), (64, 48), 3, false, false, 0),
        // Odd sizes leave SIMD padding after the last plane row.
        ((331, 203), (64, 48), 3, true, false, 1),
        // Gray, so the store applies the level shift; tall tiles.
        ((180, 120), (48, 96), 1, true, false, 0),
        // A fourth plane, shifted after the MCT; wide tiles.
        ((160, 128), (80, 40), 4, false, true, 0),
        // Every reduction that leaves work for two workers.
        ((768, 448), (192, 160), 3, true, false, 2),
    ];
    let settings = DecodeSettings::default();
    for (size, tile, components, ht, reversible, reductions) in fixtures {
        let bytes = encode_tiled(size, tile, components, ht, reversible);
        for reduction in 0..=reductions {
            let (serial, _) = decode(&bytes, &settings, reduction, SERIAL, 2);
            assert!(serial.bitmap.is_ok());
            for threads in [2, 3] {
                let (auto, _) = decode(&bytes, &settings, reduction, AUTO, threads);
                let label = format!("{size:?} in {tile:?} reduced {reduction} on {threads}");
                assert_eq!(auto, serial, "{label}");
            }
        }
    }
}

#[test]
fn concurrent_tile_workspaces_add_up_in_the_peak() {
    let bytes = encode_tiled((256, 192), (64, 64), 3, true, true);
    let settings = DecodeSettings::default();
    let (_, serial) = decode(&bytes, &settings, 0, SERIAL, 3);
    let (_, auto) = decode(&bytes, &settings, 0, AUTO, 3);
    // Each worker that takes a tile adds its own tile's coefficients, so the
    // peak lies between one worker's and all three workers'. How many take a
    // tile depends on scheduling, so the bounds are what every run shows.
    assert_eq!(auto.channels, serial.channels);
    assert!(
        (serial.coefficients..=3 * serial.coefficients).contains(&auto.coefficients),
        "{auto:?} vs {serial:?}"
    );
}

#[test]
fn failing_tiles_report_the_serial_error() {
    let valid = encode_tiled((256, 192), (64, 64), 3, true, true);
    let tile_parts: Vec<usize> = valid
        .windows(4)
        .enumerate()
        .filter(|(_, window)| *window == [0xff, 0x90, 0x00, 0x0a])
        .map(|(offset, _)| offset)
        .collect();
    assert_eq!(tile_parts.len(), 12);
    // Break the last packet byte of the given tiles, or cut the codestream.
    let broken = |tiles: &[usize]| {
        let mut bytes = valid.clone();
        for &tile in tiles {
            bytes[tile_parts[tile + 1] - 1] = 0xff;
        }
        bytes
    };
    let cases = [
        broken(&[1]),
        broken(&[3, 9]),
        broken(&[10]),
        valid[..tile_parts[7] + 100].to_vec(),
    ];
    for (index, bytes) in cases.iter().enumerate() {
        for strict in [true, false] {
            let settings = DecodeSettings {
                strict,
                ..DecodeSettings::default()
            };
            let (serial, _) = decode(bytes, &settings, 0, SERIAL, 2);
            for threads in [2, 4] {
                let (auto, _) = decode(bytes, &settings, 0, AUTO, threads);
                assert_eq!(auto, serial, "case {index} strict {strict} on {threads}");
            }
        }
    }
}
