// SPDX-License-Identifier: MIT OR Apache-2.0

//! The free batch API's retained workspaces are process-wide, so this binary
//! holds a single test to observe them without interference.

use std::num::NonZeroUsize;

use j2k::{
    decode_tiles_into, release_tile_batch_workspaces, retained_tile_batch_workspace_bytes,
    J2kDecoder, PixelFormat, TileBatchOptions, TileDecodeJob,
};
use j2k_native::{encode_htj2k, EncodeOptions};

#[test]
fn decode_tiles_into_retains_workspaces_until_released() {
    let pixels = (0..32_u32 * 32 * 3)
        .map(|index| u8::try_from((index * 11) & 0xff).expect("masked"))
        .collect::<Vec<_>>();
    let input = encode_htj2k(
        &pixels,
        32,
        32,
        3,
        8,
        false,
        &EncodeOptions {
            reversible: true,
            num_decomposition_levels: 2,
            ..EncodeOptions::default()
        },
    )
    .expect("encode RGB8 tile");
    let mut expected = std::iter::repeat_n(0_u8, pixels.len()).collect::<Vec<_>>();
    J2kDecoder::new(&input)
        .expect("reference decoder")
        .decode_into(&mut expected, 32 * 3, PixelFormat::Rgb8)
        .expect("reference decode");
    assert_eq!(retained_tile_batch_workspace_bytes(), 0);

    let mut outputs = (0..2)
        .map(|_| std::iter::repeat_n(0_u8, pixels.len()).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let mut jobs = outputs
        .iter_mut()
        .map(|out| TileDecodeJob {
            input: &input,
            out,
            stride: 32 * 3,
        })
        .collect::<Vec<_>>();
    decode_tiles_into(
        &mut jobs,
        PixelFormat::Rgb8,
        TileBatchOptions {
            workers: NonZeroUsize::new(2),
        },
    )
    .expect("batch decode");
    drop(jobs);
    assert!(outputs.iter().all(|out| *out == expected));
    assert!(retained_tile_batch_workspace_bytes() > 0);

    release_tile_batch_workspaces();
    assert_eq!(retained_tile_batch_workspace_bytes(), 0);
}
