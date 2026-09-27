// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{ForwardBitReader, MelDecoder, ReverseBitReader};

#[test]
fn reader_state_and_bit_consumption_match_pre_split_goldens() {
    let data = [0xAA, 0xFF, 0x01, 0x7F, 0x80];
    let mut forward = ForwardBitReader::<0xFF>::new(&data);
    assert_eq!(u64::from(forward.fetch()), 0x3F81_FFAA);
    forward.advance(5);
    assert_eq!(u64::from(forward.fetch()), 0x01FC_0FFD);
    forward.advance(19);
    assert_eq!(u64::from(forward.fetch()), 0xFFFF_C03F);
    assert_eq!(
        (forward.pos, forward.bits, forward.tmp, forward.unstuff),
        (5, 37, 0x0000_001F_FFFF_C03F, true)
    );

    let mut reverse = ReverseBitReader::new_mrp(&data);
    assert_eq!(u64::from(reverse.fetch()), 0xFF01_7F80);
    assert_eq!(reverse.advance(7), 0x55FE_02FF);
    assert_eq!(u64::from(reverse.fetch()), 0x55FE_02FF);
    assert_eq!(
        (
            reverse.pos,
            reverse.remaining,
            reverse.bits,
            reverse.tmp,
            reverse.unstuff,
        ),
        (-1, 0, 33, 0x0000_0001_55FE_02FF, true)
    );

    let mel_data = [0x12, 0x34, 0x56, 0x78];
    let mut mel = MelDecoder::new(&mel_data, mel_data.len(), 2);
    let mut runs = [0i32; 8];
    for run in &mut runs {
        *run = mel.get_run().expect("MEL run");
    }
    assert_eq!(runs, [1, 0, 1, 0, 0, 0, 2, 2]);
    assert_eq!(
        (
            mel.pos,
            mel.remaining,
            mel.bits_left,
            mel.k,
            mel.num_runs,
            mel.runs,
            mel.unstuff,
        ),
        (3, 0, 0, 5, 0, 0, false)
    );

    let vlc_data = [0x12, 0x34, 0x56, 0x78, 0x9A];
    let mut vlc = ReverseBitReader::new_vlc(&vlc_data, vlc_data.len(), 3);
    assert_eq!(vlc.fetch(), 0x0000_02B7);
    assert_eq!(vlc.advance(9), 0x0000_0001);
    assert_eq!(vlc.fetch(), 0x0000_0001);
    assert_eq!(
        (vlc.pos, vlc.remaining, vlc.bits, vlc.tmp, vlc.unstuff),
        (1, 0, 34, 1, false)
    );
}

#[test]
fn word_refills_yield_the_same_bits_as_byte_refills() {
    let mut seed = 0x1234_5678_u32;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed
    };
    // Bytes around the stuffing thresholds are over-represented.
    let special = [0xFF_u8, 0x7F, 0x8F, 0x90, 0x00, 0xFE];
    for len in 0..=40_usize {
        for _ in 0..20 {
            let data = (0..len)
                .map(|_| {
                    let draw = next();
                    if draw % 3 == 0 {
                        special[(draw >> 8) as usize % special.len()]
                    } else {
                        (draw >> 16).to_le_bytes()[0]
                    }
                })
                .collect::<Vec<_>>();
            let advances = (0..24).map(|_| next() % 33).collect::<Vec<_>>();

            let mut forward = ForwardBitReader::<0xFF>::new(&data);
            let mut forward_bytes = forward.clone();
            let mut reverse = ReverseBitReader::new_mrp(&data);
            let mut reverse_bytes = reverse.clone();
            for &count in &advances {
                if forward_bytes.bits < 32 {
                    forward_bytes = forward_bytes.fill_bytes();
                }
                assert_eq!(
                    u64::from(forward.fetch()),
                    forward_bytes.tmp & u64::from(u32::MAX),
                    "forward len {len}"
                );
                forward.advance(count);
                forward_bytes.advance(count);

                if reverse_bytes.bits < 32 {
                    reverse_bytes = reverse_bytes.fill_bytes();
                }
                assert_eq!(
                    u64::from(reverse.fetch()),
                    reverse_bytes.tmp & u64::from(u32::MAX),
                    "reverse len {len}"
                );
                reverse.advance(count);
                reverse_bytes.advance(count);
            }
        }
    }
}
