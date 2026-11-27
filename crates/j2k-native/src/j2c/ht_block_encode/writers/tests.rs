// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{terminate_mel_vlc, MagSgnEncoder, MelEncoder, VlcEncoder};

#[test]
fn writer_state_and_termination_match_pre_split_goldens() {
    let mut mel = MelEncoder::new();
    for bit in [
        false, false, true, false, true, true, false, false, false, true,
    ] {
        mel.encode(bit).expect("MEL bit");
    }
    let mut vlc = VlcEncoder::new();
    vlc.encode(0b10_1101, 6).expect("VLC bits");
    vlc.encode(0x1ff, 9).expect("VLC bits");
    terminate_mel_vlc(&mut mel, &mut vlc).expect("terminate MEL/VLC");

    assert_eq!(
        (
            mel.pos,
            mel.remaining_bits,
            mel.tmp,
            mel.run,
            mel.k,
            mel.threshold,
        ),
        (2, 5, 0x80, 0, 2, 1)
    );
    assert_eq!(&mel.buffer[..mel.pos], &[0xD3, 0x87]);
    assert_eq!(
        (vlc.pos, vlc.used_bits, vlc.tmp, vlc.last_greater_than_8f),
        (3, 3, 0x07, true)
    );
    assert_eq!(
        &vlc.buffer[vlc.buffer.len() - vlc.pos..],
        &[0xFE, 0xDF, 0xFF]
    );

    let mut magsgn = MagSgnEncoder::new();
    magsgn.encode(0xff, 8).expect("MagSgn bits");
    magsgn.encode(0x55, 7).expect("MagSgn bits");
    magsgn.encode(0x3, 2).expect("MagSgn bits");
    magsgn.terminate().expect("terminate MagSgn");

    assert_eq!(magsgn.pos, 2);
    assert_eq!(&magsgn.buffer[..magsgn.pos], &[0xFF, 0x55]);
}

#[test]
fn magsgn_stuffs_after_ff_across_a_full_reservoir() {
    let mut magsgn = MagSgnEncoder::new();
    magsgn.encode(u64::from(u32::MAX), 32).expect("MagSgn bits");
    magsgn.encode(u64::from(u32::MAX), 32).expect("MagSgn bits");
    magsgn.terminate().expect("terminate MagSgn");

    assert_eq!(
        &magsgn.buffer[..magsgn.pos],
        &[0xFF, 0x7F, 0xFF, 0x7F, 0xFF, 0x7F, 0xFF, 0x7F]
    );
}

#[test]
fn reverse_vlc_handles_stuffing_and_a_partial_reservoir_tail() {
    let mut mel = MelEncoder::new();
    let mut vlc = VlcEncoder::new();
    vlc.encode(0b111, 3).expect("VLC stuffing boundary");
    terminate_mel_vlc(&mut mel, &mut vlc).expect("terminate MEL/VLC");
    assert_eq!(&vlc.buffer[vlc.buffer.len() - vlc.pos..], &[0x7F, 0xFF]);

    mel.reset();
    vlc.reset();
    vlc.encode(0b1010, 4)
        .expect("VLC byte after greater-than-0x8f byte");
    terminate_mel_vlc(&mut mel, &mut vlc).expect("terminate MEL/VLC");
    assert_eq!(&vlc.buffer[vlc.buffer.len() - vlc.pos..], &[0xAF, 0xFF]);

    mel.reset();
    vlc.reset();
    for _ in 0..16 {
        vlc.encode(0, 4).expect("VLC reservoir bits");
    }
    terminate_mel_vlc(&mut mel, &mut vlc).expect("terminate MEL/VLC");
    assert_eq!((mel.pos, mel.buffer[0]), (1, 0));
    assert_eq!((vlc.pos, vlc.used_bits, vlc.tmp), (9, 4, 0));
    assert_eq!(
        &vlc.buffer[vlc.buffer.len() - vlc.pos..],
        &[0, 0, 0, 0, 0, 0, 0, 0x0F, 0xFF]
    );
}

#[test]
fn writers_ignore_bits_above_the_declared_codeword_length() {
    let mut mel_a = MelEncoder::new();
    let mut mel_b = MelEncoder::new();
    let mut vlc_a = VlcEncoder::new();
    let mut vlc_b = VlcEncoder::new();
    vlc_a.encode(1, 1).expect("VLC bit");
    vlc_b.encode(u64::MAX, 1).expect("masked VLC bit");
    terminate_mel_vlc(&mut mel_a, &mut vlc_a).expect("terminate MEL/VLC");
    terminate_mel_vlc(&mut mel_b, &mut vlc_b).expect("terminate MEL/VLC");
    assert_eq!(&mel_a.buffer[..mel_a.pos], &mel_b.buffer[..mel_b.pos]);
    assert_eq!(
        &vlc_a.buffer[vlc_a.buffer.len() - vlc_a.pos..],
        &vlc_b.buffer[vlc_b.buffer.len() - vlc_b.pos..]
    );

    let mut magsgn_a = MagSgnEncoder::new();
    let mut magsgn_b = MagSgnEncoder::new();
    magsgn_a.encode(1, 1).expect("MagSgn bit");
    magsgn_b.encode(u64::MAX, 1).expect("masked MagSgn bit");
    magsgn_a.terminate().expect("terminate MagSgn");
    magsgn_b.terminate().expect("terminate MagSgn");
    assert_eq!(
        &magsgn_a.buffer[..magsgn_a.pos],
        &magsgn_b.buffer[..magsgn_b.pos]
    );
}

#[test]
fn vlc_swar_bulk_drain_matches_bytewise_stuffing_boundaries() {
    let patterns: &[&[(u64, u8)]] = &[
        // Full 64-bit word with no 0x7F/0xFF low-7-bit bytes (exercises SWAR fast path).
        &[(0x1234_5678_9ABC_DE01, 60), (0x0A, 4), (0x33, 8)],
        // Byte boundary transitions around 0x8F -> 0x7F / 0x90 -> 0x7F / 0xFF.
        &[
            (0x7F, 8),
            (0x90, 8),
            (0x7F, 8),
            (0x8F, 8),
            (0xFF, 8),
            (0x7F, 8),
            (0x00, 8),
            (0x91, 8),
        ],
        // Mix of safe 64-bit chunks and stuffed chunks.
        &[
            (0x0102_0304_0506_0708, 60),
            (0x09, 4),
            (0xFF, 8),
            (0x7F, 8),
            (0x1020_3040_5060_7080, 60),
            (0x05, 4),
        ],
    ];
    for pattern in patterns {
        let mut vlc_fast = VlcEncoder::new();
        let mut mel_fast = MelEncoder::new();
        for &(cw, len) in *pattern {
            vlc_fast.encode(cw, len).expect("fast VLC encode");
        }
        terminate_mel_vlc(&mut mel_fast, &mut vlc_fast).expect("terminate fast");

        // Compare against byte-at-a-time insertion (never fills 64 bits at once).
        let mut vlc_ref = VlcEncoder::new();
        let mut mel_ref = MelEncoder::new();
        for &(cw, len) in *pattern {
            let mut rem = len;
            let mut val = cw;
            while rem > 0 {
                let step = rem.min(4);
                vlc_ref.encode(val, step).expect("ref VLC encode");
                val >>= step;
                rem -= step;
            }
        }
        terminate_mel_vlc(&mut mel_ref, &mut vlc_ref).expect("terminate ref");

        assert_eq!(
            &vlc_fast.buffer[vlc_fast.buffer.len() - vlc_fast.pos..],
            &vlc_ref.buffer[vlc_ref.buffer.len() - vlc_ref.pos..]
        );
        assert_eq!(
            &mel_fast.buffer[..mel_fast.pos],
            &mel_ref.buffer[..mel_ref.pos]
        );
    }
}
