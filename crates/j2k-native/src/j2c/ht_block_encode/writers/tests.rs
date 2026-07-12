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

    assert_eq!(
        (magsgn.pos, magsgn.max_bits, magsgn.used_bits, magsgn.tmp),
        (2, 8, 8, 0xFF)
    );
    assert_eq!(&magsgn.buffer[..magsgn.pos], &[0xFF, 0x55]);
}
