// SPDX-License-Identifier: MIT OR Apache-2.0

//! Shared MQ arithmetic-coder probability table (ITU-T T.800 Table C.2).

pub(crate) use j2k_codec_math::classic::MQ_STATES as QE_TABLE;

#[expect(
    clippy::cast_possible_truncation,
    clippy::similar_names,
    reason = "MQ state indices fit in 7 bits and the paired MPS/LPS transitions are domain terms"
)]
const fn packed_decoder_states() -> [u64; 256] {
    let mut packed = [0; 256];
    let mut mps = 0u8;
    while mps <= 1 {
        let mut index = 0usize;
        while index < QE_TABLE.len() {
            let state = QE_TABLE[index];
            let context = index as u8 | (mps << 7);
            let next_mps = state.nmps | (mps << 7);
            let next_lps = state.nlps | ((mps ^ state.switch as u8) << 7);
            packed[context as usize] = state.qe as u64
                | ((mps as u64) << 16)
                | ((next_mps as u64) << 24)
                | ((next_lps as u64) << 32);
            index += 1;
        }
        mps += 1;
    }
    packed
}

pub(crate) const PACKED_DECODER_STATES: [u64; 256] = packed_decoder_states();

#[cfg(test)]
mod tests {
    use super::{PACKED_DECODER_STATES, QE_TABLE};

    #[test]
    fn packed_decoder_states_preserve_every_mq_transition_and_mps_bit() {
        for mps in 0u8..=1 {
            for (index, transition) in QE_TABLE.iter().copied().enumerate() {
                let context = u8::try_from(index).expect("MQ index fits u8") | (mps << 7);
                let packed = PACKED_DECODER_STATES[context as usize];
                assert_eq!(
                    u32::try_from(packed & 0xffff).expect("masked MQ value fits u32"),
                    transition.qe
                );
                assert_eq!(u8::from((packed >> 16) & 1 != 0), mps);
                assert_eq!(
                    u8::try_from((packed >> 24) & 0xff).expect("masked MPS state fits u8"),
                    transition.nmps | (mps << 7)
                );
                assert_eq!(
                    u8::try_from((packed >> 32) & 0xff).expect("masked LPS state fits u8"),
                    transition.nlps | ((mps ^ u8::from(transition.switch)) << 7)
                );
            }
        }
    }
}
