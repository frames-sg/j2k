// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{CodecContext, DecoderContext};

#[derive(Debug, PartialEq, Eq)]
struct SeededContext {
    retained_entries: usize,
}

impl Default for SeededContext {
    fn default() -> Self {
        Self {
            retained_entries: 7,
        }
    }
}

impl CodecContext for SeededContext {
    fn clear(&mut self) {
        self.retained_entries = 0;
    }
}

#[test]
fn new_constructs_the_codec_context_from_its_default() {
    let context = DecoderContext::<SeededContext>::new();

    assert_eq!(context.codec().retained_entries, 7);
}
