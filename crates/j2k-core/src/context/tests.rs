// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{CacheStats, CodecContext};

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
fn default_constructs_the_codec_context_state() {
    let context = SeededContext::default();

    assert_eq!(context.retained_entries, 7);
}

#[test]
fn cache_stats_constructors_preserve_explicit_and_defaulted_counters() {
    assert_eq!(
        CacheStats::new(3, 5),
        CacheStats {
            hits: 3,
            misses: 5,
            occupied_slots: 0,
            evictions: 0,
        }
    );
    assert_eq!(
        CacheStats::with_slots(8, 13, 21, 34),
        CacheStats {
            hits: 8,
            misses: 13,
            occupied_slots: 21,
            evictions: 34,
        }
    );
}
