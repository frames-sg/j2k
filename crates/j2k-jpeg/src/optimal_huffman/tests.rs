// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn frequencies(counts: &[(u8, u64)]) -> HuffmanFrequencies {
    let mut frequencies = HuffmanFrequencies::new();
    for &(symbol, count) in counts {
        frequencies.add_count(symbol, count);
    }
    frequencies
}

/// Kraft sum of the table's codes, in units of 2^-16.
fn kraft_units(spec: &HuffmanSpec) -> u64 {
    spec.bits
        .iter()
        .enumerate()
        .map(|(index, &count)| u64::from(count) << (MAX_CODE_LEN - 1 - index))
        .sum()
}

/// Check the properties every table the encoder writes must have.
fn assert_valid(spec: &HuffmanSpec, frequencies: &HuffmanFrequencies) {
    let table = spec.table();
    let coded: usize = spec.bits.iter().map(|&count| usize::from(count)).sum();
    assert_eq!(coded, spec.values().len());
    // Strictly below one: the all-ones code point stays unused.
    assert!(kraft_units(spec) < 1 << MAX_CODE_LEN);
    for symbol in 0..=u8::MAX {
        let len = table.lens[usize::from(symbol)];
        let occurs = frequencies.0[usize::from(symbol)] > 0;
        assert_eq!(len > 0, occurs, "symbol {symbol} coded iff it occurs");
        if len > 0 {
            assert!(usize::from(len) <= MAX_CODE_LEN);
            let all_ones = (1_u32 << len) - 1;
            assert_ne!(u32::from(table.codes[usize::from(symbol)]), all_ones);
        }
    }
    let mut seen = [false; SYMBOLS];
    for &symbol in spec.values() {
        assert!(!core::mem::replace(&mut seen[usize::from(symbol)], true));
    }
}

#[test]
fn a_single_symbol_gets_a_one_bit_zero_code() {
    let frequencies = frequencies(&[(5, 1000)]);
    let spec = optimal_huffman_spec(&frequencies).expect("table");
    assert_eq!(spec.bits[0], 1);
    assert_eq!(spec.values(), &[5]);
    let table = spec.table();
    assert_eq!((table.codes[5], table.lens[5]), (0, 1));
}

#[test]
fn two_equal_symbols_avoid_the_all_ones_code() {
    let frequencies = frequencies(&[(0, 10), (1, 10)]);
    let spec = optimal_huffman_spec(&frequencies).expect("table");
    // The reserved symbol takes the longest code, which is then dropped.
    assert_eq!(&spec.bits[..2], &[1, 1]);
    assert_valid(&spec, &frequencies);
}

#[test]
fn more_frequent_symbols_never_get_longer_codes() {
    let counts: Vec<(u8, u64)> = (0..=u8::MAX)
        .map(|symbol| (symbol, u64::from(symbol) * 37 % 101 + 1))
        .collect();
    let frequencies = frequencies(&counts);
    let spec = optimal_huffman_spec(&frequencies).expect("table");
    assert_valid(&spec, &frequencies);
    let table = spec.table();
    for &(a, count_a) in &counts {
        for &(b, count_b) in &counts {
            if count_a > count_b {
                assert!(table.lens[usize::from(a)] <= table.lens[usize::from(b)]);
            }
        }
    }
}

#[test]
fn skewed_statistics_are_limited_to_sixteen_bits() {
    // Fibonacci counts give the deepest possible unconstrained tree.
    let mut counts = Vec::new();
    let (mut a, mut b) = (1_u64, 1_u64);
    for symbol in 0..40_u8 {
        counts.push((symbol, a));
        (a, b) = (b, a + b);
    }
    let frequencies = frequencies(&counts);
    assert!(unlimited_code_sizes(&frequencies)
        .iter()
        .any(|&size| size > MAX_CODE_LEN));
    let spec = optimal_huffman_spec(&frequencies).expect("table");
    assert_valid(&spec, &frequencies);
    assert!(spec.bits[MAX_CODE_LEN - 1] > 0);
}

#[test]
fn every_symbol_at_once_still_fits() {
    let frequencies = frequencies(&(0..=u8::MAX).map(|symbol| (symbol, 1)).collect::<Vec<_>>());
    let spec = optimal_huffman_spec(&frequencies).expect("table");
    assert_eq!(spec.values().len(), SYMBOLS);
    assert_valid(&spec, &frequencies);
}

#[test]
fn huge_counts_do_not_overflow() {
    let frequencies = frequencies(&[(0, u64::MAX), (1, u64::MAX), (2, 3)]);
    let spec = optimal_huffman_spec(&frequencies).expect("table");
    assert_valid(&spec, &frequencies);
}

#[test]
fn an_empty_table_is_an_error() {
    assert!(matches!(
        optimal_huffman_spec(&HuffmanFrequencies::new()),
        Err(JpegEncodeError::InternalInvariant { .. })
    ));
}

#[test]
fn counts_accumulate() {
    let mut counts = frequencies(&[(1, 2), (3, 4)]);
    counts.add(3);
    counts.add_count(9, 7);
    assert_eq!(counts, frequencies(&[(1, 2), (3, 5), (9, 7)]));
    assert_eq!(counts.count(3), 5);
}
