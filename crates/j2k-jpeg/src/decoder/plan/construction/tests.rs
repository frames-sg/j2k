// SPDX-License-Identifier: MIT OR Apache-2.0

use super::PreparedConstructionBudget;
use crate::{
    context::DecoderContext,
    parse::tables::{HuffmanTableRole, HuffmanValues, RawHuffmanTable},
    JpegError,
};
use alloc::vec::Vec;
use core::mem::size_of;

#[test]
fn parsed_and_context_baseline_accepts_exact_and_rejects_one_over() {
    assert!(PreparedConstructionBudget::with_cap(7, 5, 12).is_ok());
    assert!(matches!(
        PreparedConstructionBudget::with_cap(7, 5, 11),
        Err(JpegError::MemoryCapExceeded {
            requested: 12,
            cap: 11
        })
    ));
}

#[test]
fn external_parsed_and_context_baseline_has_one_exact_ledger() {
    let external = 3;
    let parsed = 7;
    let context = 5;
    let exact_cap = external + parsed + context;
    assert!(PreparedConstructionBudget::with_external_live_and_cap(
        external, parsed, context, exact_cap,
    )
    .is_ok());
    assert!(matches!(
        PreparedConstructionBudget::with_external_live_and_cap(
            external,
            parsed,
            context,
            exact_cap - 1,
        ),
        Err(JpegError::MemoryCapExceeded { requested, cap })
            if requested == exact_cap && cap == exact_cap - 1
    ));
}

#[test]
fn forced_spare_vector_capacity_is_counted_exactly() {
    let mut spare = Vec::<u32>::with_capacity(8);
    spare.push(1);
    assert!(spare.capacity() > spare.len());

    let base = 12usize;
    let vector_bytes = spare.capacity() * size_of::<u32>();
    let exact_cap = base + vector_bytes;
    let mut exact = PreparedConstructionBudget::with_cap(7, 5, exact_cap).unwrap();
    exact
        .include_retained_capacity::<u32>(spare.capacity())
        .unwrap();
    assert_eq!(exact.live_bytes, exact_cap);

    let mut one_over = PreparedConstructionBudget::with_cap(7, 5, exact_cap - 1).unwrap();
    assert!(matches!(
        one_over.include_retained_capacity::<u32>(spare.capacity()),
        Err(JpegError::MemoryCapExceeded { requested, cap })
            if requested == exact_cap && cap == exact_cap - 1
    ));
}

#[test]
fn prior_actual_capacity_is_used_by_the_next_reserve() {
    let mut spare = Vec::<u32>::with_capacity(5);
    spare.push(1);
    let base = 12usize;
    let prior_bytes = spare.capacity() * size_of::<u32>();
    let next_bytes = 3 * size_of::<u32>();
    let requested_boundary = base + prior_bytes + next_bytes;

    let mut reconciled = PreparedConstructionBudget::with_cap(7, 5, usize::MAX).unwrap();
    reconciled
        .include_retained_capacity::<u32>(spare.capacity())
        .unwrap();
    let next = reconciled.try_vec::<u32>(3).unwrap();
    assert_eq!(
        reconciled.live_bytes,
        base + prior_bytes + next.capacity() * size_of::<u32>()
    );

    let mut one_over = PreparedConstructionBudget::with_cap(7, 5, requested_boundary - 1).unwrap();
    one_over
        .include_retained_capacity::<u32>(spare.capacity())
        .unwrap();
    assert!(matches!(
        one_over.try_vec::<u32>(3),
        Err(JpegError::MemoryCapExceeded { requested, cap })
            if requested == requested_boundary && cap == requested_boundary - 1
    ));
}

#[test]
fn context_and_prepared_owners_each_charge_the_shared_huffman_payload() {
    let raw = RawHuffmanTable {
        bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        values: HuffmanValues::from_slice(&[0]),
    };
    let mut ctx = DecoderContext::new();
    let mut budget =
        PreparedConstructionBudget::with_cap(0, 0, j2k_core::DEFAULT_MAX_HOST_ALLOCATION_BYTES)
            .unwrap();
    let mut prepared = budget.try_huffman_tables(1).unwrap();

    budget
        .push_huffman_table(&mut ctx, &raw, HuffmanTableRole::Dc, &mut prepared)
        .unwrap();

    let context_bytes = ctx.retained_allocation_bytes();
    let prepared_bytes = prepared.retained_allocation_bytes().unwrap();
    budget
        .verify_retained(context_bytes, prepared_bytes)
        .unwrap();
    assert!(context_bytes > 0);
    assert!(prepared_bytes > 0);
}
