// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

fn vector_with_capacity<T>(capacity: usize) -> Vec<T> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .expect("small retained-input test allocation");
    values
}

#[test]
fn retained_baseline_accepts_exact_cap_and_rejects_one_byte_over() {
    let owner = vector_with_capacity::<u32>(3);
    let bytes = owner.capacity() * core::mem::size_of::<u32>();
    let retained = NativeEncodeRetainedInput::from_owner_bytes(&owner, bytes);
    NativeEncodeSession::try_with_cap(retained, bytes).expect("exact baseline cap");

    let retained = NativeEncodeRetainedInput::from_owner_bytes(&owner, bytes);
    let error = NativeEncodeSession::try_with_cap(retained, bytes - 1)
        .err()
        .expect("one-byte-over baseline");
    assert_eq!(
        error,
        EncodeError::AllocationTooLarge {
            what: "retained native encode inputs",
            requested: bytes,
            cap: bytes - 1,
        }
    );
}

#[test]
fn checked_phase_baseline_includes_every_retained_owner() {
    let first = vector_with_capacity::<u16>(5);
    let second = vector_with_capacity::<u32>(7);
    let owner_bytes = first.capacity() * core::mem::size_of::<u16>()
        + second.capacity() * core::mem::size_of::<u32>();
    let owners = (&first, &second);
    let retained = NativeEncodeRetainedInput::from_owner_bytes(&owners, owner_bytes);
    let session = NativeEncodeSession::try_new(retained).expect("encode session");
    assert_eq!(
        session
            .checked_phase_retained_bytes(11, "test phase")
            .expect("phase bytes"),
        first.capacity() * core::mem::size_of::<u16>()
            + second.capacity() * core::mem::size_of::<u32>()
            + 11
    );
}

#[test]
fn accelerator_output_phase_accepts_exact_cap_and_rejects_one_byte_over() {
    let output = vector_with_capacity::<u8>(9);
    let phase_bytes = 7;
    let exact_cap = phase_bytes + output.capacity();
    let exact = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_cap)
        .expect("exact output session");
    exact
        .checked_phase(phase_bytes, "test accelerator phase")
        .expect("phase baseline")
        .reconcile_accelerator_vec(&output, "test accelerator output")
        .expect("exact accelerator output cap");

    let over = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_cap - 1)
        .expect("phase baseline remains below cap");
    let error = over
        .checked_phase(phase_bytes, "test accelerator phase")
        .expect("phase baseline")
        .reconcile_accelerator_vec(&output, "test accelerator output")
        .expect_err("accelerator output is one byte over cap");
    assert_eq!(
        error,
        EncodeError::AllocationTooLarge {
            what: "test accelerator output",
            requested: exact_cap,
            cap: exact_cap - 1,
        }
    );
}

#[test]
fn child_session_carries_parent_and_phase_owners_exactly_once() {
    let owner = vector_with_capacity::<u16>(5);
    let phase_owner = vector_with_capacity::<u8>(7);
    let owner_bytes = owner.capacity() * core::mem::size_of::<u16>();
    let phase_bytes = phase_owner.capacity();
    let child_bytes = 11;
    let exact_cap = owner_bytes + phase_bytes + child_bytes;
    let retained = NativeEncodeRetainedInput::from_owner_bytes(&owner, owner_bytes);
    let session = NativeEncodeSession::try_with_cap(retained, exact_cap).expect("parent session");
    let child = session
        .checked_child_session(&phase_owner, phase_bytes, "retained parent tile owners")
        .expect("child session at exact parent phase");
    assert_eq!(
        child
            .checked_phase_retained_bytes(child_bytes, "nested tile phase")
            .expect("nested phase at exact cap"),
        exact_cap
    );

    let retained = NativeEncodeRetainedInput::from_owner_bytes(&owner, owner_bytes);
    let session = NativeEncodeSession::try_with_cap(retained, exact_cap - 1)
        .expect("parent baseline remains below cap");
    let child = session
        .checked_child_session(&phase_owner, phase_bytes, "retained parent tile owners")
        .expect("parent phase remains below cap");
    let error = child
        .checked_phase(child_bytes, "nested tile phase")
        .err()
        .expect("nested tile is one byte over cap");
    assert_eq!(
        error,
        EncodeError::AllocationTooLarge {
            what: "nested tile phase",
            requested: exact_cap,
            cap: exact_cap - 1,
        }
    );
}
