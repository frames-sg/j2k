// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::j2c::encode::NativeEncodeSession;
use crate::{EncodeError, NativeEncodeRetainedInput};

fn vector_with_capacity<T>(capacity: usize) -> Vec<T> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .expect("small transform ownership test allocation");
    values
}

#[test]
fn transform_graph_uses_actual_nested_capacities_at_exact_boundary() {
    let mut components = vector_with_capacity::<Vec<f32>>(2);
    components.push(vector_with_capacity::<f32>(5));
    components.push(vector_with_capacity::<f32>(7));
    let component_bytes = component_planes_retained_bytes(&components, components.capacity())
        .expect("component bytes");
    assert_eq!(
        component_bytes,
        components.capacity() * core::mem::size_of::<Vec<f32>>()
            + (components[0].capacity() + components[1].capacity()) * core::mem::size_of::<f32>()
    );

    let mut levels = vector_with_capacity::<DwtLevel>(1);
    levels.push(DwtLevel {
        hl: vector_with_capacity::<f32>(2),
        lh: vector_with_capacity::<f32>(3),
        hh: vector_with_capacity::<f32>(4),
        low_width: 1,
        low_height: 1,
        high_width: 1,
        high_height: 1,
    });
    let decomposition = DwtDecomposition {
        ll: vector_with_capacity::<f32>(3),
        ll_width: 1,
        ll_height: 1,
        levels,
    };
    let mut decompositions = vector_with_capacity::<DwtDecomposition>(1);
    decompositions.push(decomposition);
    let dwt_bytes = dwt_decompositions_retained_bytes(&decompositions, decompositions.capacity())
        .expect("DWT bytes");
    let expected_dwt_bytes = decompositions.capacity() * core::mem::size_of::<DwtDecomposition>()
        + decompositions[0].ll.capacity() * core::mem::size_of::<f32>()
        + decompositions[0].levels.capacity() * core::mem::size_of::<DwtLevel>()
        + (decompositions[0].levels[0].hl.capacity()
            + decompositions[0].levels[0].lh.capacity()
            + decompositions[0].levels[0].hh.capacity())
            * core::mem::size_of::<f32>();
    assert_eq!(dwt_bytes, expected_dwt_bytes);

    let exact_cap = component_bytes + dwt_bytes;
    let exact = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_cap)
        .expect("exact transform session");
    exact
        .checked_phase(exact_cap, "transform graph output")
        .expect("exact transform graph");

    let over = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_cap - 1)
        .expect("baseline remains below cap");
    let error = over
        .checked_phase(exact_cap, "transform graph output")
        .err()
        .expect("transform graph is one byte over cap");
    assert!(matches!(
        error,
        EncodeError::AllocationTooLarge {
            what: "transform graph output",
            requested,
            cap,
        } if requested == exact_cap && cap == exact_cap - 1
    ));
}

#[test]
fn cpu_dwt_preflight_counts_packed_and_extracted_planes() {
    let samples = 17;
    let levels = 3;
    assert_eq!(
        cpu_dwt_transient_bytes(samples, levels).expect("DWT transient bytes"),
        samples * core::mem::size_of::<f32>() * 2
            + usize::from(levels) * core::mem::size_of::<DwtLevel>()
    );
}
