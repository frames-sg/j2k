// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::size_of;

use super::*;

fn descriptor(packet_index: u32, state_index: u32) -> j2k_native::J2kPacketizationPacketDescriptor {
    j2k_native::J2kPacketizationPacketDescriptor {
        packet_index,
        state_index,
        layer: 0,
        resolution: packet_index,
        component: 0,
        precinct: 0,
    }
}

#[test]
fn tier2_metadata_plan_honors_exact_aggregate_cap() {
    let payload = [0x2a];
    let resolutions = [j2k_native::J2kPacketizationResolution {
        subbands: vec![j2k_native::J2kPacketizationSubband {
            code_blocks: vec![j2k_native::J2kPacketizationCodeBlock {
                data: &payload,
                ht_cleanup_length: 0,
                ht_refinement_length: 0,
                num_coding_passes: 1,
                num_zero_bitplanes: 0,
                previously_included: false,
                l_block: 3,
                block_coding_mode: J2kPacketizationBlockCodingMode::Classic,
            }],
            num_cbs_x: 1,
            num_cbs_y: 1,
        }],
    }];
    let packet_descriptors = [descriptor(0, 7), descriptor(0, 7)];
    let job = J2kPacketizationEncodeJob {
        resolution_count: 1,
        num_layers: 2,
        num_components: 1,
        code_block_count: 1,
        progression_order: j2k_native::J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &packet_descriptors,
        resolutions: &resolutions,
    };
    let counts = tier2_packet_allocation_counts(job).expect("metadata counts");
    assert_eq!(counts.unique_states, 1);
    assert_eq!(counts.state_blocks, 1);
    let exact_cap = size_of::<J2kPacketResolution>()
        + size_of::<J2kPacketSubband>()
        + size_of::<J2kPacketBlock>()
        + size_of::<u8>()
        + size_of::<(u32, u32, usize)>()
        + size_of::<J2kPacketStateBlock>()
        + 2 * size_of::<J2kPacketDescriptor>();
    let requests = tier2_packet_allocation_requests(&counts);
    crate::batch_allocation::BatchMetadataBudget::with_cap(
        "J2K Metal Tier-2 packet metadata",
        exact_cap,
    )
    .preflight(&requests)
    .expect("exact aggregate cap");
    assert!(matches!(
        crate::batch_allocation::BatchMetadataBudget::with_cap(
            "J2K Metal Tier-2 packet metadata",
            exact_cap - 1,
        )
        .preflight(&requests),
        Err(j2k_core::BatchInfrastructureError::AllocationTooLarge {
            requested,
            cap,
            ..
        }) if requested == exact_cap && cap == exact_cap - 1
    ));
}

#[test]
fn plan_rejects_descriptor_packet_index_before_metal_dispatch() {
    let resolutions = [j2k_native::J2kPacketizationResolution {
        subbands: Vec::new(),
    }];
    let packet_descriptors = [descriptor(1, 0)];
    let job = J2kPacketizationEncodeJob {
        resolution_count: 1,
        num_layers: 1,
        num_components: 1,
        code_block_count: 0,
        progression_order: j2k_native::J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &packet_descriptors,
        resolutions: &resolutions,
    };

    let Err(error) = plan_tier2_packetization(job) else {
        panic!("out-of-range descriptor unexpectedly planned");
    };
    assert!(matches!(
        error,
        Error::MetalKernel { ref message }
            if message == "Tier-2 Metal packet descriptor packet index out of range"
    ));
}

#[test]
fn plan_rejects_reused_state_with_a_different_block_layout() {
    let payload = [0x2a];
    let resolutions = [
        j2k_native::J2kPacketizationResolution {
            subbands: Vec::new(),
        },
        j2k_native::J2kPacketizationResolution {
            subbands: vec![j2k_native::J2kPacketizationSubband {
                code_blocks: vec![j2k_native::J2kPacketizationCodeBlock {
                    data: &payload,
                    ht_cleanup_length: 0,
                    ht_refinement_length: 0,
                    num_coding_passes: 1,
                    num_zero_bitplanes: 0,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode: J2kPacketizationBlockCodingMode::Classic,
                }],
                num_cbs_x: 1,
                num_cbs_y: 1,
            }],
        },
    ];
    let packet_descriptors = [descriptor(0, 7), descriptor(1, 7)];
    let job = J2kPacketizationEncodeJob {
        resolution_count: 2,
        num_layers: 1,
        num_components: 1,
        code_block_count: 1,
        progression_order: j2k_native::J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &packet_descriptors,
        resolutions: &resolutions,
    };

    let Err(error) = plan_tier2_packetization(job) else {
        panic!("mismatched reused state unexpectedly planned");
    };
    assert!(matches!(
        error,
        Error::MetalKernel { ref message }
            if message == "Tier-2 Metal packet descriptor state layout mismatch"
    ));
}
