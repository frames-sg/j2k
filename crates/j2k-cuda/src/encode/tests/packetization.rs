// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    cuda_ht_segment_lengths, flatten_cuda_htj2k_packetization_job, CudaHtj2kPacketizationPlanError,
    CudaHtj2kPacketizationPlanTagNodeState, J2kPacketizationBlockCodingMode,
    J2kPacketizationCodeBlock, J2kPacketizationEncodeJob, J2kPacketizationPacketDescriptor,
    J2kPacketizationProgressionOrder, J2kPacketizationResolution, J2kPacketizationSubband,
};

#[test]
fn cuda_packetization_flatten_accepts_cleanup_only_single_block_packet() {
    let payload = [0x12, 0x34, 0x56, 0x78];
    let code_block = J2kPacketizationCodeBlock {
        data: &payload,
        ht_cleanup_length: 0,
        ht_refinement_length: 0,
        num_coding_passes: 1,
        num_zero_bitplanes: 2,
        previously_included: false,
        l_block: 3,
        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
    };
    let subband = J2kPacketizationSubband {
        code_blocks: vec![code_block],
        num_cbs_x: 1,
        num_cbs_y: 1,
    };
    let resolution = J2kPacketizationResolution {
        subbands: vec![subband],
    };
    let descriptor = J2kPacketizationPacketDescriptor {
        packet_index: 0,
        state_index: 0,
        layer: 0,
        resolution: 0,
        component: 0,
        precinct: 0,
    };
    let job = J2kPacketizationEncodeJob {
        resolution_count: 1,
        num_layers: 1,
        num_components: 1,
        code_block_count: 1,
        progression_order: J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &[descriptor],
        resolutions: &[resolution],
    };

    let plan = flatten_cuda_htj2k_packetization_job(job).expect("supported CUDA packetization");

    assert_eq!(plan.payload, payload);
    assert_eq!(plan.packets.len(), 1);
    assert_eq!(plan.subbands.len(), 1);
    assert_eq!(plan.blocks.len(), 1);
    assert_eq!(plan.packets[0].block_start, 0);
    assert_eq!(plan.packets[0].block_count, 1);
    assert_eq!(plan.packets[0].subband_start, 0);
    assert_eq!(plan.packets[0].subband_count, 1);
    assert_eq!(plan.subbands[0].block_start, 0);
    assert_eq!(plan.subbands[0].block_count, 1);
    let payload_len = u32::try_from(payload.len()).expect("test payload length fits in u32");
    assert!(plan.packets[0].output_capacity >= payload_len + 256);
    assert_eq!(plan.blocks[0].data_offset, 0);
    assert_eq!(plan.blocks[0].data_len, payload_len);
    assert_eq!(plan.blocks[0].num_coding_passes, 1);
    assert_eq!(plan.blocks[0].num_zero_bitplanes, 2);
}

#[test]
fn cuda_packetization_flatten_accepts_cleanup_only_multi_block_packet() {
    let payloads = vec![
        vec![0x10, 0x11, 0x12],
        vec![0x20, 0x21],
        vec![0x30, 0x31, 0x32, 0x33],
        vec![0x40],
    ];
    let code_blocks = payloads
        .iter()
        .enumerate()
        .map(|(idx, payload)| J2kPacketizationCodeBlock {
            data: payload.as_slice(),
            ht_cleanup_length: 0,
            ht_refinement_length: 0,
            num_coding_passes: 1,
            num_zero_bitplanes: u8::try_from(idx + 1).expect("test zbp fits in u8"),
            previously_included: false,
            l_block: 3,
            block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
        })
        .collect();
    let subband = J2kPacketizationSubband {
        code_blocks,
        num_cbs_x: 2,
        num_cbs_y: 2,
    };
    let resolution = J2kPacketizationResolution {
        subbands: vec![subband],
    };
    let descriptor = J2kPacketizationPacketDescriptor {
        packet_index: 0,
        state_index: 0,
        layer: 0,
        resolution: 0,
        component: 0,
        precinct: 0,
    };
    let job = J2kPacketizationEncodeJob {
        resolution_count: 1,
        num_layers: 1,
        num_components: 1,
        code_block_count: 4,
        progression_order: J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &[descriptor],
        resolutions: &[resolution],
    };

    let plan = flatten_cuda_htj2k_packetization_job(job).expect("multi-block CUDA packetization");

    assert_eq!(plan.packets.len(), 1);
    assert_eq!(plan.subbands.len(), 1);
    assert_eq!(plan.blocks.len(), 4);
    assert_eq!(plan.packets[0].block_start, 0);
    assert_eq!(plan.packets[0].block_count, 4);
    assert_eq!(plan.packets[0].subband_start, 0);
    assert_eq!(plan.packets[0].subband_count, 1);
    assert_eq!(plan.subbands[0].block_start, 0);
    assert_eq!(plan.subbands[0].block_count, 4);
    assert_eq!(plan.subbands[0].num_cbs_x, 2);
    assert_eq!(plan.subbands[0].num_cbs_y, 2);
    assert_eq!(
        plan.payload,
        payloads.into_iter().flatten().collect::<Vec<_>>()
    );
    assert_eq!(plan.blocks[2].num_zero_bitplanes, 3);
}

#[test]
fn cuda_packetization_flatten_accepts_ht_refinement_pass_packet() {
    let payload = [0x12, 0x34, 0x56, 0x78, 0x9a];
    let code_block = J2kPacketizationCodeBlock {
        data: &payload,
        ht_cleanup_length: 3,
        ht_refinement_length: 2,
        num_coding_passes: 3,
        num_zero_bitplanes: 2,
        previously_included: false,
        l_block: 3,
        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
    };
    let subband = J2kPacketizationSubband {
        code_blocks: vec![code_block],
        num_cbs_x: 1,
        num_cbs_y: 1,
    };
    let resolution = J2kPacketizationResolution {
        subbands: vec![subband],
    };
    let descriptor = J2kPacketizationPacketDescriptor {
        packet_index: 0,
        state_index: 0,
        layer: 0,
        resolution: 0,
        component: 0,
        precinct: 0,
    };
    let job = J2kPacketizationEncodeJob {
        resolution_count: 1,
        num_layers: 1,
        num_components: 1,
        code_block_count: 1,
        progression_order: J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &[descriptor],
        resolutions: &[resolution],
    };

    let plan = flatten_cuda_htj2k_packetization_job(job).expect("HT refinement packetization");

    assert_eq!(plan.payload, payload);
    assert_eq!(plan.blocks.len(), 1);
    assert_eq!(plan.blocks[0].num_coding_passes, 3);
    assert_eq!(
        plan.blocks[0].data_len,
        u32::try_from(payload.len()).expect("test payload length fits in u32")
    );
}

#[test]
fn cuda_packetization_rejects_overflowing_ht_refinement_lengths() {
    let payload = [0x12];
    let code_block = J2kPacketizationCodeBlock {
        data: &payload,
        ht_cleanup_length: u32::MAX,
        ht_refinement_length: 1,
        num_coding_passes: 3,
        num_zero_bitplanes: 2,
        previously_included: false,
        l_block: 3,
        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
    };

    let err = cuda_ht_segment_lengths(&code_block)
        .expect_err("overflowing CUDA HT segment lengths rejected");

    assert_eq!(
        err,
        CudaHtj2kPacketizationPlanError::ArithmeticOverflow(
            "multi-pass HTJ2K packet contribution length overflow"
        )
    );
}

#[test]
fn cuda_packetization_flatten_rejects_out_of_range_ht_pass_count() {
    let payload = [0u8; 1];
    let code_block = J2kPacketizationCodeBlock {
        data: &payload,
        ht_cleanup_length: 0,
        ht_refinement_length: 0,
        num_coding_passes: 165,
        num_zero_bitplanes: 2,
        previously_included: false,
        l_block: 3,
        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
    };
    let subband = J2kPacketizationSubband {
        code_blocks: vec![code_block],
        num_cbs_x: 1,
        num_cbs_y: 1,
    };
    let resolution = J2kPacketizationResolution {
        subbands: vec![subband],
    };
    let descriptor = J2kPacketizationPacketDescriptor {
        packet_index: 0,
        state_index: 0,
        layer: 0,
        resolution: 0,
        component: 0,
        precinct: 0,
    };
    let job = J2kPacketizationEncodeJob {
        resolution_count: 1,
        num_layers: 1,
        num_components: 1,
        code_block_count: 1,
        progression_order: J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &[descriptor],
        resolutions: &[resolution],
    };

    let err = flatten_cuda_htj2k_packetization_job(job)
        .expect_err("invalid HT pass count must be rejected before CUDA launch");

    assert_eq!(
        err,
        CudaHtj2kPacketizationPlanError::Invalid(
            "CUDA HTJ2K packetization coding pass count exceeds JPEG 2000 bounds"
        )
    );
}

#[test]
fn cuda_packetization_flatten_accepts_previously_included_second_layer_packet() {
    let first_payload = [0x11u8; 20];
    let second_payload = [0x22u8; 5];
    let first_block = J2kPacketizationCodeBlock {
        data: &first_payload,
        ht_cleanup_length: 0,
        ht_refinement_length: 0,
        num_coding_passes: 1,
        num_zero_bitplanes: 2,
        previously_included: false,
        l_block: 3,
        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
    };
    let second_block = J2kPacketizationCodeBlock {
        data: &second_payload,
        ht_cleanup_length: 0,
        ht_refinement_length: 0,
        num_coding_passes: 1,
        num_zero_bitplanes: 2,
        previously_included: false,
        l_block: 3,
        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
    };
    let first_resolution = J2kPacketizationResolution {
        subbands: vec![J2kPacketizationSubband {
            code_blocks: vec![first_block],
            num_cbs_x: 1,
            num_cbs_y: 1,
        }],
    };
    let second_resolution = J2kPacketizationResolution {
        subbands: vec![J2kPacketizationSubband {
            code_blocks: vec![second_block],
            num_cbs_x: 1,
            num_cbs_y: 1,
        }],
    };
    let descriptors = [
        J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        },
        J2kPacketizationPacketDescriptor {
            packet_index: 1,
            state_index: 0,
            layer: 1,
            resolution: 0,
            component: 0,
            precinct: 0,
        },
    ];
    let resolutions = [first_resolution, second_resolution];
    let job = J2kPacketizationEncodeJob {
        resolution_count: 2,
        num_layers: 2,
        num_components: 1,
        code_block_count: 2,
        progression_order: J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &descriptors,
        resolutions: &resolutions,
    };

    let plan = flatten_cuda_htj2k_packetization_job(job).expect("stateful CUDA packetization plan");

    assert_eq!(
        plan.payload,
        [first_payload.as_slice(), second_payload.as_slice()].concat()
    );
    assert_eq!(plan.packets.len(), 2);
    assert_eq!(plan.blocks.len(), 2);
    assert_eq!(plan.packets[0].layer, 0);
    assert_eq!(plan.packets[1].layer, 1);
    assert_eq!(plan.blocks[0].l_block, 3);
    assert_eq!(plan.blocks[0].previously_included, 0);
    assert_eq!(plan.blocks[1].previously_included, 1);
    assert_eq!(plan.blocks[0].inclusion_layer, 0);
    assert_eq!(plan.blocks[1].inclusion_layer, 0);
    assert_eq!(
        plan.blocks[1].l_block, 5,
        "first layer length must update L-block for later packet state"
    );
}

#[test]
fn cuda_packetization_flatten_accepts_deferred_first_inclusion_second_layer_packet() {
    let payload = [0x44u8; 5];
    let first_block = J2kPacketizationCodeBlock {
        data: &[],
        ht_cleanup_length: 0,
        ht_refinement_length: 0,
        num_coding_passes: 0,
        num_zero_bitplanes: 2,
        previously_included: false,
        l_block: 3,
        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
    };
    let second_block = J2kPacketizationCodeBlock {
        data: &payload,
        ht_cleanup_length: 0,
        ht_refinement_length: 0,
        num_coding_passes: 1,
        num_zero_bitplanes: 2,
        previously_included: false,
        l_block: 3,
        block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
    };
    let first_resolution = J2kPacketizationResolution {
        subbands: vec![J2kPacketizationSubband {
            code_blocks: vec![first_block],
            num_cbs_x: 1,
            num_cbs_y: 1,
        }],
    };
    let second_resolution = J2kPacketizationResolution {
        subbands: vec![J2kPacketizationSubband {
            code_blocks: vec![second_block],
            num_cbs_x: 1,
            num_cbs_y: 1,
        }],
    };
    let descriptors = [
        J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        },
        J2kPacketizationPacketDescriptor {
            packet_index: 1,
            state_index: 0,
            layer: 1,
            resolution: 0,
            component: 0,
            precinct: 0,
        },
    ];
    let resolutions = [first_resolution, second_resolution];
    let job = J2kPacketizationEncodeJob {
        resolution_count: 2,
        num_layers: 2,
        num_components: 1,
        code_block_count: 2,
        progression_order: J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &descriptors,
        resolutions: &resolutions,
    };

    let plan = flatten_cuda_htj2k_packetization_job(job).expect("deferred first inclusion plan");

    assert_eq!(plan.payload, payload);
    assert_eq!(plan.packets.len(), 2);
    assert_eq!(plan.blocks.len(), 2);
    assert_eq!(plan.packets[0].layer, 0);
    assert_eq!(plan.packets[1].layer, 1);
    assert_eq!(plan.blocks[0].previously_included, 0);
    assert_eq!(plan.blocks[1].previously_included, 0);
    assert_eq!(plan.blocks[0].inclusion_layer, 1);
    assert_eq!(plan.blocks[1].inclusion_layer, 1);
}

#[test]
#[expect(
    clippy::too_many_lines,
    reason = "multi-packet deferred-inclusion fixture is one byte-structure regression"
)]
fn cuda_packetization_flatten_accepts_deferred_first_inclusion_after_non_empty_packet() {
    let first_payload = [0x11u8; 3];
    let second_payload = [0x22u8; 5];
    let first_resolution = J2kPacketizationResolution {
        subbands: vec![J2kPacketizationSubband {
            code_blocks: vec![
                J2kPacketizationCodeBlock {
                    data: &first_payload,
                    ht_cleanup_length: 0,
                    ht_refinement_length: 0,
                    num_coding_passes: 1,
                    num_zero_bitplanes: 2,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                },
                J2kPacketizationCodeBlock {
                    data: &[],
                    ht_cleanup_length: 0,
                    ht_refinement_length: 0,
                    num_coding_passes: 0,
                    num_zero_bitplanes: 2,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                },
            ],
            num_cbs_x: 2,
            num_cbs_y: 1,
        }],
    };
    let second_resolution = J2kPacketizationResolution {
        subbands: vec![J2kPacketizationSubband {
            code_blocks: vec![
                J2kPacketizationCodeBlock {
                    data: &[],
                    ht_cleanup_length: 0,
                    ht_refinement_length: 0,
                    num_coding_passes: 0,
                    num_zero_bitplanes: 2,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                },
                J2kPacketizationCodeBlock {
                    data: &second_payload,
                    ht_cleanup_length: 0,
                    ht_refinement_length: 0,
                    num_coding_passes: 1,
                    num_zero_bitplanes: 2,
                    previously_included: false,
                    l_block: 3,
                    block_coding_mode: J2kPacketizationBlockCodingMode::HighThroughput,
                },
            ],
            num_cbs_x: 2,
            num_cbs_y: 1,
        }],
    };
    let descriptors = [
        J2kPacketizationPacketDescriptor {
            packet_index: 0,
            state_index: 0,
            layer: 0,
            resolution: 0,
            component: 0,
            precinct: 0,
        },
        J2kPacketizationPacketDescriptor {
            packet_index: 1,
            state_index: 0,
            layer: 1,
            resolution: 0,
            component: 0,
            precinct: 0,
        },
    ];
    let resolutions = [first_resolution, second_resolution];
    let job = J2kPacketizationEncodeJob {
        resolution_count: 2,
        num_layers: 2,
        num_components: 1,
        code_block_count: 4,
        progression_order: J2kPacketizationProgressionOrder::Lrcp,
        packet_descriptors: &descriptors,
        resolutions: &resolutions,
    };

    let plan = flatten_cuda_htj2k_packetization_job(job)
        .expect("persistent tag-tree state is flattened for CUDA packetization");

    assert_eq!(
        plan.payload,
        [first_payload.as_slice(), second_payload.as_slice()].concat()
    );
    assert_eq!(plan.packets.len(), 2);
    assert_eq!(plan.blocks.len(), 4);
    assert_eq!(plan.blocks[0].previously_included, 0);
    assert_eq!(plan.blocks[1].previously_included, 0);
    assert_eq!(plan.blocks[2].previously_included, 1);
    assert_eq!(plan.blocks[3].previously_included, 0);
    assert_eq!(plan.blocks[0].inclusion_layer, 0);
    assert_eq!(plan.blocks[1].inclusion_layer, 1);
    assert_eq!(plan.blocks[2].inclusion_layer, 0);
    assert_eq!(plan.blocks[3].inclusion_layer, 1);
    assert_eq!(plan.tag_states.len(), 2);
    assert_eq!(plan.tag_nodes.len(), 12);
    assert_eq!(plan.tag_states[1].inclusion_node_start, 6);
    assert_eq!(plan.tag_states[1].zero_bitplane_node_start, 9);
    assert_eq!(
        &plan.tag_nodes[6..9],
        &[
            CudaHtj2kPacketizationPlanTagNodeState {
                current: 0,
                known: 1,
            },
            CudaHtj2kPacketizationPlanTagNodeState {
                current: 1,
                known: 0,
            },
            CudaHtj2kPacketizationPlanTagNodeState {
                current: 0,
                known: 1,
            },
        ]
    );
    assert_eq!(
        &plan.tag_nodes[9..12],
        &[
            CudaHtj2kPacketizationPlanTagNodeState {
                current: 2,
                known: 1,
            },
            CudaHtj2kPacketizationPlanTagNodeState {
                current: 0,
                known: 0,
            },
            CudaHtj2kPacketizationPlanTagNodeState {
                current: 2,
                known: 1,
            },
        ]
    );
}
