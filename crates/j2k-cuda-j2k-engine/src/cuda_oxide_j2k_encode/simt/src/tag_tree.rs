// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    abi::{
        J2kHtPacketBlock, J2kHtPacketSubband, J2kHtPacketSubbandTagState, J2kHtPacketTagNodeState,
    },
    constants::{J2K_PACKET_MAX_TAG_LEVELS, J2K_PACKET_MAX_TAG_NODES, J2K_PACKET_TAG_INF},
    helpers::load_job,
    packet_writer::{J2kPacketBitWriter, j2k_packet_write_bit},
};

pub(crate) struct J2kPacketTagTree<'a> {
    pub(crate) values: &'a mut [u32; J2K_PACKET_MAX_TAG_NODES],
    pub(crate) current: &'a mut [u32; J2K_PACKET_MAX_TAG_NODES],
    pub(crate) known: &'a mut [u32; J2K_PACKET_MAX_TAG_NODES],
    pub(crate) widths: [u32; J2K_PACKET_MAX_TAG_LEVELS],
    pub(crate) heights: [u32; J2K_PACKET_MAX_TAG_LEVELS],
    pub(crate) offsets: [u32; J2K_PACKET_MAX_TAG_LEVELS],
    pub(crate) levels: u32,
    pub(crate) total_nodes: u32,
    pub(crate) failed: u32,
}

#[inline(always)]
pub(crate) fn j2k_packet_tag_tree_new<'a>(scratch: *mut u32) -> J2kPacketTagTree<'a> {
    // Initialize every word before creating references to the node arrays.
    let mut index = 0;
    while index < 3 * J2K_PACKET_MAX_TAG_NODES {
        unsafe { scratch.add(index).write(0) };
        index += 1;
    }
    J2kPacketTagTree {
        values: unsafe { &mut *scratch.cast::<[u32; J2K_PACKET_MAX_TAG_NODES]>() },
        current: unsafe {
            &mut *scratch
                .add(J2K_PACKET_MAX_TAG_NODES)
                .cast::<[u32; J2K_PACKET_MAX_TAG_NODES]>()
        },
        known: unsafe {
            &mut *scratch
                .add(2 * J2K_PACKET_MAX_TAG_NODES)
                .cast::<[u32; J2K_PACKET_MAX_TAG_NODES]>()
        },
        widths: [0; J2K_PACKET_MAX_TAG_LEVELS],
        heights: [0; J2K_PACKET_MAX_TAG_LEVELS],
        offsets: [0; J2K_PACKET_MAX_TAG_LEVELS],
        levels: 0,
        total_nodes: 0,
        failed: 0,
    }
}

#[inline(always)]
pub(crate) fn j2k_packet_tag_tree_fail(tree: &mut J2kPacketTagTree<'_>) -> bool {
    tree.failed = 1;
    false
}

/// Level layout of a tag tree over a `width` x `height` grid of code blocks.
#[derive(Clone, Copy)]
pub(crate) struct J2kPacketTagTreeShape {
    pub(crate) widths: [u32; J2K_PACKET_MAX_TAG_LEVELS],
    pub(crate) heights: [u32; J2K_PACKET_MAX_TAG_LEVELS],
    pub(crate) offsets: [u32; J2K_PACKET_MAX_TAG_LEVELS],
    pub(crate) levels: u32,
    pub(crate) total_nodes: u32,
}

#[inline(always)]
pub(crate) fn j2k_packet_tag_tree_shape_new() -> J2kPacketTagTreeShape {
    J2kPacketTagTreeShape {
        widths: [0; J2K_PACKET_MAX_TAG_LEVELS],
        heights: [0; J2K_PACKET_MAX_TAG_LEVELS],
        offsets: [0; J2K_PACKET_MAX_TAG_LEVELS],
        levels: 0,
        total_nodes: 0,
    }
}

/// Lays `shape` out for a `width` x `height` grid; `false` when the grid is
/// empty or needs more levels or nodes than a tree holds. Callers keep the
/// shape in a local declared before any branch: the device toolchain cannot
/// allocate stack inside loops.
#[inline(always)]
pub(crate) fn j2k_packet_tag_tree_shape(
    shape: &mut J2kPacketTagTreeShape,
    width: u32,
    height: u32,
) -> bool {
    shape.levels = 0;
    shape.total_nodes = 0;
    if width == 0 || height == 0 {
        return false;
    }

    let mut w = width;
    let mut h = height;
    loop {
        if shape.levels as usize >= J2K_PACKET_MAX_TAG_LEVELS {
            return false;
        }
        let Some(nodes) = w.checked_mul(h) else {
            return false;
        };
        let Some(next_total) = shape.total_nodes.checked_add(nodes) else {
            return false;
        };
        if next_total as usize > J2K_PACKET_MAX_TAG_NODES {
            return false;
        }

        let level = shape.levels as usize;
        shape.widths[level] = w;
        shape.heights[level] = h;
        shape.offsets[level] = shape.total_nodes;
        shape.total_nodes = next_total;
        shape.levels += 1;
        if w <= 1 && h <= 1 {
            return true;
        }
        w = (w + 1) >> 1;
        h = (h + 1) >> 1;
    }
}

#[inline(always)]
pub(crate) fn j2k_packet_tag_tree_init(
    tree: &mut J2kPacketTagTree<'_>,
    width: u32,
    height: u32,
) -> bool {
    let mut shape = j2k_packet_tag_tree_shape_new();
    if !j2k_packet_tag_tree_shape(&mut shape, width, height) {
        return j2k_packet_tag_tree_fail(tree);
    }

    tree.widths = shape.widths;
    tree.heights = shape.heights;
    tree.offsets = shape.offsets;
    tree.levels = shape.levels;
    tree.total_nodes = shape.total_nodes;
    tree.failed = 0;
    let mut idx = 0_u32;
    while idx < shape.total_nodes {
        let slot = idx as usize;
        tree.values[slot] = 0;
        tree.current[slot] = 0;
        tree.known[slot] = 0;
        idx += 1;
    }
    true
}

#[inline(always)]
pub(crate) fn j2k_packet_tag_tree_propagate(tree: &mut J2kPacketTagTree<'_>) {
    let mut level = 1_u32;
    while level < tree.levels {
        let prev_w = tree.widths[(level - 1) as usize];
        let prev_h = tree.heights[(level - 1) as usize];
        let curr_w = tree.widths[level as usize];
        let curr_h = tree.heights[level as usize];
        let mut cy = 0_u32;
        while cy < curr_h {
            let mut cx = 0_u32;
            while cx < curr_w {
                let mut min_value = u32::MAX;
                let child_x0 = cx << 1;
                let child_y0 = cy << 1;
                let child_x1 = (child_x0 + 2).min(prev_w);
                let child_y1 = (child_y0 + 2).min(prev_h);
                let mut y = child_y0;
                while y < child_y1 {
                    let mut x = child_x0;
                    while x < child_x1 {
                        let child_idx = tree.offsets[(level - 1) as usize] + y * prev_w + x;
                        min_value = min_value.min(tree.values[child_idx as usize]);
                        x += 1;
                    }
                    y += 1;
                }
                let node_idx = tree.offsets[level as usize] + cy * curr_w + cx;
                tree.values[node_idx as usize] = min_value;
                cx += 1;
            }
            cy += 1;
        }
        level += 1;
    }
}

#[inline(always)]
pub(crate) fn j2k_packet_build_tag_trees(
    inclusion_tree: &mut J2kPacketTagTree<'_>,
    zbp_tree: &mut J2kPacketTagTree<'_>,
    subband: J2kHtPacketSubband,
    blocks: *const J2kHtPacketBlock,
    tag_states: *const J2kHtPacketSubbandTagState,
    tag_nodes: *const J2kHtPacketTagNodeState,
    tag_state_count: u64,
    tag_node_count: u64,
    subband_meta_idx: u32,
) {
    if !j2k_packet_tag_tree_init(inclusion_tree, subband.num_cbs_x, subband.num_cbs_y)
        || !j2k_packet_tag_tree_init(zbp_tree, subband.num_cbs_x, subband.num_cbs_y)
    {
        inclusion_tree.failed = 1;
        zbp_tree.failed = 1;
        return;
    }

    let mut idx = 0_u32;
    while idx < subband.block_count {
        let block = load_job(blocks, subband.block_start + idx);
        let x = idx % subband.num_cbs_x;
        let y = idx / subband.num_cbs_x;
        let leaf_idx = y * subband.num_cbs_x + x;
        inclusion_tree.values[leaf_idx as usize] = if block.previously_included == 0 {
            block.inclusion_layer
        } else {
            J2K_PACKET_TAG_INF
        };
        zbp_tree.values[leaf_idx as usize] = block.num_zero_bitplanes;
        idx += 1;
    }
    j2k_packet_tag_tree_propagate(inclusion_tree);
    j2k_packet_tag_tree_propagate(zbp_tree);

    if tag_state_count == 0 {
        return;
    }
    if subband_meta_idx as u64 >= tag_state_count {
        inclusion_tree.failed = 1;
        zbp_tree.failed = 1;
        return;
    }
    let state = load_job(tag_states, subband_meta_idx);
    if state.node_count != inclusion_tree.total_nodes {
        inclusion_tree.failed = 1;
        zbp_tree.failed = 1;
        return;
    }
    let Some(inclusion_end) =
        (state.inclusion_node_start as u64).checked_add(state.node_count as u64)
    else {
        inclusion_tree.failed = 1;
        zbp_tree.failed = 1;
        return;
    };
    let Some(zbp_end) =
        (state.zero_bitplane_node_start as u64).checked_add(state.node_count as u64)
    else {
        inclusion_tree.failed = 1;
        zbp_tree.failed = 1;
        return;
    };
    if inclusion_end > tag_node_count || zbp_end > tag_node_count {
        inclusion_tree.failed = 1;
        zbp_tree.failed = 1;
        return;
    }

    let mut node_idx = 0_u32;
    while node_idx < state.node_count {
        let inclusion_node = load_job(tag_nodes, state.inclusion_node_start + node_idx);
        let zbp_node = load_job(tag_nodes, state.zero_bitplane_node_start + node_idx);
        let slot = node_idx as usize;
        inclusion_tree.current[slot] = inclusion_node.current;
        inclusion_tree.known[slot] = inclusion_node.known;
        zbp_tree.current[slot] = zbp_node.current;
        zbp_tree.known[slot] = zbp_node.known;
        node_idx += 1;
    }
}

#[inline(always)]
pub(crate) fn j2k_packet_tag_tree_encode(
    tree: &mut J2kPacketTagTree<'_>,
    x: u32,
    y: u32,
    max_value: u32,
    writer: &mut J2kPacketBitWriter,
) {
    let mut path = [0_u32; J2K_PACKET_MAX_TAG_LEVELS];
    let mut cx = x;
    let mut cy = y;
    let mut level = 0_u32;
    while level < tree.levels {
        path[level as usize] = tree.offsets[level as usize] + cy * tree.widths[level as usize] + cx;
        cx >>= 1;
        cy >>= 1;
        level += 1;
    }

    let mut parent_value = 0_u32;
    let mut reverse = tree.levels;
    while reverse > 0 {
        let node_idx = path[(reverse - 1) as usize];
        let slot = node_idx as usize;
        let mut start = tree.current[slot].max(parent_value);
        if tree.known[slot] == 0 {
            let target = tree.values[slot].min(max_value);
            while start < target {
                j2k_packet_write_bit(writer, 0);
                start += 1;
            }
            if tree.values[slot] < max_value {
                j2k_packet_write_bit(writer, 1);
                tree.known[slot] = 1;
            }
            tree.current[slot] = target;
        }
        parent_value = tree.current[slot];
        reverse -= 1;
    }
}
