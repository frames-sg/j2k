//! The tag tree, described in Section B.10.2.
//!
//! Tag trees are quad trees where each leaf stores an integer value.
//! Each intermediate node stores the smallest value of all of its children.
//! For example, if a node stores the value 3, it means that all children
//! have a value of 3 or higher. The root node therefore stores the smallest
//! values across all children.

use alloc::vec::Vec;

use crate::math::ceil_log2_u32;
use crate::reader::BitReader;

#[derive(Debug, PartialEq, Eq, Clone, Default)]
pub(crate) struct TagNode {
    /// The width of the area covered by the node.
    ///
    /// For leaf nodes, this value is always 1. In some cases, the width might
    /// be 0, in which case the leaf node doesn't actually "exist" and is just
    /// a dummy node.
    width: u32,
    /// The height of the area covered by the node.
    ///
    /// For leaf nodes, this value is always 1. In some cases, the height might
    /// be 0, in which case the leaf node doesn't actually "exist" and is just
    /// a dummy node.
    height: u32,
    /// The actual value stored in the node. Only valid once `initialized`
    /// is set to `true`.
    value: u32,
    /// Whether the node has been fully initialized. The tag tree is not
    /// stored in its complete form in the JP2 file, but is instead built
    /// up incrementally, each packet contributing the information of the
    /// tag tree. The node is therefore only initialized with its actual
    /// value once we cross it the first time.
    initialized: bool,
    /// The level inside the tree. Zero indicates that the given node is
    /// a leaf node, otherwise the level is > 0. The root node has the highest
    /// level.
    level: u16,
    /// The indices of the children of the node, some of which might be dummy
    /// nodes (indicated by the fact that the index is `usize::MAX`).
    children: [usize; 4],
}

impl TagNode {
    fn new(width: u32, height: u32, level: u16) -> Self {
        Self {
            width,
            height,
            level,
            value: 0,
            initialized: false,
            children: [usize::MAX, usize::MAX, usize::MAX, usize::MAX],
        }
    }

    /// The width of the top-left child.
    fn top_left_width(&self) -> u32 {
        u32::min(1 << (self.level - 1), self.width)
    }

    /// The height of the top-left child.
    fn top_left_height(&self) -> u32 {
        u32::min(1 << (self.level - 1), self.height)
    }
}

impl TagNode {
    fn build(width: u32, height: u32, level: u16, nodes: &mut Vec<Self>) -> Self {
        #[cfg(test)]
        tests::BUILD_VISITS.with(|visits| visits.set(visits.get() + 1));
        let mut tag = Self::new(width, height, level);

        // Empty quadrants have no descendants. Expanding them to leaf depth
        // would make a narrow tree cost as much as its enclosing square.
        if width == 0 || height == 0 {
            return tag;
        }

        if level == 0 {
            // We reached the leaf node.
            assert!(width <= 1 && height <= 1);

            return tag;
        }

        // Determine the width and height of the top-left child node. Based
        // on this, we can infer the dimensions of all other child nodes.
        let top_left_width = tag.top_left_width();
        let top_left_height = tag.top_left_height();

        let mut push = |node: Self, child_idx: usize, nodes: &mut Vec<Self>| {
            // If this is not the case, the child doesn't actually exist.
            if node.width > 0 && node.height > 0 {
                let node_idx = nodes.len();
                nodes.push(node);
                tag.children[child_idx] = node_idx;
            }
        };

        // We always push four children, but some nodes might in reality have
        // fewer than that. In this case, the resulting node will simply have
        // a width or height of 0 and we can recognize that it technically
        // doesn't exist.
        let n1 = Self::build(top_left_width, top_left_height, level - 1, nodes);
        push(n1, 0, nodes);

        let n2 = Self::build(width - top_left_width, top_left_height, level - 1, nodes);
        push(n2, 1, nodes);

        let n3 = Self::build(top_left_width, height - top_left_height, level - 1, nodes);
        push(n3, 2, nodes);

        let n4 = Self::build(
            width - top_left_width,
            height - top_left_height,
            level - 1,
            nodes,
        );
        push(n4, 3, nodes);

        tag
    }
}

fn read_tag_node(
    node_idx: usize,
    x: u32,
    y: u32,
    reader: &mut BitReader<'_>,
    parent_val: u32,
    max_val: u32,
    nodes: &mut [TagNode],
) -> Option<u32> {
    let node = &mut nodes[node_idx];

    if !node.initialized {
        let mut val = u32::max(parent_val, node.value);

        loop {
            if val >= max_val {
                break;
            }

            // "Each node has an associated current value, which is
            // initialized to zero (the minimum). A 0 bit in the tag tree
            // means that the minimum (or the value in the case of the
            // highest level) is larger than the current value and a 1 bit
            // means that the minimum (or the value in the case of the
            // highest level) is equal to the current value."
            match reader.read_bits_with_stuffing(1)? {
                0 => val += 1,
                1 => {
                    node.initialized = true;
                    break;
                }
                _ => unreachable!(),
            }
        }

        node.value = val;
    }

    // Abort early if we already reached the leaf node or the minimum
    // value of all children is too large.
    if node.value >= max_val || node.level == 0 {
        return Some(node.value);
    }

    let top_left_width = node.top_left_width();
    let top_left_height = node.top_left_height();

    let left = x < top_left_width;
    let top = y < top_left_height;

    match (left, top) {
        (true, true) => read_tag_node(node.children[0], x, y, reader, node.value, max_val, nodes),
        (false, true) => read_tag_node(
            node.children[1],
            x - top_left_width,
            y,
            reader,
            node.value,
            max_val,
            nodes,
        ),
        (true, false) => read_tag_node(
            node.children[2],
            x,
            y - top_left_height,
            reader,
            node.value,
            max_val,
            nodes,
        ),
        (false, false) => read_tag_node(
            node.children[3],
            x - top_left_width,
            y - top_left_height,
            reader,
            node.value,
            max_val,
            nodes,
        ),
    }
}

#[derive(Copy, Clone)]
pub(crate) struct TagTree {
    root: usize,
    width: u32,
    height: u32,
}

impl TagTree {
    pub(crate) fn new(width: u32, height: u32, nodes: &mut Vec<TagNode>) -> Self {
        // Calculate how many levels the tree has in total.
        let level = u16::from(ceil_log2_u32(width.max(height)));
        let node = TagNode::build(width, height, level, nodes);
        let idx = nodes.len();
        nodes.push(node);

        Self {
            root: idx,
            width,
            height,
        }
    }

    pub(crate) fn read(
        &mut self,
        x: u32,
        y: u32,
        reader: &mut BitReader<'_>,
        max_val: u32,
        nodes: &mut [TagNode],
    ) -> Option<u32> {
        debug_assert!(x < self.width && y < self.height);

        read_tag_node(self.root, x, y, reader, 0, max_val, nodes)
    }
}

#[cfg(test)]
mod tests {
    use crate::{DecodeError, DecodeSettings, DecoderContext, DecodingError, Image};
    use core::cell::Cell;

    std::thread_local! {
        pub(super) static BUILD_VISITS: Cell<usize> = const { Cell::new(0) };
    }

    #[test]
    fn truncated_tall_codestream_bounds_tag_tree_work() {
        let data = [
            255, 79, 255, 81, 0, 47, 0, 0, 0, 0, 0, 16, 0, 0, 90, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 16, 32, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 1, 1, 7, 1, 1, 7, 1, 1, 255, 82,
            0, 12, 0, 0, 0, 4, 0, 2, 0, 0, 38, 1, 255, 92, 0, 17, 66, 32, 82, 80, 5, 80, 5, 80, 71,
            87, 211, 87, 211, 87, 98, 255, 100, 0, 37, 0, 1, 67, 114, 101, 97, 68, 101, 100, 32,
            98, 121, 32, 79, 10, 101, 111, 110, 32, 50, 46, 53, 46, 52, 255, 144, 0, 10, 80, 32,
            101, 46, 53, 46, 52, 255, 144, 0, 10, 0, 0, 0, 0, 0, 41, 0, 1, 255, 147, 139, 186, 199,
            233, 198, 8, 40, 97, 69, 24, 38, 53, 12, 80, 199, 10, 0, 0, 0, 0, 0, 0, 0, 18, 0, 133,
            128, 255, 217,
        ];
        for reduction in 0..=2 {
            let image = Image::new_with_reduction(&data, &DecodeSettings::default(), reduction)
                .expect("valid header");

            let mut context = DecoderContext::default();
            BUILD_VISITS.with(|visits| visits.set(0));
            let result = image.decode_region_components_for_integer_output_with_context(
                (0, 0, 1, 1),
                &mut context,
            );
            assert!(matches!(
                result,
                Err(DecodeError::Decoding(DecodingError::PacketParseFailure(_)))
            ));
            // Even a full tile should require only linear tree construction work,
            // independent of the enclosing square of its highly skewed grid.
            let visits = BUILD_VISITS.with(Cell::get);
            assert!(
                visits < 1_000_000,
                "excessive tag-tree work at reduction {reduction}: {visits} visits"
            );
        }
    }
}
