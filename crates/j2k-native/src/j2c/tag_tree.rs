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

/// One node of a [`TagTree`].
#[derive(Debug, PartialEq, Eq, Clone, Copy, Default)]
pub(crate) struct TagNode {
    /// The value read so far: the node's value once `initialized`, before
    /// that a lower bound for it.
    value: u32,
    /// Whether the node's value is known. The tag tree is not stored in its
    /// complete form, but each packet contributes the information it needs,
    /// so a node is initialized when a read first crosses it completely.
    initialized: bool,
}

/// A tag tree over a `width` x `height` grid of leaves, as consecutive
/// nodes: one row-major grid per level, from the root down to the leaves.
/// The node at level `l` above leaf (`x`, `y`) is (`x >> l`, `y >> l`) of a
/// grid `ceil(width / 2^l)` wide, as in the quad tree of Section B.10.2.
#[derive(Copy, Clone)]
pub(crate) struct TagTree {
    root: usize,
    width: u32,
    height: u32,
}

impl TagTree {
    pub(crate) fn new(width: u32, height: u32, nodes: &mut Vec<TagNode>) -> Self {
        let count = node_count(width, height);
        #[cfg(test)]
        tests::BUILD_VISITS.with(|visits| visits.set(visits.get() + count));
        let root = nodes.len();
        nodes.resize(root + count, TagNode::default());
        Self {
            root,
            width,
            height,
        }
    }

    /// A tree over a `width` x `height` grid that holds no nodes until
    /// [`Self::place`] gives it some. Reads fail until then.
    pub(crate) const fn unplaced(width: u32, height: u32) -> Self {
        Self {
            root: UNPLACED,
            width,
            height,
        }
    }

    pub(crate) const fn width(&self) -> u32 {
        self.width
    }

    pub(crate) const fn height(&self) -> u32 {
        self.height
    }

    /// The nodes the tree takes.
    pub(crate) fn node_count(&self) -> usize {
        node_count(self.width, self.height)
    }

    /// Start the tree afresh on the nodes from `root`, as [`Self::new`]
    /// would build it there. Fails if they don't fit in `nodes`.
    pub(crate) fn place(&mut self, root: usize, nodes: &mut [TagNode]) -> Option<()> {
        let end = root.checked_add(self.node_count())?;
        nodes.get_mut(root..end)?.fill(TagNode::default());
        self.root = root;
        Some(())
    }

    /// Give up the tree's nodes, keeping its grid.
    pub(crate) fn unplace(&mut self) {
        self.root = UNPLACED;
    }

    /// Read the value of leaf (`x`, `y`), or as much of it as shows whether
    /// it is below `max_val`, walking the levels from the root.
    pub(crate) fn read(
        &mut self,
        x: u32,
        y: u32,
        reader: &mut BitReader<'_>,
        max_val: u32,
        nodes: &mut [TagNode],
    ) -> Option<u32> {
        debug_assert!(x < self.width && y < self.height);
        let mut level_start = self.root;
        let mut parent_val = 0;
        // The root's level; the leaves are level zero.
        let top_level = ceil_log2_u32(self.width.max(self.height));
        for level in (0..=top_level).rev() {
            let width = level_len(self.width, level);
            let index = (y >> level) as usize * width + (x >> level) as usize;
            let node = nodes.get_mut(level_start.checked_add(index)?)?;

            if !node.initialized {
                let mut val = u32::max(parent_val, node.value);

                while val < max_val {
                    // "Each node has an associated current value, which is
                    // initialized to zero (the minimum). A 0 bit in the tag tree
                    // means that the minimum (or the value in the case of the
                    // highest level) is larger than the current value and a 1 bit
                    // means that the minimum (or the value in the case of the
                    // highest level) is equal to the current value."
                    if reader.read_bits_with_stuffing(1)? == 1 {
                        node.initialized = true;
                        break;
                    }
                    val += 1;
                }

                node.value = val;
            }

            // Stop at the leaf, or when the minimum of all children is too large.
            if node.value >= max_val || level == 0 {
                return Some(node.value);
            }
            parent_val = node.value;
            level_start += width * level_len(self.height, level);
        }
        None
    }
}

/// The root of a tree that holds no nodes: no node index reaches it.
const UNPLACED: usize = usize::MAX;

/// Nodes in a tree over a `width` x `height` grid. An empty grid keeps a lone
/// root, which no read reaches.
fn node_count(width: u32, height: u32) -> usize {
    if width == 0 || height == 0 {
        return 1;
    }
    let top_level = ceil_log2_u32(width.max(height));
    (0..=top_level)
        .map(|level| level_len(width, level) * level_len(height, level))
        .sum()
}

/// Nodes along one axis of `len` leaves at `level`: `ceil(len / 2^level)`.
fn level_len(len: u32, level: u8) -> usize {
    (len.saturating_sub(1) >> level) as usize + 1
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
