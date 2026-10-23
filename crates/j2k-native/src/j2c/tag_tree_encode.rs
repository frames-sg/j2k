//! Tag tree encoder for JPEG 2000 packet headers.
//!
//! The tag tree is a hierarchical structure used to efficiently encode
//! code-block inclusion information and zero bitplane counts in packet headers.
//! See Annex B.10.2 of ITU-T T.800.

use super::encode::allocation::{BudgetedVec, EncodeAllocationLedger};
use crate::writer::FallibleBitWriter;
use crate::{EncodeError, EncodeResult};

const MAX_TAG_TREE_LEVELS: usize = 33;

/// A single node in the tag tree.
#[derive(Debug, Clone)]
struct TagNode {
    value: u32,
    current_value: u32,
    known: bool,
}

/// Tag tree encoder.
///
/// Encodes values using a hierarchical min-tree structure where each parent
/// node's value is the minimum of its children. This allows efficient encoding
/// by skipping already-known ranges.
#[derive(Debug)]
pub(crate) struct TagTreeEncoder<'a> {
    nodes: BudgetedVec<'a, TagNode>,
    width: u32,
    height: u32,
    num_levels: usize,
    level_offsets: [usize; MAX_TAG_TREE_LEVELS],
}

impl<'a> TagTreeEncoder<'a> {
    /// Create a new tag tree for a grid of `width × height` leaf values.
    pub(crate) fn try_new(
        width: u32,
        height: u32,
        allocations: &'a EncodeAllocationLedger,
    ) -> EncodeResult<Self> {
        if (width == 0) != (height == 0) {
            return Err(EncodeError::InvalidInput {
                what: "tag-tree dimensions must both be zero or both be nonzero",
            });
        }

        let mut level_offsets = [0usize; MAX_TAG_TREE_LEVELS];
        let mut total_nodes = 0usize;
        let mut w = width;
        let mut h = height;
        let mut num_levels = 0usize;

        loop {
            if num_levels >= MAX_TAG_TREE_LEVELS {
                return Err(EncodeError::InternalInvariant {
                    what: "tag-tree depth exceeded the u32 geometry bound",
                });
            }
            level_offsets[num_levels] = total_nodes;
            let level_nodes_u32 = w.checked_mul(h).ok_or(EncodeError::ArithmeticOverflow {
                what: "tag-tree level node count",
            })?;
            let level_nodes =
                usize::try_from(level_nodes_u32).map_err(|_| EncodeError::ArithmeticOverflow {
                    what: "tag-tree level node count",
                })?;
            total_nodes =
                total_nodes
                    .checked_add(level_nodes)
                    .ok_or(EncodeError::ArithmeticOverflow {
                        what: "tag-tree total node count",
                    })?;
            num_levels += 1;

            if w <= 1 && h <= 1 {
                break;
            }

            w = w.div_ceil(2);
            h = h.div_ceil(2);
        }

        let mut nodes =
            allocations.try_vec_with_capacity(total_nodes, "tag-tree node capacity exhausted")?;
        for _ in 0..total_nodes {
            nodes.try_push(TagNode {
                value: 0,
                current_value: 0,
                known: false,
            })?;
        }

        Ok(Self {
            nodes,
            width,
            height,
            num_levels,
            level_offsets,
        })
    }

    /// Set every leaf from `values` in row-major order, then each parent to
    /// the minimum of its children, one level at a time.
    pub(crate) fn set_leaves(
        &mut self,
        values: impl ExactSizeIterator<Item = u32>,
    ) -> EncodeResult<()> {
        let (mut width, mut height) = (self.width as usize, self.height as usize);
        let leaves = self
            .nodes
            .get_mut(..width * height)
            .filter(|leaves| leaves.len() == values.len())
            .ok_or(EncodeError::InvalidInput {
                what: "tag-tree leaf count differs from its grid",
            })?;
        for (leaf, value) in leaves.iter_mut().zip(values) {
            leaf.value = value;
        }
        for level in 1..self.num_levels {
            let (lower, upper) = self.nodes.split_at_mut(self.level_offsets[level]);
            let children = &lower[self.level_offsets[level - 1]..];
            let parent_width = width.div_ceil(2);
            for (index, parent) in upper[..parent_width * height.div_ceil(2)]
                .iter_mut()
                .enumerate()
            {
                let (x, y) = (2 * (index % parent_width), 2 * (index / parent_width));
                parent.value = (y..height.min(y + 2))
                    .flat_map(|row| &children[row * width + x..row * width + width.min(x + 2)])
                    .map(|child| child.value)
                    .fold(u32::MAX, u32::min);
            }
            (width, height) = (parent_width, height.div_ceil(2));
        }
        Ok(())
    }

    /// Encode the value at leaf position (x, y) up to threshold `max_val`.
    ///
    /// Writes bits to `writer` following the tag tree coding procedure (B.10.2).
    /// Returns the encoded value if it was below `max_val`, or None if >= `max_val`.
    pub(crate) fn encode(
        &mut self,
        x: u32,
        y: u32,
        max_val: u32,
        writer: &mut impl FallibleBitWriter,
    ) -> EncodeResult<()> {
        if x >= self.width || y >= self.height {
            return Err(EncodeError::InvalidInput {
                what: "tag-tree encode coordinate is out of range",
            });
        }
        // Build path from root to leaf
        let mut path = [0usize; MAX_TAG_TREE_LEVELS];
        let mut cx = x;
        let mut cy = y;
        let mut cw = self.width;

        for (level, path_entry) in path.iter_mut().enumerate().take(self.num_levels) {
            *path_entry = checked_node_index(self.level_offsets[level], cw, cx, cy)?;
            cx /= 2;
            cy /= 2;
            cw = cw.div_ceil(2);
        }

        // Encode from root to leaf
        let mut parent_val = 0u32;
        for &node_idx in path[..self.num_levels].iter().rev() {
            let node = self
                .nodes
                .get_mut(node_idx)
                .ok_or(EncodeError::InternalInvariant {
                    what: "tag-tree encode path exceeded allocated nodes",
                })?;
            let start = node.current_value.max(parent_val);

            if !node.known {
                let target = node.value.min(max_val);
                for _ in start..target {
                    writer.try_write_bit(0)?; // Value is at least the next threshold.
                }
                if node.value < max_val {
                    writer.try_write_bit(1)?; // Value is exactly this.
                    node.known = true;
                }
                node.current_value = target;
            }

            parent_val = node.current_value;
        }
        Ok(())
    }
}

fn checked_node_index(offset: usize, width: u32, x: u32, y: u32) -> EncodeResult<usize> {
    let row_offset = y
        .checked_mul(width)
        .and_then(|row| row.checked_add(x))
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "tag-tree node index",
        })?;
    let row_offset = usize::try_from(row_offset).map_err(|_| EncodeError::ArithmeticOverflow {
        what: "tag-tree node index",
    })?;
    offset
        .checked_add(row_offset)
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "tag-tree node index",
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::j2c::encode::allocation::EncodeAllocationLedger;
    use crate::j2c::tag_tree::{TagNode, TagTree};
    use crate::reader::BitReader;
    use crate::writer::CheckedBitWriter;
    use alloc::vec::Vec;

    fn checked_writer(allocations: &EncodeAllocationLedger) -> CheckedBitWriter<'_> {
        CheckedBitWriter::try_with_capacity(allocations, 64 * 1024, "test tag-tree header")
            .expect("checked tag-tree writer")
    }

    #[test]
    fn test_single_value() {
        let allocations = EncodeAllocationLedger::new(0).expect("test allocation ledger");
        let mut tree = TagTreeEncoder::try_new(1, 1, &allocations).expect("valid tree");
        tree.set_leaves(core::iter::once(3)).expect("valid leaves");

        let mut writer = checked_writer(&allocations);
        tree.encode(0, 0, 4, &mut writer).expect("encode tree");
        let data = writer.try_finish().expect("finish tag-tree header");
        // Should encode: 0, 0, 0, 1 (three zeros then one)
        assert_eq!(data.as_slice(), &[0b0001_0000]);
    }

    #[test]
    fn test_2x2_tree() {
        assert_values_round_trip(2, 2, &[0, 1, 2, 3]);
    }

    #[test]
    fn test_new_tree_dimensions() {
        let allocations = EncodeAllocationLedger::new(0).expect("test allocation ledger");
        let tree = TagTreeEncoder::try_new(4, 4, &allocations).expect("valid tree");
        // 4×4 → 2×2 → 1×1 = 3 levels
        assert_eq!(tree.num_levels, 3);
        // 16 + 4 + 1 = 21 nodes
        assert_eq!(tree.nodes.len(), 21);
    }

    #[test]
    fn one_row_odd_width_tree_sets_all_leaf_values() {
        assert_values_round_trip(31, 1, &(0..31).collect::<Vec<_>>());
    }

    #[test]
    fn one_column_odd_height_tree_sets_all_leaf_values() {
        assert_values_round_trip(1, 31, &(0..31).collect::<Vec<_>>());
    }

    #[test]
    fn varied_8x8_values_round_trip_against_decoder_tag_tree() {
        let values = [
            2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 3, 2, 1, 1, 1, 1, 2, 3, 2, 2, 1,
            1, 1, 1, 2, 3, 2, 2, 1, 1, 1, 1, 2, 2, 2, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2,
            2, 2, 2, 1, 1, 1,
        ];
        assert_values_round_trip(8, 8, &values);
    }

    fn assert_values_round_trip(width: u32, height: u32, values: &[u32]) {
        assert_eq!(values.len(), usize::try_from(width * height).unwrap());
        let allocations = EncodeAllocationLedger::new(0).expect("test allocation ledger");
        let mut encoder = TagTreeEncoder::try_new(width, height, &allocations).expect("valid tree");
        encoder
            .set_leaves(values.iter().copied())
            .expect("valid leaves");

        let mut writer = checked_writer(&allocations);
        for (idx, value) in values.iter().copied().enumerate() {
            let index = u32::try_from(idx).expect("test tag-tree index fits u32");
            encoder
                .encode(index % width, index / width, value + 1, &mut writer)
                .expect("encode tree");
        }
        let bytes = writer.try_finish().expect("finish tag-tree header");

        let mut nodes = Vec::<TagNode>::new();
        let mut decoder = TagTree::new(width, height, &mut nodes);
        let mut reader = BitReader::new(&bytes);
        for (idx, expected) in values.iter().copied().enumerate() {
            let index = u32::try_from(idx).expect("test tag-tree index fits u32");
            let actual = decoder
                .read(
                    index % width,
                    index / width,
                    &mut reader,
                    u32::MAX,
                    &mut nodes,
                )
                .expect("tag tree decode");
            assert_eq!(actual, expected, "index {idx}");
        }
    }
}
