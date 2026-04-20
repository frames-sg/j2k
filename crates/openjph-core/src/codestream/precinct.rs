//! Precinct processing.
//!
//! Port of `ojph_precinct.h/cpp`. A precinct groups codeblocks from the
//! same resolution level. It also contains tag trees for signalling
//! inclusion and zero-bitplane information.

#![allow(dead_code)]

use crate::types::*;

// =========================================================================
// Tag Tree
// =========================================================================

/// A tag tree for encoding/decoding inclusion and zero-bitplane information.
///
/// The tag tree is a hierarchical structure where each node stores a value.
/// Leaf nodes correspond to codeblocks. Internal nodes summarize children.
#[derive(Debug, Clone, Default)]
pub struct TagTree {
    /// Node values at each level, stored level by level.
    nodes: Vec<u32>,
    /// Width at each level.
    widths: Vec<u32>,
    /// Height at each level.
    heights: Vec<u32>,
    /// Number of levels.
    num_levels: u32,
}

impl TagTree {
    /// Build a tag tree for a grid of `width` × `height` codeblocks.
    pub fn new(width: u32, height: u32) -> Self {
        if width == 0 || height == 0 {
            return Self::default();
        }
        let mut widths = Vec::new();
        let mut heights = Vec::new();
        let mut w = width;
        let mut h = height;
        loop {
            widths.push(w);
            heights.push(h);
            if w == 1 && h == 1 {
                break;
            }
            w = div_ceil(w, 2);
            h = div_ceil(h, 2);
        }
        let num_levels = widths.len() as u32;
        let total: u32 = widths
            .iter()
            .zip(heights.iter())
            .map(|(&w, &h)| w * h)
            .sum();
        let nodes = vec![0u32; total as usize];
        Self {
            nodes,
            widths,
            heights,
            num_levels,
        }
    }

    /// Get the value at a leaf position (codeblock index).
    pub fn get_value(&self, x: u32, y: u32) -> u32 {
        if self.num_levels == 0 {
            return 0;
        }
        let idx = y * self.widths[0] + x;
        self.nodes[idx as usize]
    }

    /// Set the value at a leaf position.
    pub fn set_value(&mut self, x: u32, y: u32, val: u32) {
        if self.num_levels == 0 {
            return;
        }
        let idx = y * self.widths[0] + x;
        self.nodes[idx as usize] = val;
    }

    /// Number of levels in the tree.
    pub fn levels(&self) -> u32 {
        self.num_levels
    }
}

// =========================================================================
// Precinct
// =========================================================================

/// A precinct within a resolution level.
///
/// In JPEG 2000, each resolution level is partitioned into precincts. Each
/// precinct covers a rectangular region of codeblocks.
#[derive(Debug, Clone)]
pub struct Precinct {
    /// Precinct rectangle in the resolution grid.
    pub prec_rect: Rect,
    /// Number of codeblocks in x direction.
    pub num_cbs_x: u32,
    /// Number of codeblocks in y direction.
    pub num_cbs_y: u32,
    /// Tag tree for inclusion information.
    pub inclusion_tree: TagTree,
    /// Tag tree for zero-bitplane (missing MSBs) information.
    pub zero_bitplane_tree: TagTree,
    /// Whether this precinct uses SOP markers.
    pub uses_sop: bool,
    /// Whether this precinct uses EPH markers.
    pub uses_eph: bool,
}

impl Default for Precinct {
    fn default() -> Self {
        Self {
            prec_rect: Rect::new(Point::new(0, 0), Size::new(0, 0)),
            num_cbs_x: 0,
            num_cbs_y: 0,
            inclusion_tree: TagTree::default(),
            zero_bitplane_tree: TagTree::default(),
            uses_sop: false,
            uses_eph: false,
        }
    }
}

impl Precinct {
    /// Create a new precinct with the given parameters.
    pub fn new(rect: Rect, num_cbs_x: u32, num_cbs_y: u32) -> Self {
        let inclusion_tree = TagTree::new(num_cbs_x, num_cbs_y);
        let zero_bitplane_tree = TagTree::new(num_cbs_x, num_cbs_y);
        Self {
            prec_rect: rect,
            num_cbs_x,
            num_cbs_y,
            inclusion_tree,
            zero_bitplane_tree,
            uses_sop: false,
            uses_eph: false,
        }
    }

    /// Total number of codeblocks in this precinct.
    pub fn num_codeblocks(&self) -> u32 {
        self.num_cbs_x * self.num_cbs_y
    }

    /// True if this precinct has zero area.
    pub fn is_empty(&self) -> bool {
        self.prec_rect.siz.w == 0 || self.prec_rect.siz.h == 0
    }
}
