// SPDX-License-Identifier: MIT OR Apache-2.0

//! Whole-tile convex selection across alternate HT sets.

use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::j2c::bitplane_encode::EncodedCodeBlock;
use crate::{EncodeError, EncodeResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HtCandidateRange {
    pub(crate) start: usize,
    pub(crate) len: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HtCandidateSelection {
    pub(crate) candidate_index: usize,
    pub(crate) num_coding_passes: u8,
}

#[derive(Clone, Copy, Debug)]
struct HtCandidatePoint {
    candidate_index: usize,
    num_coding_passes: u8,
    rate: u64,
    distortion: f64,
}

pub(crate) fn tile_candidate_selection_workspace_bytes(
    candidate_count: usize,
    family_count: usize,
) -> EncodeResult<usize> {
    let point_count = candidate_count
        .checked_mul(9)
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "HTJ2K tile candidate workspace point count",
        })?;
    let point_bytes = point_count
        .checked_mul(core::mem::size_of::<HtCandidatePoint>())
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "HTJ2K tile candidate workspace points",
        })?;
    let range_bytes = family_count
        .checked_mul(core::mem::size_of::<HtCandidateRange>())
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "HTJ2K tile candidate workspace ranges",
        })?;
    let selection_bytes = family_count
        .checked_mul(core::mem::size_of::<HtCandidateSelection>())
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "HTJ2K tile candidate workspace selections",
        })?;
    let frontier_bytes = family_count
        .checked_mul(core::mem::size_of::<usize>())
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "HTJ2K tile candidate workspace frontiers",
        })?;
    point_bytes
        .checked_add(range_bytes)
        .and_then(|bytes| bytes.checked_add(selection_bytes))
        .and_then(|bytes| bytes.checked_add(frontier_bytes))
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "HTJ2K tile candidate workspace",
        })
}

pub(crate) fn select_tile_code_block_candidates(
    candidates: &[EncodedCodeBlock],
    ranges: &[HtCandidateRange],
    byte_budget: u64,
) -> EncodeResult<Vec<HtCandidateSelection>> {
    let point_capacity =
        candidates
            .len()
            .checked_mul(3)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "HTJ2K tile candidate point count",
            })?;
    let mut hull_points = Vec::new();
    hull_points.try_reserve_exact(point_capacity).map_err(|_| {
        EncodeError::HostAllocationFailed {
            what: "HTJ2K tile candidate points",
            bytes: point_capacity.saturating_mul(core::mem::size_of::<HtCandidatePoint>()),
        }
    })?;
    let mut hull_ranges = Vec::new();
    hull_ranges
        .try_reserve_exact(ranges.len())
        .map_err(|_| EncodeError::HostAllocationFailed {
            what: "HTJ2K tile candidate hull ranges",
            bytes: ranges
                .len()
                .saturating_mul(core::mem::size_of::<HtCandidateRange>()),
        })?;

    for &range in ranges {
        let family = candidate_family(candidates, range)?;
        let start = hull_points.len();
        append_candidate_hull(family, range.start, &mut hull_points)?;
        hull_ranges.push(HtCandidateRange {
            start,
            len: hull_points.len() - start,
        });
    }

    allocate_hull_points(&hull_points, &hull_ranges, byte_budget)
}

fn allocate_hull_points(
    hull_points: &[HtCandidatePoint],
    hull_ranges: &[HtCandidateRange],
    byte_budget: u64,
) -> EncodeResult<Vec<HtCandidateSelection>> {
    let mut selections = Vec::new();
    selections
        .try_reserve_exact(hull_ranges.len())
        .map_err(|_| EncodeError::HostAllocationFailed {
            what: "HTJ2K tile candidate selections",
            bytes: hull_ranges
                .len()
                .saturating_mul(core::mem::size_of::<HtCandidateSelection>()),
        })?;
    let mut frontiers = Vec::new();
    frontiers
        .try_reserve_exact(hull_ranges.len())
        .map_err(|_| EncodeError::HostAllocationFailed {
            what: "HTJ2K tile candidate frontiers",
            bytes: hull_ranges
                .len()
                .saturating_mul(core::mem::size_of::<usize>()),
        })?;
    let mut used = 0u64;
    for range in hull_ranges {
        let first = *hull_points
            .get(range.start)
            .ok_or(EncodeError::InternalInvariant {
                what: "HTJ2K tile candidate hull is empty",
            })?;
        used = used
            .checked_add(first.rate)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "HTJ2K tile candidate cleanup budget",
            })?;
        selections.push(selection(first));
        frontiers.push(1);
    }

    while let Some((family, point, delta_rate)) = best_fitting_edge(
        hull_points,
        hull_ranges,
        &frontiers,
        byte_budget.saturating_sub(used),
    )? {
        used = used
            .checked_add(delta_rate)
            .ok_or(EncodeError::ArithmeticOverflow {
                what: "HTJ2K tile candidate selected bytes",
            })?;
        selections[family] = selection(point);
        frontiers[family] += 1;
    }
    Ok(selections)
}

fn best_fitting_edge(
    points: &[HtCandidatePoint],
    ranges: &[HtCandidateRange],
    frontiers: &[usize],
    remaining: u64,
) -> EncodeResult<Option<(usize, HtCandidatePoint, u64)>> {
    let mut best = None::<(usize, HtCandidatePoint, u64, f64)>;
    for (family, range) in ranges.iter().enumerate() {
        let frontier = frontiers[family];
        if frontier >= range.len {
            continue;
        }
        let previous = points[range.start + frontier - 1];
        let next = points[range.start + frontier];
        let delta_rate =
            next.rate
                .checked_sub(previous.rate)
                .ok_or(EncodeError::InternalInvariant {
                    what: "HTJ2K tile candidate hull rate is not monotonic",
                })?;
        if delta_rate > remaining {
            continue;
        }
        let slope = edge_slope(previous, next);
        let replace = best.is_none_or(|(best_family, best_point, _, best_slope)| {
            slope
                .total_cmp(&best_slope)
                .then_with(|| best_family.cmp(&family))
                .then_with(|| best_point.candidate_index.cmp(&next.candidate_index))
                .then_with(|| best_point.num_coding_passes.cmp(&next.num_coding_passes))
                .is_gt()
        });
        if replace {
            best = Some((family, next, delta_rate, slope));
        }
    }
    Ok(best.map(|(family, point, rate, _)| (family, point, rate)))
}

fn candidate_family(
    candidates: &[EncodedCodeBlock],
    range: HtCandidateRange,
) -> EncodeResult<&[EncodedCodeBlock]> {
    let end = range
        .start
        .checked_add(range.len)
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "HTJ2K tile candidate family range",
        })?;
    if range.len == 0 || end > candidates.len() {
        return Err(EncodeError::InternalInvariant {
            what: "HTJ2K tile candidate family range is invalid",
        });
    }
    Ok(&candidates[range.start..end])
}

fn append_candidate_hull(
    candidates: &[EncodedCodeBlock],
    candidate_offset: usize,
    output: &mut Vec<HtCandidatePoint>,
) -> EncodeResult<()> {
    let hull_start = output.len();
    let mut points = candidate_points(candidates, candidate_offset)?;
    points.sort_by(|left, right| {
        left.rate
            .cmp(&right.rate)
            .then_with(|| right.distortion.total_cmp(&left.distortion))
            .then_with(|| left.candidate_index.cmp(&right.candidate_index))
            .then_with(|| left.num_coding_passes.cmp(&right.num_coding_passes))
    });
    let nondominated = nondominated_points(points)?;
    for point in nondominated {
        while output.len() >= hull_start + 2 {
            let left = output[output.len() - 2];
            let middle = output[output.len() - 1];
            if edge_slope(left, middle).total_cmp(&edge_slope(middle, point)) == Ordering::Greater {
                break;
            }
            output.pop();
        }
        output.push(point);
    }
    Ok(())
}

fn candidate_points(
    candidates: &[EncodedCodeBlock],
    candidate_offset: usize,
) -> EncodeResult<Vec<HtCandidatePoint>> {
    let capacity = candidates
        .len()
        .checked_mul(3)
        .ok_or(EncodeError::ArithmeticOverflow {
            what: "HTJ2K block candidate point count",
        })?;
    let mut points = Vec::new();
    points
        .try_reserve_exact(capacity)
        .map_err(|_| EncodeError::HostAllocationFailed {
            what: "HTJ2K block candidate points",
            bytes: capacity.saturating_mul(core::mem::size_of::<HtCandidatePoint>()),
        })?;
    for (local_index, candidate) in candidates.iter().enumerate() {
        if candidate.num_coding_passes == 0 {
            points.push(HtCandidatePoint {
                candidate_index: candidate_offset + local_index,
                num_coding_passes: 0,
                rate: 0,
                distortion: 0.0,
            });
            continue;
        }
        let mut rate = 0u64;
        let mut distortion = 0.0;
        for pass in 0..candidate.num_coding_passes {
            rate = rate.checked_add(pass_rate(candidate, pass)).ok_or(
                EncodeError::ArithmeticOverflow {
                    what: "HTJ2K candidate cumulative rate",
                },
            )?;
            let delta = candidate.ht_distortion_deltas[usize::from(pass)];
            if !delta.is_finite() || delta < 0.0 {
                return Err(EncodeError::InternalInvariant {
                    what: "HTJ2K candidate distortion is invalid",
                });
            }
            distortion += delta;
            points.push(HtCandidatePoint {
                candidate_index: candidate_offset + local_index,
                num_coding_passes: pass + 1,
                rate,
                distortion,
            });
        }
    }
    Ok(points)
}

fn nondominated_points(points: Vec<HtCandidatePoint>) -> EncodeResult<Vec<HtCandidatePoint>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(points.len())
        .map_err(|_| EncodeError::HostAllocationFailed {
            what: "HTJ2K nondominated candidate points",
            bytes: points
                .len()
                .saturating_mul(core::mem::size_of::<HtCandidatePoint>()),
        })?;
    let mut best_distortion = f64::NEG_INFINITY;
    let mut previous_rate = None;
    for point in points {
        if previous_rate == Some(point.rate) {
            continue;
        }
        previous_rate = Some(point.rate);
        if point.distortion <= best_distortion {
            continue;
        }
        best_distortion = point.distortion;
        output.push(point);
    }
    Ok(output)
}

fn pass_rate(candidate: &EncodedCodeBlock, pass: u8) -> u64 {
    u64::from(match pass {
        0 => candidate.ht_cleanup_length,
        1 => candidate.ht_sigprop_length,
        _ => candidate.ht_magref_length,
    })
}

#[expect(
    clippy::cast_precision_loss,
    reason = "an HT point sums at most three u32 segment lengths, which is exactly representable in f64"
)]
fn edge_slope(left: HtCandidatePoint, right: HtCandidatePoint) -> f64 {
    let rate = right.rate - left.rate;
    if rate == 0 {
        f64::INFINITY
    } else {
        (right.distortion - left.distortion) / rate as f64
    }
}

fn selection(point: HtCandidatePoint) -> HtCandidateSelection {
    HtCandidateSelection {
        candidate_index: point.candidate_index,
        num_coding_passes: point.num_coding_passes,
    }
}
