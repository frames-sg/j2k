// SPDX-License-Identifier: MIT OR Apache-2.0

use core::mem::size_of;

use super::roi_fixture;
use crate::{DecodeSettings, DecoderContext, Image};

const SIDE: usize = 64;
const COMPONENTS: usize = 3;

fn decode_context(bytes: &[u8], roi: Option<(u32, u32, u32, u32)>) -> DecoderContext<'_> {
    let image = Image::new(bytes, &DecodeSettings::default()).expect("image");
    let mut context = DecoderContext::default();
    match roi {
        Some(roi) => {
            image
                .decode_region_with_context(roi, &mut context)
                .expect("region decode");
        }
        None => {
            image
                .decode_with_context(&mut context)
                .expect("whole decode");
        }
    }
    context
}

macro_rules! assert_total_is_the_sum {
    ($peak:expr) => {
        assert_eq!(
            $peak.total(),
            $peak.coefficients + $peak.channels + $peak.idwt + $peak.tier1 + $peak.structure
        );
    };
}

#[test]
fn whole_decode_peak_holds_every_component_plane_and_one_component_of_coefficients() {
    let bytes = roi_fixture(true, 3);
    let context = decode_context(&bytes, None);
    let counters = &context.tile_decode_context.debug_counters;
    let peak = counters.workspace_peak.bytes();
    assert_total_is_the_sum!(peak);

    // Each component finishes Tier-1, the IDWT and store before the next is
    // decoded, so coefficients for only one component are live. Every
    // component plane is, until the color transform.
    assert_eq!(counters.stored_coefficients, SIDE * SIDE);
    assert_eq!(peak.coefficients, SIDE * SIDE * size_of::<f32>());
    assert!(peak.channels >= COMPONENTS * SIDE * SIDE * size_of::<f32>());
    assert!(peak.idwt >= SIDE * SIDE * size_of::<f32>());
    assert!(peak.tier1 > 0);
    assert!(peak.structure > 0);
}

#[test]
fn region_decode_peak_includes_the_windowed_idwt_buffers() {
    let bytes = roi_fixture(true, 3);
    let (width, height) = (17, 19);
    let whole = decode_context(&bytes, None)
        .tile_decode_context
        .debug_counters
        .workspace_peak
        .bytes();
    let context = decode_context(&bytes, Some((20, 18, width, height)));
    let counters = &context.tile_decode_context.debug_counters;
    let peak = counters.workspace_peak.bytes();
    assert_total_is_the_sum!(peak);

    let region_samples = (width * height) as usize;
    assert!(peak.channels >= COMPONENTS * region_samples * size_of::<f32>());
    assert!(peak.channels < whole.channels);
    assert!(peak.coefficients >= counters.stored_coefficients * size_of::<f32>());
    assert!(peak.coefficients < whole.coefficients);
    // The windowed IDWT keeps its level buffers outside the decoder context.
    // The last level's window covers at least the region, and the peak is
    // sampled while it and the level below are both live.
    assert!(peak.idwt > region_samples * size_of::<f32>());
    assert!(peak.total() < whole.total());
}
