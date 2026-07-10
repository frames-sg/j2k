use crate::{idwt_band_index, J2kDirectIdwtStep, J2kRect, J2kWaveletTransform};

/// Required region of one direct-plan coefficient band.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct J2kRequiredBandRegion {
    /// Inclusive minimum x coordinate.
    pub x0: u32,
    /// Inclusive minimum y coordinate.
    pub y0: u32,
    /// Exclusive maximum x coordinate.
    pub x1: u32,
    /// Exclusive maximum y coordinate.
    pub y1: u32,
}

impl J2kRequiredBandRegion {
    /// Construct a non-empty required region.
    #[must_use]
    pub fn new(x0: u32, y0: u32, x1: u32, y1: u32) -> Option<Self> {
        (x0 < x1 && y0 < y1).then_some(Self { x0, y0, x1, y1 })
    }

    /// Construct a region covering a band from origin.
    #[must_use]
    pub const fn full(width: u32, height: u32) -> Self {
        Self {
            x0: 0,
            y0: 0,
            x1: width,
            y1: height,
        }
    }

    /// Region width.
    #[must_use]
    pub fn width(self) -> u32 {
        self.x1.saturating_sub(self.x0)
    }

    /// Region height.
    #[must_use]
    pub fn height(self) -> u32 {
        self.y1.saturating_sub(self.y0)
    }

    /// Expand this region within a band of `width` by `height`.
    #[must_use]
    pub fn expanded_within_band(self, margin: u32, width: u32, height: u32) -> Self {
        Self {
            x0: self.x0.saturating_sub(margin),
            y0: self.y0.saturating_sub(margin),
            x1: self.x1.saturating_add(margin).min(width),
            y1: self.y1.saturating_add(margin).min(height),
        }
    }

    /// Expand this region within an absolute rectangle.
    #[must_use]
    pub fn expanded_within_rect(self, margin: u32, rect: J2kRect) -> Self {
        Self {
            x0: self.x0.saturating_sub(margin).max(rect.x0),
            y0: self.y0.saturating_sub(margin).max(rect.y0),
            x1: self.x1.saturating_add(margin).min(rect.x1),
            y1: self.y1.saturating_add(margin).min(rect.y1),
        }
    }

    /// Merge two required regions.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            x0: self.x0.min(other.x0),
            y0: self.y0.min(other.y0),
            x1: self.x1.max(other.x1),
            y1: self.y1.max(other.y1),
        }
    }

    /// Return whether this region intersects a rectangle at origin with size.
    #[must_use]
    pub fn intersects(self, x0: u32, y0: u32, width: u32, height: u32) -> bool {
        let x1 = x0.saturating_add(width);
        let y1 = y0.saturating_add(height);
        self.x0 < x1 && x0 < self.x1 && self.y0 < y1 && y0 < self.y1
    }

    /// Convert this region to the public adapter rectangle type.
    #[must_use]
    pub const fn to_rect(self) -> J2kRect {
        J2kRect {
            x0: self.x0,
            y0: self.y0,
            x1: self.x1,
            y1: self.y1,
        }
    }
}

impl From<J2kRequiredBandRegion> for J2kRect {
    fn from(region: J2kRequiredBandRegion) -> Self {
        region.to_rect()
    }
}

/// Required input windows for one direct-plan IDWT step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct J2kIdwtRequiredInputWindows {
    /// Required LL input window.
    pub ll: J2kRequiredBandRegion,
    /// Required HL input window.
    pub hl: J2kRequiredBandRegion,
    /// Required LH input window.
    pub lh: J2kRequiredBandRegion,
    /// Required HH input window.
    pub hh: J2kRequiredBandRegion,
}

/// Return the conservative output margin used when back-propagating IDWT ROI windows.
#[must_use]
pub const fn idwt_required_output_margin(transform: J2kWaveletTransform) -> u32 {
    match transform {
        J2kWaveletTransform::Reversible53 => 16,
        J2kWaveletTransform::Irreversible97 => 40,
    }
}

/// Compute required LL/HL/LH/HH input windows for one direct-plan IDWT step.
#[must_use]
pub fn idwt_required_input_windows(
    idwt: &J2kDirectIdwtStep,
    output_region: J2kRequiredBandRegion,
) -> J2kIdwtRequiredInputWindows {
    J2kIdwtRequiredInputWindows {
        ll: idwt_input_required_region(
            output_region,
            idwt.rect.x0,
            idwt.rect.y0,
            true,
            true,
            idwt.ll.width(),
            idwt.ll.height(),
        ),
        hl: idwt_input_required_region(
            output_region,
            idwt.rect.x0,
            idwt.rect.y0,
            false,
            true,
            idwt.hl.width(),
            idwt.hl.height(),
        ),
        lh: idwt_input_required_region(
            output_region,
            idwt.rect.x0,
            idwt.rect.y0,
            true,
            false,
            idwt.lh.width(),
            idwt.lh.height(),
        ),
        hh: idwt_input_required_region(
            output_region,
            idwt.rect.x0,
            idwt.rect.y0,
            false,
            false,
            idwt.hh.width(),
            idwt.hh.height(),
        ),
    }
}

/// Compute one absolute sub-band window for native ROI planning.
#[must_use]
pub fn idwt_required_input_window_for_rects(
    output_window: J2kRequiredBandRegion,
    output_rect: J2kRect,
    band_rect: J2kRect,
    low_x: bool,
    low_y: bool,
) -> J2kRequiredBandRegion {
    if output_window.width() == 0 || output_window.height() == 0 {
        return J2kRequiredBandRegion::full(0, 0);
    }

    let x0 = band_rect.x0.saturating_add(idwt_band_index(
        output_rect.x0,
        output_window.x0.saturating_sub(output_rect.x0),
        low_x,
    ));
    let x1 = band_rect.x0.saturating_add(
        idwt_band_index(
            output_rect.x0,
            output_window
                .x1
                .saturating_sub(1)
                .saturating_sub(output_rect.x0),
            low_x,
        )
        .saturating_add(1),
    );
    let y0 = band_rect.y0.saturating_add(idwt_band_index(
        output_rect.y0,
        output_window.y0.saturating_sub(output_rect.y0),
        low_y,
    ));
    let y1 = band_rect.y0.saturating_add(
        idwt_band_index(
            output_rect.y0,
            output_window
                .y1
                .saturating_sub(1)
                .saturating_sub(output_rect.y0),
            low_y,
        )
        .saturating_add(1),
    );

    J2kRequiredBandRegion {
        x0: x0.min(band_rect.x1),
        y0: y0.min(band_rect.y1),
        x1: x1.min(band_rect.x1),
        y1: y1.min(band_rect.y1),
    }
}

fn idwt_input_required_region(
    output_region: J2kRequiredBandRegion,
    output_origin_x: u32,
    output_origin_y: u32,
    low_x: bool,
    low_y: bool,
    band_width: u32,
    band_height: u32,
) -> J2kRequiredBandRegion {
    let x0 = idwt_band_index(output_origin_x, output_region.x0, low_x);
    let x1 = idwt_band_index(output_origin_x, output_region.x1 - 1, low_x).saturating_add(1);
    let y0 = idwt_band_index(output_origin_y, output_region.y0, low_y);
    let y1 = idwt_band_index(output_origin_y, output_region.y1 - 1, low_y).saturating_add(1);
    J2kRequiredBandRegion {
        x0: x0.min(band_width),
        y0: y0.min(band_height),
        x1: x1.min(band_width),
        y1: y1.min(band_height),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn j2k_rect(x0: u32, y0: u32, x1: u32, y1: u32) -> J2kRect {
        J2kRect { x0, y0, x1, y1 }
    }

    fn idwt_step(bounds: J2kRect) -> J2kDirectIdwtStep {
        J2kDirectIdwtStep {
            output_band_id: 10,
            rect: bounds,
            transform: J2kWaveletTransform::Reversible53,
            ll_band_id: 1,
            ll: j2k_rect(
                0,
                0,
                bounds.width().div_ceil(2),
                bounds.height().div_ceil(2),
            ),
            hl_band_id: 2,
            hl: j2k_rect(0, 0, bounds.width() / 2, bounds.height().div_ceil(2)),
            lh_band_id: 3,
            lh: j2k_rect(0, 0, bounds.width().div_ceil(2), bounds.height() / 2),
            hh_band_id: 4,
            hh: j2k_rect(0, 0, bounds.width() / 2, bounds.height() / 2),
        }
    }

    #[test]
    fn idwt_required_output_margin_matches_transform_support() {
        assert_eq!(
            idwt_required_output_margin(J2kWaveletTransform::Reversible53),
            16
        );
        assert_eq!(
            idwt_required_output_margin(J2kWaveletTransform::Irreversible97),
            40
        );
    }

    #[test]
    fn idwt_required_input_windows_cover_odd_dimensions() {
        let step = idwt_step(j2k_rect(1, 1, 8, 6));
        let windows = idwt_required_input_windows(
            &step,
            J2kRequiredBandRegion {
                x0: 2,
                y0: 1,
                x1: 7,
                y1: 5,
            },
        );

        assert_eq!(
            windows.ll,
            J2kRequiredBandRegion {
                x0: 1,
                y0: 0,
                x1: 4,
                y1: 3,
            }
        );
        assert_eq!(
            windows.hl,
            J2kRequiredBandRegion {
                x0: 1,
                y0: 0,
                x1: 3,
                y1: 3,
            }
        );
        assert_eq!(
            windows.lh,
            J2kRequiredBandRegion {
                x0: 1,
                y0: 1,
                x1: 4,
                y1: 2,
            }
        );
        assert_eq!(
            windows.hh,
            J2kRequiredBandRegion {
                x0: 1,
                y0: 1,
                x1: 3,
                y1: 2,
            }
        );
    }

    #[test]
    fn idwt_required_input_windows_clamp_boundary_touching_regions() {
        let step = idwt_step(j2k_rect(0, 0, 9, 7));
        let windows = idwt_required_input_windows(
            &step,
            J2kRequiredBandRegion {
                x0: 7,
                y0: 5,
                x1: 9,
                y1: 7,
            },
        );

        assert_eq!(
            windows.ll,
            J2kRequiredBandRegion {
                x0: 4,
                y0: 3,
                x1: 5,
                y1: 4,
            }
        );
        assert_eq!(
            windows.hh,
            J2kRequiredBandRegion {
                x0: 3,
                y0: 2,
                x1: 4,
                y1: 3,
            }
        );
    }

    #[test]
    fn idwt_required_input_windows_cover_full_frame() {
        let step = idwt_step(j2k_rect(0, 0, 8, 8));
        let windows = idwt_required_input_windows(&step, J2kRequiredBandRegion::full(8, 8));

        assert_eq!(windows.ll, J2kRequiredBandRegion::full(4, 4));
        assert_eq!(windows.hl, J2kRequiredBandRegion::full(4, 4));
        assert_eq!(windows.lh, J2kRequiredBandRegion::full(4, 4));
        assert_eq!(windows.hh, J2kRequiredBandRegion::full(4, 4));
    }

    #[test]
    fn idwt_required_input_window_for_rects_preserves_absolute_offsets() {
        let window = idwt_required_input_window_for_rects(
            J2kRequiredBandRegion {
                x0: 12,
                y0: 22,
                x1: 16,
                y1: 27,
            },
            j2k_rect(10, 20, 20, 30),
            j2k_rect(5, 7, 10, 12),
            true,
            false,
        );

        assert_eq!(
            window,
            J2kRequiredBandRegion {
                x0: 6,
                y0: 8,
                x1: 9,
                y1: 11,
            }
        );
    }
}
