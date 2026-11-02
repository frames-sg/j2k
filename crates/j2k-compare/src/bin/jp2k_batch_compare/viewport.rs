// SPDX-License-Identifier: MIT OR Apache-2.0

//! Viewer-style region requests over one large image.
//!
//! `J2K_BATCH_COMPARE_VIEWPORT=<reduce>:<count>:<size>:<seed>` turns the single
//! input of a run into `count` requests, each a `size` x `size` output window
//! (in output pixels) at resolution reduction `reduce`, so the source region
//! is `size << reduce` pixels on a side. Positions are drawn with a seeded
//! generator, so every decoder and every host gets the same requests; they are
//! not aligned to tiles, precincts or code blocks.

use j2k_core::Rect;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ViewportSpec {
    pub(crate) reduce: u8,
    pub(crate) count: usize,
    pub(crate) size: u32,
    pub(crate) seed: u64,
}

impl ViewportSpec {
    pub(crate) fn parse(value: &str) -> Result<Self, String> {
        let parts = value.split(':').map(str::trim).collect::<Vec<_>>();
        let [reduce, count, size, seed] = parts.as_slice() else {
            return Err(format!(
                "J2K_BATCH_COMPARE_VIEWPORT {value:?}: expected <reduce>:<count>:<size>:<seed>"
            ));
        };
        let number = |text: &str, what: &str| {
            text.parse::<u64>()
                .map_err(|err| format!("J2K_BATCH_COMPARE_VIEWPORT {what} {text:?}: {err}"))
        };
        let reduce = u8::try_from(number(reduce, "reduce")?)
            .ok()
            .filter(|&r| r <= 6)
            .ok_or("J2K_BATCH_COMPARE_VIEWPORT reduce must be 0..=6")?;
        let count = usize::try_from(number(count, "count")?)
            .ok()
            .filter(|&c| c > 0)
            .ok_or("J2K_BATCH_COMPARE_VIEWPORT count must be positive")?;
        let size = u32::try_from(number(size, "size")?)
            .ok()
            .filter(|&s| s > 0)
            .ok_or("J2K_BATCH_COMPARE_VIEWPORT size must be a positive u32")?;
        Ok(Self {
            reduce,
            count,
            size,
            seed: number(seed, "seed")?,
        })
    }

    /// The source-coordinate regions. Like a viewer at zoom level `reduce`,
    /// each window is placed on that level's pixel grid and scaled up, so its
    /// origin is a multiple of `2^reduce`: every decoder's rounding convention
    /// for reduced coordinates then gives the same output rectangle. Windows
    /// are clipped to the image.
    pub(crate) fn requests(self, dimensions: (u32, u32)) -> Vec<Rect> {
        let scale = 1_u32 << self.reduce;
        let level = (dimensions.0.div_ceil(scale), dimensions.1.div_ceil(scale));
        let (w, h) = (self.size.min(level.0), self.size.min(level.1));
        let mut state = self.seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut next = |span: u32| {
            // xorshift64*: deterministic across platforms.
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let value = state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32;
            u32::try_from(value % u64::from(span)).unwrap_or(0)
        };
        (0..self.count)
            .map(|_| {
                let x = next(level.0 - w + 1) * scale;
                let y = next(level.1 - h + 1) * scale;
                Rect {
                    x,
                    y,
                    w: (w * scale).min(dimensions.0 - x),
                    h: (h * scale).min(dimensions.1 - y),
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::ViewportSpec;

    #[test]
    fn parses_a_spec_and_rejects_reductions_beyond_six_levels() {
        assert_eq!(
            ViewportSpec::parse("2:16:1024:7"),
            Ok(ViewportSpec {
                reduce: 2,
                count: 16,
                size: 1024,
                seed: 7
            })
        );
        assert!(ViewportSpec::parse("6:16:256:7").is_ok());
        assert!(ViewportSpec::parse("7:16:256:7").is_err());
        assert!(ViewportSpec::parse("0:0:1024:7").is_err());
        assert!(ViewportSpec::parse("0:16:1024").is_err());
    }

    #[test]
    fn requests_are_deterministic_and_inside_the_image() {
        let spec = ViewportSpec::parse("1:32:1024:3").expect("spec");
        let first = spec.requests((16_384, 16_384));
        assert_eq!(first, spec.requests((16_384, 16_384)));
        assert_eq!(first.len(), 32);
        for rect in &first {
            assert_eq!((rect.w, rect.h), (2048, 2048));
            assert_eq!(
                (rect.x % 2, rect.y % 2),
                (0, 0),
                "origin on the level-1 grid"
            );
            assert!(rect.x + rect.w <= 16_384 && rect.y + rect.h <= 16_384);
        }
        assert!(first.windows(2).any(|pair| pair[0] != pair[1]));
    }

    #[test]
    fn requests_larger_than_the_image_cover_it_exactly() {
        let spec = ViewportSpec::parse("3:2:1024:1").expect("spec");
        for rect in spec.requests((4000, 3000)) {
            assert_eq!((rect.x, rect.y, rect.w, rect.h), (0, 0, 4000, 3000));
        }
    }
}
