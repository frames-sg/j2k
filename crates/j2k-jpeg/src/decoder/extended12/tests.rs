// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{upsample_h2v1_sample_at, upsample_h2v2_rows_at};

#[test]
fn extended12_generic_upsampling_keeps_edge_and_rounding_goldens() {
    let row_u8 = [0u8, 1, 255];
    let actual_u8 = (0..row_u8.len() * 2)
        .map(|x| upsample_h2v1_sample_at(&row_u8, x))
        .collect::<Vec<_>>();
    assert_eq!(actual_u8, [0, 0, 1, 65, 192, 255]);

    let row_u16 = [0u16, 1, 4095];
    let actual_u16 = (0..row_u16.len() * 2)
        .map(|x| upsample_h2v1_sample_at(&row_u16, x))
        .collect::<Vec<_>>();
    assert_eq!(actual_u16, [0, 0, 1, 1025, 3072, 4095]);

    let current = [100u16, 400, 900];
    let near = [300u16, 200, 700];
    let actual_h2v2 = (0..5)
        .map(|x| upsample_h2v2_rows_at(&current, &near, 5, x))
        .collect::<Vec<_>>();
    assert_eq!(actual_h2v2, [150, 200, 300, 475, 850]);
}

#[test]
fn extended12_420_writers_replicate_last_real_chroma_row_instead_of_padding() {
    use super::planes::Extended12Plane;
    use super::sampling::Extended12ColorSampling;
    use super::writers::{
        write_extended12_color420_planes_region, write_extended12_four_component_planes_region,
        Extended12Output, Extended12RgbProjection, Extended12WriteRegion,
    };
    use crate::info::{ColorSpace, DownscaleFactor, Rect};

    // 5x6 image in one 16x16 MCU: chroma rows 0..3 are real, 3..8 padding.
    let (width, height) = (5u32, 6u32);
    let real_chroma_rows = 3;
    let plane = |seed: u16, luma: bool, padding: Option<u16>| {
        let (stride, rows) = if luma { (16, 16) } else { (8, 8) };
        let mut pixels = vec![0u16; stride * rows];
        for (offset, sample) in pixels.iter_mut().enumerate() {
            let (row, col) = (offset / stride, offset % stride);
            let row = match padding {
                Some(value) if !luma && row >= real_chroma_rows => {
                    *sample = value;
                    continue;
                }
                None if !luma => row.min(real_chroma_rows - 1),
                _ => row,
            };
            let (row, col) = (u16::try_from(row).unwrap(), u16::try_from(col).unwrap());
            *sample = (seed + row * 397 + col * 211) % 4096;
        }
        Extended12Plane {
            pixels,
            stride,
            width: if luma { 5 } else { 3 },
        }
    };
    let region = Extended12WriteRegion {
        output_rect: Rect {
            x: 0,
            y: 0,
            w: width,
            h: height,
        },
        dimensions: (width, height),
        downscale: DownscaleFactor::Full,
        output: Extended12Output::Rgb16,
    };
    let stride = width as usize * 6;
    let color = |padding: Option<u16>| {
        let planes = [
            plane(100, true, None),
            plane(1_900, false, padding),
            plane(2_300, false, padding),
        ];
        let mut out = vec![0u8; stride * height as usize];
        write_extended12_color420_planes_region(
            &mut out,
            stride,
            region,
            Extended12RgbProjection::YCbCr,
            &planes,
        );
        out
    };
    let four_component = |padding: Option<u16>| {
        let planes = [
            plane(100, true, None),
            plane(1_900, false, padding),
            plane(2_300, false, padding),
            plane(700, false, padding),
        ];
        let mut out = vec![0u8; stride * height as usize];
        write_extended12_four_component_planes_region(
            &mut out,
            stride,
            region,
            ColorSpace::Ycck,
            Extended12ColorSampling::S420,
            &planes,
        );
        out
    };

    let color_replicated = color(None);
    let four_replicated = four_component(None);
    for padding in [0u16, 4095, 1234] {
        assert_eq!(color(Some(padding)), color_replicated, "padding {padding}");
        assert_eq!(
            four_component(Some(padding)),
            four_replicated,
            "padding {padding}"
        );
    }
}
