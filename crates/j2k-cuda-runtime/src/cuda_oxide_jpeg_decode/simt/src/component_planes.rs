use super::{
    load_u8, min_u32, store_u8, ycbcr_to_rgb, J2kJpeg420Params, Rgb420McuBlocks,
    Rgb422McuBlocks,
};

pub(super) const SAMPLING_420: u32 = 0;
pub(super) const SAMPLING_422: u32 = 1;

#[derive(Clone, Copy)]
struct ComponentPlaneLayout {
    chroma_width: u32,
    chroma_height: u32,
    cb_offset: u32,
    cr_offset: u32,
}

#[inline(always)]
fn component_plane_layout(params: J2kJpeg420Params, sampling: u32) -> ComponentPlaneLayout {
    let chroma_width = params.width / 2 + params.width % 2;
    let chroma_height = if sampling == SAMPLING_420 {
        params.height / 2 + params.height % 2
    } else {
        params.height
    };
    let cb_offset = params.width * params.height;
    let cr_offset = cb_offset + chroma_width * chroma_height;
    ComponentPlaneLayout {
        chroma_width,
        chroma_height,
        cb_offset,
        cr_offset,
    }
}

#[inline(always)]
fn store_block(
    output: *mut u8,
    plane_offset: u32,
    plane_stride: u32,
    plane_width: u32,
    plane_height: u32,
    base_x: u32,
    base_y: u32,
    block: &[u8; 64],
) {
    let mut y = 0;
    while y < 8 {
        let py = base_y + y;
        if py < plane_height {
            let mut x = 0;
            while x < 8 {
                let px = base_x + x;
                if px < plane_width {
                    store_u8(
                        output,
                        plane_offset + py * plane_stride + px,
                        block[(y * 8 + x) as usize],
                    );
                }
                x += 1;
            }
        }
        y += 1;
    }
}

#[inline(always)]
pub(super) fn store_420_mcu(
    output: *mut u8,
    params: J2kJpeg420Params,
    mx: u32,
    my: u32,
    blocks: Rgb420McuBlocks<'_>,
) {
    let base_x = mx * 16;
    let base_y = my * 16;
    store_block(output, 0, params.width, params.width, params.height, base_x, base_y, blocks.y0);
    store_block(
        output,
        0,
        params.width,
        params.width,
        params.height,
        base_x + 8,
        base_y,
        blocks.y1,
    );
    store_block(
        output,
        0,
        params.width,
        params.width,
        params.height,
        base_x,
        base_y + 8,
        blocks.y2,
    );
    store_block(
        output,
        0,
        params.width,
        params.width,
        params.height,
        base_x + 8,
        base_y + 8,
        blocks.y3,
    );

    let layout = component_plane_layout(params, SAMPLING_420);
    let chroma_x = mx * 8;
    let chroma_y = my * 8;
    store_block(
        output,
        layout.cb_offset,
        layout.chroma_width,
        layout.chroma_width,
        layout.chroma_height,
        chroma_x,
        chroma_y,
        blocks.cb,
    );
    store_block(
        output,
        layout.cr_offset,
        layout.chroma_width,
        layout.chroma_width,
        layout.chroma_height,
        chroma_x,
        chroma_y,
        blocks.cr,
    );
}

#[inline(always)]
pub(super) fn store_422_mcu(
    output: *mut u8,
    params: J2kJpeg420Params,
    mx: u32,
    my: u32,
    blocks: Rgb422McuBlocks<'_>,
) {
    let base_x = mx * 16;
    let base_y = my * 8;
    store_block(output, 0, params.width, params.width, params.height, base_x, base_y, blocks.y0);
    store_block(
        output,
        0,
        params.width,
        params.width,
        params.height,
        base_x + 8,
        base_y,
        blocks.y1,
    );

    let layout = component_plane_layout(params, SAMPLING_422);
    let chroma_x = mx * 8;
    let chroma_y = my * 8;
    store_block(
        output,
        layout.cb_offset,
        layout.chroma_width,
        layout.chroma_width,
        layout.chroma_height,
        chroma_x,
        chroma_y,
        blocks.cb,
    );
    store_block(
        output,
        layout.cr_offset,
        layout.chroma_width,
        layout.chroma_width,
        layout.chroma_height,
        chroma_x,
        chroma_y,
        blocks.cr,
    );
}

#[inline(always)]
fn h2v2_sample(
    planes: *const u8,
    plane_offset: u32,
    chroma_width: u32,
    chroma_height: u32,
    output_x: u32,
    output_y: u32,
) -> u8 {
    let chroma_y = min_u32(output_y / 2, chroma_height - 1);
    let near_y = if (output_y & 1) == 0 {
        if chroma_y == 0 { 0 } else { chroma_y - 1 }
    } else {
        min_u32(chroma_y + 1, chroma_height - 1)
    };
    let sample = min_u32(output_x / 2, chroma_width - 1);
    let current = load_u8(planes, plane_offset + chroma_y * chroma_width + sample) as u32;
    let near = load_u8(planes, plane_offset + near_y * chroma_width + sample) as u32;
    let current_sum = 3 * current + near;
    if chroma_width == 1 || output_x == 0 {
        return ((4 * current_sum + 8) >> 4) as u8;
    }
    if output_x == chroma_width * 2 - 1 {
        return ((4 * current_sum + 7) >> 4) as u8;
    }
    if (output_x & 1) == 0 {
        let previous = sample - 1;
        let previous_current =
            load_u8(planes, plane_offset + chroma_y * chroma_width + previous) as u32;
        let previous_near =
            load_u8(planes, plane_offset + near_y * chroma_width + previous) as u32;
        return ((3 * current_sum + 3 * previous_current + previous_near + 8) >> 4) as u8;
    }
    let next = min_u32(sample + 1, chroma_width - 1);
    let next_current =
        load_u8(planes, plane_offset + chroma_y * chroma_width + next) as u32;
    let next_near = load_u8(planes, plane_offset + near_y * chroma_width + next) as u32;
    ((3 * current_sum + 3 * next_current + next_near + 7) >> 4) as u8
}

#[inline(always)]
fn h2v1_sample(
    planes: *const u8,
    plane_offset: u32,
    chroma_width: u32,
    output_x: u32,
    output_y: u32,
) -> u8 {
    let row = plane_offset + output_y * chroma_width;
    if chroma_width == 1 || output_x == 0 {
        return load_u8(planes, row);
    }
    if output_x == chroma_width * 2 - 1 {
        return load_u8(planes, row + chroma_width - 1);
    }
    let sample = min_u32(output_x / 2, chroma_width - 1);
    let current = load_u8(planes, row + sample) as u32;
    if (output_x & 1) == 0 {
        let previous = load_u8(planes, row + sample - 1) as u32;
        return ((3 * current + previous + 2) >> 2) as u8;
    }
    let next = load_u8(planes, row + min_u32(sample + 1, chroma_width - 1)) as u32;
    ((3 * current + next + 2) >> 2) as u8
}

#[inline(always)]
pub(super) fn convert_pixel(
    planes: *const u8,
    output: *mut u8,
    params: J2kJpeg420Params,
    sampling: u32,
    pixel: u32,
) {
    let x = pixel - (pixel / params.width) * params.width;
    let y = pixel / params.width;
    let layout = component_plane_layout(params, sampling);
    let luma = load_u8(planes, pixel);
    let (cb, cr) = if sampling == SAMPLING_420 {
        (
            h2v2_sample(
                planes,
                layout.cb_offset,
                layout.chroma_width,
                layout.chroma_height,
                x,
                y,
            ),
            h2v2_sample(
                planes,
                layout.cr_offset,
                layout.chroma_width,
                layout.chroma_height,
                x,
                y,
            ),
        )
    } else {
        (
            h2v1_sample(planes, layout.cb_offset, layout.chroma_width, x, y),
            h2v1_sample(planes, layout.cr_offset, layout.chroma_width, x, y),
        )
    };
    let mut r = 0;
    let mut g = 0;
    let mut b = 0;
    ycbcr_to_rgb(luma, cb, cr, &mut r, &mut g, &mut b);
    let destination = y * params.out_stride + x * 3;
    store_u8(output, destination, r);
    store_u8(output, destination + 1, g);
    store_u8(output, destination + 2, b);
}
