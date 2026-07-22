// SPDX-License-Identifier: MIT OR Apache-2.0

pub(super) fn cpu_forward_dwt53_buffer(
    samples: &[f32],
    width: usize,
    height: usize,
    levels: u8,
) -> Vec<f32> {
    let mut buffer = samples.to_vec();
    let mut current_width = width;
    let mut current_height = height;

    for _ in 0..levels {
        if current_width < 2 && current_height < 2 {
            break;
        }
        if current_height >= 2 {
            let low_height = current_height.div_ceil(2);
            let mut col = vec![0.0; current_height];
            for x in 0..current_width {
                for y in 0..current_height {
                    col[y] = buffer[y * width + x];
                }
                forward_lift_53(&mut col);
                for y in 0..low_height {
                    buffer[y * width + x] = col[y * 2];
                }
                for y in 0..current_height / 2 {
                    buffer[(low_height + y) * width + x] = col[y * 2 + 1];
                }
            }
        }
        if current_width >= 2 {
            let mut row = vec![0.0; current_width];
            for y in 0..current_height {
                let row_start = y * width;
                row.copy_from_slice(&buffer[row_start..row_start + current_width]);
                forward_lift_53(&mut row);
                let low_width = current_width.div_ceil(2);
                for x in 0..low_width {
                    buffer[row_start + x] = row[x * 2];
                }
                for x in 0..current_width / 2 {
                    buffer[row_start + low_width + x] = row[x * 2 + 1];
                }
            }
        }
        current_width = current_width.div_ceil(2);
        current_height = current_height.div_ceil(2);
    }

    buffer
}

pub(super) fn cpu_forward_dwt97_buffer(
    samples: &[f32],
    width: usize,
    height: usize,
    levels: u8,
) -> Vec<f32> {
    let mut buffer = samples.to_vec();
    let mut current_width = width;
    let mut current_height = height;

    for _ in 0..levels {
        if current_width < 2 && current_height < 2 {
            break;
        }
        if current_height >= 2 {
            let low_height = current_height.div_ceil(2);
            let mut col = vec![0.0; current_height];
            for x in 0..current_width {
                for y in 0..current_height {
                    col[y] = buffer[y * width + x];
                }
                forward_lift_97(&mut col);
                for y in 0..low_height {
                    buffer[y * width + x] = col[y * 2];
                }
                for y in 0..current_height / 2 {
                    buffer[(low_height + y) * width + x] = col[y * 2 + 1];
                }
            }
        }
        if current_width >= 2 {
            let mut row = vec![0.0; current_width];
            for y in 0..current_height {
                let row_start = y * width;
                row.copy_from_slice(&buffer[row_start..row_start + current_width]);
                forward_lift_97(&mut row);
                let low_width = current_width.div_ceil(2);
                for x in 0..low_width {
                    buffer[row_start + x] = row[x * 2];
                }
                for x in 0..current_width / 2 {
                    buffer[row_start + low_width + x] = row[x * 2 + 1];
                }
            }
        }
        current_width = current_width.div_ceil(2);
        current_height = current_height.div_ceil(2);
    }

    buffer
}

fn forward_lift_53(data: &mut [f32]) {
    let n = data.len();
    if n < 2 {
        return;
    }

    let last_even = if n.is_multiple_of(2) { n - 2 } else { n - 1 };
    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n {
            data[i + 1]
        } else {
            data[last_even]
        };
        data[i] -= ((left + right) * 0.5).floor();
    }

    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += ((left + right) * 0.25 + 0.5).floor();
    }
}

fn forward_lift_97(data: &mut [f32]) {
    const ALPHA: f32 = -1.586_134_3;
    const BETA: f32 = -0.052_980_117;
    const GAMMA: f32 = 0.882_911_1;
    const DELTA: f32 = 0.443_506_87;
    const KAPPA: f32 = 1.230_174_1;
    const INV_KAPPA: f32 = 1.0 / KAPPA;

    let n = data.len();
    if n < 2 {
        return;
    }

    let last_even = if n.is_multiple_of(2) { n - 2 } else { n - 1 };
    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n {
            data[i + 1]
        } else {
            data[last_even]
        };
        data[i] += ALPHA * (left + right);
    }
    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += BETA * (left + right);
    }
    for i in (1..n).step_by(2) {
        let left = data[i - 1];
        let right = if i + 1 < n {
            data[i + 1]
        } else {
            data[last_even]
        };
        data[i] += GAMMA * (left + right);
    }
    for i in (0..n).step_by(2) {
        let left = if i > 0 { data[i - 1] } else { data[1] };
        let right = if i + 1 < n { data[i + 1] } else { left };
        data[i] += DELTA * (left + right);
    }
    for i in (0..n).step_by(2) {
        data[i] *= INV_KAPPA;
    }
    for i in (1..n).step_by(2) {
        data[i] *= KAPPA;
    }
}
