// SPDX-License-Identifier: MIT OR Apache-2.0

pub(super) fn max_sample_delta(left: &[u8], right: &[u8], bit_depth: u8) -> Result<u32, String> {
    if left.len() != right.len() {
        return Err("decoded output lengths differ".to_string());
    }
    if bit_depth <= 8 {
        return Ok(left
            .iter()
            .zip(right)
            .map(|(&a, &b)| u32::from(a.abs_diff(b)))
            .max()
            .unwrap_or_default());
    }
    if !left.len().is_multiple_of(2) {
        return Err("16-bit decoded output has an odd byte length".to_string());
    }
    Ok(left
        .chunks_exact(2)
        .zip(right.chunks_exact(2))
        .map(|(a, b)| {
            u32::from(u16::from_le_bytes([a[0], a[1]]).abs_diff(u16::from_le_bytes([b[0], b[1]])))
        })
        .max()
        .unwrap_or_default())
}

pub(super) fn psnr(reference: &[u8], actual: &[u8], bit_depth: u8) -> Result<f64, String> {
    if reference.len() != actual.len() {
        return Err("PSNR input lengths differ".to_string());
    }
    let (sum_squared_error, sample_count) = if bit_depth <= 8 {
        let error = reference
            .iter()
            .zip(actual)
            .map(|(&a, &b)| {
                let delta = f64::from(a) - f64::from(b);
                delta * delta
            })
            .sum::<f64>();
        (error, reference.len())
    } else {
        if !reference.len().is_multiple_of(2) {
            return Err("16-bit PSNR input has an odd byte length".to_string());
        }
        let error = reference
            .chunks_exact(2)
            .zip(actual.chunks_exact(2))
            .map(|(a, b)| {
                let delta = f64::from(u16::from_le_bytes([a[0], a[1]]))
                    - f64::from(u16::from_le_bytes([b[0], b[1]]));
                delta * delta
            })
            .sum::<f64>();
        (error, reference.len() / 2)
    };
    if sample_count == 0 {
        return Err("PSNR inputs are empty".to_string());
    }
    if sum_squared_error == 0.0 {
        return Ok(f64::INFINITY);
    }
    let peak = f64::from((1_u32 << bit_depth) - 1);
    let sample_count =
        u32::try_from(sample_count).map_err(|_| "PSNR sample count exceeds u32".to_string())?;
    let mse = sum_squared_error / f64::from(sample_count);
    Ok(10.0 * (peak * peak / mse).log10())
}

#[cfg(test)]
mod tests {
    use super::{max_sample_delta, psnr};

    #[test]
    fn parity_metrics_are_sample_based_for_eight_and_sixteen_bit_data() {
        assert_eq!(max_sample_delta(&[0, 2, 255], &[1, 2, 253], 8), Ok(2));
        let left = [0_u16, 1024, 65_535]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let right = [1_u16, 1022, 65_535]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        assert_eq!(max_sample_delta(&left, &right, 16), Ok(2));
        assert!(psnr(&left, &left, 16).is_ok_and(f64::is_infinite));
    }
}
