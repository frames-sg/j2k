// SPDX-License-Identifier: Apache-2.0

pub(crate) fn validate_dct_block_grid(
    block_count: usize,
    block_cols: usize,
    block_rows: usize,
    width: usize,
    height: usize,
) -> Result<(), ()> {
    let expected_blocks = block_cols.checked_mul(block_rows).ok_or(())?;
    let covered_width = block_cols.checked_mul(8).ok_or(())?;
    let covered_height = block_rows.checked_mul(8).ok_or(())?;

    if block_count != expected_blocks
        || width == 0
        || height == 0
        || width > covered_width
        || height > covered_height
    {
        return Err(());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_dct_block_grid;

    #[test]
    fn validates_non_empty_grid_covered_by_blocks() {
        assert_eq!(validate_dct_block_grid(6, 3, 2, 24, 16), Ok(()));
    }

    #[test]
    fn rejects_overflowing_grid_dimensions() {
        assert_eq!(
            validate_dct_block_grid(1, usize::MAX, usize::MAX, 8, 8),
            Err(())
        );
    }

    #[test]
    fn rejects_zero_image_extent() {
        assert_eq!(validate_dct_block_grid(1, 1, 1, 0, 8), Err(()));
        assert_eq!(validate_dct_block_grid(1, 1, 1, 8, 0), Err(()));
    }
}
