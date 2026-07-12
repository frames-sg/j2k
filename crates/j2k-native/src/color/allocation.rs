// SPDX-License-Identifier: MIT OR Apache-2.0

//! Retained allocation accounting for owned native decode outputs.

use super::RawBitmap;

fn bit_capacity_bytes(bits: usize) -> Option<usize> {
    bits.checked_add(7).map(|rounded| rounded / 8)
}

impl RawBitmap {
    /// Return the actual heap capacity retained by this owned result.
    #[must_use]
    pub(crate) fn allocated_bytes(&self) -> Option<usize> {
        self.data
            .capacity()
            .checked_add(bit_capacity_bytes(self.component_signed.capacity())?)
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn packed_bitmap_counts_bit_vector_capacity_in_bytes() {
        let mut data = Vec::new();
        data.try_reserve_exact(3).expect("small data allocation");
        let mut component_signed = Vec::new();
        component_signed
            .try_reserve_exact(65)
            .expect("small bit-vector allocation");
        let bitmap = RawBitmap {
            data,
            width: 1,
            height: 1,
            bit_depth: 8,
            signed: false,
            component_signed,
            num_components: 1,
            bytes_per_sample: 1,
        };
        assert_eq!(
            bitmap.allocated_bytes(),
            Some(bitmap.data.capacity() + bitmap.component_signed.capacity().div_ceil(8))
        );
    }
}
