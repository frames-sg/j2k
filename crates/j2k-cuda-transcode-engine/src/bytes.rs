// SPDX-License-Identifier: MIT OR Apache-2.0

pub(crate) fn i16_slice_as_bytes(values: &[i16]) -> &[u8] {
    // SAFETY: every initialized i16 has a valid byte representation and the
    // returned slice cannot outlive the source slice.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

pub(crate) fn i32_slice_as_bytes_mut(values: &mut [i32]) -> &mut [u8] {
    // SAFETY: every byte pattern is a valid initialized i32 representation and
    // the mutable byte slice has the same unique borrow and lifetime.
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}

pub(crate) fn f32_slice_as_bytes_mut(values: &mut [f32]) -> &mut [u8] {
    // SAFETY: every byte pattern is a valid initialized f32 representation and
    // the mutable byte slice has the same unique borrow and lifetime.
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(values),
        )
    }
}
