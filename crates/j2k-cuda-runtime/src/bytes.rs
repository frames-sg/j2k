use j2k_core::accelerator::GpuAbi;

macro_rules! gpu_slice_bytes {
    ($($(#[$attr:meta])* $name:ident: $ty:ty;)+) => {
        $(
            $(#[$attr])*
            pub(crate) fn $name(values: &[$ty]) -> &[u8] {
                <$ty as GpuAbi>::slice_as_bytes(values)
            }
        )+
    };
}

macro_rules! gpu_slice_bytes_mut {
    ($($(#[$attr:meta])* $name:ident: $ty:ty;)+) => {
        $(
            $(#[$attr])*
            pub(crate) fn $name(values: &mut [$ty]) -> &mut [u8] {
                <$ty as GpuAbi>::slice_as_bytes_mut(values)
            }
        )+
    };
}

gpu_slice_bytes! {
    f32_slice_as_bytes: f32;
    i16_slice_as_bytes: i16;
    #[cfg(test)]
    i32_slice_as_bytes: i32;
}

gpu_slice_bytes_mut! {
    #[cfg(test)]
    f32_slice_as_bytes_mut: f32;
    #[cfg(test)]
    i32_slice_as_bytes_mut: i32;
}
