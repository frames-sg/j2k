// SPDX-License-Identifier: MIT OR Apache-2.0

//! Compile-time assertions for owner types that must not implement `Clone`.

pub(crate) trait AmbiguousIfClone<Marker> {}

impl<T: ?Sized> AmbiguousIfClone<()> for T {}

pub(crate) struct CloneMarker;

impl<T: Clone> AmbiguousIfClone<CloneMarker> for T {}

pub(crate) struct RequireUnambiguous<T: ?Sized + AmbiguousIfClone<Marker>, Marker>(
    core::marker::PhantomData<(*const T, Marker)>,
);

macro_rules! assert_move_only {
    ($($ty:ty),* $(,)?) => {
        $(
            const _: () = {
                let _: Option<$crate::move_only::RequireUnambiguous<$ty, _>> = None;
            };
        )*
    };
}

pub(crate) use assert_move_only;
