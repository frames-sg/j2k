// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::Error;

pub(crate) fn try_vec<T>(capacity: usize, what: &'static str) -> Result<Vec<T>, Error> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|source| Error::Allocation {
            what,
            requested: capacity,
            source,
        })?;
    Ok(values)
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub(crate) fn try_clone_slice<T: Clone>(values: &[T], what: &'static str) -> Result<Vec<T>, Error> {
    let mut cloned = try_vec(values.len(), what)?;
    cloned.extend_from_slice(values);
    Ok(cloned)
}

#[cfg(all(target_arch = "aarch64", target_os = "macos"))]
pub(crate) fn try_single<T>(value: T, what: &'static str) -> Result<Vec<T>, Error> {
    let mut values = try_vec(1, what)?;
    values.push(value);
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::try_vec;
    use crate::Error;

    #[test]
    fn impossible_capacity_preserves_allocation_context() {
        let error = try_vec::<u8>(usize::MAX, "MPSGraph allocation test")
            .expect_err("impossible capacity must fail before allocation");

        assert!(matches!(
            error,
            Error::Allocation {
                what: "MPSGraph allocation test",
                requested: usize::MAX,
                ..
            }
        ));
    }
}
