// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

pub(super) fn get_or_try_init_context_bound<C, R, E>(
    bound_context: &mut Option<C>,
    cached_resource: &mut Option<Arc<R>>,
    requested_context: &C,
    same_context: impl FnOnce(&C, &C) -> bool,
    mismatch_error: impl FnOnce() -> E,
    initialize: impl FnOnce(&C) -> Result<R, E>,
) -> Result<(Arc<R>, bool), E>
where
    C: Clone,
{
    if let Some(context) = bound_context.as_ref() {
        if !same_context(context, requested_context) {
            return Err(mismatch_error());
        }
    } else {
        *bound_context = Some(requested_context.clone());
    }

    if let Some(resource) = cached_resource.as_ref() {
        return Ok((Arc::clone(resource), false));
    }

    let resource = Arc::new(initialize(requested_context)?);
    *cached_resource = Some(Arc::clone(&resource));
    Ok((resource, true))
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, sync::Arc};

    use super::get_or_try_init_context_bound;

    #[test]
    fn compatible_context_reuses_one_shared_resource() {
        let mut context = None;
        let mut resource = None;
        let initializations = Cell::new(0usize);

        let (first, first_initialized) = get_or_try_init_context_bound(
            &mut context,
            &mut resource,
            &7u64,
            u64::eq,
            || "context mismatch",
            |bound| {
                initializations.set(initializations.get() + 1);
                Ok::<_, &str>(*bound * 10)
            },
        )
        .expect("first resource initialization");
        let (second, second_initialized) = get_or_try_init_context_bound(
            &mut context,
            &mut resource,
            &7u64,
            u64::eq,
            || "context mismatch",
            |_| panic!("cached resource must bypass initialization"),
        )
        .expect("cached resource lookup");

        assert!(first_initialized);
        assert!(!second_initialized);
        assert_eq!(initializations.get(), 1);
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn mismatched_context_is_rejected_before_resource_initialization() {
        let mut context = None;
        let mut resource = None;
        let (cached, _) = get_or_try_init_context_bound(
            &mut context,
            &mut resource,
            &7u64,
            u64::eq,
            || "context mismatch",
            |_| Ok::<_, &str>(70u64),
        )
        .expect("first resource initialization");
        let initializations = Cell::new(0usize);

        let result = get_or_try_init_context_bound(
            &mut context,
            &mut resource,
            &8u64,
            u64::eq,
            || "context mismatch",
            |_| {
                initializations.set(initializations.get() + 1);
                Ok(80u64)
            },
        );

        assert_eq!(result.expect_err("mismatch must fail"), "context mismatch");
        assert_eq!(initializations.get(), 0);
        assert_eq!(context, Some(7));
        assert!(Arc::ptr_eq(
            &cached,
            resource.as_ref().expect("existing cache is preserved")
        ));
    }

    #[test]
    fn failed_initialization_keeps_binding_and_allows_compatible_retry() {
        let mut context = None;
        let mut resource = None;

        let failed = get_or_try_init_context_bound(
            &mut context,
            &mut resource,
            &7u64,
            u64::eq,
            || "context mismatch",
            |_| Err::<u64, _>("upload failed"),
        );
        assert_eq!(
            failed.expect_err("initialization must fail"),
            "upload failed"
        );
        assert_eq!(context, Some(7));
        assert!(resource.is_none());

        let (retried, initialized) = get_or_try_init_context_bound(
            &mut context,
            &mut resource,
            &7u64,
            u64::eq,
            || "context mismatch",
            |_| Ok::<_, &str>(70u64),
        )
        .expect("compatible retry");

        assert!(initialized);
        assert_eq!(*retried, 70);
    }
}
