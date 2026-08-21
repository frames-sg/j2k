// SPDX-License-Identifier: MIT OR Apache-2.0

//! Host workspace admission for CUDA JPEG entropy diagnostics.

use super::{CudaJpegEntropyOverflowState, CudaJpegEntropySyncState};
use crate::{allocation::HostPhaseBudget, error::CudaError};

pub(super) fn allocate_diagnostic_workspaces_with_cap(
    state_count: usize,
    overflow_count: usize,
    external_live_bytes: usize,
    retained_page_locked_bytes: usize,
    cap: usize,
) -> Result<
    (
        Vec<CudaJpegEntropySyncState>,
        Vec<CudaJpegEntropyOverflowState>,
    ),
    CudaError,
> {
    let mut host_budget = HostPhaseBudget::with_cap("CUDA JPEG entropy diagnostics", cap);
    host_budget.account_bytes(external_live_bytes)?;
    host_budget.account_bytes(retained_page_locked_bytes)?;
    let states = host_budget.try_vec_filled(state_count, CudaJpegEntropySyncState::default())?;
    let overflows =
        host_budget.try_vec_filled(overflow_count, CudaJpegEntropyOverflowState::default())?;
    Ok((states, overflows))
}

#[cfg(test)]
mod tests {
    use super::allocate_diagnostic_workspaces_with_cap;
    use crate::{CudaError, CudaJpegEntropyOverflowState, CudaJpegEntropySyncState};

    #[test]
    fn diagnostic_state_and_overflow_external_live_boundary_is_exact() {
        let external = 11;
        let retained_page_locked = 30;
        let workspace_bytes = 2 * core::mem::size_of::<CudaJpegEntropySyncState>()
            + core::mem::size_of::<CudaJpegEntropyOverflowState>();
        let exact = external + retained_page_locked + workspace_bytes;
        let (states, overflows) =
            allocate_diagnostic_workspaces_with_cap(2, 1, external, retained_page_locked, exact)
                .expect("exact diagnostic workspaces");
        assert_eq!(states.len(), 2);
        assert_eq!(overflows.len(), 1);
        assert!(matches!(
            allocate_diagnostic_workspaces_with_cap(
                2,
                1,
                external,
                retained_page_locked,
                exact - 1,
            ),
            Err(CudaError::HostAllocationTooLarge { requested, cap, .. })
                if requested == exact && cap == exact - 1
        ));
    }

    #[test]
    fn diagnostic_budget_counts_new_and_reused_larger_checkout_exactly() {
        let external = 7;
        let state_bytes = core::mem::size_of::<CudaJpegEntropySyncState>();
        for retained_page_locked in [4096, 8192] {
            let exact = external + retained_page_locked + state_bytes;
            allocate_diagnostic_workspaces_with_cap(1, 0, external, retained_page_locked, exact)
                .expect("new or larger reused staging fits its exact aggregate");
            assert!(matches!(
                allocate_diagnostic_workspaces_with_cap(
                    1,
                    0,
                    external,
                    retained_page_locked,
                    exact - 1,
                ),
                Err(CudaError::HostAllocationTooLarge { requested, cap, .. })
                    if requested == exact && cap == exact - 1
            ));
        }
    }
}
