// SPDX-License-Identifier: MIT OR Apache-2.0

//! Actual-capacity checks at parallel preparation collection boundaries.

use super::{
    encode::live::{float97_tile_retained_bytes, integer_tile_retained_bytes},
    Float97BatchTile, HostLiveBudget, IntegerBatchTile, JpegToHtj2kError, JpegToHtj2kScratch,
};

type IntegerPreparedResult = (usize, Result<IntegerBatchTile, JpegToHtj2kError>);
type Float97PreparedResult = (usize, Result<Float97BatchTile, JpegToHtj2kError>);

pub(super) fn validate_integer_prepared_collection(
    results: &[IntegerPreparedResult],
    outer_capacity: usize,
    scratch: &JpegToHtj2kScratch,
) -> Result<(), JpegToHtj2kError> {
    validate_prepared_collection::<IntegerPreparedResult, _>(
        results,
        outer_capacity,
        scratch,
        |(_, result)| result.as_ref().ok().map(integer_tile_retained_bytes),
    )
}

pub(super) fn validate_float97_prepared_collection(
    results: &[Float97PreparedResult],
    outer_capacity: usize,
    scratch: &JpegToHtj2kScratch,
) -> Result<(), JpegToHtj2kError> {
    validate_prepared_collection::<Float97PreparedResult, _>(
        results,
        outer_capacity,
        scratch,
        |(_, result)| result.as_ref().ok().map(float97_tile_retained_bytes),
    )
}

fn validate_prepared_collection<T, F>(
    results: &[T],
    outer_capacity: usize,
    scratch: &JpegToHtj2kScratch,
    retained: F,
) -> Result<(), JpegToHtj2kError>
where
    F: Fn(&T) -> Option<Result<usize, JpegToHtj2kError>>,
{
    let mut budget = HostLiveBudget::process_cap();
    budget.add_bytes(scratch.retained_bytes()?)?;
    budget.add_capacity::<T>(outer_capacity)?;
    for result in results {
        if let Some(bytes) = retained(result) {
            budget.add_bytes(bytes?)?;
        }
    }
    Ok(())
}
