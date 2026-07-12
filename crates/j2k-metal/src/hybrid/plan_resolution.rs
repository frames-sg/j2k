// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{batch_allocation, Error};

pub(super) fn try_resolve_plans_in_order<T, R>(
    inputs: &[T],
    mut resolve: impl FnMut(&T) -> Result<R, Error>,
) -> Result<Vec<R>, Error> {
    let mut budget = batch_allocation::BatchMetadataBudget::new(
        "J2K Metal ordered region-scaled plan resolution",
    );
    budget.preflight(&[batch_allocation::BatchMetadataRequest::of::<R>(
        inputs.len(),
    )])?;
    let mut ordered = budget.try_vec(inputs.len(), "J2K Metal ordered region-scaled plans")?;

    for input in inputs {
        ordered.push(resolve(input)?);
    }
    Ok(ordered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolution_preserves_order_and_stops_at_first_error() {
        let inputs = core::array::from_fn::<_, 17, _>(|index| {
            u8::try_from(index).expect("resolver test index fits u8")
        });
        let ordered = try_resolve_plans_in_order(&inputs, |input| Ok(usize::from(*input) * 2))
            .expect("fallible ordered resolution");
        for (index, value) in ordered.iter().enumerate() {
            assert_eq!(*value, index * 2);
        }

        let mut calls = Vec::new();
        let error = try_resolve_plans_in_order(&inputs, |input| {
            calls.push(*input);
            if *input == 2 || *input == 7 {
                return Err(Error::MetalKernel {
                    message: format!("resolver error {input}"),
                });
            }
            Ok(*input)
        })
        .expect_err("lowest-index resolver error");
        assert!(matches!(
            error,
            Error::MetalKernel { message } if message == "resolver error 2"
        ));
        assert_eq!(calls, [0, 1, 2]);
    }
}
