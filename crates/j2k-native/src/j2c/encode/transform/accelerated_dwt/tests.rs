// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::j2c::encode::{NativeEncodePipelineError, NativeEncodeRetainedInput};
use crate::{EncodeError, J2kEncodeStageError, J2kEncodeStageErrorKind};
use alloc::vec;

#[derive(Default)]
struct FakeDwtAccelerator {
    output53: Option<J2kForwardDwt53Output>,
    output97: Option<J2kForwardDwt97Output>,
    fail53: bool,
    fail97: bool,
    calls53: usize,
    calls97: usize,
}

impl J2kEncodeStageAccelerator for FakeDwtAccelerator {
    fn encode_forward_dwt53(
        &mut self,
        _job: J2kForwardDwt53Job<'_>,
    ) -> crate::J2kEncodeStageResult<Option<J2kForwardDwt53Output>> {
        self.calls53 += 1;
        if self.fail53 {
            return Err(J2kEncodeStageError::internal_invariant(
                "test forward 5/3 failure",
            ));
        }
        Ok(self.output53.take())
    }

    fn encode_forward_dwt97(
        &mut self,
        _job: J2kForwardDwt97Job<'_>,
    ) -> crate::J2kEncodeStageResult<Option<J2kForwardDwt97Output>> {
        self.calls97 += 1;
        if self.fail97 {
            return Err(J2kEncodeStageError::internal_invariant(
                "test forward 9/7 failure",
            ));
        }
        Ok(self.output97.take())
    }
}

fn samples() -> Vec<f32> {
    (0_u8..20).map(|value| f32::from(value) - 7.0).collect()
}

fn session() -> NativeEncodeSession<'static> {
    NativeEncodeSession::try_new(NativeEncodeRetainedInput::none())
        .expect("default native encode session")
}

fn run_dwt(
    component: Vec<f32>,
    reversible: bool,
    session: &NativeEncodeSession<'_>,
    accelerator: &mut FakeDwtAccelerator,
) -> NativeEncodePipelineResult<OwnedDwtComponent> {
    let mut line_scratch = vec![0.0; 5];
    encode_forward_dwt(
        ForwardDwtRequest {
            component,
            width: 5,
            height: 4,
            num_levels: 1,
            reversible,
            session,
            retained_base_bytes: 0,
            line_scratch: &mut line_scratch,
        },
        accelerator,
    )
}

fn assert_decomposition_eq(actual: &DwtDecomposition, expected: &DwtDecomposition) {
    assert_eq!(actual.ll, expected.ll);
    assert_eq!(actual.ll_width, expected.ll_width);
    assert_eq!(actual.ll_height, expected.ll_height);
    assert_eq!(actual.levels.len(), expected.levels.len());
    for (actual, expected) in actual.levels.iter().zip(&expected.levels) {
        assert_eq!(actual.hl, expected.hl);
        assert_eq!(actual.lh, expected.lh);
        assert_eq!(actual.hh, expected.hh);
        assert_eq!(actual.low_width, expected.low_width);
        assert_eq!(actual.low_height, expected.low_height);
        assert_eq!(actual.high_width, expected.high_width);
        assert_eq!(actual.high_height, expected.high_height);
    }
}

#[test]
fn declined_acceleration_preserves_scalar_packed_dwt_for_both_filters() {
    for reversible in [true, false] {
        let source = samples();
        let mut expected = source.clone();
        let mut expected_scratch = vec![0.0; 5];
        fdwt::try_forward_dwt_packed_f32(&mut expected, 5, 4, 1, reversible, &mut expected_scratch)
            .expect("scalar packed DWT reference");
        let mut accelerator = FakeDwtAccelerator::default();

        let actual = run_dwt(source, reversible, &session(), &mut accelerator)
            .expect("declined accelerator uses scalar path");

        let OwnedDwtComponent::Packed(actual) = actual else {
            panic!("declined acceleration must retain packed scalar ownership");
        };
        assert_eq!(actual.coefficients, expected);
        assert_eq!(accelerator.calls53, usize::from(reversible));
        assert_eq!(accelerator.calls97, usize::from(!reversible));
    }
}

#[test]
fn accepted_acceleration_preserves_scalar_decomposition_for_both_filters() {
    let source = samples();
    let expected53 =
        fdwt::try_forward_dwt(&source, 5, 4, 1, true).expect("scalar decomposed 5/3 reference");
    let output53 = crate::scalar::forward_dwt53_reference(&source, 5, 4, 1)
        .expect("public 5/3 accelerator output");
    let mut accelerator = FakeDwtAccelerator {
        output53: Some(output53),
        ..FakeDwtAccelerator::default()
    };
    let actual53 = run_dwt(source.clone(), true, &session(), &mut accelerator)
        .expect("accepted 5/3 accelerator output");
    let OwnedDwtComponent::Decomposed(actual53) = actual53 else {
        panic!("accepted acceleration must use decomposed ownership");
    };
    assert_decomposition_eq(&actual53, &expected53);
    assert_eq!((accelerator.calls53, accelerator.calls97), (1, 0));

    let expected97 =
        fdwt::try_forward_dwt(&source, 5, 4, 1, false).expect("scalar decomposed 9/7 reference");
    let output97 = crate::scalar::forward_dwt97_reference(&source, 5, 4, 1)
        .expect("public 9/7 accelerator output");
    let mut accelerator = FakeDwtAccelerator {
        output97: Some(output97),
        ..FakeDwtAccelerator::default()
    };
    let actual97 = run_dwt(source, false, &session(), &mut accelerator)
        .expect("accepted 9/7 accelerator output");
    let OwnedDwtComponent::Decomposed(actual97) = actual97 else {
        panic!("accepted acceleration must use decomposed ownership");
    };
    assert_decomposition_eq(&actual97, &expected97);
    assert_eq!((accelerator.calls53, accelerator.calls97), (0, 1));
}

#[test]
fn accelerator_failures_keep_the_transform_operation_and_typed_source() {
    for reversible in [true, false] {
        let mut accelerator = FakeDwtAccelerator {
            fail53: reversible,
            fail97: !reversible,
            ..FakeDwtAccelerator::default()
        };

        let Err(error) = run_dwt(samples(), reversible, &session(), &mut accelerator) else {
            panic!("accepted accelerator failure must not fall back");
        };

        let NativeEncodePipelineError::Typed(EncodeError::Accelerator { operation, source }) =
            error
        else {
            panic!("accelerator failure must retain typed encode classification");
        };
        assert_eq!(
            operation,
            if reversible {
                "forward 5/3 DWT"
            } else {
                "forward 9/7 DWT"
            }
        );
        assert_eq!(source.kind(), J2kEncodeStageErrorKind::InternalInvariant);
    }
}

fn valid_53_output() -> J2kForwardDwt53Output {
    crate::scalar::forward_dwt53_reference(&samples(), 5, 4, 1)
        .expect("valid 5/3 accelerator fixture")
}

fn valid_97_output() -> J2kForwardDwt97Output {
    crate::scalar::forward_dwt97_reference(&samples(), 5, 4, 1)
        .expect("valid 9/7 accelerator fixture")
}

#[test]
fn malformed_accelerator_outputs_are_rejected_before_ownership_conversion() {
    let mut bad53 = valid_53_output();
    bad53.ll.pop();
    let error = convert_forward_dwt53_output(bad53, &session(), 0, "test 5/3 conversion")
        .expect_err("short 5/3 LL band must reject");
    assert!(matches!(
        error,
        NativeEncodePipelineError::Typed(EncodeError::Accelerator { operation, source })
            if operation == "test 5/3 conversion"
                && source.kind() == J2kEncodeStageErrorKind::InternalInvariant
    ));

    let mut bad97 = valid_97_output();
    bad97.levels[0].hh.pop();
    let error = convert_forward_dwt97_output(bad97, &session(), 0, "test 9/7 conversion")
        .expect_err("short 9/7 detail band must reject");
    assert!(matches!(
        error,
        NativeEncodePipelineError::Typed(EncodeError::Accelerator { operation, source })
            if operation == "test 9/7 conversion"
                && source.kind() == J2kEncodeStageErrorKind::InternalInvariant
    ));
}

#[test]
fn accelerator_output_conversion_has_an_exact_aggregate_capacity_boundary() {
    let output = valid_53_output();
    let source_bytes = forward_dwt53_output_retained_bytes(&output)
        .expect("accelerator source allocation accounting");
    let owner_bytes = output.levels.len() * core::mem::size_of::<fdwt::DwtLevel>();
    let exact_cap = source_bytes + owner_bytes;
    let exact = NativeEncodeSession::try_with_cap(NativeEncodeRetainedInput::none(), exact_cap)
        .expect("exact accelerator conversion session");

    convert_forward_dwt53_output(output, &exact, 0, "exact 5/3 conversion")
        .expect("exact source plus destination overlap fits");

    let over = NativeEncodeSession::try_with_cap(
        NativeEncodeRetainedInput::none(),
        exact_cap.saturating_sub(1),
    )
    .expect("cap-minus-one session construction");
    let error = convert_forward_dwt53_output(valid_53_output(), &over, 0, "bounded 5/3 conversion")
        .expect_err("cap-minus-one conversion must reject");
    assert!(matches!(
        error,
        NativeEncodePipelineError::Typed(EncodeError::AllocationTooLarge { requested, cap, .. })
            if requested == exact_cap && cap == exact_cap - 1
    ));
}

#[test]
fn band_validators_cover_each_detail_orientation_for_both_filters() {
    let valid53 = valid_53_output();
    validate_dwt53_level(&valid53.levels[0]).expect("valid 5/3 detail geometry");
    let valid97 = valid_97_output();
    validate_dwt97_level(&valid97.levels[0]).expect("valid 9/7 detail geometry");

    let mut bad53 = valid_53_output().levels.remove(0);
    bad53.hl.pop();
    assert_eq!(
        validate_dwt53_level(&bad53),
        Err("accelerated DWT output length mismatch")
    );
    let mut bad97 = valid_97_output().levels.remove(0);
    bad97.lh.pop();
    assert_eq!(
        validate_dwt97_level(&bad97),
        Err("accelerated DWT output length mismatch")
    );
    assert_eq!(
        validate_band_len(3, 2, 2),
        Err("accelerated DWT output length mismatch")
    );
}
