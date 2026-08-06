// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    encode_with_accelerator, EncodeError, EncodeOptions, J2kDeinterleaveToF32Job, J2kEncodeContext,
    J2kEncodeStageAccelerator, J2kForwardDwt53Job, J2kForwardIctJob, J2kForwardRctJob,
};
use alloc::{vec, vec::Vec};

#[derive(Clone, Copy)]
enum FailedStage {
    Begin,
    Deinterleave,
    Rct,
    Ict,
    Dwt53,
}

struct FailingAccelerator(FailedStage);

struct MalformedDeinterleaveAccelerator;

#[derive(Default)]
struct ContextRecordingAccelerator {
    context: Option<J2kEncodeContext>,
}

impl J2kEncodeStageAccelerator for ContextRecordingAccelerator {
    fn begin_encode(&mut self, context: J2kEncodeContext) -> crate::J2kEncodeStageResult<()> {
        self.context = Some(context);
        Ok(())
    }
}

impl J2kEncodeStageAccelerator for MalformedDeinterleaveAccelerator {
    fn encode_deinterleave(
        &mut self,
        _job: J2kDeinterleaveToF32Job<'_>,
    ) -> crate::J2kEncodeStageResult<Option<Vec<Vec<f32>>>> {
        Ok(Some(vec![Vec::new()]))
    }
}

impl J2kEncodeStageAccelerator for FailingAccelerator {
    fn begin_encode(&mut self, _context: J2kEncodeContext) -> crate::J2kEncodeStageResult<()> {
        if matches!(self.0, FailedStage::Begin) {
            Err(crate::J2kEncodeStageError::internal_invariant(
                "staged test failure",
            ))
        } else {
            Ok(())
        }
    }

    fn encode_deinterleave(
        &mut self,
        _job: J2kDeinterleaveToF32Job<'_>,
    ) -> crate::J2kEncodeStageResult<Option<Vec<Vec<f32>>>> {
        if matches!(self.0, FailedStage::Deinterleave) {
            Err(crate::J2kEncodeStageError::internal_invariant(
                "staged test failure",
            ))
        } else {
            Ok(None)
        }
    }

    fn encode_forward_rct(
        &mut self,
        _job: J2kForwardRctJob<'_>,
    ) -> crate::J2kEncodeStageResult<bool> {
        if matches!(self.0, FailedStage::Rct) {
            Err(crate::J2kEncodeStageError::internal_invariant(
                "staged test failure",
            ))
        } else {
            Ok(false)
        }
    }

    fn encode_forward_ict(
        &mut self,
        _job: J2kForwardIctJob<'_>,
    ) -> crate::J2kEncodeStageResult<bool> {
        if matches!(self.0, FailedStage::Ict) {
            Err(crate::J2kEncodeStageError::internal_invariant(
                "staged test failure",
            ))
        } else {
            Ok(false)
        }
    }

    fn encode_forward_dwt53(
        &mut self,
        _job: J2kForwardDwt53Job<'_>,
    ) -> crate::J2kEncodeStageResult<Option<crate::J2kForwardDwt53Output>> {
        if matches!(self.0, FailedStage::Dwt53) {
            Err(crate::J2kEncodeStageError::internal_invariant(
                "staged test failure",
            ))
        } else {
            Ok(None)
        }
    }
}

fn assert_stage_error(
    stage: FailedStage,
    operation: &'static str,
    components: u16,
    reversible: bool,
) {
    let pixels = vec![17_u8; 8 * 8 * usize::from(components)];
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible,
        guard_bits: if reversible { 1 } else { 2 },
        use_mct: components == 3,
        ..EncodeOptions::default()
    };
    let mut accelerator = FailingAccelerator(stage);
    let error = encode_with_accelerator(
        &pixels,
        8,
        8,
        components,
        8,
        false,
        &options,
        &mut accelerator,
    )
    .expect_err("accelerator stage must fail");
    assert_eq!(
        error,
        EncodeError::Accelerator {
            operation,
            source: crate::J2kEncodeStageError::internal_invariant("staged test failure"),
        }
    );
}

#[test]
fn staged_accelerator_failures_keep_typed_operation_taxonomy() {
    assert_stage_error(FailedStage::Begin, "encode route selection", 1, true);
    assert_stage_error(FailedStage::Deinterleave, "pixel deinterleave", 1, true);
    assert_stage_error(FailedStage::Rct, "forward RCT", 3, true);
    assert_stage_error(FailedStage::Ict, "forward ICT", 3, false);
    assert_stage_error(FailedStage::Dwt53, "forward 5/3 DWT", 1, true);
}

#[test]
fn encode_supplies_validated_route_context_before_stage_dispatch() {
    let pixels = vec![17_u8; 8 * 4 * 3];
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: false,
        guard_bits: 2,
        use_mct: true,
        ..EncodeOptions::default()
    };
    let mut accelerator = ContextRecordingAccelerator::default();

    encode_with_accelerator(&pixels, 8, 4, 3, 8, false, &options, &mut accelerator)
        .expect("encode with route context");

    assert_eq!(
        accelerator.context,
        Some(J2kEncodeContext {
            num_pixels: 32,
            num_components: 3,
            bit_depth: 8,
            signed: false,
            reversible: false,
        })
    );
}

#[test]
fn malformed_accelerator_output_keeps_the_accelerator_category() {
    let pixels = vec![17_u8; 8 * 8];
    let options = EncodeOptions {
        num_decomposition_levels: 1,
        reversible: true,
        ..EncodeOptions::default()
    };
    let error = encode_with_accelerator(
        &pixels,
        8,
        8,
        1,
        8,
        false,
        &options,
        &mut MalformedDeinterleaveAccelerator,
    )
    .expect_err("malformed accelerator output must fail");

    assert_eq!(
        error,
        EncodeError::Accelerator {
            operation: "pixel deinterleave",
            source: crate::J2kEncodeStageError::internal_invariant(
                "accelerated deinterleave component length mismatch",
            ),
        }
    );
}
