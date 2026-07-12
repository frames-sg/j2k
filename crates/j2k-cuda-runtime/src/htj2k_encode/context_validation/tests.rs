// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    validate_htj2k_encode_context, validate_htj2k_encode_context_matches,
    HTJ2K_ENCODE_CONTEXT_MISMATCH,
};
use crate::{
    context::{CudaContext, HTJ2K_UVLC_ENCODE_TABLE_BYTES},
    error::CudaError,
    htj2k_encode::{
        CudaHtj2kEncodeResidentTarget, CudaHtj2kEncodeResources, CudaHtj2kEncodeTables,
    },
    memory::{CudaBufferPool, CudaDeviceBuffer},
};

struct EncodeContextFixture {
    context: CudaContext,
    foreign_context: CudaContext,
    local_coefficients: CudaDeviceBuffer,
    foreign_coefficients: CudaDeviceBuffer,
    local_resources: CudaHtj2kEncodeResources,
    foreign_resources: CudaHtj2kEncodeResources,
    local_pool: CudaBufferPool,
    foreign_pool: CudaBufferPool,
}

impl EncodeContextFixture {
    fn new() -> Self {
        let context = CudaContext::system_default().expect("launch CUDA context");
        let foreign_context = CudaContext::system_default().expect("foreign CUDA context");
        let local_coefficients = context.allocate(0).expect("local coefficient buffer");
        let foreign_coefficients = foreign_context
            .allocate(0)
            .expect("foreign coefficient buffer");
        let local_resources = empty_resources(&context);
        let foreign_resources = empty_resources(&foreign_context);
        let local_pool = context.buffer_pool();
        let foreign_pool = foreign_context.buffer_pool();
        Self {
            context,
            foreign_context,
            local_coefficients,
            foreign_coefficients,
            local_resources,
            foreign_resources,
            local_pool,
            foreign_pool,
        }
    }
}

fn empty_resources(context: &CudaContext) -> CudaHtj2kEncodeResources {
    CudaHtj2kEncodeResources {
        vlc_table0: context.allocate(0).expect("empty VLC table 0"),
        vlc_table1: context.allocate(0).expect("empty VLC table 1"),
        uvlc_table: context.allocate(0).expect("empty UVLC table"),
    }
}

fn assert_context_mismatch<T>(result: Result<T, CudaError>) {
    match result {
        Err(CudaError::InvalidArgument { message }) => {
            assert_eq!(message, HTJ2K_ENCODE_CONTEXT_MISMATCH);
        }
        Err(error) => panic!("expected an HTJ2K encode context mismatch, got {error}"),
        Ok(_) => panic!("expected an HTJ2K encode context mismatch"),
    }
}

#[test]
fn context_match_validation_accepts_empty_matching_and_aliasing_inputs() {
    assert!(validate_htj2k_encode_context_matches([]).is_ok());
    assert!(validate_htj2k_encode_context_matches([true, true, true]).is_ok());
}

#[test]
fn context_match_validation_rejects_each_mismatched_category() {
    for matches in [
        [false, true, true],
        [true, false, true],
        [true, true, false],
    ] {
        assert_context_mismatch(validate_htj2k_encode_context_matches(matches));
    }
}

#[test]
fn resident_entry_points_reject_foreign_coefficients_before_empty_fast_paths() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let fixture = EncodeContextFixture::new();
    let vlc0 = [0u16; 2048];
    let vlc1 = [0u16; 2048];
    let uvlc = vec![0u8; HTJ2K_UVLC_ENCODE_TABLE_BYTES];
    let tables = CudaHtj2kEncodeTables {
        vlc_table0: &vlc0,
        vlc_table1: &vlc1,
        uvlc_table: &uvlc,
    };

    assert_context_mismatch(fixture.context.encode_htj2k_codeblocks_resident(
        &fixture.foreign_coefficients,
        0,
        &[],
        tables,
    ));
    assert_context_mismatch(fixture.context.encode_htj2k_codeblock_regions_resident(
        &fixture.foreign_coefficients,
        0,
        &[],
        tables,
    ));
    assert_context_mismatch(
        fixture
            .context
            .encode_htj2k_codeblocks_resident_with_resources_and_pool(
                &fixture.foreign_coefficients,
                0,
                &[],
                &fixture.local_resources,
                &fixture.local_pool,
            ),
    );
    assert_context_mismatch(
        fixture
            .context
            .encode_htj2k_codeblock_regions_resident_with_resources_and_pool(
                &fixture.foreign_coefficients,
                0,
                &[],
                &fixture.local_resources,
                &fixture.local_pool,
            ),
    );
}

#[test]
fn resource_and_pool_entry_points_reject_foreign_owners_when_empty() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let fixture = EncodeContextFixture::new();

    assert_context_mismatch(fixture.context.encode_htj2k_codeblocks_with_resources(
        &[],
        &[],
        &fixture.foreign_resources,
    ));
    assert_context_mismatch(
        fixture
            .context
            .encode_htj2k_codeblocks_resident_with_resources_and_pool(
                &fixture.local_coefficients,
                0,
                &[],
                &fixture.foreign_resources,
                &fixture.local_pool,
            ),
    );
    assert_context_mismatch(
        fixture
            .context
            .encode_htj2k_codeblock_regions_resident_with_resources_and_pool(
                &fixture.local_coefficients,
                0,
                &[],
                &fixture.local_resources,
                &fixture.foreign_pool,
            ),
    );
}

#[test]
fn resource_validation_checks_every_table_allocation() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let fixture = EncodeContextFixture::new();

    for foreign_table in 0..3 {
        let table_owner = |index| {
            if index == foreign_table {
                &fixture.foreign_context
            } else {
                &fixture.context
            }
        };
        let resources = CudaHtj2kEncodeResources {
            vlc_table0: table_owner(0).allocate(0).expect("mixed VLC table 0"),
            vlc_table1: table_owner(1).allocate(0).expect("mixed VLC table 1"),
            uvlc_table: table_owner(2).allocate(0).expect("mixed UVLC table"),
        };
        assert_context_mismatch(fixture.context.encode_htj2k_codeblocks_with_resources(
            &[],
            &[],
            &resources,
        ));
    }
}

#[test]
fn multi_input_entry_points_validate_all_targets_resources_and_pool() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let fixture = EncodeContextFixture::new();
    let foreign_target = CudaHtj2kEncodeResidentTarget {
        coefficients: &fixture.foreign_coefficients,
        coefficient_count: 0,
        jobs: &[],
    };
    let local_target = CudaHtj2kEncodeResidentTarget {
        coefficients: &fixture.local_coefficients,
        coefficient_count: 0,
        jobs: &[],
    };

    assert_context_mismatch(
        fixture
            .context
            .encode_htj2k_codeblocks_multi_resident_with_resources_and_pool(
                &[local_target, foreign_target, local_target],
                &fixture.local_resources,
                &fixture.local_pool,
            ),
    );
    assert_context_mismatch(
        fixture
            .context
            .encode_htj2k_codeblocks_multi_resident_compact_with_resources_and_pool(
                &[foreign_target],
                &fixture.local_resources,
                &fixture.local_pool,
            ),
    );
    assert_context_mismatch(
        fixture
            .context
            .encode_htj2k_codeblocks_multi_resident_with_resources_and_pool(
                &[],
                &fixture.foreign_resources,
                &fixture.local_pool,
            ),
    );
    assert_context_mismatch(
        fixture
            .context
            .encode_htj2k_codeblocks_multi_resident_compact_with_resources_and_pool(
                &[],
                &fixture.local_resources,
                &fixture.foreign_pool,
            ),
    );
}

#[test]
fn matching_empty_batches_and_repeated_same_context_targets_remain_valid() {
    if !j2k_test_support::cuda_runtime_gate(module_path!()) {
        return;
    }
    let fixture = EncodeContextFixture::new();
    assert!(validate_htj2k_encode_context(
        &fixture.context,
        [
            &fixture.local_coefficients,
            &fixture.local_coefficients,
            &fixture.local_coefficients,
        ],
        Some(&fixture.local_resources),
        Some(&fixture.local_pool),
    )
    .is_ok());

    let target = CudaHtj2kEncodeResidentTarget {
        coefficients: &fixture.local_coefficients,
        coefficient_count: 0,
        jobs: &[],
    };
    let compact = fixture
        .context
        .encode_htj2k_codeblocks_multi_resident_compact_with_resources_and_pool(
            &[target, target],
            &fixture.local_resources,
            &fixture.local_pool,
        )
        .expect("matching empty multi-input encode");
    assert!(compact.payload().is_empty());
    assert!(compact.code_blocks().is_empty());

    let blocks = fixture
        .context
        .encode_htj2k_codeblocks_resident_with_resources_and_pool(
            &fixture.local_coefficients,
            0,
            &[],
            &fixture.local_resources,
            &fixture.local_pool,
        )
        .expect("matching empty resident encode");
    assert!(blocks.code_blocks().is_empty());
}
