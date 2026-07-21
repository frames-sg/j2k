// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use j2k::{prepare_batch, BatchDecodeOptions, EncodedImage};

use super::fixtures::htj2k_gray8_fixture_with_levels;

#[test]
fn grouping_separates_incompatible_backend_execution_shapes() {
    let prepared = prepare_batch(
        vec![
            EncodedImage::full(Arc::from(htj2k_gray8_fixture_with_levels(16, 16, 1))),
            EncodedImage::full(Arc::from(htj2k_gray8_fixture_with_levels(16, 16, 2))),
        ],
        BatchDecodeOptions::default(),
    )
    .expect("prepare differing execution shapes");

    assert!(prepared.errors().is_empty());
    assert_eq!(prepared.groups().len(), 2);
    assert_eq!(prepared.groups()[0].source_indices(), &[0]);
    assert_eq!(prepared.groups()[1].source_indices(), &[1]);
}
