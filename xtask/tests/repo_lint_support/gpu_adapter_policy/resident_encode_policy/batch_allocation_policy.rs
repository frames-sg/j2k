// SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;

use crate::repo_lint_support::{assert_pattern_checks, repo_root, PatternCheck};

#[test]
fn cuda_public_batch_encode_collections_remain_fallible() {
    let root = repo_root();
    let cuda = fs::read_to_string(root.join("crates/j2k-cuda/src/encode.rs"))
        .expect("read CUDA encode adapter");
    let production = cuda
        .split("#[cfg(test)]\nmod tests")
        .next()
        .expect("CUDA encode production prefix");

    assert_pattern_checks(&[
        PatternCheck::new("CUDA public batch encode allocation", production)
            .required(&[
                "HostPhaseBudget::new(",
                "host_encode_outcome_budget(",
                "host_budget.try_vec_with_capacity(",
                "host_budget.account_vec(&outcome.encoded.codestream)?;",
                "host_budget.account_vec(&outcomes)?;",
                "j2k CUDA submitted batch codestreams",
                "j2k CUDA host batch encode outcomes",
                "j2k CUDA resident batch codestreams",
                "j2k CUDA resident batch encode outcomes",
            ])
            .forbidden(&[".collect()", "Vec::with_capacity", ".collect::<Vec"]),
    ]);
    assert_eq!(
        production
            .matches("host_budget.try_vec_with_capacity(")
            .count(),
        4,
        "each public CUDA tile-batch conversion must reserve fallibly"
    );
    assert_eq!(
        production.matches("host_encode_outcome_budget(").count(),
        3,
        "both codestream-owning conversions must seed their budget from actual capacities"
    );
}
