// SPDX-License-Identifier: MIT OR Apache-2.0

fn adoption_feature_required(task: &str, args: impl Iterator<Item = String>) -> Result<(), String> {
    let _ = args.count();
    Err(format!(
        "`cargo xtask {task}` requires the xtask `adoption` feature. Run `cargo run -p xtask --features adoption -- {task}`."
    ))
}

pub(crate) fn adoption_benchmark(args: impl Iterator<Item = String>) -> Result<(), String> {
    adoption_feature_required("adoption-benchmark", args)
}

pub(crate) fn adoption_curate(args: impl Iterator<Item = String>) -> Result<(), String> {
    adoption_feature_required("adoption-curate", args)
}

pub(crate) fn adoption_manifest(args: impl Iterator<Item = String>) -> Result<(), String> {
    adoption_feature_required("adoption-manifest", args)
}

pub(crate) fn adoption_materialize(args: impl Iterator<Item = String>) -> Result<(), String> {
    adoption_feature_required("adoption-materialize", args)
}

pub(crate) fn adoption_report(args: impl Iterator<Item = String>) -> Result<(), String> {
    adoption_feature_required("adoption-report", args)
}
