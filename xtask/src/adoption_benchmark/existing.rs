use std::path::Path;

use super::options::AdoptionBenchmarkOptions;
use super::runner::skipped_step;
use super::summary::{AdoptionStep, StepStatus};
use super::support::criterion_target_dir;

#[cfg(test)]
mod tests;

pub(super) fn existing_steps(
    options: &AdoptionBenchmarkOptions,
) -> Result<Vec<AdoptionStep>, String> {
    let mut steps = vec![
        existing_ran_step("cpu-fixture-compare", None, &options.out_dir)?,
        existing_ran_step("cpu-encode-compare", None, &options.out_dir)?,
        existing_ran_step(
            "cpu-public-api-encode",
            Some(&criterion_target_dir(options, "cpu-public-api-encode")),
            &options.out_dir,
        )?,
        existing_ran_step(
            "cpu-public-api-decode",
            Some(&criterion_target_dir(options, "cpu-public-api-decode")),
            &options.out_dir,
        )?,
    ];

    if options.cuda {
        steps.push(existing_ran_step(
            "cuda-htj2k-decode",
            Some(&criterion_target_dir(options, "cuda-htj2k-decode")),
            &options.out_dir,
        )?);
        steps.push(existing_ran_step(
            "cuda-htj2k-encode",
            Some(&criterion_target_dir(options, "cuda-htj2k-encode")),
            &options.out_dir,
        )?);
    } else {
        steps.push(skipped_step(
            "cuda-htj2k-decode",
            "not requested; pass --cuda for CUDA decode/encode Criterion benches",
            &options.out_dir,
        ));
        steps.push(skipped_step(
            "cuda-htj2k-encode",
            "not requested; pass --cuda for CUDA decode/encode Criterion benches",
            &options.out_dir,
        ));
    }

    if options.metal {
        steps.push(existing_ran_step(
            "metal-decode-benchmark",
            None,
            &options.out_dir,
        )?);
        steps.push(existing_ran_step(
            "metal-encode-auto-routing",
            None,
            &options.out_dir,
        )?);
        steps.push(existing_ran_step(
            "metal-transcode-benchmark",
            Some(&criterion_target_dir(options, "metal-transcode-benchmark")),
            &options.out_dir,
        )?);
    } else {
        steps.push(skipped_step(
            "metal-decode-benchmark",
            "not requested; pass --metal for Metal decode benchmark",
            &options.out_dir,
        ));
        steps.push(skipped_step(
            "metal-encode-auto-routing",
            "not requested; pass --metal for Metal hybrid encode routing benchmark",
            &options.out_dir,
        ));
        steps.push(skipped_step(
            "metal-transcode-benchmark",
            "not requested; pass --metal for Metal transcode benchmark",
            &options.out_dir,
        ));
    }

    Ok(steps)
}

pub(super) fn existing_ran_step(
    name: &'static str,
    target_dir: Option<&Path>,
    out_dir: &Path,
) -> Result<AdoptionStep, String> {
    let stdout = out_dir.join(format!("{name}.out"));
    let stderr = out_dir.join(format!("{name}.err"));
    if !stdout.is_file() {
        return Err(format!(
            "--finalize-existing requires completed {name} stdout at {}",
            stdout.display()
        ));
    }
    let stdout_len = stdout
        .metadata()
        .map_err(|err| format!("stat {}: {err}", stdout.display()))?
        .len();
    if stdout_len == 0 {
        return Err(format!(
            "--finalize-existing found empty {name} stdout at {}",
            stdout.display()
        ));
    }
    if !stderr.is_file() {
        return Err(format!(
            "--finalize-existing requires {name} stderr at {}",
            stderr.display()
        ));
    }
    Ok(AdoptionStep {
        name,
        command: "existing artifact reused by --finalize-existing".to_string(),
        stdout,
        stderr,
        criterion_root: target_dir.map(|path| path.join("criterion")),
        status: StepStatus::Ran,
    })
}
