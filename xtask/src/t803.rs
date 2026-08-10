use crate::command_support::run_cargo_with_env;

const CUDA_RUNNER_ENV: &[(&str, &str)] = &[
    ("J2K_REQUIRE_CUDA_RUNTIME", "1"),
    ("J2K_REQUIRE_CUDA_OXIDE_BUILD", "1"),
];

pub(super) fn t803(args: impl IntoIterator<Item = String>) -> Result<(), String> {
    let args = args.into_iter().collect::<Vec<_>>();
    let (features, envs) = runner_config(&args);
    let mut cargo_args = [
        "run",
        "--quiet",
        "-p",
        "j2k-t803",
        "--features",
        features,
        "--bin",
        "j2k-t803-runner",
        "--",
    ]
    .into_iter()
    .map(str::to_string)
    .collect::<Vec<_>>();
    cargo_args.extend(args);
    let cargo_args = cargo_args.iter().map(String::as_str).collect::<Vec<_>>();
    run_cargo_with_env(&cargo_args, envs)
}

fn runner_config(
    args: &[impl AsRef<str>],
) -> (&'static str, &'static [(&'static str, &'static str)]) {
    match args
        .windows(2)
        .find_map(|pair| (pair[0].as_ref() == "--iut").then(|| pair[1].as_ref()))
    {
        Some("cuda") => ("runner,cuda-runner", CUDA_RUNNER_ENV),
        Some("metal") => ("runner,metal-runner", &[]),
        _ => ("runner", &[]),
    }
}

#[cfg(test)]
mod tests {
    use super::runner_config;

    #[test]
    fn t803_runner_configuration_follows_the_selected_adapter_iut() {
        assert_eq!(
            runner_config(&["run", "--iut", "cpu"]),
            ("runner", &[] as &[(&str, &str)])
        );
        assert_eq!(
            runner_config(&["run", "--iut", "cuda"]),
            (
                "runner,cuda-runner",
                &[
                    ("J2K_REQUIRE_CUDA_RUNTIME", "1"),
                    ("J2K_REQUIRE_CUDA_OXIDE_BUILD", "1"),
                ][..],
            )
        );
        assert_eq!(
            runner_config(&["run", "--iut", "metal"]),
            ("runner,metal-runner", &[] as &[(&str, &str)])
        );
        assert_eq!(
            runner_config(&["verify"]),
            ("runner", &[] as &[(&str, &str)])
        );
    }
}
