use crate::command_support::run_cargo;

pub(super) fn t803(args: impl IntoIterator<Item = String>) -> Result<(), String> {
    let args = args.into_iter().collect::<Vec<_>>();
    let features = runner_features(&args);
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
    run_cargo(&cargo_args)
}

fn runner_features(args: &[impl AsRef<str>]) -> &'static str {
    match args
        .windows(2)
        .find_map(|pair| (pair[0].as_ref() == "--iut").then(|| pair[1].as_ref()))
    {
        Some("cuda") => "runner,cuda-runner",
        Some("metal") => "runner,metal-runner",
        _ => "runner",
    }
}

#[cfg(test)]
mod tests {
    use super::runner_features;

    #[test]
    fn t803_runner_features_follow_the_selected_adapter_iut() {
        assert_eq!(runner_features(&["run", "--iut", "cpu"]), "runner");
        assert_eq!(
            runner_features(&["run", "--iut", "cuda"]),
            "runner,cuda-runner"
        );
        assert_eq!(
            runner_features(&["run", "--iut", "metal"]),
            "runner,metal-runner"
        );
        assert_eq!(runner_features(&["verify"]), "runner");
    }
}
