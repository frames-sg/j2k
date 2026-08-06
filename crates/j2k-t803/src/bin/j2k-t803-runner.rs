// SPDX-License-Identifier: MIT OR Apache-2.0

use std::process::ExitCode;

fn main() -> ExitCode {
    match j2k_t803::runner::run_cli(std::env::args().skip(1)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("T.803 runner failed: {error}");
            ExitCode::FAILURE
        }
    }
}
