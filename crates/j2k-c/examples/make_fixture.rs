// SPDX-License-Identifier: MIT OR Apache-2.0

//! Generate a small synthetic image for C/C++ integration tests; no external data.
#[path = "../tests/support/mod.rs"]
mod support;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .ok_or("usage: make_fixture OUTPUT.j2c")?;
    let (encoded, _) = support::gray_fixture(true);
    std::fs::write(path, encoded)?;
    Ok(())
}
