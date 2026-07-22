use std::env;
use std::error::Error;
use std::io;

fn main() -> Result<(), Box<dyn Error>> {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")
        .map_err(|_| io::Error::other("Cargo did not provide CARGO_MANIFEST_DIR"))?;
    println!("cargo::metadata=manifest_dir={manifest_dir}");
    Ok(())
}
