fn main() {
    println!("cargo:rerun-if-changed=src/signpost.c");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        cc::Build::new()
            .file("src/signpost.c")
            .compile("j2k_metal_signpost");
    }
}
