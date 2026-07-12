// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{fs, sync::atomic::Ordering};

use super::{
    cleanup_cli_temp, command_is_runnable, kakadu_temp_dir, openjph_input_extension,
    openjph_output_extension, openjph_temp_dir, read_cli_pnm_output, reduce_factor, Container,
    Downscale, PixelFormat, OPENJPH_TEMP_COUNTER,
};

#[test]
fn comparator_extensions_and_reduce_factors_cover_supported_contracts() {
    assert_eq!(openjph_input_extension(Container::RawCodestream), "j2c");
    assert_eq!(openjph_input_extension(Container::Jp2), "jp2");
    assert_eq!(openjph_input_extension(Container::Jph), "jph");
    assert_eq!(openjph_input_extension(Container::Jhc), "jhc");
    assert_eq!(openjph_output_extension(PixelFormat::Gray8), "pgm");
    assert_eq!(openjph_output_extension(PixelFormat::Rgb8), "ppm");
    assert_eq!(openjph_output_extension(PixelFormat::Rgba8), "pnm");
    assert_eq!(reduce_factor(Downscale::None).unwrap(), 0);
    assert_eq!(reduce_factor(Downscale::Half).unwrap(), 1);
    assert_eq!(reduce_factor(Downscale::Quarter).unwrap(), 2);
    assert_eq!(reduce_factor(Downscale::Eighth).unwrap(), 3);
}

#[test]
fn cli_pnm_readback_validates_format_and_cleanup() {
    let directory = openjph_temp_dir().expect("OpenJPH temp dir");
    assert_eq!(
        directory,
        kakadu_temp_dir()
            .expect("Kakadu temp dir")
            .with_file_name("j2k-openjph-expand")
    );
    let token = OPENJPH_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let gray_path = directory.join(format!("unit-{token}.pgm"));
    let rgb_path = directory.join(format!("unit-{token}.ppm"));
    fs::write(&gray_path, b"P5\n2 1\n255\n\x01\xFE").expect("gray PNM");
    fs::write(&rgb_path, b"P6\n1 1\n255\n\x01\x02\x03").expect("RGB PNM");

    assert_eq!(
        read_cli_pnm_output("unit", &gray_path, PixelFormat::Gray8).unwrap(),
        [1, 254]
    );
    assert_eq!(
        read_cli_pnm_output("unit", &rgb_path, PixelFormat::Rgb8).unwrap(),
        [1, 2, 3]
    );
    assert!(read_cli_pnm_output("unit", &gray_path, PixelFormat::Rgba8)
        .unwrap_err()
        .contains("unsupported"));
    assert!(
        read_cli_pnm_output("unit", &directory.join("missing.pgm"), PixelFormat::Gray8,)
            .unwrap_err()
            .contains("open unit output")
    );

    cleanup_cli_temp(&gray_path, true).expect("clean gray PNM");
    cleanup_cli_temp(&rgb_path, true).expect("clean RGB PNM");
    cleanup_cli_temp(&gray_path, true).expect("missing cleanup is harmless");
    assert!(!command_is_runnable(
        &directory.join("definitely-missing-program")
    ));
}
