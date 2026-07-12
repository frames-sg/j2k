// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::HashMap,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Mutex, MutexGuard,
    },
};

use super::{
    collect_source_image_paths, encode_manifest_from_env, external_corpus_category,
    external_image_metadata, external_source_label, generated_image_cases, is_pnm_path,
    is_supported_source_image_path, load_external_image_cases, manifest_column, manifest_field,
    manifest_optional_value, manifest_required_value, mixed_external_batches, pnm_extension,
    read_pnm, read_raster_image, read_source_image, select_cases, source_format_label, write_pnm,
};
use crate::encode_compare::{EncodeManifest, EncodeManifestEntry, ImageCase, PnmImage};

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);
static ENV_SERIAL: Mutex<()> = Mutex::new(());

fn temp_dir(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "j2k-encode-images-{label}-{}-{}",
        std::process::id(),
        NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).expect("create image test directory");
    root
}

struct EnvGuard {
    _serial: MutexGuard<'static, ()>,
    key: &'static str,
    previous: Option<OsString>,
}

impl EnvGuard {
    fn set(key: &'static str, value: &Path) -> Self {
        let serial = ENV_SERIAL.lock().expect("image env serial lock");
        let previous = std::env::var_os(key);
        std::env::set_var(key, value);
        Self {
            _serial: serial,
            key,
            previous,
        }
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        if let Some(previous) = &self.previous {
            std::env::set_var(self.key, previous);
        } else {
            std::env::remove_var(self.key);
        }
    }
}

fn case(name: &str, source: &str, components: u8, pixels: &[u8]) -> ImageCase {
    ImageCase {
        name: name.to_string(),
        input_source: source.to_string(),
        corpus_category: "natural-image".to_string(),
        corpus_name: "fixture".to_string(),
        license_status: "cc0".to_string(),
        source_command: "fixture".to_string(),
        manifest_status: "covered".to_string(),
        source_format: "pnm".to_string(),
        width: 1,
        height: 1,
        components,
        pixels: pixels.to_vec(),
        pnm_path: PathBuf::from(format!("{name}.pnm")),
    }
}

#[test]
fn pnm_and_raster_fixtures_preserve_shape_channels_and_pixels() {
    let root = temp_dir("formats");
    let gray_path = root.join("gray.PGM");
    write_pnm(&gray_path, &[1, 2], 2, 1, 1).expect("write gray PNM");
    let gray = read_source_image(&gray_path).expect("read gray PNM");
    assert_eq!((gray.width, gray.height, gray.components), (2, 1, 1));
    assert_eq!(gray.pixels, [1, 2]);
    assert_eq!(gray.source_command, "source-pnm");

    let rgb_path = root.join("rgb.ppm");
    write_pnm(&rgb_path, &[1, 2, 3, 4, 5, 6], 2, 1, 3).expect("write RGB PNM");
    let rgb = read_pnm(&rgb_path).expect("read RGB PNM");
    assert_eq!((rgb.width, rgb.height, rgb.components), (2, 1, 3));
    assert_eq!(rgb.pixels, [1, 2, 3, 4, 5, 6]);

    let png_path = root.join("gray.png");
    image::GrayImage::from_raw(2, 1, vec![9, 10])
        .expect("gray image")
        .save(&png_path)
        .expect("save gray PNG");
    let png = read_raster_image(&png_path).expect("read gray PNG");
    assert_eq!((png.width, png.height, png.components), (2, 1, 1));
    assert_eq!(png.pixels, [9, 10]);
    assert_eq!(png.source_command, "image-crate-decode-to-pnm");

    let rgb_png_path = root.join("rgb.png");
    image::RgbImage::from_raw(1, 1, vec![7, 8, 9])
        .expect("RGB image")
        .save(&rgb_png_path)
        .expect("save RGB PNG");
    assert_eq!(
        read_source_image(&rgb_png_path)
            .expect("read RGB PNG")
            .pixels,
        [7, 8, 9]
    );

    let rgba_path = root.join("rgba.png");
    image::RgbaImage::from_raw(1, 1, vec![1, 2, 3, 4])
        .expect("RGBA image")
        .save(&rgba_path)
        .expect("save RGBA PNG");
    let error = read_raster_image(&rgba_path)
        .err()
        .expect("alpha is unsupported");
    assert!(error.contains("unsupported source color type Rgba8"));
    assert!(read_raster_image(&root.join("missing.png"))
        .err()
        .expect("missing raster error")
        .contains("open source image"));
}

#[test]
fn generated_and_external_loaders_materialize_valid_sorted_pnm_cases() {
    let root = temp_dir("loaders");
    let generated_dir = root.join("generated");
    fs::create_dir_all(&generated_dir).expect("create generated dir");
    let generated = generated_image_cases(&generated_dir).expect("generated cases");
    assert_eq!(generated.len(), 3);
    assert_eq!(
        (
            generated[0].width,
            generated[0].height,
            generated[0].components
        ),
        (128, 128, 1)
    );
    assert_eq!(
        (
            generated[2].width,
            generated[2].height,
            generated[2].components
        ),
        (512, 512, 3)
    );
    assert!(generated.iter().all(|case| case.pnm_path.is_file()));

    let inputs = root.join("inputs");
    let nested = inputs.join("nested");
    let work = root.join("work");
    fs::create_dir_all(&nested).expect("create input tree");
    fs::create_dir_all(&work).expect("create work dir");
    write_pnm(&inputs.join("b.ppm"), &[1, 2, 3], 1, 1, 3).expect("write RGB input");
    write_pnm(&nested.join("a.pgm"), &[4], 1, 1, 1).expect("write gray input");
    fs::write(inputs.join("ignored.txt"), "not an image").expect("write ignored input");

    let loaded = load_external_image_cases(&inputs, &work, None).expect("load external cases");
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded[0].name, "external_0000_b");
    assert_eq!(loaded[1].name, "external_0001_a");
    assert!(loaded.iter().all(|case| case.pnm_path.is_file()));
    assert!(loaded
        .iter()
        .all(|case| case.manifest_status == "not-covered"));

    let error = load_external_image_cases(&root.join("missing"), &work, None)
        .err()
        .expect("missing input dir");
    assert!(error.contains("entry is not a directory"));
    let empty = root.join("empty");
    fs::create_dir_all(&empty).expect("create empty input dir");
    assert!(load_external_image_cases(&empty, &work, None)
        .err()
        .expect("empty input error")
        .contains("contains no supported source images"));
}

#[test]
fn recursive_inventory_and_extension_classification_are_case_insensitive() {
    let root = temp_dir("inventory");
    let nested = root.join("nested");
    fs::create_dir_all(&nested).expect("create nested dir");
    for name in ["a.PGM", "b.JpEg", "c.TIFF", "d.bmp", "ignored.txt"] {
        fs::write(nested.join(name), []).expect("write inventory fixture");
    }
    let mut paths = Vec::new();
    collect_source_image_paths(&root, &mut paths).expect("collect image paths");
    paths.sort();
    assert_eq!(paths.len(), 4);
    assert!(is_supported_source_image_path(Path::new("image.PNG")));
    assert!(!is_supported_source_image_path(Path::new("image.gif")));
    assert!(is_pnm_path(Path::new("image.PnM")));
    assert!(!is_pnm_path(Path::new("image.png")));
    assert_eq!(source_format_label(Path::new("image.JPEG")), "jpeg");
    assert_eq!(source_format_label(Path::new("image")), "unknown");
    assert_eq!(pnm_extension(1), Ok("pgm"));
    assert_eq!(pnm_extension(3), Ok("ppm"));
    assert!(pnm_extension(4)
        .unwrap_err()
        .contains("unsupported component count 4"));
    assert!(write_pnm(&root.join("bad.pgm"), &[1], 2, 2, 1).is_err());
}

#[test]
fn manifest_parser_handles_complete_optional_duplicate_and_malformed_rows() {
    let root = temp_dir("manifest");
    let image = root.join("fixture.pgm");
    write_pnm(&image, &[42], 1, 1, 1).expect("write manifest image");
    let manifest = root.join("manifest.tsv");
    fs::write(
        &manifest,
        "path\tcorpus_category\tcorpus_name\tlicense_status\tsource_command\tinput_fnv1a64\nfixture.pgm\tnatural-image\tunit-corpus\tcc0\tgenerated\t36a8b778f01d001b\n",
    )
    .expect("write manifest");
    let _manifest_env = EnvGuard::set("J2K_ENCODE_COMPARE_MANIFEST", &manifest);
    let parsed = encode_manifest_from_env()
        .expect("parse manifest")
        .expect("manifest present");
    let canonical = image.canonicalize().expect("canonical image");
    let entry = parsed.entries.get(&canonical).expect("manifest entry");
    assert_eq!(entry.corpus_name, "unit-corpus");
    assert_eq!(entry.license_status, "cc0");

    fs::write(
        &manifest,
        "path\tcorpus_category\nfixture.pgm\tnatural-image\nfixture.pgm\tnatural-image\n",
    )
    .expect("write duplicate manifest");
    assert!(encode_manifest_from_env()
        .err()
        .expect("duplicate manifest error")
        .contains("duplicates path"));
    fs::write(&manifest, "").expect("write empty manifest");
    assert!(encode_manifest_from_env()
        .err()
        .expect("empty manifest error")
        .contains("is empty"));
    fs::write(&manifest, "path\nfixture.pgm\n").expect("write missing column manifest");
    assert!(encode_manifest_from_env()
        .err()
        .expect("missing column error")
        .contains("corpus_category"));

    assert_eq!(manifest_column(&["path", "kind"], "path"), Ok(0));
    assert!(manifest_field(&["only"], 1, "path", 2).is_err());
    assert!(manifest_required_value(&[""], 0, "path", 2).is_err());
    assert_eq!(
        manifest_optional_value(&["value"], None, "optional", 2),
        Ok(None)
    );
}

#[test]
fn external_metadata_checks_pins_and_falls_back_when_uncovered() {
    let root = temp_dir("metadata");
    let path = root.join("kodak-image.pgm");
    write_pnm(&path, &[42], 1, 1, 1).expect("write metadata fixture");
    let image = PnmImage {
        width: 1,
        height: 1,
        components: 1,
        pixels: vec![42],
        source_command: "source-pnm".to_string(),
    };
    let fallback = external_image_metadata(&path, &image, None).expect("fallback metadata");
    assert_eq!(fallback.manifest_status, "not-covered");
    assert_eq!(fallback.corpus_name, "path-inferred");
    assert!(external_source_label(&path)
        .expect("source label")
        .starts_with("external:"));
    assert_eq!(external_corpus_category(&path), "natural-image");

    let canonical = path.canonicalize().expect("canonical fixture");
    let entry = EncodeManifestEntry {
        corpus_category: "medical-domain".to_string(),
        corpus_name: "medical-fixture".to_string(),
        license_status: "cc0".to_string(),
        source_command: "fixture-command".to_string(),
        input_fnv1a64: Some(j2k_test_support::fnv1a64_hex(&image.pixels)),
    };
    let manifest = EncodeManifest {
        entries: HashMap::from([(canonical.clone(), entry)]),
    };
    let covered =
        external_image_metadata(&path, &image, Some(&manifest)).expect("covered metadata");
    assert_eq!(covered.manifest_status, "covered");
    assert_eq!(covered.corpus_category, "medical-domain");

    let unpinned = EncodeManifest {
        entries: HashMap::from([(
            canonical.clone(),
            EncodeManifestEntry {
                corpus_category: "natural-image".to_string(),
                corpus_name: "fixture".to_string(),
                license_status: "cc0".to_string(),
                source_command: "fixture".to_string(),
                input_fnv1a64: None,
            },
        )]),
    };
    assert_eq!(
        external_image_metadata(&path, &image, Some(&unpinned))
            .expect("unpinned metadata")
            .manifest_status,
        "covered-unpinned"
    );

    let mismatched = EncodeManifest {
        entries: HashMap::from([(
            canonical,
            EncodeManifestEntry {
                corpus_category: "natural-image".to_string(),
                corpus_name: "fixture".to_string(),
                license_status: "cc0".to_string(),
                source_command: "fixture".to_string(),
                input_fnv1a64: Some("wrong".to_string()),
            },
        )]),
    };
    assert!(external_image_metadata(&path, &image, Some(&mismatched))
        .err()
        .expect("hash mismatch error")
        .contains("hash mismatch"));
}

#[test]
fn case_selection_and_mixed_batches_require_distinct_external_images() {
    let gray_a = case("gray-a", "external:a", 1, &[1]);
    let gray_b = case("gray-b", "external:b", 1, &[2]);
    let gray_copy = case("gray-copy", "external:c", 1, &[1]);
    let rgb_a = case("rgb-a", "external:d", 3, &[1, 2, 3]);
    let rgb_b = case("rgb-b", "external:e", 3, &[4, 5, 6]);
    let generated = case("generated", "j2k-generated", 1, &[9]);
    let cases = vec![gray_a, gray_b, gray_copy, rgb_a, rgb_b, generated];
    assert_eq!(
        select_cases(cases.clone(), &[]).expect("unfiltered").len(),
        6
    );
    assert_eq!(
        select_cases(cases.clone(), &["rgb-"])
            .expect("filtered")
            .len(),
        2
    );
    assert!(select_cases(cases.clone(), &["missing"])
        .err()
        .expect("missing case selection error")
        .contains("no encode cases matched"));

    let batches = mixed_external_batches(&cases);
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].name, "external_mixed_gray8_encode");
    assert_eq!(batches[0].cases.len(), 3);
    assert_eq!(batches[1].name, "external_mixed_rgb8_encode");
    assert!(mixed_external_batches(&cases[..1]).is_empty());
}
