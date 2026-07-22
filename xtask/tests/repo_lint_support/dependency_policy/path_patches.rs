// SPDX-License-Identifier: MIT OR Apache-2.0

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::Path,
    process::{Command, Stdio},
    sync::atomic::{AtomicU64, Ordering},
};

use super::super::{repo_root, sha256_hex};

struct PatchedFile {
    path: &'static str,
    upstream_sha256: &'static str,
    patched_sha256: &'static str,
}

struct PathPatch {
    name: &'static str,
    directory: &'static str,
    archive_sha256: &'static str,
    tree_sha256: &'static str,
    files: &'static [PatchedFile],
}

const PATH_PATCHES: &[PathPatch] = &[
    PathPatch {
        name: "block",
        directory: "third_party/block-0.1.6-patched",
        archive_sha256: "0d8c1fef690941d3e7788d328517591fecc684c084084702d6ff1641e993699a",
        tree_sha256: "ed11b5084e7c790c36466b1ea4033b9b8b1378739c38346b888d7c55178b3214",
        files: &[PatchedFile {
            path: "src/lib.rs",
            upstream_sha256: "eb31678adf63b53109d9b94eba23699fd5f9ebfdb950f6e1a57ad51bb6a146fa",
            patched_sha256: "bf799f4d01bb497fdcffe7a5e28d998e721ed45c1be866ed1b454df39ce876a9",
        }],
    },
    PathPatch {
        name: "cubecl-cuda",
        directory: "third_party/cubecl-cuda-0.10.0-patched",
        archive_sha256: "b6b0a69ff45688d322ad8e92c8bf645167b9ca490fa8fa087fc6adac8c5e46be",
        tree_sha256: "bc1d7a26f591d748244ac3b18d4d641969d03506dff9d0f1c3ce73e10086166d",
        files: &[
            PatchedFile {
                path: "src/compute/command.rs",
                upstream_sha256: "ea7697c7fb33fd28a598ff05bb2d7ef6f07b21c1ecd68eff57a1680dcd68b797",
                patched_sha256: "d71de50b6035d5b98895e21b0fd6994a389b25240f48812e8a1231a0b6cdbe5c",
            },
            PatchedFile {
                path: "src/compute/server.rs",
                upstream_sha256: "7411535c6ae5c72efac9a85ec42d451a8f7ece4e890c1d4ecf35ff4ff4cd2a1d",
                patched_sha256: "ad81a690415f228c5a43f9972f06a3ae1bc7ba8673e3a545d79e15a6655d5a3c",
            },
        ],
    },
    PathPatch {
        name: "cubecl-runtime",
        directory: "third_party/cubecl-runtime-0.10.0-patched",
        archive_sha256: "b68491bf5b3e997ae36bdc4e63b4ccd6d2f0e86b3b596a5d7a48d2b9e92622a0",
        tree_sha256: "0ae2be14bb10fc27c633c99b0a862dce8b237fc32768fd63eff0b6ad53ef0047",
        files: &[
            PatchedFile {
                path: "src/client.rs",
                upstream_sha256: "dd25c67906e3db68d05d916d45234e4bdf20cd5b65ac1ad2b00a2ddcb520e85a",
                patched_sha256: "2ba96df7c895b975d9e171f3c2452fc61e887091af565384061bbd816d471e1b",
            },
            PatchedFile {
                path: "src/server/base.rs",
                upstream_sha256: "c6292c16d5a73d7a1570776ffd25ec10afc339b942dbc5a693d8c791feb8ef5b",
                patched_sha256: "93ad450ed4aba9c39478be65afd428110118f6014dd080f79ced701612baafe2",
            },
        ],
    },
    PathPatch {
        name: "wgpu",
        directory: "third_party/wgpu-29.0.4-patched",
        archive_sha256: "76e8840e1ba2881d4cbb18d2147627a56af426ff064c0401eb0c8410c6325d07",
        tree_sha256: "c958c8a0308bc8c1b3753f32f5696cc5d15ef50f3dec56b64eb1936d9a22a1f5",
        files: &[
            PatchedFile {
                path: "src/api/buffer.rs",
                upstream_sha256: "c7d2c13d8031a9416d5562ec970a46470da331544d3a5e15bc8fb7dae2a9de32",
                patched_sha256: "19ba97b38b63fd491e480cfcc773fa2055f691a97ee727d7541a7255bbf82ba4",
            },
            PatchedFile {
                path: "src/backend/wgpu_core.rs",
                upstream_sha256: "8c0e1fab257ef93a86f6d6c93f1970f9756dececcaf3cebaa398a90a9d58efa9",
                patched_sha256: "efdb2d9a3351718d29ca0a4a843b1e61cc778dcb5fb46a0c365e9dd79507101c",
            },
        ],
    },
    PathPatch {
        name: "wgpu-core",
        directory: "third_party/wgpu-core-29.0.4-patched",
        archive_sha256: "2f519832254e56965a9940c4af57dcb75f702b6f6fa4a0b172f685395843a4d7",
        tree_sha256: "13631d9cf9230ea52ae5458c440d60ccf23988839b9d89ff48cb026e7c324411",
        files: &[PatchedFile {
            path: "src/device/global.rs",
            upstream_sha256: "8d8554b2400b46b595af68b0d2fab472da06b8c11fa1823299728b5bc2cf2caa",
            patched_sha256: "ceab63ff6129a22e1bea8e9e1e691edb745080541d9c65b3ec043981d0a21978",
        }],
    },
    PathPatch {
        name: "wgpu-hal",
        directory: "third_party/wgpu-hal-29.0.4-patched",
        archive_sha256: "97ace1c17727311c22a46e4e3faf56ea6de81af99dcc839bdfb54857b94d448d",
        tree_sha256: "c6cbbd822bd03017303b9270cad39adc20ebc97b3b0765bd8b088c5957674269",
        files: &[PatchedFile {
            path: "src/metal/mod.rs",
            upstream_sha256: "daed7a2d7c1cd7b9431b3009b5b7feb2ca8e45648ffd7eb0185c46cfcbeb2ada",
            patched_sha256: "ffa9a5a9767b5e458d50001a66bdf904be5a3199adfb213cac033a9765d0e5a6",
        }],
    },
];

static NEXT_PATCH_TREE_TEST_ID: AtomicU64 = AtomicU64::new(0);

#[test]
fn patched_tree_digest_includes_nested_provenance_named_files() {
    let directory = std::env::temp_dir().join(format!(
        "j2k-patched-tree-digest-{}-{}",
        std::process::id(),
        NEXT_PATCH_TREE_TEST_ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(directory.join("nested")).expect("create patched-tree test directory");
    fs::write(directory.join("PATCH_PROVENANCE.md"), "root metadata")
        .expect("write root provenance");
    fs::write(directory.join("src.rs"), "payload").expect("write patched payload");
    let nested = directory.join("nested/PATCH_PROVENANCE.md");
    fs::write(&nested, "first nested payload").expect("write nested provenance-named payload");
    let first = patched_tree_sha256(&directory);

    fs::write(&nested, "changed nested payload").expect("change nested provenance-named payload");
    let second = patched_tree_sha256(&directory);
    fs::remove_dir_all(&directory).expect("remove patched-tree test directory");

    assert_ne!(first, second, "only the root provenance record is excluded");
}

#[test]
fn patched_tree_digest_excludes_generated_root_lockfile() {
    let directory = std::env::temp_dir().join(format!(
        "j2k-patched-tree-lockfile-{}-{}",
        std::process::id(),
        NEXT_PATCH_TREE_TEST_ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&directory).expect("create patched-tree test directory");
    fs::write(directory.join("src.rs"), "payload").expect("write patched payload");
    let without_lockfile = patched_tree_sha256(&directory);

    fs::write(directory.join("Cargo.lock"), "generated lockfile")
        .expect("write generated root lockfile");
    let with_lockfile = patched_tree_sha256(&directory);
    fs::remove_dir_all(&directory).expect("remove patched-tree test directory");

    assert_eq!(
        without_lockfile, with_lockfile,
        "an ignored generated root lockfile is not part of a reproducible patch tree"
    );
}

#[test]
fn all_workspace_path_patches_have_pinned_provenance_and_local_digests() {
    let root = repo_root();
    let workspace = fs::read_to_string(root.join("Cargo.toml")).expect("read workspace manifest");
    let manifest = toml::from_str::<toml::Value>(&workspace).expect("parse workspace manifest");
    let patches = manifest
        .get("patch")
        .and_then(|patch| patch.get("crates-io"))
        .and_then(toml::Value::as_table)
        .expect("workspace [patch.crates-io] table");
    let actual = patches
        .iter()
        .map(|(name, value)| {
            let path = value
                .get("path")
                .and_then(toml::Value::as_str)
                .unwrap_or_else(|| panic!("path patch `{name}` must have a string path"));
            (name.as_str(), path)
        })
        .collect::<BTreeSet<_>>();
    let expected = PATH_PATCHES
        .iter()
        .map(|patch| (patch.name, patch.directory))
        .collect::<BTreeSet<_>>();
    assert_eq!(
        actual, expected,
        "every workspace path patch must be governed"
    );

    let excluded = manifest
        .get("workspace")
        .and_then(|workspace| workspace.get("exclude"))
        .and_then(toml::Value::as_array)
        .expect("workspace.exclude array")
        .iter()
        .filter_map(toml::Value::as_str)
        .collect::<BTreeSet<_>>();
    for patch in PATH_PATCHES {
        assert!(
            excluded.contains(patch.directory),
            "{} must be excluded from workspace membership",
            patch.directory
        );
        let directory = root.join(patch.directory);
        let provenance = fs::read_to_string(directory.join("PATCH_PROVENANCE.md"))
            .unwrap_or_else(|error| panic!("read {} provenance: {error}", patch.name));
        for digest in [patch.archive_sha256, patch.tree_sha256] {
            assert!(
                provenance.contains(digest),
                "{} provenance must pin SHA-256 {digest}",
                patch.name
            );
        }
        assert_eq!(
            patched_tree_sha256(&directory),
            patch.tree_sha256,
            "{} patched tree digest changed",
            patch.name
        );
        for file in patch.files {
            assert!(provenance.contains(file.path));
            assert!(provenance.contains(file.upstream_sha256));
            assert!(provenance.contains(file.patched_sha256));
            assert_eq!(sha256_hex(&directory.join(file.path)), file.patched_sha256);
        }
    }

    let governed_directories = expected
        .iter()
        .map(|(_, path)| *path)
        .collect::<BTreeSet<_>>();
    let discovered_directories = fs::read_dir(root.join("third_party"))
        .expect("read third_party")
        .filter_map(Result::ok)
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| {
            let name = entry.file_name();
            name.to_str()
                .filter(|name| name.ends_with("-patched"))
                .map(|name| format!("third_party/{name}"))
        })
        .collect::<BTreeSet<_>>();
    assert_eq!(
        discovered_directories,
        governed_directories
            .into_iter()
            .map(str::to_owned)
            .collect::<BTreeSet<_>>(),
        "every patched third-party directory must be governed"
    );
}

fn patched_tree_sha256(directory: &Path) -> String {
    let mut pending = vec![directory.to_path_buf()];
    let mut files = BTreeMap::new();
    while let Some(current) = pending.pop() {
        for entry in fs::read_dir(&current)
            .unwrap_or_else(|error| panic!("read {}: {error}", current.display()))
        {
            let path = entry.expect("read patched tree entry").path();
            if path.is_dir() {
                pending.push(path);
            } else if path != directory.join("PATCH_PROVENANCE.md")
                && path != directory.join("Cargo.lock")
            {
                let relative = path
                    .strip_prefix(directory)
                    .expect("patched file below patch directory")
                    .to_string_lossy()
                    .replace('\\', "/");
                assert!(
                    files.insert(relative, path).is_none(),
                    "patched tree contains a duplicate relative path"
                );
            }
        }
    }
    let mut inventory = String::new();
    for (relative, path) in files {
        inventory.push_str(&sha256_hex(&path));
        inventory.push_str("  ");
        inventory.push_str(&relative);
        inventory.push('\n');
    }
    sha256_bytes(inventory.as_bytes())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    for (program, args) in [("sha256sum", &[][..]), ("shasum", &["-a", "256"][..])] {
        let child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn();
        let Ok(mut child) = child else {
            continue;
        };
        child
            .stdin
            .take()
            .expect("SHA-256 stdin")
            .write_all(bytes)
            .expect("write SHA-256 input");
        let output = child.wait_with_output().expect("wait for SHA-256 tool");
        assert!(output.status.success(), "{program} failed");
        return String::from_utf8_lossy(&output.stdout)
            .split_whitespace()
            .next()
            .expect("SHA-256 output")
            .to_owned();
    }
    panic!("no SHA-256 command is available")
}
