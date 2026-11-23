#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Build a local j2k-c C/C++ package. Never publish or overwrite an existing package."""

import argparse
from collections import namedtuple
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

CRATE = Path(__file__).resolve().parents[1]
ROOT = CRATE.parents[1]
ABI_SOURCE = CRATE / "src" / "abi.rs"
ABI_HEADER = CRATE / "include" / "j2k.h"
# Cargo may color stderr (CARGO_TERM_COLOR=always is set on GitHub runners);
# a trailing reset would otherwise become part of the last native library flag.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]")
# The workspace's own license texts, for members without a crate-local copy.
WORKSPACE_LICENSES = ("LICENSE", "LICENSE-MIT", "LICENSE-APACHE")
# Cargo's file name, the packaged real file, its development symlink, and the
# runtime name recorded by consumers (ELF SONAME or Mach-O install name).
Library = namedtuple("Library", "built packaged link soname linker_args")


def run(*args, env=None):
    """Run a command in the workspace; a nonzero exit raises CalledProcessError."""
    subprocess.run(args, cwd=ROOT, check=True, text=True, env=env)


def run_and_capture_stderr(*args, env=None):
    """Run a command in the workspace and return stderr; a nonzero exit raises CalledProcessError."""
    proc = subprocess.run(args, cwd=ROOT, text=True, stderr=subprocess.PIPE, env=env)
    sys.stderr.write(proc.stderr)
    sys.stderr.flush()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, args, output=proc.stdout, stderr=proc.stderr)
    return proc.stderr


def parse_native_static_libs(stderr):
    """Parse the native-static-libs line emitted by rustc."""
    # Strip ANSI escapes before parsing: colored Cargo output would otherwise
    # leave a reset sequence attached to the final library flag.
    stderr = ANSI_ESCAPE.sub("", stderr)
    match = re.search(r"native-static-libs:[ \t]*([^\r\n]*)", stderr)
    if not match:
        raise RuntimeError("Could not find 'native-static-libs:' in cargo rustc output")
    libs = match.group(1).strip()
    if not libs:
        raise RuntimeError("'native-static-libs:' output was empty")
    if CONTROL_CHARACTERS.search(libs):
        raise RuntimeError("'native-static-libs:' output contains control characters")
    return libs


def command_output(*args):
    """Return a command's standard output; a nonzero exit raises CalledProcessError."""
    stdout = subprocess.run(args, cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE).stdout
    if not isinstance(stdout, str) or not stdout.strip():
        raise RuntimeError(f"{' '.join(args)} produced no output")
    return stdout


def abi_version():
    """Read the C ABI revision that versions the shared library name."""
    rust = re.findall(r"^pub const J2K_ABI_VERSION: u32 = (\d+);$",
                      ABI_SOURCE.read_text(encoding="utf-8"), re.MULTILINE)
    header = re.findall(r"^#define J2K_ABI_VERSION UINT32_C\((\d+)\)$",
                        ABI_HEADER.read_text(encoding="utf-8"), re.MULTILINE)
    if len(rust) != 1 or rust != header:
        raise RuntimeError(f"J2K_ABI_VERSION must be defined once and agree in "
                           f"{ABI_SOURCE.relative_to(ROOT)} and {ABI_HEADER.relative_to(ROOT)}")
    return int(rust[0])


def macho_version(version, limits=(65535, 255, 255)):
    """Encode an X.Y.Z version for ld64; SemVer pre-release/build suffixes are not representable."""
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)(?:[-+].*)?", version)
    if not match or any(int(part) > limit for part, limit in zip(match.groups(), limits)):
        raise RuntimeError(f"Version {version} cannot be encoded as a Mach-O dylib version")
    return ".".join(str(int(part)) for part in match.groups())


def library_for(host, version, abi):
    """Name the library by ABI revision; macOS and GNU/Linux use different linker contracts."""
    if "apple-darwin" in host:
        packaged = f"libj2k.{abi}.dylib"
        soname = f"@rpath/{packaged}"
        # Cargo names the build `libj2k_c` (crate `j2k-c`); the package ships `libj2k`.
        return Library("libj2k_c.dylib", packaged, "libj2k.dylib", soname, [
            "-C", f"link-arg=-Wl,-install_name,{soname}",
            "-C", f"link-arg=-Wl,-compatibility_version,{macho_version(f'{abi}.0.0')}",
            "-C", f"link-arg=-Wl,-current_version,{macho_version(version)}"])
    if "unknown-linux-gnu" in host:
        packaged = f"libj2k.so.{abi}"
        return Library("libj2k_c.so", packaged, "libj2k.so", packaged,
                       ["-C", f"link-arg=-Wl,-soname,{packaged}"])
    raise RuntimeError(f"Packaging is currently implemented for macOS and GNU/Linux hosts, not {host}")


def cargo_home():
    return Path(os.environ.get("CARGO_HOME") or Path.home() / ".cargo").absolute()


def user_rustflags():
    """Return the rustflags Cargo would read from the environment.

    Packaging passes CARGO_ENCODED_RUSTFLAGS, which makes Cargo ignore config
    rustflags. Refuse those sources rather than silently dropping them.
    """
    if "CARGO_ENCODED_RUSTFLAGS" in os.environ:
        return [flag for flag in os.environ["CARGO_ENCODED_RUSTFLAGS"].split("\x1f") if flag]
    if "RUSTFLAGS" in os.environ:
        return [flag.strip() for flag in os.environ["RUSTFLAGS"].split(" ") if flag.strip()]
    sources = [name for name in os.environ if name == "CARGO_BUILD_RUSTFLAGS"
               or (name.startswith("CARGO_TARGET_") and name.endswith("_RUSTFLAGS"))]
    configs = [directory / ".cargo" / name for directory in (ROOT, *ROOT.parents)
               for name in ("config", "config.toml")]
    configs += [cargo_home() / "config", cargo_home() / "config.toml"]
    key = re.compile(r"^[ \t]*(?:[^#\s][^\n]*\.)?rustflags[ \t]*=", re.MULTILINE)
    sources += [str(path) for path in configs
                if path.is_file() and key.search(path.read_text(encoding="utf-8"))]
    if sources:
        raise RuntimeError("Packaging sets CARGO_ENCODED_RUSTFLAGS, which would override rustflags from "
                           + ", ".join(sources) + "; pass those flags through RUSTFLAGS instead")
    return []


def path_remaps(sysroot, target_dir):
    """Map machine-local prefixes to fixed virtual paths; the most specific prefix wins."""
    remaps = {}
    for path, virtual in ((sysroot, "/rust-sysroot"), (cargo_home(), "/cargo-home"),
                          (ROOT, "/j2k"), (target_dir, "/cargo-target")):
        for form in (path.absolute(), path.resolve()):
            remaps[str(form)] = virtual
    # rustc applies the last matching --remap-path-prefix.
    return dict(sorted(remaps.items(), key=lambda item: len(item[0])))


def require_no_build_paths(library, remaps):
    """Fail rather than package a binary that still embeds a local build prefix."""
    data = library.read_bytes()
    # Single-component prefixes such as a container's /src occur in remapped paths.
    leaked = [prefix for prefix in remaps
              if len(Path(prefix).parts) > 2 and f"{prefix}/".encode() in data]
    if leaked:
        raise RuntimeError(f"{library} embeds local build paths: {', '.join(leaked)}")


def linked_packages(metadata, root_id):
    """The packages the library links or builds with: its normal and build
    dependency closure, without development-only dependencies."""
    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    pending, seen = [root_id], {root_id}
    while pending:
        for dependency in nodes[pending.pop()]["deps"]:
            kinds = {kind["kind"] for kind in dependency["dep_kinds"]}
            if kinds & {None, "build"} and dependency["pkg"] not in seen:
                seen.add(dependency["pkg"])
                pending.append(dependency["pkg"])
    return [package for package in metadata["packages"] if package["id"] in seen - {root_id}]


def collect_notices(packages, workspace_members, destination):
    """Include upstream notice texts, preserving package-relative paths."""
    entries = []
    for package in sorted(packages, key=lambda item: item["name"]):
        source = Path(package["manifest_path"]).parent
        files = {path for path in source.rglob("*") if path.is_file()
                 and path.name.upper().startswith(("LICENSE", "LICENCE", "NOTICE", "COPYING"))}
        if package.get("license_file"):
            files.add(source / package["license_file"])
        # Workspace members share the repository's license texts.
        if not files and package["id"] in workspace_members:
            source = ROOT
            files = {ROOT / name for name in WORKSPACE_LICENSES}
        if not files:
            raise RuntimeError(f"No license/notice text found for {package['name']}")
        relative = Path(f"{package['name']}-{package['version']}")
        for path in sorted(files):
            target = destination / relative / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        entries.append(f"- {package['name']} {package['version']}: {package.get('license') or 'see license file'}")
    (destination / "README.md").write_text(
        "# Third-party licenses\n\n"
        "Notices for the locked Cargo packages the library links or builds with, "
        "including build-time tools. Inclusion does not imply that every tool is "
        "linked into the library. Original licenses remain in effect.\n\n"
        + "\n".join(entries) + "\n", encoding="utf-8")


def stage_static_archive(built_static, output):
    """Copy the built static archive into the package's lib/ directory."""
    if not built_static.is_file():
        raise RuntimeError(f"Expected static archive was not built: {built_static}")
    destination = output / "lib" / "libj2k.a"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_static, destination)
    return destination


def render_cmake_config(template, library, static_libs):
    """Render J2KConfig.cmake from its template and reject unexpanded placeholders."""
    config = (template
              .replace("@LIBRARY_FILE@", library.packaged)
              .replace("@LIBRARY_SONAME@", library.soname)
              .replace("@STATIC_LIBS@", static_libs))
    if re.search(r"@[A-Z_]+@", config):
        raise RuntimeError("Unexpanded placeholder in crates/j2k-c/cmake/J2KConfig.cmake.in")
    return config


def render_pkgconfig(template, version, url, private_libs):
    """Render j2k.pc from its template and reject empty private libs or unexpanded placeholders."""
    if not private_libs or not private_libs.strip():
        raise RuntimeError("Derived PRIVATE_LIBS is empty")
    pkgconfig = (template
                 .replace("@VERSION@", version)
                 .replace("@URL@", url or "")
                 .replace("@PRIVATE_LIBS@", private_libs))
    if re.search(r"@[A-Z_]+@", pkgconfig):
        raise RuntimeError("Unexpanded placeholder in crates/j2k-c/pkgconfig/j2k.pc.in")
    return pkgconfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New output directory; defaults to dist/version-host")
    args = parser.parse_args()
    host = next(line.removeprefix("host: ") for line in command_output("rustc", "-vV").splitlines()
                if line.startswith("host: "))

    sysroot = Path(command_output("rustc", "--print", "sysroot").strip())
    metadata = json.loads(command_output("cargo", "metadata", "--locked", "--format-version", "1",
                                         "--filter-platform", host))
    package = next(item for item in metadata["packages"] if item["name"] == "j2k-c")
    version = package["version"]
    library = library_for(host, version, abi_version())
    output = (args.output or ROOT / "dist" / f"j2k-c-{version}-{host}").resolve()
    if output.exists():
        raise RuntimeError(f"Output already exists; choose a new directory: {output}")

    # target_directory honors CARGO_TARGET_DIR and Cargo's build.target-dir.
    target_dir = Path(metadata["target_directory"])
    remaps = path_remaps(sysroot, target_dir)
    env = {name: value for name, value in os.environ.items() if name != "RUSTFLAGS"}
    env["CARGO_TERM_COLOR"] = "never"
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(
        user_rustflags() + [f"--remap-path-prefix={source}={virtual}" for source, virtual in remaps.items()])
    # Keep only exported/dynamic symbols in the distributed library.
    stderr = run_and_capture_stderr("cargo", "rustc", "--locked", "--release", "-p", "j2k-c", "--lib",
                                    "--target", host, "--", "-C", "strip=symbols",
                                    "--print", "native-static-libs",
                                    *library.linker_args, env=env)
    private_libs = parse_native_static_libs(stderr)
    built = target_dir / host / "release" / library.built
    built_static = target_dir / host / "release" / "libj2k_c.a"
    if not built.is_file():
        raise RuntimeError(f"Expected library was not built: {built}")
    if not built_static.is_file():
        raise RuntimeError(f"Expected static archive was not built: {built_static}")
    require_no_build_paths(built_static, remaps)
    require_no_build_paths(built, remaps)

    output.mkdir(parents=True)
    (output / "lib" / "cmake" / "J2K").mkdir(parents=True)
    shutil.copy2(built, output / "lib" / library.packaged)
    (output / "lib" / library.link).symlink_to(library.packaged)
    stage_static_archive(built_static, output)
    for source, directory in (("include", "include"), ("native/examples", "examples"), ("docs", "docs")):
        shutil.copytree(CRATE / source, output / directory)
    quickstart = (CRATE / "docs" / "quickstart.md").read_text(encoding="utf-8")
    (output / "README.md").write_text(
        quickstart.replace("](api.md)", "](docs/api.md)").replace("](supported.md)", "](docs/supported.md)"),
        encoding="utf-8")
    for name in (*WORKSPACE_LICENSES, "NOTICES.md"):
        shutil.copy2(ROOT / name, output / name)
    shutil.copy2(CRATE / "CHANGELOG.md", output / "CHANGELOG.md")
    config = render_cmake_config(
        (CRATE / "cmake" / "J2KConfig.cmake.in").read_text(encoding="utf-8"),
        library,
        private_libs,
    )
    config_dir = output / "lib" / "cmake" / "J2K"
    (config_dir / "J2KConfig.cmake").write_text(config, encoding="utf-8")
    pkgconfig_dir = output / "lib" / "pkgconfig"
    pkgconfig_dir.mkdir()
    pkgconfig = render_pkgconfig(
        (CRATE / "pkgconfig" / "j2k.pc.in").read_text(encoding="utf-8"),
        version,
        package["repository"],
        private_libs,
    )
    (pkgconfig_dir / "j2k.pc").write_text(pkgconfig, encoding="utf-8")
    (config_dir / "J2KConfigVersion.cmake").write_text(
        f'set(PACKAGE_VERSION "{version}")\n'
        'if(PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)\n'
        '  set(PACKAGE_VERSION_COMPATIBLE TRUE)\n'
        '  set(PACKAGE_VERSION_EXACT TRUE)\n'
        'endif()\n', encoding="utf-8")
    collect_notices(linked_packages(metadata, package["id"]), set(metadata["workspace_members"]),
                    output / "licenses")
    rust_notices = sysroot / "share" / "doc" / "rust"
    runtime_notices = output / "licenses" / "rust-runtime"
    runtime_notices.mkdir()
    shutil.copy2(rust_notices / "COPYRIGHT-library.html", runtime_notices)
    shutil.copytree(rust_notices / "licenses", runtime_notices / "licenses")
    print(f"Package: {output}")
    codec = next(item["version"] for item in metadata["packages"] if item["name"] == "j2k")
    print(f"Host: {host}; codec: j2k {codec}; package: {version}; "
          f"library: lib/{library.packaged} ({library.soname}); static: lib/libj2k.a")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"package: {error}", file=sys.stderr)
        sys.exit(1)
