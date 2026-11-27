#!/usr/bin/env python3
# SPDX-License-Identifier: MIT OR Apache-2.0
"""Build a local j2k-c C/C++ package. Never publish or overwrite an existing package."""

import argparse
from collections import namedtuple
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile

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
# Cargo's file name; the packaged file, its development symlink and its import
# library, as package-relative paths; the runtime name recorded by consumers
# (ELF SONAME or Mach-O install name); and extra linker arguments.
Library = namedtuple("Library", "built packaged link implib soname linker_args")
# What a build produced: the shared library, its Windows import library, and
# on macOS and GNU/Linux the static archive and the native libraries it needs.
Build = namedtuple("Build", "library implib static_archive static_libs")
ARCHIVE_FORMATS = ("tar.gz", "zip")
# A package name is one path component: the archive's file name and top directory.
PACKAGE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")


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
    """Name the library; macOS, GNU/Linux and Windows use different linker contracts.

    The Unix names carry the ABI revision. Cargo names the build `j2k_c`
    (crate `j2k-c`); the package ships `j2k`.
    """
    if "apple-darwin" in host:
        packaged = f"libj2k.{abi}.dylib"
        soname = f"@rpath/{packaged}"
        return Library("libj2k_c.dylib", f"lib/{packaged}", "lib/libj2k.dylib", None, soname, [
            "-C", f"link-arg=-Wl,-install_name,{soname}",
            "-C", f"link-arg=-Wl,-compatibility_version,{macho_version(f'{abi}.0.0')}",
            "-C", f"link-arg=-Wl,-current_version,{macho_version(version)}"])
    if "unknown-linux-gnu" in host:
        packaged = f"libj2k.so.{abi}"
        return Library("libj2k_c.so", f"lib/{packaged}", "lib/libj2k.so", None, packaged,
                       ["-C", f"link-arg=-Wl,-soname,{packaged}"])
    if host == "x86_64-pc-windows-msvc":
        # Programs load the DLL named in their import library, so packaging
        # writes a new import library for the renamed DLL (windows_import_library).
        return Library("j2k_c.dll", "bin/j2k.dll", None, "lib/j2k.dll.lib", None, [])
    raise RuntimeError("Packaging is implemented for macOS, GNU/Linux and x86-64 Windows (MSVC) "
                       f"hosts, not {host}")


def header_functions(header=None):
    """Return the functions j2k.h declares, which must be exactly the library's exports."""
    text = header if header is not None else ABI_HEADER.read_text(encoding="utf-8")
    names = re.findall(r"^J2K_API\b[^;(]*?\b(j2k_\w+)\s*\(", text, re.MULTILINE)
    if not names or len(set(names)) != len(names):
        raise RuntimeError(f"{ABI_HEADER.relative_to(ROOT)} must declare each J2K_API function once")
    return names


def dll_exports(dumpbin_output):
    """Parse the export names from `dumpbin /EXPORTS` output."""
    return re.findall(r"^\s+\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]{8}\s+(\S+)\s*$",
                      dumpbin_output, re.MULTILINE)


def module_definition(dll_name, functions):
    """A module-definition file whose import library binds programs to dll_name."""
    return f'LIBRARY "{dll_name}"\nEXPORTS\n' + "".join(f"    {name}\n" for name in functions)


def msvc_version(path):
    """The MSVC toolset version in .../VC/Tools/MSVC/<version>/bin/HostX/Y/lib.exe."""
    return tuple(int(part) for part in re.findall(r"\d+", path.parents[3].name))


def msvc_tools():
    """Locate lib.exe and dumpbin.exe: on a developer prompt's PATH, or in the
    newest Visual Studio with the x64 C++ tools."""
    on_path = shutil.which("lib.exe")
    if on_path:
        directory = Path(on_path).parent
    else:
        program_files = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
        if not vswhere.is_file():
            raise RuntimeError("lib.exe is not on PATH and vswhere.exe is missing; "
                               "run from a Visual Studio developer prompt")
        found = command_output(str(vswhere), "-latest", "-products", "*", "-requires",
                               "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-find",
                               r"VC\Tools\MSVC\**\bin\Hostx64\x64\lib.exe")
        directory = max((Path(line.strip()) for line in found.splitlines() if line.strip()),
                        key=msvc_version).parent
    tools = directory / "lib.exe", directory / "dumpbin.exe"
    missing = [str(tool) for tool in tools if not tool.is_file()]
    if missing:
        raise RuntimeError(f"MSVC tools not found: {', '.join(missing)}")
    return tools


def windows_import_library(dll, dll_name, scratch):
    """Write an import library that binds programs to dll_name; return its path.

    Cargo's import library names the DLL it built (j2k_c.dll), so the renamed
    DLL needs a new one. The DLL must export exactly the functions j2k.h declares.
    """
    lib, dumpbin = msvc_tools()
    exported = dll_exports(command_output(str(dumpbin), "/NOLOGO", "/EXPORTS", str(dll)))
    declared = header_functions()
    if sorted(exported) != sorted(declared):
        raise RuntimeError(f"{dll.name} exports differ from {ABI_HEADER.relative_to(ROOT)}: "
                           f"missing {sorted(set(declared) - set(exported))}, "
                           f"unexpected {sorted(set(exported) - set(declared))}")
    scratch.mkdir(parents=True, exist_ok=True)
    definition = scratch / "j2k.def"
    definition.write_text(module_definition(dll_name, declared), encoding="utf-8")
    implib = scratch / f"{dll_name}.lib"
    run(str(lib), "/NOLOGO", "/MACHINE:X64", f"/DEF:{definition}", f"/OUT:{implib}")
    if not implib.is_file():
        raise RuntimeError(f"lib.exe did not write {implib}")
    return implib


def render_cmake_config(template, library, static_libs):
    """Expand the CMake package template for one platform's library layout.

    static_libs are the static archive's native link libraries; they are unused
    where the package has no static archive (Windows)."""
    if library.implib:
        note = "The DLL is in bin/; programs link its import library in lib/."
        link_property = f'IMPORTED_IMPLIB "${{_j2k_prefix}}/{library.implib}"'
    else:
        note = "The real file is named by C ABI revision; lib/ also holds an unversioned symlink."
        link_property = f'IMPORTED_SONAME "{library.soname}"'
    config = (template.replace("@LIBRARY_NOTE@", note)
              .replace("@LIBRARY_FILE@", library.packaged)
              .replace("@LINK_PROPERTY@", link_property)
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


def stage_static_archive(built_static, output):
    """Copy the built static archive into the package's lib/ directory."""
    if not built_static.is_file():
        raise RuntimeError(f"Expected static archive was not built: {built_static}")
    destination = output / "lib" / "libj2k.a"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_static, destination)
    return destination


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
    leaked = [prefix for prefix in remaps if len(Path(prefix).parts) > 2
              and any(f"{prefix}{separator}".encode() in data for separator in ("/", "\\"))]
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


def archive_paths(output, name, archive_format):
    """The archive and its checksum file, written beside the package directory."""
    archive = output.parent / f"{name}.{archive_format}"
    return archive, archive.parent / f"{archive.name}.sha256"


def package_entries(package):
    """The package directory and everything in it, each directory before its contents."""
    return [package, *sorted(package.rglob("*"), key=lambda path: path.relative_to(package).parts)]


def write_tar_gz(package, name, archive):
    """Archive the package under top directory `name`, with symlinks kept and
    ownership cleared."""
    with open(archive, "xb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw) as compressed, \
            tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in package_entries(package):
            info = tar.gettarinfo(str(path), arcname="/".join((name, *path.relative_to(package).parts)))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            if info.isreg():
                with open(path, "rb") as handle:
                    tar.addfile(info, handle)
            elif info.isdir() or info.issym():
                tar.addfile(info)
            else:
                raise RuntimeError(f"Cannot archive {path}: not a file, directory or symlink")


def write_zip(package, name, archive):
    """Archive the package under top directory `name`; zip has no portable symlinks."""
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in package_entries(package):
            mode = path.lstat().st_mode
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise RuntimeError(f"Cannot archive {path} in a zip: not a regular file or directory")
            zipped.write(path, "/".join((name, *path.relative_to(package).parts)))


def sha256_file(path):
    """Hex SHA-256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as binary:
        while True:
            block = binary.read(1 << 20)
            if not block:
                return digest.hexdigest()
            digest.update(block)


def write_archive(package, name, archive_format, archive, checksum):
    """Write the archive and a `sha256sum -c` file naming it; return the digest."""
    (write_tar_gz if archive_format == "tar.gz" else write_zip)(package, name, archive)
    digest = sha256_file(archive)
    with open(checksum, "x", encoding="ascii", newline="\n") as text:
        text.write(f"{digest}  {archive.name}\n")
    return digest


def build_library(host, library, sysroot, target_dir):
    """Build the stripped shared library and, on macOS and GNU/Linux, the static
    archive; return a Build.

    target_dir is Cargo's, which honors CARGO_TARGET_DIR and build.target-dir.
    """
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
    built = target_dir / host / "release" / library.built
    if not built.is_file():
        raise RuntimeError(f"Expected library was not built: {built}")
    require_no_build_paths(built, remaps)
    if library.implib:
        scratch = target_dir / host / "release" / "j2k-package"
        return Build(built, windows_import_library(built, Path(library.packaged).name, scratch), None, None)
    built_static = target_dir / host / "release" / "libj2k_c.a"
    if not built_static.is_file():
        raise RuntimeError(f"Expected static archive was not built: {built_static}")
    require_no_build_paths(built_static, remaps)
    return Build(built, None, built_static, parse_native_static_libs(stderr))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        help="New package directory; defaults to dist/NAME")
    parser.add_argument("--name", help="Package name: the archive's file name and top directory; "
                        "defaults to the --output directory's name, or j2k-c-VERSION-HOST")
    parser.add_argument("--archive", choices=ARCHIVE_FORMATS,
                        help="Also write NAME.FORMAT and NAME.FORMAT.sha256 beside the package directory")
    args = parser.parse_args()
    if args.name is not None and not PACKAGE_NAME.fullmatch(args.name):
        parser.error(f"--name must be one file name of letters, digits, '.', '_', '+' or '-': {args.name!r}")
    return args


def main():
    args = parse_args()
    host = next(line.removeprefix("host: ") for line in command_output("rustc", "-vV").splitlines()
                if line.startswith("host: "))

    sysroot = Path(command_output("rustc", "--print", "sysroot").strip())
    metadata = json.loads(command_output("cargo", "metadata", "--locked", "--format-version", "1",
                                         "--filter-platform", host))
    package = next(item for item in metadata["packages"] if item["name"] == "j2k-c")
    version = package["version"]
    library = library_for(host, version, abi_version())
    if args.output:
        output = args.output.resolve()
        name = args.name or output.name
    else:
        name = args.name or f"j2k-c-{version}-{host}"
        output = (ROOT / "dist" / name).resolve()
    archive = archive_paths(output, name, args.archive) if args.archive else ()
    for path in (output, *archive):
        if path.exists() or path.is_symlink():
            raise RuntimeError(f"Output already exists; choose a new name or directory: {path}")

    build = build_library(host, library, sysroot, Path(metadata["target_directory"]))

    output.mkdir(parents=True)
    (output / "lib" / "cmake" / "J2K").mkdir(parents=True)
    (output / library.packaged).parent.mkdir(exist_ok=True)
    shutil.copy2(build.library, output / library.packaged)
    if library.link:
        (output / library.link).symlink_to(Path(library.packaged).name)
    if build.implib:
        shutil.copy2(build.implib, output / library.implib)
    if build.static_archive:
        stage_static_archive(build.static_archive, output)
    for source, directory in (("include", "include"), ("native/examples", "examples"), ("docs", "docs")):
        shutil.copytree(CRATE / source, output / directory)
    quickstart = (CRATE / "docs" / "quickstart.md").read_text(encoding="utf-8")
    (output / "README.md").write_text(
        re.sub(r"\]\((api|supported)\.md(#[^)]*)?\)", r"](docs/\1.md\2)", quickstart),
        encoding="utf-8")
    for license_name in (*WORKSPACE_LICENSES, "NOTICES.md"):
        shutil.copy2(ROOT / license_name, output / license_name)
    config_dir = output / "lib" / "cmake" / "J2K"
    (config_dir / "J2KConfig.cmake").write_text(
        render_cmake_config((CRATE / "cmake" / "J2KConfig.cmake.in").read_text(encoding="utf-8"),
                            library, build.static_libs or ""),
        encoding="utf-8")
    # MSVC links the import library by file name; pkg-config serves the Unix linkers.
    if not library.implib:
        pkgconfig_dir = output / "lib" / "pkgconfig"
        pkgconfig_dir.mkdir()
        pkgconfig = render_pkgconfig((CRATE / "pkgconfig" / "j2k.pc.in").read_text(encoding="utf-8"),
                                     version, package["repository"], build.static_libs)
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
    runtime_name = library.soname or Path(library.packaged).name
    static = "; static: lib/libj2k.a" if build.static_archive else ""
    print(f"Host: {host}; codec: j2k {codec}; package: {version}; "
          f"library: {library.packaged} ({runtime_name}){static}")
    if archive:
        digest = write_archive(output, name, args.archive, *archive)
        print(f"Archive: {archive[0]}\nSHA-256: {digest}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"package: {error}", file=sys.stderr)
        sys.exit(1)
