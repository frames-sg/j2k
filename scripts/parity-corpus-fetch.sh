#!/usr/bin/env bash
set -euo pipefail

manifest="${SIGNINUM_PARITY_CORPUS_MANIFEST:-${1:-corpus/wsi-samples/manifest.json}}"
out_dir="${SIGNINUM_PARITY_CORPUS_DIR:-${2:-corpus/wsi-samples}}"

python3 - "$manifest" "$out_dir" <<'PY'
import hashlib
import json
import os
import pathlib
import sys
import urllib.parse
import urllib.request

MAX_DOWNLOAD_BYTES = int(os.environ.get("SIGNINUM_PARITY_CORPUS_MAX_BYTES", str(512 * 1024 * 1024)))


def fail(message):
    print(f"parity-corpus-fetch: {message}", file=sys.stderr)
    sys.exit(1)


def entry_value(entry, *keys):
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def safe_relative_path(raw, url):
    if raw:
        if "\\" in raw:
            fail(f"unsafe output path {raw!r}")
        path = pathlib.PurePosixPath(raw)
    else:
        parsed = urllib.parse.urlparse(url)
        name = pathlib.PurePosixPath(parsed.path).name
        if not name:
            fail(f"entry {url!r} has no filename; add path, file, or filename")
        path = pathlib.PurePosixPath(name)

    if path.is_absolute() or ".." in path.parts:
        fail(f"unsafe output path {str(path)!r}")
    return pathlib.Path(*path.parts)


def safe_destination(root, relative):
    destination = (root / relative).resolve()
    try:
        destination.relative_to(root)
    except ValueError:
        fail(f"unsafe output path {str(relative)!r}")
    return destination


def validate_url(url):
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        fail(f"unsafe download URL {url!r}: only https URLs are allowed")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest_path = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
out_root = out_dir.resolve()

if not manifest_path.exists():
    fail(f"manifest not found: {manifest_path}")

with manifest_path.open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)

entries = manifest.get("fixtures") or manifest.get("samples")
if not isinstance(entries, list):
    fail("manifest must contain a fixtures or samples list")

out_dir.mkdir(parents=True, exist_ok=True)

for entry in entries:
    if not isinstance(entry, dict):
        fail("manifest entries must be objects")

    url = entry_value(entry, "url", "source_url", "download_url")
    expected = entry_value(entry, "sha256", "sha-256")
    if not url or not expected:
        fail("each entry must contain url and sha256")
    validate_url(url)

    relative = safe_relative_path(entry_value(entry, "path", "file", "filename") or "", url)
    destination = safe_destination(out_root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and sha256_file(destination).lower() == expected.lower():
        print(f"ok {destination}")
        continue

    tmp = destination.with_suffix(destination.suffix + ".part")
    print(f"fetch {url} -> {destination}")
    with urllib.request.urlopen(url, timeout=60) as response:
        total = 0
        with tmp.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    tmp.unlink(missing_ok=True)
                    fail(
                        f"download for {destination} exceeds {MAX_DOWNLOAD_BYTES} byte limit"
                    )
                handle.write(chunk)

    actual = sha256_file(tmp)
    if actual.lower() != expected.lower():
        tmp.unlink(missing_ok=True)
        fail(f"sha256 mismatch for {destination}: expected {expected}, got {actual}")

    os.replace(tmp, destination)
    print(f"ok {destination}")
PY
