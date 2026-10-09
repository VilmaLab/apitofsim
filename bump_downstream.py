#!/usr/bin/env python3
"""Point the apitofsim pins of one downstream file at a GitHub release.

Takes the wheels and their sha256 digests from the release page and rewrites the
pins in a single file, `pyproject.toml` in the current directory by default. A
`.toml` file counts as a uv project, so the `uv.lock` next to it is rewritten as
well and `uv lock --check` verifies the result; a `.yaml` file is a conda
environment.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

DOWNLOAD = "https://github.com/VilmaLab/apitofsim/releases/download"
ARCHIVE = "https://github.com/VilmaLab/apitofsim/archive/refs/tags"
API = "https://api.github.com/repos/VilmaLab/apitofsim/releases/tags/{tag}"

# A pinned wheel URL, e.g. .../download/v0.0.18/apitofsim-0.0.18-<suffix>.whl.
WHEEL = re.compile(rf'{DOWNLOAD}/[^/"\s]+/(?P<asset>apitofsim-[^/"\s]+\.whl)')
# The commented-out sdist fallback, which GitHub generates from the tag.
SDIST = re.compile(rf"{ARCHIVE}/[^/\"\s]+\.tar\.gz")
# uv.lock records the sha256 of every wheel next to its URL.
HASH = re.compile(
    rf'"(?P<url>{DOWNLOAD}/[^"\s]+)"(?P<mid>, hash = "sha256:)[0-9a-f]{{64}}"'
)
# uv.lock records the version on its own line and inline with the source.
VERSION = re.compile(r'(name = "apitofsim",?\s+version = ")[^"]+(")')
# A conda environment pin, e.g. "  - apitofsim ==0.0.18".
CONDA_PIN = re.compile(r"^(\s*-\s*apitofsim\s*==\s*)[^\s#]+", re.MULTILINE)


def fetch_release(tag: str) -> dict:
    """Look up the release, tolerating a missing or extra leading `v`."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "bump-downstream"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    for candidate in dict.fromkeys(
        [tag, f"v{tag.removeprefix('v')}", tag.removeprefix("v")]
    ):
        request = urllib.request.Request(API.format(tag=candidate), headers=headers)
        try:
            with urllib.request.urlopen(request) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
    raise SystemExit(f"no GitHub release found for {tag}")


def sha256(asset: dict) -> str:
    match = re.fullmatch(r"sha256:([0-9a-f]{64})", asset.get("digest") or "")
    if not match:
        raise SystemExit(f"release asset {asset['name']} has no sha256 digest")
    return match.group(1)


def rewrite(
    path: Path, tag: str, version: str, assets: dict[str, str]
) -> tuple[str, Counter]:
    """Rewrite every apitofsim pin in one file; returns the new text and counts."""
    text = path.read_text()
    counts: Counter = Counter()

    def count(key: str, replacement):
        """Count only the matches that the replacement actually changes."""

        def sub(match: re.Match[str]) -> str:
            new = replacement(match)
            counts[key] += new != match[0]
            return new

        return sub

    def wheel(match: re.Match[str]) -> str:
        suffix = match["asset"].split("-", 2)[2]
        name = f"apitofsim-{version}-{suffix}"
        if name not in assets:
            raise SystemExit(f"{path}: release {tag} has no asset named {name}")
        return f"{DOWNLOAD}/{tag}/{name}"

    def hashed(match: re.Match[str]) -> str:
        name = match["url"].rsplit("/", 1)[1]
        if name not in assets:
            raise SystemExit(f"{path}: release {tag} has no asset named {name}")
        return f'"{match["url"]}"{match["mid"]}{assets[name]}"'

    text = WHEEL.sub(count("wheels", wheel), text)
    text = SDIST.sub(count("sdist urls", lambda m: f"{ARCHIVE}/{tag}.tar.gz"), text)
    text = VERSION.sub(count("versions", lambda m: f"{m[1]}{version}{m[2]}"), text)
    text = CONDA_PIN.sub(count("conda pins", lambda m: f"{m[1]}{version}"), text)
    text = HASH.sub(count("hashes", hashed), text)
    return text, counts


def targets(path: Path) -> list[Path]:
    """The files to patch for the path given on the command line."""
    lock = path.with_name("uv.lock")
    if path.name == "uv.lock":
        return [path]
    if path.suffix == ".toml":
        return [path] + ([lock] if lock.exists() else [])
    if path.suffix in (".yaml", ".yml"):
        return [path]
    raise SystemExit(f"{path}: expected a .toml, uv.lock, .yaml or .yml file")


def patch(
    path: Path, tag: str, version: str, assets: dict[str, str], dry_run: bool
) -> bool:
    """Rewrite one file; returns whether it changed."""
    text = path.read_text()
    new_text, counts = rewrite(path, tag, version, assets)
    if new_text == text:
        print(f"{os.path.relpath(path)}: already at {tag}")
        return False
    if not dry_run:
        path.write_text(new_text)
    print(
        f"{os.path.relpath(path)}: "
        + ", ".join(f"{n} {what}" for what, n in counts.items() if n)
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tag", help="release tag, for example v0.0.18")
    parser.add_argument(
        "file",
        nargs="?",
        type=Path,
        default=Path("pyproject.toml"),
        help="file to update (default: pyproject.toml)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="report changes without writing"
    )
    parser.add_argument(
        "--no-verify", action="store_true", help="skip `uv lock --check`"
    )
    args = parser.parse_args()

    files = targets(args.file)
    release = fetch_release(args.tag)
    tag = release["tag_name"]
    version = tag.removeprefix("v")
    if not re.fullmatch(r"[0-9][0-9A-Za-z.]*", version):
        raise SystemExit(f"cannot derive a version from tag {tag}")
    assets = {asset["name"]: sha256(asset) for asset in release["assets"]}

    changed = False
    for path in files:
        changed = patch(path, tag, version, assets, args.dry_run) or changed
    if not changed:
        return 0
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    if args.no_verify or not any(p.name == "uv.lock" for p in files):
        return 0
    if not shutil.which("uv"):
        print("uv not found, skipping `uv lock --check`", file=sys.stderr)
        return 0
    check = subprocess.run(
        ["uv", "lock", "--check"],
        cwd=args.file.parent,
        capture_output=True,
        text=True,
    )
    print(f"uv lock --check: {'ok' if check.returncode == 0 else 'FAILED'}")
    if check.returncode:
        print(check.stdout + check.stderr, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
