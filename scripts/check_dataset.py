"""Validate converted Parquet datasets against their conversion manifests."""

import argparse
import json
from pathlib import Path

KINDS = ("speeds", "crashes")


def describe(kind: str, path: Path) -> dict:
    """Manifest metadata for `path`, or raise if it is not a usable dataset.

    Checks the recorded file count against what is on disk, so a half-copied
    or interrupted conversion is rejected rather than silently half-read.
    """
    manifest = path / "_conversion.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"{path}: no _conversion.json")
    recorded = json.loads(manifest.read_text())
    if recorded.get("kind") != kind:
        raise ValueError(f"{path}: manifest kind is {recorded.get('kind')!r}, expected {kind!r}")
    if not recorded.get("rows"):
        raise ValueError(f"{path}: manifest records no rows")
    files = len(list(path.rglob("*.parquet")))
    if files != recorded.get("parquet_files"):
        raise ValueError(
            f"{path}: {files} parquet files on disk, manifest records "
            f"{recorded.get('parquet_files')}"
        )
    return {
        "path": str(path.resolve()),
        "kind": kind,
        "rows": recorded["rows"],
        "parquet_files": files,
        "first_timestamp": recorded.get("first_timestamp"),
        "last_timestamp": recorded.get("last_timestamp"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for kind in KINDS:
        parser.add_argument(f"--{kind}", type=Path)
    parser.add_argument("--record", type=Path, help="write the metadata here as JSON")
    args = parser.parse_args()
    wanted = {kind: getattr(args, kind) for kind in KINDS if getattr(args, kind)}
    if not wanted:
        parser.error("pass --speeds and/or --crashes")
    described = {}
    for kind, path in wanted.items():
        try:
            described[kind] = describe(kind, path)
        except (OSError, ValueError, json.JSONDecodeError) as problem:
            raise SystemExit(str(problem)) from problem
    if args.record:
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(described, indent=2) + "\n")
    print(json.dumps(described, indent=2))


if __name__ == "__main__":
    main()
