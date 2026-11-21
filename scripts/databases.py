#!/usr/bin/env python3
"""Record, show and verify the reference databases (lib/databases.py).

  record NAME --path P [--source URL] [--version V] [--downloaded-on YYYY-MM-DD]
          right after a download: version, date, source and file checksums
  show    the manifest
  verify  every recorded database is unchanged and all were downloaded on one day

Manifest: paths_organism.databases_manifest.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from lib import databases as db  # noqa: E402
from lib.config import load_config, resolve_path  # noqa: E402


def main():
    config = load_config()
    ap = argparse.ArgumentParser(description="Reference database manifest.")
    ap.add_argument("command", choices=["record", "show", "verify"])
    ap.add_argument("name", nargs="?", choices=db.NAMES)
    ap.add_argument("--path", type=Path)
    ap.add_argument("--source")
    ap.add_argument("--version")
    ap.add_argument("--downloaded-on")
    args = ap.parse_args()
    manifest = resolve_path("databases_manifest", config=config)
    if args.command == "record":
        if not args.name or not args.path:
            sys.exit("ERROR: `record` needs a database name and --path.")
        e = db.record(manifest, args.name, args.path, source=args.source, version=args.version,
                      downloaded_on=args.downloaded_on)
        print(f"  ✓ {args.name} {e['version']} ({e['downloaded_on']}, {len(e['files'])} files)")
    elif args.command == "show":
        print(json.dumps({k: {x: v[x] for x in ("version", "downloaded_on", "source", "path")}
                          for k, v in db.load(manifest).items()}, indent=2))
    else:
        problems = db.verify(manifest)
        if not db.same_day(manifest):
            problems.append("the databases were not downloaded on one day")
        if problems:
            sys.exit("ERROR: " + "; ".join(problems))
        print(f"  ✓ {len(db.load(manifest))} databases unchanged, downloaded on one day")


if __name__ == "__main__":
    main()
