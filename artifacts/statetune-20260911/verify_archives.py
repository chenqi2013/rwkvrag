"""Verify downloaded release archives and every member before extraction."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha(stream):
    return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("download_directory", type=Path)
    args = parser.parse_args()
    index = Path(__file__).resolve().parent
    for archive in json.loads((index / "ARCHIVES.json").read_text())["archives"]:
        path = args.download_directory / archive["asset"]
        with path.open("rb") as stream:
            if sha(stream) != archive["sha256"]:
                raise ValueError(f"Archive SHA mismatch: {path.name}")
        manifest = index / Path(archive["manifest"]).name
        with manifest.open("rb") as stream:
            if sha(stream) != archive["manifest_sha256"]:
                raise ValueError(f"Manifest SHA mismatch: {manifest.name}")
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        expected = {row["path"]: row for row in rows}
        if len(expected) != len(rows) or len(rows) != archive["member_count"]:
            raise ValueError("Duplicate or missing manifest members")
        seen = set()
        with tarfile.open(path, "r|gz") as tar:
            for member in tar:
                relative = PurePosixPath(member.name)
                if (not member.isfile() or relative.is_absolute()
                        or ".." in relative.parts or member.name in seen
                        or member.name not in expected):
                    raise ValueError(f"Unexpected or unsafe member: {member.name}")
                seen.add(member.name)
                row = expected[member.name]
                with tar.extractfile(member) as stream:
                    if member.size != row["bytes"] or sha(stream) != row["sha256"]:
                        raise ValueError(f"Member mismatch: {member.name}")
        if seen != set(expected):
            raise ValueError(f"Missing archive members: {path.name}")
        print(f"Verified {path.name}: {len(seen)} files")


if __name__ == "__main__":
    main()
