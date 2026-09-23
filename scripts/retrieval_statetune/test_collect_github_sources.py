"""Content-addressed source snapshots must match the upstream Git blob."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_github_sources import git_blob_sha


def test_git_blob_identity():
    assert git_blob_sha(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
