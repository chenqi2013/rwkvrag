from pathlib import Path

import pytest

from llamaindex_retrieval.finewiki_browser import FineWikiPathError, browse_finewiki_paths


def test_browse_finewiki_paths_lists_directories_and_parquet_files(tmp_path: Path) -> None:
    root = tmp_path / "finewiki"
    child = root / "child"
    child.mkdir(parents=True)
    parquet = root / "train.parquet"
    parquet.write_bytes(b"parquet")
    (root / "ignored.txt").write_text("ignore", encoding="utf-8")

    page = browse_finewiki_paths([root], None)

    assert page["current"] == str(root)
    assert page["parent"] is None
    assert page["roots"] == [str(root)]
    assert page["entries"] == [
        {"name": "child", "path": str(child), "type": "directory", "size": None},
        {
            "name": "train.parquet",
            "path": str(parquet),
            "type": "parquet",
            "size": 7,
        },
    ]


def test_browse_finewiki_paths_uses_parent_when_file_is_selected(tmp_path: Path) -> None:
    root = tmp_path / "finewiki"
    root.mkdir()
    parquet = root / "train.parquet"
    parquet.write_bytes(b"data")

    page = browse_finewiki_paths([root], str(parquet))

    assert page["current"] == str(root)


def test_browse_finewiki_paths_rejects_paths_outside_roots(tmp_path: Path) -> None:
    root = tmp_path / "finewiki"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()

    with pytest.raises(FineWikiPathError, match="只能浏览"):
        browse_finewiki_paths([root], str(outside))
