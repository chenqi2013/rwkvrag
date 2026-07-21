from pathlib import Path


class FineWikiPathError(ValueError):
    pass


def browse_finewiki_paths(roots: list[Path], requested_path: str | None) -> dict:
    available_roots = [root.expanduser().resolve() for root in roots if root.expanduser().is_dir()]
    if not available_roots:
        raise FineWikiPathError("没有可用的 FineWiki 浏览目录")

    current = (
        Path(requested_path).expanduser().resolve()
        if requested_path
        else available_roots[0]
    )
    if current.is_file():
        current = current.parent
    allowed_root = next(
        (root for root in available_roots if current == root or root in current.parents),
        None,
    )
    if allowed_root is None:
        raise FineWikiPathError("只能浏览配置允许的 FineWiki 目录")
    if not current.is_dir():
        raise FineWikiPathError(f"目录不存在：{current}")

    entries = []
    for item in current.iterdir():
        if item.name.startswith("."):
            continue
        resolved = item.resolve()
        if resolved != allowed_root and allowed_root not in resolved.parents:
            continue
        if resolved.is_dir():
            entries.append(
                {"name": item.name, "path": str(resolved), "type": "directory", "size": None}
            )
        elif resolved.is_file() and resolved.suffix.lower() == ".parquet":
            entries.append(
                {
                    "name": item.name,
                    "path": str(resolved),
                    "type": "parquet",
                    "size": resolved.stat().st_size,
                }
            )
    entries.sort(key=lambda entry: (entry["type"] != "directory", entry["name"].casefold()))
    parent = str(current.parent) if current != allowed_root else None
    return {
        "current": str(current),
        "parent": parent,
        "roots": [str(root) for root in available_roots],
        "entries": entries,
    }
