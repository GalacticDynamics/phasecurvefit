"""Write switcher.json and the root redirect for the versioned docs on gh-pages.

Usage: python docs_pages_index.py SITE_DIR BASE_URL

SITE_DIR holds one folder per version: ``latest`` and ``vX.Y``. Prints the
newest release (``vX.Y``), or nothing if there is none, so the workflow can
copy that folder to ``stable``.
"""

import json
import re
import sys
from pathlib import Path


def release_versions(site: Path, /) -> list[str]:
    """Release folders, newest first."""
    names = [p.name for p in site.iterdir() if p.is_dir()]
    found = [n for n in names if re.fullmatch(r"v\d+\.\d+", n)]
    return sorted(found, key=lambda n: tuple(map(int, n[1:].split("."))), reverse=True)


def write_index(site: Path, base_url: str, /) -> str | None:
    """Write switcher.json and index.html; return the newest release."""
    releases = release_versions(site)
    newest = releases[0] if releases else None
    entries = [
        {"name": "latest (main)", "version": "latest", "url": f"{base_url}latest/"}
    ]
    entries += [
        {
            "name": f"{v} (stable)" if v == newest else v,
            "version": v,
            "url": f"{base_url}{v}/",
            **({"preferred": True} if v == newest else {}),
        }
        for v in releases
    ]
    (site / "switcher.json").write_text(json.dumps(entries, indent=2) + "\n")
    target = "stable/" if newest else "latest/"
    (site / "index.html").write_text(
        f'<!doctype html><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="0; url={target}">'
        f'<link rel="canonical" href="{base_url}{target}">'
        f'<a href="{target}">{target}</a>\n'
    )
    (site / ".nojekyll").touch()
    return newest


def _demo() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        site = Path(d)
        for n in ("latest", "v0.3", "v0.10", "v0.9", "stable", "pr-1"):
            (site / n).mkdir()
        assert write_index(site, "https://x/") == "v0.10"
        entries = json.loads((site / "switcher.json").read_text())
        assert [e["version"] for e in entries] == ["latest", "v0.10", "v0.9", "v0.3"]
        assert entries[1]["preferred"] is True
        assert "stable/" in (site / "index.html").read_text()


if __name__ == "__main__":
    if sys.argv[1:] == ["--demo"]:
        _demo()
    else:
        print(write_index(Path(sys.argv[1]), sys.argv[2]) or "")
