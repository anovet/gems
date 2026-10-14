"""Every path the docs point at exists, so a moved or renamed file can't strand a pointer."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".venv", ".claude", ".git", "data", "runs"}
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
BACKTICK_PATH = re.compile(r"`([\w./-]+\.md|[\w./-]+/)`")


def doc_files() -> list[Path]:
    return sorted(
        p for p in ROOT.rglob("*.md") if not SKIP_DIRS.intersection(p.relative_to(ROOT).parts)
    )


def markdown_links(path: Path) -> list[tuple[str, Path]]:
    links = []
    for target in MD_LINK.findall(path.read_text(encoding="utf-8")):
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            continue
        links.append((target, path.parent / target.split("#")[0]))
    return links


@pytest.mark.parametrize("doc", doc_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_markdown_links_resolve(doc):
    missing = [target for target, resolved in markdown_links(doc) if not resolved.exists()]
    assert not missing, f"{doc.relative_to(ROOT)} links to missing paths: {missing}"


def test_agents_md_pointers_resolve():
    """AGENTS.md names files in backticks, relative to the repo root."""
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    missing = [p for p in BACKTICK_PATH.findall(text) if not (ROOT / p).exists()]
    assert not missing, f"AGENTS.md points at missing paths: {missing}"
