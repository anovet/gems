"""The shared results table: one CSV row per run, compared across runs and iterations."""

import csv
from pathlib import Path
from typing import Any

LEADERBOARD_COLUMN = "leaderboard_score"  # filled in by hand after submitting; always last


def append_result(path: Path, row: dict[str, Any]) -> None:
    """Append one run's row, creating the table on first use.

    Lists are written as ';'-separated values. If the row brings new columns, the table is
    rewritten with them added (before the leaderboard column), keeping every existing value,
    including hand-entered leaderboard scores.
    """
    row = {k: ";".join(str(x) for x in v) if isinstance(v, (list, tuple)) else v for k, v in row.items()}
    row.setdefault(LEADERBOARD_COLUMN, "")

    existing: list[dict[str, str]] = []
    header: list[str] = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            header = list(reader.fieldnames or [])
            existing = list(reader)

    columns = [c for c in header if c != LEADERBOARD_COLUMN]
    columns += [c for c in row if c not in columns and c != LEADERBOARD_COLUMN]
    columns.append(LEADERBOARD_COLUMN)

    path.parent.mkdir(parents=True, exist_ok=True)
    if columns == header:
        with open(path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=columns).writerow(row)
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, restval="")
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(row)
