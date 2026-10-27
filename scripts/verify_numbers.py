"""Compare the numbers produced by experiments/ with the ones in the paper.

Every experiment writes the numbers quoted in the paper to
paper/generated/<experiment>.tex as LaTeX macros. This script reads those
files and compares them with results/tables/paper_numbers.csv, the reference
snapshot of the submitted paper.

    python scripts/verify_numbers.py            # compare, exit 1 on any mismatch
    python scripts/verify_numbers.py --update   # rewrite the snapshot (authors only)
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GENERATED = REPO / "paper" / "generated"
SNAPSHOT = REPO / "results" / "tables" / "paper_numbers.csv"
MACRO = re.compile(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$")


def read_generated() -> dict[tuple[str, str], str]:
    numbers = {}
    for path in sorted(GENERATED.glob("*.tex")):
        for line in path.read_text().splitlines():
            if m := MACRO.match(line):
                numbers[(path.stem, m.group(1))] = m.group(2)
    return numbers


def read_snapshot() -> dict[tuple[str, str], str]:
    with SNAPSHOT.open(newline="") as fh:
        return {(r["experiment"], r["macro"]): r["value"] for r in csv.DictReader(fh)}


def write_snapshot(numbers: dict[tuple[str, str], str]) -> None:
    with SNAPSHOT.open("w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(["experiment", "macro", "value"])
        for (experiment, macro), value in sorted(numbers.items()):
            writer.writerow([experiment, macro, value])


def main() -> int:
    numbers = read_generated()
    if not numbers:
        print(f"No generated numbers in {GENERATED}; run the experiments first.")
        return 1
    if "--update" in sys.argv:
        write_snapshot(numbers)
        print(f"Wrote {len(numbers)} numbers to {SNAPSHOT.relative_to(REPO)}")
        return 0

    expected = read_snapshot()
    missing = sorted(expected.keys() - numbers.keys())
    differ = sorted(k for k in expected.keys() & numbers.keys() if expected[k] != numbers[k])
    for experiment in sorted({e for e, _ in missing}):
        print(f"MISSING  {experiment}: run experiments/{experiment}.py")
    for key in differ:
        print(f"DIFFERS  {key[0]}  \\{key[1]}: paper {expected[key]}, yours {numbers[key]}")
    matched = len(expected) - len(missing) - len(differ)
    print(f"{matched} of {len(expected)} numbers match the paper.")
    return 0 if matched == len(expected) else 1


if __name__ == "__main__":
    sys.exit(main())
