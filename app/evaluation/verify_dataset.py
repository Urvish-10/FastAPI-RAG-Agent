"""
Verify that every evidence entry in an eval dataset really appears on the stated
page of the stated PDF (so expected pages are checked, not trusted).

Matching ignores case and all whitespace, because PDF text extraction can split
words ("Eligib le") or change spacing.

Usage:
    uv run python -m app.evaluation.verify_dataset [dataset.json] [--docs-dir docs]
Exit code is non-zero if any evidence is missing.
"""
import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

from pypdf import PdfReader

THIS_DIR = Path(__file__).parent
DEFAULT_DATASET = THIS_DIR / "datasets" / "policy_docs_qa.json"
DEFAULT_DOCS = THIS_DIR.parents[1] / "docs"


def _norm(s: str) -> str:
    # normalise quotes/dashes, drop whitespace and control chars, lowercase
    s = (s.replace("’", "'").replace("‘", "'")
          .replace("–", "-").replace("—", "-").replace("\x00", ""))
    return re.sub(r"\s+", "", s).lower()


@lru_cache(maxsize=None)
def _pages(pdf_path: str) -> tuple:
    return tuple(_norm(p.extract_text() or "") for p in PdfReader(pdf_path).pages)


def verify(dataset_path: Path, docs_dir: Path) -> list[str]:
    errors: list[str] = []
    items = json.loads(dataset_path.read_text())
    ids = [i["id"] for i in items]
    if len(ids) != len(set(ids)):
        errors.append("duplicate question ids")

    for item in items:
        if item["category"] == "unanswerable" and item["evidence"]:
            errors.append(f"{item['id']}: unanswerable question must have no evidence")
        if item["category"] != "unanswerable" and not item["evidence"]:
            errors.append(f"{item['id']}: answerable question has no evidence")
        for e in item["evidence"]:
            pdf = docs_dir / e["document"]
            if not pdf.exists():
                errors.append(f"{item['id']}: missing document {e['document']}")
                continue
            pages = _pages(str(pdf))
            if not 1 <= e["page"] <= len(pages):
                errors.append(f"{item['id']}: {e['document']} has no page {e['page']}")
            elif _norm(e["contains"]) not in pages[e["page"] - 1]:
                where = [i + 1 for i, t in enumerate(pages) if _norm(e["contains"]) in t]
                errors.append(
                    f"{item['id']}: '{e['contains']}' not on {e['document']} p{e['page']} "
                    f"(found on pages {where or 'nowhere'})"
                )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset", nargs="?", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS)
    args = parser.parse_args()

    errors = verify(args.dataset, args.docs_dir)
    n = len(json.loads(args.dataset.read_text()))
    if errors:
        print(f"{len(errors)} problem(s) in {n} questions:")
        for e in errors:
            print("  -", e)
        sys.exit(1)
    print(f"OK: all evidence verified for {n} questions.")


if __name__ == "__main__":
    main()
