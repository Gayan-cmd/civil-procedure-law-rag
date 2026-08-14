"""Regex scan for inline amendment annotations in extracted CPC text."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
TEXT = ROOT / "corpus" / "text"

# Patterns covering common Sri Lankan legislative amendment annotation styles:
#   [§2, Law 20 of 1977]          -- section-mark bracket form
#   (Amended by Act No. 8 of 2017) -- prose parenthetical form
#   ss. 4, Act 11 of 2010          -- section-list form
#   [Act No. 8 of 2017]            -- bare bracket citation
#   As amended by ... Act No. X of YYYY
PATTERNS = {
    "bracket_section_mark": re.compile(
        r"\[\s*(?:§|ss?\.?|Sec(?:tion)?s?\.?)\s*[\d,\s()a-zA-Z]*,?\s*"
        r"(?:Law|Act|Ordinance)s?\s*No\.?\s*\d+[\w\s]*of\s*\d{4}\s*\]",
        re.IGNORECASE,
    ),
    "bracket_bare_citation": re.compile(
        r"\[\s*(?:Law|Act|Ordinance)s?\s*No\.?\s*\d+[\w\s]*of\s*\d{4}\s*\]",
        re.IGNORECASE,
    ),
    "parenthetical_amended_by": re.compile(
        r"\(\s*[Aa]mended\s+by\s+(?:Law|Act|Ordinance)s?\s*No\.?\s*\d+\s*of\s*\d{4}\s*\)",
        re.IGNORECASE,
    ),
    "inline_section_list": re.compile(
        r"\bss?\.\s*\d+(?:\s*,\s*\d+)*\s*,?\s*(?:Law|Act|Ordinance)\s*(?:No\.?)?\s*\d+\s*of\s*\d{4}",
        re.IGNORECASE,
    ),
    "as_amended_by_prose": re.compile(
        r"[Aa]s\s+amended\s+by\s+(?:Law|Act|Ordinance)s?\s*No\.?\s*\d+\s*of\s*\d{4}",
        re.IGNORECASE,
    ),
    "inserted_by_prose": re.compile(
        r"[Ii]nserted\s+by\s+(?:Law|Act|Ordinance)s?\s*No\.?\s*\d+\s*of\s*\d{4}",
        re.IGNORECASE,
    ),
    "substituted_by_prose": re.compile(
        r"[Ss]ubstituted\s+by\s+(?:Law|Act|Ordinance)s?\s*No\.?\s*\d+\s*of\s*\d{4}",
        re.IGNORECASE,
    ),
    "repealed_by_prose": re.compile(
        r"[Rr]epealed\s+by\s+(?:Law|Act|Ordinance)s?\s*No\.?\s*\d+\s*of\s*\d{4}",
        re.IGNORECASE,
    ),
}


def scan_file(path: Path):
    text = path.read_text(encoding="utf-8")
    found = []
    for name, pat in PATTERNS.items():
        for m in pat.finditer(text):
            found.append((name, m.group(0).strip(), m.start()))
    found.sort(key=lambda x: x[2])
    return found


def main():
    results = {}
    for txt_path in sorted(TEXT.glob("cpc-consolidated*.txt")):
        matches = scan_file(txt_path)
        results[txt_path.stem] = matches
        print(f"\n=== {txt_path.stem} ===")
        print(f"Total matches: {len(matches)}")
        by_type = {}
        for name, snippet, pos in matches:
            by_type.setdefault(name, 0)
            by_type[name] += 1
        for name, count in by_type.items():
            print(f"  {name}: {count}")
        print("\n  First 15 examples:")
        for name, snippet, pos in matches[:15]:
            snippet_clean = re.sub(r"\s+", " ", snippet)
            print(f"    [{name}] {snippet_clean!r}  (char offset {pos})")

    out = {k: [{"type": n, "text": s, "offset": p} for n, s, p in v] for k, v in results.items()}
    with open(ROOT / "corpus" / "amendment_annotation_scan.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
