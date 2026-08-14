"""
Batch 2: retry the 5 related statutes with newly-located lankalaw.net URLs,
and retry Act 50/2024 status (flagged as mislabeled -- not downloaded).
Updates existing manifest.json entries in place (matched by slug) rather
than duplicating them.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from build_corpus import download, audit_extract, sha256_of, RAW, TEXT, ROOT, DELAY_SECONDS

UPDATES = [
    dict(slug="judicature-act-1978",
         title="Judicature Act No. 2 of 1978 (consolidated to 2023) — lankalaw.net edition",
         source_url="https://lankalaw.net/wp-content/uploads/2024/03/Judicature-Act.pdf"),
    dict(slug="evidence-ordinance",
         title="Evidence Ordinance, Chapter 14 (consolidated to 2024) — lankalaw.net edition",
         source_url="https://lankalaw.net/wp-content/uploads/2025/03/Evidence-Ordinance-Consolidated-2024.pdf"),
    dict(slug="prescription-ordinance",
         title="Prescription Ordinance (consolidated to 2024) — lankalaw.net edition",
         source_url="https://lankalaw.net/wp-content/uploads/2025/03/Prescription-Consolidated-2024.pdf"),
    dict(slug="arbitration-act-1995",
         title="Arbitration Act No. 11 of 1995 — lankalaw.net edition",
         source_url="https://lankalaw.net/wp-content/uploads/2024/02/3107.pdf"),
    dict(slug="mediation-boards-act-1988",
         title="Mediation Boards Act No. 72 of 1988 (consolidated to 2024) — lankalaw.net edition",
         source_url="https://lankalaw.net/wp-content/uploads/2025/03/Mediation-Boards-Act-Consolidated-2024.pdf"),
]


def main():
    manifest_path = ROOT / "corpus" / "manifest.json"
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    by_slug = {e["slug"]: e for e in manifest}
    now = datetime.now(timezone.utc).isoformat()

    for i, upd in enumerate(UPDATES):
        slug = upd["slug"]
        print(f"[{i+1}/{len(UPDATES)}] {slug} ...", end=" ", flush=True)
        entry = by_slug[slug]
        entry["title"] = upd["title"]
        entry["source_url"] = upd["source_url"]

        dl = download({"slug": slug, "source_url": upd["source_url"]})
        entry["download_status"] = dl["status"]
        entry["download_error"] = dl["error"]

        if dl["status"] == "ok":
            entry["downloaded_at"] = now
            entry["sha256"] = sha256_of(dl["raw_path"])
            audit = audit_extract(dl["raw_path"])
            entry["pages"] = audit["pages"]
            entry["chars_extracted"] = audit["chars_extracted"]
            entry["chars_per_page_avg"] = audit["chars_per_page_avg"]
            entry["middle_page_chars"] = audit["middle_page_chars"]
            entry["extraction_verdict"] = audit["verdict"]
            if audit["verdict"] != "likely scanned":
                text_path = TEXT / f"{slug}.txt"
                with open(text_path, "w", encoding="utf-8") as f:
                    f.write(audit["text"])
                entry["text_path"] = str(text_path.relative_to(ROOT)).replace("\\", "/")
            print(f"OK ({audit['pages']}p, verdict={audit['verdict']})")
        else:
            print(f"FAILED: {dl['error']}")

        if i < len(UPDATES) - 1:
            time.sleep(DELAY_SECONDS)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print("\nManifest updated.")


if __name__ == "__main__":
    main()
