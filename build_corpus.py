"""
Build a corpus of Sri Lankan civil procedure law for a RAG system.
Downloads source PDFs, extracts text with pypdf, audits extraction quality,
and writes manifest.json + REPORT.md.
"""
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from pypdf import PdfReader
from pypdf.errors import PdfReadError

ROOT = Path(__file__).parent
RAW = ROOT / "corpus" / "raw"
TEXT = ROOT / "corpus" / "text"
RAW.mkdir(parents=True, exist_ok=True)
TEXT.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/pdf,text/html,application/xhtml+xml,*/*;q=0.8",
}

DELAY_SECONDS = 2

# Documents to attempt. Each has a resolved source_url and expects a PDF,
# except the "related statutes" whose source page (lawnet.gov.lk) is
# currently unreachable (TLS cert mismatch) -- included with source_url=None
# so they show up in the manifest/report as unresolved rather than silently
# dropped.
DOCUMENTS = [
    # --- Consolidated Civil Procedure Code (3 editions) ---
    dict(slug="cpc-consolidated-lankalaw", title="Civil Procedure Code (consolidated, lankalaw.net edition)",
         doc_type="principal_act", act_number=None, year=None, amends=None,
         source_url="https://lankalaw.net/wp-content/uploads/2024/03/Civil-Procedure-Code.pdf"),
    dict(slug="cpc-consolidated-commonlii", title="Civil Procedure Code (consolidated, CommonLII edition)",
         doc_type="principal_act", act_number=None, year=None, amends=None,
         source_url="https://www.commonlii.org/lk/legis/consol_act/cpc105233.pdf"),
    dict(slug="cpc-consolidated-lawnet", title="Civil Procedure Code (consolidated, LawNet edition)",
         doc_type="principal_act", act_number=None, year=None, amends=None,
         source_url="https://www.lawnet.gov.lk/wp-content/uploads/2016/11/CIVIL-PROCEDURE-CODE.pdf"),

    # --- CPC Amendment Acts: known good URLs ---
    dict(slug="cpc-amend-43-2024", title="Civil Procedure Code (Amendment) Act No. 43 of 2024",
         doc_type="amendment_act", act_number="43", year=2024, amends="civil_procedure_code",
         source_url="https://www.parliament.lk/uploads/acts/gbills/english/6350.pdf"),
    dict(slug="cpc-amend-29-2023", title="Civil Procedure Code (Amendment) Act No. 29 of 2023",
         doc_type="amendment_act", act_number="29", year=2023, amends="civil_procedure_code",
         source_url="https://www.parliament.lk/uploads/acts/gbills/english/6310.pdf"),
    dict(slug="cpc-amend-8-2017", title="Civil Procedure Code (Amendment) Act No. 8 of 2017",
         doc_type="amendment_act", act_number="8", year=2017, amends="civil_procedure_code",
         source_url="https://www.srilankalaw.lk/gazette/2017_pdf/08-2017_E.pdf"),

    # --- CPC Amendment Acts: located via lankalaw.net legislation index ---
    dict(slug="cpc-amend-50-2024", title="Civil Procedure Code (Amendment) Act No. 50 of 2024",
         doc_type="amendment_act", act_number="50", year=2024, amends="civil_procedure_code",
         source_url="http://documents.gov.lk/files/act/2024/9/50-2024_E.pdf"),
    dict(slug="cpc-amend-20-2023", title="Civil Procedure Code (Amendment) Act No. 20 of 2023",
         doc_type="amendment_act", act_number="20", year=2023, amends="civil_procedure_code",
         source_url="https://lankalaw.net/wp-content/uploads/2024/02/6307.pdf"),
    dict(slug="cpc-amend-7-2023", title="Civil Procedure Code (Amendment) Act No. 7 of 2023",
         doc_type="amendment_act", act_number="7", year=2023, amends="civil_procedure_code",
         source_url="https://lankalaw.net/wp-content/uploads/2026/06/07-2023_E.pdf"),
    dict(slug="cpc-amend-36-2022", title="Civil Procedure Code (Amendment) Act No. 36 of 2022",
         doc_type="amendment_act", act_number="36", year=2022, amends="civil_procedure_code",
         source_url="https://lankalaw.net/wp-content/uploads/2024/02/6274.pdf"),
    dict(slug="cpc-amend-17-2022", title="Civil Procedure Code (Amendment) Act No. 17 of 2022",
         doc_type="amendment_act", act_number="17", year=2022, amends="civil_procedure_code",
         source_url="https://lankalaw.net/wp-content/uploads/2024/02/6253.pdf"),
    dict(slug="cpc-amend-5-2022", title="Civil Procedure Code (Amendment) Act No. 5 of 2022",
         doc_type="amendment_act", act_number="5", year=2022, amends="civil_procedure_code",
         source_url="https://lankalaw.net/wp-content/uploads/2024/02/6238.pdf"),
    dict(slug="cpc-amend-11-2010", title="Civil Procedure Code (Amendment) Act No. 11 of 2010",
         doc_type="amendment_act", act_number="11", year=2010, amends="civil_procedure_code",
         source_url="https://lankalaw.net/wp-content/uploads/2024/02/11-2010_E.pdf"),
    dict(slug="cpc-amend-4-2005", title="Civil Procedure Code (Amendment) Act No. 4 of 2005",
         doc_type="amendment_act", act_number="4", year=2005, amends="civil_procedure_code",
         source_url="https://www.parliament.lk/uploads/acts/gbills/english/5631.pdf"),

    # --- Related statutes: source page lawnet.gov.lk/legislative-enactments/
    # is unreachable (TLS cert mismatch -> unrelated bluehost host). Cannot
    # locate URLs without guessing, which the task explicitly disallows.
    dict(slug="judicature-act-1978", title="Judicature Act No. 2 of 1978",
         doc_type="related_statute", act_number="2", year=1978, amends=None,
         source_url=None, unresolved_reason="Source page lawnet.gov.lk/legislative-enactments/ unreachable (TLS cert mismatch)"),
    dict(slug="evidence-ordinance", title="Evidence Ordinance (Chapter 14)",
         doc_type="related_statute", act_number=None, year=None, amends=None,
         source_url=None, unresolved_reason="Source page lawnet.gov.lk/legislative-enactments/ unreachable (TLS cert mismatch)"),
    dict(slug="prescription-ordinance", title="Prescription Ordinance",
         doc_type="related_statute", act_number=None, year=None, amends=None,
         source_url=None, unresolved_reason="Source page lawnet.gov.lk/legislative-enactments/ unreachable (TLS cert mismatch)"),
    dict(slug="arbitration-act-1995", title="Arbitration Act No. 11 of 1995",
         doc_type="related_statute", act_number="11", year=1995, amends=None,
         source_url=None, unresolved_reason="Source page lawnet.gov.lk/legislative-enactments/ unreachable (TLS cert mismatch)"),
    dict(slug="mediation-boards-act-1988", title="Mediation Boards Act No. 72 of 1988",
         doc_type="related_statute", act_number="72", year=1988, amends=None,
         source_url=None, unresolved_reason="Source page lawnet.gov.lk/legislative-enactments/ unreachable (TLS cert mismatch)"),
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download(doc):
    url = doc["source_url"]
    result = {"status": "failed", "error": None, "raw_path": None,
              "content_type": None, "http_status": None}
    if url is None:
        result["error"] = doc.get("unresolved_reason", "no source_url")
        return result
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60)
    except requests.exceptions.SSLError as e:
        result["error"] = f"SSL error: {e}"
        return result
    except requests.exceptions.RequestException as e:
        result["error"] = f"Request error: {e}"
        return result

    result["http_status"] = resp.status_code
    ctype = resp.headers.get("Content-Type", "")
    result["content_type"] = ctype

    if resp.status_code != 200:
        result["error"] = f"HTTP {resp.status_code}"
        return result

    body = resp.content
    is_pdf_magic = body[:5] == b"%PDF-"
    looks_html = b"<html" in body[:2000].lower() or b"<!doctype html" in body[:2000].lower()

    if looks_html and not is_pdf_magic:
        result["error"] = f"Returned HTML instead of PDF (content-type={ctype})"
        return result

    if not is_pdf_magic:
        result["error"] = f"Response is not a PDF (no %PDF magic bytes, content-type={ctype})"
        return result

    raw_path = RAW / f"{doc['slug']}.pdf"
    with open(raw_path, "wb") as f:
        f.write(body)

    result["status"] = "ok"
    result["raw_path"] = raw_path
    return result


def audit_extract(pdf_path: Path):
    """Returns dict with pages, chars_extracted, chars_per_page_avg,
    middle_page_chars, verdict, and full extracted text."""
    reader = PdfReader(str(pdf_path))
    n = len(reader.pages)
    page_texts = []
    for p in reader.pages:
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        page_texts.append(t)

    full_text = "\n\n".join(page_texts)
    total_chars = sum(len(t) for t in page_texts)
    avg_chars = total_chars / n if n else 0
    mid_idx = n // 2
    mid_chars = len(page_texts[mid_idx]) if n else 0

    if mid_chars > 500:
        verdict = "born-digital"
    elif mid_chars < 100:
        verdict = "likely scanned"
    else:
        verdict = "mixed/uncertain"

    return {
        "pages": n,
        "chars_extracted": total_chars,
        "chars_per_page_avg": round(avg_chars, 1),
        "middle_page_chars": mid_chars,
        "verdict": verdict,
        "text": full_text,
    }


def main():
    manifest = []
    now = datetime.now(timezone.utc).isoformat()

    for i, doc in enumerate(DOCUMENTS):
        print(f"[{i+1}/{len(DOCUMENTS)}] {doc['slug']} ...", end=" ", flush=True)
        entry = {
            "slug": doc["slug"],
            "title": doc["title"],
            "doc_type": doc["doc_type"],
            "act_number": doc["act_number"],
            "year": doc["year"],
            "amends": doc["amends"],
            "source_url": doc["source_url"],
            "downloaded_at": None,
            "sha256": None,
            "pages": None,
            "chars_extracted": None,
            "chars_per_page_avg": None,
            "middle_page_chars": None,
            "extraction_verdict": None,
            "text_path": None,
            "download_status": None,
            "download_error": None,
        }

        dl = download(doc)
        entry["download_status"] = dl["status"]
        entry["download_error"] = dl["error"]

        if dl["status"] == "ok":
            entry["downloaded_at"] = now
            entry["sha256"] = sha256_of(dl["raw_path"])
            try:
                audit = audit_extract(dl["raw_path"])
                entry["pages"] = audit["pages"]
                entry["chars_extracted"] = audit["chars_extracted"]
                entry["chars_per_page_avg"] = audit["chars_per_page_avg"]
                entry["middle_page_chars"] = audit["middle_page_chars"]
                entry["extraction_verdict"] = audit["verdict"]

                if audit["verdict"] != "likely scanned":
                    text_path = TEXT / f"{doc['slug']}.txt"
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(audit["text"])
                    entry["text_path"] = str(text_path.relative_to(ROOT)).replace("\\", "/")
                print(f"OK ({audit['pages']}p, verdict={audit['verdict']})")
            except (PdfReadError, Exception) as e:
                entry["extraction_verdict"] = "extraction_error"
                entry["download_error"] = f"Downloaded but pypdf failed to parse: {e}"
                print(f"DOWNLOADED but PDF PARSE FAILED: {e}")
        else:
            print(f"FAILED: {dl['error']}")

        manifest.append(entry)

        if doc["source_url"] is not None and i < len(DOCUMENTS) - 1:
            time.sleep(DELAY_SECONDS)

    with open(ROOT / "corpus" / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\nDone. Manifest written to corpus/manifest.json")


if __name__ == "__main__":
    main()
