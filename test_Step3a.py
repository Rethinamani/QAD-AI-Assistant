# test_step3a.py
import logging
import os
from ingestion.cleaner.pdf_cleaner import clean_pdf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

PDF_FILE = os.path.join("data", "uploads", "qad_manual.pdf")

print(f"\n── Cleaning PDF: {PDF_FILE} ──\n")
result = clean_pdf(PDF_FILE)

print(f"\n── Results ──")
print(f"  TOC entries:   {len(result['toc'])}")
print(f"  Text blocks:   {len(result['text'])}")
print(f"  Tables:        {len(result['tables'])}")
print(f"  Figures:       {len(result['figures'])}")

print(f"\n── First 2 TOC entries ──")
for entry in result["toc"][:2]:
    print(f"  {entry}")

print(f"\n── First 2 text blocks ──")
for block in result["text"][:2]:
    print(f"\n  Page {block['page']} | Section: {block['section']}")
    print(f"  {block['text'][:200]}")

if result["tables"]:
    print(f"\n── First table (page {result['tables'][0]['page']}) ──")
    print(result["tables"][0]["text"][:300])

if result["figures"]:
    print(f"\n── First figure context (page {result['figures'][0]['page']}) ──")
    print(result["figures"][0]["text"][:200])

print("\n✅ PDF Cleaner test complete.")