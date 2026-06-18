# test_step2a.py
import logging
import os
from ingestion.cleaner.excel_cleaner import clean_excel

# Show logs in terminal
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Point to your actual Excel file ───────────────────────────────────────────
EXCEL_FILE = os.path.join("data", "uploads", "qad_errors.xlsx")

print(f"\n── Cleaning Excel: {EXCEL_FILE} ──\n")

rows = clean_excel(EXCEL_FILE)

print(f"\n✅ Total clean rows: {len(rows)}")
print(f"   With resolution:    {len([r for r in rows if r['resolution_available']])}")
print(f"   Without resolution: {len([r for r in rows if not r['resolution_available']])}")

print("\n── First 3 cleaned rows ──")
for row in rows[:3]:
    print()
    for k, v in row.items():
        print(f"  {k}: {v}")