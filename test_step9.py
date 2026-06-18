# test_step9.py
import logging
import os
import time
import threading
from ingestion.file_watcher import start_file_watcher
from ingestion.nightly_ingester import run_nightly_ingestion

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Test 1: Nightly ingestion with a sample CSV ────────────────────────────────
print("\n── Test 1: Nightly Ingestion ──\n")

# Create a sample ServiceNow CSV in nightly_drop/
sample_csv = """number,short_description,description,close_notes,category,state
INC001,Tax environment error,Customer reported blank tax environment on EDI 850,Resolved by BBC Pricing Team. Updated tax environment configuration.,QAD ERP,Resolved
INC002,Credit terms missing,Credit term code invalid on incoming 850 transaction,Verified and corrected credit term code in QAD master data.,QAD ERP,Resolved
INC003,Ship-to address not found,EDI 850 rejected due to missing ship-to address 28000757,Added ship-to address to QAD customer master.,QAD ERP,Resolved
"""

csv_path = os.path.join("data", "nightly_drop", "servicenow_export.csv")
with open(csv_path, "w") as f:
    f.write(sample_csv)
print(f"Created sample CSV: {csv_path}")

summary = run_nightly_ingestion(csv_path)
print("\nNightly Ingestion Summary:")
for k, v in summary.items():
    print(f"  {k}: {v}")

# ── Test 2: Run again — should skip already-ingested tickets ──────────────────
print("\n── Test 2: Re-run — deduplication check ──\n")
summary2 = run_nightly_ingestion(csv_path)
print("\nSecond Run Summary:")
for k, v in summary2.items():
    print(f"  {k}: {v}")
print(f"  Expected: new_ingested=0, skipped=3")

# ── Test 3: File watcher — run briefly then stop ──────────────────────────────
print("\n── Test 3: File Watcher (5 second demo) ──\n")
print("Starting file watcher for 5 seconds...")
print("Drop a file into data/uploads/ to see auto-ingestion trigger.")

watcher_thread = threading.Thread(target=start_file_watcher, daemon=True)
watcher_thread.start()
time.sleep(5)
print("File watcher demo complete.")

print("\n✅ Step 9 complete.")