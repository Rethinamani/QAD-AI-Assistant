# test_step4.py
import logging
from security.pii_scrubber import scrub_text, scrub_chunk
from security.sensitivity_tagger import tag_sensitivity, filter_high_sensitivity

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

print("\n── Test 1: PII Scrubbing ──\n")

test_cases = [
    "Contact John Smith at john.smith@company.com or call 555-123-4567.",
    "Server IP is 192.168.1.100. Please do not share externally.",
    "Error 4023: Tax environment missing. Resolution: Contact BBC team.",
    "No PII here — just a normal technical error message.",
]

for text in test_cases:
    result = scrub_text(text)
    print(f"  Original : {text}")
    print(f"  Scrubbed : {result['scrubbed_text']}")
    print(f"  PII found: {result['pii_types_found']}")
    print()

print("\n── Test 2: Chunk Scrubbing ──\n")

chunk = {
    "text":       "Submitted by Jane Doe (jane.doe@qad.com). Error on PO #4441.",
    "page":       5,
    "chunk_type": "text",
    "section":    "Purchasing",
}
scrubbed = scrub_chunk(chunk)
print(f"  Scrubbed text: {scrubbed['text']}")
print(f"  PII detected : {scrubbed['pii_detected']}")
print(f"  PII types    : {scrubbed['pii_types_found']}")

print("\n── Test 3: Figure chunk flag ──\n")

figure_chunk = {
    "text":       "Fig. 2.1 Requisition Process overview diagram.",
    "page":       12,
    "chunk_type": "figure",
    "section":    "Purchasing",
}
scrubbed_fig = scrub_chunk(figure_chunk)
print(f"  image_may_contain_pii: {scrubbed_fig['image_may_contain_pii']}")

print("\n── Test 4: Sensitivity Tagging ──\n")

chunks = [
    {"text": "Normal error description about purchase orders.", "chunk_type": "text", "page": 10},
    {"text": "This document is confidential and for internal use only.", "chunk_type": "text", "page": 20},
    {"text": "Executive salary details for Q3 payroll processing.", "chunk_type": "text", "page": 30},
    {"text": "Draft version — do not distribute to external parties.", "chunk_type": "text", "page": 40},
]

tagged = [tag_sensitivity(c) for c in chunks]
for c in tagged:
    print(f"  Page {c['page']} | {c['sensitivity_level'].upper():6} | {c['text'][:60]}")

print("\n── Test 5: Filter high sensitivity ──\n")
safe, blocked = filter_high_sensitivity(tagged)
print(f"  Safe chunks   : {len(safe)}")
print(f"  Blocked chunks: {len(blocked)}")
for b in blocked:
    print(f"    Blocked → Page {b['page']}: {b['text'][:60]}")

print("\n✅ Step 4 complete.")