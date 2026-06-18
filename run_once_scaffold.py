# run_once_scaffold.py
import os

folders = [
    "ingestion/cleaner",
    "retrieval",
    "guardrails",
    "security",
    "registry",
    "job_queue",
    "chatbot",
    "api",
    "vectorstore",
    "data/uploads",
    "data/processed",
    "data/nightly_drop",
    "data/figures",
    "logs",
]

files = [
    "ingestion/__init__.py",
    "ingestion/cleaner/__init__.py",
    "ingestion/pdf_ingester.py",
    "ingestion/excel_ingester.py",
    "ingestion/nightly_ingester.py",
    "ingestion/cleaner/pdf_cleaner.py",
    "ingestion/cleaner/excel_cleaner.py",
    "ingestion/cleaner/servicenow_cleaner.py",
    "retrieval/__init__.py",
    "retrieval/hybrid_search.py",
    "retrieval/confidence_scorer.py",
    "guardrails/__init__.py",
    "guardrails/input_guardrail.py",
    "guardrails/output_guardrail.py",
    "security/__init__.py",
    "security/pii_scrubber.py",
    "security/sensitivity_tagger.py",
    "registry/__init__.py",
    "registry/document_registry.py",
    "job_queue/__init__.py",
    "job_queue/job_queue.py",
    "chatbot/__init__.py",
    "chatbot/chain.py",
    "chatbot/servicenow_handler.py",
    "api/__init__.py",
    "api/main.py",
    "vectorstore/__init__.py",
    "vectorstore/chroma_store.py",
    "config.py",
    ".env",
    ".gitignore",
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"✅ Created folder: {folder}")

for file in files:
    if not os.path.exists(file):
        with open(file, "w") as f:
            pass
        print(f"✅ Created file: {file}")
    else:
        print(f"⏭️  Already exists: {file}")

print("\n✅ Project scaffold complete.")