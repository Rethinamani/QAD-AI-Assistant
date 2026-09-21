# docs/_build_docx.py — build the blueprint as an editable Word document.
import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.join(HERE, "_diagrams")
OUT  = os.path.join(HERE, "Infor Support Assistant - Blueprint.docx")

INK   = RGBColor(0x14, 0x27, 0x3D)
BLUE  = RGBColor(0x0D, 0x4B, 0x96)
MUTED = RGBColor(0x56, 0x69, 0x7F)

doc = Document()

# ── base styles ───────────────────────────────────────────────────────────────
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(10.5)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.15

for i, sz in ((1, 17), (2, 12)):
    st = doc.styles[f"Heading {i}"]
    st.font.name = "Calibri"
    st.font.size = Pt(sz)
    st.font.color.rgb = INK
    st.font.bold = True
    st.paragraph_format.space_before = Pt(18 if i == 1 else 10)
    st.paragraph_format.space_after = Pt(6)

for section in doc.sections:
    section.top_margin = section.bottom_margin = Inches(0.8)
    section.left_margin = section.right_margin = Inches(0.9)

CODE = "Consolas"


def add_body(text, *, style=None, italic=False, color=None, size=None):
    p = doc.add_paragraph(style=style)
    run = p.add_run(text)
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color
    if size is not None:
        run.font.size = Pt(size)
    return p


def add_rich(segments, *, style=None):
    """segments: list of (text, is_code) -> one paragraph with mixed fonts."""
    p = doc.add_paragraph(style=style)
    for text, is_code in segments:
        r = p.add_run(text)
        if is_code:
            r.font.name = CODE
            r.font.size = Pt(9.5)
    return p


def add_diagram(png, caption):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(os.path.join(DIAG, png), width=Inches(6.6))
    c = doc.add_paragraph()
    c.paragraph_format.space_before = Pt(2)
    r = c.add_run(caption)
    r.italic = True
    r.font.size = Pt(9)
    r.font.color.rgb = MUTED


def shade(cell, hex_fill):
    el = OxmlElement("w:shd")
    el.set(qn("w:val"), "clear")
    el.set(qn("w:fill"), hex_fill)
    cell._tc.get_or_add_tcPr().append(el)


def add_table(headers, rows, code_cols=()):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = True
    for j, h in enumerate(headers):
        cell = t.rows[0].cells[j]
        cell.text = h
        shade(cell, "E9F2FC")
        run = cell.paragraphs[0].runs[0]
        run.bold = True
        run.font.size = Pt(9)
        run.font.color.rgb = BLUE
    for row in rows:
        cells = t.add_row().cells
        for j, val in enumerate(row):
            cells[j].text = ""
            para = cells[j].paragraphs[0]
            run = para.add_run(val)
            run.font.size = Pt(9)
            if j in code_cols:
                run.font.name = CODE
                run.font.size = Pt(8.5)
                run.font.color.rgb = BLUE
    doc.add_paragraph()
    return t


# ── masthead ──────────────────────────────────────────────────────────────────
add_body("PROOF OF CONCEPT  ·  TECHNICAL BLUEPRINT", color=BLUE, size=9)
title = doc.add_paragraph()
tr = title.add_run("Infor Support Assistant")
tr.bold = True
tr.font.size = Pt(26)
tr.font.color.rgb = INK
title.paragraph_format.space_after = Pt(4)

add_body(
    "A retrieval-augmented support assistant for Infor WMS that runs entirely on one "
    "machine. Operators ask questions in plain language; the assistant answers from "
    "uploaded manuals, incident and defect exports, and recorded walkthroughs — "
    "grounded, source-cited, and confidence-gated, with no data leaving the host.",
    color=MUTED,
)

add_table(
    ["Project", "Classification", "Revision", "Date", "Runtime"],
    [["Infor Support Assistant", "Proof of Concept", "0.1 — draft", "2026-09-04",
      "Single host / offline / CPU"]],
)

# ── §1 ────────────────────────────────────────────────────────────────────────
doc.add_heading("§1  What this is", level=1)
add_body(
    "The assistant is a local RAG (retrieval-augmented generation) pipeline. Documents "
    "are ingested once into a vector store; at question time the system retrieves the "
    "passages most likely to answer the query, hands them to a local language model as "
    "context, and returns the model's answer alongside the exact sources it drew from."
)
add_body(
    "Every moving part — the language model, the embedding model, the vector store, the "
    "reranker, the PII scrubber — runs on the same host as the API. There is no external "
    "API key, no cloud inference, and no outbound request in the answer path. That "
    "constraint shapes every decision below."
)
doc.add_heading("Design principles", level=2)
for lead, rest in [
    ("Offline by construction.", " Ollama serves generation and embeddings from localhost:11434; ChromaDB, BM25, the cross-encoder and spaCy all run in-process. The only network dependency is a one-time model download."),
    ("Answer only from what was uploaded.", " The model is instructed to use retrieved context exclusively and to name sources by their real file names. A low-confidence retrieval routes to escalation rather than a guess."),
    ("Guarded on both sides.", " Input is checked for injection and abuse; output is scanned for leaked PII, internal identifiers, and fabricated error numbers before it reaches the user."),
    ("Confidence is explicit.", " Each answer carries a band — high, medium, or low — derived from the reranker score of the top passage."),
    ("Ingestion is hands-off.", " A file dropped into data/uploads/ is picked up, routed by type, cleaned, scrubbed, embedded and indexed without operator action."),
]:
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(lead)
    r.bold = True
    p.add_run(rest)

# ── §2 ────────────────────────────────────────────────────────────────────────
doc.add_heading("§2  System architecture", level=1)
add_diagram(
    "diagram_architecture.png",
    "The answer path never leaves the host. The blue dashed boundary is one machine: "
    "browser → FastAPI, then the orchestrator fans out to the retrieval engine, the "
    "Ollama model server, and the SQLite registry. The ingestion pipeline (right) is the "
    "only component that reads external files, and it writes only to the vector store "
    "and registry. The ServiceNow box is a stub — it mints a fake INC-#### id and makes "
    "no outbound call.",
)
add_table(
    ["Component", "Module", "Responsibility"],
    [
        ["Chat UI", "streamlit_app.py",
         "ChatGPT-style front end: conversation sidebar, message history, rendered source citations and figure images. Talks to the API over HTTP only."],
        ["API & lifespan", "api/main.py",
         "/chat, /upload, /escalate, /conversations*, /documents, /stats, /health. Starts the file watcher and builds BM25 indexes on boot."],
        ["Orchestrator", "chatbot/chain.py",
         "The RAG chain: guardrail checks, intent detection, follow-up query rewriting, hybrid search, confidence scoring, prompt assembly, LLM call, output validation."],
        ["Retrieval", "retrieval/hybrid_search.py\nretrieval/bm25_index.py\nretrieval/confidence_scorer.py",
         "Per-collection dense + keyword search, reciprocal rank fusion, cross-encoder rerank, exact record-id pinning, intent-driven promotion of figures / summaries / transcripts, confidence banding."],
        ["Vector store", "vectorstore/chroma_store.py",
         "ChromaDB PersistentClient, cosine HNSW, four collections. Upsert, filtered query, delete-by-source helpers."],
        ["Ingestion", "ingestion/pdf_ingester.py\ningestion/excel_ingester.py\ningestion/video_ingester.py\ningestion/file_watcher.py",
         "Type-specific extraction and chunking, shared PII scrub + sensitivity filter, batched embedding, upsert, BM25 rebuild, registry updates. Auto-triggered by the watcher."],
        ["Security", "security/pii_scrubber.py\nsecurity/sensitivity_tagger.py\nguardrails/*.py",
         "Presidio + spaCy PII redaction, keyword sensitivity tagging, and the input / output guardrails."],
        ["Registry", "registry/document_registry.py\nchatbot/conversation_store.py",
         "SQLite: document lifecycle (doc_id, filename, type, version, status, chunk count) and durable chat history that survives restarts."],
        ["Escalation", "chatbot/servicenow_handler.py",
         "Builds an incident payload from the conversation and returns a dummy INC<timestamp>. No integration — a placeholder for a real connector."],
        ["Desktop shell", "desktop_app.py",
         "Optional pywebview wrapper that packages the Streamlit UI as a native window."],
    ],
    code_cols=(1,),
)

# ── §3 ────────────────────────────────────────────────────────────────────────
doc.add_heading("§3  Request lifecycle", level=1)
add_body(
    "A question passes through a fixed sequence of gates. Two of them — an explicit "
    "escalation request and a low-confidence retrieval — can divert the request away "
    "from the language model entirely."
)
add_diagram(
    "diagram_lifecycle.png",
    "Two exits skip the model. The intent router catches an explicit \"escalate / raise "
    "an incident\" before any retrieval runs. Otherwise the confidence band on the top "
    "reranked passage decides: ≥ 0.40 goes to the LLM; < 0.40 offers escalation instead "
    "of guessing. A short factual query is nudged up one band so a one-line lookup is "
    "not forced to escalate.",
)
for step, detail in [
    ("0 · pending check", "If the previous turn offered escalation, a yes / no here is handled before anything else."),
    ("1 · input guardrail", "Empty, over-length (> 500 chars), prompt-injection patterns, or abuse → rejected with a plain message."),
    ("2 · intent router", "Escalation phrasing → confirmation prompt. Acronym / \"what does X stand for\" → BM25 definition lookup, no LLM."),
    ("3 · query rewrite", "Short follow-ups are enriched with the entity or topic from the previous turn (heuristic, no model call)."),
    ("4 · hybrid search", "Dense + BM25 per collection → RRF → cross-encoder rerank → top 5. Confidence = normalized score of rank 1."),
    ("5 · escalation gate", "Low band → offer to raise an incident. The original query is held for the payload."),
    ("6 · generation", "Context block + conversation history + banded instructions → llama3.1:8b at temperature 0.1."),
    ("7 · output guardrail", "Redact PII, strip internal identifiers, flag error numbers absent from the context, truncate to the token cap."),
]:
    p = doc.add_paragraph()
    r = p.add_run(step + "  —  ")
    r.bold = True
    r.font.color.rgb = BLUE
    p.add_run(detail)

# ── §4 ────────────────────────────────────────────────────────────────────────
doc.add_heading("§4  Ingestion pipelines", level=1)
add_body(
    "Three source types share one back half. Each has its own extractor; from the "
    "chunking step onward the path is identical — scrub, filter, embed, index. The "
    "target collection is chosen by file type, and for spreadsheets by file name."
)
add_diagram(
    "diagram_ingestion.png",
    "One shared back half. Extraction differs per type; everything after Normalise is "
    "common code. A spreadsheet whose name contains neither \"incident\" nor \"defect\" "
    "is rejected at upload, as is any file whose name was ingested before. The embed "
    "step batches 128 chunks per Ollama call.",
)
for h, body in [
    ("PDF", "Text blocks, tables (pdfplumber), and embedded raster figures. Figures are detected by image size and page position — not by caption — then run through Tesseract so diagram labels become searchable text. Figure images are saved and served back with the answer."),
    ("Excel", "Generic single-sheet reader: first row is the header, columns are arbitrary. One chunk per row plus a sheet-overview chunk for \"how many\" questions. Every column is kept as col_* metadata. Routed to incident_tickets or defect_records by file name."),
    ("Audio / video", "ffmpeg extracts a 16 kHz mono track; faster-whisper (\"medium\" + voice activity detection) transcribes it. Segments are merged into ~60-second / 700-character windows, each carrying its timestamp, plus one full-transcript overview chunk. Sidecar .srt / .json are written too."),
]:
    p = doc.add_paragraph()
    r = p.add_run(h + "  —  ")
    r.bold = True
    p.add_run(body)

# ── §5 ────────────────────────────────────────────────────────────────────────
doc.add_heading("§5  Retrieval & confidence", level=1)
add_body(
    "Retrieval runs every collection in parallel as independent ranked lists, then fuses "
    "them so a large collection cannot outweigh a small one by sheer volume. Fusion "
    "feeds a cross-encoder, which is the score that actually decides the answer."
)
for term, detail in [
    ("dense", "Top 20 per collection from ChromaDB, cosine similarity on nomic-embed-text vectors."),
    ("sparse", "Top 20 per collection from a per-collection BM25 Okapi index, rebuilt after every ingest."),
    ("fusion", "Reciprocal rank fusion, k = 60, ranked within each list — every list's rank 1 contributes 1/(k+1). Fused to 10, plus each collection's own best candidate."),
    ("rerank", "cross-encoder/ms-marco-MiniLM-L-6-v2 scores every query–passage pair; top 5 survive."),
    ("exact id", "A bare record token in the query (e.g. SNOW8729267) is looked up by row_key and pinned to rank 0."),
    ("promotions", "Figure-intent, aggregate-intent (\"how many\"), and transcript-intent queries each force their matching chunk type into the result set if absent."),
]:
    p = doc.add_paragraph()
    r = p.add_run(term + "  —  ")
    r.bold = True
    r.font.color.rgb = BLUE
    p.add_run(detail)
doc.add_paragraph()
add_table(
    ["Band", "Score", "Behaviour"],
    [
        ["high", "≥ 0.75", "Answer directly and completely from context."],
        ["medium", "0.40 – 0.75", "Answer, but instructed to extract inline detail and note gaps."],
        ["low", "< 0.40", "No LLM call — offer to raise an incident instead."],
    ],
)

# ── §6 ────────────────────────────────────────────────────────────────────────
doc.add_heading("§6  Security & guardrails", level=1)
for h, body in [
    ("Input guardrail", "Rejects empty input, queries over 500 characters, prompt-injection patterns (\"ignore previous instructions\", \"you are now…\", [system]), and abusive language — each with a specific message."),
    ("Output guardrail", "Regex-redacts email / phone / IP / SSN from the model's reply, replaces leaked internal terms (chromadb, doc_id, file paths) with [internal], flags error numbers not present in the retrieved context, and truncates to the response token cap."),
    ("PII scrub (ingest)", "Presidio + spaCy (en_core_web_sm, trimmed pipeline) redact PERSON, EMAIL_ADDRESS, PHONE_NUMBER, IP_ADDRESS, US_SSN, CREDIT_CARD, IBAN_CODE before anything is embedded. Runs as one batched NLP pass over all chunks."),
    ("Sensitivity filter", "Keyword tagging (salary, payroll, confidential, password, credential…). A chunk tagged \"high\" is dropped — never embedded, never retrievable. Figure chunks are flagged as possibly containing PII in the image itself."),
]:
    p = doc.add_paragraph()
    r = p.add_run(h + "  —  ")
    r.bold = True
    p.add_run(body)
add_body(
    "URL and LOCATION are deliberately not redacted: URL redaction removes document "
    "links and hurts retrieval; LOCATION rarely carries real PII in ERP material and is "
    "costly to detect.",
    color=MUTED, size=9,
)

# ── §7 ────────────────────────────────────────────────────────────────────────
doc.add_heading("§7  Data stores", level=1)
add_table(
    ["Collection", "Fed by", "Chunk types", "Key metadata"],
    [
        ["pdf_chunks", "PDF ingester", "text, table, figure", "page, section, chunk_type, image_path"],
        ["incident_tickets", "Excel, name contains \"incident\"", "excel_row, sheet_summary", "row_number, row_key, col_*"],
        ["defect_records", "Excel, name contains \"defect\" / \"bug\"", "excel_row, sheet_summary", "row_number, row_key, col_*"],
        ["video_transcripts", "Audio / video ingester", "transcript_segment, transcript_summary", "start_sec, end_sec, start_time, segment_index"],
    ],
    code_cols=(0,),
)
for term, detail in [
    ("Vector index", "ChromaDB PersistentClient at vectorstore/chroma_db/, cosine HNSW. Every collection also has a pickled BM25 index under vectorstore/bm25_indexes/."),
    ("document_registry.db", "One row per ingested file: doc_id, filename, source type, version, status, chunk count, timestamp. Drives duplicate-name rejection."),
    ("conversations.db", "Conversations and messages with confidence, sources and images. The chat window is rehydrated from here after a restart."),
    ("data/ layout", "uploads/ (watched), figures/ (served), transcripts/ (sidecars). Nothing here is authoritative — the vector store and registry are."),
]:
    p = doc.add_paragraph()
    r = p.add_run(term + "  —  ")
    r.bold = True
    r.font.color.rgb = BLUE
    p.add_run(detail)

# ── §8 ────────────────────────────────────────────────────────────────────────
doc.add_heading("§8  Technology stack", level=1)
for h, body in [
    ("Models (local)", "Ollama — llama3.1:8b, nomic-embed-text (768-dim). sentence-transformers 4.1 cross-encoder. spaCy en_core_web_sm 3.8. faster-whisper 1.1."),
    ("Retrieval & store", "chromadb, rank-bm25 0.2.2. Orchestration glue from langchain 0.3.25 / langchain-community / langchain-ollama."),
    ("Ingestion", "pymupdf 1.25.5, pdfplumber 0.11.6, pytesseract 0.3.13 + Tesseract binary, pandas 2.3.3 / openpyxl 3.1.5, ffmpeg binary, watchdog 6.0."),
    ("Service & UI", "fastapi 0.115 / uvicorn 0.34, streamlit 1.45, pywebview 6.2, SQLAlchemy 2.0, presidio-analyzer / anonymizer 2.2.356."),
]:
    p = doc.add_paragraph()
    r = p.add_run(h + "  —  ")
    r.bold = True
    p.add_run(body)
add_body(
    "External binaries required on the host: Ollama, Tesseract OCR, ffmpeg. Everything "
    "else installs from requirements.txt.",
    color=MUTED, size=9,
)

# ── §9 ────────────────────────────────────────────────────────────────────────
doc.add_heading("§9  Known limits & next steps", level=1)
add_table(
    ["Limit", "Detail", "Direction"],
    [
        ["CPU-only inference", "Embedding is ~83% of ingestion wall-clock; generation is the slowest step at query time. Concurrency gave no gain on shared cores.", "A GPU for Ollama is the single biggest speedup. Batch size and concurrency are already env-configurable for that day."],
        ["Vector-drawn diagrams", "Only embedded raster images are extracted from PDFs. Diagrams drawn as vector paths are not captured as figures.", "Cluster drawing-operation bounding boxes and render those regions to raster before OCR."],
        ["Table-of-contents noise", "Dotted TOC lines can rank as high-confidence sources.", "Filter leader-dot / page-number-only chunks at clean time."],
        ["Escalation is a stub", "No real ticketing integration; the incident id is synthetic.", "Replace servicenow_handler with a real connector behind the same interface."],
        ["Single tenant", "No authentication on the API; one shared conversation store; assumes a trusted local network.", "Add auth and per-user scoping before any multi-user deployment."],
        ["Transcription quality", "Accuracy tracks source audio; quiet or noisy recordings degrade badly even at the \"medium\" model.", "Surface a confidence signal per transcript; allow a larger model where the host can afford it."],
    ],
)

foot = doc.add_paragraph()
foot.paragraph_format.space_before = Pt(14)
fr = foot.add_run("Infor Support Assistant · Blueprint rev 0.1   ·   Proof of concept — not a production system   ·   2026-09-04")
fr.font.size = Pt(8)
fr.font.color.rgb = MUTED

doc.save(OUT)
print("saved", OUT)
