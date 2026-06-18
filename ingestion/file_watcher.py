# ingestion/file_watcher.py

import os
import time
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from config import UPLOADS_DIR
from ingestion.excel_ingester import ingest_excel
from ingestion.pdf_ingester import ingest_pdf
from retrieval.bm25_index import build_bm25_index
from config import CHROMA_COLLECTION_PDF, CHROMA_COLLECTION_EXCEL

logger = logging.getLogger(__name__)

# Supported file extensions and their ingester
SUPPORTED_EXTENSIONS = {
    ".pdf":  "pdf",
    ".xlsx": "excel",
    ".xls":  "excel",
}

# How long to wait after a file appears before processing
# (avoids processing incomplete uploads)
STABILIZATION_DELAY_SECONDS = 3


class UploadEventHandler(FileSystemEventHandler):
    """
    Watchdog event handler for the uploads directory.
    Triggers ingestion pipeline when a new supported file is created.
    """

    def __init__(self):
        super().__init__()
        # Track files currently being processed to avoid double-trigger
        self._processing = set()

    def on_created(self, event):
        """Called when a file or directory is created."""
        if event.is_directory:
            return

        file_path = event.src_path
        _, ext    = os.path.splitext(file_path)
        ext       = ext.lower()

        if ext not in SUPPORTED_EXTENSIONS:
            logger.debug(f"Ignoring unsupported file type: {file_path}")
            return

        if file_path in self._processing:
            return

        self._processing.add(file_path)
        logger.info(f"New file detected: {file_path}")

        # Wait for file to stabilize (finish copying)
        time.sleep(STABILIZATION_DELAY_SECONDS)

        # Verify file still exists and is non-empty
        if not os.path.exists(file_path):
            logger.warning(f"File disappeared before processing: {file_path}")
            self._processing.discard(file_path)
            return

        if os.path.getsize(file_path) == 0:
            logger.warning(f"File is empty — skipping: {file_path}")
            self._processing.discard(file_path)
            return

        self._trigger_ingestion(file_path, ext)
        self._processing.discard(file_path)

    def _trigger_ingestion(self, file_path: str, ext: str):
        """Run the appropriate ingestion pipeline and rebuild BM25."""
        source_type = SUPPORTED_EXTENSIONS[ext]
        logger.info(f"Triggering {source_type} ingestion for: {file_path}")

        try:
            if source_type == "pdf":
                summary = ingest_pdf(file_path)
                build_bm25_index(CHROMA_COLLECTION_PDF, force_rebuild=True)

            elif source_type == "excel":
                summary = ingest_excel(file_path)
                build_bm25_index(CHROMA_COLLECTION_EXCEL, force_rebuild=True)

            logger.info(f"Auto-ingestion complete: {summary}")

        except Exception as e:
            logger.error(
                f"Auto-ingestion failed for {file_path}: {e}",
                exc_info=True,
            )


def start_file_watcher():
    """
    Start watching the uploads directory for new files.
    Blocks indefinitely — run in a background thread or separate process.
    """
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    logger.info(f"Starting file watcher on: {UPLOADS_DIR}")

    handler  = UploadEventHandler()
    observer = Observer()
    observer.schedule(handler, path=UPLOADS_DIR, recursive=False)
    observer.start()

    logger.info("File watcher running. Drop PDF or Excel files into uploads/ to auto-ingest.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("File watcher stopped.")
        observer.stop()

    observer.join()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    start_file_watcher()