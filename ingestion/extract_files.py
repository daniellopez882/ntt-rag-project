"""PDF extraction with Docling.

Docling (and the torch stack under it) is imported when an extractor is
built, not when this module is: the API never needs it, and the previous
module both imported it at the top and swallowed a failed converter
initialisation, leaving ``self.converter`` unset until ``extract_pdf_content``
crashed with an unrelated ``AttributeError``.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class PDFExtractionConfig:
    enable_ocr: bool = True
    images_scale: float = 2.0
    include_images: bool = True
    include_tables: bool = True


class Extractor(Protocol):
    """What the ingestion pipeline needs from an extractor."""

    def extract_pdf_content(self, pdf_path: str) -> tuple[str, dict[str, Any]]: ...


class PDFExtractor:
    """Docling-backed extractor. Raises at construction if Docling is missing or broken."""

    def __init__(self, config: PDFExtractionConfig | None = None) -> None:
        self.config = config or PDFExtractionConfig()
        self.converter = self._build_converter()

    def _build_converter(self) -> Any:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError as exc:
            raise RuntimeError(
                "Docling is not installed; install the project with the [ingest] extra"
            ) from exc
        options = PdfPipelineOptions()
        options.do_ocr = self.config.enable_ocr
        options.do_picture_description = self.config.include_images
        options.do_table_structure = self.config.include_tables
        options.images_scale = self.config.images_scale
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

    def extract_pdf_content(self, pdf_path: str) -> tuple[str, dict[str, Any]]:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        logger.info("Extracting content from %s", path.name)
        started = time.time()
        doc = self.converter.convert(str(path)).document
        metadata = {
            "source": str(path),
            "title": path.stem,
            "processing_time": round(time.time() - started, 2),
            "pages": len(doc.pages),
            "texts": len(doc.texts),
            "pictures": len(doc.pictures),
            "tables": len(doc.tables),
            "extraction_method": "docling",
            "content_type": "pdf",
        }
        return doc.export_to_markdown(), metadata


def create_pdf_extractor(config: PDFExtractionConfig | None = None) -> PDFExtractor:
    return PDFExtractor(config)
