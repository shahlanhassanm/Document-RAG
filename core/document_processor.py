"""
Multi-format document processor.
Supports: PDF, DOCX, PPTX, XLSX, CSV, TXT
Each format extracts text with metadata (filename, page/slide/sheet number).
"""

import os
import csv
import io
from langchain_core.documents import Document
from core.logger import get_logger

log = get_logger("document_processor")


def process_file(file_path: str, original_name: str = None) -> list[Document]:
    """
    Extract text from a file and return a list of Document objects with metadata.
    Each document chunk corresponds to a page/slide/sheet for traceability.
    """
    ext = os.path.splitext(file_path)[1].lower()
    name = original_name or os.path.basename(file_path)
    log.info(f"Processing file: {name} (type: {ext})")

    try:
        if ext == '.pdf':
            docs = _process_pdf(file_path, name)
        elif ext == '.docx':
            docs = _process_docx(file_path, name)
        elif ext == '.pptx':
            docs = _process_pptx(file_path, name)
        elif ext in ('.xlsx', '.xls'):
            docs = _process_excel(file_path, name)
        elif ext == '.csv':
            docs = _process_csv(file_path, name)
        elif ext == '.txt':
            docs = _process_txt(file_path, name)
        else:
            log.warning(f"Unsupported file type: {ext}")
            raise ValueError(f"Unsupported file format: {ext}")

        log.info(f"Extracted {len(docs)} document segments from {name}")
        return docs

    except Exception as e:
        log.error(f"Failed to process {name}: {e}", exc_info=True)
        raise


def _process_pdf(file_path: str, name: str) -> list[Document]:
    """Extract text page-by-page from PDF using PyMuPDF."""
    import fitz
    docs = []
    pdf = fitz.open(file_path)
    for page_num, page in enumerate(pdf, 1):
        text = page.get_text().strip()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"source": name, "page": page_num, "type": "pdf"}
            ))
    pdf.close()
    log.debug(f"PDF '{name}': {len(docs)} pages with text")
    return docs


def _process_docx(file_path: str, name: str) -> list[Document]:
    """Extract paragraphs from DOCX."""
    import docx
    doc = docx.Document(file_path)
    full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
    if not full_text.strip():
        return []
    return [Document(
        page_content=full_text,
        metadata={"source": name, "type": "docx"}
    )]


def _process_pptx(file_path: str, name: str) -> list[Document]:
    """Extract text slide-by-slide from PPTX."""
    from pptx import Presentation
    docs = []
    prs = Presentation(file_path)
    for slide_num, slide in enumerate(prs.slides, 1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = paragraph.text.strip()
                    if text:
                        texts.append(text)
        if texts:
            docs.append(Document(
                page_content="\n".join(texts),
                metadata={"source": name, "slide": slide_num, "type": "pptx"}
            ))
    log.debug(f"PPTX '{name}': {len(docs)} slides with text")
    return docs


def _process_excel(file_path: str, name: str) -> list[Document]:
    """Extract data sheet-by-sheet from Excel, preserving table structure."""
    from openpyxl import load_workbook
    docs = []
    wb = load_workbook(file_path, read_only=True, data_only=True)
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = []
        for row in ws.iter_rows(values_only=True):
            # Convert each cell to string, handle None
            row_text = " | ".join([str(cell) if cell is not None else "" for cell in row])
            if row_text.strip().replace("|", "").strip():
                rows.append(row_text)
        if rows:
            # First row as header, rest as data
            content = f"Sheet: {sheet_name}\n" + "\n".join(rows)
            docs.append(Document(
                page_content=content,
                metadata={"source": name, "sheet": sheet_name, "type": "excel"}
            ))
    wb.close()
    log.debug(f"Excel '{name}': {len(docs)} sheets with data")
    return docs


def _process_csv(file_path: str, name: str) -> list[Document]:
    """Extract data from CSV, preserving table structure."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        rows = []
        for row in reader:
            row_text = " | ".join(row)
            if row_text.strip():
                rows.append(row_text)
    if not rows:
        return []
    content = "\n".join(rows)
    return [Document(
        page_content=content,
        metadata={"source": name, "type": "csv"}
    )]


def _process_txt(file_path: str, name: str) -> list[Document]:
    """Read plain text file."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read().strip()
    if not text:
        return []
    return [Document(
        page_content=text,
        metadata={"source": name, "type": "txt"}
    )]


def process_batch(file_paths: list[tuple[str, str]]) -> list[Document]:
    """
    Process a batch of files.
    Args: list of (file_path, original_name) tuples
    Returns: combined list of Document objects from all files
    """
    all_docs = []
    log.info(f"Starting batch processing of {len(file_paths)} files")
    for file_path, original_name in file_paths:
        try:
            docs = process_file(file_path, original_name)
            all_docs.extend(docs)
        except Exception as e:
            log.error(f"Skipping {original_name}: {e}")
    log.info(f"Batch complete: {len(all_docs)} total document segments from {len(file_paths)} files")
    return all_docs
