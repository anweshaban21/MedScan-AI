"""
utils.py – extract plain text from uploaded files
Supports: .txt, .pdf, .png / .jpg / .jpeg (via pytesseract OCR)
"""

from __future__ import annotations
import io


def extract_text_from_file(uploaded_file) -> str:
    """Return plain text from a Streamlit UploadedFile object."""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    # ── Plain text ────────────────────────────────────────────────────────────
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")

    # ── PDF ───────────────────────────────────────────────────────────────────
    if name.endswith(".pdf"):
        return _extract_pdf(data)

    # ── Images ────────────────────────────────────────────────────────────────
    if name.endswith((".png", ".jpg", ".jpeg")):
        return _extract_image(data)

    return ""


def _extract_pdf(data: bytes) -> str:
    try:
        import pdfplumber
        text_parts: list[str] = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        full = "\n".join(text_parts).strip()
        if full:
            return full
    except Exception:
        pass

    # Fallback: PyMuPDF
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        return "\n".join(page.get_text() for page in doc).strip()
    except Exception:
        pass

    return ""


def _extract_image(data: bytes) -> str:
    try:
        from PIL import Image
        import pytesseract
        image = Image.open(io.BytesIO(data))
        return pytesseract.image_to_string(image)
    except Exception as e:
        return f"[OCR failed: {e}]"