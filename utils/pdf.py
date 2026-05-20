"""PDF text extraction for resumes."""
from io import BytesIO
from pypdf import PdfReader


def extract_resume_text(uploaded_file) -> str:
    """Extract clean plain text from a Streamlit-uploaded PDF.

    Strips empty pages, collapses excessive whitespace, and limits the result
    to a reasonable token budget so it fits in the LLM context.
    """
    raw = uploaded_file.read() if hasattr(uploaded_file, "read") else uploaded_file
    reader = PdfReader(BytesIO(raw))

    chunks = []
    for page in reader.pages:
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            chunks.append(text)

    full = "\n\n".join(chunks)
    # Collapse runs of blank lines for a tighter prompt.
    while "\n\n\n" in full:
        full = full.replace("\n\n\n", "\n\n")
    return full.strip()
