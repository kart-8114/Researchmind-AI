import fitz  # PyMuPDF

def extract_pdf_pages(path):
    """Extract page-by-page text from PDF using PyMuPDF with provenance."""
    full_text = ""
    analysis_text = ""
    page_texts = {}
    try:
        with fitz.open(path) as doc:
            page_count = len(doc)
            for i in range(min(15, page_count)):
                page_text = doc[i].get_text()
                page_texts[i + 1] = page_text
                full_text += page_text
                if i < 10:
                    analysis_text += page_text
    except Exception as e:
        print(f"PDF extraction error: {e}")
    return full_text, analysis_text, page_texts
