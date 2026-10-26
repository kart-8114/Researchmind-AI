import unittest
from services.pdf_processor import extract_pdf_pages

class TestPdfProcessor(unittest.TestCase):
    def test_pdf_extractor_missing_file(self):
        full, analysis, pages = extract_pdf_pages("nonexistent.pdf")
        self.assertEqual(full, "")
        self.assertEqual(pages, {})

if __name__ == "__main__":
    unittest.main()
