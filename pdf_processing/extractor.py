
""" The PDFTextExtractor class provides functionality to extract text from PDF documents."""

import fitz
from pathlib import Path


class PDFTextExtractor:

    # Initialize the PDFTextExtractor with the path to the PDF document
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    # Extract text from each page of the PDF document and return a list of dictionaries 
    # containing the source, page number, and extracted content
    def extract_text(self):

        extracted_pages = []

        # Iterate through each page in the PDF document, extract the text content,
        # and store it in a structured format
        for page_num, page in enumerate(self.doc):
            text = page.get_text("text")

            extracted_pages.append({
                "source": Path(self.pdf_path).name,
                "page": page_num + 1,
                "content": text
            })

        return extracted_pages
