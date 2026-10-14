"""
The PDFImageExtractor class provides functionality to extract images from PDF documents 
and save them as separate image files. 
It also generates metadata for each extracted image
which can be used for tracking and referencing the extracted images.
This metadata can be useful for agents to link back to the source PDF
and the specific image extracted when a user asks questions about the images in the PDF.

"""
import fitz
from pathlib import Path

class PDFImageExtractor:
    # Initialize the PDFImageExtractor with the path to the PDF document
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)

    # Extract images from each page of the PDF document and save them as separate image files.
    # Return a list of dictionaries containing the source, page number, 
    # and path to the extracted images

    def extract_images(self, output_dir="data/extracted/images"):

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_metadata = []

        # Iterate through each page in the PDF document, extract the images,
        # and save them as separate image files. Store the metadata for each extracted image
        # This metadata can be used for tracking and referencing the extracted images
        # Agent can use this metadata to link back to the source PDF and the specific image
        # extracted when a user asks question about the images in the PDF    
        for page_index in range(len(self.doc)):

            page = self.doc[page_index]

            # Extract images from the page using PyMuPDF's get_images method
            image_list = page.get_images(full=True)

            # Iterate through the list of images, extract the image data, 
            # and save it as a separate file

            for img_index, img in enumerate(image_list):

                xref = img[0]
                base_image = self.doc.extract_image(xref)

                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                image_name = (
                    f"{Path(self.pdf_path).stem}_"
                    f"page_{page_index + 1}_img_{img_index + 1}.{image_ext}"
                )

                image_path = output_dir / image_name

                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                # Create metadata for each extracted image
                # This metadata can be used for tracking and referencing the extracted images
                # Agent can use this metadata to link back to the source PDF and the specific image
                # extracted when a user asks question about the images in the PDF
                image_metadata.append({
                    "source": Path(self.pdf_path).name,
                    "page": page_index + 1,
                    "image_path": str(image_path)
                })

        return image_metadata