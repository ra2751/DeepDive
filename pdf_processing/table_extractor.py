import camelot
from pathlib import Path

""" 
The PDFTableExtractor class provides functionality to extract tables 
from PDF documents and save them as CSV files. 
It also generates metadata for each extracted table, 
which can be used for tracking and referencing the extracted tables. 

This metadata can be useful for agents to link back to the source PDF
and the specific table extracted when a user asks questions about the tables in the PDF. 

"""

class PDFTableExtractor:

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract_tables(self, output_dir="data/extracted/tables"):
        # Ensure the output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract tables from the PDF using Camelot
        tables = camelot.read_pdf(
            self.pdf_path,
            pages="all"
        )

        metadata = []

        # Save each extracted table as a CSV file and create metadata
        for idx, table in enumerate(tables):

            file_name = (
                f"{Path(self.pdf_path).stem}_table_{idx + 1}.csv"
            )

            save_path = output_dir / file_name

            table.df.to_csv(save_path, index=False)

            # create metadata for each extracted table
            # This metadata can be used for tracking and referencing the extracted tables
            # Agent can use this metadata to link back to the source PDF and the specific table 
            # extracted when a user asks question about the tables in the PDF
            metadata.append({
                "table_id": idx + 1,
                "source": Path(self.pdf_path).name,
                "path": str(save_path)
            })

        return metadata