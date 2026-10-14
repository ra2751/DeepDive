from pathlib import Path
from monitoring.logger import setup_logger
from pdf_processing.extractor import PDFTextExtractor
from pdf_processing.table_extractor import PDFTableExtractor
from pdf_processing.image_extractor import PDFImageExtractor
from pdf_processing.document_builder import DocumentBuilder
from pdf_processing.chunking import TextChunker

from pdf_processing.embeddings import EmbeddingGenerator
from vector_database.milvus_setup import MilvusDB
from vector_database.indexing import IndexManager


RAW_PDF_DIR = "data/raw_pdfs"

logger = setup_logger()

class MultiPDFPipeline:

    def __init__(self):
        self.all_text_pages = []

    # =========================================================
    # STEP 1 — PROCESS ALL PDFs
    # =========================================================
    # This method processes all PDF files in the specified directory. 
    # For each PDF, it performs text extraction, table extraction, and image extraction.
    def process_pdfs(self,single_pdf_path: Path = None):
        # If a single path is provided, isolate execution to just that file; else look at directory
        if single_pdf_path:
            pdf_files = [single_pdf_path] if single_pdf_path.exists() else []
        else:
            pdf_files = list(Path(RAW_PDF_DIR).glob("*.pdf"))
        
        #pdf_files = list(Path(RAW_PDF_DIR).glob("*.pdf"))

        if not pdf_files:
            print("No PDF files found.")
            return

        print(f"\nTargeting {len(pdf_files)} PDF file(s) for parsing operations.")

        for pdf_path in pdf_files:
            print("\n" + "=" * 60)
            print(f"Processing PDF: {pdf_path.name}")
            print("=" * 60)

            # -------------------------------------------------
            # TEXT EXTRACTION
            # -------------------------------------------------
            try:
                text_extractor = PDFTextExtractor(str(pdf_path))
                pages = text_extractor.extract_text()
                self.all_text_pages.extend(pages)
                print(f"Text pages extracted: {len(pages)}")
            except Exception as e:
                print(f"Text extraction failed for {pdf_path.name}")
                print(e)

            # -------------------------------------------------
            # TABLE EXTRACTION
            # -------------------------------------------------
            try:
                table_extractor = PDFTableExtractor(str(pdf_path))
                tables = table_extractor.extract_tables()
                print(f"Tables extracted: {len(tables)}")
            except Exception as e:
                print(f"Table extraction failed for {pdf_path.name}")
                print(e)

            # -------------------------------------------------
            # IMAGE EXTRACTION
            # -------------------------------------------------
            try:
                image_extractor = PDFImageExtractor(str(pdf_path))
                images = image_extractor.extract_images()
                print(f"Images extracted: {len(images)}")
            except Exception as e:
                print(f"Image extraction failed for {pdf_path.name}")
                print(e)

    # =========================================================
    # STEP 2 — BUILD VECTOR PIPELINE
    # =========================================================
    def build_vector_pipeline(self):
        
        if not self.all_text_pages:
            print("No text data gathered to chunk or insert.")
            return False


        # -------------------------------------------------
        # BUILD DOCUMENTS
        # -------------------------------------------------
        print("\nBuilding LangChain documents...")
        documents = DocumentBuilder.build_documents(
            self.all_text_pages
        )
        print(f"Documents created: {len(documents)}")

        # -------------------------------------------------
        # CHUNKING
        # -------------------------------------------------
        
        chunker = TextChunker(
            chunk_size=500,
            chunk_overlap=100
        )
        chunks = chunker.split_documents(documents)
        logger.info(f"MultiPDFPipeline | Total chunks created: {len(chunks)}")

        # -------------------------------------------------
        # PREPARE CHUNK DATA
        # -------------------------------------------------
        print("\nPreparing chunk metadata...")

        chunk_texts = []
        chunk_sources = []
        chunk_pages = []

        for chunk in chunks:
            chunk_texts.append(chunk.page_content)
            source_val =  chunk.metadata.get("source", "unknown")
            chunk_sources.append(Path(source_val).name)
            chunk_pages.append(
                chunk.metadata.get("page", -1)
            )

        # -------------------------------------------------
        # GENERATE EMBEDDINGS
        # -------------------------------------------------
        logger.info(f"MultiPDFPipeline | Generating embeddings...")
        embedding_generator = EmbeddingGenerator()
        embeddings = embedding_generator.generate_embeddings(
            chunk_texts
        )
        logger.info(f"MultiPDFPipeline | Embeddings generated: {len(embeddings)}")

        # -------------------------------------------------
        # CONNECT TO MILVUS CLOUD
        # -------------------------------------------------
        logger.info(f"MultiPDFPipeline | Connecting to Milvus Cloud...")
        milvus = MilvusDB()


        collection = milvus.create_collection(collection_name="ai_research_assistant")

        # -------------------------------------------------
        # CREATE INDEX
        # -------------------------------------------------
        
        IndexManager.create_hnsw_index(collection)

        logger.info(f"MultiPDFPipeline | HNSW index created successfully")

        # -------------------------------------------------
        # INSERT DATA INTO MILVUS
        # -------------------------------------------------
        logger.info(f"MultiPDFPipeline | Inserting vectors into Milvus...")

        insert_data = [
            embeddings,
            chunk_texts,
            chunk_sources,
            chunk_pages
        ]

        collection.insert(insert_data)

        logger.info(f"MultiPDFPipeline | Data inserted successfully")

        # -------------------------------------------------
        # LOAD COLLECTION
        # -------------------------------------------------
        collection.load()

        logger.info(f"MultiPDFPipeline | Collection reloaded successfully for immediate RAG querying")
        #print("\nCollection loaded successfully")
        return True


    # =========================================================
    # RUN COMPLETE PIPELINE
    # =========================================================
    def run(self,single_pdf_path:Path=None):
        self.process_pdfs(single_pdf_path=single_pdf_path)
        status = self.build_vector_pipeline()
        return status


# =============================================================
# MAIN - this method is when running it manually. 
# =============================================================
# if __name__ == "__main__":

   #pipeline = MultiPDFPipeline()
   #pipeline.run()