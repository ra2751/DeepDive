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


class IngestionLimitError(RuntimeError):
    """An upload exceeds the ingestion budget (file size, chunk count or embedding requests)."""


class IngestionCompatibilityError(RuntimeError):
    """The target collection cannot safely receive Mesh embeddings."""

class MultiPDFPipeline:

    def __init__(self, settings=None, budget=None):
        from config.settings import get_settings
        from budget import new_ingestion_budget

        self.all_text_pages = []
        self.settings = settings or get_settings()
        # Ingestion budget is separate from research-run budgets
        self.budget = budget or new_ingestion_budget(self.settings)

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

        max_bytes = int(self.settings.max_upload_mb * 1024 * 1024)
        for pdf_path in pdf_files:
            if Path(pdf_path).stat().st_size > max_bytes:
                raise IngestionLimitError(
                    f"{Path(pdf_path).name} exceeds the {self.settings.max_upload_mb:g} MB upload limit"
                )

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

        # Enforce the ingestion budget before any embedding request is sent
        max_chunks = self.settings.max_ingest_chunks
        if len(chunks) > max_chunks:
            raise IngestionLimitError(
                f"Upload produced {len(chunks)} chunks; the limit is {max_chunks} "
                f"({self.settings.max_ingest_embedding_calls} embedding requests x "
                f"{self.settings.embedding_batch_size} chunks). Nothing was indexed."
            )
        if self.settings.milvus_embedding_compat == "incompatible":
            raise IngestionCompatibilityError(
                "Mesh embeddings are marked incompatible with the existing collection; "
                "refusing to mix vectors. Nothing was indexed."
            )

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
        embedding_generator = EmbeddingGenerator(budget=self.budget, settings=self.settings)
        embeddings = embedding_generator.generate_embeddings(
            chunk_texts
        )
        usage = self.budget.snapshot()
        logger.info(
            f"MultiPDFPipeline | Embeddings generated: {len(embeddings)} | "
            f"Mesh embedding requests: {usage.mesh_embedding_calls}/{self.settings.max_ingest_embedding_calls}"
        )

        # -------------------------------------------------
        # CONNECT TO MILVUS CLOUD
        # -------------------------------------------------
        logger.info(f"MultiPDFPipeline | Connecting to Milvus Cloud...")
        milvus = MilvusDB()


        # Attaches to the existing collection if present (never drops it)
        collection = milvus.create_collection(collection_name=self.settings.milvus_collection)

        existing_dim = next(
            (f.params.get("dim") for f in collection.schema.fields if f.name == "embedding"), None
        )
        if existing_dim is not None and int(existing_dim) != self.settings.embedding_dim:
            raise IngestionCompatibilityError(
                f"Collection embedding dimension {existing_dim} != Mesh embedding dimension "
                f"{self.settings.embedding_dim}. Nothing was indexed."
            )

        # -------------------------------------------------
        # CREATE INDEX (only if none exists; existing indexes are preserved)
        # -------------------------------------------------
        
        created = IndexManager.ensure_hnsw_index(collection)

        logger.info(f"MultiPDFPipeline | HNSW index {'created' if created else 'already present, preserved'}")

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