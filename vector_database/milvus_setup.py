from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os
from dotenv import load_dotenv
from monitoring.logger import setup_logger


load_dotenv()
logger = setup_logger()

""" 
MilvusDB class for managing Milvus collections and connections. 
This class handles the connection to Milvus Cloud and 
create collections with the appropriate schema for 
storing document embeddings and metadata.
"""

class MilvusDB:
    # Initialize the connection to Milvus Cloud using environment variables
    def __init__(self):
        self.uri = os.getenv("Milvus_ENDPOINT")
        self.token = os.getenv("Milvus_API_KEY")
        self.name = None # Initialize collection name attribute
        self.collection = None # Initialize collection attribute

        connections.connect(
            alias="default",
            uri=self.uri,
            token=self.token
        )
        print("Connected to Milvus Cloud successfully")

    # Method to create a collection with the specified schema
    def create_collection(self,collection_name="ai_research_assistant"):    
        if utility.has_collection(collection_name):
            logger.info(f"Milvs DB|create collection|Collection '{collection_name}' already exists. Attaching and loading incrementally...")
            self.collection = Collection(name=collection_name)
            self.name = collection_name
            return self.collection
            # utility.drop_collection(collection_name)

        # if doesn't exist, build it for very first time
        logger.info(f"Milvs DB|create collection| Collection {collection_name} not found. Initializing new schema set up...")    

        # Define the collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),  # Assuming 1536-dimensional embeddings
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000), # Store the original text chunk
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="page", dtype=DataType.INT64)
        ]
        schema = CollectionSchema(fields, description="DeepDive AI document chunks")

        # Create the collection
        self.collection = Collection(name=collection_name, schema=schema)
        self.name = collection_name # Store the collection name for later use
        
        logger.info(f"Milvs DB|create collection|Collection '{self.name}' created successfully.")

        return self.collection
    
