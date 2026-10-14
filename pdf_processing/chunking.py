""" 
The TextChunker class provides functionality to split large text documents 
into smaller chunks.This is important for processing and embedding the text 
in vector databases, as many embedding models have a maximum token limit.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter



class TextChunker:

    def __init__(self,
                 chunk_size=500,
                 chunk_overlap=100):

        # Initialize the TextChunker with the specified chunk size and overlap.
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )
    # Split the input documents into smaller chunks using the RecursiveCharacterTextSplitter.
    def split_documents(self, documents):
        return self.splitter.split_documents(documents)