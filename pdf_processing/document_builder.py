"""
The DocumentBuilder class is responsible for converting the extracted text 
from PDF pages into a format that can be easily used by the agent.
It takes the extracted text, 
along with its associated metadata (such as source and page number),    
and converts them into Document objects that can be processed by the LangChain framework.

LangChain's search tools (Vector Stores) cannot read raw Python dictionaries.

They only accept these standardized Document objects.
"""

from langchain_core.documents import Document


class DocumentBuilder:

    @staticmethod
    def build_documents(text_pages):

        documents = []

        for item in text_pages:

            doc = Document(
                page_content=item["content"],
                metadata={
                    "source": item["source"],
                    "page": item["page"]
                }
            )

            documents.append(doc)

        return documents