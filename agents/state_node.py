from pydantic import BaseModel , Field
from typing import List, Optional, TypedDict, Annotated, Sequence
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import BaseMessage
import operator

"""
This module defines the state structure for the RAG Agent, 
including the conversation history, retrieved documents, 
current query, and validation metadata. 

It also includes a Pydantic model for parsing topic selection 
responses from the Supervisor Agent.

"""

class TopicSelectionParser(BaseModel):
    topic: str = Field(
        description="The topic to be analyzed for relevance."
        )
    
    reasoning: str = Field(
        description="The reasoning behind the relevance of the topic."
        )
    
    next_route: str = Field(
        description="The explicit node to execute next. " \
        "Strict rule: If this is a retry due to validation failure, you MUST choose 'web_search_agent' or 'general_llm_agent'." \
        "Otherwise, use 'rag_agent' for document queries. " \
        "or used 'finalize' "
    )

topic_selection_parser = PydanticOutputParser(pydantic_object=TopicSelectionParser)

class AgentRagState(TypedDict):
     
    # The conversation history (Appends via operator.add)
    messages: Annotated[Sequence[BaseMessage], operator.add] 

    # Context from Milvus (Overwrites each turn to keep context fresh)   
    retrieved_docs: List[dict]

    # The current query (can be the raw input or a rewritten version)
    query: str

    # Metadata related to the validation process (e.g., validation results, reasoning, etc.)
    # "performed" is True only when the validator actually evaluated the current draft
    validation_metadata: dict

    # The answer produced by the last executed agent (rag / web / general llm)
    draft_answer: Optional[str]

    # Set when the supervisor rejects a query as out of scope
    out_of_scope: bool

    # Why the run stopped early (e.g. "max_llm_calls", "max_run_seconds"), if it did
    termination_reason: Optional[str]

    # Honest final label: answer_check_passed | ungrounded | not_validated |
    # incomplete_budget | out_of_scope (set by the finalize node, never upgraded)
    answer_status: str

    # Budget usage snapshot at the end of the run
    usage: dict

    # User-visible notices (errors, degraded features) accumulated during the run
    notices: Annotated[List[str], operator.add]

    # tracking final generated answer
    final_response: str 

    # add this tracking path key
    next_step: str

    # Tracks the iteration count to protect the budget
    loop_count: int  