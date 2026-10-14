from monitoring.logger import setup_logger
from agents.state_node import AgentRagState
from config.config import Config
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

logger = setup_logger()

def WebSearchAgent(state:AgentRagState):
    """
    Role: Active web search researcher. Executes a web query when local docs
    are missing information on non AI related topics
    
    """

    question = state['query']
    logger.info(f"WebSearchAgent|Query:{question}")


    TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
    

    if not TAVILY_API_KEY:
        logger.error(f"WebSearchAgent | TAVILY_API_KEY environment variable is missing!")
        # Optional: Fail gracefully or throw an explicit instruction error

    # execute live web search
    search_tool = TavilySearchResults(max=3,TAVILY_API_KEY=TAVILY_API_KEY)
    search_results = search_tool.invoke({"query":question})
    logger.info(f"WebSearchAgent|Search Results:{search_results}")

    # Tavily returns a list of dicts with 'url' and 'content'
    retrieved_docs = [
        {"content": result["content"], "source": result.get("url","Web Search")}
        for result in search_results
    ]

    # convert to string context for the prompt
    formatted_docs = "\n\n".join(
        [
            f"Source: {d['source']}\nContent: {d['content']}"   
            for d in retrieved_docs
        ]
    )

    # Generate Response
    template = """
    You are an expert research assistant answering a query using real-time web search results.
    Provide a comprehensive, factual answer based strictly on the provided web snippets.

    User Query: {question}

    Web Search Context:
    {formatted_docs}

    Please provide a professional, verified answer.
    """

    prompt = PromptTemplate.from_template(template)

    config = Config()
    model = config.get_llm()
    chain = prompt | model | StrOutputParser()

    response = chain.invoke({
        "question":question,"formatted_docs":formatted_docs
    })

    logger.info(f"WebSearchAgent| Response Generated Successfully")

    return {
        "retrieved_docs": retrieved_docs,
        "messages":[AIMessage(content=response)]
    }