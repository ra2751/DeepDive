from monitoring.logger import setup_logger
from agents.state_node import AgentRagState
from budget import BudgetExceeded, budget_from_config
from gateway import MeshError, chat_model
from retriever.web_search import get_web_search_provider
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

logger = setup_logger()

def WebSearchAgent(state:AgentRagState, config: RunnableConfig):
    """
    Role: Active web search researcher. Executes a web query when local docs
    are missing information on non AI related topics.

    Search goes through the governed WebSearchProvider (Tavily, a non-model
    provider) with per-run quotas, timeouts and bounded retries.
    """

    question = state['query']
    budget = budget_from_config(config)
    notices = []
    logger.info("WebSearchAgent| Query received")

    # execute live web search
    try:
        provider = get_web_search_provider(budget)
        search_results = provider.search(question)
    except BudgetExceeded as e:
        return {"termination_reason": e.reason, "retrieved_docs": [], "draft_answer": None}
    except Exception as e:  # quota, timeout, missing key, provider error
        logger.error(f"WebSearchAgent | Web search failed: {type(e).__name__}")
        search_results = []
        notices.append(f"Web search unavailable: {e}")

    logger.info(f"WebSearchAgent| {len(search_results)} search results")

    retrieved_docs = [
        {"content": result.content, "source": result.url, "retrieved_at": result.retrieved_at}
        for result in search_results
    ]

    if not retrieved_docs:
        # Do not spend a model call writing an answer without any web context
        answer = "Web search returned no usable results, so no web-grounded answer could be produced."
        return {"retrieved_docs": [], "messages": [AIMessage(content=answer)],
                "draft_answer": answer, "notices": notices}

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

    model = chat_model("generator", budget)
    chain = prompt | model | StrOutputParser()

    try:
        response = chain.invoke({
            "question":question,"formatted_docs":formatted_docs
        })
    except BudgetExceeded as e:
        return {"termination_reason": e.reason, "retrieved_docs": retrieved_docs, "draft_answer": None, "notices": notices}
    except MeshError as e:
        logger.error(f"WebSearchAgent | Mesh call failed: {type(e).__name__}")
        return {"termination_reason": "mesh_error", "retrieved_docs": retrieved_docs, "draft_answer": None,
                "notices": notices + [f"Answer generation failed: {e}"]}

    logger.info(f"WebSearchAgent| Response Generated Successfully")

    return {
        "retrieved_docs": retrieved_docs,
        "messages":[AIMessage(content=response)],
        "draft_answer": response,
        "notices": notices,
    }
