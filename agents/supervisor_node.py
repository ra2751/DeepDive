
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from agents.state_node import AgentRagState, topic_selection_parser
from budget import BudgetExceeded, budget_from_config
from gateway import MeshError, chat_model
from monitoring.logger import setup_logger


logger = setup_logger()

OUT_OF_SCOPE_MESSAGE = "This Research Intelligence Platform is specialized for AI, Generative AI, and Agentic AI research use cases."


def _out_of_scope(current_loops):
    return {
        "messages": [AIMessage(content=OUT_OF_SCOPE_MESSAGE)],
        "next_step": "end_node",
        "loop_count": current_loops,
        "out_of_scope": True,
        "draft_answer": OUT_OF_SCOPE_MESSAGE,
        "final_response": "This research assistance can only be for AI related topics"
    }


def _terminate(current_loops, reason):
    logger.warning(f"SupervisorAgent | Stopping run: {reason}")
    return {"next_step": "end_node", "loop_count": current_loops, "termination_reason": reason}


def SupervisorAgent(state:AgentRagState, config: RunnableConfig):
    """
    The SupervisorAgent oversees the routing workflow. It analyzes the query and the 
    validation history to decide the content topic category and explicitly commands 
    which target agent node should execute.
    """

    question = state['query']
    budget = budget_from_config(config)
    logger.info("SupervisorAgent received query")

    # Track and increment the sequence pass immediately upon entry
    current_loops = state.get('loop_count', 0) + 1
    logger.info(f"SupervisorAgent received query. Iteration Loop: {current_loops}") 

    # check if this agent is called for re routing due to a validation failure
    validation_metadata = state.get('validation_metadata',{}) 
    is_valid_signal = str(validation_metadata.get('is_valid', 'yes')).lower().strip()
    is_retry = is_valid_signal == "no"
    failure_reason = validation_metadata.get('reasoning',"")

    # --- retry and cost protection ---
    # Retry rounds are bounded by the BudgetManager (MAX_RETRY_ROUNDS). The router only
    # sends us back here while rounds remain; a failed validation is never flipped to "yes".
    if is_retry:
        try:
            round_no = budget.next_retry_round()
            logger.info(f"SupervisorAgent | Retry round {round_no}/{budget.limits.max_retry_rounds}")
        except BudgetExceeded as e:
            return _terminate(current_loops, e.reason)
    

    # --- DOMAIN GUARDRAIL PRE-CHECK ---
    # Static checking pattern to prevent out-of-domain processing costs before reaching the LLM
    non_ai_triggers = ["weather", "politics", "election", "sports", "football", "cricket", "entertainment","finance","relationships"]
    if any(word in question.lower() for word in non_ai_triggers):
        logger.info("SupervisorAgent | Out-of-scope topic caught by keyword pre-check.")
        return _out_of_scope(current_loops)
    
    # ROUTING STRATEGY PROMPT
    #strategy_guidance_old = """
    #=========================================
    #CRITICAL ROUTING & DOMAIN RULES (STRICT)
    #=========================================
    #1. DOMAIN GUARDRAIL:
    #   - Evaluate the user's intent immediately. This assistant is strictly restricted to Machine Learning, 
    #   Deep Learning, and Artificial Intelligence domains.
    #   - If the query focuses on out-of-scope topics (e.g., politics, weather, sports, general entertainment, 
    #   or casual chat unrelated to technology), you MUST immediately select route: 'end_node' and set your 
    #   classification topic to 'Others'.

    #2. ROUTING PRIORITY (TURN 1):
    #   - If the query explicitly references source documentation (e.g., "attached papers", "survey documents", 
    #   "in the PDFs") OR targets explicit structural/mathematical neural network sub-components (e.g., "ODE solvers", 
    #   "attention weights", "feed-forward networks", "embeddings"), you MUST classify the topic as 'AI Transformer 
    #   Related' and select route: 'rag_agent'.
    #   - MANDATORY: Do NOT bypass the 'rag_agent' on Turn 1 just because the query demands a creative delivery wrapper 
    #   (such as a dialogue, analogy, podcast script, or story). Content origin takes absolute priority over formatting style.
    #"""

    strategy_guidance = """
    =========================================
    CRITICAL ROUTING & DOMAIN RULES (STRICT)
    =========================================
    1. DOMAIN GUARDRAIL:
       - Evaluate the user's intent immediately. This assistant is strictly restricted to Machine Learning, Deep Learning, and Artificial Intelligence domains.
       - If the query focuses on out-of-scope topics (e.g., politics, weather, sports, general entertainment, or casual chat unrelated to technology), you MUST immediately select route: 'end_node' and set your classification topic to 'Others'. Do not process further.

    2. ROUTING PRIORITY (TURN 1):
       - HIGH-LEVEL CONCEPTUAL DEFINITIONS: If the query is a basic, introductory definition request (e.g., "what is a transformer", "explain neural networks simply", "what does LLM stand for"), you MUST classify the topic as 'AI Non-Transformer Related' or 'AI Transformer Related' depending on context, but you MUST select route: 'general_llm_agent'. Do not send basic introductory definitions to RAG, as advanced research papers lack elementary definitions.
       
       - DEEP TECHNICAL / SPECIFIC ARCHITECTURES: If the query explicitly references source documentation (e.g., "attached papers", "survey documents", 
         "in the PDFs") OR targets explicit structural, mathematical, or advanced neural network sub-components (e.g., "ODE solvers", "attention weights",
         "feed-forward networks", "embeddings", "LoFTR", "multi-head attention blocks"), you MUST classify the topic as 'AI Transformer Related' and 
         select route: 'rag_agent'. (EXCEPTION: If the query is just a basic, introductory request for a transformer definition, 
         you MUST follow the HIGH-LEVEL CONCEPTUAL rule above and choose 'general_llm_agent').

       - FACTUAL LOOKUP / CODE / IMPLEMENTATION: If the query requires factual lookup, real-world data, implementation code scripts, or broad external context, you MUST classify the topic appropriately and select route: 'web_search_agent'.

    3. MANDATORY FORMATTING OVERRIDE RULE:
       - Do NOT bypass the assigned execution agent on Turn 1 just because the query demands a creative delivery wrapper (such as a dialogue, analogy, podcast script, or story). Content and technical origin (e.g., high-level request vs. deep paper grounding) take absolute priority over formatting style.
    """

    # If this is a retry due to validation failure, switch
    # include instructions to avoid routing back to the same agent if its data source was insufficient.
    if is_retry:
        strategy_guidance = f"""
        
        ### CRITICAL RETRY NOTICE (FALLBACK MODE) ###
        A previous attempt to handle this query FAILED validation.
        Reason for failure: {failure_reason}

        SYSTEM DIRECTIVE: 
        The local 'rag_agent' is completely frozen. You are strictly FORBIDDEN from choosing 'rag_agent'. 
        You must look at the user's current goal to choose between WEB SEARCH and GENERAL LLM:

        1. Route to 'general_llm_agent' (INTERNAL REASONING) if:
           - The user is asking for text generation, explanations, brainstorming, analogies, summaries, formatting, or theoretical conceptual reasoning.
           - Even if they use technical keywords (like attention, matrices, or transformers), if they are asking to "explain", "conceptualize", "summarize", or "format", choose 'general_llm_agent'. Do not search the web for standard baseline textbook theories.

        2. Route to 'web_search_agent' (EXTERNAL FACTS) ONLY if:
           - The user explicitly asks for code implementations, functional scripts (Python, PyTorch), recent market releases, live benchmarks, or real-time context.
        """

        logger.info(f"SupervisorAgent| retry_instruction: {failure_reason}")

    template="""
    You are the central conductor orchestration agent for a research multi-agent system.
    Your task is to analyze the user request and determine both the core subject category and the next explicit node route.

    User query: {question}

    ### Routing Constraints
    {strategy_guidance}

    Select one topic category from: [AI Transformer Related, AI Non-Transformer Related, Others]
    Select one next execution route from: ['rag_agent', 'web_search_agent', 'general_llm_agent','end_node']

    {format_instructions}
    """

    # Create a prompt template with the specified template and input variables
    prompt = PromptTemplate.from_template(
        template = template,
        #input_variables=["question"],
        partial_variables={"format_instructions": topic_selection_parser.get_format_instructions()} 
        )
    
    # Model via the Mesh API gateway (all inference goes through Mesh)
    model = chat_model("supervisor", budget)

    # Combine the prompt, model, and parser into a single chain
    chain = prompt | model | topic_selection_parser

    try:
        response = chain.invoke(
            {   "question":question,
                "strategy_guidance": strategy_guidance
            }
        )
        selected_topic = response.topic
        chosen_route = response.next_route
        reasoning = response.reasoning

    except BudgetExceeded as e:
        return _terminate(current_loops, e.reason)
    except MeshError as e:
        logger.error(f"SupervisorAgent | Mesh call failed: {type(e).__name__}")
        return _terminate(current_loops, "mesh_error")
    except Exception as e:
        logger.info(f"SupervisorAgent failed to parse structured output: {type(e).__name__}")
        selected_topic = "AI_Technical_Fallback"
        chosen_route = "general_llm_agent"
        reasoning = "Fallback triggered due to structural parsing error."

    # Guard against routes the graph does not define (e.g. 'finalize'), which would crash
    if chosen_route not in ("rag_agent", "web_search_agent", "general_llm_agent", "end_node"):
        logger.warning(f"SupervisorAgent | Unknown route {chosen_route!r}; using general_llm_agent")
        chosen_route = "general_llm_agent"

    # Intercept out-of-scope LLM classifications to deliver the required block response
    if chosen_route == "end_node" or selected_topic == "Others":
        logger.info("SupervisorAgent | Out-of-scope routing confirmed by LLM classification.")
        
        return _out_of_scope(current_loops)

    logger.info(
            f"SupervisorAgent | Topic:{selected_topic} | Reason: {reasoning}| Route Selected: {chosen_route} | Retry: {is_retry}"
    )

    return {
        "messages": [AIMessage(content=f"Topic: {selected_topic}. Reason: {reasoning}")],  
        "next_step": chosen_route,
        "loop_count":current_loops # Sync updated loop index back into the global graph state
    }