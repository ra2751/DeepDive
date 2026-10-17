"""
Answer-level chat workflow (fallback for testing and rollback).

It runs the answer-level graph (supervisor -> agent -> validator -> finalize) and is
NOT claim-level verification.
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError

from agents.final_response_node import STATUS_TEXT
from budget import BudgetConfigError, format_usage, new_research_budget
from budget.manager import UsageLedger
from config.settings import ConfigError, get_settings
from graph_builder.graph_builder import build_graph


@st.cache_resource
def get_agent_graph():
    return build_graph()


def render_answer_status(status, termination_reason=None):
    text = STATUS_TEXT.get(status, status or "Unknown status")
    if termination_reason and status != "answer_check_passed":
        text += f" — reason: {termination_reason}"
    if status == "answer_check_passed":
        st.success(text)
    elif status == "out_of_scope":
        st.info(text)
    else:
        st.warning(text)


def render_legacy():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # Display historical chat conversations
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("User"):
                st.markdown(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("Assistant"):
                meta = getattr(message, 'metadata', None) or {}
                if meta.get('answer_status'):
                    render_answer_status(meta['answer_status'], meta.get('termination_reason'))
                st.markdown(message.content)
                if meta.get('usage_line'):
                    st.caption(meta['usage_line'])
            
                # --- RENDER HISTORICAL SOURCES (NAME ONLY) ---
                if meta.get('citations'):
                    with st.expander("🔗 Sources consulted"):
                        seen_sources_history = set()
                        for src in message.metadata['citations']:
                            source_name = src.get('source')
                            if source_name and source_name not in seen_sources_history:
                                seen_sources_history.add(source_name)
                                st.markdown(f"**Source:** `{source_name}`")

    # ==========================================
    # 6. STREAMING GRAPH RUNTIME LOGIC
    # ==========================================
    if user_query := st.chat_input("Ask research assistant..."):
    
        # Render User Turn Immediately
        with st.chat_message("User"):
            st.markdown(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        # Render Assistant Loading Turn
        with st.chat_message("Assistant"):
            response_placeholder = st.empty()
        
            # Telemetry updates for UI
            st.session_state.system_logs.append(f"Received query: '{user_query}'")
            st.session_state.system_logs.append("Invoking primary SupervisorAgent routing algorithm...")
        
            # Invoke your compiled LangGraph pipeline under a fresh per-question budget
            final_output = None
            try:
                settings = get_settings()
                budget = new_research_budget(settings)
                with st.spinner("Agents coordinating strategy..."):
                    initial_state = {
                        "query": user_query, 
                        "messages": st.session_state.chat_history, 
                        "loop_count": 0,
                        # A fresh question has not been validated yet (never pre-set to "yes")
                        "validation_metadata": {"is_valid": "pending", "reasoning": "", "performed": False},
                        "draft_answer": None,
                        "out_of_scope": False,
                        "termination_reason": None,
                        "notices": [],
                    }
                    final_output = get_agent_graph().invoke(
                        initial_state,
                        config={"configurable": {"budget": budget}, "recursion_limit": 20},
                    )
            except (ConfigError, BudgetConfigError) as config_error:
                response_placeholder.error(f"Configuration error: {config_error}")
            except GraphRecursionError:
                response_placeholder.error("The run was stopped by the graph safety limit before producing an answer.")

            if final_output is None:
                st.stop()

            # Extract values safely out of your standard State Dict schema
            final_answer = final_output.get("final_response", "No response generated.")
            retrieved_docs = final_output.get("retrieved_docs", [])
            validation_meta = final_output.get("validation_metadata", {})
            answer_status = final_output.get("answer_status")
            termination_reason = final_output.get("termination_reason")
            usage_line = format_usage(UsageLedger(**final_output.get("usage", {})), budget.limits)
        
            # Map diagnostic outputs into the Observability Sidebar
            st.session_state.system_logs.append(f"Answer status: {answer_status} | termination: {termination_reason or 'none'}")
            st.session_state.system_logs.append(f"Usage: {usage_line}")
            if validation_meta.get("reasoning"):
                st.session_state.system_logs.append(f"Evaluator notes: {validation_meta.get('reasoning')}")

            # Render the answer with its honest status label
            with response_placeholder.container():
                render_answer_status(answer_status, termination_reason)
                st.markdown(final_answer)
                st.caption(usage_line)
                for notice in dict.fromkeys(final_output.get("notices", [])):  # unique, in order
                    st.caption(f"⚠️ {notice}")
        
            # --- PROCESS LIVE CITATIONS (NAME ONLY) ---
            citations_to_store = []
            if retrieved_docs:
                with st.expander("🔗 Sources consulted"):
                    seen_sources_live = set()
                    display_idx = 1
                
                    for doc in retrieved_docs:
                        source_name = doc.get('source', '').strip()
                        if not source_name:
                            continue
                        
                        if source_name not in seen_sources_live:
                            seen_sources_live.add(source_name)
                            st.markdown(f"**[{display_idx}] Source Document:** `{source_name}`")
                            citations_to_store.append({"source": source_name})
                            display_idx += 1
                        else:
                            citations_to_store.append({"source": source_name})

            # Save Assistant turn to Chat History (including its citation metadata)
            ai_message_state = AIMessage(content=final_answer)
            ai_message_state.metadata = {
                "citations": citations_to_store,
                "answer_status": answer_status,
                "termination_reason": termination_reason,
                "usage_line": usage_line,
            }
            st.session_state.chat_history.append(ai_message_state)
        
            # Force refresh to update the sidebar telemetry layout smoothly
            st.rerun()