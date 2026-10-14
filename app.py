import streamlit as st
import time
import os
from pathlib import Path
from langchain_core.messages import AIMessage, HumanMessage
# Import your compiled LangGraph application
from graph_builder.graph_builder import app as agent_graph

# --- DIRECT IMPORT OF YOUR NATIVE PIPELINE ENTITIES ---
try:
    from main_pipeline import MultiPDFPipeline
except ImportError:
    st.error("Engine Error: 'main_pipeline.py' not found in the application path root layout.")

# Establish directory structures relative to root
RAW_PDF_DIR = Path("data/raw_pdfs")
RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 1. PAGE CONFIGURATION & THEME
# ==========================================
st.set_page_config(
    page_title="Enterprise AI Research Assistant",
    page_icon="🔬",  
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injection for a professional, polished aesthetic
st.markdown("""
<style>
    .reportview-container { background: #f5f7f9; }
    .main .block-container { padding-top: 2rem; }
    div.stButton > button:first-child {
        background-color: #0066cc; color: white; border-radius: 6px;
    }
    .citation-header { font-size: 14px; font-weight: 600; color: #4A5568; margin-top: 10px; }
    .source-tag { 
        background-color: #EDF2F7; padding: 4px 8px; border-radius: 4px; 
        font-size: 12px; font-family: monospace; color: #2D3748; display: inline-block; margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. SESSION STATE INITIALIZATION
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "system_logs" not in st.session_state:
    st.session_state.system_logs = ["System initialized. Ready for user query."]

# ==========================================
# 4. SIDEBAR: UPLOADER & OBSERVABILITY LOGS
# ==========================================
with st.sidebar:
    st.title("🛡️ Agent Observability")
    st.subheader("Multi-Agent Control Panel")
    st.markdown("---")
    
    # --- AUTOMATED RAW_PDF INCREMENTAL INGESTION LAYER ---
    st.subheader("📁 Knowledge Base Ingestion")
    uploaded_file = st.file_uploader(
        "Upload reference PDFs (Max 10MB)",
        type=["pdf"],
        help="Documents uploaded here are instantly analyzed and securely integrated into your corporate knowledge base for real-time research and insights."
    )
    
    if uploaded_file is not None:
        # Enforce file boundary rules (10MB ceiling)
        MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
        if uploaded_file.size > MAX_FILE_SIZE_BYTES:
            st.error("❌ File rejection notice: The uploaded document exceeds the maximum 10MB size limit.")
        else:
            # Prevent processing loops if the file is already processed in this session instance
            if f"processed_{uploaded_file.name}" not in st.session_state:
                
                # Setup target output tracking paths
                saved_pdf_path = RAW_PDF_DIR / uploaded_file.name
                
                with st.spinner(f"Saving and indexing {uploaded_file.name} to vector store..."):
                    try:
                        # Write stream buffer contents out to disk under data/raw_pdfs
                        with open(saved_pdf_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        
                        st.session_state.system_logs.append(f"Saved file '{uploaded_file.name}' to root location data/raw_pdfs/")
                        
                        # Instantiate your custom MultiPDFPipeline to handle incremental processing
                        pipeline_engine = MultiPDFPipeline()
                        st.session_state.system_logs.append("Triggering incremental MultiPDFPipeline extraction workflow...")
                        
                        # Force processing loop strictly against the single target path file
                        pipeline_status = pipeline_engine.run(single_pdf_path=saved_pdf_path)
                        
                        if pipeline_status:
                            st.session_state[f"processed_{uploaded_file.name}"] = True
                            st.session_state.system_logs.append(f"Successfully processed and index-loaded '{uploaded_file.name}' into Milvus.")
                            st.success(f"✅ Ingested '{uploaded_file.name}' successfully!")
                            st.balloons()
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Vector store pipeline failed to yield usable text segments.")
                            
                    except Exception as pipeline_error:
                        st.error(f"❌ Internal Processing Loop Failure: {str(pipeline_error)}")
                        st.session_state.system_logs.append(f"Pipeline error: {str(pipeline_error)}")
            else:
                st.info(f"ℹ️ '{uploaded_file.name}' is already loaded and available for query.")

    st.markdown("---")
    st.markdown("**Active Agents Monitor:**")
    
    # Render active logs dynamically as they happen
    log_container = st.empty()
    with log_container.container():
        for log in st.session_state.system_logs[::-1]: # Show freshest logs on top
            st.caption(f"⏱️ {log}")
            
    if st.button("Clear Systems Diagnostics"):
        st.session_state.system_logs = ["System telemetry cleared."]
        st.rerun()

# ==========================================
# 5. MAIN INTERFACE LAYOUT
# ==========================================
st.title("🤖 AI Research Assistant")
st.subheader("Your Intelligent Agent for Deep Research and Fact-Checked Insights")
st.markdown("---")

# Display historical chat conversations
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("User"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("Assistant"):
            st.markdown(message.content)
            
            # --- RENDER HISTORICAL SOURCES (NAME ONLY) ---
            if hasattr(message, 'metadata') and message.metadata.get('citations'):
                with st.expander("🔗 Sources"):
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
        
        # Invoke your compiled LangGraph pipeline
        with st.spinner("Agents coordinating strategy..."):
            initial_state = {
                "query": user_query, 
                "messages": st.session_state.chat_history, 
                "loop_count": 0,
                "validation_metadata": {"is_valid": "yes", "reasoning": "turn 1 reset verification pass"}
            }
            final_output = agent_graph.invoke(initial_state)

        # Extract values safely out of your standard State Dict schema
        final_answer = final_output.get("final_response", "No response generated.")
        retrieved_docs = final_output.get("retrieved_docs", [])
        validation_meta = final_output.get("validation_metadata", {})
        
        # Map diagnostic outputs into the Observability Sidebar
        st.session_state.system_logs.append(f"Validation verification verdict: {validation_meta.get('is_valid', 'N/A')}")
        if validation_meta.get("reasoning"):
            st.session_state.system_logs.append(f"Evaluator notes: {validation_meta.get('reasoning')}")

        # Render final verified text answer to the chat interface
        response_placeholder.markdown(final_answer)
        
        # --- PROCESS LIVE CITATIONS (NAME ONLY) ---
        citations_to_store = []
        if retrieved_docs:
            with st.expander("🔗 Verified Grounding Sources"):
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
        ai_message_state.metadata = {"citations": citations_to_store}
        st.session_state.chat_history.append(ai_message_state)
        
        # Force refresh to update the sidebar telemetry layout smoothly
        st.rerun()