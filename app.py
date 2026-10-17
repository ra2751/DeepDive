import streamlit as st
import time
import os
from pathlib import Path
from config.settings import ConfigError
from domains.registry import load_domains
from ui.legacy_view import render_legacy
from ui.research_view import SOURCE_OPTIONS, render_research

WORKFLOW_VERIFIED = "Verified research (default)"
WORKFLOW_LEGACY = "Legacy answer-level (fallback)"

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
    page_title="DeepDive AI",
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
if "system_logs" not in st.session_state:
    st.session_state.system_logs = ["System initialized. Ready for user query."]

# ==========================================
# 4. SIDEBAR: UPLOADER & OBSERVABILITY LOGS
# ==========================================
with st.sidebar:
    st.title("🛡️ Agent Observability")
    st.subheader("Multi-Agent Control Panel")
    st.markdown("---")

    # --- RESEARCH WORKFLOW CONTROLS ---
    workflow = st.radio("Workflow", [WORKFLOW_VERIFIED, WORKFLOW_LEGACY], index=0,
                        help="The legacy workflow is kept temporarily for testing and rollback.")
    domains = load_domains()
    domain_id, scope, claims_to_check = "general", SOURCE_OPTIONS["Web only"], []
    if workflow == WORKFLOW_VERIFIED:
        domain_id = st.selectbox("Research domain", list(domains), index=list(domains).index("general"),
                                 format_func=lambda d: domains[d].name)
        source_choice = st.radio("Sources", list(SOURCE_OPTIONS), index=list(SOURCE_OPTIONS).index("Web only"),
                                 help="Documents = the indexed Milvus collection; Web = Tavily search. "
                                      "Web is used only when you select it.")
        scope = SOURCE_OPTIONS[source_choice]
        claims_text = st.text_area("Optional: claims to fact-check (one per line)", value="",
                                   help="If provided, these claims are verified instead of extracting new ones.")
        claims_to_check = [c.strip() for c in claims_text.splitlines() if c.strip()]
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
                            
                    except ConfigError as config_error:
                        st.error(f"❌ Configuration error: {config_error}")
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
st.title("🤖 DeepDive AI — Self-Verifying Research Analyst")
if workflow == WORKFLOW_VERIFIED:
    st.subheader("Self-verifying research: every finding is checked claim by claim against cited evidence")
    st.markdown("---")
    render_research(domain_id, scope, claims_to_check)
else:
    st.subheader("Legacy answer-level workflow")
    st.info("Fallback mode for testing and rollback: answers here are checked at answer level only, "
            "not claim by claim, and are never labelled verified.")
    st.markdown("---")
    render_legacy()
