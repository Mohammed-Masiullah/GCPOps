# ============================================================================
# COMPLETE STREAMLIT RAG CHATBOT APP
# Ready to deploy - just run: streamlit run streamlit_rag_app.py
# ============================================================================

import streamlit as st
from rag_pipeline import rag_pipeline, print_detailed_results
import time
from datetime import datetime

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🎬 Movie RAG Chat",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .title-container {
        text-align: center;
        padding: 2rem 0;
        border-bottom: 2px solid #1f77b4;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZATION
# ============================================================================

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "settings" not in st.session_state:
    st.session_state.settings = {
        "reranker_type": "crossencoder",
        "initial_top_k": 50,
        "rerank_top_k": 8,
        "max_context_tokens": 30000
    }

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

with st.sidebar:
    st.markdown("# ⚙️ Configuration")
    
    # Mode selection
    mode = st.radio(
        "📍 Select Mode:",
        ["💬 Query Mode", "🔧 Infra Mode (Coming Soon)"],
        help="Choose between chatting with movies or infrastructure settings"
    )
    
    if "💬" in mode:
        st.markdown("### 🎯 Query Settings")
        
        # Reranker selection
        reranker_type = st.selectbox(
            "🤖 Reranker Type:",
            ["crossencoder", "vertex_ai"],
            format_func=lambda x: "CrossEncoder (Local, Data-Private)" if x == "crossencoder" else "Vertex AI (Enterprise Support)",
            help="CrossEncoder: Local execution, data stays private\nVertex AI: Cloud-based, enterprise support"
        )
        st.session_state.settings["reranker_type"] = reranker_type
        
        if reranker_type == "crossencoder":
            st.info("✅ Data-Private Mode: No data leaves your environment")
        else:
            st.warning("⚠️ Cloud Mode: Data sent to Google Cloud")
        
        # Initial top_k
        initial_top_k = st.slider(
            "🔍 Initial Candidates:",
            min_value=10,
            max_value=150,
            value=50,
            step=10,
            help="Number of chunks to retrieve from BigQuery (higher = more thorough search)"
        )
        st.session_state.settings["initial_top_k"] = initial_top_k
        
        # Rerank top_k
        rerank_top_k = st.slider(
            "📊 Final Chunks After Reranking:",
            min_value=3,
            max_value=20,
            value=8,
            step=1,
            help="Number of chunks to use for answer generation (higher = longer context)"
        )
        st.session_state.settings["rerank_top_k"] = rerank_top_k
        
        # Token budget
        max_context_tokens = st.slider(
            "🪙 Max Context Tokens:",
            min_value=5000,
            max_value=100000,
            value=30000,
            step=5000,
            help="Maximum tokens for context (higher = more complete answers but slower)"
        )
        st.session_state.settings["max_context_tokens"] = max_context_tokens
        
        st.markdown("---")
        
        # Quick presets
        st.markdown("### ⚡ Quick Presets")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Speed Mode"):
                st.session_state.settings.update({
                    "initial_top_k": 20,
                    "rerank_top_k": 3,
                    "max_context_tokens": 10000
                })
                st.rerun()
        
        with col2:
            if st.button("🎯 Quality Mode"):
                st.session_state.settings.update({
                    "initial_top_k": 100,
                    "rerank_top_k": 15,
                    "max_context_tokens": 50000
                })
                st.rerun()
    
    st.markdown("---")
    
    # Info section
    with st.expander("ℹ️ About This App"):
        st.markdown("""
        ### 🎬 Movie RAG Chatbot
        
        **What is RAG?**
        - Retrieval-Augmented Generation combines search + AI
        - Retrieves relevant context before generating answers
        
        **Pipeline:**
        1. Convert your query to embeddings
        2. Search BigQuery for similar movies
        3. Rerank using semantic similarity
        4. Generate answer with Gemini 2.5 Flash
        
        **Tech Stack:**
        - **Embedding:** Gemini Embedding 001
        - **Reranking:** CrossEncoder (local) or Vertex AI
        - **LLM:** Gemini 2.5 Flash
        - **Vector DB:** BigQuery
        - **UI:** Streamlit
        """)
    
    # Statistics
    with st.expander("📈 Session Stats"):
        if st.session_state.messages:
            st.metric("Queries", len([m for m in st.session_state.messages if m["role"] == "user"]))
            st.metric("Responses", len([m for m in st.session_state.messages if m["role"] == "assistant"]))
        else:
            st.info("No queries yet. Start chatting! 👆")
    
    # Clear history button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# ============================================================================
# MAIN CONTENT
# ============================================================================

# Title
st.markdown("""
<div class="title-container">
    <h1>🎬 Movie RAG Chatbot</h1>
    <p>Ask questions about movies powered by RAG + Gemini</p>
</div>
""", unsafe_allow_html=True)

# Mode-specific content
if "💬" in mode:
    # Instructions
    st.markdown("""
    ### How to use:
    1. **Enter your question** in the chat box below
    2. **System retrieves** relevant movie information
    3. **Reranks** results for accuracy
    4. **Generates answer** with Gemini 2.5 Flash
    5. **View metrics** to see pipeline performance
    
    **Example queries:**
    - "What are highly rated comedy movies?"
    - "Tell me about movies released in 2020"
    - "Best action movies with IMDb rating > 8"
    """)
    
    # Chat interface
    chat_container = st.container(border=True)
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🤖"):
                st.markdown(message["content"])
                
                # Display metrics for assistant messages
                if message["role"] == "assistant" and "metrics" in message:
                    with st.expander("📊 View Pipeline Metrics"):
                        metrics = message["metrics"]
                        
                        # Performance metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Time (s)", f"{metrics.get('pipeline_time_seconds', 0):.2f}")
                        with col2:
                            st.metric("Candidates", metrics.get('candidates_retrieved', 0))
                        with col3:
                            st.metric("Sources Used", metrics.get('sources_used', 0))
                        with col4:
                            st.metric("Reranker", metrics.get('reranker_used', 'N/A'))
                        
                        # Token metrics
                        col5, col6, col7 = st.columns(3)
                        with col5:
                            st.metric("Prompt Tokens", metrics.get('prompt_tokens', 0))
                        with col6:
                            st.metric("Output Tokens", metrics.get('output_tokens', 0))
                        with col7:
                            st.metric("Total Tokens", metrics.get('total_tokens', 0))
    
    # Input area
    user_input = st.chat_input(
        "Ask about movies...",
        key="chat_input"
    )
    
    if user_input:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Display user message
        with chat_container:
            with st.chat_message("user", avatar="🧑"):
                st.markdown(user_input)
        
        # Generate response
        with st.spinner("🔍 Searching and generating answer..."):
            try:
                # Record start time
                start_time = time.time()
                
                # Call RAG pipeline
                result = rag_pipeline(
                    user_query=user_input,
                    reranker_type=st.session_state.settings["reranker_type"],
                    initial_top_k=st.session_state.settings["initial_top_k"],
                    rerank_top_k=st.session_state.settings["rerank_top_k"],
                    max_context_tokens=st.session_state.settings["max_context_tokens"]
                )
                
                answer = result.get('answer', 'Error generating answer')
                
                # Add assistant message to chat
                assistant_message = {
                    "role": "assistant",
                    "content": answer,
                    "metrics": {
                        'pipeline_time_seconds': result.get('pipeline_time_seconds', 0),
                        'candidates_retrieved': result.get('candidates_retrieved', 0),
                        'sources_used': result.get('sources_used', 0),
                        'prompt_tokens': result.get('prompt_tokens', 0),
                        'output_tokens': result.get('output_tokens', 0),
                        'total_tokens': result.get('total_tokens', 0),
                        'reranker_used': result.get('reranker_used', 'N/A')
                    }
                }
                
                st.session_state.messages.append(assistant_message)
                
                # Display assistant message and metrics
                with chat_container:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(answer)
                        
                        # Show metrics
                        with st.expander("📊 View Pipeline Metrics"):
                            metrics = assistant_message["metrics"]
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Time (s)", f"{metrics['pipeline_time_seconds']:.2f}")
                            with col2:
                                st.metric("Candidates", metrics['candidates_retrieved'])
                            with col3:
                                st.metric("Sources", metrics['sources_used'])
                            with col4:
                                st.metric("Reranker", metrics['reranker_used'])
                            
                            col5, col6, col7 = st.columns(3)
                            with col5:
                                st.metric("Prompt Tokens", metrics['prompt_tokens'])
                            with col6:
                                st.metric("Output Tokens", metrics['output_tokens'])
                            with col7:
                                st.metric("Total Tokens", metrics['total_tokens'])
            
            except Exception as e:
                error_message = f"❌ Error: {str(e)}"
                
                # Add error message to chat
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
                
                # Display error
                with chat_container:
                    with st.chat_message("assistant", avatar="⚠️"):
                        st.error(error_message)

else:  # Infra Mode
    st.markdown("""
    ### 🔧 Infrastructure Mode (Coming Soon)
    
    #### Planned Features:
    - 📊 Vector Index Management
    - 📈 Pipeline Performance Monitoring
    - 🔍 System Health Dashboard
    - ⚙️ Configuration Management
    - 📝 Query History & Analytics
    - 🐛 Debugging & Logs
    
    #### Stay Tuned!
    This mode will be implemented in the next phase.
    """)
    
    st.info("👷 Under development. Check back soon!")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    🎬 Movie RAG Chatbot | Built with Streamlit, Gemini & CrossEncoder
    <br>
    Questions? Check the sidebar for more information.
</div>
""", unsafe_allow_html=True)
