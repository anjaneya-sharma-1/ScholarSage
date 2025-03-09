import streamlit as st
import os
import tempfile
import config
from utils import fetch_data, faiss_helper, embeddings
import logging


st.set_page_config(
    page_title="ScholarSage", 
    page_icon="📚",
    layout="wide"
)


config.ensure_directories()


logging.basicConfig(
    filename=config.LOG_PATH,
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger('app')


if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = faiss_helper.FAISSIndex()
 
    if not st.session_state.faiss_index.load():
        st.session_state.faiss_index.create_new_index()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def local_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #4169e1;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-container {
        height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .input-container {
        position: sticky;
        bottom: 0;
        padding: 1rem;
        background-color: white;
        border-top: 1px solid #ddd;
    }
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #4169e1;
    }
    </style>
    """, unsafe_allow_html=True)

def process_research_topic(topic):
    """Process a research topic by fetching data and updating the FAISS index"""
    st.info(f"Researching about: {topic}... This may take a moment.")
    
   
    wiki_data = fetch_data.fetch_wikipedia(topic)
    arxiv_papers = fetch_data.fetch_arxiv(topic, max_results=10)  # Increased from 5 to 10
    

    chunks = []
    metadata_list = []
    
  
    if wiki_data:
        wiki_chunks = embeddings.chunk_text(wiki_data, chunk_size=1000)
        for i, chunk in enumerate(wiki_chunks):
            chunks.append(chunk)
            metadata_list.append({
                'source': 'wikipedia',
                'topic': topic,
                'chunk_id': i
            })
    
    # Process ArXiv data
    for i, paper in enumerate(arxiv_papers):
        paper_text = f"Title: {paper['title']}\nSummary: {paper['summary']}"
        paper_chunks = embeddings.chunk_text(paper_text, chunk_size=1000)
        for j, chunk in enumerate(paper_chunks):
            chunks.append(chunk)
            metadata_list.append({
                'source': 'arxiv',
                'paper_id': i,
                'paper_title': paper['title'],
                'chunk_id': j,
                'url': paper['url']
            })
    
    # Generate embeddings and update index
    if chunks:
        batch_embeddings = embeddings.batch_embed(chunks)
        st.session_state.faiss_index.add_to_index(batch_embeddings, metadata_list, chunks)
        st.session_state.faiss_index.save()
        return True
    return False

def process_query(query):
    """Process a user query using the FAISS index"""
    # Generate query embedding - use the new search_query_embedding function
    if config.ENABLE_QUERY_EXPANSION:
        # Generate embedding with query expansion
        query_embedding = embeddings.search_query_embedding(query)
        
        # You might want to add this for user feedback when the expanded query is different
        expanded_query = embeddings.expand_query(query)
        if expanded_query != query:
            st.info(f"Optimized search query: '{expanded_query}'")
    else:
        # Use regular embedding without expansion
        query_embedding = embeddings.get_embedding(query)
    
    # Search for relevant information
    if st.session_state.faiss_index.index is None or st.session_state.faiss_index.index.ntotal == 0:
        # No index exists yet - perform direct research
        st.info(f"No existing research found. Searching for information about: '{query}'...")
        success = process_research_topic(query)
        if success:
            st.success(f"Found information about '{query}'!")
            # Now try the search again with the new information
            return process_query(query)
        else:
            return f"I couldn't find specific information about '{query}'. Please try a different query or use the research tool to add more information on this topic."
    
    # Convert to numpy array and reshape
    import numpy as np
    query_embedding_np = np.array([query_embedding])
    
   
    k = 10 
    D, I = st.session_state.faiss_index.search(query_embedding_np, k=k)
    

    max_distance_threshold = 1.2  # Adjust as needed - lower is more strict
    
    # Retrieve relevant context
    context = []
    for i, idx in enumerate(I[0]):
        if idx < 0 or idx >= len(st.session_state.faiss_index.metadata) or D[0][i] > max_distance_threshold:
            continue
        metadata = st.session_state.faiss_index.metadata[idx]
        chunk_id = idx
        context.append((chunk_id, metadata, D[0][i]))  # Store distance for debugging
    
    if not context:
        # No relevant context found or results weren't relevant enough
        st.info(f"No relevant information found. Searching the web for: '{query}'...")
        
        # First try to research the specific topic
        success = process_research_topic(query)
        
        if success:
            st.success(f"Found new information about '{query}'!")
            # Try searching again with the new data
            return process_query(query)
        else:
            # If specific topic research failed, try a more general approach
            # Extract key concepts from the query for broader research
            import re
            key_terms = re.findall(r'\b[A-Za-z]{3,}\b', query)
            research_success = False
            
            # Try to research each key term
            for term in key_terms[:2]:  # Limit to first 2 terms to avoid too many searches
                if term.lower() not in ['what', 'when', 'where', 'why', 'who', 'how', 'the', 'and', 'that']:
                    st.info(f"Researching related topic: '{term}'...")
                    if process_research_topic(term):
                        research_success = True
            
            if research_success:
                st.success("Found some related information!")
                # Try searching one more time with the new data
                return process_query(query)
            else:
                return f"I don't have enough information about '{query}' in my knowledge base. Please use the research tool in the sidebar to add information on this topic first."
    
    # Sort context by relevance (lowest distance first)
    context.sort(key=lambda x: x[2])
    
    # Build context from top results
    context_text = ""
    sources = []
    
    for chunk_id, metadata, _ in context:
        chunk_index = st.session_state.faiss_index.chunk_indices[chunk_id]
        context_text += st.session_state.faiss_index.chunks[chunk_index] + "\n\n"
        
        if metadata['source'] == 'wikipedia':
            sources.append(f"Wikipedia article on {metadata['topic']}")
        elif metadata['source'] == 'arxiv':
            sources.append(f"ArXiv paper: {metadata['paper_title']}")
        elif metadata['source'] == 'pdf':
            sources.append(f"PDF: {metadata['filename']}")
    
    # Generate response
    answer = embeddings.generate_answer(
        query, 
        context_text,
    )
    
    return answer

def main():
    local_css()
    
    # App header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">📚 ScholarSage</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Your AI-powered research assistant</p>', unsafe_allow_html=True)
    
    # Sidebar for research topic
    with st.sidebar:
        st.markdown('<p class="sidebar-header">Research Setup</p>', unsafe_allow_html=True)
        research_topic = st.text_input("What topic would you like to research?", key="research_topic_input")
        col1, col2 = st.columns(2)
        with col1:
            research_btn = st.button("Start Research", type="primary", use_container_width=True)
        with col2:
            clear_btn = st.button("Clear Chat", type="secondary", use_container_width=True)
        
        if research_btn and research_topic:
            with st.spinner(f"Researching about {research_topic}..."):
                success = process_research_topic(research_topic)
                if success:
                    st.success(f"Research on '{research_topic}' completed!")
                else:
                    st.error("Failed to find relevant information.")
        
        if clear_btn:
            st.session_state.chat_history = []
            st.rerun()
        
        # PDF upload with improved UI
        st.markdown('<p class="sidebar-header">Upload Document</p>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
        if uploaded_file:
            if st.button("Process PDF", type="primary", use_container_width=True):
                with st.spinner("Processing PDF..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    pdf_text = fetch_data.read_pdf(tmp_path)
                    if pdf_text:
                        # Process PDF content
                        chunks = embeddings.chunk_text(pdf_text, chunk_size=1000)
                        metadata_list = []
                        
                        for i, chunk in enumerate(chunks):
                            metadata_list.append({
                                'source': 'pdf',
                                'filename': uploaded_file.name,
                                'chunk_id': i
                            })
                        
                        # Generate embeddings and update index
                        batch_embeddings = embeddings.batch_embed(chunks)
                        st.session_state.faiss_index.add_to_index(batch_embeddings, metadata_list, chunks)
                        st.session_state.faiss_index.save()
                        st.success(f"PDF '{uploaded_file.name}' processed successfully!")
                    
                    # Clean up
                    os.unlink(tmp_path)
        
        # Advanced settings with toggles
        st.markdown('<p class="sidebar-header">Advanced Settings</p>', unsafe_allow_html=True)
        enable_expansion = st.toggle("Enable Query Expansion", value=config.ENABLE_QUERY_EXPANSION)
        if enable_expansion != config.ENABLE_QUERY_EXPANSION:
            config.ENABLE_QUERY_EXPANSION = enable_expansion
            st.success(f"Query expansion {'enabled' if enable_expansion else 'disabled'}.")
        
        # Model selection with accessibility notes
        model_options = {
            "mistralai/Mistral-7B-Instruct-v0.2": "Mistral 7B Instruct (Recommended for API)",
            "google/gemma-7b-it": "Gemma 7B-IT (Requires special permissions)",
            "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral 8x7B (May have quotas)",
            "meta-llama/Llama-2-13b-chat-hf": "Llama 2 13B (Requires special permissions)"
        }
        
        selected_model = st.selectbox(
            "Select AI Model", 
            options=list(model_options.keys()),
            format_func=lambda x: model_options.get(x),
            index=list(model_options.keys()).index(config.GENERATION_MODEL) if config.GENERATION_MODEL in model_options else 0
        )
        
        if selected_model != config.GENERATION_MODEL:
            config.GENERATION_MODEL = selected_model
            st.success(f"Model changed to {model_options[selected_model].split(' (')[0]}")
            
            # Show warning for models with access restrictions
            if "special permission" in model_options[selected_model].lower() or "quota" in model_options[selected_model].lower():
                st.warning("⚠️ This model may have API access restrictions. If you get a 403 error, try using the local model option or switch to Mistral 7B.")
        
        # Index stats
        st.markdown('<p class="sidebar-header">Index Statistics</p>', unsafe_allow_html=True)
        if st.session_state.faiss_index.index:
            st.metric("Documents in Index", st.session_state.faiss_index.index.ntotal)
        else:
            st.info("No index available yet.")
    
    # Main chat container
    container = st.container()
    
    # Chat history display - show messages in proper chat bubbles
    with container:
        # Display initial prompt if no chat history
        if not st.session_state.chat_history:
            st.info("👋 Hello! I'm ScholarSage. Ask me anything about your research topic or upload a PDF to get started.")
        
        # Display chat history with proper chat bubbles
        for role, message in st.session_state.chat_history:
            if role == "You":
                with st.chat_message("user", avatar="👤"):
                    st.write(message)
            else:
                with st.chat_message("assistant", avatar="📚"):
                    st.write(message)
    
    # Input area at the bottom
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    query = st.chat_input("Ask a question about your research", key="query_input")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if query:
        # Add user message to chat
        with container:
            with st.chat_message("user", avatar="👤"):
                st.write(query)
        
        # Process query and get response
        with st.spinner("ScholarSage is thinking..."):
            response = process_query(query)
        
        # Add assistant response to chat
        with container:
            with st.chat_message("assistant", avatar="📚"):
                st.write(response)
        
        # Update chat history
        st.session_state.chat_history.append(("You", query))
        st.session_state.chat_history.append(("ScholarSage", response))

if __name__ == "__main__":
    main()
