# Debug for Streamlit app to run
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain import hub
from langgraph.graph import START, StateGraph

# Set page config
st.set_page_config(
    page_title="Aviation Incident Q&A",
    page_icon="✈️",
    layout="wide"
)

# Define State class (same as in notebook)
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

@st.cache_resource
def load_vector_store():
    """Load the existing vector store (same as notebook)"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory="./data/vector_db",  # Where to save data locally, remove if not necessary
        )
        return vector_store
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

@st.cache_resource
def create_rag_graph():
    """Create the RAG graph using LangChain and LangGraph (same as notebook)"""
    try:
        # Load vector store
        vector_store = load_vector_store()
        if not vector_store:
            return None
        
        # Initialize LLM (same model as notebook)
        llm = GoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20")
        
        # Load prompt (same as notebook)
        prompt = hub.pull("rlm/rag-prompt")
        
        def retrieve(state: State):
            retrieved_docs = vector_store.similarity_search(state["question"])
            return {"context": retrieved_docs}

        def generate(state: State):
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
            messages = prompt.invoke({"question": state["question"], "context": docs_content})
            response = llm.invoke(messages)
            return {"answer": response}
        
        # Build the graph (same as notebook)
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()
        
        return graph
    except Exception as e:
        st.error(f"Error creating RAG graph: {str(e)}")
        return None

# Main app
def main():
    st.title("✈️ Aviation Incident Q&A System")
    st.markdown("Ask questions about aviation incidents and accidents from the NTSB database using RAG (Retrieval-Augmented Generation).")
    
    # Initialize the RAG system
    with st.spinner("Loading RAG system..."):
        graph = create_rag_graph()
    
    if not graph:
        st.error("❌ Failed to load the RAG system. Please check your setup.")
        st.stop()
    
    st.success("✅ RAG system loaded successfully!")
    
    # Create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Ask a Question")
        
        # Question input
        question = st.text_area(
            "Enter your question about aviation incidents:",
            placeholder="e.g., Were there any incidents involving the CESSNA 172?",
            height=100,
            key="question_input"
        )
        
        # Sample questions
        st.markdown("**Sample questions:**")
        sample_questions = [
            "Were there any incidents involving the CESSNA 172?",
            "What are the most common causes of aviation accidents?",
            "Tell me about incidents in California",
            "What types of aircraft have the most incidents?",
            "Show me accidents involving engine failure"
        ]
        
        selected_question = st.selectbox(
            "Or choose a sample question:",
            [""] + sample_questions,
            key="sample_selector"
        )
        
        if selected_question:
            question = selected_question
        
        # Submit button
        if st.button("🔍 Get Answer", type="primary") and question:
            with st.spinner("Searching documents and generating answer..."):
                try:
                    # Use the RAG graph to get the answer (same as notebook)
                    result = graph.invoke({"question": question})
                    
                    # Display the answer
                    st.subheader("Answer")
                    st.write(result["answer"])
                    
                    # Store results in session state
                    st.session_state.last_result = result
                    
                except Exception as e:
                    st.error(f"Error processing question: {str(e)}")
    
    with col2:
        st.subheader("Retrieved Context")
        
        if hasattr(st.session_state, 'last_result') and st.session_state.last_result:
            context_docs = st.session_state.last_result.get("context", [])
            
            if context_docs:
                st.markdown(f"**Found {len(context_docs)} relevant documents:**")
                
                for i, doc in enumerate(context_docs):
                    with st.expander(f"Document {i+1}", expanded=(i == 0)):
                        # Show document content
                        content = doc.page_content
                        if len(content) > 500:
                            content = content[:500] + "..."
                        st.text_area(
                            f"Content:",
                            content,
                            height=150,
                            key=f"doc_{i}",
                            disabled=True
                        )
                        
                        # Show metadata if available
                        if hasattr(doc, 'metadata') and doc.metadata:
                            st.json(doc.metadata)
            else:
                st.info("No context documents found.")
        else:
            st.info("Submit a question to see relevant context documents here.")
    
    # Additional features
    st.markdown("---")
    
    with st.expander("ℹ️ About this System"):
        st.markdown("""
        This Q&A system uses:
        - **RAG (Retrieval-Augmented Generation)** to find relevant documents
        - **Chroma Vector Database** for efficient similarity search
        - **Google Gemini Embeddings** for document vectorization
        - **Google Gemini AI** for answer generation
        - **LangChain & LangGraph** for orchestration
        
        The system searches through NTSB aviation incident reports to provide 
        accurate, context-aware answers to your questions.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("**Data Source:** NTSB Aviation Incident & Accident Reports")
    st.markdown("**Powered by:** Google Gemini AI, LangChain, and LangGraph")

if __name__ == "__main__":
    main()
