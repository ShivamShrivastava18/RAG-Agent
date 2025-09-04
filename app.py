"""
Streamlit AI Agent Chatbot with Multimodal Capabilities
Uses Google Gemini API for LLM with image processing support
"""

import streamlit as st
import os
import json
import sqlite3
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import asyncio
from pathlib import Path
import base64
from io import BytesIO
import tempfile
import sys

# Core dependencies
import google.generativeai as genai
import chromadb
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize
import PyPDF2
import docx
from bs4 import BeautifulSoup
from PIL import Image
import requests

# Streamlit page config
st.set_page_config(
    page_title="AI Agent Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download required NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

download_nltk_data()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentResponse:
    """Structured response from the agent"""
    content: str
    sources: List[str]
    confidence_score: float
    reasoning_steps: List[str]
    tool_calls: List[str]
    image_analysis: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class DocumentProcessor:
    """Handles document loading and processing including images"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt', '.html', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']
    
    def load_document(self, uploaded_file) -> Dict[str, Any]:
        """Load and extract content from various document formats including images"""
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        metadata = {
            "filename": uploaded_file.name,
            "file_type": file_extension,
            "size": uploaded_file.size,
            "created": datetime.now()
        }
        
        content = ""
        image_data = None
        
        try:
            if file_extension == '.pdf':
                content = self._extract_pdf(uploaded_file)
            elif file_extension == '.docx':
                content = self._extract_docx(uploaded_file)
            elif file_extension == '.txt':
                content = self._extract_txt(uploaded_file)
            elif file_extension == '.html':
                content = self._extract_html(uploaded_file)
            elif file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                content, image_data = self._process_image(uploaded_file)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            logger.error(f"Error processing {uploaded_file.name}: {str(e)}")
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            raise
        
        return {
            "content": content, 
            "metadata": metadata,
            "image_data": image_data
        }
    
    def _extract_pdf(self, uploaded_file) -> str:
        """Extract text from PDF"""
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def _extract_docx(self, uploaded_file) -> str:
        """Extract text from DOCX"""
        doc = docx.Document(uploaded_file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    
    def _extract_txt(self, uploaded_file) -> str:
        """Extract text from TXT"""
        return str(uploaded_file.read(), "utf-8")
    
    def _extract_html(self, uploaded_file) -> str:
        """Extract text from HTML"""
        soup = BeautifulSoup(uploaded_file.read(), 'html.parser')
        return soup.get_text()
    
    def _process_image(self, uploaded_file) -> Tuple[str, bytes]:
        """Process image file and return description + image data"""
        # Read image data
        image_bytes = uploaded_file.read()
        
        # Reset file pointer for PIL
        uploaded_file.seek(0)
        
        # Open image with PIL for metadata
        try:
            img = Image.open(uploaded_file)
            content = f"Image file: {uploaded_file.name}, Format: {img.format}, Size: {img.size}, Mode: {img.mode}"
        except Exception as e:
            content = f"Image file: {uploaded_file.name} (could not read metadata: {str(e)})"
        
        return content, image_bytes

class StreamlitMemoryManager:
    """Manages conversation memory in Streamlit session state"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.max_memory_items = 50
        
        # Initialize session state if not exists
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        if 'document_store' not in st.session_state:
            st.session_state.document_store = {}
        if 'facts_store' not in st.session_state:
            st.session_state.facts_store = []
    
    def add_conversation(self, user_input: str, response: AgentResponse):
        """Store conversation in session state"""
        conversation_item = {
            "user_input": user_input,
            "response": response,
            "timestamp": response.timestamp
        }
        
        st.session_state.conversation_history.append(conversation_item)
        
        # Keep only recent items
        if len(st.session_state.conversation_history) > self.max_memory_items:
            st.session_state.conversation_history = st.session_state.conversation_history[-self.max_memory_items:]
    
    def get_context(self, max_items: int = 5) -> str:
        """Get recent conversation context"""
        recent_items = st.session_state.conversation_history[-max_items:] if st.session_state.conversation_history else []
        context = []
        
        for item in recent_items:
            context.append(f"User: {item['user_input']}")
            context.append(f"Assistant: {item['response'].content}")
        
        return "\n".join(context)
    
    def add_fact(self, fact: str, source: str, confidence: float):
        """Store extracted facts"""
        fact_item = {
            "fact": fact,
            "source": source,
            "confidence": confidence,
            "timestamp": datetime.now()
        }
        
        # Check if fact already exists
        existing_facts = [f["fact"] for f in st.session_state.facts_store]
        if fact not in existing_facts:
            st.session_state.facts_store.append(fact_item)
    
    def clear_memory(self):
        """Clear all memory"""
        st.session_state.conversation_history = []
        st.session_state.facts_store = []

@st.cache_resource
def initialize_vector_store():
    """Initialize ChromaDB vector store"""
    client = chromadb.PersistentClient(path="./streamlit_chroma_db")
    collection = client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )
    return client, collection

@st.cache_resource
def initialize_encoder():
    """Initialize sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

class VectorStore:
    """Manages document embeddings and similarity search"""
    
    def __init__(self):
        self.client, self.collection = initialize_vector_store()
        self.encoder = initialize_encoder()
    
    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any]):
        """Add document to vector store"""
        if not content.strip():
            return
            
        # Store in session state for easy access
        st.session_state.document_store[doc_id] = {
            "content": content,
            "metadata": metadata
        }
        
        # Chunk the document
        chunks = self._chunk_document(content)
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            embedding = self.encoder.encode([chunk])[0].tolist()
            
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": i,
                "chunk_size": len(chunk),
                "document_id": doc_id
            })
            
            try:
                # Remove existing chunk if it exists
                try:
                    self.collection.delete(ids=[chunk_id])
                except:
                    pass
                
                self.collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[chunk_metadata]
                )
            except Exception as e:
                logger.error(f"Error adding chunk {chunk_id}: {str(e)}")
        
        logger.info(f"Added {len(chunks)} chunks for document {doc_id}")
    
    def _chunk_document(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Smart chunking with context preservation"""
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap for context
                words = current_chunk.split()
                overlap_words = words[-overlap//10:] if len(words) > overlap//10 else words
                current_chunk = " ".join(overlap_words) + " " + sentence
            else:
                current_chunk += " " + sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text]
    
    def search(self, query: str, n_results: int = 5, filters: Dict = None) -> List[Dict]:
        """Perform similarity search"""
        try:
            query_embedding = self.encoder.encode([query])[0].tolist()
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filters
            )
            
            return [
                {
                    "content": doc,
                    "metadata": meta,
                    "distance": dist
                }
                for doc, meta, dist in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )
            ]
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")
            return []

class CustomTools:
    """Custom tools for the agent"""
    
    def __init__(self, vector_store: VectorStore, memory_manager: StreamlitMemoryManager):
        self.vector_store = vector_store
        self.memory_manager = memory_manager
    
    def search_documents(self, query: str, filters: Dict = None) -> List[Dict]:
        """Search through indexed documents"""
        return self.vector_store.search(query, n_results=5, filters=filters)
    
    def summarize_document(self, doc_content: str, max_length: int = 200) -> str:
        """Create a summary of document content"""
        sentences = sent_tokenize(doc_content)
        if len(sentences) <= 3:
            return doc_content
        
        # Simple extractive summarization
        summary_sentences = [
            sentences[0],
            sentences[len(sentences)//2] if len(sentences) > 2 else "",
            sentences[-1]
        ]
        
        summary = " ".join([s for s in summary_sentences if s])
        
        if len(summary) > max_length:
            summary = summary[:max_length] + "..."
        
        return summary
    
    def extract_facts(self, content: str) -> List[str]:
        """Extract key facts from content"""
        sentences = sent_tokenize(content)
        facts = []
        
        for sentence in sentences:
            if any(char.isdigit() for char in sentence) or \
               any(word in sentence.lower() for word in ['is', 'was', 'are', 'were', 'has', 'have']) and \
               len(sentence.split()) > 5:
                facts.append(sentence.strip())
        
        return facts[:5]

class GeminiAgent:
    """Main AI Agent using Google Gemini API"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        self.vector_store = VectorStore()
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.memory_manager = StreamlitMemoryManager(self.session_id)
        self.tools = CustomTools(self.vector_store, self.memory_manager)
        self.doc_processor = DocumentProcessor()
    
    def load_documents(self, uploaded_files):
        """Load multiple documents into the system"""
        success_count = 0
        
        # Only create progress bar if in Streamlit context
        progress_bar = None
        if 'streamlit' in sys.modules:
            progress_bar = st.progress(0)
        
        try:
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    doc_data = self.doc_processor.load_document(uploaded_file)
                    doc_id = Path(uploaded_file.name).stem
                    
                    # Convert datetime to string for metadata
                    if 'created' in doc_data.get('metadata', {}):
                        doc_data['metadata']['created'] = doc_data['metadata']['created'].isoformat()
                    
                    # For images, store the image data separately
                    if doc_data.get("image_data"):
                        st.session_state.document_store[f"{doc_id}_image"] = doc_data["image_data"]
                    
                    self.vector_store.add_document(
                        doc_id=doc_id,
                        content=doc_data["content"],
                        metadata=doc_data["metadata"]
                    )
                    
                    success_count += 1
                    logger.info(f"Successfully loaded document: {uploaded_file.name}")
                    
                except Exception as e:
                    logger.error(f"Failed to load document {uploaded_file.name}: {str(e)}")
                    if 'streamlit' in sys.modules:
                        st.error(f"Failed to load {uploaded_file.name}: {str(e)}")
                
                # Update progress if progress bar exists
                if progress_bar is not None:
                    progress = (i + 1) / len(uploaded_files)
                    progress_bar.progress(min(progress, 1.0))
        finally:
            # Ensure progress bar is cleared
            if progress_bar is not None and 'streamlit' in sys.modules:
                progress_bar.empty()
        
        return success_count
    
    def analyze_image_with_query(self, image_data: bytes, query: str) -> str:
        """Analyze image using Gemini's multimodal capabilities"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(BytesIO(image_data))
            
            prompt = f"""
            Analyze this image in the context of the following query: {query}
            
            Please provide:
            1. A detailed description of what you see
            2. How it relates to the query
            3. Any specific information that answers the query
            4. Any text visible in the image
            """
            
            response = self.model.generate_content([prompt, image])
            return response.text
            
        except Exception as e:
            logger.error(f"Error analyzing image: {str(e)}")
            return f"Error analyzing image: {str(e)}"
    
    def calculate_confidence(self, query: str, response: str, sources: List[str]) -> float:
        """Calculate confidence score for the response"""
        confidence = 0.5
        
        if sources:
            confidence += min(len(sources) * 0.1, 0.3)
        
        if any(word in response.lower() for word in ['according to', 'based on', 'the document states']):
            confidence += 0.1
        
        if any(phrase in response.lower() for phrase in ['might be', 'possibly', 'unclear', 'not sure']):
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def process_query(self, query: str, include_images: bool = True) -> AgentResponse:
        """Main query processing with multi-step reasoning"""
        reasoning_steps = []
        tool_calls = []
        all_sources = []
        image_analysis = None
        
        reasoning_steps.append("Starting query processing")
        
        # Step 1: Search for relevant documents
        search_results = self.tools.search_documents(query)
        tool_calls.append("search_documents")
        
        relevant_docs = []
        image_docs = []
        
        for result in search_results:
            relevant_docs.append(result["content"])
            if result["metadata"].get("filename"):
                all_sources.append(result["metadata"]["filename"])
                
                # Check if there's an associated image
                doc_id = result["metadata"].get("document_id", "")
                image_key = f"{doc_id}_image"
                if include_images and image_key in st.session_state.document_store:
                    image_docs.append((doc_id, st.session_state.document_store[image_key]))
        
        reasoning_steps.append(f"Found {len(search_results)} relevant document chunks")
        
        # Step 2: Analyze images if present and relevant
        if image_docs and include_images:
            image_analyses = []
            for doc_id, image_data in image_docs[:3]:  # Limit to 3 images
                analysis = self.analyze_image_with_query(image_data, query)
                image_analyses.append(f"Image {doc_id}: {analysis}")
            
            if image_analyses:
                image_analysis = "\n\n".join(image_analyses)
                tool_calls.append("image_analysis")
                reasoning_steps.append(f"Analyzed {len(image_analyses)} images")
        
        # Step 3: Extract facts from relevant documents
        all_facts = []
        for doc_content in relevant_docs[:3]:
            facts = self.tools.extract_facts(doc_content)
            all_facts.extend(facts)
        
        if all_facts:
            tool_calls.append("extract_facts")
            reasoning_steps.append(f"Extracted {len(all_facts)} key facts")
        
        # Step 4: Get conversation context
        context = self.memory_manager.get_context()
        
        # Step 5: Generate response using Gemini
        system_prompt = """You are an intelligent AI agent with access to documents and images. 
        Provide accurate, well-reasoned answers based on the available information. 
        Always cite your sources and be transparent about limitations."""
        
        user_prompt = f"""
        Question: {query}
        
        Relevant Documents:
        {chr(10).join(relevant_docs[:3])}
        
        Extracted Facts:
        {chr(10).join(all_facts[:5])}
        
        Image Analysis:
        {image_analysis if image_analysis else "No relevant images found"}
        
        Conversation Context:
        {context}
        
        Please provide a comprehensive answer, citing specific sources when possible.
        """
        
        try:
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            response = self.model.generate_content(full_prompt)
            content = response.text
            reasoning_steps.append("Generated response using Gemini")
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            content = f"I apologize, but I encountered an error while processing your query: {str(e)}"
        
        # Step 6: Calculate confidence score
        confidence = self.calculate_confidence(query, content, all_sources)
        reasoning_steps.append(f"Calculated confidence score: {confidence:.2f}")
        
        # Create response object
        agent_response = AgentResponse(
            content=content,
            sources=list(set(all_sources)),
            confidence_score=confidence,
            reasoning_steps=reasoning_steps,
            tool_calls=tool_calls,
            image_analysis=image_analysis
        )
        
        # Store in memory
        self.memory_manager.add_conversation(query, agent_response)
        
        # Store extracted facts
        for fact in all_facts[:5]:
            self.memory_manager.add_fact(fact, "document_analysis", confidence)
        
        return agent_response

def display_chat_message(message: Dict, is_user: bool = True):
    """Display a chat message in the UI"""
    with st.chat_message("user" if is_user else "assistant"):
        if is_user:
            st.write(message.get("content", ""))
        else:
            response = message.get("response")
            if response:
                st.write(response.content)
                
                # Show confidence and sources in an expander
                with st.expander("📊 Response Details", expanded=False):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Confidence Score", f"{response.confidence_score:.2f}")
                        if response.sources:
                            st.write("**Sources:**")
                            for source in response.sources:
                                st.write(f"• {source}")
                    
                    with col2:
                        if response.tool_calls:
                            st.write("**Tools Used:**")
                            for tool in response.tool_calls:
                                st.write(f"• {tool}")
                        
                        if response.reasoning_steps:
                            st.write("**Reasoning Steps:**")
                            for i, step in enumerate(response.reasoning_steps, 1):
                                st.write(f"{i}. {step}")
                
                # Show image analysis if available
                if response.image_analysis:
                    with st.expander("🖼️ Image Analysis", expanded=False):
                        st.write(response.image_analysis)

def main():
    """Main Streamlit application"""
    
    # App title and description
    st.title("🤖 AI Agent Assistant")
    st.markdown("*Intelligent document analysis with multimodal capabilities*")
    
    # Sidebar for configuration and document management
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key input
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Get your API key from Google AI Studio"
        )
        
        if not api_key:
            st.error("Please enter your Google Gemini API key to continue")
            st.stop()
        
        st.header("📚 Document Management")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'txt', 'html', 'jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
            accept_multiple_files=True,
            help="Upload documents and images for analysis"
        )
        
        # Initialize agent (cached)
        if 'agent' not in st.session_state:
            try:
                st.session_state.agent = GeminiAgent(api_key)
                st.success("Agent initialized successfully!")
            except Exception as e:
                st.error(f"Failed to initialize agent: {str(e)}")
                st.stop()
        
        # Process uploaded files
        if uploaded_files and st.button("📤 Process Documents"):
            with st.spinner("Processing documents..."):
                success_count = st.session_state.agent.load_documents(uploaded_files)
                st.success(f"Successfully processed {success_count}/{len(uploaded_files)} documents")
        
        # Document status
        if st.session_state.document_store:
            st.write("**Loaded Documents:**")
            for doc_id, doc_info in st.session_state.document_store.items():
                if not doc_id.endswith("_image"):
                    filename = doc_info.get("metadata", {}).get("filename", doc_id)
                    file_type = doc_info.get("metadata", {}).get("file_type", "unknown")
                    st.write(f"• {filename} ({file_type})")
        
        st.divider()
        
        # Memory management
        st.header("🧠 Memory")
        conv_count = len(st.session_state.get('conversation_history', []))
        facts_count = len(st.session_state.get('facts_store', []))
        
        st.write(f"**Conversations:** {conv_count}")
        st.write(f"**Facts Stored:** {facts_count}")
        
        if st.button("🗑️ Clear Memory"):
            st.session_state.agent.memory_manager.clear_memory()
            st.success("Memory cleared!")
            st.rerun()
        
        # Settings
        st.header("⚙️ Settings")
        include_images = st.checkbox("Include Image Analysis", value=True)
        
        # Help section
        with st.expander("ℹ️ How to Use"):
            st.write("""
            1. Enter your Google Gemini API key
            2. Upload documents (PDF, Word, text, images)
            3. Click 'Process Documents'
            4. Start asking questions about your documents
            5. View detailed analysis and sources
            """)
    
    # Main chat interface
    st.header("💬 Chat Interface")
    
    # Display conversation history
    for item in st.session_state.get('conversation_history', []):
        display_chat_message({"content": item["user_input"]}, is_user=True)
        display_chat_message({"response": item["response"]}, is_user=False)
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your documents..."):
        if not st.session_state.document_store:
            st.warning("Please upload and process some documents first!")
        else:
            # Display user message
            display_chat_message({"content": prompt}, is_user=True)
            
            # Generate and display response
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.process_query(prompt, include_images=include_images)
                    display_chat_message({"response": response}, is_user=False)
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
    
    # Statistics in footer
    if st.session_state.get('conversation_history'):
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Conversations", len(st.session_state.conversation_history))
        
        with col2:
            doc_count = len([k for k in st.session_state.document_store.keys() if not k.endswith("_image")])
            st.metric("Documents Loaded", doc_count)
        
        with col3:
            image_count = len([k for k in st.session_state.document_store.keys() if k.endswith("_image")])
            st.metric("Images Loaded", image_count)
        
        with col4:
            avg_confidence = sum(item["response"].confidence_score for item in st.session_state.conversation_history) / len(st.session_state.conversation_history)
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")

if __name__ == "__main__":
    main()