# AI Agent Assistant

A Streamlit-based AI assistant that processes and analyzes documents using Google's Gemini API. The application supports multiple document formats, maintains conversation memory, and provides source attribution for responses.

## Features

- **Document Processing**: Supports PDF, DOCX, TXT, HTML, and various image formats
- **Vector Search**: Utilizes ChromaDB for efficient document retrieval
- **Conversation Memory**: Maintains context across interactions
- **Multimodal Support**: Processes both text and image inputs
- **Source Attribution**: Provides references for all generated responses

## Prerequisites

- Python 3.8+
- Google Gemini API key
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ShivamShrivastava18/RAG-Agent.git
   cd klypup2
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Create a `.env` file in the project root and add your Google Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser and navigate to the provided local URL (typically http://localhost:8501)

3. In the sidebar:
   - Enter your Google Gemini API key
   - Upload documents using the file uploader
   - Click "Process Documents" to index them

4. Use the chat interface to ask questions about your documents

## Project Structure

- `app.py`: Main application file containing the Streamlit interface and core logic
- `requirements.txt`: Lists all Python dependencies
- `Architecture.md`: Detailed system architecture documentation
- `.gitignore`: Specifies intentionally untracked files to ignore

## Dependencies

Key dependencies include:
- streamlit: Web application framework
- google-generativeai: Google's Gemini API client
- chromadb: Vector database for document storage and retrieval
- sentence-transformers: Text embedding generation
- PyPDF2, python-docx: Document processing

## License

[Specify your license here]

## Support

For support or feature requests, please open an issue in the repository.
