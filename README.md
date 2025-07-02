# PepperonIA

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) application using Large Language Models (LLMs). RAG combines powerful LLMs with vector-based retrieval of external knowledge sources to provide more accurate and contextually relevant responses.

## Features

- **Document Embeddings**: Index and embed documents using vector databases.
- **Semantic Search**: Retrieve relevant documents based on user queries.
- **LLM Integration**: Generate answers by combining retrieved documents with prompts.
- **Web UI**: Interactive Gradio interface for easy testing.

## Prerequisites

- Python 3.8+
- `requirements.txt`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-llm-project.git
   cd rag-llm-project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3.  Pull the LLM model :
    ```bash
    ollama pull Llama3.2
    ```

## Usage

Simply run the Gradio app:
```bash
python gradio_rag_app.py
```
This will start a local web server. Open your browser and navigate to `http://localhost:XXXX` to interact with the RAG interface.

## License

This project is licensed under the MIT License.