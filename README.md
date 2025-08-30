# AI Assistant for Technical Documentation (Conversational RAG Prototype)

This project is a functional prototype of a **conversational Retrieval-Augmented Generation (RAG) system**. It's designed to answer questions about a specific knowledge base—in this case, technical documentation for Azure AI, Drupal, and React—while maintaining the context of an ongoing conversation.

This was developed as a hands-on project to gain practical experience with the full lifecycle of a modern AI application, from a sophisticated data pre-processing pipeline to building a memory-enabled query engine.

## How it Works

The application follows a robust, multi-stage RAG architecture to ensure high-quality, context-aware answers:

1.  **Data Ingestion & OCR (`preprocess.py`):**
    *   Source PDF documents from the `data/` directory are processed using **OCR (Tesseract)** via the `Unstructured` library. This allows for reliable text extraction even from complex, scanned, or image-based PDFs where standard text extraction fails.
    *   A first pass of basic cleaning is applied, and the raw extracted text is saved to a `processed_data/` directory.

2.  **Data Refinement (`refine_processed_data.py`):**
    *   A second script applies a more aggressive and specific set of cleaning rules (using regular expressions) to the text files in `processed_data/`.
    *   This two-step process efficiently separates the slow, intensive OCR step from the fast, iterative text-cleaning step, allowing for rapid refinement of the knowledge base.

3.  **Indexing (`main.py`):**
    *   The final, cleaned text documents are loaded and split into overlapping chunks to preserve semantic context.
    *   The chunks are converted into numerical vectors (embeddings) using Google's `embedding-001` model and stored in a local **ChromaDB** vector database.

4.  **Conversational Retrieval and Generation (`main.py`):**
    *   The system uses a **History-Aware Retriever**. When a follow-up question is asked, it first uses an LLM to intelligently rephrase the new question into a detailed, standalone query that incorporates context from the chat history.
    *   This rephrased query is used to retrieve the most relevant text chunks from ChromaDB.
    *   The user's question, the chat history, and the retrieved context are then passed to a **Google Gemini** model to generate a final, context-aware, and conversational answer.

## Key Features

*   **Robust Data Ingestion:** Handles complex or scanned PDFs by forcing OCR.
*   **Iterative Cleaning Pipeline:** A smart two-script approach that separates slow OCR from fast text refinement, demonstrating a practical data engineering workflow.
*   **Conversational Memory:** Remembers the context of the conversation to answer follow-up questions intelligently.
*   **Advanced RAG Architecture:** Implements a sophisticated, multi-chain pipeline that reflects modern best practices.

## Technology Stack

*   **Core Language:** Python (3.9)
*   **AI/LLM Framework:** LangChain
*   **LLM & Embeddings:** Google Gemini (via `langchain-google-genai`)
*   **Vector Database:** ChromaDB (local, file-based)
*   **Document Processing:** `Unstructured`, `PyPDF`, `Pillow`
*   **OCR Engine:** Tesseract (via `pytesseract`)
*   **PDF Rendering Engine:** Poppler

## How to Run Locally

1.  **Prerequisites:**
    *   Ensure you have [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) and [Poppler](https://github.com/oschwartz10612/poppler-windows/releases/) installed and available in your system's PATH.

2.  **Clone the repository:**
    ```bash
    git clone [URL-of-your-repository]
    cd [repository-name]
    ```
3.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    ```
4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Set up your API Key:**
    *   Create a `.env` file in the root of the project.
    *   Add your Google API key: `GOOGLE_API_KEY="AIza..."`
6.  **Add your documents:**
    *   Place your source PDF files inside the `data/` directory.
7.  **Run the pre-processing pipeline:**
    *   First, run the slow OCR extraction step:
    ```bash
    python preprocess.py
    ```
    *   (Optional) You can now inspect the files in `processed_data/` and refine the cleaning rules in `refine_processed_data.py`. Then, run the fast cleaning script:
    ```bash
    python refine_processed_data.py
    ```
8.  **Run the main application:**
    ```bash
    python main.py
    ```