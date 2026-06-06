# NCERT AI Tutor — Offline Personalized Learning

An entirely offline, personalized AI tutor desktop application designed to help students learn Social Studies (History, Political Science, Geography, and Economics) using official NCERT curriculum for Classes 6 to 10. The application leverages a Retrieval-Augmented Generation (RAG) pipeline to provide accurate, textbook-grounded answers without needing an internet connection.

## 🌟 Key Features

*   **100% Offline Generation:** Powered by the open-source **Phi-3 (Mini)** model via Ollama. No data leaves your machine.
*   **Curriculum-Grounded (RAG):** Combines the intelligence of LLMs with semantic search (FAISS + MiniLM Embeddings) to retrieve exact facts directly from processed NCERT textbook PDFs before generating answers.
*   **Four Core Subjects:** History, Political Science (Polity), Geography, and Economics.
*   **Personalized Dashboard:** Tracks questions asked, topics explored, identifies weak areas, and suggests practice questions.
*   **Chat Management:** Creates separate, organized chat sessions that automatically pick up on subjects. You have the ability to review history and delete old chats.
*   **Modern Premium UI:** A dynamic and beautiful dark-mode interface powered by vanilla HTML/CSS and JavaScript. 

---

## 🏗️ Architecture & Tech Stack

The application is structured as a full-stack local web desktop app:

*   **Backend:** FastAPI (Python)
    *   **Database:** SQLite + SQLAlchemy ORM (Asynchronous queries).
    *   **State Management:** Stores Users, Sessions, Chat Logs, Subject Progress, and Auto-Topic Tracking.
*   **AI & RAG Pipeline:**
    *   **LLM Engine:** Microsoft `phi3:mini` served locally via Ollama.
    *   **Semantic DB:** FAISS (Facebook AI Similarity Search) index.
    *   **Embeddings:** `all-MiniLM-L6-v2` (Sentence-Transformers) for extremely fast offline embedding.
*   **Extraction:** `pdfplumber` and `PyMuPDF` to intelligently extract and chunk text from textbooks.
*   **Frontend:** Vanilla ES6 Javascript + HTML5 + Native CSS variable integration.
    *   Served statically straight from the Python FastAPI server.

---

## 📁 Project Structure

```text
Offline-tutor-application/
│
├── backend/                  # FastAPI Application Logic
│   ├── api/                  # Routes (auth.py, chat.py, dashboard.py)
│   ├── database/             # SQLAlchemy Models & CRUD operations
│   ├── rag/                  # RAG logic (pipeline, FAISS handling, phi3 stream)
│   ├── config.py             # App configurations
│   └── main.py               # Main app factory
│
├── frontend/                 # User Interface Files
│   ├── static/
│   │   ├── css/style.css     # Premium UI styling
│   │   └── js/app.js         # Frontend application logic
│   └── index.html            # Main SPA wrapper
│
├── scripts/                  # Data Ingestion Tools
│   ├── extract_text.py       # Converts PDFs to raw text
│   ├── preprocess_chunks.py  # Cleans & splits texts into semantic chunks
│   └── generate_embeddings.py# Builds the FAISS vector database
│
├── data/                     # Raw Input Directory (Place textbook PDFs here)
├── extracted_text/           # Output of Phase 1 (Raw Texts)
├── chunks/                   # Output of Phase 2 (Text Chunk dictionaries)
├── vector_db/                # Output of Phase 3 (FAISS index & metadata)
│
├── requirements.txt          # Python dependencies
└── main.py                   # Root execution script (Starts uvicorn backend)
```

---

## 🚀 How to Set Up & Run the Application

Follow these steps to set up the knowledge base and run the application locally on your Windows machine.

### Step 1: System Requirements
1.  **Python 3.10+**
2.  **Ollama:** You must have [Ollama](https://ollama.com/) installed on your machine.
3.  Once Ollama is installed, pull the Phi-3 LLM to your machine by running this in your terminal:
    ```powershell
    ollama pull phi3:mini
    ```

### Step 2: Set Up Python Environment
Open a terminal in the root project folder (`e:\Offline-tutor-application`).
```powershell
# Activate the virtual environment
.\venv\Scripts\activate

# Install the required dependencies
pip install -r requirements.txt
```

### Step 3: Build the Knowledge Base (Do this only once!)
The AI tutor needs to digest the textbooks before it can answer questions based on them. If you ever add new PDFs into the `data/` folder, you must re-run these three scripts in sequence:

```powershell
# 1. Parse all PDFs in the data folder into raw text.
python scripts/extract_text.py

# 2. Split the raw text into logical, readable chunks for the AI.
python scripts/preprocess_chunks.py

# 3. Create Vector embeddings and save the FAISS database to disk.
python scripts/generate_embeddings.py
```

### Step 4: Run the Application!
Whenever you want to start learning, make sure your virtual environment is active, Ollama is running in the background of your PC, and simply run:

```powershell
python main.py
```

The application server will start and load up the models. 
1. Open your web browser (Chrome, Edge, etc.).
2. Navigate to **[http://localhost:8000](http://localhost:8000)**.
3. Register an account and start asking your new offline AI Tutor questions!
