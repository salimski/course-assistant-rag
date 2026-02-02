# Course Assistant RAG

A Retrieval-Augmented Generation (RAG) system that answers questions using course materials.

The system uses:
- **LangChain + LangGraph** for agent orchestration
- **ChromaDB** as the vector database
- **Ollama** for both embeddings and the language model (local, offline)

✅ This repository includes a **prebuilt Chroma vector database**, so it can be run immediately without re-indexing.

---

## System Requirements (Windows)

- **Windows 10 / 11**
- **Python 3.12 (REQUIRED)**
  - ⚠️ Python 3.14 is **not supported** and may cause installation failures
- **Ollama** installed and running

---

## Python Dependencies

Installed automatically from `requirements.txt`:

```txt
langgraph==0.2.45
langchain==0.3.7
langchain-ollama==0.2.0
langchain-community==0.3.7
chromadb>=0.5.24
langchain-chroma>=0.1.5
requests==2.31.0
python-dotenv==1.0.0
pypdf==5.1.0
pydantic==2.10.3

Installation & Setup (Windows CMD)
1️⃣ Open Command Prompt (CMD)

Press:

Windows key → type "cmd" → Enter

2️⃣ Clone the repository
cd %USERPROFILE%\Documents\GitHub
git clone https://github.com/salimski/course-assistant-rag.git
cd course-assistant-rag


(Alternatively: download ZIP from GitHub and extract it.)

3️⃣ Create a virtual environment (Python 3.12)

⚠️ Important: Make sure Python 3.12 is installed.

Create the virtual environment explicitly using Python 3.12:

C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe -m venv venv


Activate it:

venv\Scripts\activate


Verify Python version:

python --version


✅ Must show Python 3.12.x

4️⃣ Install Python packages
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Ollama Setup (MANDATORY)

This project requires two Ollama models:

One for embeddings

One for chat generation

1️⃣ Make sure Ollama is running

Open Ollama normally (background service).

2️⃣ Pull the embedding model
ollama pull nomic-embed-text

3️⃣ Pull the chat model (example)
ollama pull llama3.1:8b


If your system uses a different model, pull that instead.

To see available models:

ollama list

Running the Project
▶️ Run the main interactive agent

From the project root:

cd C:\Users\%USERNAME%\Documents\GitHub\course-assistant-rag
venv\Scripts\activate
python main.py


You should see:

“System ready”

“Interactive Mode”

Then you can ask questions like:

What is TF-IDF?

What is PageRank?

Type quit to exit.

Running the RAG Tool Test (Optional)

⚠️ Must be run from the project root, not inside tools/.

cd C:\Users\%USERNAME%\Documents\GitHub\course-assistant-rag
venv\Scripts\activate
python tools\rag_tool.py


This verifies:

ChromaDB loading

Embeddings

Similarity search

Prebuilt Chroma Vector Database

This repository includes a prebuilt vector store:

./chroma_db


No indexing is required on first run

Queries will work immediately

Rebuilding the database (optional)

Delete chroma_db/

Place PDFs or TXT files into:

data/course_materials/


Run the RAG tool again to rebuild embeddings

Common Issues & Fixes
❌ pip not recognized

Use:

python -m pip install -r requirements.txt

❌ NumPy / C++ build errors

You are using the wrong Python version.

Fix:

python --version


Must be Python 3.12

❌ Ollama model not found (404)

Example:

model 'llama3.1:8b' not found


Fix:

ollama pull llama3.1:8b

❌ Embedding model not found

Example:

model 'nomic-embed-text' not found


Fix:

ollama pull nomic-embed-text

❌ No documents found

If rebuilding the DB, ensure files exist in:

data/course_materials/

Notes for Reviewers / Lecturers

Project is fully reproducible

Uses local models only (no external APIs)

Prebuilt vector database included

Python 3.12 required for stability on Windows

Author

Salim
