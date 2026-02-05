# Course Assistant RAG - Agentic RAG System

**Information Retrieval Course - Final Assignment**

**Track 3: Agentic RAG**

**Students:**
- 314777475 Salim B
- 213017049 Habib N

---

## Overview

Agentic RAG system using LangGraph that combines:
- RAG (ChromaDB + Ollama) for course materials
- Weather API (Open-Meteo)
- Holiday API (Hebcal)
- Calendar management

**Agentic Features:**
- Auto-chaining (e.g., weather on exam day)
- Multi-tool orchestration
- Conditional routing

---

## Requirements

- **Windows 10/11**
- **Python 3.12** (not 3.14)
- **Ollama** with models:
  - `nomic-embed-text`
  - `llama3.1:8b`

---

## Quick Setup

```cmd
ollama pull nomic-embed-text
ollama pull llama3.1:8b
git clone https://github.com/salimski/course-assistant-rag.git
cd course-assistant-rag
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook course_assistant_rag.ipynb

```



## Dependencies
```
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
jupyter>=1.0.0
notebook>=7.0.0
ipykernel>=6.25.0
```

## Running
```cmd
venv\Scripts\activate
jupyter notebook course_assistant_rag.ipynb
```

**Command Line:**
```markdown
venv\Scripts\activate
python main.py
```

## Project Structure
```python
course-assistant-rag/
├── course_assistant_rag.ipynb    # Main notebook
├── main.py                       # CLI interface
├── requirements.txt              
├── tools/                        # RAG, weather, holiday, calendar
├── data/course_materials/        # Source documents
└── chroma_db/                    # Prebuilt vector DB
```

## Salim B (314777475) | Habib N (213017049)

Information Retrieval Course - February 2026

