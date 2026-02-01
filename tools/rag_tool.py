"""
RAG Tool - Handles course material retrieval using ChromaDB and Ollama embeddings
"""

import os

# ---- Disable Chroma telemetry noise (prevents the capture() error spam) ----
# Must be set before Chroma client is created.
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"] = "False"

from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document


class RAGTool:
    """
    Retrieval-Augmented Generation tool for course materials.

    This tool:
    1. Loads documents from the data/course_materials folder (PDF + TXT)
    2. Splits them into chunks
    3. Embeds them using Ollama's nomic-embed-text model
    4. Stores embeddings in ChromaDB
    5. Provides semantic search functionality
    """

    def __init__(self, data_dir: str = "data/course_materials", persist_dir: str = "chroma_db"):
        self.data_dir = data_dir
        self.persist_dir = persist_dir

        # Initialize Ollama embeddings
        print("Initializing embeddings model...")
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url="http://localhost:11434"
        )

        # Initialize or load the vector store
        self.vector_store = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """
        Initialize ChromaDB vector store.
        If it exists, load it. Otherwise, create it from documents.
        """
        if os.path.exists(self.persist_dir):
            print(f"Loading existing vector store from {self.persist_dir}...")
            self.vector_store = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings
            )
            # Safer than using private attributes when possible
            try:
                count = self.vector_store._collection.count()
            except Exception:
                count = "unknown"
            print(f"Loaded {count} documents from vector store.")
        else:
            print("No existing vector store found. Creating new one...")
            self._create_vector_store()

    def _load_documents(self) -> List[Document]:
        """
        Load all documents from the data directory.
        Supports both PDF and text files.

        Returns:
            List of LangChain Document objects
        """
        print(f"Loading documents from {self.data_dir}...")
        all_documents: List[Document] = []

        # Load PDF files
        print("Loading PDF files...")
        pdf_loader = DirectoryLoader(
            self.data_dir,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True
        )

        try:
            pdf_docs = pdf_loader.load()
            print(f"  ✓ Loaded {len(pdf_docs)} pages from PDF files")
            all_documents.extend(pdf_docs)
        except Exception as e:
            print(f"  ⚠ No PDFs found or error loading PDFs: {e}")

        # Load text files
        print("Loading text files...")
        txt_loader = DirectoryLoader(
            self.data_dir,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True
        )

        try:
            txt_docs = txt_loader.load()
            print(f"  ✓ Loaded {len(txt_docs)} text files")
            all_documents.extend(txt_docs)
        except Exception as e:
            print(f"  ⚠ No text files found or error loading: {e}")

        print(f"\n📚 Total documents loaded: {len(all_documents)}")
        return all_documents

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks for embedding.
        """
        print("Splitting documents into chunks...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)

        # Filter out chunks that are too short or just metadata
        filtered_chunks: List[Document] = []
        for chunk in chunks:
            content = chunk.page_content.strip()
            if len(content) > 100 and not content.startswith("Credit:"):
                filtered_chunks.append(chunk)

        print(f"Created {len(chunks)} chunks, filtered to {len(filtered_chunks)} quality chunks.")
        return filtered_chunks

    def _create_vector_store(self):
        """
        Create a new vector store from documents.
        """
        documents = self._load_documents()

        if not documents:
            print("No documents found! Please add files to data/course_materials/")
            return

        chunks = self._split_documents(documents)

        print("Creating embeddings and storing in ChromaDB...")
        print("(This may take a minute...)")

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.persist_dir
        )

        print(f"Vector store created successfully with {len(chunks)} chunks!")

    def search(self, query: str, k: int = 5) -> str:
        """
        Search for relevant course material.
        """
        if not self.vector_store:
            return "Error: No course materials loaded in the vector store."

        print(f"\nSearching for: '{query}'")

        results = self.vector_store.similarity_search(query, k=k)
        if not results:
            return "No relevant information found in course materials."

        context = "\n\n" + "=" * 50 + "\n\n"
        context += "\n\n".join(
            [f"📄 Chunk {i + 1}:\n{doc.page_content}" for i, doc in enumerate(results)]
        )

        print(f"Found {len(results)} relevant chunks.")
        return context


RAG_TOOL_DESCRIPTION = """
Search through Information Retrieval course materials including lecture notes, papers, and documentation.
Use this tool when the user asks about IR concepts, course content, algorithms, or theory.
"""


def create_rag_tool():
    return RAGTool()


if __name__ == "__main__":
    print("Testing RAG Tool...")
    print("=" * 50)

    rag = create_rag_tool()

    test_query = "What is PageRank?"
    print(f"\nTest Query: {test_query}")
    print("=" * 50)

    result = rag.search(test_query)
    print("\nResult:")
    print(result)
