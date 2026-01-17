#!/usr/bin/env python3
"""
RAG-Based Question Answering System

A command-line tool for answering questions based on PDF documents using
Retrieval-Augmented Generation (RAG) with LangChain and HuggingFace.
"""

import os
import argparse
from getpass import getpass
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint


class RAGQASystem:
    """RAG-based Question Answering System"""
    
    def __init__(self, config=None):
        """
        Initialize the RAG QA system.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config = config or {
            "chunk_size": 500,
            "chunk_overlap": 100,
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "HuggingFaceH4/zephyr-7b-beta",
            "temperature": 0.5,
            "max_tokens": 512,
            "retriever_k": 4
        }
        self.db = None
        self.qa_chain = None
        
    def load_pdf(self, pdf_path):
        """Load and chunk PDF document."""
        print(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        print(f"✓ Loaded {len(pages)} pages")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap']
        )
        documents = text_splitter.split_documents(pages)
        print(f"✓ Split into {len(documents)} chunks")
        return documents
    
    def create_vector_store(self, documents):
        """Create FAISS vector store from documents."""
        print(f"Creating embeddings using {self.config['embedding_model']}...")
        embedding_model = HuggingFaceEmbeddings(
            model_name=self.config['embedding_model']
        )
        
        print("Building FAISS vector store...")
        self.db = FAISS.from_documents(documents, embedding_model)
        print("✓ Vector store created successfully")
    
    def setup_qa_chain(self):
        """Set up the RAG chain with LLM."""
        print(f"Initializing LLM: {self.config['llm_model']}...")
        llm = HuggingFaceEndpoint(
            repo_id=self.config['llm_model'],
            temperature=self.config['temperature'],
            max_new_tokens=self.config['max_tokens']
        )
        
        print("Creating RAG chain...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=self.db.as_retriever(
                search_kwargs={"k": self.config['retriever_k']}
            ),
            return_source_documents=True
        )
        print("✓ RAG chain ready")
    
    def ask(self, query, show_sources=True, max_source_length=500):
        """
        Ask a question and get an answer.
        
        Args:
            query (str): The question to ask
            show_sources (bool): Whether to display source documents
            max_source_length (int): Maximum characters to show from each source
        
        Returns:
            dict: Result containing answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call setup_qa_chain() first.")
        
        result = self.qa_chain.invoke({"query": query})
        
        print("=" * 80)
        print(f"Question: {query}")
        print("=" * 80)
        print(f"\nAnswer:\n{result['result']}\n")
        
        if show_sources and result.get("source_documents"):
            print("\n" + "=" * 80)
            print("Source Documents:")
            print("=" * 80)
            for i, doc in enumerate(result["source_documents"], 1):
                print(f"\n--- Source {i} ---")
                content = doc.page_content[:max_source_length]
                if len(doc.page_content) > max_source_length:
                    content += "..."
                print(content)
                
                if doc.metadata:
                    print(f"\nMetadata: {doc.metadata}")
                print("-" * 80)
        
        return result


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="RAG-based Question Answering System"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to analyze"
    )
    parser.add_argument(
        "-q", "--question",
        help="Question to ask (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "--no-sources",
        action="store_true",
        help="Don't show source documents"
    )
    
    args = parser.parse_args()
    
    # Check for HuggingFace API token
    if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
        token = getpass("Enter your HuggingFace API token: ")
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = token
    
    # Initialize system
    try:
        system = RAGQASystem()
        documents = system.load_pdf(args.pdf_path)
        system.create_vector_store(documents)
        system.setup_qa_chain()
        
        # Ask question or enter interactive mode
        if args.question:
            system.ask(args.question, show_sources=not args.no_sources)
        else:
            print("\n" + "=" * 80)
            print("Interactive Mode - Enter your questions (type 'quit' to exit)")
            print("=" * 80)
            while True:
                user_query = input("\nQuestion: ").strip()
                if user_query.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                if user_query:
                    system.ask(user_query, show_sources=not args.no_sources)
    
    except FileNotFoundError:
        print(f"Error: PDF file '{args.pdf_path}' not found.")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
