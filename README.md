# 🤖 RAG-Based Question Answering System

A powerful **Retrieval-Augmented Generation (RAG)** system that enables intelligent question answering from PDF documents using LangChain, HuggingFace models, and FAISS vector search.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)](https://github.com/langchain-ai/langchain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Installation](#-installation)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Examples](#-examples)
- [Security](#-security)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

## ✨ Features

- 📄 **PDF Document Processing**: Automatically loads and processes PDF documents
- 🔍 **Semantic Search**: Uses FAISS for efficient vector similarity search
- 🧠 **Smart Chunking**: Intelligently splits documents with configurable overlap
- 💬 **Context-Aware Answers**: Provides answers grounded in source documents
- 📚 **Source Attribution**: Shows which parts of the document informed the answer
- 🔒 **Secure Token Management**: Never hardcodes API tokens
- 🎯 **Flexible Interface**: Both Jupyter notebook and command-line interfaces
- ⚙️ **Configurable**: Easy-to-adjust parameters for different use cases

## 🔬 How It Works

The RAG system operates in several stages:

```
1. Document Loading → 2. Text Chunking → 3. Embedding Generation
                                                    ↓
6. Answer Generation ← 5. Context Retrieval ← 4. Vector Storage
```

1. **Document Loading**: Reads PDF and extracts text
2. **Text Chunking**: Splits text into manageable chunks with overlap
3. **Embedding Generation**: Converts chunks to vector embeddings using `all-MiniLM-L6-v2`
4. **Vector Storage**: Stores embeddings in FAISS index for fast retrieval
5. **Context Retrieval**: Finds most relevant chunks for a given question
6. **Answer Generation**: Uses LLM (Zephyr-7B) to generate answers based on context

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- HuggingFace account (for API token)

### Step 1: Clone the Repository

```bash
git clone https://github.com/tushar-patel28/rag-qa-system.git
cd rag-qa-system
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Get your HuggingFace API token:
   - Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
   - Create a new token (read access is sufficient)
   
3. Add your token to `.env`:
   ```
   HUGGINGFACEHUB_API_TOKEN=your_actual_token_here
   ```

## 💻 Usage

### Option 1: Jupyter Notebook (Interactive)

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Open `rag_qa_system.ipynb`

3. Run cells sequentially

4. Modify the example questions or add your own!

### Option 2: Command Line Interface

#### Single Question Mode

```bash
python rag_qa_system.py path/to/your/document.pdf -q "What is artificial intelligence?"
```

#### Interactive Mode

```bash
python rag_qa_system.py path/to/your/document.pdf
```

Then enter questions interactively. Type `quit` to exit.

#### Without Source Documents

```bash
python rag_qa_system.py path/to/your/document.pdf -q "Your question" --no-sources
```

### Option 3: Python Script

```python
from rag_qa_system import RAGQASystem

# Initialize system
system = RAGQASystem()

# Load PDF
documents = system.load_pdf("your_document.pdf")

# Create vector store
system.create_vector_store(documents)

# Setup QA chain
system.setup_qa_chain()

# Ask questions
result = system.ask("What is machine learning?")
```

## ⚙️ Configuration

You can customize the system behavior by modifying the configuration:

```python
config = {
    "chunk_size": 500,           # Size of text chunks
    "chunk_overlap": 100,        # Overlap between chunks
    "embedding_model": "all-MiniLM-L6-v2",  # Embedding model
    "llm_model": "HuggingFaceH4/zephyr-7b-beta",  # Language model
    "temperature": 0.5,          # LLM creativity (0-1)
    "max_tokens": 512,           # Maximum response length
    "retriever_k": 4             # Number of chunks to retrieve
}

system = RAGQASystem(config=config)
```

### Configuration Parameters Explained

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `chunk_size` | Number of characters per chunk | 500 | 200-1000 |
| `chunk_overlap` | Overlapping characters between chunks | 100 | 50-200 |
| `embedding_model` | HuggingFace embedding model | all-MiniLM-L6-v2 | - |
| `llm_model` | HuggingFace language model | zephyr-7b-beta | - |
| `temperature` | Randomness in generation | 0.5 | 0.0-1.0 |
| `max_tokens` | Maximum answer length | 512 | 256-2048 |
| `retriever_k` | Chunks retrieved per query | 4 | 2-10 |

## 📁 Project Structure

```
rag-qa-system/
├── rag_qa_system.py          # Command-line interface
├── rag_qa_system.ipynb        # Jupyter notebook interface
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variable template
├── .gitignore                # Git ignore rules
├── README.md                 # This file
└── LICENSE                   # MIT License
```

## 📝 Examples

### Example 1: Basic Question

**Input:**
```python
ask_question("What is artificial intelligence?")
```

**Output:**
```
Question: What is artificial intelligence?
================================================================================

Answer:
Artificial Intelligence is a computing concept that enables machines to think 
and solve complex problems like humans do through a self-correcting learning 
process...

Source Documents:
================================================================================
--- Source 1 ---
Generally speaking, Artificial Intelligence is a computing concept that helps 
a machine think and solve complex problems as we humans do with our intelligence...
```

### Example 2: Specific Technical Query

```python
ask_question("What are the subfields of AI mentioned in the document?")
```

### Example 3: Without Sources (Cleaner Output)

```python
ask_question("How does machine learning work?", show_sources=False)
```

## 🔒 Security

### Best Practices

✅ **DO:**
- Store API tokens in environment variables or `.env` files
- Add `.env` to `.gitignore`
- Use `getpass` for interactive token input in notebooks
- Regularly rotate your API tokens
- Use read-only tokens when possible

❌ **DON'T:**
- Hardcode tokens in source code
- Commit `.env` files to version control
- Share tokens in public repositories
- Use tokens with more permissions than needed

### Revoking Exposed Tokens

If you accidentally expose a token:

1. Go to [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Delete the exposed token
3. Generate a new token
4. Update your `.env` file

## 🔧 Troubleshooting

### Common Issues

#### Issue: `FileNotFoundError: PDF file not found`

**Solution:** Ensure the PDF file path is correct and the file exists.

```bash
# Check if file exists
ls path/to/your/file.pdf
```

#### Issue: `Authentication Error` from HuggingFace

**Solution:** 
1. Check your token is set correctly
2. Verify the token has not expired
3. Ensure you have internet connection

```bash
# Test token
echo $HUGGINGFACEHUB_API_TOKEN
```

#### Issue: `Out of Memory Error`

**Solution:** Reduce chunk size or process fewer documents at once:

```python
config = {
    "chunk_size": 300,  # Smaller chunks
    "retriever_k": 2    # Fewer retrieved chunks
}
```

#### Issue: Poor Answer Quality

**Solution:** Try adjusting these parameters:
- Increase `chunk_overlap` for better context
- Adjust `temperature` (lower = more focused)
- Increase `retriever_k` for more context

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [HuggingFace](https://huggingface.co/) for models and embeddings
- [FAISS](https://github.com/facebookresearch/faiss) for efficient vector search
- Original inspiration from Assignment_3a

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**⭐ If you find this project helpful, please consider giving it a star!**