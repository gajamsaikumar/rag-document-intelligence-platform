# Document Portal Analysis with RAG

## Project Overview
The Document Portal Analysis platform automates the process of reviewing, comparing, and querying business documents such as invoices and reports from multiple vendors. By leveraging Retrieval-Augmented Generation (RAG) and large language models, it provides insights, highlights differences, and enables interactive document exploration, reducing manual work and improving decision-making efficiency.

## Use Case
Businesses often receive numerous reports or invoices from global vendors, which are time-consuming to review manually. This portal provides a unified interface for analyzing documents, comparing them, and querying content interactively, streamlining operational workflows.

## Features

### Secure User Access
- Users can register and log in with credentials  
- All document services are restricted to authenticated users

### Document Analysis
- Upload a single PDF to extract insights  
- Provides detailed information from the document content

![Document Analysis](doc_analysis.PNG)

### Document Comparison
- Upload two PDFs to view differences side-by-side  
- Useful for tracking changes across vendor reports or invoices

![Document Compare](doc_compare.PNG)

### Single Document Chat
- Query a single document using natural language  
- System returns relevant answers based on the document's content

![Single Document Chat](doc_chat.PNG)

### Multi-Document Chat
- Upload multiple PDFs and query across all documents  
- Enables comprehensive analysis from multiple sources

![Multi-Document Chat](doc_chat.PNG)

## Technologies
- Python  
- LangChain  
- FastAPI  
- Streamlit  
- Google Embeddings  
- FAISS  
- Groq LLM Models  
- Google Gemini LLM  
- Docker  
- CI/CD with GitHub Actions  
- AWS (ECR, ECS, Fargate, Secret Manager)

## How It Works
1. Documents are converted into vector embeddings  
2. FAISS is used for efficient retrieval  
3. Relevant document sections are passed to LLMs via a RAG pipeline  
4. Users can interact, compare, and query documents based on context
