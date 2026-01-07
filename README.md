# Agentic RAG Project

A modern Agentic RAG (Retrieval-Augmented Generation) system built with Pydantic AI, FastAPI, and PostgreSQL (pgvector). This project provides a scalable, modular, and production-ready foundation for document-based AI applications.

## 📋 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [License](#license)

## ✨ Features

- 🧠 Pydantic AI-based intelligent agent system
- 🔍 Multiple search strategies: Vector Search and Hybrid Search
- 📄 Advanced PDF processing: Table and image extraction with Docling
- 💾 Database: PostgreSQL + pgvector extension
- 🌊 Real-time streaming: Live responses via Server-Sent Events
- 🎯 Session management: Conversation history and context retention
- 🐳 Docker containerization: Easy deployment
- 🔧 Type safety: Reliable data handling with Pydantic models
- ⚡ Instant ingestion: Upload documents and query immediately

## 🛠️ Tech Stack

### Backend

- [Pydantic AI](https://ai.pydantic.dev/) - AI Agent Framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [LangChain](https://langchain.readthedocs.io/) - Document processing and embeddings
- [Docling](https://github.com/DS4SD/docling) - PDF extraction and analysis
- [AsyncPG](https://magicstack.github.io/asyncpg/) - PostgreSQL async client
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity search

### Frontend

- [Streamlit](https://streamlit.io/) - Interactive web interface

### Infrastructure

- [PostgreSQL 17](https://www.postgresql.org/) - Primary database
- [Docker Compose](https://docs.docker.com/compose/) - Multi-container deployment
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer

## 🚀 Installation

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/daniellopez882/ntt-rag-project.git
cd ntt-rag-project
```

### 2. Set Up Environment Variables

Create a `.env` file from `.env.example`:

```bash
# API Configuration
APP_ENV=development
LOG_LEVEL=INFO
APP_HOST=0.0.0.0
APP_PORT=8058
API_URL=http://api:8058

# Streamlit Configuration
SERVER_PORT=8501
SERVER_HOST=0.0.0.0

# PostgreSQL Vector DB Configuration
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=postgres
DB_PORT=5432
DB_NAME=vector_db

# OpenAI API Configuration (required)
OPENAI_API_KEY=your_openai_api_key
LLM_CHOICE=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

### 3. Start with Docker

```bash
docker-compose up -d
docker-compose logs -f
```

## 📚 Usage

### Web Interface

Access the Streamlit UI: <http://localhost:8501>

### API Usage

FastAPI documentation: <http://localhost:8058/docs>

## 🏗️ Project Structure

```text
ntt_rag_project/
├── agent/                  # AI Agent and business logic
│   ├── agent.py           # Main Pydantic AI agent
│   ├── api.py             # FastAPI endpoints
│   ├── db_utils.py        # Database operations
│   ├── models.py          # Pydantic models
│   ├── prompts.py         # System prompts
│   ├── providers.py       # LLM and embedding providers
│   └── tools.py           # Agent tools
├── ingestion/             # Document processing
│   ├── chunker.py         # Text chunking
│   ├── extract_files.py   # PDF extraction
│   └── ingest.py          # Main ingestion pipeline
├── ui/                    # Streamlit UI
│   └── app.py
├── sql/                   # Database schema
│   └── schema.sql
├── docker-compose.yml     # Container orchestration
├── Dockerfile
└── pyproject.toml         # Python dependencies
```

## 📄 License

Apache2.0 License - See `LICENCE` for details.

---

Author: daniellopez882
