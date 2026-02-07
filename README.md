# 🧭 Semantic Search API [Project ID: P-697]

A FastAPI-based semantic search system that processes company and product data from Excel/CSV, generates AI embeddings with OpenAI, and enables intelligent search via PostgreSQL + pgvector.

---

## 📚 Table of Contents

[About](#-about)  
[Features](#-features)  
[Tech Stack](#-tech-stack)  
[Installation](#-installation)  
[Usage](#-usage)  
[Configuration](#-configuration)  
[Screenshots](#-screenshots)  
[API Documentation](#-api-documentation)  
[Contact](#-contact)  
[Acknowledgements](#-acknowledgements)

---

## 🧩 About

This project provides an intuitive, production-ready API for **semantic search** over company and product data. It addresses the need to search by meaning (e.g. “high-quality fastener suppliers”) rather than exact keywords. Data is uploaded as Excel/CSV, grouped by industry, scored for quality, embedded with OpenAI’s text-embedding model, and stored in PostgreSQL with the pgvector extension. Users can run natural-language queries with optional filters and get ranked results combining completeness and semantic similarity.

**Key goals:** scalable vector search, flexible filtering (industry/country), quality-aware ranking, and a simple upload → embed → search workflow.

---

## ✨ Features

- **Semantic search** – Natural-language and Chinese/English queries with meaning-based matching via embeddings.
- **Excel/CSV upload** – Ingest company/product data with industry grouping and automatic quality scoring.
- **Vector storage** – PostgreSQL + pgvector for embedding storage and similarity search.
- **Multi-factor ranking** – Combines completeness score (60%) and semantic similarity (40%).
- **Filtering** – Industry and country filters; product-code and metric-intent detection (e.g. “highest quantity”).
- **Feedback API** – Submit user feedback (keep/reject/compare) on search results.
- **Production-ready** – Async FastAPI, Gunicorn + Uvicorn workers, connection pooling, Render.com deployment support.

---

## 🧠 Tech Stack

| Category   | Technologies |
|-----------|--------------|
| **Languages** | Python 3.8+ |
| **Frameworks** | FastAPI, Uvicorn, Gunicorn |
| **Database** | PostgreSQL with pgvector |
| **AI / Embeddings** | OpenAI API (text-embedding-3-small) |
| **Data & ORM** | Pandas, SQLAlchemy (async) |
| **Tools** | python-dotenv, Docker-friendly, Render.com |

---

## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/Phoenix-dev11/Semantic_search_V2.git

# Navigate to the project directory
cd Semantic_search_V2

# Create virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Prerequisites:** Python 3.8+, PostgreSQL with pgvector extension, OpenAI API key.

Enable pgvector in your database:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

---

## 🚀 Usage

**Development (with auto-reload):**

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Production (Gunicorn):**

```bash
python start.py
# or: web: python start.py (Procfile)
```

Then open your browser or API client:

👉 **Base URL:** [http://localhost:8000](http://localhost:8000)  
👉 **Interactive API docs:** [http://localhost:8000/docs](http://localhost:8000/docs) (when `DISABLE_DOCS` is not set)

---

## 🧾 Configuration

Create a `.env` file (use `env.example` as a template):

**Required:**

- `DATABASE_URL` – PostgreSQL connection string (e.g. `postgresql://user:password@host:5432/database_name`)
- `OPENAI_API_KEY` – Your OpenAI API key for embeddings

**Optional (defaults shown):**

- `EMBEDDING_MODEL=text-embedding-3-small`
- `DISABLE_DOCS=false` – Set to `true` to disable `/docs` and `/redoc`
- `PORT=8000`
- `WEB_CONCURRENCY=4`
- `ENVIRONMENT=development`

**Example:**

```env
DATABASE_URL=postgresql://user:password@localhost:5432/semantic_search
OPENAI_API_KEY=your_openai_api_key_here
```

---

## 🖼 Screenshots

Add demo images, GIFs, or UI preview screenshots here.

*Example: Swagger UI at `/docs`, sample search request/response, or dashboard screens.*

---

## 📜 API Documentation

Main endpoints (see [http://localhost:8000/docs](http://localhost:8000/docs) for full request/response schemas when docs are enabled):

| Method | Endpoint | Description |
|--------|----------|-------------|
| **GET** | `/` | Health check |
| **GET** | `/health` | Detailed health status |
| **POST** | `/api/upload` | Upload Excel/CSV for processing and embedding |
| **POST** | `/api/search` | Semantic search (body: `query_text`, `filters`, `top_k`) |
| **GET** | `/api/debug/industries` | Debug: list industries |
| **GET** | `/api/debug/standard-scoring` | Debug: standard scoring info |
| **POST** | `/api/feedback` | Submit feedback on a search result (e.g. keep/reject/compare) |

**Example search request:**

```json
POST /api/search
{
  "query_text": "I need Q02 highest quantity product",
  "filters": "扣件",
  "top_k": 5
}
```

---

## 📬 Contact

- **Author:** Hiroshi Nagaya
- **Email:** phoenixryan1111@gmail.com  
- **GitHub:** @Phoenix-dev11
- **Website/Portfolio:** hiroshi-nagaya.vercel.app 

*(Replace with your details.)*

---

## 🌟 Acknowledgements

- **OpenAI** – Text embedding API (text-embedding-3-small).
- **FastAPI** – Modern async API framework.
- **pgvector** – PostgreSQL extension for vector similarity search.
- **Render.com** – Deployment configuration (`render.yaml`, `Procfile`).

---

**Version:** 2.0.0  
**Status:** Production Ready
