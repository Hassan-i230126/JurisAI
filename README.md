# Juris AI — Pakistani Criminal Law Intelligence System

### Group Members
- Muhammad Wajdan   23i-0033
- Hassan Ali Waqar  23i-0126
- Balaaj Raza       23i-0046

---

## 1. Business Use Case
**Domain:** Pakistani Criminal Defense Chatbot.
**Value Proposition:** Juris AI serves as a specialized, real-time research companion for criminal defense lawyers in Pakistan. The legal field requires absolute precision; generic LLMs frequently hallucinate case laws and statutes. 
- **How RAG Adds Value:** By grounding the model in an offline, indexed vector database of the Pakistan Penal Code (PPC), Code of Criminal Procedure (CrPC), Qanun-e-Shahadat Order (QSO), Anti-Terrorism Act (ATA), and over 2,800 Supreme Court judgments, RAG ensures all responses are legally factual and citeable.
- **How Tools Add Value:** Tools allow the model to interact dynamically with structured data. Instead of relying solely on similarity search, the model can execute full-text SQL searches on specific penal codes, explicitly calculate legal deadlines based on CrPC rules, and persist user-session bounds via a Client CRM system.

---

## 2. Architecture Diagram & Explanation

![Architecture Diagram](docs/architecture.png)
*(Note: Please insert the architecture image at docs/architecture.png)*

**Component Breakdown:**
- **Frontend (React/Vite):** A professional, dark-themed UI with a streaming chat interface. Uses WebSockets to establish a persistent connection.
- **WebSocket Handler & Conversation Manager:** Maps incoming WebSocket messages to specific client sessions. Maintains conversation history and orchestrates the turn-by-turn LLM reasoning loop.
- **Tool Orchestrator:** Intercepts LLM tool-call requests and routes them to the appropriate Python utility (CRM, Statute Lookup, Case Search, Deadline Calculator).
- **RAG Pipeline (ChromaDB + bge-m3):** Handles document chunking and vector storage. Intercepts user queries to pull the Top-K most relevant document chunks before passing them to the LLM context.
- **LLM Engine (Ollama / phi4-mini):** The core reasoning engine running locally. Generates responses and determines when to trigger specific tools.

---

## 3. Model Selection
- **LLM Selected:** phi4-mini (Quantized via Ollama)
- **Why this model:** phi4-mini offers an exceptional balance between reasoning capability and computational efficiency. Since the system is designed to run entirely on CPU-only hardware locally (for data privacy), a small footprint is critical.
- **Performance Characteristics:** 
  - **Memory Usage:** ~4.5 GB total application memory footprint (the LLM utilizes ~3-3.5 GB).
  - **Throughput:** Averages 12-18 tokens per second on a standard modern CPU.
  - **Quantization:** Int4/Int8 quantization via Ollama's GGUF format significantly reduces memory requirements while preserving instruction-following accuracy for legal text structuring.

---

## 4. Document Collection
- **Number of Documents:** 2,800+ Supreme Court criminal judgments + 1 JSON multi-act corpus + 1 full PPC Markdown document.
- **Sources:** Custom scraped Pakistani jurisprudence and openly available legislative markdown/JSON datasets.
- **Chunking Strategy:** Documents are processed using a recursive character text splitter with overlapping boundaries to preserve legal context (e.g., keeping section clauses together).
- **Embedding Model:** bge-m3 (Running locally via Ollama). Chosen for its strong performance on multilingual and dense legal texts.
- **Vector Database:** ChromaDB (local SQLite-backed instance).
- **Retrieval Parameters:** 
  - **Top-K:** 3
  - **Similarity Metric:** Cosine Distance
  - **Distance Threshold:** 1.4

---

## 5. Tools Description

### 1. Statute Lookup
- **Description:** Performs rapid, section-level lookup of Pakistani statutes (PPC, CrPC, QSO, ATA) using FTS5 SQLite full-text search.
- **Input Schema:** `{"query": "string (the legal term or section to search)"}`
- **LLM Invocation Example:**
  ```json
  {"name": "statute_lookup", "arguments": {"query": "Section 302 PPC"}}
  ```

### 2. Case Search
- **Description:** Executes semantic search across the 2,800+ indexed Supreme Court criminal judgments in ChromaDB to find relevant precedents.
- **Input Schema:** `{"query": "string (legal concept or case topic)"}`
- **LLM Invocation Example:**
  ```json
  {"name": "case_search", "arguments": {"query": "post-arrest bail in non-bailable offeces"}}
  ```

### 3. Deadline Calculator
- **Description:** Computes procedural deadlines based on the Code of Criminal Procedure (CrPC).
- **Input Schema:** `{"action": "string (e.g. appeal)", "date": "string (YYYY-MM-DD)"}`
- **LLM Invocation Example:**
  ```json
  {"name": "deadline_calc", "arguments": {"action": "High Court Appeal", "date": "2023-10-01"}}
  ```

### 4. Client CRM
- **Description:** Creates, searches, and manages confidential client profiles, allowing the LLM to contextually bind a chat session to a specific case.
- **Input Schema:** `{"action": "string (create|read|delete)", "client_id": "string (optional)", "details": "string (optional)"}`
- **LLM Invocation Example:**
  ```json
  {"name": "client_crm", "arguments": {"action": "create", "details": "Mr. Ali, charged under 302 PPC"}}
  ```

---

## 6. Real-time Optimisation
To provide a seamless, real-time experience on CPU hardware, the following optimizations were implemented:
- **WebSocket Token Streaming:** The LLM yields tokens via async generators over WebSockets, bringing Time-To-First-Token (TTFT) down.
- **Embedding Caching (LRU):** An LRU cache (utils/cache.py) memoizes frequent bge-m3 embedding queries, skipping the costly embedding inference step for repeated semantic searches.
- **Asynchronous I/O:** FastAPI and async SQLite/ChromaDB integrations ensure that the main thread is never blocked during disk reads or API calls.
- **Benchmarks (Typical CPU constraints):**
  - **Average Retrieval Time:** ~150-300ms 
  - **Tool Call Latency:** ~50ms
  - **End-to-End Response Time (TTFT):** ~800ms - 1.2s for a typical query.
  - **Generation Speed:** ~15 tokens/sec

---

## 7. Setup Instructions

### Pre-Requisites
- Docker & Docker Compose (or Python 3.11+)
- Ollama running locally

### Local & Docker Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/juris-ai.git
   cd juris-ai
   ```

2. **Configure Environment Variables:**
   Create a `.env` file in the root directory based on `.env.example`:
   ```bash
   LLM_MODEL=phi4-mini
   EMBEDDING_MODEL=bge-m3
   RAG_TOP_K=3
   ```

3. **Start Models (Ollama):**
   ```bash
   ollama serve
   ollama pull phi4-mini
   ollama pull bge-m3
   ```

4. **Build and Run via Docker:**
   ```bash
   docker-compose up --build -d
   ```

5. **(Alternative) Local Python Setup:**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Or venv\Scriptsctivate on Windows
   pip install -r requirements.txt
   
   # Prepare the Databases
   python scripts/populate_statutes.py
   python scripts/run_indexer.py
   
   # Start Server
   cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

---

## 8. Known Limitations
- **High Concurrency Constraints:** Running on CPU architecture limits the system to a few concurrent users. Under heavy parallel load, Ollama queuing significantly degrades token delivery times.
- **Long Document Truncation:** The phi4-mini model is configured with a 3072 token context window (LLM_NUM_CTX=3072). Extreme edge cases requiring the ingestion of lengthy, 100+ page comprehensive court rulings in a single prompt may force context truncation.
- **Multi-Hop Reasoning:** When queries require complex, deeply nested cross-referencing between three entirely separate laws (e.g., linking a specific ATA clause to a CrPC exemption while factoring in a QSO evidentiary rule), the quantized model may occasionally drop subtle procedural nuance.
