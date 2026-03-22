# PJI-RAG: Clinical Decision Support for Periprosthetic Joint Infection

> Version 0.1.0 | MIT License

Hệ thống hỗ trợ chẩn đoán và điều trị PJI (Periprosthetic Joint Infection) dựa trên Knowledge Graph RAG, kết hợp:
- **LightRAG** — knowledge graph + vector retrieval từ tài liệu y tế
- **ICM 2018 scoring** — tính điểm tự động theo tiêu chuẩn quốc tế
- **Structured JSON output** — trả về phác đồ điều trị có cấu trúc cho web integration
- **HuggingFace / OpenAI** — hỗ trợ cả 2 LLM provider

---

## Hướng dẫn chạy từ đầu (Step-by-step)

### Bước 1: Cài đặt môi trường

```bash
cd Pji_rag

# Cài uv (nếu chưa có)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Tạo virtual environment + cài dependencies
uv venv --python 3.10 .venv
source .venv/bin/activate       # Linux/WSL
# .venv\Scripts\activate        # Windows PowerShell
# source .venv/Scripts/activate # Windows Git Bash

uv pip install -e ".[dev]"
```

### Bước 2: Cấu hình .env

```bash
cp .env.example .env
```

Mở `.env` và điền tokens:

```env
# --- LLM (bắt buộc, chọn 1 trong 2) ---

# Option A: HuggingFace (miễn phí)
# Lấy token tại https://huggingface.co/settings/tokens
LLM_PROVIDER=huggingface
HF_TOKEN=hf_your-token-here

# Option B: OpenAI (trả phí)
# LLM_PROVIDER=openai
# OPENAI_API_KEY=sk-your-key-here

# --- Jina AI: Embedding + Reranker (khuyến nghị) ---
# Lấy API key tại https://jina.ai/
# Nếu có → dùng Jina embedding (1024 dim) + reranker v3
# Nếu không → fallback về HuggingFace embedding (384 dim), không reranker
JINA_API_KEY=jina_your-api-key-here
JINA_EMBEDDING_MODEL=jina-embeddings-v5-text-small
JINA_EMBEDDING_DIM=1024
JINA_RERANKER_MODEL=jina-reranker-v3
```

### Bước 3: Chuẩn bị dữ liệu PDF

Đặt tài liệu y tế vào `data/pdf/` (hệ thống scan recursive tất cả subfolder):

```
data/pdf/
├── Các nghiên cứu hướng dẫn điều trị/
├── Các tài liệu bàn luận về điều trị thay khớp/
├── Hỗ trợ chẩn đoán/
├── Một số guideline hướng dẫn chẩn đoán và báo cáo chuyên sâu/
└── Sách và Guideline về điều trị nhiễm khuẩn khớp/
```

### Bước 4: Khởi động server

```bash
source .venv/bin/activate
uvicorn src.api.endpoints:app --reload
```

Kiểm tra server đã chạy:
```bash
curl http://localhost:8000/health
# → {"status":"ok","initialized":true}
```

Swagger UI: http://localhost:8000/docs

### Bước 5: Ingest tài liệu → Tạo Knowledge Graph + Vector Store

**Đây là bước quan trọng nhất.** Nếu chưa ingest thì hệ thống không có dữ liệu để retrieve.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory": "data/pdf"}'
```

Hoặc dùng Swagger UI: mở http://localhost:8000/docs → `POST /ingest` → Try it out.

**Quá trình ingest sẽ:**

```
36 PDFs
  → PyMuPDF extract text (giữ page numbers)
  → LightRAG chunk (1200 tokens, overlap 200)
  → LLM extract entities + relations (2 lần/chunk)
      Entities: organism, biomarker, antibiotic, procedure, guideline...
  → Build Knowledge Graph (NetworkX .graphml)
  → Embed vào Vector Store (NanoVectorDB .json)
  → Lưu file_paths cho citation
```

**Thời gian & chi phí ước tính:**

| Provider | Thời gian (~36 PDFs) | Chi phí |
|---|---|---|
| HuggingFace Free | 30-60 phút | Miễn phí (giới hạn credits/tháng) |
| HuggingFace PRO | 15-30 phút | $9/tháng |
| OpenAI gpt-4o-mini | 10-20 phút | ~$2-5 |

**Response khi hoàn thành:**
```json
{"documents_ingested": 36}
```

**Sau khi ingest, `rag_storage/` sẽ chứa:**
```
rag_storage/
├── graph_chunk_entity_relation.graphml   # Knowledge Graph
├── vdb_entities.json                     # Entity embeddings
├── vdb_relationships.json                # Relation embeddings
├── vdb_chunks.json                       # Chunk embeddings
├── kv_store_full_docs.json               # Original documents
├── kv_store_text_chunks.json             # Text chunks
├── kv_store_full_entities.json           # Extracted entities
├── kv_store_full_relations.json          # Extracted relations
├── kv_store_llm_response_cache.json      # LLM cache (tránh gọi lại)
└── kv_store_doc_status.json              # Indexing status
```

### Bước 6: Test query

**Simple query test:**
```bash
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "clinical_question": "68 tuổi, nam, 2 năm sau thay khớp gối, đau và sưng mạn tính. Đánh giá PJI.",
    "lab_values": {"serum_CRP": 15, "synovial_WBC": 4500, "synovial_alpha_defensin": "positive"},
    "is_acute": false
  }'
```

**Full snapshot test (endpoint chính cho web team):**

Dùng Swagger UI: http://localhost:8000/docs → `POST /process-snapshot` → copy JSON từ `docs/api_contract.json` (field `snapshot_data_json`).

---

## Xử lý lỗi thường gặp khi Ingest

### Lỗi 402 — HuggingFace hết credits
```
HF embedding error (402): ...
```
- Chờ đầu tháng reset credits, hoặc mua HF PRO ($9/tháng)
- Hoặc đổi sang OpenAI trong `.env`

### Lỗi 503 — Model đang loading
```
HF model loading, waiting 20s...
```
- Bình thường — hệ thống tự retry sau 20-30s
- HF Inference API cold-start model khi lâu không dùng

### Ingest bị timeout
- Ingest 36 PDFs mất 30-60 phút với HF Free — đừng tắt terminal
- Nếu bị timeout uvicorn, chạy ingest bằng CLI:
  ```bash
  source .venv/bin/activate
  python -m src.main
  ```

### Muốn ingest lại từ đầu
```bash
rm -rf rag_storage/*
# Restart server → ingest lại
```

---

## Pipeline xử lý `/process-snapshot`

Đây là endpoint chính cho web team. Flow xử lý:

```
Web team gửi snapshot_data_json (bệnh nhân, xét nghiệm, cấy khuẩn, hình ảnh)
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 0: Data Completeness Check (deterministic)         │
│   Kiểm tra thiếu dữ liệu gì → trả về cảnh báo         │
│   Không cần LLM, chạy ngay lập tức                      │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 1: Parse snapshot (snapshot_parser.py)              │
│   → extract_icm_lab_values()     # CRP, ESR, WBC → IDs │
│   → extract_major_criteria()     # sinus tract, 2 cấy   │
│   → determine_is_acute()         # ACUTE / CHRONIC       │
│   → build_clinical_context()     # ALL fields → text     │
│   → build_culture_summary()      # Vi khuẩn + KSĐ       │
│   → extract_patient_risk_factors() # Dị ứng, tiểu đường │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 2: ICM 2018 Scoring (pji_criteria.py)              │
│   Deterministic, không cần LLM                          │
│   Major criterion → PJI confirmed ngay                  │
│   Minor score ≥6 → Infected | 2-5 → Inconclusive       │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 3: RAG Retrieve (rag_engine.py) — chạy song song   │
│   4 queries cho 4 loại recommendation:                  │
│   • Diagnostic criteria context                         │
│   • Surgery strategy context                            │
│   • Systemic antibiotic context                         │
│   • Local antibiotic context                            │
│   Mode: "mix" = Knowledge Graph + Vector search         │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 4: LLM Generate (direct call, bypass retrieval)    │
│   Context đã có từ Step 3 → gọi LLM trực tiếp          │
│   4 structured JSON outputs:                            │
│   • DIAGNOSTIC_TEST      (ICM scoring + reasoning)      │
│   • SURGERY_PROCEDURE    (DAIR / revision strategy)     │
│   • SYSTEMIC_ANTIBIOTIC  (IV → oral phases + dosing)    │
│   • LOCAL_ANTIBIOTIC     (cement spacer + mixing)       │
│   + Citations linking evidence → recommendations        │
└─────────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ Step 5: Validate & Return                               │
│   Parse JSON → Pydantic models → ProcessSnapshotResponse│
│   {                                                     │
│     recommendation_items: {...},                        │
│     citations: [...],                                   │
│     data_completeness: {is_complete, missing_items}     │
│   }                                                     │
└─────────────────────────────────────────────────────────┘
```

---

## Kiến trúc dự án

```
Pji_rag/
├── .env.example                 # Template cấu hình (copy → .env)
├── .gitignore
├── LICENSE
├── pyproject.toml               # Dependencies + project metadata
├── README.md
│
├── docs/
│   └── api_contract.json        # API contract cho web team (input/output schema)
│
├── data/pdf/                    # Tài liệu y tế (gitignored)
├── rag_storage/                 # LightRAG auto-generated (gitignored)
│
└── src/
    ├── __init__.py
    ├── main.py                  # PJIDiagnosisSystem — pipeline orchestrator
    │
    ├── core/                    # Infrastructure layer
    │   ├── __init__.py
    │   ├── config.py            # Centralized config (thresholds, env vars, constants)
    │   ├── rag_engine.py        # LightRAG wrapper (HF/OpenAI, KG, retrieval)
    │   └── document_loader.py   # PDF extraction (PyMuPDF) + recursive scan
    │
    ├── medical/                 # Domain logic layer
    │   ├── __init__.py
    │   ├── pji_criteria.py      # ICM 2018 scoring (major + minor, acute/chronic)
    │   ├── completeness_checker.py  # Data completeness check (deterministic)
    │   ├── prompt_templates.py  # Prompt cho simple query endpoints
    │   ├── recommendation_prompts.py  # Prompt cho structured JSON output (web team)
    │   └── snapshot_parser.py   # Parse snapshot → lab values, context, cultures
    │
    └── api/                     # API layer
        ├── __init__.py
        ├── endpoints.py         # FastAPI REST endpoints + dependency injection
        └── schemas/             # Pydantic models (API contract)
            ├── __init__.py      # Re-exports tất cả models
            ├── input.py         # Input: snapshot_data_json (web → AI)
            └── output.py        # Output: recommendations + citations + completeness (AI → web)
```

---

## API Endpoints

### Web Integration (primary)
| Method | Path | Mô tả |
|---|---|---|
| `POST` | `/process-snapshot` | Nhận snapshot lâm sàng → trả 4 phác đồ + citations + data completeness |

### System
| Method | Path | Mô tả |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/ingest` | Ingest documents vào RAG (recursive) — **chạy 1 lần sau khi chuẩn bị PDF** |
| `DELETE` | `/documents/{doc_id}` | Xóa document, rebuild KG |
| `DELETE` | `/entities/{entity_name}` | Xóa entity khỏi KG |

### Simple Query (testing)
| Method | Path | Mô tả |
|---|---|---|
| `POST` | `/diagnose` | Chẩn đoán PJI (free-text question) |
| `POST` | `/compare-guidelines` | So sánh guidelines |
| `POST` | `/interpret-labs` | Giải thích kết quả xét nghiệm |

---

## ICM 2018 Scoring

### Major Criteria (bất kỳ 1 = PJI confirmed)
| Tiêu chí | ID |
|---|---|
| Sinus tract communicating with joint | `sinus_tract` |
| Two positive cultures (same organism) | `two_cultures` |

### Minor Criteria (tổng >= 6 = confirmed, 2-5 = inconclusive)
| Marker | ID | Chronic | Acute | Points |
|---|---|---|---|---|
| Serum CRP | `serum_CRP` | 10 mg/L | 100 mg/L | 2 |
| Serum D-Dimer | `serum_D_dimer` | 860 ng/mL | 860 ng/mL | 2 |
| Serum ESR | `serum_ESR` | 30 mm/hr | N/A | 1 |
| Synovial WBC | `synovial_WBC` | 3000 cells/uL | 10000 cells/uL | 3 |
| Synovial PMN% | `synovial_PMN` | 80% | 90% | 2 |
| Alpha-Defensin | `synovial_alpha_defensin` | positive | positive | 3 |
| Leukocyte Esterase | `synovial_LE` | ++ | ++ | 3 |
| Synovial CRP | `synovial_CRP` | 6.9 mg/L | 6.9 mg/L | 1 |
| Histology | `positive_histology` | 5 PMN/HPF | 5 PMN/HPF | 3 |
| Single culture | `single_culture` | positive | positive | 2 |

---

## Lưu ý quan trọng

### Embedding & Reranker (Jina AI)

Hệ thống ưu tiên dùng **Jina AI** cho embedding + reranker nếu có `JINA_API_KEY`:

| Thành phần | Model | Chi tiết |
|---|---|---|
| Embedding | `jina-embeddings-v5-text-small` | 1024 dim, 32K tokens, multilingual |
| Reranker | `jina-reranker-v3` | 131K context, 24 languages, tự động re-rank kết quả search |

Nếu không có `JINA_API_KEY`, fallback về HuggingFace embedding (`bge-small-en-v1.5`, 384 dim) và không dùng reranker.

### HuggingFace Embedding (fallback)

Chỉ dùng khi không có `JINA_API_KEY`. HF Inference API **không hỗ trợ** `/v1/embeddings` cho hầu hết models — code gọi trực tiếp endpoint native với `?task=feature-extraction`.

### Thay đổi Embedding Model

Khi đổi embedding model (bất kể Jina hay HF):
1. **Phải xóa toàn bộ `rag_storage/`** vì vector dimension thay đổi
2. **Phải ingest lại** tất cả documents
3. Không thể mix embeddings từ 2 models khác nhau

```bash
rm -rf rag_storage/*
# Restart server → POST /ingest lại
```

---

## Tech Stack

| Thành phần | Thư viện | Vai trò |
|---|---|---|
| Knowledge Graph + Retrieval | LightRAG | Chunking, entity extraction, KG, vector search, reranking |
| REST API | FastAPI | Endpoints cho web team |
| PDF Extraction | PyMuPDF (fitz) | Extract text từ PDF guidelines |
| LLM | HuggingFace Qwen 72B / OpenAI | Entity extraction + response generation |
| Embedding | Jina v5-text-small (1024 dim) | Vector embeddings cho similarity search |
| Reranker | Jina reranker-v3 | Re-rank retrieved results (131K context) |
| Data Validation | Pydantic v2 | Input/output JSON schema |
| Scoring | Custom Python | ICM 2018 deterministic scoring |
