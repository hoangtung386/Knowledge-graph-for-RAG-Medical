"""FastAPI endpoints for PJI Diagnosis System."""

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

from src.api.schemas import ProcessSnapshotRequest, ProcessSnapshotResponse
from src.core.config import API_HOST, API_PORT
from src.main import PJIDiagnosisSystem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models (simple endpoints)
# ---------------------------------------------------------------------------


class DiagnoseRequest(BaseModel):
    clinical_question: str = Field(
        ...,
        description="Clinical scenario or question about a patient",
        examples=["68-year-old male, 2 years post TKA, chronic knee pain and swelling"],
    )
    lab_values: dict[str, float | str] | None = Field(
        default=None,
        description="Lab marker values. Keys are marker IDs (e.g. serum_CRP, synovial_WBC)",
        examples=[{"serum_CRP": 15, "synovial_WBC": 4500, "synovial_alpha_defensin": "positive"}],
    )
    major_criteria: list[str] | None = Field(
        default=None,
        description="Major criteria present: 'sinus_tract', 'two_cultures'",
    )
    is_acute: bool = Field(
        default=False,
        description="True if infection is acute (<90 days post-op)",
    )
    conversation_history: list[dict[str, str]] | None = None


class DiagnoseResponse(BaseModel):
    answer: str
    icm_score: dict | None = None


class GuidelineCompareRequest(BaseModel):
    question: str = Field(
        ...,
        description="Question to compare across guidelines",
        examples=["What CRP threshold defines PJI?"],
    )
    conversation_history: list[dict[str, str]] | None = None


class GuidelineCompareResponse(BaseModel):
    answer: str


class LabInterpretRequest(BaseModel):
    lab_values: dict[str, float | str]
    is_acute: bool = False
    conversation_history: list[dict[str, str]] | None = None


class LabInterpretResponse(BaseModel):
    answer: str
    icm_score: dict


class IngestRequest(BaseModel):
    directory: str = Field(
        default="data/pdf",
        description="Path to directory containing documents to ingest (scans subdirectories)",
    )


class IngestResponse(BaseModel):
    documents_ingested: int


class HealthResponse(BaseModel):
    status: str
    initialized: bool


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------

_system = PJIDiagnosisSystem()


async def get_system() -> PJIDiagnosisSystem:
    """FastAPI dependency that provides the initialized PJIDiagnosisSystem."""
    return _system


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _system.initialize()
    logger.info("PJI Diagnosis System initialized")
    yield
    await _system.finalize()
    logger.info("PJI Diagnosis System finalized")


app = FastAPI(
    title="PJI Diagnosis API",
    description="RAG-based Clinical Decision Support System for Periprosthetic Joint Infection",
    version="0.3.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# PRIMARY ENDPOINT — Web team calls this
# ---------------------------------------------------------------------------


@app.post(
    "/process-snapshot",
    response_model=ProcessSnapshotResponse,
    summary="Process clinical snapshot → structured recommendations + citations",
    tags=["Web Integration"],
)
async def process_snapshot(
    req: ProcessSnapshotRequest,
    system: PJIDiagnosisSystem = Depends(get_system),
):
    """Main endpoint for web team integration.

    Receives full clinical snapshot and returns:
    - 4 recommendation categories (diagnostic, surgery, systemic AB, local AB)
    - Evidence citations from RAG knowledge base
    """
    try:
        return await system.process_snapshot(req.snapshot_data_json)
    except Exception as e:
        logger.exception("process_snapshot failed")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


# ---------------------------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health(system: PJIDiagnosisSystem = Depends(get_system)):
    return HealthResponse(status="ok", initialized=system._initialized)


@app.post("/ingest", response_model=IngestResponse, tags=["System"])
async def ingest_documents(
    req: IngestRequest,
    system: PJIDiagnosisSystem = Depends(get_system),
):
    """Ingest documents from a directory into the RAG system (recursive)."""
    if not os.path.isdir(req.directory):
        raise HTTPException(status_code=400, detail=f"Directory not found: {req.directory}")
    count = await system.ingest_documents(req.directory)
    return IngestResponse(documents_ingested=count)


@app.delete("/documents/{doc_id}", tags=["System"])
async def delete_document(
    doc_id: str,
    system: PJIDiagnosisSystem = Depends(get_system),
):
    """Delete a document and rebuild affected knowledge graph."""
    await system.delete_document(doc_id)
    return {"status": "deleted", "doc_id": doc_id}


@app.delete("/entities/{entity_name}", tags=["System"])
async def delete_entity(
    entity_name: str,
    system: PJIDiagnosisSystem = Depends(get_system),
):
    """Delete an entity and all its relations from the knowledge graph."""
    await system.delete_entity(entity_name)
    return {"status": "deleted", "entity_name": entity_name}


# ---------------------------------------------------------------------------
# Simple query endpoints (for testing / ad-hoc queries)
# ---------------------------------------------------------------------------


@app.post("/diagnose", response_model=DiagnoseResponse, tags=["Simple Query"])
async def diagnose(
    req: DiagnoseRequest,
    system: PJIDiagnosisSystem = Depends(get_system),
):
    """Simple PJI diagnosis with free-text question."""
    result = await system.diagnose(
        clinical_question=req.clinical_question,
        lab_values=req.lab_values,
        major_criteria=req.major_criteria,
        is_acute=req.is_acute,
        conversation_history=req.conversation_history,
    )
    return DiagnoseResponse(**result)


@app.post("/compare-guidelines", response_model=GuidelineCompareResponse, tags=["Simple Query"])
async def compare_guidelines(
    req: GuidelineCompareRequest,
    system: PJIDiagnosisSystem = Depends(get_system),
):
    """Compare how different PJI guidelines address a question."""
    result = await system.compare_guidelines(
        question=req.question,
        conversation_history=req.conversation_history,
    )
    return GuidelineCompareResponse(**result)


@app.post("/interpret-labs", response_model=LabInterpretResponse, tags=["Simple Query"])
async def interpret_labs(
    req: LabInterpretRequest,
    system: PJIDiagnosisSystem = Depends(get_system),
):
    """Interpret lab results in PJI context with ICM scoring + LLM analysis."""
    result = await system.interpret_labs(
        lab_values=req.lab_values,
        is_acute=req.is_acute,
        conversation_history=req.conversation_history,
    )
    return LabInterpretResponse(**result)


# ---------------------------------------------------------------------------
# Run with: uvicorn src.api.endpoints:app --reload
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.endpoints:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
