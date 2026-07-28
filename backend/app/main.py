"""FastAPI Application Entrypoint for ResearchPilot AI."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.app.api.routes import router as api_router

app = FastAPI(
    title="ResearchPilot AI Engine",
    description="Agentic Multimodal Research Assistant API powered by LangGraph & Gemini",
    version="1.0.0",
)

# Configure CORS for Streamlit frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API V1 Router
app.include_router(api_router)


class HealthResponse(BaseModel):
    status: str
    version: str
    message: str


@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        message="ResearchPilot AI Backend is operational."
    )
