from pydantic import BaseModel
from enum import Enum
from typing import Optional


class AnalysisStatus(str, Enum):
    """Status of video analysis."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalyzeRequest(BaseModel):
    """Request model for video analysis (future use)."""
    pass


class AnalyzeResponse(BaseModel):
    """Response model for triggering analysis."""
    attempt_id: int
    status: AnalysisStatus
    message: str


class StatusResponse(BaseModel):
    """Response model for checking analysis status."""
    attempt_id: int
    status: AnalysisStatus
    progress: Optional[int] = None  # Percentage 0-100
    result: Optional[dict] = None  # Analysis results when completed


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
