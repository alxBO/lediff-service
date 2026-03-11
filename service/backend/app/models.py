"""Pydantic models for API request/response schemas."""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    job_id: str
    filename: str
    width: int = 0
    height: int = 0
    file_size_bytes: int = 0
    format: str = ""
    histogram: Optional[Dict[str, List[int]]] = None
    dynamic_range_ev: float = 0.0
    mean_brightness: float = 0.0
    median_brightness: float = 0.0
    clipping_percent: float = 0.0
    # For .npy uploads (generation mode)
    npy_shape: Optional[List[int]] = None


class GenerateRequest(BaseModel):
    mode: Literal["itm", "generation"] = "itm"
    model_type: Literal["highlight", "shadow"] = "highlight"
    prompt: str = "A photograph with natural lighting"
    seed: int = 42
    num_inference_steps: int = Field(default=50, ge=10, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    tiling: bool = True


class ProgressEvent(BaseModel):
    stage: str
    progress: float
    message: str
    queue_position: int = 0


class HdrAnalysis(BaseModel):
    dynamic_range_ev: float
    peak_luminance: float
    mean_luminance: float
    luminance_percentiles: Dict[str, float]
    hdr_histogram: dict


class ResultResponse(BaseModel):
    job_id: str
    download_url: str
    analysis: HdrAnalysis
    processing_time_seconds: float


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
