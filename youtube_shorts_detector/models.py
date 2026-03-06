"""
LMM 응답 파싱을 위한 Pydantic 모델 정의
"""
from pydantic import BaseModel, Field, validator, computed_field
from typing import Optional, Dict, Any
from enum import Enum

class ContentCategory(str, Enum):
    """콘텐츠 분류 카테고리"""
    C1 = "C1"  # 어그로/스팸
    C2 = "C2"  # 공장형 패턴
    C3 = "C3"  # 품질 불량
    C4 = "C4"  # 무단 도용
    C5 = "C5"  # 정상 영상

class AnalysisStatus(str, Enum):
    """분석 상태"""
    AUTO_APPROVE = "AUTO_APPROVE"
    HUMAN_REVIEW = "HUMAN_REVIEW" 
    AUTO_REJECT = "AUTO_REJECT"
    ANALYSIS_FAILED = "ANALYSIS_FAILED"

class AnalysisResult(BaseModel):
    """최종 분석 결과 (Pydantic 버그 완전 수정)"""
    c_category: ContentCategory
    reasoning_log: str
    confidence_score: float
    raw_response: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    
    @computed_field
    @property
    def status(self) -> AnalysisStatus:
        """신뢰도 기반 자동 상태 결정"""
        if self.error_message:
            return AnalysisStatus.ANALYSIS_FAILED
        elif self.confidence_score >= 0.8:
            return AnalysisStatus.AUTO_APPROVE
        elif self.confidence_score >= 0.5:
            return AnalysisStatus.HUMAN_REVIEW
        else:
            return AnalysisStatus.AUTO_REJECT

class LLMResponse(BaseModel):
    """LMM 원시 응답 스키마"""
    c_category: ContentCategory = Field(..., description="콘텐츠 분류 카테고리")
    reasoning_log: str = Field(..., min_length=10, description="판별 근거")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="신뢰도 점수")
    
    @validator('reasoning_log')
    def validate_reasoning(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("판별 근거는 최소 10자 이상이어야 합니다")
        return v.strip()

class VideoMetadata(BaseModel):
    """영상 메타데이터"""
    video_id: str
    title: str
    description: str
    duration: int  # 초
    view_count: int
    upload_date: str
    channel_name: str
    thumbnail_url: Optional[str] = None

class PreprocessingResult(BaseModel):
    """전처리 결과"""
    video_metadata: VideoMetadata
    keyframes: list[str]  # 이미지 경로 또는 base64
    ocr_text: str
    layout_score: float
    roi_data: Dict[str, Any]
    processing_log: list[str]

class MockVideoData(BaseModel):
    """테스트용 Mock 데이터"""
    video_id: str
    title: str
    description: str
    keyframes: list[str]
    ocr_text: str
    metadata: dict