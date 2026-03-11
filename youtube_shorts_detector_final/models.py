"""
기획서 기반 응답 모델 정의
CIS 점수 체계 및 정밀 점수들 포함
"""
from pydantic import BaseModel, Field, validator, computed_field
from typing import Optional, Dict, Any, List
from enum import Enum

class ContentCategory(str, Enum):
    """콘텐츠 분류 카테고리"""
    C1 = "C1"  # 어그로/스팸
    C2 = "C2"  # 공장형 패턴
    C3 = "C3"  # 품질 불량
    C5 = "C5"  # 정상 영상

class AnalysisStatus(str, Enum):
    """분석 상태"""
    AUTO_APPROVE = "AUTO_APPROVE"
    HUMAN_REVIEW = "HUMAN_REVIEW" 
    AUTO_REJECT = "AUTO_REJECT"
    ANALYSIS_FAILED = "ANALYSIS_FAILED"

class PreciseScores(BaseModel):
    """기획서 기반 정밀 점수들"""
    c1_spam_score: float = Field(..., ge=0.0, description="C1 어그로/스팸 점수")
    c2_pattern_score: float = Field(..., ge=0.0, le=1.0, description="C2 공장형 패턴 점수") 
    c3_context_score: float = Field(..., ge=0.0, le=1.0, description="C3 맥락 품질 점수")
    cis_final: float = Field(..., description="CIS 최종 통합 점수")
    
    @computed_field
    @property
    def score_breakdown(self) -> Dict[str, str]:
        """점수 해석"""
        return {
            "c1_level": "높음" if self.c1_spam_score > 1.0 else "보통" if self.c1_spam_score > 0.5 else "낮음",
            "c2_level": "높음" if self.c2_pattern_score > 0.85 else "보통" if self.c2_pattern_score > 0.5 else "낮음", 
            "c3_level": "높음" if self.c3_context_score > 0.7 else "보통" if self.c3_context_score > 0.4 else "낮음",
            "cis_level": "우수" if self.cis_final > 0.3 else "보통" if self.cis_final > -0.2 else "불량"
        }

class AnalysisResult(BaseModel):
    """최종 분석 결과"""
    c_category: ContentCategory
    reasoning_log: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    raw_response: Optional[str] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    precise_scores: Optional[PreciseScores] = None  # 🆕 기획서 기반 정밀 점수
    
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
    
    @computed_field
    @property
    def category_description(self) -> str:
        """카테고리 설명"""
        descriptions = {
            "C1": "어그로성 또는 스팸성 콘텐츠",
            "C2": "공장형 패턴으로 대량 생산된 콘텐츠", 
            "C3": "품질이 불량하거나 맥락이 불일치하는 콘텐츠",
            "C5": "정상적이고 양질의 콘텐츠"
        }
        return descriptions.get(self.c_category.value, "알 수 없는 카테고리")

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
    keyframes: List[str]  # base64 이미지들
    ocr_text: str
    layout_score: float
    roi_data: Dict[str, Any]
    processing_log: List[str]

class LLMResponse(BaseModel):
    """LMM 분석 응답 (기획서 기반)"""
    visual_elements: List[str] = Field(..., description="시각적 요소들")
    frame_descriptions: List[str] = Field(..., description="프레임별 묘사")
    layout_consistency: str = Field(..., description="레이아웃 일관성 (높음/보통/낮음)")
    layout_analysis: str = Field(..., description="레이아웃 분석 상세")
    spam_indicators: List[str] = Field(..., description="스팸/어그로 지표들") 
    content_object_matching: Dict[str, Any] = Field(..., description="내용-객체 매칭")
    action_elements: List[str] = Field(..., description="액션/동작 요소들")
    quality_issues: List[str] = Field(..., description="품질 문제들")
    overall_analysis: str = Field(..., description="종합 분석")

# 🆕 API 응답 모델들
class VideoAnalysisResponse(BaseModel):
    """영상 분석 API 응답"""
    success: bool = Field(..., description="분석 성공 여부")
    video_id: str = Field(..., description="비디오 ID")
    
    # 기본 분석 결과
    analysis_result: Dict[str, Any] = Field(..., description="분석 결과")
    
    # 🆕 기획서 기반 점수들
    precise_scores: Optional[PreciseScores] = Field(None, description="정밀 점수들")
    
    # 기술적 세부사항
    technical_details: Dict[str, Any] = Field(..., description="기술적 세부사항")
    
    # 사용자 액션
    confidence_level: str = Field(..., description="신뢰도 수준")
    recommended_actions: List[str] = Field(..., description="추천 액션들")
    
    # 메타 정보
    processing_time: float = Field(..., description="처리 시간")
    report_url: str = Field(..., description="상세 리포트 URL")

class BatchAnalysisResponse(BaseModel):
    """배치 분석 응답"""
    total_videos: int
    successful_analyses: int
    failed_analyses: int
    results: List[VideoAnalysisResponse]
    processing_time: float
    
class ErrorResponse(BaseModel):
    """에러 응답"""
    success: bool = False
    error: bool = True
    error_message: str
    error_code: Optional[str] = None
    timestamp: str

class HealthCheckResponse(BaseModel):
    """헬스체크 응답"""
    status: str = Field(..., description="서비스 상태")
    timestamp: str
    pipeline_status: str
    database_connected: bool
    model_ready: bool
    uptime: float

# 🆕 대시보드용 모델들
class DashboardMetrics(BaseModel):
    """대시보드 메트릭"""
    total_analyses: int
    category_distribution: Dict[str, int]
    average_scores: Dict[str, float]
    processing_stats: Dict[str, float]
    quality_metrics: Dict[str, float]

class ScoreVisualization(BaseModel):
    """점수 시각화 데이터"""
    labels: List[str]
    values: List[float]
    colors: List[str]
    descriptions: List[str]

# Mock 데이터 모델 (테스트용)
class MockVideoData(BaseModel):
    """테스트용 Mock 데이터"""
    video_id: str
    title: str
    description: str
    keyframes: List[str]
    ocr_text: str
    metadata: Dict[str, Any]

# 피드백 모델들
class UserFeedbackRequest(BaseModel):
    """사용자 피드백 요청"""
    video_id: str
    action: str = Field(..., description="like, dislike, report, block_channel")
    feedback_text: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5)
    analysis_accuracy: Optional[bool] = Field(None, description="분석 결과 정확도 피드백")

class FeedbackResponse(BaseModel):
    """피드백 응답"""
    feedback_id: str
    status: str
    message: str
    timestamp: str

# 요청 모델들  
class VideoAnalysisRequest(BaseModel):
    """영상 분석 요청"""
    video_url: str = Field(..., description="YouTube URL")
    request_source: str = Field("web", description="요청 출처")
    priority: Optional[str] = Field("normal", description="처리 우선순위")
    include_details: bool = Field(True, description="상세 정보 포함 여부")

class BatchAnalysisRequest(BaseModel):
    """배치 분석 요청"""
    video_urls: List[str] = Field(..., min_items=1, max_items=10)
    request_source: str = Field("batch_api", description="요청 출처")
    callback_url: Optional[str] = Field(None, description="완료 시 콜백 URL")

# 🎯 사용성 향상을 위한 유틸리티 함수들
def create_mock_precise_scores(category: str) -> PreciseScores:
    """카테고리에 맞는 Mock 정밀 점수 생성"""
    if category == "C1":
        return PreciseScores(
            c1_spam_score=1.5,
            c2_pattern_score=0.3,
            c3_context_score=0.2,
            cis_final=-0.7
        )
    elif category == "C2":
        return PreciseScores(
            c1_spam_score=0.4,
            c2_pattern_score=0.92,
            c3_context_score=0.5,
            cis_final=-0.6
        )
    elif category == "C3":
        return PreciseScores(
            c1_spam_score=0.2,
            c2_pattern_score=0.4,
            c3_context_score=0.3,
            cis_final=-0.1
        )
    else:  # C5
        return PreciseScores(
            c1_spam_score=0.1,
            c2_pattern_score=0.3,
            c3_context_score=0.8,
            cis_final=0.5
        )

def format_score_for_display(score: float, score_type: str) -> str:
    """점수를 사용자 친화적으로 포맷"""
    if score_type == "cis":
        if score > 0.3:
            return f"{score:.3f} (우수)"
        elif score > -0.2:
            return f"{score:.3f} (보통)" 
        else:
            return f"{score:.3f} (불량)"
    elif score_type in ["c2", "c3"]:
        if score > 0.8:
            return f"{score:.3f} (높음)"
        elif score > 0.5:
            return f"{score:.3f} (보통)"
        else:
            return f"{score:.3f} (낮음)"
    else:  # c1
        if score > 1.0:
            return f"{score:.3f} (높음)"
        elif score > 0.5:
            return f"{score:.3f} (보통)"
        else:
            return f"{score:.3f} (낮음)"