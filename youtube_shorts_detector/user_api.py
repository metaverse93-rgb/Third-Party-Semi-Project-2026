"""
사용자용 FastAPI 서버
크롬 확장 프로그램 및 웹 서비스용 API
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import asyncio
import time
from datetime import datetime
import logging
import uuid

# 기존 파이프라인 import
from pipeline import YouTubeShortsAnalysisPipeline
from context_score_calculator import ContextScoreCalculator
from evaluation_metrics import EvaluationMetrics
from performance_logger import PerformanceLogger
from database_manager import db_manager
from database_models import UserFeedback, Contents, AnalysisResults

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성 (사용자용)
user_app = FastAPI(
    title="YouTube Shorts Detector API",
    description="유튜브 쇼츠 콘텐츠 분석 API",
    version="1.0.0"
)

# CORS 설정 (크롬 확장 프로그램 지원)
user_app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://*", "http://localhost:*", "https://*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 파이프라인 인스턴스
analysis_pipeline = YouTubeShortsAnalysisPipeline()
context_calculator = ContextScoreCalculator()
evaluator = EvaluationMetrics()
perf_logger = PerformanceLogger()

# 요청/응답 모델
class VideoAnalysisRequest(BaseModel):
    """영상 분석 요청"""
    video_url: str = Field(..., description="유튜브 쇼츠 URL")
    request_source: Optional[str] = Field("web", description="요청 출처 (chrome_extension, web)")
    user_agent: Optional[str] = Field(None, description="사용자 에이전트")
    
class VideoAnalysisResponse(BaseModel):
    """영상 분석 응답"""
    video_id: str
    analysis_result: Dict[str, Any]
    context_score: Dict[str, Any]
    processing_time: float
    confidence_level: str  # high, medium, low
    recommended_actions: List[str]
    report_url: str
    model_used: Optional[str] = None
    
class UserFeedbackRequest(BaseModel):
    """사용자 피드백 요청"""
    video_id: str
    action: str = Field(..., description="사용자 액션: like, dislike, report, block_channel")
    feedback_text: Optional[str] = None
    rating: Optional[int] = Field(None, ge=1, le=5)

@user_app.get("/")
def root():
    """API 루트"""
    return {
        "message": "YouTube Shorts Detector API",
        "version": "1.0.0",
        "endpoints": {
            "analyze": "/analyze (POST) - 영상 분석",
            "feedback": "/feedback (POST) - 피드백 제출", 
            "report": "/report/{video_id} (GET) - 상세 리포트",
            "health": "/health (GET) - 서비스 상태"
        },
        "chrome_extension_ready": True
    }

@user_app.post("/analyze", response_model=VideoAnalysisResponse)
async def analyze_video(
    request: VideoAnalysisRequest,
    background_tasks: BackgroundTasks,
    client_request: Request
):
    """
    영상 분석 API (크롬 확장 프로그램용)
    60초 타임아웃 보장
    """
    start_time = time.time()
    
    try:
        logger.info(f"🎬 영상 분석 요청: {request.video_url}")
        
        # 비디오 ID 추출
        video_id = _extract_video_id(request.video_url)
        
        # 기존 분석 결과 확인 (캐시)
        cached_result = await _get_cached_analysis(video_id)
        if cached_result:
            logger.info(f"📋 캐시된 결과 반환: {video_id}")
            return cached_result
        
        # 60초 타임아웃으로 분석 실행
        try:
            analysis_result = await asyncio.wait_for(
                _perform_analysis(request.video_url),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408,
                detail="분석 타임아웃 (60초 초과). 잠시 후 다시 시도해주세요."
            )
        
        processing_time = time.time() - start_time
        
        # 신뢰도 레벨 결정
        confidence_level = _determine_confidence_level(
            analysis_result["analysis_result"]["confidence_score"]
        )
        
        # 추천 액션 생성
        recommended_actions = _generate_user_actions(analysis_result)
        
        # 리포트 URL 생성
        report_url = f"/report/{video_id}"
        # Context Score만 추출
        context_data = {
            "context_score": analysis_result["technical_details"].get("context_score", 0.75),
            "s_semantic": analysis_result["technical_details"].get("s_semantic", 0.8),
            "o_existence": analysis_result["technical_details"].get("o_existence", 0.7),
            "a_sync": analysis_result["technical_details"].get("a_sync", 0.8),
            "layout_score": analysis_result["technical_details"].get("layout_score", 0.0)
}
        # 응답 구성
        response = VideoAnalysisResponse(
            video_id=video_id,
            analysis_result=analysis_result["analysis_result"],
            context_score=context_data,  # ← 수정됨!
            processing_time=processing_time,
            confidence_level=confidence_level,
            recommended_actions=recommended_actions,
            report_url=report_url,
            model_used=analysis_result.get("model_used", "gpt-4o-mini")
        )
        
        # 백그라운드에서 성능 로깅
        background_tasks.add_task(
            _log_analysis_performance,
            video_id,
            analysis_result,
            processing_time,
            str(client_request.client.host) if client_request.client else "unknown"
        )
        
        logger.info(f"✅ 분석 완료: {video_id} ({processing_time:.2f}초)")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 분석 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}")

@user_app.post("/feedback")
async def submit_feedback(
    feedback: UserFeedbackRequest,
    background_tasks: BackgroundTasks,
    client_request: Request
):
    """사용자 피드백 수집 API"""
    
    try:
        logger.info(f"📝 피드백 수신: {feedback.video_id} - {feedback.action}")
        
        # 피드백 ID 생성
        feedback_id = f"feedback_{uuid.uuid4().hex[:8]}"
        
        # 백그라운드에서 DB 저장
        background_tasks.add_task(
            _save_user_feedback,
            feedback_id,
            feedback,
            str(client_request.client.host) if client_request.client else "unknown",
            client_request.headers.get("user-agent", "unknown")
        )
        
        # 즉시 응답
        return {
            "status": "success",
            "feedback_id": feedback_id,
            "message": "피드백이 성공적으로 접수되었습니다.",
            "action_taken": feedback.action
        }
        
    except Exception as e:
        logger.error(f"❌ 피드백 저장 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="피드백 처리 중 오류가 발생했습니다.")

@user_app.get("/report/{video_id}")
async def get_analysis_report(video_id: str):
    """
    상세 분석 리포트 페이지
    Streamlit 대시보드로 리다이렉트
    """
    # Streamlit 대시보드 URL로 리다이렉트
    streamlit_url = f"http://localhost:8501/?video_id={video_id}"
    return RedirectResponse(url=streamlit_url)

@user_app.get("/health")
def health_check():
    """서비스 상태 확인"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "pipeline_status": "ready",
        "database_status": "connected" if db_manager else "disconnected"
    }

# ========== 헬퍼 함수들 ==========

def _extract_video_id(video_url: str) -> str:
    """YouTube URL에서 비디오 ID 추출"""
    import re
    
    # YouTube Shorts URL 패턴들
    patterns = [
        r'youtube\.com/shorts/([a-zA-Z0-9_-]+)',
        r'youtu\.be/([a-zA-Z0-9_-]+)',
        r'youtube\.com/watch\?v=([a-zA-Z0-9_-]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, video_url)
        if match:
            return match.group(1)
    
    # 패턴이 없으면 URL의 해시값 사용
    return f"video_{hash(video_url) % 100000}"

async def _get_cached_analysis(video_id: str) -> Optional[VideoAnalysisResponse]:
    """캐시된 분석 결과 조회"""
    
    try:
        with db_manager.get_db_session() as session:
            # 최근 24시간 내 분석 결과 찾기
            from datetime import timedelta
            cutoff = datetime.now() - timedelta(hours=24)
            
            analysis = session.query(AnalysisResults).filter(
                AnalysisResults.video_id == video_id,
                AnalysisResults.created_at >= cutoff
            ).order_by(AnalysisResults.created_at.desc()).first()
            
            if analysis and analysis.content:
                # 캐시된 결과로 응답 구성
                return VideoAnalysisResponse(
                    video_id=video_id,
                    analysis_result={
                        "category": analysis.c_category,
                        "confidence_score": analysis.confidence_score,
                        "reasoning_log": analysis.reasoning_log,
                        "status": analysis.status.value
                    },
                    context_score={
                        "context_score": analysis.context_score or 0.75,
                        "s_semantic": analysis.s_semantic or 0.8,
                        "o_existence": analysis.o_existence or 0.7,
                        "a_sync": analysis.a_sync or 0.8
                    },
                    processing_time=0.1,  # 캐시 조회 시간
                    confidence_level=_determine_confidence_level(analysis.confidence_score),
                    recommended_actions=_generate_user_actions_from_category(analysis.c_category),
                    report_url=f"/report/{video_id}"
                )
    except Exception as e:
        logger.warning(f"캐시 조회 실패: {e}")
    
    return None

async def _perform_analysis(video_url: str) -> Dict[str, Any]:
    """실제 영상 분석 수행"""
    
    # 기존 파이프라인 사용
    result = analysis_pipeline.analyze_video(video_url)
    
    if result.get("error"):
        raise Exception(result["error_message"])
    
    return result

def _determine_confidence_level(confidence_score: float) -> str:
    """신뢰도 레벨 결정"""
    if confidence_score >= 0.8:
        return "high"
    elif confidence_score >= 0.5:
        return "medium"
    else:
        return "low"

def _generate_user_actions(analysis_result: Dict) -> List[str]:
    """분석 결과 기반 사용자 액션 추천"""
    category = analysis_result["analysis_result"]["category"]
    return _generate_user_actions_from_category(category)

def _generate_user_actions_from_category(category: str) -> List[str]:
    """카테고리별 사용자 액션 추천"""
    
    action_map = {
        "C1": ["🚫 채널 추천 안 함", "📢 신고하기", "💬 의견 보내기"],
        "C2": ["⚠️ 품질 문제 신고", "🔇 채널 음소거", "💬 개선 의견"],
        "C3": ["🐛 품질 문제 신고", "🔄 재분석 요청", "💬 의견 보내기"],
        "C4": ["⚖️ 저작권 신고", "🚫 채널 차단", "📢 신고하기"],
        "C5": ["👍 좋아요", "🔔 구독 고려", "📤 공유하기"]
    }
    
    return action_map.get(category, ["💬 의견 보내기"])

async def _log_analysis_performance(
    video_id: str,
    analysis_result: Dict,
    processing_time: float,
    client_ip: str
):
    """분석 성능 로깅 (백그라운드)"""
    
    try:
        # Context Score 계산
        context_result = context_calculator.calculate_context_score(
            analysis_result.get("keyframes", []),
            analysis_result.get("ocr_text", ""),
            analysis_result.get("metadata", {})
        )
        
        # 성능 지표 계산 (Mock)
        performance_metrics = {
            "cer": evaluator.calculate_cer("mock_predicted", "mock_reference"),
            "rouge_l": evaluator.calculate_rouge_l("mock_predicted", "mock_reference"), 
            "bert_score": evaluator.calculate_bert_score("mock_predicted", "mock_reference"),
            "f1_weighted": 0.89
        }
        
        # 성능 로거에 기록
        perf_logger.log_single_result(
            video_id=video_id,
            analysis_result={
                "category": analysis_result["analysis_result"]["category"],
                "confidence_score": analysis_result["analysis_result"]["confidence_score"],
                "status": analysis_result["analysis_result"]["status"],
                "processing_time": processing_time
            },
            context_score=context_result,
            performance_metrics=performance_metrics
        )
        
        logger.info(f"📊 성능 로깅 완료: {video_id}")
        
    except Exception as e:
        logger.error(f"성능 로깅 실패: {e}")

async def _save_user_feedback(
    feedback_id: str,
    feedback: UserFeedbackRequest,
    client_ip: str,
    user_agent: str
):
    """사용자 피드백 DB 저장 (백그라운드)"""
    
    try:
        with db_manager.get_db_session() as session:
            feedback_record = UserFeedback(
                feedback_id=feedback_id,
                video_id=feedback.video_id,
                user_action=feedback.action,
                feedback_type="user_action",
                feedback_text=feedback.feedback_text,
                rating=feedback.rating,
                ip_address=client_ip,
                user_agent=user_agent,
                is_processed=False
            )
            
            session.add(feedback_record)
            session.commit()
            
        logger.info(f"💾 피드백 저장 완료: {feedback_id}")
        
    except Exception as e:
        logger.error(f"피드백 저장 실패: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(user_app, host="0.0.0.0", port=8000)