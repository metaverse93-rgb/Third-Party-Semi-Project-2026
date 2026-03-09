"""
사용자용 FastAPI 서버 - 기획서 기반
CIS 점수 체계 및 정밀 분석 결과 제공
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import time
from datetime import datetime
import logging
import uuid

# 🆕 기획서 기반 모델들 import
from pipeline import YouTubeShortsAnalysisPipeline
from models import (
    VideoAnalysisRequest, VideoAnalysisResponse, 
    UserFeedbackRequest, FeedbackResponse,
    HealthCheckResponse, ErrorResponse,
    PreciseScores, create_mock_precise_scores
)
from database_manager import db_manager
from database_models import UserFeedback
from config import MOCK_MODE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성 (사용자용)
user_app = FastAPI(
    title="YouTube Shorts Detector API - 기획서 기반",
    description="기획서 CIS 점수 체계 기반 유튜브 쇼츠 콘텐츠 분석 API",
    version="2.0.0"
)

# CORS 설정
user_app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://*", "http://localhost:*", "https://*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 파이프라인 인스턴스
analysis_pipeline = YouTubeShortsAnalysisPipeline(mock_mode=MOCK_MODE)

@user_app.get("/")
def root():
    """API 루트"""
    return {
        "message": "YouTube Shorts Detector API - 기획서 CIS 점수 체계",
        "version": "2.0.0",
        "features": {
            "cis_scoring": "통합 맥락 점수 시스템",
            "precise_analysis": "C1, C2, C3 세부 점수",
            "real_time": "실시간 분석",
            "batch_support": "배치 처리 지원"
        },
        "endpoints": {
            "analyze": "/analyze (POST) - 영상 분석",
            "feedback": "/feedback (POST) - 피드백 제출", 
            "health": "/health (GET) - 서비스 상태",
            "report": "/report/{video_id} (GET) - 상세 리포트"
        },
        "chrome_extension_ready": True,
        "mock_mode": MOCK_MODE
    }

@user_app.get("/health", response_model=HealthCheckResponse)
def health_check():
    """서비스 상태 확인"""
    try:
        pipeline_status = analysis_pipeline.get_pipeline_status()
        
        return HealthCheckResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            pipeline_status="ready",
            database_connected=not MOCK_MODE,
            model_ready=True,
            uptime=time.time()
        )
    except Exception as e:
        logger.error(f"헬스체크 실패: {e}")
        raise HTTPException(status_code=503, detail="서비스 일시 불가")

@user_app.post("/analyze", response_model=VideoAnalysisResponse)
async def analyze_video(
    request: VideoAnalysisRequest,
    background_tasks: BackgroundTasks,
    client_request: Request
):
    """
    기획서 기반 영상 분석 API
    CIS 점수 체계 및 정밀 분석 결과 제공
    """
    start_time = time.time()
    
    try:
        logger.info(f"🎬 기획서 기반 분석 요청: {request.video_url}")
        
        # 비디오 ID 추출
        video_id = _extract_video_id(request.video_url)
        
        # 기존 분석 결과 확인 (캐시)
        if not MOCK_MODE:
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
        
        if not analysis_result["success"]:
            raise HTTPException(
                status_code=500, 
                detail=f"분석 실패: {analysis_result.get('error_message', '알 수 없는 오류')}"
            )
        
        processing_time = time.time() - start_time
        
        # 🆕 기획서 기반 응답 구성
        response = _build_api_response(analysis_result, video_id, processing_time)
        
        # 백그라운드 로깅
        background_tasks.add_task(
            _log_analysis_request,
            request.video_url,
            analysis_result,
            str(client_request.client.host) if client_request.client else "unknown"
        )
        
        logger.info(f"✅ 분석 완료: {video_id} → {analysis_result['analysis_result']['category']} (CIS: {analysis_result['technical_details'].get('cis_final', 0):.3f})")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 분석 API 오류: {e}")
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {str(e)}")

@user_app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: UserFeedbackRequest,
    background_tasks: BackgroundTasks,
    client_request: Request
):
    """사용자 피드백 수집 API"""
    try:
        logger.info(f"📝 피드백 수신: {feedback.video_id} - {feedback.action}")
        
        feedback_id = f"feedback_{uuid.uuid4().hex[:8]}"
        
        # 백그라운드에서 DB 저장
        background_tasks.add_task(
            _save_user_feedback,
            feedback_id,
            feedback,
            str(client_request.client.host) if client_request.client else "unknown",
            client_request.headers.get("user-agent", "unknown")
        )
        
        return FeedbackResponse(
            feedback_id=feedback_id,
            status="success",
            message="피드백이 성공적으로 접수되었습니다.",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ 피드백 저장 실패: {str(e)}")
        raise HTTPException(status_code=500, detail="피드백 처리 중 오류가 발생했습니다.")

@user_app.get("/report/{video_id}")
async def get_analysis_report(video_id: str):
    """상세 분석 리포트 페이지 (Streamlit 대시보드로 리다이렉트)"""
    streamlit_url = f"http://localhost:8501/?video_id={video_id}"
    return RedirectResponse(url=streamlit_url)

# =============================================
# 🆕 헬퍼 함수들 (기획서 기반)
# =============================================

async def _perform_analysis(video_url: str) -> Dict[str, Any]:
    """비동기 분석 수행"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, analysis_pipeline.analyze_video, video_url)

def _build_api_response(analysis_result: Dict, video_id: str, processing_time: float) -> VideoAnalysisResponse:
    """🆕 기획서 기반 API 응답 구성"""
    
    # 기본 분석 결과
    analysis = analysis_result["analysis_result"]
    tech_details = analysis_result["technical_details"]
    
    # 🆕 정밀 점수 추출
    precise_scores = PreciseScores(
        c1_spam_score=tech_details.get("c1_spam_score", 0.0),
        c2_pattern_score=tech_details.get("c2_pattern_score", 0.0),
        c3_context_score=tech_details.get("c3_context_score", 0.5),
        cis_final=tech_details.get("cis_final", 0.0)
    )
    
    # 신뢰도 레벨 결정
    confidence_level = _determine_confidence_level(analysis["confidence_score"])
    
    # 🆕 기획서 기반 추천 액션
    recommended_actions = _generate_enhanced_actions(
        analysis["category"], 
        precise_scores,
        analysis_result["recommended_actions"]
    )
    
    return VideoAnalysisResponse(
        success=True,
        video_id=video_id,
        analysis_result={
            "category": analysis["category"],
            "category_name": analysis["category_name"],
            "status": analysis["status"],
            "confidence_score": analysis["confidence_score"],
            "reasoning_summary": _extract_reasoning_summary(analysis["reasoning_log"])
        },
        precise_scores=precise_scores,
        technical_details={
            "model_used": tech_details["model_used"],
            "processing_time": tech_details["total_time"],
            "keyframe_count": tech_details["keyframe_count"],
            "ocr_text_length": len(tech_details.get("ocr_text", "")),
            # 🆕 점수 해석
            "score_interpretation": precise_scores.score_breakdown
        },
        confidence_level=confidence_level,
        recommended_actions=recommended_actions,
        processing_time=processing_time,
        report_url=f"/report/{video_id}"
    )

def _determine_confidence_level(confidence_score: float) -> str:
    """신뢰도 레벨 결정"""
    if confidence_score >= 0.9:
        return "very_high"
    elif confidence_score >= 0.8:
        return "high" 
    elif confidence_score >= 0.6:
        return "medium"
    elif confidence_score >= 0.4:
        return "low"
    else:
        return "very_low"

def _generate_enhanced_actions(category: str, scores: PreciseScores, base_actions: List[str]) -> List[str]:
    """🆕 정밀 점수 기반 향상된 추천 액션"""
    
    enhanced_actions = []
    
    # CIS 점수 기반 기본 권장사항
    if scores.cis_final < -0.5:
        enhanced_actions.append(f"⚠️ CIS 점수 매우 낮음 ({scores.cis_final:.3f}) - 시청 주의 필요")
    elif scores.cis_final < 0:
        enhanced_actions.append(f"📊 CIS 점수 보통 ({scores.cis_final:.3f}) - 내용 검토 권장")
    else:
        enhanced_actions.append(f"✅ CIS 점수 양호 ({scores.cis_final:.3f}) - 안전한 콘텐츠")
    
    # 카테고리별 세부 액션
    if category == "C1" and scores.c1_spam_score > 1.5:
        enhanced_actions.append("🚨 고위험 어그로 콘텐츠 - 즉시 차단 권장")
    elif category == "C2" and scores.c2_pattern_score > 0.9:
        enhanced_actions.append("🏭 명백한 공장형 패턴 - 채널 전체 검토 권장")
    elif category == "C3" and scores.c3_context_score < 0.3:
        enhanced_actions.append("📉 심각한 품질 문제 - 개선 요청 고려")
    
    # 기존 액션 추가
    enhanced_actions.extend(base_actions[:3])  # 최대 3개만
    
    return enhanced_actions

def _extract_reasoning_summary(reasoning_log: str) -> str:
    """판단 근거에서 요약 추출"""
    lines = reasoning_log.split('\n')
    
    # CIS 점수가 포함된 라인 찾기
    for line in lines:
        if "CIS" in line and "점수" in line:
            return line.strip()
    
    # 분류 관련 라인 찾기
    for line in lines:
        if any(cat in line for cat in ["C1", "C2", "C3", "C5"]) and "판정" in line:
            return line.strip()
    
    # 기본 요약
    return reasoning_log.split('\n')[0] if reasoning_log else "분석 완료"

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
    
    # 패턴 매칭 실패시 Mock ID 생성
    return f"mock_video_{hash(video_url) % 100000}"

async def _get_cached_analysis(video_id: str) -> Optional[VideoAnalysisResponse]:
    """캐시된 분석 결과 조회"""
    if MOCK_MODE:
        return None
    
    try:
        with db_manager.get_db_session() as session:
            from database_models import AnalysisResults
            
            result = session.query(AnalysisResults).filter_by(
                video_id=video_id
            ).order_by(AnalysisResults.created_at.desc()).first()
            
            if result and result.created_at:
                # 24시간 이내 결과만 캐시로 사용
                time_diff = datetime.now() - result.created_at
                if time_diff.total_seconds() < 86400:  # 24시간
                    
                    # 캐시된 결과를 API 응답으로 변환
                    cached_scores = PreciseScores(
                        c1_spam_score=result.performance_metrics.get("c1_spam_score", 0) if result.performance_metrics else 0,
                        c2_pattern_score=result.performance_metrics.get("c2_pattern_score", 0) if result.performance_metrics else 0,
                        c3_context_score=result.performance_metrics.get("c3_context_score", 0.5) if result.performance_metrics else 0.5,
                        cis_final=result.performance_metrics.get("cis_final", 0) if result.performance_metrics else 0
                    )
                    
                    return VideoAnalysisResponse(
                        success=True,
                        video_id=video_id,
                        analysis_result={
                            "category": result.c_category,
                            "category_name": result.c_category,
                            "status": result.status.value,
                            "confidence_score": result.confidence_score,
                            "reasoning_summary": "캐시된 분석 결과"
                        },
                        precise_scores=cached_scores,
                        technical_details={
                            "model_used": result.model_used,
                            "processing_time": result.processing_time or 0,
                            "keyframe_count": 0,
                            "ocr_text_length": 0,
                            "score_interpretation": cached_scores.score_breakdown
                        },
                        confidence_level=_determine_confidence_level(result.confidence_score),
                        recommended_actions=["📋 캐시된 분석 결과"],
                        processing_time=0.1,
                        report_url=f"/report/{video_id}"
                    )
        
        return None
        
    except Exception as e:
        logger.error(f"캐시 조회 실패: {e}")
        return None

async def _log_analysis_request(video_url: str, analysis_result: Dict, client_ip: str):
    """분석 요청 로깅 (백그라운드)"""
    try:
        logger.info(f"📊 분석 로그: {video_url} → {analysis_result['analysis_result']['category']} (IP: {client_ip})")
        
        # 필요시 추가 로깅 로직 구현
        # - 사용량 통계
        # - 성능 모니터링
        # - 오류 추적
        
    except Exception as e:
        logger.error(f"분석 로깅 실패: {e}")

async def _save_user_feedback(
    feedback_id: str,
    feedback: UserFeedbackRequest,
    client_ip: str,
    user_agent: str
):
    """사용자 피드백 DB 저장 (백그라운드)"""
    
    if MOCK_MODE:
        logger.info(f"🎭 Mock 피드백 저장: {feedback_id}")
        return
    
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
                is_processed=False,
                # 🆕 분석 정확도 피드백
                feedback_context={
                    "analysis_accuracy": feedback.analysis_accuracy,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            session.add(feedback_record)
            session.commit()
            
        logger.info(f"💾 피드백 저장 완료: {feedback_id}")
        
    except Exception as e:
        logger.error(f"피드백 저장 실패: {e}")

# 배치 분석 엔드포인트 (선택사항)
@user_app.post("/batch-analyze")
async def batch_analyze(urls: List[str]):
    """배치 영상 분석 (최대 5개)"""
    if len(urls) > 5:
        raise HTTPException(status_code=400, detail="최대 5개 URL까지 지원")
    
    results = []
    for url in urls:
        try:
            result = await _perform_analysis(url)
            results.append(result)
        except Exception as e:
            results.append({
                "success": False,
                "video_url": url,
                "error": str(e)
            })
    
    return {
        "success": True,
        "total_videos": len(urls),
        "results": results,
        "processing_time": time.time()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(user_app, host="0.0.0.0", port=8000)