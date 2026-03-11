"""
사용자용 FastAPI 서버 - 기획서 기반
CIS 점수 체계 및 정밀 분석 결과 제공
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from typing import Optional, Dict, Any, List
import asyncio
import time
from datetime import datetime
import logging
import uuid

from pipeline import YouTubeShortsAnalysisPipeline
from models import (
    VideoAnalysisRequest, VideoAnalysisResponse,
    UserFeedbackRequest, FeedbackResponse,
    HealthCheckResponse,
    PreciseScores
)
from database_manager import db_manager
from database_models import UserFeedback
from config import MOCK_MODE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# FastAPI 앱
# ─────────────────────────────────────────
user_app = FastAPI(
    title="YouTube Shorts Detector API",
    description="기획서 CIS 점수 체계 기반 유튜브 쇼츠 콘텐츠 분석 API",
    version="2.0.0"
)

user_app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://*", "http://localhost:*", "https://*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analysis_pipeline = YouTubeShortsAnalysisPipeline(mock_mode=MOCK_MODE)

# ─────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────

@user_app.get("/")
def root():
    return {
        "message": "YouTube Shorts Detector API",
        "version": "2.0.0",
        "endpoints": {
            "analyze": "/analyze (POST) - 영상 분석",
            "feedback": "/feedback (POST) - 피드백 제출",
            "report": "/report/{video_id} (GET) - Streamlit 리포트",
            "health": "/health (GET) - 서비스 상태",
            "batch": "/batch-analyze (POST) - 배치 분석"
        },
        "mock_mode": MOCK_MODE
    }


@user_app.get("/health", response_model=HealthCheckResponse)
def health_check():
    """서비스 상태 확인 (Streamlit 사이드바 연동)"""
    try:
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
    영상 분석 API — Streamlit 대시보드 연동
    CIS 점수 체계 및 정밀 분석 결과 반환 (60초 타임아웃)
    """
    start_time = time.time()

    try:
        logger.info(f"🎬 분석 요청: {request.video_url}")
        video_id = _extract_video_id(request.video_url)

        # 캐시 확인 (MOCK_MODE 아닐 때만)
        if not MOCK_MODE:
            cached = await _get_cached_analysis(video_id)
            if cached:
                logger.info(f"📋 캐시 반환: {video_id}")
                return cached

        # 분석 실행 (60초 타임아웃)
        try:
            analysis_result = await asyncio.wait_for(
                _perform_analysis(request.video_url),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(status_code=408, detail="분석 타임아웃 (60초 초과)")

        if not analysis_result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"분석 실패: {analysis_result.get('error_message', '알 수 없는 오류')}"
            )

        processing_time = time.time() - start_time
        response = _build_api_response(analysis_result, video_id, processing_time)

        background_tasks.add_task(
            _log_analysis_request,
            request.video_url,
            analysis_result,
            str(client_request.client.host) if client_request.client else "unknown"
        )

        logger.info(f"✅ 완료: {video_id} → {analysis_result['analysis_result']['category']}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 분석 오류: {e}")
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {str(e)}")


@user_app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    feedback: UserFeedbackRequest,
    background_tasks: BackgroundTasks,
    client_request: Request
):
    """사용자 피드백 수집 — Streamlit 액션 버튼 연동"""
    try:
        feedback_id = f"feedback_{uuid.uuid4().hex[:8]}"

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
            message="피드백이 접수되었습니다.",
            timestamp=datetime.now().isoformat()
        )

    except Exception as e:
        logger.error(f"❌ 피드백 오류: {e}")
        raise HTTPException(status_code=500, detail="피드백 처리 중 오류가 발생했습니다.")


@user_app.get("/report/{video_id}")
async def get_analysis_report(video_id: str):
    """Streamlit 대시보드로 리다이렉트"""
    return RedirectResponse(url=f"http://localhost:8501/?video_id={video_id}")


@user_app.post("/batch-analyze")
async def batch_analyze(urls: List[str]):
    """배치 분석 (최대 5개)"""
    if len(urls) > 5:
        raise HTTPException(status_code=400, detail="최대 5개 URL까지 지원")

    results = []
    for url in urls:
        try:
            result = await _perform_analysis(url)
            results.append(result)
        except Exception as e:
            results.append({"success": False, "video_url": url, "error": str(e)})

    return {"success": True, "total": len(urls), "results": results}


# ─────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────

def _extract_video_id(video_url: str) -> str:
    import re
    for pattern in [
        r'youtube\.com/shorts/([a-zA-Z0-9_-]+)',
        r'youtu\.be/([a-zA-Z0-9_-]+)',
        r'youtube\.com/watch\?v=([a-zA-Z0-9_-]+)'
    ]:
        m = re.search(pattern, video_url)
        if m:
            return m.group(1)
    return f"video_{hash(video_url) % 100000}"


async def _perform_analysis(video_url: str) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, analysis_pipeline.analyze_video, video_url)


def _build_api_response(analysis_result: Dict, video_id: str, processing_time: float) -> VideoAnalysisResponse:
    analysis  = analysis_result["analysis_result"]
    tech      = analysis_result["technical_details"]

    precise_scores = PreciseScores(
        c1_spam_score    = tech.get("c1_spam_score", 0.0),
        c2_pattern_score = tech.get("c2_pattern_score", 0.0),
        c3_context_score = tech.get("c3_context_score", 0.5),
        cis_final        = tech.get("cis_final", 0.0)
    )

    return VideoAnalysisResponse(
        success=True,
        video_id=video_id,
        analysis_result={
            "category":          analysis["category"],
            "category_name":     analysis.get("category_name", analysis["category"]),
            "status":            analysis["status"],
            "confidence_score":  analysis["confidence_score"],
            "reasoning_log":     analysis.get("reasoning_log", ""),
            "reasoning_summary": _extract_reasoning_summary(analysis.get("reasoning_log", ""))
        },
        precise_scores=precise_scores,
        technical_details={
            "model_used":          tech.get("model_used", "gpt-4o-mini"),
            "processing_time":     tech.get("total_time", processing_time),
            "keyframe_count":      tech.get("keyframe_count", 0),
            "ocr_text_length":     len(tech.get("ocr_text", "")),
            # streamlit_dashboard_pre.py의 get_analysis_data()가 td에서 읽는 필드들
            "ocr_text":            tech.get("ocr_text", ""),
            "layout_score":        tech.get("layout_score", 0.0),
            "spam_detected":       tech.get("spam_detected", False),
            "short_circuit_c4":    tech.get("short_circuit_c4", False),
            "score_interpretation": precise_scores.score_breakdown
        },
        confidence_level=_determine_confidence_level(analysis["confidence_score"]),
        recommended_actions=_generate_actions(analysis["category"], precise_scores),
        processing_time=processing_time,
        report_url=f"/report/{video_id}"
    )


def _determine_confidence_level(score: float) -> str:
    if score >= 0.9:   return "very_high"
    elif score >= 0.8: return "high"
    elif score >= 0.6: return "medium"
    elif score >= 0.4: return "low"
    else:              return "very_low"


def _generate_actions(category: str, scores: PreciseScores) -> List[str]:
    base = {
        "C1": ["🚫 채널 추천 안 함", "📢 신고하기", "💬 의견 보내기"],
        "C2": ["⚠️ 채널 음소거", "📢 품질 신고", "💬 의견 보내기"],
        "C3": ["🐛 품질 문제 신고", "🔄 재분석 요청", "💬 의견 보내기"],
        "C4": ["⚖️ 저작권 신고", "🚫 채널 차단", "📢 신고하기"],
        "C5": ["👍 좋아요", "🔔 구독 고려", "📤 공유하기"]
    }.get(category, ["💬 의견 보내기"])

    extras = []
    if scores.cis_final < -0.5:
        extras.append(f"⚠️ CIS 점수 매우 낮음 ({scores.cis_final:.3f})")
    elif scores.cis_final >= 0.2:
        extras.append(f"✅ CIS 점수 양호 ({scores.cis_final:.3f})")

    return extras + base


def _extract_reasoning_summary(reasoning_log: str) -> str:
    for line in reasoning_log.split('\n'):
        if "CIS" in line and "점수" in line:
            return line.strip()
    for line in reasoning_log.split('\n'):
        if any(c in line for c in ["C1","C2","C3","C5"]) and "판정" in line:
            return line.strip()
    return reasoning_log.split('\n')[0] if reasoning_log else "분석 완료"


async def _get_cached_analysis(video_id: str) -> Optional[VideoAnalysisResponse]:
    try:
        with db_manager.get_db_session() as session:
            from database_models import AnalysisResults
            result = session.query(AnalysisResults).filter_by(
                video_id=video_id
            ).order_by(AnalysisResults.created_at.desc()).first()

            if result and result.created_at:
                if (datetime.now() - result.created_at).total_seconds() < 86400:
                    pm = result.performance_metrics or {}
                    scores = PreciseScores(
                        c1_spam_score    = pm.get("c1_spam_score", 0),
                        c2_pattern_score = pm.get("c2_pattern_score", 0),
                        c3_context_score = pm.get("c3_context_score", 0.5),
                        cis_final        = pm.get("cis_final", 0)
                    )
                    return VideoAnalysisResponse(
                        success=True,
                        video_id=video_id,
                        analysis_result={
                            "category": result.c_category,
                            "category_name": result.c_category,
                            "status": result.status.value,
                            "confidence_score": result.confidence_score,
                            "reasoning_log":     result.reasoning_log or "",
                            "reasoning_summary": "캐시된 분석 결과"
                        },
                        precise_scores=scores,
                        technical_details={
                            "model_used": result.model_used,
                            "processing_time": result.processing_time or 0,
                            "keyframe_count": len(result.content.keyframes or []) if result.content else 0,
                            "ocr_text_length": len(result.content.raw_ocr_text or "") if result.content else 0,
                            # streamlit_dashboard_pre.py 호환 필드
                            "ocr_text":         result.content.raw_ocr_text or "" if result.content else "",
                            "layout_score":     result.content.layout_score or 0.0 if result.content else 0.0,
                            "spam_detected":    False,
                            "short_circuit_c4": result.c_category in ["C1", "C2", "C3"],
                            "score_interpretation": scores.score_breakdown
                        },
                        confidence_level=_determine_confidence_level(result.confidence_score),
                        recommended_actions=["📋 캐시된 분석 결과"],
                        processing_time=0.1,
                        report_url=f"/report/{video_id}"
                    )
    except Exception as e:
        logger.warning(f"캐시 조회 실패: {e}")
    return None


async def _log_analysis_request(video_url: str, analysis_result: Dict, client_ip: str):
    try:
        logger.info(f"📊 로그: {video_url} → {analysis_result['analysis_result']['category']} (IP: {client_ip})")
    except Exception as e:
        logger.error(f"로깅 실패: {e}")


async def _save_user_feedback(
    feedback_id: str,
    feedback: UserFeedbackRequest,
    client_ip: str,
    user_agent: str
):
    if MOCK_MODE:
        logger.info(f"🎭 Mock 피드백: {feedback_id}")
        return

    try:
        with db_manager.get_db_session() as session:
            session.add(UserFeedback(
                feedback_id   = feedback_id,
                video_id      = feedback.video_id,
                user_action   = feedback.action,
                feedback_type = "user_action",
                feedback_text = feedback.feedback_text,
                rating        = feedback.rating,
                ip_address    = client_ip,
                user_agent    = user_agent,
                is_processed  = False,
                feedback_context={
                    "analysis_accuracy": getattr(feedback, "analysis_accuracy", None),
                    "timestamp": datetime.now().isoformat()
                }
            ))
            session.commit()
        logger.info(f"💾 피드백 저장: {feedback_id}")
    except Exception as e:
        logger.error(f"피드백 저장 실패: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(user_app, host="0.0.0.0", port=8000)
