"""
크롬 확장 프로그램 전용 API
미니멀 오버레이용 경량화된 응답
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List
import asyncio
import logging

from user_api import analysis_pipeline, _extract_video_id, _perform_analysis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 크롬 확장 전용 FastAPI 앱
chrome_app = FastAPI(
    title="Chrome Extension API",
    description="크롬 확장 프로그램용 경량 API",
    version="1.0.0"
)

# CORS 설정 (크롬 확장만)
chrome_app.add_middleware(
    CORSMiddleware,
    allow_origins=["chrome-extension://*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# 응답 모델 (미니멀)
class ChromeAnalysisResponse(BaseModel):
    """크롬 확장용 미니멀 응답"""
    video_id: str
    category: str
    confidence: float
    status: str
    risk_level: str  # safe, warning, danger
    icon_color: str  # green, yellow, red
    overlay_text: str
    detail_url: str

@chrome_app.get("/")
def chrome_root():
    """크롬 확장 API 루트"""
    return {
        "message": "YouTube Shorts Detector Chrome Extension API",
        "version": "1.0.0",
        "extension_ready": True
    }

@chrome_app.post("/quick-analyze", response_model=ChromeAnalysisResponse)
async def quick_analyze(video_url: str):
    """
    빠른 분석 (크롬 확장용)
    30초 타임아웃, 미니멀 응답
    """
    
    try:
        logger.info(f"🔍 크롬 확장 빠른 분석: {video_url}")
        
        video_id = _extract_video_id(video_url)
        
        # 30초 타임아웃으로 빠른 분석
        try:
            analysis_result = await asyncio.wait_for(
                _perform_analysis(video_url),
                timeout=30.0
            )
        except asyncio.TimeoutError:
            # 타임아웃 시 기본 안전 응답
            return ChromeAnalysisResponse(
                video_id=video_id,
                category="C5",
                confidence=0.5,
                status="TIMEOUT",
                risk_level="warning",
                icon_color="yellow",
                overlay_text="분석 중...",
                detail_url=f"http://localhost:8501/?video_id={video_id}"
            )
        
        # 미니멀 응답 구성
        category = analysis_result["analysis_result"]["category"]
        confidence = analysis_result["analysis_result"]["confidence_score"]
        
        # 위험도 및 색상 결정
        risk_level, icon_color = _determine_risk_level(category, confidence)
        
        # 오버레이 텍스트 생성
        overlay_text = _generate_overlay_text(category, confidence)
        
        return ChromeAnalysisResponse(
            video_id=video_id,
            category=category,
            confidence=confidence,
            status=analysis_result["analysis_result"]["status"],
            risk_level=risk_level,
            icon_color=icon_color,
            overlay_text=overlay_text,
            detail_url=f"http://localhost:8501/?video_id={video_id}"
        )
        
    except Exception as e:
        logger.error(f"크롬 확장 분석 실패: {e}")
        raise HTTPException(status_code=500, detail="분석 실패")

@chrome_app.get("/overlay-config")
def get_overlay_config():
    """오버레이 설정 반환"""
    return {
        "position": "top-right",
        "size": "compact",
        "animation": "fade-in",
        "timeout": 5000,  # 5초 후 자동 숨김
        "colors": {
            "safe": "#28a745",
            "warning": "#ffc107", 
            "danger": "#dc3545"
        },
        "icons": {
            "safe": "✅",
            "warning": "⚠️",
            "danger": "🚫"
        }
    }

def _determine_risk_level(category: str, confidence: float) -> tuple:
    """위험도 및 아이콘 색상 결정"""
    
    if category in ["C1", "C2", "C4"]:  # 어그로, 공장형, 도용
        if confidence > 0.8:
            return "danger", "red"
        else:
            return "warning", "yellow"
    elif category == "C3":  # 품질 불량
        return "warning", "yellow"
    else:  # C5 정상
        return "safe", "green"

def _generate_overlay_text(category: str, confidence: float) -> str:
    """오버레이 텍스트 생성"""
    
    text_map = {
        "C1": f"어그로 의심 ({confidence:.0%})",
        "C2": f"공장형 콘텐츠 ({confidence:.0%})",
        "C3": f"품질 문제 ({confidence:.0%})",
        "C4": f"도용 의심 ({confidence:.0%})",
        "C5": f"정상 콘텐츠 ({confidence:.0%})"
    }
    
    return text_map.get(category, f"분석 완료 ({confidence:.0%})")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(chrome_app, host="0.0.0.0", port=8002)