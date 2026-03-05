"""
Phase 1 + Phase 2 통합 파이프라인
유튜브 URL → 최종 분류 결과
"""
import time
import logging
from typing import Dict, Any

from preprocessing import VideoPreprocessor
from content_analyzer import ContentAnalyzer
from models import AnalysisResult, PreprocessingResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeShortsAnalysisPipeline:
    """유튜브 쇼츠 분석 통합 파이프라인"""
    
    def __init__(self, analyzer_mode: str = "GPT4o"):
        """
        초기화
        Args:
            analyzer_mode: ContentAnalyzer 모드 ("GPT4o" 또는 "Qwen")
        """
        self.preprocessor = VideoPreprocessor()
        self.analyzer = ContentAnalyzer(mode=analyzer_mode)
        
        logger.info(f"파이프라인 초기화 완료 (모드: {analyzer_mode})")
    
    def analyze_video(self, video_url: str) -> Dict[str, Any]:
        """
        전체 분석 파이프라인 실행
        
        Args:
            video_url: 유튜브 영상 URL
            
        Returns:
            Dict: 전체 분석 결과
        """
        total_start_time = time.time()
        
        logger.info(f"🎬 영상 분석 시작: {video_url}")
        
        try:
            # Phase 1: 전처리
            logger.info("📊 Phase 1: 전처리 시작")
            preprocessing_result = self.preprocessor.process(video_url)
            
            logger.info(f"✅ Phase 1 완료: {preprocessing_result.video_metadata.title}")
            
            # Phase 2: LMM 추론
            logger.info("🤖 Phase 2: LMM 추론 시작")
            analysis_result = self.analyzer.analyze(
                frames=preprocessing_result.keyframes,
                ocr_text=preprocessing_result.ocr_text,
                layout_score=preprocessing_result.layout_score,
                metadata=preprocessing_result.video_metadata.model_dump()
            )
            
            logger.info(f"✅ Phase 2 완료: {analysis_result.c_category.value} ({analysis_result.status.value})")
            
            # 통합 결과 구성
            total_processing_time = time.time() - total_start_time
            
            result = {
                "video_info": {
                    "url": video_url,
                    "title": preprocessing_result.video_metadata.title,
                    "channel": preprocessing_result.video_metadata.channel_name,
                    "duration": preprocessing_result.video_metadata.duration,
                    "view_count": preprocessing_result.video_metadata.view_count
                },
                "analysis_result": {
                    "category": analysis_result.c_category.value,
                    "category_name": analysis_result.c_category.name,
                    "status": analysis_result.status.value,
                    "confidence_score": analysis_result.confidence_score,
                    "reasoning_log": analysis_result.reasoning_log
                },
                "technical_details": {
                    "ocr_text": preprocessing_result.ocr_text,
                    "layout_score": preprocessing_result.layout_score,
                    "keyframe_count": len(preprocessing_result.keyframes),
                    "preprocessing_time": sum([float(log.split(':')[-1].strip().replace('초', '')) 
                                             for log in preprocessing_result.processing_log 
                                             if '초' in log] or [0]),
                    "analysis_time": analysis_result.processing_time,
                    "total_time": total_processing_time
                },
                "logs": preprocessing_result.processing_log + [
                    f"LMM 분석 완료: {analysis_result.c_category.value}",
                    f"전체 처리 시간: {total_processing_time:.2f}초"
                ]
            }
            
            # 상태별 추천 액션
            result["recommended_actions"] = self._get_recommended_actions(analysis_result)
            
            return result
            
        except Exception as e:
            error_result = {
                "error": True,
                "error_message": str(e),
                "video_info": {"url": video_url},
                "total_time": time.time() - total_start_time
            }
            
            logger.error(f"❌ 파이프라인 실행 실패: {e}")
            return error_result
    
    def _get_recommended_actions(self, result: AnalysisResult) -> list[str]:
        """분석 결과에 따른 추천 액션"""
        actions = []
        
        if result.status.value == "AUTO_APPROVE":
            actions = ["✅ 정상 콘텐츠", "👍 좋아요 추천", "🔔 구독 고려"]
        elif result.status.value == "HUMAN_REVIEW":
            actions = ["⚠️ 추가 검토 필요", "👤 사람 판단 권장", "🔍 상세 분석 요청"]
        elif result.status.value == "AUTO_REJECT":
            actions = ["❌ 부적절 콘텐츠", "🚫 시청 비추천", "📢 신고 고려"]
        else:
            actions = ["🔧 분석 오류", "🔄 재시도 권장"]
        
        return actions
    
    def switch_analyzer_mode(self, new_mode: str):
        """분석기 모드 변경"""
        logger.info(f"분석기 모드 변경: {self.analyzer.mode} → {new_mode}")
        self.analyzer.switch_model(new_mode)