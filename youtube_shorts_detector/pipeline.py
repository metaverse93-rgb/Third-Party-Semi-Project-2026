"""
Phase 1 + Phase 2 통합 파이프라인
유튜브 URL → 최종 분류 결과 → DB 저장
"""
import time
import logging
import uuid
from typing import Dict, Any

from preprocessing import VideoPreprocessor
from content_analyzer import ContentAnalyzer
from models import AnalysisResult, PreprocessingResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeShortsAnalysisPipeline:
    """유튜브 쇼츠 분석 통합 파이프라인"""

    def __init__(self, analyzer_mode: str = "GPT4o"):
        self.preprocessor = VideoPreprocessor()
        self.analyzer = ContentAnalyzer(mode=analyzer_mode)
        logger.info(f"파이프라인 초기화 완료 (모드: {analyzer_mode})")

    def analyze_video(self, video_url: str) -> Dict[str, Any]:
        """전체 분석 파이프라인 실행"""
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

            total_processing_time = time.time() - total_start_time

            result = {
                "video_info": {
                    "url": video_url,
                    "title": preprocessing_result.video_metadata.title,
                    "channel": preprocessing_result.video_metadata.channel_name,
                    "duration": preprocessing_result.video_metadata.duration,
                    "view_count": preprocessing_result.video_metadata.view_count
                },
                "model_used": self.analyzer.mode,
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
                    "analysis_time": analysis_result.processing_time,
                    "total_time": total_processing_time,
                    "raw_response": analysis_result.raw_response,
                    "spam_detected": analysis_result.raw_response == "spam_pattern_detected",
                    "short_circuit_c4": analysis_result.c_category.value in ["C1", "C2", "C3"],
                },
                "logs": preprocessing_result.processing_log + [
                    f"LMM 분석 완료: {analysis_result.c_category.value}",
                    f"전체 처리 시간: {total_processing_time:.2f}초"
                ]
            }

            result["recommended_actions"] = self._get_recommended_actions(analysis_result)

            # ✅ DB 저장
            db_saved = self._save_to_db(
                video_url, preprocessing_result, analysis_result, total_processing_time
            )
            result["db_saved"] = db_saved if db_saved is not None else False

            return result

        except Exception as e:
            logger.error(f"❌ 파이프라인 실행 실패: {e}")
            return {
                "error": True,
                "error_message": str(e),
                "video_info": {"url": video_url},
                "total_time": time.time() - total_start_time
            }

    def _save_to_db(self, video_url, preprocessing_result, analysis_result, processing_time):
        """분석 결과 DB 저장"""
        try:
            from database_manager import db_manager
            from database_models import Contents, AnalysisResults, ContentStatus, AnalysisResultStatus

            meta = preprocessing_result.video_metadata

            with db_manager.get_db_session() as session:
                # ✅ Contents 저장 (없으면 추가, 있으면 스킵)
                existing = session.query(Contents).filter_by(video_id=meta.video_id).first()
                if not existing:
                    content = Contents(
                        video_id=meta.video_id,
                        url=video_url,
                        title=meta.title,
                        description=meta.description,
                        channel_name=meta.channel_name,
                        duration=meta.duration,
                        view_count=meta.view_count,
                        upload_date=meta.upload_date,
                        raw_ocr_text=preprocessing_result.ocr_text,
                        layout_score=preprocessing_result.layout_score,
                        status=ContentStatus.COMPLETED
                    )
                    session.add(content)
                    session.flush()

                # ✅ AnalysisResults 저장
                # 상태 매핑
                status_map = {
                    "AUTO_APPROVE": AnalysisResultStatus.AUTO_APPROVE,
                    "HUMAN_REVIEW": AnalysisResultStatus.HUMAN_REVIEW,
                    "AUTO_REJECT": AnalysisResultStatus.AUTO_REJECT,
                    "ANALYSIS_FAILED": AnalysisResultStatus.ANALYSIS_FAILED,
                }
                result = AnalysisResults(
                    result_id=f"result_{uuid.uuid4().hex[:12]}",
                    video_id=meta.video_id,
                    c_category=analysis_result.c_category.value,
                    reasoning_log=analysis_result.reasoning_log,
                    confidence_score=analysis_result.confidence_score,
                    status=status_map.get(analysis_result.status.value, AnalysisResultStatus.AUTO_APPROVE),
                    model_used="GPT4o",
                    processing_time=processing_time,
                    raw_response=analysis_result.raw_response
                )
                session.add(result)
                session.commit()

            logger.info(f"💾 DB 저장 완료: {meta.video_id} → {analysis_result.c_category.value}")
            return True

        except Exception as e:
            logger.error(f"⚠️ DB 저장 실패 (분석 결과는 정상): {e}")
            # 분석 결과 dict에 플래그 추가 (user_api.py에서 경고 응답 가능)
            return False

    def _get_recommended_actions(self, result: AnalysisResult) -> list[str]:
        """분석 결과에 따른 추천 액션"""
        if result.status.value == "AUTO_APPROVE":
            return ["✅ 정상 콘텐츠", "👍 좋아요 추천", "🔔 구독 고려"]
        elif result.status.value == "HUMAN_REVIEW":
            return ["⚠️ 추가 검토 필요", "👤 사람 판단 권장", "🔍 상세 분석 요청"]
        elif result.status.value == "AUTO_REJECT":
            return ["❌ 부적절 콘텐츠", "🚫 시청 비추천", "📢 신고 고려"]
        else:
            return ["🔧 분석 오류", "🔄 재시도 권장"]

    def switch_analyzer_mode(self, new_mode: str):
        """분석기 모드 변경"""
        logger.info(f"분석기 모드 변경: {self.analyzer.mode} → {new_mode}")
        self.analyzer.switch_model(new_mode)
