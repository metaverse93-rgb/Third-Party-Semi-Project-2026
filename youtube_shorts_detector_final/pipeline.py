"""
기획서 기반 YouTube Shorts 분석 파이프라인
Phase 1: 데이터 추출 → Phase 2: LLM 분석 → Phase 3: 알고리즘 점수 계산 및 분류
"""
import time
import uuid
import logging
from typing import Dict, Any
from datetime import datetime

from preprocessing import VideoPreprocessor
from content_analyzer import ContentAnalyzer
from database_manager import db_manager, DatabaseManager
from database_models import Contents, AnalysisResults, ContentStatus, AnalysisResultStatus
from models import VideoMetadata, PreprocessingResult, AnalysisResult
from config import MOCK_MODE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeShortsAnalysisPipeline:
    """기획서 기반 YouTube Shorts 분석 파이프라인"""

    def __init__(self, analyzer_mode: str = "GPT4o", mock_mode: bool = None):
        self.mock_mode = mock_mode if mock_mode is not None else MOCK_MODE
        self.analyzer_mode = analyzer_mode
        self.preprocessor = VideoPreprocessor()
        self.analyzer = ContentAnalyzer(mode=analyzer_mode)
        self.db_manager = DatabaseManager(mock_mode=True) if self.mock_mode else db_manager

        logger.info(f"🚀 파이프라인 초기화 (모드: {analyzer_mode}, Mock: {self.mock_mode})")

    def analyze_video(self, video_url: str) -> Dict[str, Any]:
        """
        영상 분석 메인 프로세스 — Streamlit /analyze 엔드포인트 연동
        Phase 1 → Phase 2 → Phase 3(DB 저장) 순으로 실행
        """
        total_start_time = time.time()

        try:
            # Phase 1: 데이터 추출 및 전처리
            logger.info(f"📥 Phase 1: 전처리 시작 ({video_url})")
            preprocessing_result = self.preprocessor.process(video_url)

            if preprocessing_result.video_metadata.video_id is None:
                raise ValueError("비디오 ID 추출 실패")

            logger.info(f"✅ Phase 1 완료: {preprocessing_result.video_metadata.title}")

            # Phase 2: 멀티모달 분석 (LLM + 알고리즘)
            logger.info("🤖 Phase 2: 멀티모달 분석 시작")
            analysis_result = self.analyzer.analyze(
                frames=preprocessing_result.keyframes,
                ocr_text=preprocessing_result.ocr_text,
                layout_score=preprocessing_result.layout_score,
                metadata=preprocessing_result.video_metadata.model_dump()
            )

            cis = analysis_result.precise_scores.cis_final if analysis_result.precise_scores else 0
            logger.info(f"✅ Phase 2 완료: {analysis_result.c_category.value} (CIS: {cis:.3f})")

            # Phase 3: DB 저장 (Mock 아닐 때만)
            if not self.mock_mode:
                logger.info("💾 Phase 3: DB 저장 시작")
                self._save_to_database(preprocessing_result, analysis_result)
                logger.info("✅ Phase 3 완료")

            total_time = time.time() - total_start_time
            result = self._build_result(preprocessing_result, analysis_result, video_url, total_time)
            logger.info(f"🎉 분석 완료: {result['analysis_result']['category']} ({total_time:.2f}초)")
            return result

        except Exception as e:
            total_time = time.time() - total_start_time
            logger.error(f"❌ 파이프라인 실패: {e}")
            return self._build_error_result(video_url, str(e), total_time)

    def _save_to_database(self, preprocessing_result: PreprocessingResult, analysis_result: AnalysisResult):
        """분석 결과 DB 저장"""
        try:
            meta = preprocessing_result.video_metadata

            with self.db_manager.get_db_session() as session:
                # Contents 저장 (없으면 추가, 있으면 업데이트)
                existing = session.query(Contents).filter_by(video_id=meta.video_id).first()
                if not existing:
                    session.add(Contents(
                        video_id=meta.video_id,
                        url=meta.thumbnail_url or f"https://youtube.com/watch?v={meta.video_id}",
                        title=meta.title,
                        description=meta.description,
                        channel_name=meta.channel_name,
                        duration=meta.duration,
                        view_count=meta.view_count,
                        upload_date=meta.upload_date,
                        raw_ocr_text=preprocessing_result.ocr_text,
                        layout_score=preprocessing_result.layout_score,
                        keyframes=preprocessing_result.keyframes,
                        roi_data=preprocessing_result.roi_data,
                        status=ContentStatus.COMPLETED
                    ))
                    session.flush()
                else:
                    existing.raw_ocr_text = preprocessing_result.ocr_text
                    existing.layout_score = preprocessing_result.layout_score
                    existing.status = ContentStatus.COMPLETED
                    existing.updated_at = datetime.now()

                # AnalysisResults 저장
                status_map = {
                    "AUTO_APPROVE":    AnalysisResultStatus.AUTO_APPROVE,
                    "HUMAN_REVIEW":    AnalysisResultStatus.HUMAN_REVIEW,
                    "AUTO_REJECT":     AnalysisResultStatus.AUTO_REJECT,
                    "ANALYSIS_FAILED": AnalysisResultStatus.ANALYSIS_FAILED,
                }
                ps = analysis_result.precise_scores

                session.add(AnalysisResults(
                    result_id=f"result_{uuid.uuid4().hex[:12]}",
                    video_id=meta.video_id,
                    c_category=analysis_result.c_category.value,
                    reasoning_log=analysis_result.reasoning_log,
                    confidence_score=analysis_result.confidence_score,
                    status=status_map.get(analysis_result.status.value, AnalysisResultStatus.AUTO_APPROVE),
                    model_used=self.analyzer_mode,
                    processing_time=analysis_result.processing_time,
                    raw_response=analysis_result.raw_response,
                    context_score=ps.cis_final if ps else 0,
                    s_semantic=ps.c3_context_score if ps else 0,
                    o_existence=ps.c3_context_score if ps else 0,
                    a_sync=0.0,
                    performance_metrics={
                        "c1_spam_score":    ps.c1_spam_score if ps else 0,
                        "c2_pattern_score": ps.c2_pattern_score if ps else 0,
                        "c3_context_score": ps.c3_context_score if ps else 0,
                        "cis_final":        ps.cis_final if ps else 0
                    }
                ))
                session.commit()

            logger.info(f"💾 DB 저장 완료: {meta.video_id} → {analysis_result.c_category.value}")

        except Exception as e:
            logger.error(f"⚠️ DB 저장 실패 (분석 결과는 정상): {e}")

    def _build_result(self,
                      preprocessing_result: PreprocessingResult,
                      analysis_result: AnalysisResult,
                      video_url: str,
                      total_time: float) -> Dict[str, Any]:
        """통합 분석 결과 구성 — Streamlit 리포트 화면에서 사용"""
        meta = preprocessing_result.video_metadata
        ps   = analysis_result.precise_scores

        return {
            "success": True,
            "video_info": {
                "url":         video_url,
                "video_id":    meta.video_id,
                "title":       meta.title,
                "channel":     meta.channel_name,
                "duration":    meta.duration,
                "view_count":  meta.view_count,
                "upload_date": meta.upload_date
            },
            "analysis_result": {
                "category":         analysis_result.c_category.value,
                "category_name":    analysis_result.c_category.name,
                "status":           analysis_result.status.value,
                "confidence_score": analysis_result.confidence_score,
                "reasoning_log":    analysis_result.reasoning_log
            },
            "technical_details": {
                "ocr_text":          preprocessing_result.ocr_text,
                "layout_score":      preprocessing_result.layout_score,
                "keyframe_count":    len(preprocessing_result.keyframes),
                "analysis_time":     analysis_result.processing_time,
                "total_time":        total_time,
                "model_used":        self.analyzer_mode,
                # CIS 점수 (Streamlit 리포트 게이지/레이더 차트 연동)
                "c1_spam_score":     ps.c1_spam_score    if ps else 0.0,
                "c2_pattern_score":  ps.c2_pattern_score if ps else 0.0,
                "c3_context_score":  ps.c3_context_score if ps else 0.5,
                "cis_final":         ps.cis_final        if ps else 0.0,
                "spam_detected":     analysis_result.raw_response == "spam_pattern_detected",
                "short_circuit_c4":  analysis_result.c_category.value in ["C1", "C2", "C3"]
            },
            "recommended_actions": self._generate_actions(analysis_result.c_category.value),
            "logs": preprocessing_result.processing_log + [
                f"LMM 분석 완료: {analysis_result.c_category.value}",
                f"CIS 최종 점수: {ps.cis_final if ps else 0:.3f}",
                f"전체 처리 시간: {total_time:.2f}초"
            ]
        }

    def _build_error_result(self, video_url: str, error_msg: str, total_time: float) -> Dict[str, Any]:
        """에러 결과 구성"""
        return {
            "success": False,
            "error": True,
            "error_message": error_msg,
            "video_info": {
                "url": video_url, "video_id": "unknown",
                "title": "분석 실패", "channel": "unknown",
                "duration": 0, "view_count": 0
            },
            "analysis_result": {
                "category": "C5", "category_name": "분석 실패 (안전 분류)",
                "status": "ANALYSIS_FAILED",
                "confidence_score": 0.1,
                "reasoning_log": f"분석 실패: {error_msg}"
            },
            "technical_details": {
                "ocr_text": "", "layout_score": 0.0, "keyframe_count": 0,
                "analysis_time": 0.0, "total_time": total_time,
                "model_used": self.analyzer_mode,
                "c1_spam_score": 0.0, "c2_pattern_score": 0.0,
                "c3_context_score": 0.5, "cis_final": 0.5,
                "spam_detected": False, "short_circuit_c4": False
            },
            "recommended_actions": ["🔄 재시도 권장", "🔧 시스템 관리자 문의"],
            "logs": [f"파이프라인 오류: {error_msg}"]
        }

    def _generate_actions(self, category: str) -> list:
        """카테고리별 추천 액션 — Streamlit 액션 버튼 연동"""
        return {
            "C1": ["⚠️ 어그로/스팸 콘텐츠", "🚫 시청 주의 권장", "📢 플랫폼 신고 고려", "🔍 채널 신뢰도 재검토"],
            "C2": ["🏭 공장형 패턴 감지", "⚠️ 자동 생성 가능성", "🔄 유사 영상 주의", "📊 채널 패턴 분석 권장"],
            "C3": ["📉 품질 불량 콘텐츠", "🎯 제목-내용 불일치 가능", "⚠️ 시청 주의", "📝 개선 피드백 고려"],
            "C5": ["✅ 정상 콘텐츠 판정", "👍 안전한 시청 가능", "📈 양질 콘텐츠 추천", "💡 관련 콘텐츠 탐색"],
        }.get(category, ["❓ 분류 결과를 확인해주세요"])

    def get_pipeline_status(self) -> Dict[str, Any]:
        """파이프라인 상태 — Streamlit 사이드바 시스템 현황 연동"""
        return {
            "analyzer_mode":      self.analyzer_mode,
            "mock_mode":          self.mock_mode,
            "database_connected": not self.mock_mode,
            "components_status": {
                "preprocessor": "ready",
                "analyzer":     "ready",
                "database":     "ready" if not self.mock_mode else "mock"
            }
        }

    def switch_analyzer_mode(self, new_mode: str):
        """분석기 모드 전환"""
        logger.info(f"🔄 모드 전환: {self.analyzer_mode} → {new_mode}")
        self.analyzer_mode = new_mode
        self.analyzer.switch_model(new_mode)


if __name__ == "__main__":
    pipeline = YouTubeShortsAnalysisPipeline(mock_mode=True)
    result = pipeline.analyze_video("https://youtube.com/shorts/test")
    print(result["analysis_result"]["category"])
