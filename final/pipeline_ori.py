"""
기획서 기반 YouTube Shorts 분석 파이프라인
Phase 1: 데이터 추출 → Phase 2: LLM 분석 → Phase 3: 알고리즘 점수 계산 및 분류
"""
import time
import uuid
import logging
from typing import Dict, Any, Optional
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
        """
        파이프라인 초기화
        
        Args:
            analyzer_mode: "GPT4o" 또는 "Qwen"
            mock_mode: None이면 config.MOCK_MODE 사용
        """
        self.mock_mode = mock_mode if mock_mode is not None else MOCK_MODE
        self.analyzer_mode = analyzer_mode
        
        # 구성 요소 초기화
        self.preprocessor = VideoPreprocessor()
        self.analyzer = ContentAnalyzer(mode=analyzer_mode)
        
        if not self.mock_mode:
            self.db_manager = db_manager
        else:
            self.db_manager = DatabaseManager(mock_mode=True)
        
        logger.info(f"🚀 YouTube Shorts 분석 파이프라인 초기화")
        logger.info(f"   📊 분석 모드: {analyzer_mode}")
        logger.info(f"   🎭 Mock 모드: {self.mock_mode}")
    
    def switch_analyzer_mode(self, new_mode: str):
        """분석기 모드 전환"""
        logger.info(f"🔄 분석기 모드 전환: {self.analyzer_mode} → {new_mode}")
        self.analyzer_mode = new_mode
        self.analyzer.switch_model(new_mode)
    
    def analyze_video(self, video_url: str) -> Dict[str, Any]:
        """
        영상 분석 메인 프로세스
        
        Args:
            video_url: YouTube Shorts URL
            
        Returns:
            Dict: 통합 분석 결과
        """
        total_start_time = time.time()
        
        try:
            logger.info(f"🎬 영상 분석 시작: {video_url}")
            
            # ✅ Phase 1: 데이터 추출 및 전처리
            logger.info("📥 Phase 1: 데이터 추출 및 전처리 시작")
            preprocessing_result = self.preprocessor.process(video_url)
            
            if preprocessing_result.video_metadata.video_id is None:
                raise ValueError("비디오 ID 추출 실패")
            
            logger.info(f"✅ Phase 1 완료: {preprocessing_result.video_metadata.title}")
            
            # ✅ Phase 2: 멀티모달 분석 (LLM + 알고리즘)
            logger.info("🤖 Phase 2: 멀티모달 분석 시작")
            analysis_result = self.analyzer.analyze(
                frames=preprocessing_result.keyframes,
                ocr_text=preprocessing_result.ocr_text,
                layout_score=preprocessing_result.layout_score,
                metadata=preprocessing_result.video_metadata.model_dump()
            )
            logger.info(f"✅ Phase 2 완료: {analysis_result.c_category.value} (CIS: {analysis_result.precise_scores.cis_final if analysis_result.precise_scores else 0:.3f})")

            # ✅ Phase 3: 데이터베이스 저장
            if not self.mock_mode:
                logger.info("💾 Phase 3: 데이터베이스 저장 시작")
                self._save_to_database(preprocessing_result, analysis_result)
                logger.info("✅ Phase 3 완료: DB 저장 완료")

            # ✅ 통합 결과 구성
            total_processing_time = time.time() - total_start_time
            
            result = self._build_comprehensive_result(
                preprocessing_result, 
                analysis_result, 
                video_url,
                total_processing_time
            )

            logger.info(f"🎉 전체 분석 완료: {result['analysis_result']['category']} ({total_processing_time:.2f}초)")
            return result

        except Exception as e:
            total_processing_time = time.time() - total_start_time
            logger.error(f"❌ 파이프라인 실행 실패: {e}")
            return self._build_error_result(video_url, str(e), total_processing_time)

    def _save_to_database(self, preprocessing_result: PreprocessingResult, analysis_result: AnalysisResult):
        """데이터베이스 저장"""
        try:
            meta = preprocessing_result.video_metadata

            with self.db_manager.get_db_session() as session:
                # ✅ Contents 저장 (없으면 추가, 있으면 업데이트)
                existing_content = session.query(Contents).filter_by(video_id=meta.video_id).first()
                
                if not existing_content:
                    content = Contents(
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
                        keyframes=preprocessing_result.keyframes,  # JSON 형태로 저장
                        roi_data=preprocessing_result.roi_data,
                        status=ContentStatus.COMPLETED
                    )
                    session.add(content)
                    session.flush()
                else:
                    # 기존 콘텐츠 업데이트
                    existing_content.raw_ocr_text = preprocessing_result.ocr_text
                    existing_content.layout_score = preprocessing_result.layout_score
                    existing_content.status = ContentStatus.COMPLETED
                    existing_content.updated_at = datetime.now()

                # ✅ AnalysisResults 저장
                # 상태 매핑
                status_map = {
                    "AUTO_APPROVE": AnalysisResultStatus.AUTO_APPROVE,
                    "HUMAN_REVIEW": AnalysisResultStatus.HUMAN_REVIEW,
                    "AUTO_REJECT": AnalysisResultStatus.AUTO_REJECT,
                    "ANALYSIS_FAILED": AnalysisResultStatus.ANALYSIS_FAILED,
                }

                analysis_db_result = AnalysisResults(
                    result_id=f"result_{uuid.uuid4().hex[:12]}",
                    video_id=meta.video_id,
                    c_category=analysis_result.c_category.value,
                    reasoning_log=analysis_result.reasoning_log,
                    confidence_score=analysis_result.confidence_score,
                    status=status_map.get(analysis_result.status.value, AnalysisResultStatus.AUTO_APPROVE),
                    model_used=self.analyzer_mode,
                    processing_time=analysis_result.processing_time,
                    raw_response=analysis_result.raw_response,
                    # 🆕 기획서 기반 정밀 점수 저장
                    context_score=analysis_result.precise_scores.cis_final if analysis_result.precise_scores else 0,
                    s_semantic=analysis_result.precise_scores.c3_context_score if analysis_result.precise_scores else 0,
                    o_existence=analysis_result.precise_scores.c3_context_score if analysis_result.precise_scores else 0,
                    a_sync=0.0,  # STT 제거로 0
                    performance_metrics={
                        "c1_spam_score": analysis_result.precise_scores.c1_spam_score if analysis_result.precise_scores else 0,
                        "c2_pattern_score": analysis_result.precise_scores.c2_pattern_score if analysis_result.precise_scores else 0,
                        "c3_context_score": analysis_result.precise_scores.c3_context_score if analysis_result.precise_scores else 0,
                        "cis_final": analysis_result.precise_scores.cis_final if analysis_result.precise_scores else 0
                    }
                )
                session.add(analysis_db_result)
                session.commit()

            logger.info(f"💾 DB 저장 완료: {meta.video_id} → {analysis_result.c_category.value}")

        except Exception as e:
            logger.error(f"⚠️ DB 저장 실패 (분석 결과는 정상): {e}")

    def _build_comprehensive_result(self, 
                                  preprocessing_result: PreprocessingResult, 
                                  analysis_result: AnalysisResult,
                                  video_url: str,
                                  total_time: float) -> Dict[str, Any]:
        """포괄적인 결과 구성"""
        
        meta = preprocessing_result.video_metadata
        precise_scores = analysis_result.precise_scores
        
        # 추천 액션 생성
        recommended_actions = self._generate_recommended_actions(analysis_result.c_category.value)
        
        return {
            "success": True,
            "video_info": {
                "url": video_url,
                "video_id": meta.video_id,
                "title": meta.title,
                "channel": meta.channel_name,
                "duration": meta.duration,
                "view_count": meta.view_count,
                "upload_date": meta.upload_date
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
                "analysis_time": analysis_result.processing_time,
                "total_time": total_time,
                "model_used": self.analyzer_mode,
                # 🆕 기획서 기반 점수들
                "c1_spam_score": precise_scores.c1_spam_score if precise_scores else 0,
                "c2_pattern_score": precise_scores.c2_pattern_score if precise_scores else 0,
                "c3_context_score": precise_scores.c3_context_score if precise_scores else 0,
                "cis_final": precise_scores.cis_final if precise_scores else 0
            },
            "recommended_actions": recommended_actions,
            "logs": preprocessing_result.processing_log + [
                f"LMM 분석 완료: {analysis_result.c_category.value}",
                f"CIS 최종 점수: {precise_scores.cis_final if precise_scores else 0:.3f}",
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
                "url": video_url,
                "video_id": "unknown",
                "title": "분석 실패",
                "channel": "unknown",
                "duration": 0,
                "view_count": 0
            },
            "analysis_result": {
                "category": "C5",
                "category_name": "분석 실패 (안전 분류)",
                "status": "ANALYSIS_FAILED",
                "confidence_score": 0.1,
                "reasoning_log": f"분석 실패: {error_msg}"
            },
            "technical_details": {
                "ocr_text": "",
                "layout_score": 0.0,
                "keyframe_count": 0,
                "analysis_time": 0.0,
                "total_time": total_time,
                "model_used": self.analyzer_mode,
                "c1_spam_score": 0.0,
                "c2_pattern_score": 0.0,
                "c3_context_score": 0.5,
                "cis_final": 0.5
            },
            "recommended_actions": ["시스템 관리자에게 문의"],
            "logs": [f"파이프라인 오류: {error_msg}"]
        }

    def _generate_recommended_actions(self, category: str) -> list:
        """카테고리별 추천 액션 생성"""
        action_map = {
            "C1": [
                "⚠️ 어그로/스팸 콘텐츠로 판정됨",
                "🚫 시청 주의 권장", 
                "📢 필요시 플랫폼에 신고",
                "🔍 채널 신뢰도 재검토 필요"
            ],
            "C2": [
                "🏭 공장형 패턴 콘텐츠 감지",
                "⚠️ 자동 생성 콘텐츠 가능성",
                "🔄 유사한 패턴의 다른 영상 주의",
                "📊 채널 업로드 패턴 분석 권장"
            ],
            "C3": [
                "📉 품질 불량 콘텐츠",
                "🎯 제목-내용 불일치 또는 기술적 문제",
                "⚠️ 시청 시 주의 필요",
                "📝 개선 피드백 제출 고려"
            ],
            "C5": [
                "✅ 정상 콘텐츠로 판정",
                "👍 안전한 시청 가능",
                "📈 양질의 콘텐츠 추천",
                "💡 관련 콘텐츠 탐색 권장"
            ]
        }
        
        return action_map.get(category, ["❓ 분류 결과를 확인해주세요"])

    def get_pipeline_status(self) -> Dict[str, Any]:
        """파이프라인 상태 정보"""
        return {
            "analyzer_mode": self.analyzer_mode,
            "mock_mode": self.mock_mode,
            "database_connected": not self.mock_mode,
            "components_status": {
                "preprocessor": "ready",
                "analyzer": "ready", 
                "database": "ready" if not self.mock_mode else "mock"
            }
        }


# 테스트 함수
def test_pipeline():
    """파이프라인 테스트"""
    print("🧪 기획서 기반 YouTube Shorts 분석 파이프라인 테스트")
    print("=" * 70)
    
    # Mock 모드로 파이프라인 초기화
    pipeline = YouTubeShortsAnalysisPipeline(analyzer_mode="GPT4o", mock_mode=True)
    
    # 상태 확인
    status = pipeline.get_pipeline_status()
    print(f"\n📊 파이프라인 상태:")
    for key, value in status.items():
        print(f"  {key}: {value}")
    
    # 테스트 URL (실제 분석은 Mock 데이터로)
    test_url = "https://youtube.com/shorts/test_video_123"
    
    print(f"\n🎬 테스트 분석 시작: {test_url}")
    
    # 분석 실행
    result = pipeline.analyze_video(test_url)
    
    if result["success"]:
        print(f"\n✅ 분석 성공!")
        print(f"📋 영상 정보: {result['video_info']['title']}")
        print(f"🏷️ 분류: {result['analysis_result']['category']} ({result['analysis_result']['category_name']})")
        print(f"📈 신뢰도: {result['analysis_result']['confidence_score']:.3f}")
        print(f"⏱️ 처리 시간: {result['technical_details']['total_time']:.2f}초")
        
        print(f"\n📊 기획서 기반 점수:")
        tech = result['technical_details']
        print(f"  C1 (어그로/스팸): {tech['c1_spam_score']:.3f}")
        print(f"  C2 (공장형 패턴): {tech['c2_pattern_score']:.3f}")
        print(f"  C3 (맥락 품질): {tech['c3_context_score']:.3f}")
        print(f"  CIS 최종: {tech['cis_final']:.3f}")
        
        print(f"\n💡 추천 액션:")
        for action in result['recommended_actions']:
            print(f"  {action}")
    else:
        print(f"\n❌ 분석 실패: {result['error_message']}")
    
    print(f"\n🎉 파이프라인 테스트 완료!")

if __name__ == "__main__":
    test_pipeline()