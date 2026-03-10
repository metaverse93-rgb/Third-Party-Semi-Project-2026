"""
지식 증류 파이프라인
GPT-4o → Qwen2.5-VL 자동 학습 데이터 생성
"""
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import logging
from uuid import uuid4

from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from database_manager import db_manager
from database_models import (
    AnalysisResults, ValidationLabels, DistillationBatches,
    AnalysisResultStatus, ReviewStatus
)
from config import PERFORMANCE_TARGETS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeDistillationPipeline:
    """지식 증류 파이프라인 클래스"""
    
    def __init__(self):
        self.high_confidence_threshold = 0.9    # 자동 학습 데이터
        self.hard_example_min = 0.5             # Hard Example 최소값
        self.hard_example_max = 0.8             # Hard Example 최대값
        self.min_batch_size = 100               # 최소 배치 크기
        
        logger.info("KnowledgeDistillationPipeline 초기화")
    
    def auto_distillation_pipeline(self, days: int = 30) -> Dict[str, Any]:
        """
        자동 지식 증류 파이프라인 실행
        
        Args:
            days: 최근 N일간의 데이터 사용
            
        Returns:
            Dict: 처리 결과
        """
        logger.info(f"🔄 자동 지식 증류 파이프라인 시작 (최근 {days}일)")
        
        with db_manager.get_db_session() as session:
            # 1. 고신뢰도 데이터 수집 (자동 학습용)
            high_confidence_data = self._collect_high_confidence_data(session, days)
            
            # 2. Hard Example 수집 (사람 검토용)
            hard_examples = self._collect_hard_examples(session, days)
            
            # 3. 배치 생성
            if len(high_confidence_data) >= self.min_batch_size:
                batch_result = self._create_distillation_batch(
                    session, high_confidence_data, hard_examples
                )
                logger.info(f"✅ 배치 생성 완료: {batch_result['batch_id']}")
            else:
                logger.warning(f"⚠️ 충분한 데이터 없음: {len(high_confidence_data)}/{self.min_batch_size}")
                batch_result = {"status": "insufficient_data"}
            
            # 4. Hard Example을 검토 대기열에 추가
            hard_example_result = self._queue_hard_examples(session, hard_examples)
            
            return {
                "pipeline_status": "completed",
                "high_confidence_samples": len(high_confidence_data),
                "hard_examples": len(hard_examples),
                "batch_creation": batch_result,
                "hard_example_queuing": hard_example_result,
                "timestamp": datetime.now().isoformat()
            }
    
    def _collect_high_confidence_data(self, session: Session, days: int) -> List[Dict]:
        """고신뢰도 분석 결과 수집 (confidence >= 0.9)"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # GPT-4o 결과 중 고신뢰도이고 아직 학습에 사용되지 않은 데이터
        query = session.query(AnalysisResults).filter(
            and_(
                AnalysisResults.model_used == "GPT4o",
                AnalysisResults.confidence_score >= self.high_confidence_threshold,
                AnalysisResults.status == AnalysisResultStatus.AUTO_APPROVE,
                AnalysisResults.is_used_for_training == False,
                AnalysisResults.created_at >= cutoff_date
            )
        )
        
        high_confidence_results = query.all()
        
        # 학습 데이터 형태로 변환
        training_data = []
        for result in high_confidence_results:
            training_data.append({
                "result_id": result.result_id,
                "video_id": result.video_id,
                "input_text": result.content.raw_ocr_text,
                "target_category": result.c_category,
                "reasoning": result.reasoning_log,
                "confidence_score": result.confidence_score,
                "context_score": result.context_score,
                "performance_metrics": result.performance_metrics,
                "created_at": result.created_at.isoformat()
            })
        
        logger.info(f"📊 고신뢰도 데이터 수집: {len(training_data)}개")
        return training_data
    
    def _collect_hard_examples(self, session: Session, days: int) -> List[Dict]:
        """Hard Example 수집 (0.5 <= confidence < 0.8)"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        query = session.query(AnalysisResults).filter(
            and_(
                AnalysisResults.confidence_score >= self.hard_example_min,
                AnalysisResults.confidence_score < self.hard_example_max,
                AnalysisResults.status == AnalysisResultStatus.HUMAN_REVIEW,
                AnalysisResults.created_at >= cutoff_date
            )
        )
        
        hard_results = query.all()
        
        hard_examples = []
        for result in hard_results:
            hard_examples.append({
                "result_id": result.result_id,
                "video_id": result.video_id,
                "input_text": result.content.raw_ocr_text,
                "predicted_category": result.c_category,
                "confidence_score": result.confidence_score,
                "reasoning": result.reasoning_log,
                "difficulty_score": 1.0 - result.confidence_score,  # 신뢰도 역비례
                "requires_review": True
            })
        
        logger.info(f"🤔 Hard Example 수집: {len(hard_examples)}개")
        return hard_examples
    
    def _create_distillation_batch(self, 
                                  session: Session, 
                                  high_confidence_data: List[Dict],
                                  hard_examples: List[Dict]) -> Dict[str, Any]:
        """지식 증류 배치 생성"""
        
        batch_id = f"distillation_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 카테고리 분포 계산
        category_dist = {}
        for data in high_confidence_data:
            category = data["target_category"]
            category_dist[category] = category_dist.get(category, 0) + 1
        
        # 평균 신뢰도 계산
        avg_confidence = sum(d["confidence_score"] for d in high_confidence_data) / len(high_confidence_data)
        
        # 데이터 품질 점수 계산 (간단한 휴리스틱)
        quality_score = min(1.0, avg_confidence + (len(high_confidence_data) / 1000) * 0.1)
        
        # 배치 메타데이터 생성
        batch = DistillationBatches(
            batch_id=batch_id,
            source_model="GPT4o",
            target_model="Qwen2.5-VL",
            total_samples=len(high_confidence_data),
            high_confidence_samples=len(high_confidence_data),
            medium_confidence_samples=len(hard_examples),
            avg_confidence_score=avg_confidence,
            data_quality_score=quality_score,
            category_distribution=category_dist,
            status="preparing",
            created_by="auto_pipeline"
        )
        
        session.add(batch)
        session.flush()  # ID 생성을 위해
        
        # 학습 데이터 파일 생성
        dataset_path = self._generate_training_dataset(batch_id, high_confidence_data)
        batch.file_path = dataset_path
        batch.status = "ready"
        
        # 사용된 데이터 플래그 업데이트
        result_ids = [d["result_id"] for d in high_confidence_data]
        session.query(AnalysisResults).filter(
            AnalysisResults.result_id.in_(result_ids)
        ).update({"is_used_for_training": True}, synchronize_session=False)
        
        session.commit()
        
        return {
            "batch_id": batch_id,
            "total_samples": len(high_confidence_data),
            "dataset_path": dataset_path,
            "category_distribution": category_dist,
            "quality_score": quality_score,
            "status": "ready"
        }
    
    def _generate_training_dataset(self, batch_id: str, training_data: List[Dict]) -> str:
        """학습 데이터셋 파일 생성"""
        
        # 데이터셋 디렉토리 생성
        dataset_dir = "datasets/distillation"
        os.makedirs(dataset_dir, exist_ok=True)
        
        # JSON Lines 형태로 저장 (Hugging Face 형식)
        dataset_path = os.path.join(dataset_dir, f"{batch_id}.jsonl")
        
        with open(dataset_path, 'w', encoding='utf-8') as f:
            for data in training_data:
                # Instruction-following 형태로 변환
                training_sample = {
                    "instruction": f"다음 유튜브 쇼츠 콘텐츠를 C1~C5 카테고리로 분류하세요.",
                    "input": f"OCR 텍스트: {data['input_text']}",
                    "output": f"카테고리: {data['target_category']}\n근거: {data['reasoning']}",
                    "metadata": {
                        "video_id": data["video_id"],
                        "confidence_score": data["confidence_score"],
                        "context_score": data["context_score"]
                    }
                }
                f.write(json.dumps(training_sample, ensure_ascii=False) + '\n')
        
        logger.info(f"📄 학습 데이터셋 생성: {dataset_path}")
        return dataset_path
    
    def _queue_hard_examples(self, session: Session, hard_examples: List[Dict]) -> Dict[str, Any]:
        """Hard Example을 검토 대기열에 추가"""
        
        queued_count = 0
        
        for example in hard_examples:
            # 이미 검토 대기 중인지 확인
            existing = session.query(ValidationLabels).filter_by(
                video_id=example["video_id"],
                result_id=example["result_id"]
            ).first()
            
            if not existing:
                # 새로운 검토 라벨 생성
                label = ValidationLabels(
                    label_id=f"hard_example_{uuid4().hex[:8]}",
                    video_id=example["video_id"],
                    result_id=example["result_id"],
                    is_hard_example=True,
                    difficulty_score=example["difficulty_score"],
                    review_status=ReviewStatus.PENDING,
                    review_comments=f"Hard Example: 신뢰도 {example['confidence_score']:.2f}"
                )
                session.add(label)
                queued_count += 1
        
        session.commit()
        
        logger.info(f"📋 검토 대기열 추가: {queued_count}개")
        
        return {
            "queued_count": queued_count,
            "total_hard_examples": len(hard_examples),
            "status": "completed"
        }
    
    def get_monthly_distillation_summary(self, session: Session) -> Dict[str, Any]:
        """월별 지식 증류 현황 요약"""
        
        # 최근 30일간 배치 통계
        cutoff_date = datetime.now() - timedelta(days=30)
        
        batches = session.query(DistillationBatches).filter(
            DistillationBatches.created_at >= cutoff_date
        ).all()
        
        if not batches:
            return {"error": "최근 30일간 배치 없음"}
        
        # 통계 계산
        total_samples = sum(b.total_samples for b in batches)
        avg_quality = sum(b.data_quality_score for b in batches if b.data_quality_score) / len(batches)
        
        # 상태별 배치 수
        status_counts = {}
        for batch in batches:
            status_counts[batch.status] = status_counts.get(batch.status, 0) + 1
        
        return {
            "total_batches": len(batches),
            "total_training_samples": total_samples,
            "avg_data_quality": round(avg_quality, 3),
            "status_distribution": status_counts,
            "latest_batch": batches[-1].batch_id if batches else None
        }

def test_knowledge_distillation():
    """지식 증류 파이프라인 테스트"""
    print("🧪 지식 증류 파이프라인 테스트 시작")
    print("=" * 50)
    
    # Mock 데이터로 테스트
    pipeline = KnowledgeDistillationPipeline()
    
    # 실제로는 DB에 데이터가 있어야 하지만, Mock 환경에서는 시뮬레이션
    print("\n1️⃣ 자동 지식 증류 파이프라인 실행")
    
    try:
        result = pipeline.auto_distillation_pipeline(days=7)
        
        print(f"   파이프라인 상태: {result['pipeline_status']}")
        print(f"   고신뢰도 샘플: {result['high_confidence_samples']}개")
        print(f"   Hard Example: {result['hard_examples']}개")
        
        if result['batch_creation'].get('batch_id'):
            print(f"   생성된 배치: {result['batch_creation']['batch_id']}")
        else:
            print(f"   배치 생성 실패: {result['batch_creation'].get('status', 'unknown')}")
            
    except Exception as e:
        print(f"   ❌ 파이프라인 실행 실패: {e}")
    
    print("\n✅ 지식 증류 파이프라인 테스트 완료")

if __name__ == "__main__":
    test_knowledge_distillation()