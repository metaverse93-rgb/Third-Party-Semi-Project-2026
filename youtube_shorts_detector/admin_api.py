"""
관리자용 CRUD API
FastAPI 기반 데이터 관리 엔드포인트
"""
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

from database_manager import get_db
from database_models import (
    Contents, AnalysisResults, ValidationLabels, PerformanceLogs,
    UserFeedback, DistillationBatches, ReviewStatus
)
from knowledge_distillation import KnowledgeDistillationPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
admin_app = FastAPI(
    title="YouTube Shorts Detector Admin API",
    description="관리자용 데이터 관리 API",
    version="1.0.0"
)

# 지식 증류 파이프라인 인스턴스
distillation_pipeline = KnowledgeDistillationPipeline()

@admin_app.get("/")
def admin_root():
    """관리자 API 루트"""
    return {
        "message": "YouTube Shorts Detector Admin API",
        "version": "1.0.0",
        "endpoints": [
            "/contents", "/analysis-results", "/validation-labels",
            "/performance-logs", "/user-feedback", "/distillation-batches"
        ]
    }

# ========== Contents CRUD ==========
@admin_app.get("/contents")
def get_contents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """콘텐츠 목록 조회"""
    
    query = db.query(Contents)
    
    if status:
        query = query.filter(Contents.status == status)
    
    contents = query.offset(skip).limit(limit).all()
    total = query.count()
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "contents": [
            {
                "id": c.id,
                "video_id": c.video_id,
                "title": c.title,
                "status": c.status.value,
                "created_at": c.created_at.isoformat()
            } for c in contents
        ]
    }

@admin_app.get("/contents/{video_id}")
def get_content_detail(video_id: str, db: Session = Depends(get_db)):
    """특정 콘텐츠 상세 조회"""
    
    content = db.query(Contents).filter(Contents.video_id == video_id).first()
    
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    
    return {
        "content": {
            "id": content.id,
            "video_id": content.video_id,
            "url": content.url,
            "title": content.title,
            "channel_name": content.channel_name,
            "duration": content.duration,
            "view_count": content.view_count,
            "raw_ocr_text": content.raw_ocr_text,
            "layout_score": content.layout_score,
            "status": content.status.value,
            "created_at": content.created_at.isoformat()
        },
        "analysis_results_count": len(content.analysis_results),
        "validation_labels_count": len(content.validation_labels)
    }

# ========== Analysis Results CRUD ==========
@admin_app.get("/analysis-results")
def get_analysis_results(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    model: Optional[str] = Query(None),
    category: Optional[str] = Query(None),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    db: Session = Depends(get_db)
):
    """분석 결과 목록 조회"""
    
    query = db.query(AnalysisResults)
    
    if model:
        query = query.filter(AnalysisResults.model_used == model)
    if category:
        query = query.filter(AnalysisResults.c_category == category)
    if min_confidence:
        query = query.filter(AnalysisResults.confidence_score >= min_confidence)
    
    results = query.order_by(desc(AnalysisResults.created_at)).offset(skip).limit(limit).all()
    total = query.count()
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "results": [
            {
                "result_id": r.result_id,
                "video_id": r.video_id,
                "c_category": r.c_category,
                "confidence_score": r.confidence_score,
                "model_used": r.model_used,
                "status": r.status.value,
                "created_at": r.created_at.isoformat()
            } for r in results
        ]
    }

@admin_app.get("/analysis-results/statistics")
def get_analysis_statistics(
    days: int = Query(7, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """분석 결과 통계"""
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    # 모델별 통계
    model_stats = db.query(
        AnalysisResults.model_used,
        func.count(AnalysisResults.id).label('count'),
        func.avg(AnalysisResults.confidence_score).label('avg_confidence')
    ).filter(
        AnalysisResults.created_at >= cutoff_date
    ).group_by(AnalysisResults.model_used).all()
    
    # 카테고리별 통계
    category_stats = db.query(
        AnalysisResults.c_category,
        func.count(AnalysisResults.id).label('count')
    ).filter(
        AnalysisResults.created_at >= cutoff_date
    ).group_by(AnalysisResults.c_category).all()
    
    return {
        "period": f"최근 {days}일",
        "model_statistics": [
            {
                "model": stat.model_used,
                "count": stat.count,
                "avg_confidence": round(stat.avg_confidence, 3)
            } for stat in model_stats
        ],
        "category_statistics": [
            {
                "category": stat.c_category,
                "count": stat.count
            } for stat in category_stats
        ]
    }

# ========== Validation Labels (HITL) ==========
@admin_app.get("/validation-labels")
def get_validation_labels(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None),
    hard_examples_only: bool = Query(False),
    db: Session = Depends(get_db)
):
    """검증 라벨 목록 조회 (HITL 워크플로우)"""
    
    query = db.query(ValidationLabels)
    
    if status:
        query = query.filter(ValidationLabels.review_status == status)
    if hard_examples_only:
        query = query.filter(ValidationLabels.is_hard_example == True)
    
    labels = query.order_by(desc(ValidationLabels.created_at)).offset(skip).limit(limit).all()
    total = query.count()
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "labels": [
            {
                "label_id": l.label_id,
                "video_id": l.video_id,
                "review_status": l.review_status.value,
                "is_hard_example": l.is_hard_example,
                "difficulty_score": l.difficulty_score,
                "ground_truth_category": l.ground_truth_category,
                "created_at": l.created_at.isoformat(),
                "reviewed_at": l.reviewed_at.isoformat() if l.reviewed_at else None
            } for l in labels
        ]
    }

@admin_app.put("/validation-labels/{label_id}/review")
def update_validation_label(
    label_id: str,
    ground_truth_category: str,
    review_comments: Optional[str] = None,
    reviewer_id: str = "admin",
    db: Session = Depends(get_db)
):
    """검증 라벨 검토 업데이트"""
    
    label = db.query(ValidationLabels).filter(ValidationLabels.label_id == label_id).first()
    
    if not label:
        raise HTTPException(status_code=404, detail="Validation label not found")
    
    # 검토 정보 업데이트
    label.ground_truth_category = ground_truth_category
    label.review_status = ReviewStatus.APPROVED
    label.review_comments = review_comments
    label.human_reviewer_id = reviewer_id
    label.reviewed_at = datetime.now()
    
    db.commit()
    
    return {
        "message": "Validation label updated successfully",
        "label_id": label_id,
        "ground_truth_category": ground_truth_category
    }

# ========== Distillation Batches ==========
@admin_app.get("/distillation-batches")
def get_distillation_batches(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """지식 증류 배치 목록 조회"""
    
    query = db.query(DistillationBatches)
    
    if status:
        query = query.filter(DistillationBatches.status == status)
    
    batches = query.order_by(desc(DistillationBatches.created_at)).offset(skip).limit(limit).all()
    total = query.count()
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "batches": [
            {
                "batch_id": b.batch_id,
                "source_model": b.source_model,
                "target_model": b.target_model,
                "total_samples": b.total_samples,
                "avg_confidence_score": b.avg_confidence_score,
                "data_quality_score": b.data_quality_score,
                "status": b.status,
                "created_at": b.created_at.isoformat(),
                "file_path": b.file_path
            } for b in batches
        ]
    }

@admin_app.post("/distillation-batches/create")
def create_distillation_batch(
    days: int = Query(30, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """새로운 지식 증류 배치 생성"""
    
    try:
        result = distillation_pipeline.auto_distillation_pipeline(days=days)
        return {
            "message": "Distillation batch creation initiated",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"배치 생성 실패: {str(e)}")

@admin_app.get("/distillation-batches/{batch_id}/download")
def download_training_dataset(batch_id: str, db: Session = Depends(get_db)):
    """학습 데이터셋 파일 다운로드"""
    
    batch = db.query(DistillationBatches).filter(DistillationBatches.batch_id == batch_id).first()
    
    if not batch or not batch.file_path:
        raise HTTPException(status_code=404, detail="Training dataset not found")
    
    return FileResponse(
        path=batch.file_path,
        filename=f"{batch_id}_training_data.jsonl",
        media_type='application/octet-stream'
    )

# ========== Performance Monitoring ==========
@admin_app.get("/performance-summary")
def get_performance_summary(
    days: int = Query(7, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """성능 요약 통계"""
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    # 기본 통계
    total_analyses = db.query(AnalysisResults).filter(
        AnalysisResults.created_at >= cutoff_date
    ).count()
    
    # 평균 성능 지표
    perf_stats = db.query(
        func.avg(PerformanceLogs.context_score).label('avg_context_score'),
        func.avg(PerformanceLogs.processing_time).label('avg_processing_time')
    ).filter(
        PerformanceLogs.timestamp >= cutoff_date
    ).first()
    
    # HITL 통계
    pending_reviews = db.query(ValidationLabels).filter(
        ValidationLabels.review_status == ReviewStatus.PENDING
    ).count()
    
    return {
        "period": f"최근 {days}일",
        "total_analyses": total_analyses,
        "avg_context_score": round(perf_stats.avg_context_score or 0, 3),
        "avg_processing_time": round(perf_stats.avg_processing_time or 0, 3),
        "pending_reviews": pending_reviews,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(admin_app, host="0.0.0.0", port=8001)