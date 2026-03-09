"""
관리자용 CRUD API - 기획서 CIS 점수 체계 기반
FastAPI 기반 데이터 관리 및 모니터링 엔드포인트
"""
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import json

from database_manager import get_db
from database_models import (
    Contents, AnalysisResults, ValidationLabels, PerformanceLogs,
    UserFeedback, DistillationBatches, ReviewStatus, AnalysisResultStatus
)
from knowledge_distillation import KnowledgeDistillationPipeline
from models import PreciseScores, create_mock_precise_scores
from config import CONTENT_CATEGORIES, CIS_CLASSIFICATION_THRESHOLDS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
admin_app = FastAPI(
    title="YouTube Shorts Detector Admin API - 기획서 기반",
    description="기획서 CIS 점수 체계 기반 관리자용 데이터 관리 API",
    version="2.0.0"
)

# 지식 증류 파이프라인 인스턴스
distillation_pipeline = KnowledgeDistillationPipeline()

@admin_app.get("/")
def admin_root():
    """관리자 API 루트"""
    return {
        "message": "YouTube Shorts Detector Admin API - 기획서 CIS 점수 체계",
        "version": "2.0.0",
        "features": {
            "cis_monitoring": "CIS 점수 체계 모니터링",
            "precise_analytics": "C1, C2, C3 세부 분석",
            "advanced_hitl": "향상된 HITL 워크플로우",
            "knowledge_distillation": "지식 증류 관리"
        },
        "endpoints": {
            "performance": "/performance-summary (GET) - 성능 요약",
            "analytics": "/cis-analytics (GET) - CIS 분석",
            "contents": "/contents (GET) - 콘텐츠 관리",
            "results": "/analysis-results (GET) - 분석 결과",
            "hitl": "/validation-labels (GET) - HITL 워크플로우"
        }
    }

# =============================================
# 🆕 CIS 점수 체계 기반 성능 모니터링
# =============================================

@admin_app.get("/performance-summary")
def get_performance_summary(
    days: int = Query(7, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """🆕 기획서 기반 성능 요약 통계"""
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    try:
        # 기본 통계
        total_analyses = db.query(AnalysisResults).filter(
            AnalysisResults.created_at >= cutoff_date
        ).count()
        
        if total_analyses == 0:
            return {
                "message": "분석 데이터가 없습니다",
                "period": f"최근 {days}일",
                "total_analyses": 0
            }
        
        # 🆕 CIS 점수 통계
        cis_stats = db.query(
            func.avg(AnalysisResults.context_score).label('avg_cis'),
            func.min(AnalysisResults.context_score).label('min_cis'),
            func.max(AnalysisResults.context_score).label('max_cis'),
            func.stddev(AnalysisResults.context_score).label('std_cis')
        ).filter(
            AnalysisResults.created_at >= cutoff_date,
            AnalysisResults.context_score.isnot(None)
        ).first()
        
        # 🆕 카테고리별 CIS 분포
        category_cis_stats = db.query(
            AnalysisResults.c_category,
            func.avg(AnalysisResults.context_score).label('avg_cis'),
            func.count(AnalysisResults.id).label('count')
        ).filter(
            AnalysisResults.created_at >= cutoff_date,
            AnalysisResults.context_score.isnot(None)
        ).group_by(AnalysisResults.c_category).all()
        
        # 🆕 CIS 임계값 기반 분포
        cis_distribution = {}
        for threshold_name, threshold_value in CIS_CLASSIFICATION_THRESHOLDS.items():
            if threshold_name.endswith("_threshold"):
                continue
            
            if threshold_name == "positive":  # >= 0.3
                count = db.query(AnalysisResults).filter(
                    AnalysisResults.created_at >= cutoff_date,
                    AnalysisResults.context_score >= threshold_value
                ).count()
                cis_distribution["우수 (≥0.3)"] = count
            elif threshold_name == "neutral":  # -0.2 ~ 0.3
                count = db.query(AnalysisResults).filter(
                    AnalysisResults.created_at >= cutoff_date,
                    AnalysisResults.context_score >= -0.2,
                    AnalysisResults.context_score < 0.3
                ).count()
                cis_distribution["보통 (-0.2~0.3)"] = count
            elif threshold_name == "low_negative":  # -0.5 ~ -0.2
                count = db.query(AnalysisResults).filter(
                    AnalysisResults.created_at >= cutoff_date,
                    AnalysisResults.context_score >= -0.5,
                    AnalysisResults.context_score < -0.2
                ).count()
                cis_distribution["불량 (-0.5~-0.2)"] = count
            elif threshold_name == "high_negative":  # < -0.5
                count = db.query(AnalysisResults).filter(
                    AnalysisResults.created_at >= cutoff_date,
                    AnalysisResults.context_score < -0.5
                ).count()
                cis_distribution["매우 불량 (<-0.5)"] = count
        
        # 평균 처리 시간
        avg_processing_time = db.query(
            func.avg(AnalysisResults.processing_time)
        ).filter(
            AnalysisResults.created_at >= cutoff_date,
            AnalysisResults.processing_time.isnot(None)
        ).scalar() or 0
        
        # HITL 통계
        pending_reviews = db.query(ValidationLabels).filter(
            ValidationLabels.review_status == ReviewStatus.PENDING
        ).count()
        
        # 🆕 정밀 점수 통계 (performance_metrics에서 추출)
        precise_stats = _calculate_precise_scores_stats(db, cutoff_date)
        
        return {
            "period": f"최근 {days}일",
            "timestamp": datetime.now().isoformat(),
            
            # 기본 통계
            "basic_stats": {
                "total_analyses": total_analyses,
                "avg_processing_time": round(float(avg_processing_time), 3),
                "pending_reviews": pending_reviews
            },
            
            # 🆕 CIS 점수 통계
            "cis_statistics": {
                "average": round(float(cis_stats.avg_cis or 0), 3),
                "minimum": round(float(cis_stats.min_cis or 0), 3),
                "maximum": round(float(cis_stats.max_cis or 0), 3),
                "std_deviation": round(float(cis_stats.std_cis or 0), 3),
                "distribution": cis_distribution
            },
            
            # 🆕 카테고리별 CIS 평균
            "category_cis_averages": [
                {
                    "category": stat.c_category,
                    "category_name": CONTENT_CATEGORIES.get(stat.c_category, stat.c_category),
                    "avg_cis": round(float(stat.avg_cis), 3),
                    "count": stat.count
                } for stat in category_cis_stats
            ],
            
            # 🆕 정밀 점수 통계
            "precise_scores_stats": precise_stats,
            
            # 품질 지표
            "quality_indicators": {
                "high_confidence_rate": _calculate_high_confidence_rate(db, cutoff_date),
                "auto_approval_rate": _calculate_auto_approval_rate(db, cutoff_date),
                "cis_quality_score": _calculate_cis_quality_score(cis_stats)
            }
        }
        
    except Exception as e:
        logger.error(f"성능 요약 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"성능 요약 조회 실패: {str(e)}")

@admin_app.get("/cis-analytics")
def get_cis_analytics(
    days: int = Query(30, ge=1, le=365),
    category: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """🆕 CIS 점수 상세 분석"""
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    try:
        query = db.query(AnalysisResults).filter(
            AnalysisResults.created_at >= cutoff_date,
            AnalysisResults.context_score.isnot(None)
        )
        
        if category:
            query = query.filter(AnalysisResults.c_category == category)
        
        results = query.all()
        
        if not results:
            return {
                "message": "분석 데이터가 없습니다",
                "period": f"최근 {days}일",
                "category_filter": category
            }
        
        # CIS 점수 분포 히스토그램 데이터
        cis_scores = [float(r.context_score) for r in results if r.context_score is not None]
        
        # 시계열 데이터 (일별 평균 CIS)
        daily_cis = db.query(
            func.date(AnalysisResults.created_at).label('date'),
            func.avg(AnalysisResults.context_score).label('avg_cis'),
            func.count(AnalysisResults.id).label('count')
        ).filter(
            AnalysisResults.created_at >= cutoff_date,
            AnalysisResults.context_score.isnot(None)
        ).group_by(func.date(AnalysisResults.created_at)).order_by('date').all()
        
        # 🆕 정밀 점수 상관관계 분석
        correlation_data = []
        for result in results[:100]:  # 최근 100개만
            if result.performance_metrics and isinstance(result.performance_metrics, dict):
                correlation_data.append({
                    "cis_final": float(result.context_score or 0),
                    "c1_spam_score": result.performance_metrics.get('c1_spam_score', 0),
                    "c2_pattern_score": result.performance_metrics.get('c2_pattern_score', 0),
                    "c3_context_score": result.performance_metrics.get('c3_context_score', 0),
                    "category": result.c_category
                })
        
        return {
            "period": f"최근 {days}일",
            "category_filter": category,
            "total_samples": len(results),
            
            # CIS 분포
            "cis_distribution": {
                "scores": cis_scores,
                "bins": _create_cis_histogram_bins(cis_scores),
                "statistics": {
                    "mean": round(sum(cis_scores) / len(cis_scores), 3),
                    "median": round(sorted(cis_scores)[len(cis_scores)//2], 3),
                    "q25": round(sorted(cis_scores)[len(cis_scores)//4], 3),
                    "q75": round(sorted(cis_scores)[len(cis_scores)*3//4], 3)
                }
            },
            
            # 시계열 데이터
            "time_series": [
                {
                    "date": stat.date.isoformat(),
                    "avg_cis": round(float(stat.avg_cis), 3),
                    "count": stat.count
                } for stat in daily_cis
            ],
            
            # 상관관계 데이터
            "correlation_analysis": correlation_data,
            
            # 이상치 탐지
            "anomalies": _detect_cis_anomalies(cis_scores),
            
            # 품질 트렌드
            "quality_trends": _analyze_quality_trends(daily_cis)
        }
        
    except Exception as e:
        logger.error(f"CIS 분석 실패: {e}")
        raise HTTPException(status_code=500, detail=f"CIS 분석 실패: {str(e)}")

# =============================================
# 🆕 정밀 분석 결과 관리
# =============================================

@admin_app.get("/analysis-results/detailed")
def get_detailed_analysis_results(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    category: Optional[str] = Query(None),
    min_cis: Optional[float] = Query(None, ge=-2.0, le=2.0),
    max_cis: Optional[float] = Query(None, ge=-2.0, le=2.0),
    include_scores: bool = Query(True, description="정밀 점수 포함 여부"),
    db: Session = Depends(get_db)
):
    """🆕 상세 분석 결과 조회 (정밀 점수 포함)"""
    
    try:
        query = db.query(AnalysisResults)
        
        # 필터링
        if category:
            query = query.filter(AnalysisResults.c_category == category)
        if min_cis is not None:
            query = query.filter(AnalysisResults.context_score >= min_cis)
        if max_cis is not None:
            query = query.filter(AnalysisResults.context_score <= max_cis)
        
        total = query.count()
        results = query.order_by(desc(AnalysisResults.created_at)).offset(skip).limit(limit).all()
        
        detailed_results = []
        for result in results:
            result_data = {
                "result_id": result.result_id,
                "video_id": result.video_id,
                "category": result.c_category,
                "confidence_score": result.confidence_score,
                "status": result.status.value,
                "model_used": result.model_used,
                "processing_time": result.processing_time,
                "created_at": result.created_at.isoformat() if result.created_at else None,
                
                # 🆕 CIS 관련
                "cis_score": result.context_score,
                "s_semantic": result.s_semantic,
                "o_existence": result.o_existence,
                "a_sync": result.a_sync
            }
            
            # 🆕 정밀 점수 추가
            if include_scores and result.performance_metrics:
                if isinstance(result.performance_metrics, dict):
                    result_data["precise_scores"] = {
                        "c1_spam_score": result.performance_metrics.get('c1_spam_score', 0),
                        "c2_pattern_score": result.performance_metrics.get('c2_pattern_score', 0),
                        "c3_context_score": result.performance_metrics.get('c3_context_score', 0),
                        "cis_final": result.performance_metrics.get('cis_final', result.context_score)
                    }
                else:
                    result_data["precise_scores"] = None
            
            # 콘텐츠 정보 (조인)
            if result.content:
                result_data["content_info"] = {
                    "title": result.content.title,
                    "channel_name": result.content.channel_name,
                    "duration": result.content.duration,
                    "view_count": result.content.view_count
                }
            
            detailed_results.append(result_data)
        
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "filters": {
                "category": category,
                "min_cis": min_cis,
                "max_cis": max_cis,
                "include_scores": include_scores
            },
            "results": detailed_results,
            
            # 🆕 집계 정보
            "summary": {
                "avg_cis": round(sum(r["cis_score"] or 0 for r in detailed_results) / len(detailed_results), 3) if detailed_results else 0,
                "category_counts": _count_categories(detailed_results),
                "confidence_distribution": _analyze_confidence_distribution(detailed_results)
            }
        }
        
    except Exception as e:
        logger.error(f"상세 분석 결과 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"상세 분석 결과 조회 실패: {str(e)}")

@admin_app.get("/analysis-results/statistics")
def get_analysis_statistics(
    days: int = Query(7, ge=1, le=365),
    db: Session = Depends(get_db)
):
    """🆕 분석 결과 통계 (기획서 기반)"""
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    try:
        # 모델별 통계 (CIS 점수 포함)
        model_stats = db.query(
            AnalysisResults.model_used,
            func.count(AnalysisResults.id).label('count'),
            func.avg(AnalysisResults.confidence_score).label('avg_confidence'),
            func.avg(AnalysisResults.context_score).label('avg_cis'),
            func.avg(AnalysisResults.processing_time).label('avg_processing_time')
        ).filter(
            AnalysisResults.created_at >= cutoff_date
        ).group_by(AnalysisResults.model_used).all()
        
        # 카테고리별 통계 (CIS 점수 포함)
        category_stats = db.query(
            AnalysisResults.c_category,
            func.count(AnalysisResults.id).label('count'),
            func.avg(AnalysisResults.context_score).label('avg_cis'),
            func.avg(AnalysisResults.confidence_score).label('avg_confidence')
        ).filter(
            AnalysisResults.created_at >= cutoff_date
        ).group_by(AnalysisResults.c_category).all()
        
        # 🆕 CIS 품질 단계별 통계
        cis_quality_stats = {}
        quality_ranges = [
            ("excellent", 0.3, 2.0),
            ("good", 0.0, 0.3),
            ("fair", -0.2, 0.0),
            ("poor", -0.5, -0.2),
            ("very_poor", -2.0, -0.5)
        ]
        
        for quality_name, min_val, max_val in quality_ranges:
            count = db.query(AnalysisResults).filter(
                AnalysisResults.created_at >= cutoff_date,
                AnalysisResults.context_score >= min_val,
                AnalysisResults.context_score < max_val
            ).count()
            cis_quality_stats[quality_name] = count
        
        # 🆕 정밀 점수 평균 (performance_metrics에서)
        precise_averages = _calculate_precise_scores_averages(db, cutoff_date)
        
        return {
            "period": f"최근 {days}일",
            
            # 모델별 통계
            "model_statistics": [
                {
                    "model": stat.model_used,
                    "count": stat.count,
                    "avg_confidence": round(float(stat.avg_confidence), 3),
                    "avg_cis": round(float(stat.avg_cis or 0), 3),
                    "avg_processing_time": round(float(stat.avg_processing_time or 0), 3)
                } for stat in model_stats
            ],
            
            # 카테고리별 통계
            "category_statistics": [
                {
                    "category": stat.c_category,
                    "category_name": CONTENT_CATEGORIES.get(stat.c_category, stat.c_category),
                    "count": stat.count,
                    "avg_cis": round(float(stat.avg_cis or 0), 3),
                    "avg_confidence": round(float(stat.avg_confidence), 3)
                } for stat in category_stats
            ],
            
            # 🆕 CIS 품질 단계별 통계
            "cis_quality_distribution": cis_quality_stats,
            
            # 🆕 정밀 점수 평균
            "precise_scores_averages": precise_averages,
            
            # 전체 요약
            "summary": {
                "total_analyses": sum(stat.count for stat in model_stats),
                "overall_avg_cis": round(
                    sum(stat.avg_cis * stat.count for stat in model_stats if stat.avg_cis) / 
                    sum(stat.count for stat in model_stats), 3
                ) if model_stats else 0
            }
        }
        
    except Exception as e:
        logger.error(f"분석 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"분석 통계 조회 실패: {str(e)}")

# =============================================
# 🆕 향상된 HITL 워크플로우
# =============================================

@admin_app.get("/validation-labels/prioritized")
def get_prioritized_validation_labels(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    priority_method: str = Query("cis_based", regex="^(cis_based|confidence_based|mixed)$"),
    db: Session = Depends(get_db)
):
    """🆕 CIS 기반 우선순위 검토 대기 항목"""
    
    try:
        query = db.query(ValidationLabels).filter(
            ValidationLabels.review_status == ReviewStatus.PENDING
        ).join(AnalysisResults, ValidationLabels.video_id == AnalysisResults.video_id)
        
        # 🆕 우선순위 정렬 방식
        if priority_method == "cis_based":
            # CIS 점수가 낮을수록 우선순위 높음
            query = query.order_by(AnalysisResults.context_score.asc())
        elif priority_method == "confidence_based":
            # 신뢰도가 낮을수록 우선순위 높음
            query = query.order_by(AnalysisResults.confidence_score.asc())
        else:  # mixed
            # CIS + 신뢰도 혼합 점수
            query = query.order_by(
                (AnalysisResults.context_score - AnalysisResults.confidence_score).asc()
            )
        
        total = query.count()
        labels = query.offset(skip).limit(limit).all()
        
        prioritized_items = []
        for label in labels:
            # 관련 분석 결과 조회
            analysis_result = db.query(AnalysisResults).filter_by(
                video_id=label.video_id
            ).order_by(desc(AnalysisResults.created_at)).first()
            
            item_data = {
                "label_id": label.label_id,
                "video_id": label.video_id,
                "review_status": label.review_status.value,
                "is_hard_example": label.is_hard_example,
                "difficulty_score": label.difficulty_score,
                "created_at": label.created_at.isoformat(),
                
                # 🆕 우선순위 정보
                "priority_info": {
                    "cis_score": float(analysis_result.context_score or 0) if analysis_result else 0,
                    "confidence_score": analysis_result.confidence_score if analysis_result else 0,
                    "category": analysis_result.c_category if analysis_result else "Unknown",
                    "priority_reason": _determine_priority_reason(analysis_result) if analysis_result else "No analysis data"
                }
            }
            
            # 콘텐츠 정보
            if analysis_result and analysis_result.content:
                item_data["content_info"] = {
                    "title": analysis_result.content.title[:100] + "..." if len(analysis_result.content.title) > 100 else analysis_result.content.title,
                    "channel_name": analysis_result.content.channel_name
                }
            
            prioritized_items.append(item_data)
        
        return {
            "total": total,
            "skip": skip,
            "limit": limit,
            "priority_method": priority_method,
            "items": prioritized_items,
            
            # 🆕 우선순위 통계
            "priority_stats": {
                "high_priority": len([item for item in prioritized_items 
                                    if item["priority_info"]["cis_score"] < -0.5]),
                "medium_priority": len([item for item in prioritized_items 
                                      if -0.5 <= item["priority_info"]["cis_score"] < -0.2]),
                "low_priority": len([item for item in prioritized_items 
                                   if item["priority_info"]["cis_score"] >= -0.2])
            }
        }
        
    except Exception as e:
        logger.error(f"우선순위 검토 항목 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"우선순위 검토 항목 조회 실패: {str(e)}")

@admin_app.post("/validation-labels/{label_id}/review")
def review_validation_label(
    label_id: str,
    review_data: dict,
    db: Session = Depends(get_db)
):
    """🆕 검증 라벨 검토 처리"""
    
    try:
        label = db.query(ValidationLabels).filter_by(label_id=label_id).first()
        if not label:
            raise HTTPException(status_code=404, detail="검증 라벨을 찾을 수 없습니다")
        
        # 검토 상태 업데이트
        new_status = review_data.get("status")
        if new_status in ["approved", "rejected", "needs_revision"]:
            label.review_status = ReviewStatus[new_status.upper()]
            label.reviewed_at = datetime.now()
        
        # 🆕 CIS 기반 피드백 점수 업데이트
        if "cis_feedback" in review_data:
            label.feedback_context = {
                **(label.feedback_context or {}),
                "cis_feedback": review_data["cis_feedback"],
                "reviewer_cis_assessment": review_data.get("reviewer_cis_score"),
                "review_timestamp": datetime.now().isoformat()
            }
        
        db.commit()
        
        return {
            "success": True,
            "label_id": label_id,
            "new_status": label.review_status.value,
            "message": "검토가 완료되었습니다"
        }
        
    except Exception as e:
        logger.error(f"검증 라벨 검토 실패: {e}")
        raise HTTPException(status_code=500, detail=f"검토 처리 실패: {str(e)}")

# =============================================
# 🔧 헬퍼 함수들
# =============================================

def _calculate_precise_scores_stats(db: Session, cutoff_date: datetime) -> Dict:
    """정밀 점수 통계 계산"""
    try:
        results = db.query(AnalysisResults).filter(
            AnalysisResults.created_at >= cutoff_date,
            AnalysisResults.performance_metrics.isnot(None)
        ).all()
        
        c1_scores = []
        c2_scores = []
        c3_scores = []
        cis_scores = []
        for result in results:
            if isinstance(result.performance_metrics, dict):
                c1_scores.append(result.performance_metrics.get('c1_spam_score', 0))
                c2_scores.append(result.performance_metrics.get('c2_pattern_score', 0))
                c3_scores.append(result.performance_metrics.get('c3_context_score', 0))
                cis_scores.append(result.performance_metrics.get('cis_final', result.context_score or 0))
        
        if not c1_scores:
            return {"message": "정밀 점수 데이터 없음"}
        
        return {
            "c1_spam": {
                "average": round(sum(c1_scores) / len(c1_scores), 3),
                "max": round(max(c1_scores), 3),
                "min": round(min(c1_scores), 3),
                "high_risk_count": len([s for s in c1_scores if s > 1.0])
            },
            "c2_pattern": {
                "average": round(sum(c2_scores) / len(c2_scores), 3),
                "max": round(max(c2_scores), 3),
                "factory_pattern_count": len([s for s in c2_scores if s > 0.85])
            },
            "c3_context": {
                "average": round(sum(c3_scores) / len(c3_scores), 3),
                "high_quality_count": len([s for s in c3_scores if s > 0.7]),
                "poor_quality_count": len([s for s in c3_scores if s < 0.4])
            },
            "cis_final": {
                "average": round(sum(cis_scores) / len(cis_scores), 3),
                "excellent_count": len([s for s in cis_scores if s >= 0.3]),
                "poor_count": len([s for s in cis_scores if s < -0.2])
            }
        }
        
    except Exception as e:
        logger.error(f"정밀 점수 통계 계산 실패: {e}")
        return {"error": str(e)}

def _calculate_high_confidence_rate(db: Session, cutoff_date: datetime) -> float:
    """고신뢰도 분석 비율 계산"""
    try:
        total = db.query(AnalysisResults).filter(
            AnalysisResults.created_at >= cutoff_date
        ).count()
        
        high_confidence = db.query(AnalysisResults).filter(
            AnalysisResults.created_at >= cutoff_date,
            AnalysisResults.confidence_score >= 0.8
        ).count()
        
        return round(high_confidence / total * 100, 1) if total > 0 else 0
        
    except Exception:
        return 0

def _calculate_auto_approval_rate(db: Session, cutoff_date: datetime) -> float:
    """자동 승인 비율 계산"""
    try:
        total = db.query(AnalysisResults).filter(
            AnalysisResults.created_at >= cutoff_date
        ).count()
        
        auto_approved = db.query(AnalysisResults).filter(
            AnalysisResults.created_at >= cutoff_date,
            AnalysisResults.status == AnalysisResultStatus.AUTO_APPROVE
        ).count()
        
        return round(auto_approved / total * 100, 1) if total > 0 else 0
        
    except Exception:
        return 0

def _calculate_cis_quality_score(cis_stats) -> float:
    """CIS 기반 품질 점수 계산"""
    try:
        if not cis_stats or not cis_stats.avg_cis:
            return 0
        
        avg_cis = float(cis_stats.avg_cis)
        
        # CIS 점수를 0-100 점수로 변환
        if avg_cis >= 0.3:
            return 90 + min(10, (avg_cis - 0.3) * 20)  # 90-100점
        elif avg_cis >= 0:
            return 70 + (avg_cis / 0.3) * 20  # 70-90점
        elif avg_cis >= -0.2:
            return 50 + ((avg_cis + 0.2) / 0.2) * 20  # 50-70점
        elif avg_cis >= -0.5:
            return 30 + ((avg_cis + 0.5) / 0.3) * 20  # 30-50점
        else:
            return max(0, 30 + (avg_cis + 0.5) * 20)  # 0-30점
            
    except Exception:
        return 0

def _create_cis_histogram_bins(scores: List[float]) -> Dict:
    """CIS 점수 히스토그램 빈 생성"""
    try:
        import numpy as np
        
        # 고정 구간으로 히스토그램 생성
        bins = [-2, -1, -0.5, -0.2, 0, 0.3, 0.5, 1, 2]
        hist, bin_edges = np.histogram(scores, bins=bins)
        
        bin_data = []
        for i in range(len(hist)):
            bin_data.append({
                "range": f"{bin_edges[i]:.1f} ~ {bin_edges[i+1]:.1f}",
                "count": int(hist[i]),
                "percentage": round(int(hist[i]) / len(scores) * 100, 1)
            })
        
        return {
            "bins": bin_data,
            "total_samples": len(scores)
        }
        
    except Exception as e:
        logger.error(f"히스토그램 생성 실패: {e}")
        return {"bins": [], "total_samples": 0}

def _detect_cis_anomalies(scores: List[float]) -> Dict:
    """CIS 점수 이상치 탐지"""
    try:
        if len(scores) < 10:
            return {"anomalies": [], "method": "insufficient_data"}
        
        import numpy as np
        
        # IQR 방법으로 이상치 탐지
        q1 = np.percentile(scores, 25)
        q3 = np.percentile(scores, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        anomalies = [score for score in scores if score < lower_bound or score > upper_bound]
        
        return {
            "anomalies": anomalies,
            "anomaly_count": len(anomalies),
            "anomaly_rate": round(len(anomalies) / len(scores) * 100, 2),
            "thresholds": {
                "lower_bound": round(lower_bound, 3),
                "upper_bound": round(upper_bound, 3)
            },
            "method": "iqr"
        }
        
    except Exception as e:
        logger.error(f"이상치 탐지 실패: {e}")
        return {"anomalies": [], "method": "error"}

def _analyze_quality_trends(daily_stats) -> Dict:
    """품질 트렌드 분석"""
    try:
        if len(daily_stats) < 3:
            return {"trend": "insufficient_data"}
        
        # 최근 7일 평균 vs 이전 7일 평균
        recent_scores = [float(stat.avg_cis) for stat in daily_stats[-7:]]
        previous_scores = [float(stat.avg_cis) for stat in daily_stats[-14:-7]] if len(daily_stats) >= 14 else []
        
        recent_avg = sum(recent_scores) / len(recent_scores)
        
        if previous_scores:
            previous_avg = sum(previous_scores) / len(previous_scores)
            trend_direction = "improving" if recent_avg > previous_avg else "declining"
            change_rate = round((recent_avg - previous_avg) / abs(previous_avg) * 100, 1)
        else:
            trend_direction = "stable"
            change_rate = 0
        
        return {
            "trend_direction": trend_direction,
            "change_rate": change_rate,
            "recent_average": round(recent_avg, 3),
            "previous_average": round(sum(previous_scores) / len(previous_scores), 3) if previous_scores else None
        }
        
    except Exception as e:
        logger.error(f"트렌드 분석 실패: {e}")
        return {"trend": "error"}

def _count_categories(results: List[Dict]) -> Dict:
    """카테고리별 개수 계산"""
    counts = {}
    for result in results:
        category = result.get("category", "Unknown")
        counts[category] = counts.get(category, 0) + 1
    return counts

def _analyze_confidence_distribution(results: List[Dict]) -> Dict:
    """신뢰도 분포 분석"""
    confidences = [result.get("confidence_score", 0) for result in results]
    
    if not confidences:
        return {}
    
    return {
        "high_confidence": len([c for c in confidences if c >= 0.8]),
        "medium_confidence": len([c for c in confidences if 0.5 <= c < 0.8]),
        "low_confidence": len([c for c in confidences if c < 0.5]),
        "average": round(sum(confidences) / len(confidences), 3)
    }

def _calculate_precise_scores_averages(db: Session, cutoff_date: datetime) -> Dict:
    """정밀 점수 평균 계산"""
    try:
        results = db.query(AnalysisResults).filter(
            AnalysisResults.created_at >= cutoff_date,
            AnalysisResults.performance_metrics.isnot(None)
        ).all()
        
        c1_scores = []
        c2_scores = []
        c3_scores = []
        
        for result in results:
            if isinstance(result.performance_metrics, dict):
                c1_scores.append(result.performance_metrics.get('c1_spam_score', 0))
                c2_scores.append(result.performance_metrics.get('c2_pattern_score', 0))
                c3_scores.append(result.performance_metrics.get('c3_context_score', 0))
        
        return {
            "c1_average": round(sum(c1_scores) / len(c1_scores), 3) if c1_scores else 0,
            "c2_average": round(sum(c2_scores) / len(c2_scores), 3) if c2_scores else 0,
            "c3_average": round(sum(c3_scores) / len(c3_scores), 3) if c3_scores else 0,
            "sample_count": len(c1_scores)
        }
        
    except Exception:
        return {"c1_average": 0, "c2_average": 0, "c3_average": 0, "sample_count": 0}

def _determine_priority_reason(analysis_result) -> str:
    """우선순위 결정 근거"""
    if not analysis_result:
        return "분석 데이터 없음"
    
    cis_score = analysis_result.context_score or 0
    confidence = analysis_result.confidence_score
    category = analysis_result.c_category
    
    reasons = []
    
    if cis_score < -0.5:
        reasons.append("CIS 점수 매우 낮음")
    elif cis_score < -0.2:
        reasons.append("CIS 점수 불량")
    
    if confidence < 0.5:
        reasons.append("낮은 신뢰도")
    elif confidence < 0.8:
        reasons.append("중간 신뢰도")
    
    if category in ["C1", "C2"]:
        reasons.append(f"고위험 카테고리 ({category})")
    
    return " | ".join(reasons) if reasons else "정상 범위"

# =============================================
# 🆕 기존 CRUD 엔드포인트들 (업데이트)
# =============================================

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
                "channel_name": c.channel_name,
                "duration": c.duration,
                "view_count": c.view_count,
                "status": c.status.value,
                "created_at": c.created_at.isoformat() if c.created_at else None
            } for c in contents
        ]
    }

@admin_app.get("/user-feedback")
def get_user_feedback(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    action_type: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """사용자 피드백 조회"""
    
    query = db.query(UserFeedback)
    
    if action_type:
        query = query.filter(UserFeedback.user_action == action_type)
    
    feedbacks = query.order_by(desc(UserFeedback.created_at)).offset(skip).limit(limit).all()
    total = query.count()
    
    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "feedbacks": [
            {
                "feedback_id": f.feedback_id,
                "video_id": f.video_id,
                "user_action": f.user_action,
                "feedback_text": f.feedback_text,
                "rating": f.rating,
                "created_at": f.created_at.isoformat() if f.created_at else None,
                "is_processed": f.is_processed,
                "feedback_context": f.feedback_context
            } for f in feedbacks
        ]
    }

@admin_app.get("/export/analysis-results")
def export_analysis_results(
    format: str = Query("json", regex="^(json|csv)$"),
    days: int = Query(30, ge=1, le=365),
    include_scores: bool = Query(True),
    db: Session = Depends(get_db)
):
    """🆕 분석 결과 내보내기 (CIS 점수 포함)"""
    
    cutoff_date = datetime.now() - timedelta(days=days)
    
    try:
        results = db.query(AnalysisResults).filter(
            AnalysisResults.created_at >= cutoff_date
        ).order_by(desc(AnalysisResults.created_at)).all()
        
        export_data = []
        for result in results:
            row_data = {
                "result_id": result.result_id,
                "video_id": result.video_id,
                "category": result.c_category,
                "confidence_score": result.confidence_score,
                "cis_score": result.context_score,
                "model_used": result.model_used,
                "processing_time": result.processing_time,
                "status": result.status.value,
                "created_at": result.created_at.isoformat() if result.created_at else None
            }
            
            # 🆕 정밀 점수 포함
            if include_scores and result.performance_metrics:
                if isinstance(result.performance_metrics, dict):
                    row_data.update({
                        "c1_spam_score": result.performance_metrics.get('c1_spam_score', 0),
                        "c2_pattern_score": result.performance_metrics.get('c2_pattern_score', 0),
                        "c3_context_score": result.performance_metrics.get('c3_context_score', 0),
                        "cis_final": result.performance_metrics.get('cis_final', result.context_score)
                    })
            
            # 콘텐츠 정보
            if result.content:
                row_data.update({
                    "title": result.content.title,
                    "channel_name": result.content.channel_name,
                    "duration": result.content.duration,
                    "view_count": result.content.view_count
                })
            
            export_data.append(row_data)
        
        if format == "json":
            return {
                "export_info": {
                    "format": "json",
                    "period": f"최근 {days}일",
                    "total_records": len(export_data),
                    "include_scores": include_scores,
                    "exported_at": datetime.now().isoformat()
                },
                "data": export_data
            }
        else:
            # CSV 형태로 변환
            import io
            import csv
            from fastapi.responses import StreamingResponse
            
            output = io.StringIO()
            if export_data:
                writer = csv.DictWriter(output, fieldnames=export_data[0].keys())
                writer.writeheader()
                writer.writerows(export_data)
            
            output.seek(0)
            
            return StreamingResponse(
                io.BytesIO(output.getvalue().encode('utf-8')),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=analysis_results_{days}days.csv"}
            )
            
    except Exception as e:
        logger.error(f"데이터 내보내기 실패: {e}")
        raise HTTPException(status_code=500, detail=f"데이터 내보내기 실패: {str(e)}")

# =============================================
# 🆕 대시보드용 실시간 데이터
# =============================================

@admin_app.get("/dashboard/real-time-stats")
def get_real_time_dashboard_stats(db: Session = Depends(get_db)):
    """실시간 대시보드 통계"""
    
    try:
        now = datetime.now()
        today = now.date()
        
        # 오늘의 기본 통계
        today_analyses = db.query(AnalysisResults).filter(
            func.date(AnalysisResults.created_at) == today
        ).count()
        
        # 최근 1시간 분석 수
        hour_ago = now - timedelta(hours=1)
        recent_analyses = db.query(AnalysisResults).filter(
            AnalysisResults.created_at >= hour_ago
        ).count()
        
        # CIS 점수 분포 (최근 24시간)
        day_ago = now - timedelta(days=1)
        recent_cis_scores = db.query(AnalysisResults.context_score).filter(
            AnalysisResults.created_at >= day_ago,
            AnalysisResults.context_score.isnot(None)
        ).all()
        
        cis_scores = [float(score.context_score) for score in recent_cis_scores]
        
        return {
            "timestamp": now.isoformat(),
            "today_stats": {
                "total_analyses": today_analyses,
                "recent_hour_analyses": recent_analyses,
                "avg_cis_today": round(sum(cis_scores) / len(cis_scores), 3) if cis_scores else 0
            },
            "cis_distribution_24h": {
                "excellent": len([s for s in cis_scores if s >= 0.3]),
                "good": len([s for s in cis_scores if 0 <= s < 0.3]),
                "fair": len([s for s in cis_scores if -0.2 <= s < 0]),
                "poor": len([s for s in cis_scores if -0.5 <= s < -0.2]),
                "very_poor": len([s for s in cis_scores if s < -0.5])
            },
            "system_health": {
                "database_connected": True,
                "api_responsive": True,
                "processing_queue": 0  # 실제 구현시 큐 크기
            }
        }
        
    except Exception as e:
        logger.error(f"실시간 통계 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"실시간 통계 조회 실패: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(admin_app, host="0.0.0.0", port=8001)

    