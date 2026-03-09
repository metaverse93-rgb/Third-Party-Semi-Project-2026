"""
HITL (Human-in-the-Loop) 워크플로우
보류 데이터 관리자 검토 및 피드백 기반 모델 개선
"""
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func
from database_manager import db_manager
from database_models import (
    ValidationLabels, AnalysisResults, UserFeedback,
    ReviewStatus, AnalysisResultStatus
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HITLWorkflow:
    """Human-in-the-Loop 워크플로우 관리"""
    
    def __init__(self):
        self.priority_thresholds = {
            "high": 0.3,      # 신뢰도 0.3 이하 고우선순위
            "medium": 0.5,    # 신뢰도 0.3-0.5 중우선순위
            "low": 0.8        # 신뢰도 0.5-0.8 저우선순위
        }
        logger.info("HITLWorkflow 초기화")
    
    def get_pending_reviews(self, 
                           limit: int = 50,
                           priority: Optional[str] = None) -> List[Dict[str, Any]]:
        """검토 대기 중인 항목 조회 (우선순위별)"""
        
        with db_manager.get_db_session() as session:
            query = session.query(ValidationLabels).filter(
                ValidationLabels.review_status == ReviewStatus.PENDING
            )
            
            # 우선순위 필터링
            if priority:
                if priority == "high":
                    # 고우선순위: Hard Example + 낮은 신뢰도
                    query = query.join(AnalysisResults).filter(
                        and_(
                            ValidationLabels.is_hard_example == True,
                            AnalysisResults.confidence_score <= self.priority_thresholds["high"]
                        )
                    )
                elif priority == "medium":
                    query = query.join(AnalysisResults).filter(
                        AnalysisResults.confidence_score.between(
                            self.priority_thresholds["high"],
                            self.priority_thresholds["medium"]
                        )
                    )
                elif priority == "low":
                    query = query.join(AnalysisResults).filter(
                        AnalysisResults.confidence_score.between(
                            self.priority_thresholds["medium"],
                            self.priority_thresholds["low"]
                        )
                    )
            
            # 우선순위 정렬: 어려운 것부터
            labels = query.order_by(
                desc(ValidationLabels.is_hard_example),
                ValidationLabels.difficulty_score.desc(),
                ValidationLabels.created_at
            ).limit(limit).all()
            
            review_items = []
            for label in labels:
                # 관련 분석 결과 조회
                analysis = session.query(AnalysisResults).filter(
                    AnalysisResults.result_id == label.result_id
                ).first()
                
                review_items.append({
                    "label_id": label.label_id,
                    "video_id": label.video_id,
                    "content_title": analysis.content.title if analysis else "Unknown",
                    "predicted_category": analysis.c_category if analysis else "Unknown",
                    "confidence_score": analysis.confidence_score if analysis else 0.0,
                    "reasoning": analysis.reasoning_log if analysis else "",
                    "ocr_text": analysis.content.raw_ocr_text if analysis else "",
                    "is_hard_example": label.is_hard_example,
                    "difficulty_score": label.difficulty_score,
                    "priority": self._determine_priority(analysis.confidence_score if analysis else 0.0),
                    "created_at": label.created_at.isoformat(),
                    "review_comments": label.review_comments
                })
            
            logger.info(f"📋 검토 대기 항목 조회: {len(review_items)}개 (우선순위: {priority or 'all'})")
            return review_items
    
    def _determine_priority(self, confidence_score: float) -> str:
        """신뢰도 기반 우선순위 결정"""
        if confidence_score <= self.priority_thresholds["high"]:
            return "high"
        elif confidence_score <= self.priority_thresholds["medium"]:
            return "medium"
        else:
            return "low"
    
    def submit_review(self, 
                     label_id: str,
                     ground_truth_category: str,
                     reviewer_id: str,
                     review_comments: Optional[str] = None) -> Dict[str, Any]:
        """검토 결과 제출"""
        
        with db_manager.get_db_session() as session:
            label = session.query(ValidationLabels).filter(
                ValidationLabels.label_id == label_id
            ).first()
            
            if not label:
                return {"error": "Validation label not found"}
            
            # 검토 결과 업데이트
            label.ground_truth_category = ground_truth_category
            label.review_status = ReviewStatus.APPROVED
            label.human_reviewer_id = reviewer_id
            label.review_comments = review_comments
            label.reviewed_at = datetime.now()
            
            # 관련 분석 결과와 비교하여 정확도 계산
            analysis = session.query(AnalysisResults).filter(
                AnalysisResults.result_id == label.result_id
            ).first()
            
            accuracy_info = {}
            if analysis:
                is_correct = (analysis.c_category == ground_truth_category)
                accuracy_info = {
                    "predicted_category": analysis.c_category,
                    "ground_truth_category": ground_truth_category,
                    "is_correct": is_correct,
                    "confidence_score": analysis.confidence_score,
                    "model_used": analysis.model_used
                }
            
            session.commit()
            
            logger.info(f"✅ 검토 완료: {label_id} by {reviewer_id}")
            
            return {
                "status": "success",
                "label_id": label_id,
                "reviewer_id": reviewer_id,
                "accuracy_info": accuracy_info,
                "reviewed_at": datetime.now().isoformat()
            }
    
    def get_review_statistics(self, days: int = 30) -> Dict[str, Any]:
        """검토 통계 생성"""
        
        with db_manager.get_db_session() as session:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # 전체 검토 현황
            total_pending = session.query(ValidationLabels).filter(
                ValidationLabels.review_status == ReviewStatus.PENDING
            ).count()
            
            total_reviewed = session.query(ValidationLabels).filter(
                and_(
                    ValidationLabels.review_status == ReviewStatus.APPROVED,
                    ValidationLabels.reviewed_at >= cutoff_date
                )
            ).count()
            
            # 우선순위별 대기 현황
            priority_stats = {}
            for priority in ["high", "medium", "low"]:
                priority_items = self.get_pending_reviews(limit=1000, priority=priority)
                priority_stats[priority] = len(priority_items)
            
            # 검토자별 통계
            reviewer_stats = session.query(
                ValidationLabels.human_reviewer_id,
                func.count(ValidationLabels.id).label('review_count')
            ).filter(
                and_(
                    ValidationLabels.review_status == ReviewStatus.APPROVED,
                    ValidationLabels.reviewed_at >= cutoff_date
                )
            ).group_by(ValidationLabels.human_reviewer_id).all()
            
            return {
                "period": f"최근 {days}일",
                "total_pending": total_pending,
                "total_reviewed": total_reviewed,
                "priority_distribution": priority_stats,
                "reviewer_activity": [
                    {
                        "reviewer_id": stat.human_reviewer_id,
                        "review_count": stat.review_count
                    } for stat in reviewer_stats
                ],
                "timestamp": datetime.now().isoformat()
            }
    
    def get_feedback_insights(self, days: int = 30) -> Dict[str, Any]:
        """사용자 피드백 기반 인사이트"""
        
        with db_manager.get_db_session() as session:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # 피드백 유형별 통계
            feedback_stats = session.query(
                UserFeedback.user_action,
                session.query().count(UserFeedback.id).label('count')
            ).filter(
                UserFeedback.created_at >= cutoff_date
            ).group_by(UserFeedback.user_action).all()
            
            # 부정적 피드백 높은 카테고리 찾기
            negative_feedback = session.query(UserFeedback).filter(
                and_(
                    UserFeedback.user_action.in_(["dislike", "report"]),
                    UserFeedback.created_at >= cutoff_date
                )
            ).all()
            
            # 카테고리별 부정 피드백 집계
            category_issues = {}
            for feedback in negative_feedback:
                # 해당 비디오의 분석 결과 찾기
                analysis = session.query(AnalysisResults).filter(
                    AnalysisResults.video_id == feedback.video_id
                ).first()
                
                if analysis:
                    category = analysis.c_category
                    if category not in category_issues:
                        category_issues[category] = {"count": 0, "issues": []}
                    category_issues[category]["count"] += 1
                    category_issues[category]["issues"].append({
                        "video_id": feedback.video_id,
                        "feedback_type": feedback.user_action,
                        "feedback_text": feedback.feedback_text
                    })
            
            return {
                "period": f"최근 {days}일",
                "feedback_distribution": [
                    {
                        "action": stat.user_action,
                        "count": stat.count
                    } for stat in feedback_stats
                ],
                "problematic_categories": [
                    {
                        "category": category,
                        "negative_feedback_count": data["count"],
                        "sample_issues": data["issues"][:3]  # 최대 3개 샘플
                    } for category, data in category_issues.items()
                ],
                "recommendations": self._generate_improvement_recommendations(category_issues)
            }
    
    def _generate_improvement_recommendations(self, category_issues: Dict) -> List[str]:
        """개선 권장사항 생성"""
        recommendations = []
        
        # 가장 문제가 많은 카테고리 찾기
        if category_issues:
            most_problematic = max(category_issues.items(), key=lambda x: x[1]["count"])
            category, issue_data = most_problematic
            
            if issue_data["count"] > 10:
                recommendations.append(
                    f"{category} 카테고리의 분류 정확도 개선 필요 (부정 피드백 {issue_data['count']}개)"
                )
            
            if issue_data["count"] > 20:
                recommendations.append(
                    f"{category} 카테고리에 대한 추가 학습 데이터 수집 권장"
                )
        
        # 일반적 권장사항
        total_issues = sum(data["count"] for data in category_issues.values())
        if total_issues > 50:
            recommendations.append("전반적인 모델 재학습 고려")
        
        return recommendations

def test_hitl_workflow():
    """HITL 워크플로우 테스트"""
    print("🧪 HITL 워크플로우 테스트 시작")
    print("=" * 50)
    
    workflow = HITLWorkflow()
    
    print("\n1️⃣ 검토 대기 항목 조회")
    try:
        pending_items = workflow.get_pending_reviews(limit=5)
        print(f"   검토 대기 항목: {len(pending_items)}개")
        
        for item in pending_items[:2]:  # 처음 2개만 표시
            print(f"   - {item['label_id']}: {item['content_title'][:30]}... (우선순위: {item['priority']})")
            
    except Exception as e:
        print(f"   ❌ 검토 대기 항목 조회 실패: {e}")
    
    print("\n2️⃣ 검토 통계")
    try:
        stats = workflow.get_review_statistics(days=7)
        print(f"   전체 대기: {stats['total_pending']}개")
        print(f"   최근 검토 완료: {stats['total_reviewed']}개")
        print(f"   우선순위 분포: {stats['priority_distribution']}")
        
    except Exception as e:
        print(f"   ❌ 검토 통계 조회 실패: {e}")
    
    print("\n3️⃣ 피드백 인사이트")
    try:
        insights = workflow.get_feedback_insights(days=7)
        print(f"   피드백 분포: {len(insights['feedback_distribution'])}개 유형")
        print(f"   문제 카테고리: {len(insights['problematic_categories'])}개")
        print(f"   개선 권장사항: {len(insights['recommendations'])}개")
        
    except Exception as e:
        print(f"   ❌ 피드백 인사이트 조회 실패: {e}")
    
    print("\n✅ HITL 워크플로우 테스트 완료")

if __name__ == "__main__":
    test_hitl_workflow()