"""
성능 로깅 시스템
JSON/CSV 형태로 성능 지표 히스토리 저장
"""
import os
import json
import csv
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

from config import PERFORMANCE_LOGGING, PERFORMANCE_TARGETS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceLogger:
    """성능 지표 로깅 클래스"""
    
    def __init__(self):
        """초기화"""
        self.log_dir = PERFORMANCE_LOGGING["log_directory"]
        self.log_format = PERFORMANCE_LOGGING["log_format"]
        self.batch_size = PERFORMANCE_LOGGING["batch_size"]
        self.enable_logging = PERFORMANCE_LOGGING["enable_logging"]
        
        # 로그 디렉토리 생성
        if self.enable_logging:
            os.makedirs(self.log_dir, exist_ok=True)
            logger.info(f"PerformanceLogger 초기화: {self.log_dir} ({self.log_format} 형식)")
        
        self.batch_buffer = []
    
    def log_single_result(self, 
                         video_id: str,
                         analysis_result: Dict[str, Any],
                         context_score: Dict[str, float],
                         performance_metrics: Dict[str, float]) -> None:
        """단일 분석 결과 로깅"""
        if not self.enable_logging:
            return
        
        timestamp = datetime.now()
        
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "video_id": video_id,
            "category": analysis_result.get("category", ""),
            "confidence_score": analysis_result.get("confidence_score", 0.0),
            "status": analysis_result.get("status", ""),
            "context_score": context_score.get("context_score", 0.0),
            "s_semantic": context_score.get("s_semantic", 0.0),
            "o_existence": context_score.get("o_existence", 0.0),
            "a_sync": context_score.get("a_sync", 0.0),
            "cer": performance_metrics.get("cer", 0.0),
            "rouge_l": performance_metrics.get("rouge_l", 0.0),
            "bert_score": performance_metrics.get("bert_score", 0.0),
            "f1_score": performance_metrics.get("f1_weighted", 0.0),
            "processing_time": analysis_result.get("processing_time", 0.0)
        }
        
        self.batch_buffer.append(log_entry)
        
        # 배치 크기에 도달하면 저장
        if len(self.batch_buffer) >= self.batch_size:
            self._flush_batch()
    
    def _flush_batch(self) -> None:
        """배치 버퍼를 파일로 저장"""
        if not self.batch_buffer:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.log_format == "json":
            self._save_as_json(timestamp)
        else:
            self._save_as_csv(timestamp)
        
        logger.info(f"배치 저장 완료: {len(self.batch_buffer)}개 로그")
        self.batch_buffer.clear()
    
    def _save_as_json(self, timestamp: str) -> None:
        """JSON 형태로 저장"""
        filename = f"performance_log_{timestamp}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.batch_buffer, f, ensure_ascii=False, indent=2)
    
    def _save_as_csv(self, timestamp: str) -> None:
        """CSV 형태로 저장"""
        filename = f"performance_log_{timestamp}.csv"
        filepath = os.path.join(self.log_dir, filename)
        
        if self.batch_buffer:
            df = pd.DataFrame(self.batch_buffer)
            df.to_csv(filepath, index=False, encoding='utf-8')
    
    def get_recent_performance(self, days: int = 7) -> Dict[str, Any]:
        """최근 N일간 성능 통계"""
        if not self.enable_logging:
            return {"error": "로깅 비활성화"}
        
        cutoff_date = datetime.now() - timedelta(days=days)
        all_logs = self._load_recent_logs(cutoff_date)
        
        if not all_logs:
            return {"error": "로그 데이터 없음"}
        
        # 통계 계산
        stats = self._calculate_statistics(all_logs)
        
        return {
            "period": f"최근 {days}일",
            "total_samples": len(all_logs),
            "statistics": stats,
            "target_achievement": self._check_target_achievement(stats)
        }
    
    def _load_recent_logs(self, cutoff_date: datetime) -> List[Dict]:
        """최근 로그 파일들 로드"""
        all_logs = []
        
        if not os.path.exists(self.log_dir):
            return all_logs
        
        for filename in os.listdir(self.log_dir):
            if not filename.startswith("performance_log"):
                continue
            
            filepath = os.path.join(self.log_dir, filename)
            
            try:
                if filename.endswith(".json"):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        logs = json.load(f)
                elif filename.endswith(".csv"):
                    df = pd.read_csv(filepath)
                    logs = df.to_dict('records')
                else:
                    continue
                
                # 날짜 필터링
                for log in logs:
                    log_date = datetime.fromisoformat(log["timestamp"])
                    if log_date >= cutoff_date:
                        all_logs.append(log)
                        
            except Exception as e:
                logger.warning(f"로그 파일 로드 실패 {filename}: {e}")
        
        return all_logs
    
    def _calculate_statistics(self, logs: List[Dict]) -> Dict[str, float]:
        """성능 통계 계산"""
        import numpy as np
        
        metrics = ["context_score", "cer", "rouge_l", "bert_score", "f1_score", "processing_time"]
        stats = {}
        
        for metric in metrics:
            values = [log.get(metric, 0) for log in logs if log.get(metric) is not None]
            
            if values:
                stats[f"{metric}_mean"] = round(np.mean(values), 4)
                stats[f"{metric}_std"] = round(np.std(values), 4)
                stats[f"{metric}_min"] = round(np.min(values), 4)
                stats[f"{metric}_max"] = round(np.max(values), 4)
                stats[f"{metric}_median"] = round(np.median(values), 4)
        
        # 카테고리 분포
        categories = [log.get("category", "") for log in logs]
        category_counts = {}
        for cat in categories:
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        stats["category_distribution"] = category_counts
        
        return stats
    
    def _check_target_achievement(self, stats: Dict) -> Dict[str, bool]:
        """목표 달성 여부 확인"""
        achievements = {}
        
        # CER 달성 여부 (낮을수록 좋음)
        if "cer_mean" in stats:
            achievements["CER"] = stats["cer_mean"] <= PERFORMANCE_TARGETS["CER"]["target"]
        
        # ROUGE-L 달성 여부 (높을수록 좋음)
        if "rouge_l_mean" in stats:
            achievements["ROUGE_L"] = stats["rouge_l_mean"] >= PERFORMANCE_TARGETS["ROUGE_L"]["target"]
        
        # BERTScore 달성 여부
        if "bert_score_mean" in stats:
            achievements["BERT_SCORE"] = stats["bert_score_mean"] >= PERFORMANCE_TARGETS["BERT_SCORE"]["target"]
        
        # F1-Score 달성 여부
        if "f1_score_mean" in stats:
            achievements["F1_SCORE"] = stats["f1_score_mean"] >= PERFORMANCE_TARGETS["F1_SCORE"]["target"]
        
        # Context Score 달성 여부
        if "context_score_mean" in stats:
            achievements["CONTEXT_SCORE"] = stats["context_score_mean"] >= PERFORMANCE_TARGETS["CONTEXT_SCORE"]["target"]
        
        return achievements
    
    def cleanup_old_logs(self) -> None:
        """오래된 로그 파일 정리"""
        if not self.enable_logging:
            return
        
        retention_days = PERFORMANCE_LOGGING["retention_days"]
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        deleted_count = 0
        
        for filename in os.listdir(self.log_dir):
            if not filename.startswith("performance_log"):
                continue
            
            filepath = os.path.join(self.log_dir, filename)
            
            try:
                # 파일 생성 시간 확인
                file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                
                if file_time < cutoff_date:
                    os.remove(filepath)
                    deleted_count += 1
                    
            except Exception as e:
                logger.warning(f"로그 파일 삭제 실패 {filename}: {e}")
        
        if deleted_count > 0:
            logger.info(f"오래된 로그 파일 {deleted_count}개 삭제")
    
    def generate_performance_report(self, days: int = 30) -> str:
        """성능 리포트 생성"""
        recent_perf = self.get_recent_performance(days)
        
        if "error" in recent_perf:
            return f"리포트 생성 실패: {recent_perf['error']}"
        
        report = f"""
📊 성능 리포트 ({recent_perf['period']})
{'='*50}

📈 전체 통계:
   총 샘플 수: {recent_perf['total_samples']:,}개

🎯 핵심 지표:
   Context Score: {recent_perf['statistics'].get('context_score_mean', 0):.3f}
   CER: {recent_perf['statistics'].get('cer_mean', 0):.4f}
   ROUGE-L: {recent_perf['statistics'].get('rouge_l_mean', 0):.3f}
   BERTScore: {recent_perf['statistics'].get('bert_score_mean', 0):.3f}
   F1-Score: {recent_perf['statistics'].get('f1_score_mean', 0):.3f}

✅ 목표 달성 현황:
"""
        
        achievements = recent_perf.get('target_achievement', {})
        for metric, achieved in achievements.items():
            status = "✅ 달성" if achieved else "❌ 미달성"
            report += f"   {metric}: {status}\n"
        
        return report

def test_performance_logger():
    """성능 로거 테스트"""
    print("🧪 성능 로거 테스트 시작")
    
    logger = PerformanceLogger()
    
    # 샘플 데이터 로깅
    for i in range(5):
        logger.log_single_result(
            video_id=f"test_{i}",
            analysis_result={
                "category": f"C{(i%5)+1}",
                "confidence_score": 0.85,
                "status": "AUTO_APPROVE",
                "processing_time": 1.5
            },
            context_score={
                "context_score": 0.78,
                "s_semantic": 0.8,
                "o_existence": 0.7,
                "a_sync": 0.85
            },
            performance_metrics={
                "cer": 0.04,
                "rouge_l": 0.82,
                "bert_score": 0.86,
                "f1_weighted": 0.90
            }
        )
    
    # 강제 저장
    logger._flush_batch()
    
    # 성능 리포트 생성
    report = logger.generate_performance_report(7)
    print(report)
    
    print("✅ 성능 로거 테스트 완료")

if __name__ == "__main__":
    test_performance_logger()