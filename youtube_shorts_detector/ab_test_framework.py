"""
A/B 테스트 프레임워크
GPT-4o vs Qwen2.5-VL 성능 비교 및 통계적 유의성 검정
"""
import random
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from scipy import stats
import logging

from config import AB_TEST_CONFIG, MOCK_MODE, PERFORMANCE_TARGETS
from content_analyzer import ContentAnalyzer
from evaluation_metrics import EvaluationMetrics
from context_score_calculator import ContextScoreCalculator
from performance_logger import PerformanceLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ABTestFramework:
    """A/B 테스트 프레임워크 클래스"""
    
    def __init__(self, mock_mode: bool = None):
        """
        초기화
        Args:
            mock_mode: None이면 config.MOCK_MODE 사용
        """
        self.mock_mode = mock_mode if mock_mode is not None else MOCK_MODE
        self.enable_ab_test = AB_TEST_CONFIG["enable_ab_test"]
        self.test_ratio = AB_TEST_CONFIG["test_ratio"]
        self.min_samples = AB_TEST_CONFIG["minimum_samples"]
        self.significance_level = AB_TEST_CONFIG["significance_level"]
        
        # 분석기 초기화
        self.analyzer_a = ContentAnalyzer(mode="GPT4o", mock_mode=self.mock_mode)
        self.analyzer_b = ContentAnalyzer(mode="Qwen", mock_mode=self.mock_mode)
        
        # 평가 도구 초기화
        self.evaluator = EvaluationMetrics(mock_mode=self.mock_mode)
        self.context_calculator = ContextScoreCalculator(mock_mode=self.mock_mode)
        self.logger = PerformanceLogger()
        
        # 결과 저장소
        self.results_a = []  # GPT-4o 결과
        self.results_b = []  # Qwen 결과
        
        logger.info(f"ABTestFramework 초기화 (Mock: {self.mock_mode}, A/B 활성: {self.enable_ab_test})")
    
    def assign_to_group(self, video_id: str) -> str:
        """
        비디오를 A/B 그룹에 할당
        
        Args:
            video_id: 비디오 ID
            
        Returns:
            str: "A" (GPT-4o) 또는 "B" (Qwen)
        """
        if not self.enable_ab_test:
            return "A"  # A/B 테스트 비활성화시 기본값
        
        # 일관된 할당을 위해 video_id 해시값 사용
        hash_value = hash(video_id) % 100
        return "A" if hash_value < (self.test_ratio * 100) else "B"
    
    def analyze_with_ab_test(self, 
                            video_id: str,
                            frames: List[str], 
                            ocr_text: str, 
                            metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        A/B 테스트를 포함한 분석 수행
        
        Args:
            video_id: 비디오 ID
            frames: 키프레임 리스트
            ocr_text: OCR 텍스트
            metadata: 메타데이터
            
        Returns:
            Dict: 분석 결과 + A/B 테스트 정보
        """
        start_time = datetime.now()
        
        # 그룹 할당
        assigned_group = self.assign_to_group(video_id)
        
        # 해당 그룹의 분석기로 분석
        analyzer = self.analyzer_a if assigned_group == "A" else self.analyzer_b
        
        logger.info(f"🧪 A/B 테스트 분석: {video_id} → 그룹 {assigned_group} ({analyzer.mode})")
        
        # 분석 수행
        analysis_result = analyzer.analyze(frames, ocr_text, metadata=metadata)
        
        # Context Score 계산
        context_result = self.context_calculator.calculate_context_score(
            frames, ocr_text, metadata
        )
        
        # 성능 지표 계산 (Mock 데이터)
        performance_metrics = self._calculate_performance_metrics(
            analysis_result, ocr_text
        )
        
        # 결과 구성
        ab_result = {
            "video_id": video_id,
            "assigned_group": assigned_group,
            "model_used": analyzer.mode,
            "analysis_result": {
                "category": analysis_result.c_category.value,
                "confidence_score": analysis_result.confidence_score,
                "reasoning_log": analysis_result.reasoning_log,
                "status": analysis_result.status.value,
                "processing_time": analysis_result.processing_time
            },
            "context_score": context_result,
            "performance_metrics": performance_metrics,
            "timestamp": start_time.isoformat()
        }
        
        # 그룹별 결과 저장
        if assigned_group == "A":
            self.results_a.append(ab_result)
        else:
            self.results_b.append(ab_result)
        
        # 로깅
        if self.logger.enable_logging:
            self.logger.log_single_result(
                video_id=video_id,
                analysis_result=ab_result["analysis_result"],
                context_score=context_result,
                performance_metrics=performance_metrics
            )
        
        return ab_result
    
    def _calculate_performance_metrics(self, 
                                     analysis_result, 
                                     ocr_text: str) -> Dict[str, float]:
        """성능 지표 계산 (Mock 기반)"""
        
        # 가상의 reference 텍스트 생성
        reference_text = ocr_text + " 참조 텍스트"  # 간단한 변형
        
        return {
            "cer": self.evaluator.calculate_cer(ocr_text, reference_text),
            "rouge_l": self.evaluator.calculate_rouge_l(ocr_text, reference_text),
            "bert_score": self.evaluator.calculate_bert_score(ocr_text, reference_text),
            "f1_weighted": 0.89 + random.uniform(-0.05, 0.05)  # Mock F1 점수
        }
    
    def run_batch_ab_test(self, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        배치 A/B 테스트 실행
        
        Args:
            test_data: 테스트할 데이터 리스트
            
        Returns:
            Dict: A/B 테스트 결과 및 통계
        """
        logger.info(f"🧪 배치 A/B 테스트 시작: {len(test_data)}개 샘플")
        
        batch_results = []
        
        for i, data in enumerate(test_data):
            logger.info(f"처리 중: {i+1}/{len(test_data)}")
            
            result = self.analyze_with_ab_test(
                video_id=data.get("video_id", f"batch_test_{i}"),
                frames=data.get("frames", []),
                ocr_text=data.get("ocr_text", ""),
                metadata=data.get("metadata", {})
            )
            
            batch_results.append(result)
        
        # 통계 분석
        statistics = self.calculate_ab_statistics()
        
        return {
            "total_samples": len(batch_results),
            "group_a_count": len(self.results_a),
            "group_b_count": len(self.results_b),
            "batch_results": batch_results,
            "statistics": statistics
        }
    
    def calculate_ab_statistics(self) -> Dict[str, Any]:
        """A/B 테스트 통계 분석"""
        
        if len(self.results_a) < 2 or len(self.results_b) < 2:
            return {
                "error": "통계 분석을 위한 충분한 샘플 없음",
                "min_required": self.min_samples,
                "current_a": len(self.results_a),
                "current_b": len(self.results_b)
            }
        
        # 그룹별 메트릭 추출
        metrics_a = self._extract_group_metrics(self.results_a)
        metrics_b = self._extract_group_metrics(self.results_b)
        
        # 통계적 유의성 검정
        statistical_tests = self._perform_statistical_tests(metrics_a, metrics_b)
        
        # 성능 비교
        performance_comparison = self._compare_performance(metrics_a, metrics_b)
        
        return {
            "group_a_stats": self._calculate_group_stats(metrics_a, "GPT-4o"),
            "group_b_stats": self._calculate_group_stats(metrics_b, "Qwen"),
            "statistical_tests": statistical_tests,
            "performance_comparison": performance_comparison,
            "recommendation": self._generate_recommendation(statistical_tests, performance_comparison)
        }
    
    def _extract_group_metrics(self, group_results: List[Dict]) -> Dict[str, List[float]]:
        """그룹 결과에서 메트릭 추출"""
        metrics = {
            "confidence_scores": [],
            "processing_times": [],
            "context_scores": [],
            "cer_scores": [],
            "rouge_l_scores": [],
            "bert_scores": [],
            "f1_scores": []
        }
        
        for result in group_results:
            metrics["confidence_scores"].append(result["analysis_result"]["confidence_score"])
            metrics["processing_times"].append(result["analysis_result"]["processing_time"] or 0)
            metrics["context_scores"].append(result["context_score"]["context_score"])
            metrics["cer_scores"].append(result["performance_metrics"]["cer"])
            metrics["rouge_l_scores"].append(result["performance_metrics"]["rouge_l"])
            metrics["bert_scores"].append(result["performance_metrics"]["bert_score"])
            metrics["f1_scores"].append(result["performance_metrics"]["f1_weighted"])
        
        return metrics
    
    def _calculate_group_stats(self, metrics: Dict[str, List[float]], model_name: str) -> Dict[str, Any]:
        """그룹 통계 계산"""
        stats = {"model_name": model_name}
        
        for metric_name, values in metrics.items():
            if values:
                stats[metric_name] = {
                    "mean": round(np.mean(values), 4),
                    "std": round(np.std(values), 4),
                    "min": round(np.min(values), 4),
                    "max": round(np.max(values), 4),
                    "median": round(np.median(values), 4),
                    "count": len(values)
                }
        
        return stats
    
    def _perform_statistical_tests(self, metrics_a: Dict, metrics_b: Dict) -> Dict[str, Any]:
        """통계적 유의성 검정"""
        
        tests = {}
        
        # 주요 메트릭에 대한 t-검정
        key_metrics = ["confidence_scores", "context_scores", "cer_scores", "rouge_l_scores"]
        
        for metric in key_metrics:
            if metric in metrics_a and metric in metrics_b:
                values_a = metrics_a[metric]
                values_b = metrics_b[metric]
                
                if len(values_a) >= 2 and len(values_b) >= 2:
                    try:
                        t_stat, p_value = stats.ttest_ind(values_a, values_b)
                        
                        tests[metric] = {
                            "t_statistic": round(t_stat, 4),
                            "p_value": round(p_value, 4),
                            "is_significant": p_value < self.significance_level,
                            "significance_level": self.significance_level,
                            "interpretation": self._interpret_test_result(t_stat, p_value, metric)
                        }
                    except Exception as e:
                        tests[metric] = {"error": str(e)}
        
        return tests
    
    def _interpret_test_result(self, t_stat: float, p_value: float, metric: str) -> str:
        """검정 결과 해석"""
        if p_value >= self.significance_level:
            return f"{metric}: 통계적으로 유의한 차이 없음"
        
        if t_stat > 0:
            return f"{metric}: 그룹 A(GPT-4o)가 그룹 B(Qwen)보다 높음 (유의미)"
        else:
            return f"{metric}: 그룹 B(Qwen)가 그룹 A(GPT-4o)보다 높음 (유의미)"
    
    def _compare_performance(self, metrics_a: Dict, metrics_b: Dict) -> Dict[str, Any]:
        """성능 비교 분석"""
        
        comparison = {}
        
        # 주요 지표별 승자 결정
        comparisons = {
            "confidence_scores": "higher_better",
            "processing_times": "lower_better", 
            "context_scores": "higher_better",
            "cer_scores": "lower_better",
            "rouge_l_scores": "higher_better",
            "bert_scores": "higher_better",
            "f1_scores": "higher_better"
        }
        
        wins_a = 0
        wins_b = 0
        
        for metric, better_direction in comparisons.items():
            if metric in metrics_a and metric in metrics_b:
                mean_a = np.mean(metrics_a[metric])
                mean_b = np.mean(metrics_b[metric])
                
                if better_direction == "higher_better":
                    winner = "A" if mean_a > mean_b else "B"
                else:
                    winner = "A" if mean_a < mean_b else "B"
                
                comparison[metric] = {
                    "group_a_mean": round(mean_a, 4),
                    "group_b_mean": round(mean_b, 4),
                    "winner": winner,
                    "difference": round(abs(mean_a - mean_b), 4),
                    "improvement_rate": round(abs(mean_a - mean_b) / max(mean_a, mean_b) * 100, 2)
                }
                
                if winner == "A":
                    wins_a += 1
                else:
                    wins_b += 1
        
        comparison["overall_summary"] = {
            "group_a_wins": wins_a,
            "group_b_wins": wins_b, 
            "total_metrics": len(comparisons),
            "overall_winner": "A (GPT-4o)" if wins_a > wins_b else "B (Qwen)" if wins_b > wins_a else "Tie"
        }
        
        return comparison
    
    def _generate_recommendation(self, statistical_tests: Dict, performance_comparison: Dict) -> Dict[str, str]:
        """최종 추천 생성"""
        
        overall_winner = performance_comparison.get("overall_summary", {}).get("overall_winner", "Tie")
        significant_differences = sum(1 for test in statistical_tests.values() 
                                    if isinstance(test, dict) and test.get("is_significant", False))
        
        if overall_winner == "Tie":
            recommendation = "두 모델의 성능이 비슷합니다. 비용과 속도를 고려하여 선택하세요."
            action = "CONTINUE_AB_TEST"
        elif "A (GPT-4o)" in overall_winner:
            if significant_differences >= 2:
                recommendation = "GPT-4o가 통계적으로 유의하게 우수합니다. GPT-4o 사용을 권장합니다."
                action = "USE_MODEL_A"
            else:
                recommendation = "GPT-4o가 약간 우수하지만 통계적 유의성이 부족합니다. 더 많은 데이터 수집이 필요합니다."
                action = "COLLECT_MORE_DATA"
        else:  # Qwen 승리
            if significant_differences >= 2:
                recommendation = "Qwen이 통계적으로 유의하게 우수합니다. Qwen 사용을 권장합니다."
                action = "USE_MODEL_B"
            else:
                recommendation = "Qwen이 약간 우수하지만 통계적 유의성이 부족합니다. 더 많은 데이터 수집이 필요합니다."
                action = "COLLECT_MORE_DATA"
        
        return {
            "recommendation": recommendation,
            "action": action,
            "confidence": "high" if significant_differences >= 3 else "medium" if significant_differences >= 1 else "low"
        }
    
    def generate_ab_test_report(self) -> str:
        """A/B 테스트 리포트 생성"""
        
        if not self.results_a and not self.results_b:
            return "📊 A/B 테스트 리포트\n" + "="*50 + "\n❌ 테스트 데이터가 없습니다."
        
        stats = self.calculate_ab_statistics()
        
        if "error" in stats:
            return f"📊 A/B 테스트 리포트\n" + "="*50 + f"\n❌ {stats['error']}"
        
        report = f"""
📊 A/B 테스트 리포트
{'='*50}

🎯 테스트 개요:
   그룹 A (GPT-4o): {len(self.results_a)}개 샘플
   그룹 B (Qwen): {len(self.results_b)}개 샘플
   
📈 성능 비교:
"""
        
        # 성능 비교 추가
        perf_comp = stats.get("performance_comparison", {})
        overall = perf_comp.get("overall_summary", {})
        
        report += f"   전체 승자: {overall.get('overall_winner', 'Unknown')}\n"
        report += f"   그룹 A 승리: {overall.get('group_a_wins', 0)}개 지표\n"
        report += f"   그룹 B 승리: {overall.get('group_b_wins', 0)}개 지표\n"
        
        # 통계적 유의성 추가
        stat_tests = stats.get("statistical_tests", {})
        significant_count = sum(1 for test in stat_tests.values() 
                               if isinstance(test, dict) and test.get("is_significant", False))
        
        report += f"\n📊 통계적 유의성:\n"
        report += f"   유의한 차이: {significant_count}개 지표\n"
        
        # 추천사항 추가
        recommendation = stats.get("recommendation", {})
        report += f"\n💡 추천사항:\n"
        report += f"   {recommendation.get('recommendation', '추가 분석 필요')}\n"
        report += f"   신뢰도: {recommendation.get('confidence', 'medium')}\n"
        
        return report

def test_ab_framework():
    """A/B 테스트 프레임워크 테스트"""
    print("🧪 A/B 테스트 프레임워크 테스트 시작")
    print("=" * 60)
    
    # 프레임워크 초기화
    ab_framework = ABTestFramework(mock_mode=True)
    
    # 샘플 테스트 데이터 생성
    test_data = []
    for i in range(20):  # 20개 샘플로 테스트
        test_data.append({
            "video_id": f"test_video_{i}",
            "frames": [f"frame_{i}_1.jpg", f"frame_{i}_2.jpg"],
            "ocr_text": f"테스트 텍스트 {i} Python 강의" if i % 2 == 0 else f"🔥충격🔥 돈버는법 {i}",
            "metadata": {
                "title": f"테스트 영상 제목 {i}",
                "duration": 60 + i,
                "view_count": 1000 * i
            }
        })
    
    print("\n1️⃣ 배치 A/B 테스트 실행")
    batch_result = ab_framework.run_batch_ab_test(test_data)
    
    print(f"   총 샘플: {batch_result['total_samples']}개")
    print(f"   그룹 A: {batch_result['group_a_count']}개")
    print(f"   그룹 B: {batch_result['group_b_count']}개")
    
    print("\n2️⃣ A/B 테스트 리포트")
    report = ab_framework.generate_ab_test_report()
    print(report)
    
    print("\n✅ A/B 테스트 프레임워크 테스트 완료!")

if __name__ == "__main__":
    test_ab_framework()