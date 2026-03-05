"""
End-to-End 테스트
유튜브 URL → 최종 분류 결과까지 전체 흐름 테스트
"""
import json
from pipeline import YouTubeShortsAnalysisPipeline
from config import MOCK_MODE

def create_test_urls():
    """테스트용 URL 리스트 생성"""
    return [
        "https://youtube.com/shorts/test_aggro_video",     # 어그로 테스트
        "https://youtube.com/shorts/test_educational",     # 교육 콘텐츠 테스트
        "https://youtube.com/shorts/test_factory_pattern", # 공장형 패턴 테스트
        "https://youtube.com/shorts/test_quality_issue",   # 품질 문제 테스트
        "https://youtube.com/shorts/test_copyright"        # 저작권 문제 테스트
    ]

def run_single_test(pipeline: YouTubeShortsAnalysisPipeline, video_url: str, test_name: str):
    """개별 테스트 실행"""
    print(f"\n{'='*60}")
    print(f"🧪 {test_name} 테스트")
    print(f"📹 URL: {video_url}")
    print(f"{'='*60}")
    
    try:
        result = pipeline.analyze_video(video_url)
        
        if result.get("error"):
            print(f"❌ 테스트 실패: {result['error_message']}")
            return False
        
        # 결과 출력
        video_info = result["video_info"]
        analysis = result["analysis_result"]
        technical = result["technical_details"]
        
        print(f"\n📋 영상 정보:")
        print(f"   제목: {video_info['title']}")
        print(f"   채널: {video_info['channel']}")
        print(f"   조회수: {video_info['view_count']:,}회")
        print(f"   길이: {video_info['duration']}초")
        
        print(f"\n🎯 분석 결과:")
        print(f"   분류: {analysis['category']} ({analysis['category_name']})")
        print(f"   상태: {analysis['status']}")
        print(f"   신뢰도: {analysis['confidence_score']:.2f}")
        print(f"   근거: {analysis['reasoning_log']}")
        
        print(f"\n🔧 기술적 세부사항:")
        print(f"   OCR 텍스트: {technical['ocr_text'][:100]}...")
        print(f"   레이아웃 점수: {technical['layout_score']:.2f}")
        print(f"   처리 시간: {technical['total_time']:.2f}초")
        
        print(f"\n💡 추천 액션:")
        for action in result["recommended_actions"]:
            print(f"   {action}")
        
        print(f"\n📝 상세 로그:")
        for log in result["logs"][-3:]:  # 마지막 3개 로그만 표시
            print(f"   {log}")
        
        return True
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류: {e}")
        return False

def run_performance_benchmark(pipeline: YouTubeShortsAnalysisPipeline):
    """성능 벤치마크 테스트"""
    print(f"\n{'='*60}")
    print(f"⚡ 성능 벤치마크 테스트")
    print(f"{'='*60}")
    
    test_urls = create_test_urls()
    results = []
    
    for i, url in enumerate(test_urls, 1):
        print(f"\n🔄 벤치마크 테스트 {i}/{len(test_urls)}")
        result = pipeline.analyze_video(url)
        
        if not result.get("error"):
            processing_time = result["technical_details"]["total_time"]
            results.append(processing_time)
            print(f"   처리 시간: {processing_time:.2f}초")
        else:
            print(f"   실패: {result['error_message']}")
    
    if results:
        avg_time = sum(results) / len(results)
        min_time = min(results)
        max_time = max(results)
        
        print(f"\n📊 성능 통계:")
        print(f"   평균 처리 시간: {avg_time:.2f}초")
        print(f"   최소 처리 시간: {min_time:.2f}초")
        print(f"   최대 처리 시간: {max_time:.2f}초")
        print(f"   성공률: {len(results)}/{len(test_urls)} ({len(results)/len(test_urls)*100:.1f}%)")

def test_model_switching(pipeline: YouTubeShortsAnalysisPipeline):
    """모델 전환 테스트"""
    print(f"\n{'='*60}")
    print(f"🔄 모델 전환 테스트")
    print(f"{'='*60}")
    
    test_url = create_test_urls()[0]
    
    # GPT4o 모드 테스트
    print(f"\n1️⃣ GPT4o 모드 테스트")
    pipeline.switch_analyzer_mode("GPT4o")
    result_gpt = pipeline.analyze_video(test_url)
    
    if not result_gpt.get("error"):
        print(f"   분류: {result_gpt['analysis_result']['category']}")
        print(f"   신뢰도: {result_gpt['analysis_result']['confidence_score']:.2f}")
    
    # Qwen 모드 테스트
    print(f"\n2️⃣ Qwen 모드 테스트")
    pipeline.switch_analyzer_mode("Qwen")
    result_qwen = pipeline.analyze_video(test_url)
    
    if not result_qwen.get("error"):
        print(f"   분류: {result_qwen['analysis_result']['category']}")
        print(f"   신뢰도: {result_qwen['analysis_result']['confidence_score']:.2f}")
    
    # 결과 비교
    if not result_gpt.get("error") and not result_qwen.get("error"):
        gpt_category = result_gpt['analysis_result']['category']
        qwen_category = result_qwen['analysis_result']['category']
        
        print(f"\n📊 모델 비교:")
        print(f"   GPT4o: {gpt_category}")
        print(f"   Qwen: {qwen_category}")
        print(f"   일치 여부: {'✅' if gpt_category == qwen_category else '❌'}")

def main():
    """메인 테스트 함수"""
    print(f"🚀 유튜브 쇼츠 분석 파이프라인 End-to-End 테스트")
    print(f"Mock 모드: {'✅ 활성' if MOCK_MODE else '❌ 비활성'}")
    
    # 파이프라인 초기화
    pipeline = YouTubeShortsAnalysisPipeline(analyzer_mode="GPT4o")
    
    # 테스트 케이스 정의
    test_cases = [
        ("어그로/스팸 콘텐츠", create_test_urls()[0]),
        ("교육 콘텐츠", create_test_urls()[1]),
        ("공장형 패턴", create_test_urls()[2]),
        ("품질 문제", create_test_urls()[3]),
        ("저작권 문제", create_test_urls()[4])
    ]
    
    # 개별 테스트 실행
    success_count = 0
    for test_name, test_url in test_cases:
        if run_single_test(pipeline, test_url, test_name):
            success_count += 1
    
    print(f"\n🎉 개별 테스트 완료!")
    print(f"성공: {success_count}/{len(test_cases)} ({success_count/len(test_cases)*100:.1f}%)")
    
    # 성능 벤치마크
    run_performance_benchmark(pipeline)
    
    # 모델 전환 테스트
    test_model_switching(pipeline)
    
    print(f"\n✅ 전체 End-to-End 테스트 완료!")
    print(f"📝 1-B단계 구현 성공: Phase 1 + Phase 2 통합 파이프라인")

if __name__ == "__main__":
    main()