"""
ContentAnalyzer 테스트 함수
Mock 데이터로 즉시 실행 가능
"""
import json
from content_analyzer import ContentAnalyzer
from models import MockVideoData

def create_mock_data() -> MockVideoData:
    """테스트용 Mock 데이터 생성"""
    return MockVideoData(
        video_id="test_001",
        title="🔥충격🔥 이것만 알면 100만원 번다!! (진짜임)",
        description="돈 버는 비법 공개합니다",
        keyframes=[
            "https://example.com/frame1.jpg",  # 실제로는 base64 이미지
            "https://example.com/frame2.jpg"
        ],
        ocr_text="100만원 돈버는법 클릭 지금바로",
        metadata={
            "duration": 58,
            "views": 1250000,
            "upload_date": "2024-01-15",
            "channel": "돈버는법알려주는채널"
        }
    )

def test_analyzer():
    """ContentAnalyzer 전체 테스트"""
    print("🧪 ContentAnalyzer 테스트 시작")
    print("=" * 50)
    
    # Mock 데이터 준비
    mock_data = create_mock_data()
    print(f"📋 테스트 영상: {mock_data.title}")
    
    # GPT-4o 모드 테스트
    print("\n1️⃣ GPT-4o 모드 테스트")
    analyzer_gpt = ContentAnalyzer(mode="GPT4o")
    
    try:
        result_gpt = analyzer_gpt.analyze(
            frames=mock_data.keyframes,
            ocr_text=mock_data.ocr_text,
            layout_score=0.75,
            metadata=mock_data.metadata
        )
        
        print(f"✅ 분류: {result_gpt.c_category.value}")
        print(f"✅ 상태: {result_gpt.status.value}")
        print(f"✅ 신뢰도: {result_gpt.confidence_score:.2f}")
        print(f"✅ 근거: {result_gpt.reasoning_log[:100]}...")
        
    except Exception as e:
        print(f"❌ GPT-4o 테스트 실패: {e}")
    
    # Qwen 모드 테스트 (vLLM 서버 없으면 실패 예상)
    print("\n2️⃣ Qwen 모드 테스트")
    analyzer_qwen = ContentAnalyzer(mode="Qwen")
    
    try:
        result_qwen = analyzer_qwen.analyze(
            frames=mock_data.keyframes,
            ocr_text=mock_data.ocr_text,
            layout_score=0.75,
            metadata=mock_data.metadata
        )
        
        print(f"✅ 분류: {result_qwen.c_category.value}")
        print(f"✅ 상태: {result_qwen.status.value}")
        print(f"✅ 신뢰도: {result_qwen.confidence_score:.2f}")
        
    except Exception as e:
        print(f"⚠️ Qwen 테스트 실패 (예상됨 - vLLM 서버 없음): {e}")
    
    # 모델 교체 테스트
    print("\n3️⃣ 모델 교체 테스트")
    analyzer_gpt.switch_model("Qwen")
    print(f"✅ 모델 교체 완료: {analyzer_gpt.mode}")
    
    print("\n🎉 테스트 완료!")

def test_error_handling():
    """에러 핸들링 테스트"""
    print("\n🔧 에러 핸들링 테스트")
    
    analyzer = ContentAnalyzer(mode="GPT4o")
    
    # 빈 데이터로 테스트
    result = analyzer.analyze(
        frames=[],
        ocr_text="",
        layout_score=0.0,
        metadata={}
    )
    
    print(f"빈 데이터 처리: {result.status.value}")
    print(f"에러 메시지: {result.error_message}")

if __name__ == "__main__":
    test_analyzer()
    test_error_handling()