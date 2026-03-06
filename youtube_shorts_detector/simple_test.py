"""
API 키 없이도 테스트 가능한 간단 버전
"""
from models import AnalysisResult, ContentCategory, MockVideoData

def create_mock_data():
    """테스트용 Mock 데이터 생성"""
    return MockVideoData(
        video_id="test_001",
        title="🔥충격🔥 이것만 알면 100만원 번다!! (진짜임)",
        description="돈 버는 비법 공개합니다",
        keyframes=["frame1.jpg", "frame2.jpg"],
        ocr_text="100만원 돈버는법 클릭 지금바로",
        metadata={
            "duration": 58,
            "views": 1250000,
            "upload_date": "2024-01-15"
        }
    )

def test_basic_functionality():
    """기본 기능 테스트 (API 호출 없음)"""
    print("🧪 1-A단계 기본 기능 테스트 시작")
    print("=" * 50)
    
    # Mock 데이터 생성 테스트
    try:
        mock_data = create_mock_data()
        print(f"✅ Mock 데이터 생성 성공")
        print(f"📋 테스트 영상: {mock_data.title}")
        print(f"📋 OCR 텍스트: {mock_data.ocr_text}")
    except Exception as e:
        print(f"❌ Mock 데이터 생성 실패: {e}")
        return
    
    # 분석 결과 시뮬레이션 테스트
    print("\n📊 다양한 분석 결과 시뮬레이션:")
    
    test_cases = [
        {
            "name": "자동 승인 케이스 (정상 콘텐츠)",
            "category": ContentCategory.C5,
            "reasoning": "정상적인 교육 콘텐츠로 판단됨. 명확한 내러티브와 유용한 정보 제공",
            "confidence": 0.95
        },
        {
            "name": "사람 검토 케이스 (경계선)",
            "category": ContentCategory.C1,
            "reasoning": "자극적인 제목이 의심스러우나 내용은 양호함. 추가 검토 필요",
            "confidence": 0.65
        },
        {
            "name": "자동 거부 케이스 (어그로)",
            "category": ContentCategory.C1,
            "reasoning": "전형적인 어그로 콘텐츠. 사기성 키워드 다수 포함",
            "confidence": 0.25
        },
        {
            "name": "에러 케이스",
            "category": ContentCategory.C5,
            "reasoning": "분석 실패",
            "confidence": 0.0,
            "error": "API 호출 실패"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        try:
            result = AnalysisResult(
                c_category=case["category"],
                reasoning_log=case["reasoning"],
                confidence_score=case["confidence"],
                error_message=case.get("error")
            )
            
            print(f"\n{i}️⃣ {case['name']}")
            print(f"   분류: {result.c_category.value}")
            print(f"   상태: {result.status.value}")
            print(f"   신뢰도: {result.confidence_score:.2f}")
            print(f"   근거: {result.reasoning_log}")
            if result.error_message:
                print(f"   에러: {result.error_message}")
                
        except Exception as e:
            print(f"❌ 테스트 케이스 {i} 실패: {e}")
    
    print(f"\n🎉 1-A단계 기본 기능 테스트 완료!")

if __name__ == "__main__":
    test_basic_functionality()