"""
4단계 UI/API 통합 테스트
"""
import requests
import time
import json

def test_ui_integration():
    """UI/API 통합 테스트"""
    print("🧪 4단계 UI/API 통합 테스트 시작")
    print("=" * 60)
    
    # 1. 사용자 API 테스트
    print("\n1️⃣ 사용자 API 테스트")
    test_user_api()
    
    # 2. 크롬 확장 API 테스트
    print("\n2️⃣ 크롬 확장 API 테스트") 
    test_chrome_api()
    
    # 3. 피드백 API 테스트
    print("\n3️⃣ 피드백 API 테스트")
    test_feedback_api()
    
    print(f"\n🎉 4단계 UI/API 통합 테스트 완료!")

def test_user_api():
    """사용자 API 테스트"""
    
    try:
        # 헬스 체크
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("   ✅ 사용자 API 서버 정상")
        else:
            print("   ❌ 사용자 API 서버 오류")
            return
        
        # 영상 분석 테스트
        print("   🎬 영상 분석 테스트...")
        
        analysis_request = {
            "video_url": "https://youtube.com/shorts/test_video",
            "request_source": "test"
        }
        
        response = requests.post(
            "http://localhost:8000/analyze",
            json=analysis_request,
            timeout=65
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ 분석 성공: {result['video_id']}")
            print(f"      카테고리: {result['analysis_result']['category']}")
            print(f"      신뢰도: {result['confidence_level']}")
            return result['video_id']
        else:
            print(f"   ❌ 분석 실패: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ 사용자 API 테스트 실패: {e}")

def test_chrome_api():
    """크롬 확장 API 테스트"""
    
    try:
        # 빠른 분석 테스트
        response = requests.post(
            "http://localhost:8002/quick-analyze",
            params={"video_url": "https://youtube.com/shorts/chrome_test"},
            timeout=35
        )
        
        if response.status_code == 200:
            result = response.json()
            print("   ✅ 크롬 확장 빠른 분석 성공")
            print(f"      위험도: {result['risk_level']}")
            print(f"      오버레이: {result['overlay_text']}")
        else:
            print(f"   ❌ 크롬 확장 분석 실패: {response.status_code}")
        
        # 오버레이 설정 테스트
        response = requests.get("http://localhost:8002/overlay-config")
        if response.status_code == 200:
            print("   ✅ 오버레이 설정 조회 성공")
        
    except Exception as e:
        print(f"   ❌ 크롬 확장 API 테스트 실패: {e}")

def test_feedback_api():
    """피드백 API 테스트"""
    
    try:
        feedback_request = {
            "video_id": "test_video_123",
            "action": "like",
            "feedback_text": "테스트 피드백입니다"
        }
        
        response = requests.post(
            "http://localhost:8000/feedback",
            json=feedback_request,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print("   ✅ 피드백 제출 성공")
            print(f"      피드백 ID: {result['feedback_id']}")
        else:
            print(f"   ❌ 피드백 제출 실패: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ 피드백 API 테스트 실패: {e}")

if __name__ == "__main__":
    test_ui_integration()