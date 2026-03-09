"""
사용 가능한 모델 확인
"""
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def check_available_models():
    """사용 가능한 모델 확인"""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("❌ API 키가 없습니다")
        return
    
    try:
        client = OpenAI(api_key=api_key)
        
        # 간단한 테스트 (gpt-3.5-turbo)
        print("🧪 GPT-3.5-turbo 테스트...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        print("✅ GPT-3.5-turbo 접근 가능")
        
        # GPT-4 테스트
        print("\n🧪 GPT-4 접근 권한 테스트...")
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Hi"}],
                max_tokens=5
            )
            print("✅ GPT-4 접근 가능")
        except Exception as e:
            if "does not exist" in str(e) or "model" in str(e):
                print("❌ GPT-4 접근 권한 없음")
            else:
                print(f"❌ GPT-4 테스트 실패: {e}")
                
    except Exception as e:
        print(f"❌ 전체 테스트 실패: {e}")

if __name__ == "__main__":
    check_available_models()