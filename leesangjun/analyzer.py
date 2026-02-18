import os
import base64
import requests
import pandas as pd
from dotenv import load_dotenv
from config import FULL_OCR_DIR, SHORTS_OCR_DIR, FULL_FRAMES_DIR, SHORTS_FRAMES_DIR
import time  # 지연 시간을 위해 추가

# --- [.env 로드 로직] ---
env_path = r'D:\python\.env'
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
    API_KEY = os.getenv("OPENAI_API_KEY")
    if API_KEY:
        print(f"✅ API Key 로드 완료 (앞글자: {API_KEY[:7]}...)")
    else:
        print("❌ .env는 로드되었으나 OPENAI_API_KEY가 비어있습니다.")
else:
    print("❌ .env 파일을 찾을 수 없습니다.")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# [수정] 재시도 로직 추가 (기본 3번 재시도)
def get_visual_summary(image_path, retries=3):
    if not API_KEY: return "키 에러"
    
    base64_image = encode_image(image_path)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": "이 영상 프레임에서 자막을 제외하고, 어떤 시각적 상황(인물의 표정, 행동, 배경 등)이 벌어지고 있는지 한국어로 한 문장 요약해줘."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ],
        "max_tokens": 150
    }

    for i in range(retries):
        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            
            # 429 Error (Too Many Requests) 처리
            if response.status_code == 429:
                wait_time = (i + 1) * 15  # 15초, 30초... 점진적으로 대기 시간 증가
                print(f"⏳ 과부하 발생! {wait_time}초 후 재시도합니다... ({i+1}/{retries})")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()

        except Exception as e:
            if i == retries - 1:
                print(f"❌ GPT API 최종 오류: {e}")
                return f"요약 실패: {e}"
            time.sleep(2)

def run_visual_summary_step(csv_path, image_dir):
    if not os.path.exists(csv_path):
        print(f"⚠️ OCR 결과 파일 없음: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"🚀 분석 시작: {len(df)}개 프레임 대상")

    visual_summaries = []
    for idx, row in df.iterrows():
        img_name = row['file_name']
        img_path = os.path.join(image_dir, img_name)
        
        if os.path.exists(img_path):
            print(f"📸 [{idx+1}/{len(df)}] 분석 중: {img_name}...")
            summary = get_visual_summary(img_path)
            visual_summaries.append(summary)
            
            # [수정] 요청 간 최소 1.2초 지연 (Rate Limit 방지용)
            time.sleep(1.2)
        else:
            print(f"⚠️ 이미지 없음: {img_path}")
            visual_summaries.append("파일 없음")

    df['visual_summary'] = visual_summaries
    output_path = csv_path.replace(".csv", "_final.csv")
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ 완료! 저장됨: {output_path}")

if __name__ == "__main__":
    print("=== 4단계: 이미지 상황 요약 (GPT-4o mini) ===")
    choice = input("선택 (1.원본 / 2.쇼츠): ").strip()
    
    if choice == '1':
        ocr_base = FULL_OCR_DIR
        img_base = FULL_FRAMES_DIR
    else:
        ocr_base = SHORTS_OCR_DIR
        img_base = SHORTS_FRAMES_DIR
    
    files = [f for f in os.listdir(ocr_base) if f.endswith('_ocr_result.csv')]
    
    if not files:
        print("⚠️ 처리할 CSV 파일이 없습니다.")
    else:
        for i, f in enumerate(files): print(f"[{i}] {f}")
        idx = int(input("분석할 파일 번호 선택: "))
        selected_csv = files[idx]
        
        video_folder_name = selected_csv.replace("_ocr_result.csv", "")
        target_img_dir = os.path.join(img_base, video_folder_name)
        
        run_visual_summary_step(os.path.join(ocr_base, selected_csv), target_img_dir)