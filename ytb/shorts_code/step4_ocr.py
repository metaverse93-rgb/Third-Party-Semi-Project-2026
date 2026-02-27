import os
import cv2
import easyocr
import pandas as pd
import re
from difflib import SequenceMatcher
from config import PATHS

def get_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def run_ocr_analysis(video_name_only):
    print(f"\n[Step 4] OCR 분석 시작 (한글 발음 최적화 모드)...")
    
    image_dir = os.path.join(PATHS["shorts_frames"], video_name_only)
    output_csv = os.path.join(PATHS["shorts_ocr"], f"{video_name_only}_ocr.csv")
    os.makedirs(PATHS["shorts_ocr"], exist_ok=True)

    if not os.path.exists(image_dir):
        print(f"❌ 프레임 폴더를 찾을 수 없습니다.")
        return

    # 한글 발음을 잘 읽기 위해 ko, en 유지
    reader = easyocr.Reader(['ko', 'en'], gpu=True)
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    
    results = []
    last_text = ""

    for filename in image_files:
        img_path = os.path.join(image_dir, filename)
        image = cv2.imread(img_path)
        if image is None: continue
        
        h, w, _ = image.shape
        y_start, y_end = int(h * 0.3), int(h * 0.85)
        cropped_img = image[y_start:y_end, :]

        # OCR 실행
        ocr_result = reader.readtext(cropped_img, detail=0, paragraph=True)
        full_text = " ".join(ocr_result).strip()

        # 1. 오직 '순수 한글'과 '공백'만 남김 (일어/특수문자/알파벳 완전 제거)
        # 이 과정에서 '무손표' 같은 오타 중 한글이 아닌 기호 기반 오타들이 먼저 걸러집니다.
        korean_only = re.sub(r'[^가-힣\s]', '', full_text).strip()
        
        # 2. 연속된 공백 하나로 정리
        korean_only = re.sub(r'\s+', ' ', korean_only)

        # 3. [지능형 필터] 문장 끝에 붙는 불필요한 한 글자 노이즈 제거 (선택 사항)
        # 예: "사랑할 거야 무" -> "사랑할 거야"
        if len(korean_only) > 5 and (korean_only.endswith('득') or korean_only.endswith('표')):
            korean_only = korean_only[:-1].strip()

        # 4. 중복 및 유사도 검사
        if len(korean_only) >= 2:
            similarity = get_similarity(korean_only, last_text)
            # 75% 이상 유사하면 중복으로 간주하여 패스
            if similarity < 0.75:
                time_match = re.search(r'at_(\d{2}_\d{2}_\d{2})', filename)
                timestamp = time_match.group(1).replace("_", ":") if time_match else "unknown"

                results.append({
                    "timestamp": timestamp,
                    "text": korean_only
                })
                print(f"📝 [{timestamp}] {korean_only}")
                last_text = korean_only

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"✅ 분석 완료! 노이즈가 제거된 {len(results)}개 문장 저장.")
    
    return output_csv

