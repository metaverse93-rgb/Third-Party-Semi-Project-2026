import os
import cv2
import easyocr
import pandas as pd
import numpy as np
import re
from config import FULL_FRAMES_DIR, FULL_OCR_DIR, SHORTS_FRAMES_DIR, SHORTS_OCR_DIR

def run_ocr_for_folder(image_dir, output_ocr_dir, video_name, is_shorts=True):
    # [수정] 원본은 한/일/영 모두 대응, 쇼츠는 한/영 대응
    lang_list = ['ko', 'en'] if is_shorts else ['ja', 'en'] 
    reader = easyocr.Reader(lang_list, gpu=True)
    
    results = []
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])
    
    print(f"📖 OCR 분석 시작: {video_name} ({'쇼츠' if is_shorts else '원본'})")

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img_array = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is None: continue
        
        h, w = img.shape[:2]
        
        # [영역] 원본은 보통 하단, 쇼츠는 좀 더 넓게 설정
        y_start, y_end = (int(h * 0.3), int(h * 0.95)) if is_shorts else (int(h * 0.4), int(h * 0.99))
            
        roi_img = img[y_start:y_end, 0:w]
        roi_img_gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        
        ocr_res = reader.readtext(roi_img_gray, detail=1, mag_ratio=1.5)
        
        if ocr_res:
            texts, y_pos = [], []
            for (bbox, text, prob) in ocr_res:
                # [수정] 한글, 일본어(히라/가타/한자), 영어, 숫자만 허용하고 특수문자 제거
                clean_text = re.sub(r'[^가-힣ぁ-んァ-ン一-龥a-zA-Z0-9\s]', '', text).strip()
                threshold = 0.35 if is_shorts else 0.2

                # 신뢰도 0.35 이상, 2글자 이상인 경우 수집
                if len(clean_text) > 1 and prob > threshold:
                    texts.append(clean_text)
                    y_center = (bbox[0][1] + bbox[2][1]) / 2 # bbox 구조에 따라 인덱스 확인 필요
                    y_pos.append(round((y_start + y_center) / h, 3))
            
            if texts:
                full_text = " ".join(texts).strip()
                if full_text:
                    results.append({
                        'file_name': img_file, 
                        'text': full_text, 
                        'y_position': sum(y_pos)/len(y_pos) if y_pos else 0
                    })

    if results:
        df = pd.DataFrame(results)
        
        # [정제] 연속 중복 제거
        df = df[df['text'].shift() != df['text']]
        
        # [정제] 의미 없는 숫자만 있는 행 등은 필터링 (최소 한 글자 이상의 문자가 있어야 함)
        df = df[df['text'].str.contains('[가-힣ぁ-んァ-ン一-龥a-zA-Z]', na=False)]
        
        if not df.empty:
            csv_path = os.path.join(output_ocr_dir, f"{video_name}_ocr_result.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8-sig')
            print(f"💾 저장 완료: {csv_path} (총 {len(df)}행)")
        else:
            print("⚠️ 필터링 후 남은 데이터가 없습니다.")

if __name__ == "__main__":
    choice = input("선택 (1.원본 / 2.쇼츠): ").strip()
    is_shorts_mode = (choice == '2')
    
    base_dir = SHORTS_FRAMES_DIR if is_shorts_mode else FULL_FRAMES_DIR
    out_dir = SHORTS_OCR_DIR if is_shorts_mode else FULL_OCR_DIR
    
    folders = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    if folders:
        for i, f in enumerate(folders): print(f"[{i}] {f}")
        idx = int(input("번호 선택: "))
        run_ocr_for_folder(os.path.join(base_dir, folders[idx]), out_dir, folders[idx], is_shorts=is_shorts_mode)