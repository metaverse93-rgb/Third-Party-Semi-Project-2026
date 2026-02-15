#pip install scenedetect[opencv] easyocr opencv-python
#pip install opencv-python
#winget install OpenJS.NodeJS.LTS
#pip install yt_dlp
#pip install open_clip_torch torch torchvision
# pip install transformers
import yt_dlp
import os
from scenedetect import detect, ContentDetector
import easyocr
import cv2
import csv
import torch
from PIL import Image
# [필수 추가] BLIP 모델을 위한 라이브러리
from transformers import BlipProcessor, BlipForConditionalGeneration
from deep_translator import GoogleTranslator
# --- 설정 구간 ---
save_path = 'D:/project/shorts/%(title)s.%(ext)s'
ydl_opts = {
    'format': 'best[ext=mp4]/worst',
    'outtmpl': save_path,
    'js_runtimes': {'node': {}},
    'remote_components': ['ejs:github'],
}

# --- 1단계: 다운로드 ---
shorts_url = input("영상 링크 입력: ")
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(shorts_url, download=True)
    downloaded_file = ydl.prepare_filename(info)

# --- 2단계: 장면 분석 ---
print("\n[분석] 장면 감지 시작...")
scene_list = detect(downloaded_file, ContentDetector(threshold=27.0))
print(f"총 {len(scene_list)}개의 장면이 감지되었습니다.")

# --- 3단계: AI 모델 준비 (OCR & CLIP) ---
print("[준비] 이미지 설명 생성 모델(BLIP) 로드 중...")
reader = easyocr.Reader(['ko', 'en'])

# [추가] CLIP 모델 로드 (이 부분이 있어야 작동합니다)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
translator = GoogleTranslator(source='en', target='ko')

# --- 4단계: 통합 분석 및 CSV 저장 (중복 반복문 합침) ---
output_csv = downloaded_file.replace('.mp4', '_analysis.csv')
print(f"\n[분석] 텍스트 및 이미지 분석 시작...")

with open(output_csv, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['장면 번호', '추출된 텍스트', '이미지 분석(BLIP )'])
    
    cap = cv2.VideoCapture(downloaded_file)

    for i, scene in enumerate(scene_list):
        start_frame = scene[0].get_frames()
        end_frame = scene[1].get_frames()
        mid_frame = (start_frame + end_frame) // 2
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
        ret, frame = cap.read()
        
        if ret:
            # 1. OCR (텍스트)
            text_list = reader.readtext(frame, detail=0)
            scene_text = " ".join(text_list).strip()
            
            # 2. BLIP (라벨 없이 이미지 설명 생성)
            raw_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(raw_image, return_tensors="pt")

            with torch.no_grad():
                out = model.generate(**inputs)
                caption_en = processor.decode(out[0], skip_special_tokens=True) # 영어 변수
            try:
                # translator.translate()의 결과를 caption_ko에 저장
                caption_ko = translator.translate(caption_en)
            except Exception as e:
                print(f"번역 오류: {e}")
                caption_ko = caption_en 

            # 4. [중요!] 출력과 저장을 'caption_ko'로 변경
            print(f"[{i+1}번 장면] AI 설명(한글): {caption_ko}") # 여기를 ko로!
            writer.writerow([i+1, scene_text, caption_ko])  

    cap.release()

print(f"\n[최종 완료] 결과 파일: {output_csv}")
