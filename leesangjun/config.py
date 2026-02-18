import os

# 프로젝트 기본 경로
BASE_DIR = os.path.normpath(r"D:/project/Third-Party-semi-Project-2026")

# 1. 원본 영상 관련 (입력/출력)
FULL_SHORTS_DIR = os.path.join(BASE_DIR, "full_shorts")
FULL_FRAMES_DIR = os.path.join(BASE_DIR, "full_frames")
FULL_OCR_DIR = os.path.join(BASE_DIR, "full_ocr")

# 2. 쇼츠 영상 관련 (입력/출력)
SHORTS_DIR = os.path.join(BASE_DIR, "shorts")
SHORTS_FRAMES_DIR = os.path.join(BASE_DIR, "shorts_frames")
SHORTS_OCR_DIR = os.path.join(BASE_DIR, "shorts_ocr")

# 기본 루트 폴더들 생성
paths = [FULL_SHORTS_DIR, FULL_FRAMES_DIR, FULL_OCR_DIR, 
         SHORTS_DIR, SHORTS_FRAMES_DIR, SHORTS_OCR_DIR]

for p in paths:
    if not os.path.exists(p):
        os.makedirs(p)