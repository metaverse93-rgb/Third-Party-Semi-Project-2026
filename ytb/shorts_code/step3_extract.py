import os
import cv2
import numpy as np
import re
from config import PATHS

def get_subtitle_mask(hsv_roi):
    """자막 색상(흰색, 노란색)만 추출하는 마스크 생성"""
    lower_white = np.array([0, 0, 200]) 
    upper_white = np.array([180, 40, 255])
    lower_yellow = np.array([15, 100, 150]) 
    upper_yellow = np.array([35, 255, 255])

    mask_w = cv2.inRange(hsv_roi, lower_white, upper_white)
    mask_y = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
    
    return cv2.bitwise_or(mask_w, mask_y)

def extract_subtitle_frames_and_text(video_path):
    if not video_path or not os.path.exists(video_path):
        return

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    video_name = re.sub(r'[\\/:*?"<>|]', '_', os.path.splitext(os.path.basename(video_path))[0])
    output_dir = os.path.abspath(os.path.join(PATHS["shorts_frames"], video_name))
    os.makedirs(output_dir, exist_ok=True)

    # 자막 검출 영역 (중앙 40% ~ 하단 90%)
    y_start, y_end = int(height * 0.4), int(height * 0.9)
    
    last_saved_mask = None
    count = 0
    
    print(f"🎬 추출 최적화 모드 시작: {video_name}")

    while True:
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret: break

        # 1. 전처리
        roi = frame[y_start:y_end, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        current_mask = get_subtitle_mask(hsv)

        # 2. 자막 존재 여부 확인
        if np.sum(current_mask) > 15000:
            
            is_new_subtitle = True
            current_max_val = 0.0 # 초기값 설정 (오류 방지)

            if last_saved_mask is not None:
                # 3. 유사도 계산
                res = cv2.matchTemplate(current_mask, last_saved_mask, cv2.TM_CCOEFF_NORMED)
                _, current_max_val, _, _ = cv2.minMaxLoc(res)
                
                if current_max_val > 0.85:
                    is_new_subtitle = False

            if is_new_subtitle:
                timestamp = frame_id / fps
                time_str = f"{int(timestamp//60):02d}_{int(timestamp%60):02d}_{int((timestamp*100)%100):02d}"
                save_path = os.path.join(output_dir, f"frame_{count:03d}_at_{time_str}.jpg")
                
                # 프레임 저장
                cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                
                count += 1
                last_saved_mask = current_mask
                print(f"📸 캡처: {time_str} (유사도: {current_max_val:.2f})")
                
                # 자막 저장 후 다음 1.5초간 건너뛰기 (중복 방지 강화)
                for _ in range(int(fps * 1.5)):
                    cap.grab()
                    
        # 기본 스킵 (약 0.2초 단위로 검사하도록 변경)
        for _ in range(int(fps * 0.2)):
            cap.grab()

    cap.release()
    print(f"✅ 최적화 추출 완료! 총 {count}개 프레임 저장.")