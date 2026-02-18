import os
import cv2
import numpy as np
import re
from config import FULL_SHORTS_DIR, FULL_FRAMES_DIR, SHORTS_DIR, SHORTS_FRAMES_DIR

def extract_subtitle_frames(video_path, base_output_dir, sensitivity=0.05):
    if not video_path or not os.path.exists(video_path):
        print(f"❌ 영상을 찾을 수 없습니다.")
        return

    video_filename = os.path.basename(video_path)
    raw_name = os.path.splitext(video_filename)[0]
    # 특수문자 제거하여 안전한 폴더명 생성
    video_name_only = re.sub(r'[\\/:*?"<>|]', '_', raw_name)
    
    output_dir = os.path.abspath(os.path.join(base_output_dir, video_name_only))
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 저장 경로 확정: {output_dir}")

    cap = cv2.VideoCapture(video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 자막 감지 영역 설정
    y_start, y_end = int(height * 0.4), int(height * 0.95)
    
    prev_area = None
    frame_idx, saved_count = 0, 0
    min_interval = int(fps * 0.8)
    last_saved_frame = -min_interval

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        roi = frame[y_start:y_end, 0:width]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev_area is not None and (frame_idx - last_saved_frame) > min_interval:
            diff = cv2.absdiff(prev_area, gray)
            _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            change_ratio = np.count_nonzero(diff_thresh) / (roi.size / 3)
            
            if change_ratio > sensitivity:
                img_name = f"scene_{saved_count:03d}_f{frame_idx}.jpg"
                img_path = os.path.join(output_dir, img_name)
                
                # 한글 경로 대응 저장
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_encoded.tofile(img_path)
                
                saved_count += 1
                last_saved_frame = frame_idx
                print(f"\r📸 저장 중... ({saved_count}개) -> {img_name}", end="")

        prev_area = gray
        frame_idx += 1
        
    cap.release()
    print(f"\n✅ 완료: {saved_count}개 이미지 저장됨.")
    os.startfile(output_dir) 

if __name__ == "__main__":
    print("=== 영상별 폴더 생성 프레임 추출기 ===")
    choice = input("선택 (1.원본 / 2.쇼츠): ").strip()
    
    # 1. 모드에 따른 경로 및 감도 설정
    if choice == '1':
        target_video_dir = FULL_SHORTS_DIR
        target_output_dir = FULL_FRAMES_DIR
        sensitivity_val = 0.15
        mode_name = "원본"
    else:
        target_video_dir = SHORTS_DIR
        target_output_dir = SHORTS_FRAMES_DIR
        sensitivity_val = 0.05
        mode_name = "쇼츠"

    # 2. 해당 폴더 내의 모든 mp4 파일 목록 가져오기
    if not os.path.exists(target_video_dir):
        print(f"⚠️ 폴더가 존재하지 않습니다: {target_video_dir}")
    else:
        video_files = sorted([f for f in os.listdir(target_video_dir) if f.endswith('.mp4')])

        if not video_files:
            print(f"⚠️ {mode_name} 폴더에 분석할 mp4 파일이 없습니다.")
        else:
            print(f"\n📂 [{mode_name}] 분석할 영상을 선택하세요:")
            for i, f in enumerate(video_files):
                print(f"[{i}] {f}")
            
            try:
                idx_input = input(f"\n번호 입력 (0~{len(video_files)-1}): ").strip()
                if idx_input.isdigit():
                    idx = int(idx_input)
                    if 0 <= idx < len(video_files):
                        selected_video = video_files[idx]
                        full_video_path = os.path.join(target_video_dir, selected_video)
                        
                        print(f"🚀 선택됨: {selected_video} (감도: {sensitivity_val})")
                        extract_subtitle_frames(full_video_path, target_output_dir, sensitivity=sensitivity_val)
                    else:
                        print("❌ 잘못된 번호입니다.")
                else:
                    print("❌ 숫자를 입력해주세요.")
            except Exception as e:
                print(f"❌ 오류 발생: {e}")