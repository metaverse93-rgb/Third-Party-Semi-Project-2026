import yt_dlp
import os
import re
from datetime import datetime
from config import FULL_SHORTS_DIR, SHORTS_DIR

def download_video(url, save_dir):
    # 1. 먼저 영상 정보만 추출해서 제목 확인
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'video')
            
        # 2. 제목에 한글이 포함되어 있는지 체크
        has_korean = re.search('[가-힣]', title)
        
        if has_korean:
            # 한글이 있으면: "20240522_153022.mp4" 형식으로 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_format = f"{timestamp}.%(ext)s"
            print(f"📝 한글 제목 감지 -> 시간 기반 파일명으로 변경: {timestamp}")
        else:
            # 한글이 없으면: "English_Title.mp4" 형식으로 저장
            filename_format = "%(title)s.%(ext)s"

        # 3. 실제 다운로드 옵션 설정
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': f'{save_dir}/{filename_format}',
            'restrictfilenames': True,  # 영어 제목일 때 공백/특수문자 제거
            'windowsfilenames': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            print(f"\n✅ 다운로드 완료: {filename}")
            return filename

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        return None

if __name__ == "__main__":
    print("=== 유튜브 영상 다운로더 (한글 자동 변환) ===")
    print("1. 원본 영상 (full_shorts 폴더 저장)")
    print("2. 쇼츠 영상 (shorts 폴더 저장)")
    
    choice = input("선택 (1 또는 2): ").strip()
    video_url = input("유튜브 URL 입력: ").strip()

    if choice == '1':
        target_dir = FULL_SHORTS_DIR
        print(f"📂 원본 모드: {target_dir}")
    elif choice == '2':
        target_dir = SHORTS_DIR
        print(f"📂 쇼츠 모드: {target_dir}")
    else:
        print("⚠️ 잘못된 선택입니다.")
        exit()

    download_video(video_url, target_dir)