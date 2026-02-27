import yt_dlp
import os
import re
#import whisper
import csv
import torch
from datetime import datetime
from unicodedata import normalize
from moviepy.video.io.VideoFileClip import VideoFileClip
from config import PATHS 

# 모델은 모듈 로드 시 한 번만 초기화
device = "cuda" if torch.cuda.is_available() else "cpu"
stt_model = None # 지연 로딩을 위해 None으로 설정

def download_video(url):
    """영상 다운로드만 담당"""
    save_dir = PATHS["shorts"]
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            title = normalize('NFC', info.get('title', 'video'))
        
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.%(ext)s"
        ydl_opts = {
            'format': 'bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]',
            'outtmpl': os.path.join(save_dir, filename),
            'quiet': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_full = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(info_full)
            print(f"🎬 다운로드 완료: {os.path.basename(video_path)}")
            return video_path
    except Exception as e:
        print(f"❌ 다운로드 오류: {e}")
        return None


# whisper를 사용하여 소리 추출 부분

# def run_stt_and_save(video_path):
#     """STT 실행 및 각 비디오별 폴더에 CSV 저장"""
#     if not video_path or not os.path.exists(video_path):
#         return
    
#     # 1. 파일명 및 저장 경로 설정
#     # 확장자를 제외한 비디오 파일명 (예: 20240101_120000)
#     video_filename = os.path.splitext(os.path.basename(video_path))[0]
#     # 저장될 폴더 경로: D:\project\ytb\STT_result\비디오명
#     save_folder = os.path.join(PATHS["stt_results"], video_filename)
#     os.makedirs(save_folder, exist_ok=True) # 폴더가 없으면 생성
    
#     # CSV 파일 경로: D:\project\ytb\STT_result\비디오명\비디오명.csv
#     csv_path = os.path.join(save_folder, f"{video_filename}.csv")
    
#     audio_temp = os.path.join(save_folder, "temp_audio.mp3")
    
#     try:
#         # 1. 오디오 추출
#         video = VideoFileClip(video_path)
#         video.audio.write_audiofile(audio_temp, codec='libmp3lame', logger=None)
#         video.close()

#         # 2. STT 분석
#         model = get_model()
#         print(f"🎤 [{video_filename}] 분석 및 CSV 생성 중...")
        
#         result = model.transcribe(
#             audio_temp, 
#             fp16=False,
#             condition_on_previous_text=False,
#             no_speech_threshold=0.6,
#             logprob_threshold=-1.0
#         )
        
#         # 3. CSV 저장
#         segments = result.get('segments', [])
        
#         with open(csv_path, mode='w', newline='', encoding='utf-8-sig') as f:
#             # 개별 파일이므로 header는 항상 작성(mode='w')
#             writer = csv.DictWriter(f, fieldnames=["start_time", "stt_text"])
#             writer.writeheader()
            
#             if not segments and result['text'].strip():
#                 segments = [{'start': 0, 'text': result['text']}]

#             for seg in segments:
#                 clean_text = seg['text'].strip()
#                 if clean_text:
#                     writer.writerow({
#                         "start_time": datetime.utcfromtimestamp(seg['start']).strftime('%H:%M:%S'),
#                         "stt_text": clean_text
#                     })
#         print(f"💾 가사 저장 완료: {csv_path}")
        
#     except Exception as e:
#         print(f"❌ STT 오류 발생: {e}")
#     finally:
        if os.path.exists(audio_temp): 
            os.remove(audio_temp)