from yt_dlp import YoutubeDL

url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
ydl_opts = {'format': 'best'}

with YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])  # 이 줄 앞에 스페이스 4칸이 꼭 있어야 해요!