import yt_dlp #한 채널에서 영상 긁어오기

url = "https://www.youtube.com/@시니어인사이트-h4y/shorts"

ydl_opts = {}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])


import glob
import yt_dlp

files = glob.glob("short_channel01/*.mp4")  # 폴더 안 모든 영상

ydl_opts = {
    "writeautomaticsub": True,
    "subtitleslangs": ["ko"],
    "skip_download": True,
    "outtmpl": "data/subtitles/%(id)s.%(ext)s"
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download(files)
