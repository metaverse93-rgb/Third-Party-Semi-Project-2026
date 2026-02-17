import yt_dlp

url = "https://www.youtube.com/shorts/LwQRUFyDKNU"

ydl_opts = {
    'quiet': True,
    'skip_download': True
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)

print("제목:", info.get("title"))
print("조회수:", info.get("view_count"))
print("업로더:", info.get("uploader"))
print("길이(초):", info.get("duration"))
print("설명:", info.get("description"))