# winget install OpenJS.NodeJS.LTS
# pip install yt_dlp

import yt_dlp

save_path = 'D:/project/shorts/%(title)s.%(ext)s'

ydl_opts = {
    'format': 'best[ext=mp4]/worst',
    'outtmpl': save_path,
    # 1. 런타임 설정
    'js_runtimes': {'node': {}},
    # 2. 형식을 리스트로 감싸서 전달 (문자열 분해 방지)
    'remote_components': ['ejs:github'], 
}


shorts_url = input("영상 링크 입력:")
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([shorts_url])
