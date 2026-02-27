import yt_dlp
import re
import csv
import os
import datetime
from unicodedata import normalize
from config import PATHS, CSV_FILE

# 1. 차단 패턴 설정
SPAM_PATTERNS = [
    r'무\s?료\s?증\s?정', r'카\s?톡\s?문\s?의', r'수\s?익\s?률\s?\d+%', 
    r'선\s?착\s?순\s?\d+명', r'텔\s?레\s?그\s?램', r'bit\.ly/[\w\d]+', 
    r'공\s?식\s?리\s?방', r'최\s?저\s?가\s?보\s?장',
    r'근\s?황', r'망\s?함', r'위\s?험', r'논\s?란', r'현\s?실'
]

def clean_text(text):
    """분석을 위한 텍스트 정제"""
    if not text: return ""
    text = normalize('NFC', text)
    # 패턴 매칭을 위해 최소한의 기호(./)와 공백은 유지
    text = re.sub(r'[^가-힣a-zA-Z0-9\./\s]', '', text).lower()
    return text

def fetch_metadata(url):
    """yt-dlp 메타데이터 추출"""
    ydl_opts = {
        'quiet': True, 
        'no_warnings': True, 
        'simulate': True, 
        'dump_single_json': True
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            return ydl.extract_info(url, download=False)
    except Exception as e:
        print(f"⚠️ 메타데이터 추출 실패: {e}")
        return None

def save_to_csv(data):
    """필터링 결과를 CSV에 저장 (config.CSV_FILE 사용)"""
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, mode='a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=["datetime", "url", "status", "reason", "title", "uploader"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

# 메인 파일(main.py)에서 호출할 이름으로 변경: stage_1_filter -> run_filter
def run_filter(url):
    """1단계: 키워드 기반 필터링"""
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    info = fetch_metadata(url)
    if not info:
        res = {"datetime": now, "url": url, "status": "FAIL", "reason": "데이터 추출 불가", "title": "-", "uploader": "-"}
        save_to_csv(res)
        return False, res

    title = info.get('title', '')
    description = info.get('description', '')
    uploader = info.get('uploader', 'Unknown')
    
    raw_content = f"{title} {description}"
    cleaned_content = clean_text(raw_content)

    for pattern in SPAM_PATTERNS:
        if re.search(pattern, cleaned_content) or re.search(pattern, raw_content):
            res = {
                "datetime": now, "url": url, "status": "BLOCKED", 
                "reason": f"패턴 감지: {pattern}", "title": title, "uploader": uploader
            }
            save_to_csv(res)
            return False, res

    res = {"datetime": now, "url": url, "status": "PASSED", "reason": "1단계 통과", "title": title, "uploader": uploader}
    save_to_csv(res)
    return True, res

if __name__ == "__main__":
    # 단독 실행 시 테스트용
    print("\n🚀 유튜브 필터링 시스템 (단독 실행 테스트)")
    url_input = input("🔗 분석할 URL 입력: ").strip()
    
    if url_input:
        is_passed, res = run_filter(url_input)
        print("\n" + "="*50)
        if is_passed:
            print(f"✅ [통과] {res['title']}")
        else:
            print(f"❌ [차단/오류] 사유: {res['reason']}")
        print("="*50)