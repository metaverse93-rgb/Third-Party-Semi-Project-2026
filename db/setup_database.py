import sqlite3
import os

# DB 파일 경로
DB_PATH = os.path.join(os.path.dirname(__file__), 'spam_detector.db')

def create_database():
    """YouTube Spam Detector용 데이터베이스 생성"""
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("📁 YouTube Spam Detector DB 생성 중...")
    print(f"📍 위치: {DB_PATH}")
    
    # 기존 테이블 삭제 (초기화용)
    cursor.execute("DROP TABLE IF EXISTS report_urls")
    cursor.execute("DROP TABLE IF EXISTS video_meta")
    cursor.execute("DROP TABLE IF EXISTS analysis_results")
    cursor.execute("DROP TABLE IF EXISTS spam_patterns")
    
    # 1. URL 신고/검사 기록
    cursor.execute('''
    CREATE TABLE report_urls (
        report_id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT NOT NULL,
        ip_address TEXT,
        user_agent TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 2. YouTube 비디오 메타데이터
    cursor.execute('''
    CREATE TABLE video_meta (
        video_id INTEGER PRIMARY KEY AUTOINCREMENT,
        youtube_id TEXT UNIQUE NOT NULL,
        url TEXT NOT NULL,
        title TEXT,
        channel_name TEXT,
        duration INTEGER,
        view_count INTEGER,
        like_count INTEGER,
        comment_count INTEGER,
        upload_date TEXT,
        is_shorts INTEGER DEFAULT 0,
        fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 3. 스팸 분석 결과
    cursor.execute('''
    CREATE TABLE analysis_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id INTEGER NOT NULL,
        is_spam INTEGER CHECK (is_spam IN (0, 1)),
        spam_score REAL,
        spam_type TEXT,
        spam_reasons TEXT,
        ai_model_version TEXT,
        checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (video_id) REFERENCES video_meta(video_id)
    )
    ''')
    
    # 4. 스팸 패턴 (학습용)
    cursor.execute('''
    CREATE TABLE spam_patterns (
        pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern_type TEXT NOT NULL,
        pattern_value TEXT NOT NULL,
        weight REAL DEFAULT 1.0,
        hit_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # 5. Ground Truth (검증된 데이터)
    cursor.execute('''
    CREATE TABLE ground_truth_labels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id INTEGER NOT NULL,
        is_spam_verified INTEGER CHECK (is_spam_verified IN (0, 1)),
        is_shorts_verified INTEGER CHECK (is_shorts_verified IN (0, 1)),
        verified_by TEXT,
        verification_notes TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (video_id) REFERENCES video_meta(video_id)
    )
    ''')
    
    # 인덱스 생성 (검색 성능 향상)
    cursor.execute('CREATE INDEX idx_youtube_id ON video_meta(youtube_id)')
    cursor.execute('CREATE INDEX idx_report_created ON report_urls(created_at)')
    cursor.execute('CREATE INDEX idx_spam_score ON analysis_results(spam_score)')
    
    conn.commit()
    
    # 초기 스팸 패턴 삽입
    spam_patterns = [
        ('title_keyword', '무료', 2.0),
        ('title_keyword', '돈버는', 2.5),
        ('title_keyword', '클릭', 1.5),
        ('description_pattern', 'bit.ly', 3.0),
        ('description_pattern', '텔레그램', 2.0),
        ('channel_pattern', '구독자 없음', 1.5),
    ]
    
    cursor.executemany(
        'INSERT INTO spam_patterns (pattern_type, pattern_value, weight) VALUES (?, ?, ?)',
        spam_patterns
    )
    
    conn.commit()
    conn.close()
    
    print("✅ 데이터베이스 생성 완료!")
    print("✅ 초기 스팸 패턴 등록 완료!")

if __name__ == "__main__":
    create_database()