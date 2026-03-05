import sqlite3
import os
import json
from datetime import datetime
from contextlib import contextmanager
from urllib.parse import urlparse, parse_qs

class YouTubeSpamDB:
    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(__file__), 'spam_detector.db')
    
    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def extract_youtube_id(self, url):
        """YouTube URL에서 video ID 추출"""
        parsed = urlparse(url)
        
        # youtube.com/watch?v=VIDEO_ID
        if 'youtube.com' in parsed.netloc and 'v' in parse_qs(parsed.query):
            return parse_qs(parsed.query)['v'][0]
        
        # youtu.be/VIDEO_ID
        elif 'youtu.be' in parsed.netloc:
            return parsed.path.strip('/')
        
        # youtube.com/shorts/VIDEO_ID
        elif 'youtube.com' in parsed.netloc and '/shorts/' in parsed.path:
            return parsed.path.split('/shorts/')[1].split('/')[0]
        
        return None
    
    def add_report(self, url, ip_address=None, user_agent=None):
        """URL 검사 요청 기록"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO report_urls (url, ip_address, user_agent) 
                   VALUES (?, ?, ?)""",
                (url, ip_address, user_agent)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_or_create_video(self, url, video_data=None):
        """비디오 정보 조회 또는 생성"""
        youtube_id = self.extract_youtube_id(url)
        if not youtube_id:
            return None
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 기존 데이터 확인
            cursor.execute(
                "SELECT * FROM video_meta WHERE youtube_id = ?", 
                (youtube_id,)
            )
            result = cursor.fetchone()
            
            if result:
                return dict(result)
            
            # 새로 생성
            if video_data:
                is_shorts = 1 if 'shorts' in url.lower() else 0
                
                cursor.execute(
                    """INSERT INTO video_meta 
                       (youtube_id, url, title, channel_name, duration, 
                        view_count, like_count, comment_count, upload_date, is_shorts) 
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (youtube_id, url, 
                     video_data.get('title'),
                     video_data.get('channel_name'),
                     video_data.get('duration'),
                     video_data.get('view_count'),
                     video_data.get('like_count'),
                     video_data.get('comment_count'),
                     video_data.get('upload_date'),
                     is_shorts)
                )
                conn.commit()
                
                return {
                    'video_id': cursor.lastrowid,
                    'youtube_id': youtube_id,
                    'url': url,
                    'is_shorts': is_shorts
                }
            
            return None
    
    def add_spam_analysis(self, video_id, is_spam, spam_score, spam_reasons=None, model_version='v1.0'):
        """스팸 분석 결과 저장"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            spam_reasons_json = json.dumps(spam_reasons) if spam_reasons else None
            
            cursor.execute(
                """INSERT INTO analysis_results 
                   (video_id, is_spam, spam_score, spam_reasons, ai_model_version) 
                   VALUES (?, ?, ?, ?, ?)""",
                (video_id, is_spam, spam_score, spam_reasons_json, model_version)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_spam_patterns(self):
        """스팸 패턴 조회"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM spam_patterns ORDER BY weight DESC")
            return [dict(row) for row in cursor.fetchall()]
    
    def update_pattern_hit(self, pattern_id):
        """패턴 적중 횟수 증가"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE spam_patterns SET hit_count = hit_count + 1 WHERE pattern_id = ?",
                (pattern_id,)
            )
            conn.commit()
    
    def get_recent_analyses(self, limit=10):
        """최근 분석 결과 조회"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    vm.video_id,
                    vm.title,
                    vm.channel_name,
                    vm.youtube_id,
                    vm.url,
                    ar.is_spam,
                    ar.spam_score,
                    ar.spam_type,
                    ar.spam_reasons,
                    ar.checked_at
                FROM analysis_results ar
                JOIN video_meta vm ON ar.video_id = vm.video_id
                ORDER BY ar.checked_at DESC
                LIMIT ?
            ''', (limit,))
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                # JSON 문자열을 파싱
                if result.get('spam_reasons'):
                    try:
                        result['spam_reasons'] = json.loads(result['spam_reasons'])
                    except:
                        pass
                results.append(result)
            
            return results
    
    def get_statistics(self):
        """통계 정보 조회"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # 전체 검사 수
            cursor.execute("SELECT COUNT(*) as total FROM report_urls")
            stats['total_checks'] = cursor.fetchone()['total']
            
            # 스팸 비율 (analysis_results 테이블 기준)
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN is_spam = 1 THEN 1 ELSE 0 END) as spam_count
                FROM analysis_results
            """)
            result = cursor.fetchone()
            stats['total_analyses'] = result['total']
            stats['spam_count'] = result['spam_count']
            stats['spam_rate'] = (result['spam_count'] / result['total'] * 100) if result['total'] > 0 else 0
            
            # Shorts 비율
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(is_shorts) as shorts_count
                FROM video_meta
            """)
            result = cursor.fetchone()
            stats['total_videos'] = result['total']
            stats['shorts_count'] = result['shorts_count']
            stats['shorts_rate'] = (result['shorts_count'] / result['total'] * 100) if result['total'] > 0 else 0
            
            # 평균 스팸 점수
            cursor.execute('SELECT AVG(spam_score) FROM analysis_results')
            avg_spam_score = cursor.fetchone()[0] or 0
            stats['average_spam_score'] = round(avg_spam_score, 2)
            
            return stats

# 테스트
if __name__ == "__main__":
    db = YouTubeSpamDB()
    
    # 테스트 URL
    test_url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
    
    # 1. 신고 추가
    report_id = db.add_report(test_url, "127.0.0.1", "Mozilla/5.0")
    print(f"Report ID: {report_id}")
    
    # 2. 비디오 정보 추가
    video_info = db.get_or_create_video(test_url, {
        'title': '테스트 쇼츠 영상',
        'channel_name': '테스트 채널',
        'duration': 30,
        'view_count': 1000
    })
    print(f"Video Info: {video_info}")
    
    # 3. 스팸 분석 결과
    if video_info:
        analysis_id = db.add_spam_analysis(
            video_info['video_id'],
            is_spam=0,
            spam_score=0.15,
            spam_reasons=['깨끗한 콘텐츠']
        )
        print(f"Analysis ID: {analysis_id}")
    
    # 4. 통계
    stats = db.get_statistics()
    print(f"Statistics: {stats}")