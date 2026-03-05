# db/view_db.py
import sqlite3
import os
import json
from datetime import datetime
from tabulate import tabulate
import argparse

class DBViewer:
    def __init__(self):
        self.db_path = os.path.join(os.path.dirname(__file__), 'spam_detector.db')
        
    def view_recent_analyses(self, limit=10):
        """최근 분석 결과 보기"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                ar.id,
                vm.title,
                vm.channel_name,
                ar.is_spam,
                ar.spam_score,
                ar.spam_type,
                ar.checked_at
            FROM analysis_results ar
            JOIN video_meta vm ON ar.video_id = vm.video_id
            ORDER BY ar.checked_at DESC
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            print("분석 결과가 없습니다.")
            return
        
        # 데이터 정리
        data = []
        for row in results:
            data.append([
                row['id'],
                row['title'][:30] + '...' if len(row['title']) > 30 else row['title'],
                row['channel_name'][:20] + '...' if len(row['channel_name']) > 20 else row['channel_name'],
                '🚫 스팸' if row['is_spam'] else '✅ 정상',
                f"{row['spam_score']:.2f}",
                row['spam_type'] or '-',
                row['checked_at'][:19]  # 시간 부분만
            ])
        
        headers = ['ID', '제목', '채널', '판정', '점수', '유형', '검사일']
        print(f"\n=== 최근 분석 결과 (최근 {limit}개) ===")
        print(tabulate(data, headers=headers, tablefmt='grid'))
    
    def view_videos(self, limit=10):
        """비디오 목록 보기"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                video_id,
                title,
                channel_name,
                view_count,
                is_shorts,
                fetched_at
            FROM video_meta
            ORDER BY video_id DESC
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            print("저장된 비디오가 없습니다.")
            return
        
        data = []
        for row in results:
            data.append([
                row['video_id'],
                row['title'][:40] + '...' if len(row['title']) > 40 else row['title'],
                row['channel_name'][:20] + '...' if len(row['channel_name']) > 20 else row['channel_name'],
                f"{row['view_count']:,}" if row['view_count'] else '-',
                '📱 Shorts' if row['is_shorts'] else '📺 Video',
                row['fetched_at'][:10] if row['fetched_at'] else '-'
            ])
        
        headers = ['ID', '제목', '채널', '조회수', '유형', '수집일']
        print(f"\n=== 비디오 목록 (최근 {limit}개) ===")
        print(tabulate(data, headers=headers, tablefmt='grid'))
    
    def view_statistics(self):
        """통계 정보 보기"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 전체 통계
        stats = []
        
        # 1. 전체 비디오 수
        cursor.execute("SELECT COUNT(*) FROM video_meta")
        total_videos = cursor.fetchone()[0]
        
        # 2. 전체 분석 수
        cursor.execute("SELECT COUNT(*) FROM analysis_results")
        total_analyses = cursor.fetchone()[0]
        
        # 3. 스팸/정상 비율
        cursor.execute("""
            SELECT 
                is_spam, 
                COUNT(*) as count,
                AVG(spam_score) as avg_score
            FROM analysis_results 
            GROUP BY is_spam
        """)
        spam_stats = cursor.fetchall()
        
        # 4. Shorts vs 일반 비디오
        cursor.execute("""
            SELECT 
                is_shorts, 
                COUNT(*) as count 
            FROM video_meta 
            GROUP BY is_shorts
        """)
        video_type_stats = cursor.fetchall()
        
        conn.close()
        
        print("\n=== 전체 통계 ===")
        
        # 기본 통계
        basic_stats = [
            ['전체 비디오 수', f"{total_videos:,}개"],
            ['전체 분석 수', f"{total_analyses:,}개"],
            ['분석률', f"{(total_analyses/total_videos*100 if total_videos > 0 else 0):.1f}%"]
        ]
        print(tabulate(basic_stats, headers=['항목', '값'], tablefmt='grid'))
        
        # 스팸 통계
        print("\n=== 스팸 분석 통계 ===")
        spam_data = []
        for row in spam_stats:
            status = '스팸' if row[0] == 1 else '정상'
            spam_data.append([
                status,
                f"{row[1]:,}개",
                f"{(row[1]/total_analyses*100 if total_analyses > 0 else 0):.1f}%",
                f"{row[2]:.3f}"
            ])
        
        print(tabulate(spam_data, headers=['구분', '개수', '비율', '평균점수'], tablefmt='grid'))
        
        # 비디오 유형 통계
        print("\n=== 비디오 유형 통계 ===")
        type_data = []
        for row in video_type_stats:
            type_name = 'Shorts' if row[0] == 1 else '일반 비디오'
            type_data.append([
                type_name,
                f"{row[1]:,}개",
                f"{(row[1]/total_videos*100 if total_videos > 0 else 0):.1f}%"
            ])
        
        print(tabulate(type_data, headers=['유형', '개수', '비율'], tablefmt='grid'))
    
    def view_spam_patterns(self):
        """스팸 패턴 보기"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM spam_patterns
            ORDER BY hit_count DESC, weight DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            print("등록된 스팸 패턴이 없습니다.")
            return
        
        data = []
        for row in results:
            data.append([
                row['pattern_id'],
                row['pattern_name'],
                row['pattern_regex'][:40] + '...' if len(row['pattern_regex']) > 40 else row['pattern_regex'],
                row['category'],
                f"{row['weight']:.2f}",
                row['hit_count']
            ])
        
        headers = ['ID', '패턴명', '정규식', '카테고리', '가중치', '적중수']
        print("\n=== 스팸 패턴 목록 ===")
        print(tabulate(data, headers=headers, tablefmt='grid'))

def main():
    parser = argparse.ArgumentParser(description='YouTube 스팸 탐지기 DB 뷰어')
    parser.add_argument('command', choices=['analyses', 'videos', 'stats', 'patterns', 'all'],
                        help='보고 싶은 정보 선택')
    parser.add_argument('-n', '--limit', type=int, default=10,
                        help='표시할 항목 수 (기본값: 10)')
    
    args = parser.parse_args()
    viewer = DBViewer()
    
    if args.command == 'analyses':
        viewer.view_recent_analyses(args.limit)
    elif args.command == 'videos':
        viewer.view_videos(args.limit)
    elif args.command == 'stats':
        viewer.view_statistics()
    elif args.command == 'patterns':
        viewer.view_spam_patterns()
    elif args.command == 'all':
        viewer.view_recent_analyses(args.limit)
        viewer.view_videos(args.limit)
        viewer.view_statistics()
        viewer.view_spam_patterns()

if __name__ == "__main__":
    # 명령줄 인수가 없으면 기본 동작
    if len(os.sys.argv) == 1:
        viewer = DBViewer()
        viewer.view_recent_analyses(5)
        viewer.view_statistics()
    else:
        main()