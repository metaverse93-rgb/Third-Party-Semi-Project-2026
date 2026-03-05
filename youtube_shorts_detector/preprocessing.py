"""
Phase 1: 전처리 모듈
OpenCV ROI 추출, OCR, 메타데이터 파싱 (Mock 구현)
"""
import cv2
import numpy as np
import re
import random
import time
from typing import List, Tuple, Dict, Any
import logging

from config import ROI_CONFIG, MOCK_MODE
from models import VideoMetadata, PreprocessingResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoPreprocessor:
    """영상 전처리 클래스"""
    
    def __init__(self):
        logger.info(f"VideoPreprocessor 초기화 (Mock 모드: {MOCK_MODE})")
    
    def extract_roi(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """ROI(Region of Interest) 추출"""
        if MOCK_MODE:
            return self._mock_extract_roi(frame)
        
        height, width = frame.shape[:2]
        roi_regions = {}
        
        for region_name, (x1, y1, x2, y2) in ROI_CONFIG.items():
            # 백분율을 픽셀로 변환
            x1_px = int(x1 * width / 100)
            y1_px = int(y1 * height / 100)
            x2_px = int(x2 * width / 100)
            y2_px = int(y2 * height / 100)
            
            roi_regions[region_name] = frame[y1_px:y2_px, x1_px:x2_px]
        
        return roi_regions
    
    def _mock_extract_roi(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Mock ROI 추출"""
        logger.info("🎭 Mock ROI 추출 수행")
        
        # 가짜 ROI 영역 생성
        h, w = 100, 200  # 임의 크기
        mock_regions = {}
        
        for region_name in ROI_CONFIG.keys():
            # 색상이 다른 가짜 영역 생성
            color = (random.randint(50, 200), random.randint(50, 200), random.randint(50, 200))
            mock_regions[region_name] = np.full((h, w, 3), color, dtype=np.uint8)
        
        return mock_regions
    
    def extract_text_ocr(self, roi_regions: Dict[str, np.ndarray]) -> str:
        """OCR을 통한 텍스트 추출"""
        if MOCK_MODE:
            return self._mock_extract_text(roi_regions)
        
        # 실제 Tesseract OCR 구현 (여기서는 placeholder)
        try:
            import pytesseract
            
            extracted_texts = []
            for region_name, region in roi_regions.items():
                text = pytesseract.image_to_string(region, lang='kor+eng')
                if text.strip():
                    extracted_texts.append(f"[{region_name}] {text.strip()}")
            
            return " ".join(extracted_texts)
            
        except ImportError:
            logger.warning("pytesseract 미설치, Mock 모드로 전환")
            return self._mock_extract_text(roi_regions)
    
    def _mock_extract_text(self, roi_regions: Dict[str, np.ndarray]) -> str:
        """Mock OCR 텍스트 추출"""
        logger.info("🎭 Mock OCR 텍스트 추출 수행")
        
        # 영역별 가짜 텍스트 생성
        mock_texts = {
            "title_region": random.choice([
                "🔥충격🔥 이것만 알면 100만원",
                "Python 기초 강의 완전 정복",
                "TTS로 만든 반복 콘텐츠",
                "맛집 리뷰 솔직 후기",
                "무단 복사된 영상 의심"
            ]),
            "content_region": random.choice([
                "돈버는법 클릭 지금바로 수익창출",
                "변수 선언 print 함수 설명",
                "템플릿 자동생성 따라하기",
            ]),
            "ui_region": "좋아요 구독 알림 설정"
        }
        
        extracted_text = ""
        for region_name in roi_regions.keys():
            if region_name in mock_texts:
                extracted_text += f"[{region_name}] {mock_texts[region_name]} "
        
        return extracted_text.strip()
    
    def calculate_layout_score(self, roi_regions: Dict[str, np.ndarray]) -> float:
        """레이아웃 품질 점수 계산"""
        if MOCK_MODE:
            return self._mock_calculate_layout_score()
        
        # 실제 레이아웃 분석 로직
        scores = []
        
        for region_name, region in roi_regions.items():
            # 텍스트 밀도, 색상 분포, 구조적 정보 등 분석
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            
            # 예시: 텍스트 영역 비율
            edges = cv2.Canny(gray, 50, 150)
            text_ratio = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            scores.append(text_ratio)
        
        return np.mean(scores) if scores else 0.0
    
    def _mock_calculate_layout_score(self) -> float:
        """Mock 레이아웃 점수 계산"""
        logger.info("🎭 Mock 레이아웃 점수 계산")
        
        # 0.0 ~ 1.0 사이의 랜덤 점수 생성
        score = random.uniform(0.2, 0.95)
        return round(score, 2)
    
    def parse_metadata(self, video_url: str) -> VideoMetadata:
        """유튜브 메타데이터 파싱"""
        if MOCK_MODE:
            return self._mock_parse_metadata(video_url)
        
        # 실제 yt-dlp 구현
        try:
            import yt_dlp
            
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(video_url, download=False)
                
                return VideoMetadata(
                    video_id=info.get('id', ''),
                    title=info.get('title', ''),
                    description=info.get('description', ''),
                    duration=info.get('duration', 0),
                    view_count=info.get('view_count', 0),
                    upload_date=info.get('upload_date', ''),
                    channel_name=info.get('uploader', ''),
                    thumbnail_url=info.get('thumbnail', '')
                )
                
        except ImportError:
            logger.warning("yt-dlp 미설치, Mock 모드로 전환")
            return self._mock_parse_metadata(video_url)
    
    def _mock_parse_metadata(self, video_url: str) -> VideoMetadata:
        """Mock 메타데이터 생성"""
        logger.info(f"🎭 Mock 메타데이터 파싱: {video_url}")
        
        # 가짜 메타데이터 생성
        fake_data = [
            {
                "title": "🔥충격🔥 이것만 알면 100만원 번다!! (진짜임)",
                "description": "돈 버는 비법을 공개합니다",
                "duration": 58,
                "view_count": 1250000,
                "channel": "돈버는법알려주는채널"
            },
            {
                "title": "Python 변수와 함수 완벽 마스터",
                "description": "초보자를 위한 파이썬 기초 강의",
                "duration": 1200,
                "view_count": 45000,
                "channel": "코딩교육채널"
            },
            {
                "title": "TTS로 만든 반복 템플릿 영상",
                "description": "자동 생성된 콘텐츠",
                "duration": 35,
                "view_count": 500,
                "channel": "자동생성채널"
            }
        ]
        
        selected = random.choice(fake_data)
        video_id = video_url.split('/')[-1] if '/' in video_url else f"mock_{random.randint(1000, 9999)}"
        
        return VideoMetadata(
            video_id=video_id,
            title=selected["title"],
            description=selected["description"],
            duration=selected["duration"],
            view_count=selected["view_count"],
            upload_date="2024-01-15",
            channel_name=selected["channel"],
            thumbnail_url=f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        )
    
    def process(self, video_url: str) -> PreprocessingResult:
        """전체 전처리 파이프라인 실행"""
        start_time = time.time()
        processing_log = []
        
        logger.info(f"전처리 시작: {video_url}")
        processing_log.append(f"전처리 시작: {video_url}")
        
        # 1. 메타데이터 파싱
        metadata = self.parse_metadata(video_url)
        processing_log.append(f"메타데이터 추출 완료: {metadata.title}")
        
        # 2. 키프레임 추출 (Mock)
        if MOCK_MODE:
            keyframes = [f"frame_{i}.jpg" for i in range(1, 4)]
            processing_log.append("Mock 키프레임 생성 완료")
        else:
            # 실제 키프레임 추출 로직
            keyframes = ["frame_1.jpg", "frame_2.jpg", "frame_3.jpg"]
        
        # 3. ROI 추출 (첫 번째 프레임 기준)
        mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)  # 가짜 프레임
        roi_regions = self.extract_roi(mock_frame)
        processing_log.append(f"ROI 추출 완료: {len(roi_regions)}개 영역")
        
        # 4. OCR 텍스트 추출
        ocr_text = self.extract_text_ocr(roi_regions)
        processing_log.append(f"OCR 완료: {len(ocr_text)}자 추출")
        
        # 5. 레이아웃 점수 계산
        layout_score = self.calculate_layout_score(roi_regions)
        processing_log.append(f"레이아웃 점수: {layout_score}")
        
        processing_time = time.time() - start_time
        logger.info(f"전처리 완료: {processing_time:.2f}초")
        
        return PreprocessingResult(
            video_metadata=metadata,
            keyframes=keyframes,
            ocr_text=ocr_text,
            layout_score=layout_score,
            roi_data={"regions": list(roi_regions.keys())},
            processing_log=processing_log
        )