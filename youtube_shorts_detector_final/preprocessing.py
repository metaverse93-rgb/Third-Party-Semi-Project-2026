"""
Phase 1: 전처리 모듈
- yt-dlp: 메타데이터 수집 + 영상 다운로드
- OpenCV: 자막 변동 감지 기반 프레임 추출
  - 35~85% 영역만 비교 (상단/하단 고정 UI 제외)
  - 자막 변동 시 저장, 중복 프레임 제거
  - 최대 10장, 첫 프레임 무조건 포함

디렉토리 구조:
  downloads/
    videos/{video_id}.mp4
    frames/{video_id}/frame_001.jpg ~ frame_010.jpg
"""
import cv2
import numpy as np
import os
import base64
import random
import time
import logging
from typing import List, Dict, Any, Optional, Tuple

from config import ROI_CONFIG, MOCK_MODE
from models import VideoMetadata, PreprocessingResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ 디렉토리 설정
BASE_DIR = "downloads"
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
FRAME_DIR = os.path.join(BASE_DIR, "frames")

# ✅ 프레임 추출 설정
MAX_FRAMES = 15          # 최대 저장 프레임 수
ROI_TOP = 0.35           # 자막 감지 영역 상단 (35%)
ROI_BOTTOM = 0.85        # 자막 감지 영역 하단 (85%)
SAMPLE_INTERVAL = 0.5    # 샘플링 간격 (초) - 0.5초마다 비교
CHANGE_THRESHOLD = 2.0   # 자막 변동 감지 임계값 (샘플 프레임 간 비교)
DUPLICATE_THRESHOLD = 5.0   # 중복 프레임 판단 임계값


class VideoPreprocessor:
    """영상 전처리 클래스"""

    def __init__(self):
        os.makedirs(VIDEO_DIR, exist_ok=True)
        os.makedirs(FRAME_DIR, exist_ok=True)
        logger.info(f"VideoPreprocessor 초기화 (Mock: {MOCK_MODE})")
        logger.info(f"📁 저장 경로: {os.path.abspath(BASE_DIR)}")

    # =============================================
    # 1. 메타데이터 수집
    # =============================================

    def parse_metadata(self, video_url: str) -> VideoMetadata:
        """yt-dlp로 실제 메타데이터 수집"""
        if MOCK_MODE:
            return self._mock_parse_metadata(video_url)

        try:
            import yt_dlp
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"🔍 메타데이터 수집 중: {video_url}")
                info = ydl.extract_info(video_url, download=False)
                return VideoMetadata(
                    video_id=info.get('id', ''),
                    title=info.get('title', ''),
                    description=(info.get('description', '') or '')[:500],
                    duration=info.get('duration', 0) or 0,
                    view_count=info.get('view_count', 0) or 0,
                    upload_date=info.get('upload_date', '') or '',
                    channel_name=info.get('uploader', '') or info.get('channel', ''),
                    thumbnail_url=info.get('thumbnail', '')
                )
        except Exception as e:
            logger.error(f"❌ 메타데이터 수집 실패: {e}")
            return self._mock_parse_metadata(video_url)

    def _mock_parse_metadata(self, video_url: str) -> VideoMetadata:
        """Mock 메타데이터 (테스트용)"""
        fake_data = [
            {
                "title": "🔥충격🔥 이것만 알면 100만원 번다!!",
                "description": "돈 버는 비법을 공개합니다.",
                "duration": 58, "view_count": 1250000, "channel": "돈버는법채널"
            },
            {
                "title": "Python 변수와 함수 완벽 마스터",
                "description": "초보자를 위한 파이썬 기초 강의입니다.",
                "duration": 55, "view_count": 45000, "channel": "코딩교육채널"
            },
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

    # =============================================
    # 2. 영상 다운로드
    # =============================================

    def download_video(self, video_url: str, video_id: str) -> Optional[str]:
        """
        yt-dlp로 영상 다운로드
        저장: downloads/videos/{video_id}.mp4
        이미 존재하면 재사용
        """
        if MOCK_MODE:
            logger.info("🎭 Mock 모드: 영상 다운로드 생략")
            return None

        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")

        if os.path.exists(video_path):
            logger.info(f"♻️ 기존 영상 재사용: {video_path}")
            return video_path

        try:
            import yt_dlp
            ydl_opts = {
                'quiet': False,
                'no_warnings': True,
                'format': 'best[ext=mp4]/best',
                'outtmpl': video_path,
            }
            logger.info(f"⬇️ 영상 다운로드 중: {video_url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            if os.path.exists(video_path):
                size_mb = os.path.getsize(video_path) / (1024 * 1024)
                logger.info(f"✅ 다운로드 완료: {video_path} ({size_mb:.1f}MB)")
                return video_path
            else:
                logger.error("❌ 다운로드 후 파일 없음")
                return None
        except Exception as e:
            logger.error(f"❌ 영상 다운로드 실패: {e}")
            return None

    # =============================================
    # 3. 자막 변동 감지 기반 프레임 추출
    # =============================================

    def _get_roi(self, frame: np.ndarray) -> np.ndarray:
        """
        35~85% 영역만 잘라서 반환
        상단(채널명/UI), 하단(좋아요/UI) 제외
        """
        h = frame.shape[0]
        top = int(h * ROI_TOP)
        bottom = int(h * ROI_BOTTOM)
        return frame[top:bottom, :]

    def _calc_diff(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """두 프레임의 ROI 영역 차이값 계산"""
        roi1 = self._get_roi(frame1)
        roi2 = self._get_roi(frame2)
        # 그레이스케일로 변환 후 차이 계산
        gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        return float(diff.mean())

    def _is_duplicate(self, new_frame: np.ndarray, saved_frames: List[np.ndarray]) -> bool:
        """
        저장된 프레임들과 중복 여부 확인
        가장 최근 저장 프레임과 비교
        """
        if not saved_frames:
            return False
        diff = self._calc_diff(new_frame, saved_frames[-1])
        return diff < DUPLICATE_THRESHOLD

    def extract_frames(self, video_path: str, video_id: str) -> List[str]:
        """
        자막 변동 감지 기반 프레임 추출
        
        로직:
          1. 매 프레임 순회하며 이전 프레임과 35~85% 영역 비교
          2. 변동값 > CHANGE_THRESHOLD → 자막 변동으로 판단
          3. 저장된 마지막 프레임과 중복 체크 → 중복이면 스킵
          4. 첫 프레임은 무조건 저장 (시작 시점 파악용)
          5. 최대 MAX_FRAMES(10)장까지 저장

        저장: downloads/frames/{video_id}/frame_001.jpg ~
        반환: base64 인코딩 리스트 (gpt-4o-mini 전송용)
        """
        frame_save_dir = os.path.join(FRAME_DIR, video_id)
        os.makedirs(frame_save_dir, exist_ok=True)

        if not video_path or not os.path.exists(video_path):
            logger.warning("⚠️ 영상 파일 없음")
            return []

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"❌ 영상 열기 실패: {video_path}")
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0

            logger.info(
                f"🎬 영상 정보: {total_frames}프레임 / "
                f"{fps:.1f}fps / {duration:.1f}초"
            )
            logger.info(
                f"🔍 자막 감지 영역: {int(ROI_TOP*100)}~{int(ROI_BOTTOM*100)}% "
                f"(변동 임계값: {CHANGE_THRESHOLD}, 중복 임계값: {DUPLICATE_THRESHOLD})"
            )

            saved_frames_raw = []   # 중복 체크용 원본 프레임
            saved_frames_b64 = []   # 반환용 base64 프레임
            saved_count = 0
            prev_frame = None

            # ✅ 0.5초 간격으로 샘플 프레임 인덱스 생성
            sample_interval_frames = max(1, int(fps * SAMPLE_INTERVAL))
            sample_indices = set(range(0, total_frames, sample_interval_frames))
            sample_indices.add(0)  # 첫 프레임 보장

            logger.info(
                f"🔍 샘플링 간격: {SAMPLE_INTERVAL}초 ({sample_interval_frames}프레임마다) "
                f"→ 총 {len(sample_indices)}개 후보"
            )

            for frame_idx in range(total_frames):
                if saved_count >= MAX_FRAMES:
                    logger.info(f"✅ 최대 프레임 수 도달 ({MAX_FRAMES}장), 추출 종료")
                    break

                # 샘플링 간격에 해당하는 프레임만 처리
                if frame_idx not in sample_indices:
                    cap.read()  # 버퍼 소비
                    continue

                ret, frame = cap.read()
                if not ret:
                    break

                # ✅ 첫 프레임 무조건 저장
                if frame_idx == 0:
                    self._save_frame(
                        frame, frame_save_dir, saved_count,
                        frame_idx, fps, "첫 프레임 (무조건 포함)"
                    )
                    b64 = self._to_base64(frame)
                    saved_frames_raw.append(frame.copy())
                    saved_frames_b64.append(b64)
                    saved_count += 1
                    prev_frame = frame.copy()
                    continue

                # ✅ 마지막 저장 프레임과 자막 변동 비교 (35~85% 영역)
                diff_score = self._calc_diff(frame, prev_frame)

                if diff_score > CHANGE_THRESHOLD:
                    # ✅ 중복 프레임 체크
                    if self._is_duplicate(frame, saved_frames_raw):
                        logger.debug(f"  🔄 프레임 {frame_idx}: 중복 감지 (diff={diff_score:.2f}), 스킵")
                    else:
                        self._save_frame(
                            frame, frame_save_dir, saved_count,
                            frame_idx, fps,
                            f"자막 변동 감지 (diff={diff_score:.2f})"
                        )
                        b64 = self._to_base64(frame)
                        saved_frames_raw.append(frame.copy())
                        saved_frames_b64.append(b64)
                        saved_count += 1
                        # ✅ 저장된 프레임 기준으로 갱신
                        prev_frame = frame.copy()

            cap.release()

            logger.info(
                f"🎉 프레임 추출 완료: {saved_count}장 "
                f"→ {frame_save_dir}"
            )
            return saved_frames_b64

        except Exception as e:
            logger.error(f"❌ 프레임 추출 실패: {e}")
            return []

    def _save_frame(
        self,
        frame: np.ndarray,
        save_dir: str,
        saved_count: int,
        frame_idx: int,
        fps: float,
        reason: str
    ) -> str:
        """프레임 파일 저장 및 로그 출력"""
        filename = f"frame_{saved_count + 1:03d}.jpg"
        filepath = os.path.join(save_dir, filename)
        cv2.imwrite(filepath, frame)

        timestamp = frame_idx / fps if fps > 0 else 0
        logger.info(
            f"  📸 [{saved_count + 1:02d}/{MAX_FRAMES}] "
            f"{filename} | {timestamp:.1f}초 | {reason}"
        )
        return filepath

    def _to_base64(self, frame: np.ndarray) -> str:
        """프레임을 base64 문자열로 변환 (gpt-4o-mini 전송용)"""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')

    # =============================================
    # 4. ROI / OCR (추후 구현)
    # =============================================

    def extract_roi(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        h, w = 100, 200
        return {name: np.zeros((h, w, 3), dtype=np.uint8) for name in ROI_CONFIG.keys()}

    def extract_text_ocr(self, roi_regions: Dict[str, np.ndarray]) -> str:
        return ""

    def calculate_layout_score(self, roi_regions: Dict[str, np.ndarray]) -> float:
        return round(random.uniform(0.5, 0.9), 2)

    # =============================================
    # 5. 전체 파이프라인
    # =============================================

    def process(self, video_url: str) -> PreprocessingResult:
        """전체 전처리 파이프라인"""
        start_time = time.time()
        processing_log = []

        logger.info(f"🚀 전처리 시작: {video_url}")
        processing_log.append(f"전처리 시작: {video_url}")

        # Step 1. 메타데이터 수집
        metadata = self.parse_metadata(video_url)
        processing_log.append(f"✅ 메타데이터: {metadata.title}")
        logger.info(f"📋 제목: {metadata.title} | 채널: {metadata.channel_name}")

        # Step 2. 영상 다운로드
        video_path = self.download_video(video_url, metadata.video_id)
        if video_path:
            processing_log.append(f"✅ 다운로드: downloads/videos/{metadata.video_id}.mp4")
        else:
            processing_log.append("⚠️ 영상 다운로드 없음 (Mock 또는 실패)")

        # Step 3. 자막 변동 감지 프레임 추출
        keyframes = self.extract_frames(video_path, metadata.video_id)
        processing_log.append(
            f"✅ 프레임 추출: {len(keyframes)}장 "
            f"→ downloads/frames/{metadata.video_id}/"
        )

        # Step 4. ROI / OCR
        mock_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        roi_regions = self.extract_roi(mock_frame)
        ocr_text = self.extract_text_ocr(roi_regions)
        layout_score = self.calculate_layout_score(roi_regions)

        processing_time = time.time() - start_time
        processing_log.append(f"⏱️ 전처리 완료: {processing_time:.2f}초")
        logger.info(f"✅ 전처리 완료: {processing_time:.2f}초")

        return PreprocessingResult(
            video_metadata=metadata,
            keyframes=keyframes,
            ocr_text=ocr_text,
            layout_score=layout_score,
            roi_data={
                "video_path": video_path or "",
                "frame_dir": os.path.join(FRAME_DIR, metadata.video_id),
                "frame_count": len(keyframes),
                "regions": list(roi_regions.keys()),
                "roi_range": f"{int(ROI_TOP*100)}~{int(ROI_BOTTOM*100)}%",
            },
            processing_log=processing_log
        )
