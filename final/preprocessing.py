"""
Phase 1: 전처리 모듈 (v6.2) — 자막 유무 자동 감지 + 미달 시 보충
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[v6.2 핵심 변경 사항 — 3가지 문제 해결]

  ① 자막 없는 정상 영상 처리 (_detect_has_subtitles)
     - 영상 전체에서 20장을 균등 샘플링하여 자막 존재 비율 측정
     - 25% 이상이 자막 있음 → "자막형" 모드  (자막 필터 적용)
     - 25% 미만           → "무자막형" 모드  (자막 필터 건너뜀, diff만 사용)

  ② 자막 없는 영상의 15장 추출
     - 무자막형: 자막 필터 없이 diff 기반 장면 전환 + pHash 중복 제거로 15장 추출
     - 미달 시: 전체 영상을 target_frames 등분하여 균등 보충

  ③ 자막 있는 영상 15장 미달 시 보충
     - 기존 선택 프레임 사이 구간을 세밀하게 재스캔 (DIFF_STEP//2 간격)
     - 자막 있는 프레임 + pHash 미중복 프레임만 추가
     - 부족한 수만큼만 보충하여 최종 target_frames(≤15)장 확보

[파이프라인 흐름 v6.2]
  Step 0 : _detect_has_subtitles → 자막형 / 무자막형 판정
  Phase A : grab루프 → uint8 absdiff → 후보 리스트 (공통)
  Phase B : [자막형]   자막 필터 → pHash 중복 제거
            [무자막형] pHash 중복 제거만
  Step C  : 15장 미달 시 보충
            [자막형]   기존 프레임 사이 구간 재스캔 → 자막 있는 프레임 추가
            [무자막형] 균등 분포로 장면 추가
  저장    : 병렬 JPEG 저장 → LMM(Phase 2)으로 전달

[역할 분담 — 기획서 "라. OpenCV vs LMM"]
  OpenCV (본 모듈): 픽셀 변화 감지, pHash 중복 제거, 키프레임 추출
  LMM (Phase 2)   : 키프레임을 보고 자막 텍스트 해석 + 맥락 분석

[파일명]  frame_{번호:05d}_{HHhMMmSSsmmm}.jpg
[구조]
  downloads/
    videos/{video_id}.mp4
    frames/{video_id}/frame_*.jpg
    frames/{video_id}/_contact_sheet.jpg
  previews/{video_id}_preview.png
"""
import cv2
import numpy as np
import os
import base64
import random
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from config import MOCK_MODE
from models import VideoMetadata, PreprocessingResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 디렉토리 ──────────────────────────────────────────────────────────────────
BASE_DIR  = "downloads"
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
FRAME_DIR = os.path.join(BASE_DIR, "frames")

# ── 프레임 추출 — 기획서 Nf/Ncap 설계 근거 반영 ─────────────────────────────
Nf           = 25    # 쇼츠 정보 밀도 상한선 (60s ÷ 2.4s/컷 ≈ 25)
Ncap         = 15    # API 비용·정확도 최적 배치 크기
MAX_FRAMES   = Ncap  # N_final = min(Nf, Ncap) → 15
MIN_FRAMES   = 5

CHANGE_THRESHOLD = 6.0   # ROI 픽셀 변화 감지 임계값

# ── 자막 ROI (하단 고정) ─────────────────────────────────────────────────────
SUBTITLE_TOP    = 0.60
SUBTITLE_BOTTOM = 0.92

# ── 블랙 프레임 필터 ─────────────────────────────────────────────────────────
BLACK_FRAME_THRESHOLD = 8

# ── pHash 설정 — 기획서 "유사도 기반 중복 제거 (Nf)" ────────────────────────
PHASH_SIZE          = 8    # pHash 해시 크기 (8×8 = 64비트)
PHASH_HAMMING_MAX   = 8    # 해밍 거리 ≤ 이 값이면 "동일 자막" 처리 (중복 제거)

# ── 자막 존재 필터 설정 (v6.1 신규) ──────────────────────────────────────────
# 신호 A: 엣지 밀도 — 텍스트 윤곽선은 Canny 엣지 비율이 높음
SUBTITLE_EDGE_DENSITY_MIN   = 0.06   # ROI 픽셀 중 엣지 비율 하한 (6%)
# 신호 B: 고대비 픽셀 비율 — 자막은 밝은(>200) + 어두운(<55) 픽셀이 공존
SUBTITLE_BRIGHT_RATIO_MIN   = 0.08   # 밝은 픽셀 비율 하한 (8%)
SUBTITLE_DARK_RATIO_MIN     = 0.08   # 어두운 픽셀 비율 하한 (8%)
# 신호 C: 수평 연속 런 — 자막 획(stroke)은 수평으로 길게 이어짐
SUBTITLE_HRUN_DENSITY_MIN   = 0.04   # 길이≥5인 수평 런의 픽셀 비율 하한 (4%)
# 3개 신호 중 몇 개 이상 통과해야 "자막 있음"으로 판정
SUBTITLE_SIGNAL_MIN_PASS    = 2      # 2/3 다수결

# ── v6.1 병렬 저장 설정 ───────────────────────────────────────────────────────
PARALLEL_SAVE_WORKERS = 4

# ── 저장 포맷 ─────────────────────────────────────────────────────────────────
FRAME_EXT          = ".jpg"
FRAME_JPEG_QUALITY = 92


@dataclass
class KeyframeEvent:
    """OCR 없는 키프레임 이벤트 (LMM이 텍스트 해석 담당)"""
    frame_index  : int
    timestamp_sec: float
    timestamp_str: str
    diff_score   : float
    phash        : int     = 0    # pHash 값 (중복 제거에 사용)
    image_path   : str     = ""


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────────

def _sec_to_hhmmss(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _compute_phash(frame: np.ndarray, hash_size: int = PHASH_SIZE) -> int:
    """
    OpenCV만으로 구현한 pHash (Perceptual Hash).
    EasyOCR 없이 순수 numpy/cv2 연산으로 자막 유사도 측정.

    1. ROI 추출 → 그레이스케일
    2. hash_size×hash_size 로 리사이즈 (고주파 제거)
    3. DCT 적용 → 좌상단 (hash_size//4 × hash_size//4) 저주파 성분 추출
    4. 평균값 기준으로 비트 배열 생성 → 정수 해시로 변환
    """
    roi_y0 = int(frame.shape[0] * SUBTITLE_TOP)
    roi_y1 = int(frame.shape[0] * SUBTITLE_BOTTOM)
    roi = frame[roi_y0:roi_y1, :]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # DCT 입력: float32 필요
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4),
                         interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    # 저주파 성분만 사용
    dct_low = dct[:hash_size, :hash_size]
    avg = np.mean(dct_low)
    bits = (dct_low > avg).flatten()
    # 64비트 정수로 변환
    hash_val = 0
    for bit in bits:
        hash_val = (hash_val << 1) | int(bit)
    return hash_val


def _hamming_distance(h1: int, h2: int) -> int:
    """두 pHash 간 해밍 거리 계산"""
    return bin(h1 ^ h2).count('1')


def _has_subtitle_text(frame: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    """
    OCR 없이 OpenCV 신호 3가지로 자막 존재 여부를 판단.

    자막이 있는 프레임의 특성:
      A) 엣지 밀도 높음   — 글자 윤곽선이 Canny로 잘 검출됨
      B) 고대비 픽셀 공존 — 흰 글씨(밝음) + 검은 테두리(어두움) or 배경 대비
      C) 수평 연속 런 많음 — 글자 획(stroke)은 수평 방향으로 연속된 픽셀

    3개 신호 중 SUBTITLE_SIGNAL_MIN_PASS(=2)개 이상 통과 시 True 반환.

    Returns:
        (has_text: bool, signals: dict)  — signals는 디버그용 수치
    """
    vid_h  = frame.shape[0]
    roi_y0 = int(vid_h * SUBTITLE_TOP)
    roi_y1 = int(vid_h * SUBTITLE_BOTTOM)
    roi    = frame[roi_y0:roi_y1, :]
    gray   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    total  = gray.size
    signals = {}

    # ── 신호 A: 엣지 밀도 ────────────────────────────────────────────────────
    # CLAHE로 대비 강화 후 Canny 적용 (저해상도/흐린 자막도 검출)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray_eq   = clahe.apply(gray)
    edges     = cv2.Canny(gray_eq, 40, 120)
    edge_density = float(np.count_nonzero(edges)) / total
    signals["edge_density"] = edge_density
    pass_A = edge_density >= SUBTITLE_EDGE_DENSITY_MIN

    # ── 신호 B: 고대비 픽셀 공존 ─────────────────────────────────────────────
    bright_ratio = float(np.sum(gray > 200)) / total
    dark_ratio   = float(np.sum(gray < 55))  / total
    signals["bright_ratio"] = bright_ratio
    signals["dark_ratio"]   = dark_ratio
    pass_B = (bright_ratio >= SUBTITLE_BRIGHT_RATIO_MIN and
              dark_ratio   >= SUBTITLE_DARK_RATIO_MIN)

    # ── 신호 C: 수평 연속 런 밀도 ────────────────────────────────────────────
    # 엣지 맵에서 행 단위로 길이 ≥ 5인 연속 True 픽셀 개수 카운트
    _, bw = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hrun_pixels = 0
    MIN_RUN = 5
    for row in bw:
        in_run   = False
        run_len  = 0
        for px in row:
            if px > 0:
                run_len += 1
                in_run   = True
            else:
                if in_run and run_len >= MIN_RUN:
                    hrun_pixels += run_len
                run_len = 0
                in_run  = False
        if in_run and run_len >= MIN_RUN:
            hrun_pixels += run_len
    hrun_density = float(hrun_pixels) / total
    signals["hrun_density"] = hrun_density
    pass_C = hrun_density >= SUBTITLE_HRUN_DENSITY_MIN

    passed   = sum([pass_A, pass_B, pass_C])
    has_text = passed >= SUBTITLE_SIGNAL_MIN_PASS

    logger.debug(
        f"  [자막필터] 엣지={edge_density:.3f}({'✓' if pass_A else '✗'}) "
        f"밝기={bright_ratio:.3f}/어둠={dark_ratio:.3f}({'✓' if pass_B else '✗'}) "
        f"수평런={hrun_density:.3f}({'✓' if pass_C else '✗'}) "
        f"→ {passed}/3 → {'자막있음' if has_text else '자막없음'}"
    )
    return has_text, signals


def calculate_dynamic_settings(video_duration: float) -> Dict[str, Any]:
    dynamic_target     = min(MAX_FRAMES, max(MIN_FRAMES, int(video_duration / 4)))
    adaptive_threshold = (4.0 if video_duration <= 15
                          else 5.0 if video_duration <= 30
                          else CHANGE_THRESHOLD)
    logger.info(f"🎯 동적 설정: 영상 {video_duration:.0f}초 → 목표 {dynamic_target}장, 임계값 {adaptive_threshold}")
    return {"DYNAMIC_TARGET": dynamic_target, "CHANGE_THRESHOLD": adaptive_threshold}


def calibrate_noise_threshold(cap: cv2.VideoCapture, sample_count: int = 6) -> float:
    """
    grab → retrieve() 방식 캘리브레이션.
    ROI의 uint8 absdiff로 배경 노이즈 기준값을 측정하여 임계값을 자동 보정.
    """
    diffs      = []
    prev_small = None
    orig_pos   = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    vid_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    y0, y1     = int(vid_h * SUBTITLE_TOP), int(vid_h * SUBTITLE_BOTTOM)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    grabbed = 0
    while len(diffs) < sample_count:
        if not cap.grab():
            break
        grabbed += 1
        if grabbed % 5 != 0:
            continue
        ret, frame = cap.retrieve()
        if not ret:
            continue
        roi   = frame[y0:y1, :]
        small = cv2.resize(
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
            (120, 30), interpolation=cv2.INTER_AREA
        )
        if prev_small is not None:
            diff = float(np.mean(cv2.absdiff(small, prev_small)))
            diffs.append(diff)
        prev_small = small

    cap.set(cv2.CAP_PROP_POS_FRAMES, orig_pos)
    if not diffs:
        return CHANGE_THRESHOLD

    baseline   = float(np.mean(sorted(diffs)[:max(1, int(len(diffs) * 0.75))]))
    calibrated = max(2.0, min(7.0, baseline * 1.5))
    logger.info(f"🔬 노이즈 보정: {baseline:.2f} → 임계값 {calibrated:.2f}")
    return calibrated


# ── 메인 클래스 ───────────────────────────────────────────────────────────────

class VideoPreprocessor:

    def __init__(self):
        os.makedirs(VIDEO_DIR, exist_ok=True)
        os.makedirs(FRAME_DIR, exist_ok=True)
        logger.info(f"VideoPreprocessor 초기화 (Mock: {MOCK_MODE})")
        logger.info(f"📁 저장 경로: {os.path.abspath(BASE_DIR)}")
        logger.info(f"🚫 EasyOCR 미사용 — 자막 해석은 LMM(Phase 2) 담당")
        logger.info(f"🔍 자막 존재 필터: 엣지≥{SUBTITLE_EDGE_DENSITY_MIN} / 대비 / 수평런≥{SUBTITLE_HRUN_DENSITY_MIN} (2/3 통과)")

    # ── 1. 메타데이터 ─────────────────────────────────────────────────────────

    def parse_metadata(self, video_url: str) -> VideoMetadata:
        if MOCK_MODE:
            return self._mock_parse_metadata(video_url)
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True, 'skip_download': True}) as ydl:
                logger.info(f"🔍 메타데이터 수집 중: {video_url}")
                info = ydl.extract_info(video_url, download=False)
                return VideoMetadata(
                    video_id      = info.get('id', ''),
                    title         = info.get('title', ''),
                    description   = (info.get('description', '') or '')[:500],
                    duration      = info.get('duration', 0) or 0,
                    view_count    = info.get('view_count', 0) or 0,
                    upload_date   = info.get('upload_date', '') or '',
                    channel_name  = info.get('uploader', '') or info.get('channel', ''),
                    thumbnail_url = info.get('thumbnail', '')
                )
        except Exception as e:
            logger.error(f"❌ 메타데이터 수집 실패: {e}")
            return self._mock_parse_metadata(video_url)

    def _mock_parse_metadata(self, video_url: str) -> VideoMetadata:
        fake_data = [
            {"title": "🔥충격🔥 이것만 알면 100만원 번다!!", "description": "돈 버는 비법.", "duration": 58, "view_count": 1250000, "channel": "돈버는법채널"},
            {"title": "Python 변수와 함수 완벽 마스터",       "description": "파이썬 기초.",   "duration": 55, "view_count":   45000, "channel": "코딩교육채널"},
        ]
        sel      = random.choice(fake_data)
        video_id = video_url.split('/')[-1] if '/' in video_url else f"mock_{random.randint(1000,9999)}"
        return VideoMetadata(
            video_id=video_id, title=sel["title"], description=sel["description"],
            duration=sel["duration"], view_count=sel["view_count"],
            upload_date="2024-01-15", channel_name=sel["channel"],
            thumbnail_url=f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"
        )

    # ── 2. 다운로드 ───────────────────────────────────────────────────────────

    def download_video(self, video_url: str, video_id: str) -> Optional[str]:
        if MOCK_MODE:
            logger.info("🎭 Mock 모드: 영상 다운로드 생략")
            return None
        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        if os.path.exists(video_path):
            logger.info(f"♻️ 기존 영상 재사용: {video_path}")
            return video_path
        try:
            import yt_dlp
            logger.info(f"⬇️ 영상 다운로드 중: {video_url}")
            with yt_dlp.YoutubeDL({
                'quiet': False, 'no_warnings': True,
                'format': 'best[ext=mp4]/best', 'outtmpl': video_path
            }) as ydl:
                ydl.download([video_url])
            if os.path.exists(video_path):
                logger.info(f"✅ 다운로드 완료: {video_path} ({os.path.getsize(video_path)/1024/1024:.1f}MB)")
                return video_path
            logger.error("❌ 다운로드 후 파일 없음")
            return None
        except Exception as e:
            logger.error(f"❌ 영상 다운로드 실패: {e}")
            return None

    # ── 3. 자막 영상 여부 사전 판단 ──────────────────────────────────────────

    def _detect_has_subtitles(
        self, cap: cv2.VideoCapture, total_frames: int, fps: float, sample_n: int = 20
    ) -> bool:
        """
        영상 전체에서 균등 샘플 sample_n장을 뽑아 자막 존재 신호를 투표.
        전체 샘플 중 25% 이상이 자막 있음으로 판정되면 "자막 있는 영상"으로 분류.

        - 자막 없는 정상 영상: 자막 필터를 건너뛰고 diff 기반으로만 15장 추출
        - 자막 있는 영상   : 자막 있는 프레임만 골라 15장 추출 + 미달 시 보충
        """
        orig_pos  = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        step      = max(1, total_frames // sample_n)
        hit_count = 0

        for i in range(sample_n):
            fidx = min(i * step, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue
            has_text, _ = _has_subtitle_text(frame)
            if has_text:
                hit_count += 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, orig_pos)
        ratio = hit_count / sample_n
        logger.info(f"🔎 자막 영상 판별: {hit_count}/{sample_n} ({ratio*100:.0f}%) → "
                    f"{'자막 있음 ✓' if ratio >= 0.25 else '자막 없음 — diff 모드'}")
        return ratio >= 0.25

    # ── 4. 메인 프레임 추출 파이프라인 ───────────────────────────────────────

    def extract_keyframes(self, video_path: str, video_id: str) -> List[KeyframeEvent]:
        """
        v6.2 — 자막 유무 자동 감지 + 미달 시 보충 로직

        [영상 타입 자동 분기]
          - 자막 있는 영상 : 자막 필터 적용 → 자막 있는 프레임만 추출
                            → 15장 미달 시 기존 프레임 사이 구간에서 보충
          - 자막 없는 영상 : 자막 필터 건너뜀 → diff 기반 장면 전환으로 15장 추출

        [Phase A] grab 루프 → uint8 absdiff → 후보 리스트 (공통)
        [Phase B-자막형] 자막 필터 → pHash 중복 제거 → 15장 미달 시 보충
        [Phase B-무자막형] pHash 중복 제거만 → 균등 분포 보충
        """
        frame_save_dir = os.path.join(FRAME_DIR, video_id)
        os.makedirs(frame_save_dir, exist_ok=True)

        if not video_path or not os.path.exists(video_path):
            logger.warning("⚠️ 영상 파일 없음 → 키프레임 추출 건너뜀")
            return []

        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"❌ 영상 열기 실패: {video_path}")
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
            duration     = total_frames / fps
            vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            roi_y0       = int(vid_h * SUBTITLE_TOP)
            roi_y1       = int(vid_h * SUBTITLE_BOTTOM)

            logger.info(f"🎬 {total_frames}프레임 / {fps:.1f}fps / {duration:.1f}초")

            settings       = calculate_dynamic_settings(duration)
            target_frames  = settings["DYNAMIC_TARGET"]   # ≤ Ncap=15
            diff_threshold = min(calibrate_noise_threshold(cap), settings["CHANGE_THRESHOLD"])
            DIFF_STEP      = max(3, int(fps / 3))
            skip_frames    = int(fps * 0.5)

            # ── Step 0: 자막 영상 여부 사전 판단 ─────────────────────────────
            has_subtitles = self._detect_has_subtitles(cap, total_frames, fps)
            logger.info(f"📋 추출 모드: {'자막 기반' if has_subtitles else 'diff 기반(무자막)'}")

            # ── Phase A: grab 루프 — diff 기반 후보 생성 (공통) ──────────────
            candidates: List[Tuple[int, float]] = []
            prev_small  = None
            frame_idx   = skip_frames

            cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
            t_phaseA = time.time()

            while frame_idx < total_frames:
                skipped = 0
                while skipped < DIFF_STEP - 1 and frame_idx < total_frames:
                    if not cap.grab():
                        frame_idx = total_frames
                        break
                    frame_idx += 1
                    skipped   += 1

                if frame_idx >= total_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                roi_gray = cv2.cvtColor(frame[roi_y0:roi_y1, :], cv2.COLOR_BGR2GRAY)
                if cv2.mean(roi_gray)[0] < BLACK_FRAME_THRESHOLD:
                    prev_small = None
                    continue

                curr_small = cv2.resize(roi_gray, (120, 30), interpolation=cv2.INTER_AREA)
                if prev_small is not None:
                    diff_score = float(np.mean(cv2.absdiff(curr_small, prev_small)))
                    if diff_score >= diff_threshold:
                        candidates.append((frame_idx, diff_score))

                prev_small = curr_small

            cap.release()

            t_phaseA_done = time.time()
            logger.info(f"🔎 Phase A 완료: 후보 {len(candidates)}개 ({t_phaseA_done - t_phaseA:.2f}초)")

            if not candidates:
                logger.warning("⚠️ 후보 없음 → 빈 결과 반환")
                return []

            # ── Phase B: 후보 필터링 (자막형 / 무자막형 분기) ────────────────
            t_phaseB   = time.time()
            cap2       = cv2.VideoCapture(video_path)
            save_targets: List[Tuple[int, float, np.ndarray, int]] = []
            # (frame_idx, diff_score, frame, phash)
            prev_hash: Optional[int] = None

            for cand_fidx, diff_score in candidates:
                if len(save_targets) >= Ncap:
                    break

                cap2.set(cv2.CAP_PROP_POS_FRAMES, cand_fidx)
                ret, frame = cap2.read()
                if not ret:
                    continue

                roi_gray = cv2.cvtColor(frame[roi_y0:roi_y1, :], cv2.COLOR_BGR2GRAY)
                if cv2.mean(roi_gray)[0] < BLACK_FRAME_THRESHOLD:
                    continue

                # 자막 있는 영상 → 자막 필터 적용 / 자막 없는 영상 → 필터 건너뜀
                if has_subtitles:
                    ok, sigs = _has_subtitle_text(frame)
                    if not ok:
                        logger.debug(
                            f"  [자막없음 스킵] f={cand_fidx} "
                            f"edge={sigs['edge_density']:.3f} "
                            f"bright={sigs['bright_ratio']:.3f} "
                            f"hrun={sigs['hrun_density']:.3f}"
                        )
                        continue

                curr_hash = _compute_phash(frame)
                if prev_hash is not None:
                    if _hamming_distance(curr_hash, prev_hash) <= PHASH_HAMMING_MAX:
                        logger.debug(f"  [pHash 중복] f={cand_fidx} → 스킵")
                        continue

                save_targets.append((cand_fidx, diff_score, frame, curr_hash))
                prev_hash = curr_hash

            cap2.release()
            logger.info(f"🗂️ Phase B 완료: {len(save_targets)}/{len(candidates)} 유효 ({time.time()-t_phaseB:.2f}초)")

            # ── Step C: 15장 미달 시 보충 ─────────────────────────────────────
            #
            # [자막 있는 영상] 기존 선택된 프레임 사이 구간마다 후보를 다시 스캔.
            #   자막 있는 프레임만 추가하되 pHash 중복 검사도 재적용.
            # [자막 없는 영상] 전체 영상을 target_frames 등분하여 균등 보충.
            #
            if len(save_targets) < target_frames:
                need = target_frames - len(save_targets)
                logger.info(f"🔧 보충 필요: {len(save_targets)}장 → 목표 {target_frames}장 ({need}장 부족)")

                used_fidxs = {t[0] for t in save_targets}
                all_hashes = [t[3] for t in save_targets]
                cap3       = cv2.VideoCapture(video_path)

                if has_subtitles:
                    # 기존 프레임 사이 구간을 세밀하게 재스캔 (DIFF_STEP//2 간격)
                    sorted_fidxs = sorted(used_fidxs)
                    # 영상 전체를 구간 분할: 처음~첫프레임, 각 구간, 마지막~끝
                    boundaries = (
                        [(skip_frames, sorted_fidxs[0])] +
                        [(sorted_fidxs[i], sorted_fidxs[i+1]) for i in range(len(sorted_fidxs)-1)] +
                        [(sorted_fidxs[-1], total_frames)]
                    )
                    fine_step = max(2, DIFF_STEP // 2)
                    extra_candidates: List[Tuple[int, float]] = []

                    for seg_start, seg_end in boundaries:
                        f = seg_start + fine_step
                        while f < seg_end and f < total_frames:
                            if f not in used_fidxs:
                                extra_candidates.append((f, 0.0))
                            f += fine_step

                    for cand_fidx, _ in extra_candidates:
                        if len(save_targets) >= target_frames:
                            break
                        cap3.set(cv2.CAP_PROP_POS_FRAMES, cand_fidx)
                        ret, frame = cap3.read()
                        if not ret:
                            continue
                        roi_gray = cv2.cvtColor(frame[roi_y0:roi_y1, :], cv2.COLOR_BGR2GRAY)
                        if cv2.mean(roi_gray)[0] < BLACK_FRAME_THRESHOLD:
                            continue
                        ok, _ = _has_subtitle_text(frame)
                        if not ok:
                            continue
                        curr_hash = _compute_phash(frame)
                        if any(_hamming_distance(curr_hash, h) <= PHASH_HAMMING_MAX for h in all_hashes):
                            continue
                        save_targets.append((cand_fidx, 0.0, frame, curr_hash))
                        all_hashes.append(curr_hash)
                        logger.info(f"  ➕ 보충 [자막] f={cand_fidx} t={cand_fidx/fps:.1f}s")

                else:
                    # 무자막 영상: 전체를 target_frames 등분해서 균등 보충
                    step_fill = max(1, total_frames // (target_frames + 1))
                    for i in range(1, target_frames * 2):
                        if len(save_targets) >= target_frames:
                            break
                        cand_fidx = min(skip_frames + i * step_fill, total_frames - 1)
                        if cand_fidx in used_fidxs:
                            continue
                        cap3.set(cv2.CAP_PROP_POS_FRAMES, cand_fidx)
                        ret, frame = cap3.read()
                        if not ret:
                            continue
                        roi_gray = cv2.cvtColor(frame[roi_y0:roi_y1, :], cv2.COLOR_BGR2GRAY)
                        if cv2.mean(roi_gray)[0] < BLACK_FRAME_THRESHOLD:
                            continue
                        curr_hash = _compute_phash(frame)
                        if any(_hamming_distance(curr_hash, h) <= PHASH_HAMMING_MAX for h in all_hashes):
                            continue
                        save_targets.append((cand_fidx, 0.0, frame, curr_hash))
                        all_hashes.append(curr_hash)
                        used_fidxs.add(cand_fidx)
                        logger.info(f"  ➕ 보충 [무자막균등] f={cand_fidx} t={cand_fidx/fps:.1f}s")

                cap3.release()
                logger.info(f"✅ 보충 후 최종: {len(save_targets)}장")

            # 시간 순 정렬 후 Ncap 상한 적용
            save_targets.sort(key=lambda x: x[0])
            save_targets = save_targets[:Ncap]

            # ── 병렬 프레임 저장 ──────────────────────────────────────────────
            def _save_one(save_idx: int, fidx: int, diff: float,
                          fr: np.ndarray, phash: int) -> Tuple[int, str]:
                ts   = fidx / fps
                path = self._save_frame(fr, frame_save_dir, save_idx, fidx, ts)
                return save_idx, path

            saved_paths: Dict[int, str] = {}
            with ThreadPoolExecutor(max_workers=PARALLEL_SAVE_WORKERS) as executor:
                futs = {
                    executor.submit(_save_one, si, fi, di, fr, ph): si
                    for si, (fi, di, fr, ph) in enumerate(save_targets)
                }
                for fut in as_completed(futs):
                    si, path = fut.result()
                    saved_paths[si] = path

            # 이벤트 목록 구성
            events: List[KeyframeEvent] = []
            for save_idx, (fidx, diff_score, frame, phash) in enumerate(save_targets):
                img_path = saved_paths.get(save_idx, "")
                if not img_path:
                    continue
                ts = fidx / fps
                events.append(KeyframeEvent(
                    frame_index   = fidx,
                    timestamp_sec = round(ts, 3),
                    timestamp_str = _sec_to_hhmmss(ts),
                    diff_score    = round(diff_score, 3),
                    phash         = phash,
                    image_path    = img_path,
                ))
                logger.info(f"  ✅ [#{save_idx+1:03d}] t={_sec_to_hhmmss(ts)} | diff={diff_score:.2f}")

            total_time = time.time() - t_phaseA
            logger.info(
                f"📊 완료 ({'자막형' if has_subtitles else '무자막형'}): "
                f"후보 {len(candidates)}개 → 최종 {len(events)}장 | 총 {total_time:.2f}초"
            )
            logger.info(f"📤 {len(events)}장의 키프레임을 LMM(Phase 2)으로 전달합니다.")
            return events

        except Exception as e:
            logger.error(f"❌ 키프레임 추출 실패: {e}", exc_info=True)
            return []

    # ── 4. 프레임 저장 (JPEG) ─────────────────────────────────────────────────

    def _save_frame(
        self, frame: np.ndarray, save_dir: str,
        saved_count: int, frame_idx: int, timestamp_sec: float,
    ) -> str:
        roi_gray = cv2.cvtColor(frame[int(frame.shape[0]*SUBTITLE_TOP):, :], cv2.COLOR_BGR2GRAY)
        if float(cv2.mean(roi_gray)[0]) < BLACK_FRAME_THRESHOLD:
            logger.warning(f"  ⚠️ 검정 프레임 → 저장 생략 (f={frame_idx})")
            return ""

        ts  = _sec_to_hhmmss(timestamp_sec)
        hh, mm, ss_ms = ts.split(":")
        ss, ms = ss_ms.split(".")
        filename = f"frame_{saved_count+1:05d}_{hh}h{mm}m{ss}s{ms}{FRAME_EXT}"
        filepath = os.path.join(save_dir, filename)

        if not cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY]):
            logger.error(f"❌ 저장 실패: {filepath}")
            return ""

        logger.info(f"  💾 [{saved_count+1:03d}] {filename} | {timestamp_sec:.1f}초")
        return filepath

    # ── 5. Contact Sheet ──────────────────────────────────────────────────────

    def save_contact_sheet(
        self, frame_dir: str, video_id: str, metadata: Dict[str, Any] = None
    ) -> Optional[str]:
        if not os.path.exists(frame_dir):
            return None
        frame_files = sorted([
            f for f in os.listdir(frame_dir)
            if f.startswith("frame_") and (f.endswith(".jpg") or f.endswith(".png"))
        ])
        if not frame_files:
            logger.warning("⚠️ Contact Sheet: 프레임 없음")
            return None
        frames = [cv2.imread(os.path.join(frame_dir, f)) for f in frame_files]
        frames = [f for f in frames if f is not None]
        if not frames:
            return None

        cols = 3
        rows = (len(frames) + cols - 1) // cols
        tw, th, hh, pad = 480, 270, 70, 2
        canvas = np.full((hh + rows * (th + pad), cols * tw, 3), 28, dtype=np.uint8)

        title   = (metadata or {}).get("title", video_id)[:60]
        channel = (metadata or {}).get("channel_name", "")
        cv2.rectangle(canvas, (0,0), (cols*tw, hh), (40,40,40), -1)
        cv2.putText(canvas, f"{title}  |  {channel}  |  {len(frames)}장",
                    (14,26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200,220,255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"video_id: {video_id}  |  OpenCV v6.0 (pHash 중복제거, EasyOCR 없음)",
                    (14,54), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140,140,140), 1, cv2.LINE_AA)

        for idx, fr in enumerate(frames):
            row, col = divmod(idx, cols)
            y0, x0 = hh + row*(th+pad), col*tw
            canvas[y0:y0+th, x0:x0+tw] = cv2.resize(fr, (tw, th))
            cv2.rectangle(canvas, (x0,y0), (x0+tw-1,y0+th-1), (70,70,70), 1)

        out_path = os.path.join(frame_dir, "_contact_sheet.jpg")
        cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        logger.info(f"🖼️  Contact Sheet: {out_path}")
        return out_path

    # ── 6. 전체 파이프라인 ────────────────────────────────────────────────────

    def process(self, video_url: str) -> PreprocessingResult:
        start_time     = time.time()
        processing_log = []

        logger.info(f"🚀 전처리 시작 (v6.0 — OpenCV 전용): {video_url}")
        processing_log.append(f"전처리 시작 (OpenCV 전용, EasyOCR 없음): {video_url}")

        metadata = self.parse_metadata(video_url)
        processing_log.append(f"✅ 메타데이터: {metadata.title}")
        logger.info(f"📋 제목: {metadata.title} | 채널: {metadata.channel_name}")

        video_path = self.download_video(video_url, metadata.video_id)
        processing_log.append(
            f"✅ 다운로드: downloads/videos/{metadata.video_id}.mp4" if video_path
            else "⚠️ 영상 다운로드 없음 (Mock 또는 실패)"
        )

        frame_dir      = os.path.join(FRAME_DIR, metadata.video_id)

        # ── Phase A+B: OpenCV 키프레임 추출 ──────────────────────────────────
        keyframe_events = self.extract_keyframes(video_path, metadata.video_id)

        # ── 키프레임 base64 인코딩 (병렬) ────────────────────────────────────
        def _encode(ev: KeyframeEvent) -> str:
            if not ev.image_path or not os.path.exists(ev.image_path):
                return ""
            with open(ev.image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        with ThreadPoolExecutor(max_workers=PARALLEL_SAVE_WORKERS) as executor:
            encoded = list(executor.map(_encode, keyframe_events))
        keyframes = [e for e in encoded if e]

        # ocr_text는 비워둠 — LMM이 이미지를 보고 직접 추출
        ocr_text = ""

        processing_log.append(
            f"✅ OpenCV 키프레임 추출: {len(keyframes)}장 "
            f"(변화 감지 후보 → pHash 중복 제거 → N_final ≤ {Ncap})"
        )
        processing_log.append("📤 자막 텍스트 추출: LMM(Phase 2) 위임 (OCR 엔진 미사용)")
        logger.info(f"📝 키프레임 {len(keyframes)}장 → LMM으로 전달")

        sheet_path = self.save_contact_sheet(
            frame_dir=frame_dir, video_id=metadata.video_id,
            metadata={"title": metadata.title, "channel_name": metadata.channel_name}
        )
        if sheet_path:
            processing_log.append(f"✅ Contact Sheet: {sheet_path}")

        preview_path = make_preview(metadata.video_id)
        if preview_path:
            processing_log.append(f"✅ 프리뷰: {preview_path}")

        elapsed = time.time() - start_time
        processing_log.append(f"⏱️ 전처리 완료: {elapsed:.2f}초 (T_dl + T_opencv)")
        logger.info(f"✅ 전처리 완료: {elapsed:.2f}초")

        return PreprocessingResult(
            video_metadata = metadata,
            keyframes      = keyframes,
            ocr_text       = ocr_text,   # 빈 문자열 — LMM이 이미지에서 직접 읽음
            layout_score   = 0.0,
            roi_data       = {
                "video_path"        : video_path or "",
                "frame_dir"         : frame_dir,
                "frame_count"       : len(keyframes),
                "n_candidates"      : 0,   # process() 완료 후에는 이미 필터됨
                "subtitle_filtered" : True,
                "contact_sheet"     : sheet_path or "",
                "zone_detection"    : f"fixed_{int(SUBTITLE_TOP*100)}_{int(SUBTITLE_BOTTOM*100)}_v6.2",
                "ocr_engine"        : "none (LMM delegates)",
                "extraction_method" : "opencv_subtitle_autodetect+phash+fill",
                "subtitle_filter"   : {
                    "edge_density_min" : SUBTITLE_EDGE_DENSITY_MIN,
                    "bright_ratio_min" : SUBTITLE_BRIGHT_RATIO_MIN,
                    "dark_ratio_min"   : SUBTITLE_DARK_RATIO_MIN,
                    "hrun_density_min" : SUBTITLE_HRUN_DENSITY_MIN,
                    "min_pass"         : SUBTITLE_SIGNAL_MIN_PASS,
                },
                "phash_hamming_max" : PHASH_HAMMING_MAX,
                "n_cap"             : Ncap,
                "nf_limit"          : Nf,
                "black_frame_filter": True,
            },
            processing_log = processing_log
        )


# ── CLI 프리뷰 ────────────────────────────────────────────────────────────────

PREVIEW_DIR = "downloads/previews"


def _get_latest_video_id() -> Optional[str]:
    if not os.path.exists(FRAME_DIR):
        return None
    folders = [f for f in os.listdir(FRAME_DIR)
               if os.path.isdir(os.path.join(FRAME_DIR, f)) and not f.startswith(".")]
    if not folders:
        return None
    folders.sort(key=lambda f: os.path.getmtime(os.path.join(FRAME_DIR, f)), reverse=True)
    return folders[0]


def make_preview(video_id: str) -> Optional[str]:
    from datetime import datetime
    folder = os.path.join(FRAME_DIR, video_id)
    if not os.path.exists(folder):
        logger.error(f"❌ 폴더 없음: {folder}")
        return None
    files = sorted([f for f in os.listdir(folder)
                    if f.startswith("frame_") and (f.endswith(".jpg") or f.endswith(".png"))])
    frames = [(f, cv2.imread(os.path.join(folder, f))) for f in files]
    frames = [(f, img) for f, img in frames if img is not None]
    if not frames:
        logger.error(f"❌ 프레임 없음: {folder}")
        return None

    os.makedirs(PREVIEW_DIR, exist_ok=True)
    cols, tw, th, lh, hh, pad = 3, 540, 304, 28, 64, 6
    rows     = (len(frames) + cols - 1) // cols
    canvas_w = cols * (tw + pad) + pad
    canvas_h = hh + rows * (th + lh + pad) + pad
    canvas   = np.full((canvas_h, canvas_w, 3), 22, dtype=np.uint8)

    cv2.rectangle(canvas, (0,0), (canvas_w, hh), (45,45,45), -1)
    cv2.putText(canvas, f"video_id: {video_id}   |   {len(frames)}장  (OpenCV v6.0 — EasyOCR 없음)",
                (14,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,220,255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                (14,54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130,130,130), 1, cv2.LINE_AA)

    for idx, (fname, fr) in enumerate(frames):
        row, col = divmod(idx, cols)
        x0 = pad + col * (tw + pad)
        y0 = hh  + pad + row * (th + lh + pad)
        canvas[y0:y0+th, x0:x0+tw] = cv2.resize(fr, (tw, th))
        cv2.rectangle(canvas, (x0-1,y0-1), (x0+tw,y0+th), (70,70,70), 1)
        ly = y0 + th
        cv2.rectangle(canvas, (x0,ly), (x0+tw,ly+lh), (35,35,35), -1)
        cv2.putText(canvas, fname, (x0+6,ly+19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1, cv2.LINE_AA)

    out_path = os.path.join(PREVIEW_DIR, f"{video_id}_preview.png")
    if not cv2.imwrite(out_path, canvas):
        logger.error(f"❌ 프리뷰 저장 실패: {out_path}")
        return None
    logger.info(f"🖼️  프리뷰 저장: {out_path}")
    return out_path


def open_preview(path: str) -> None:
    import platform, subprocess
    system = platform.system()
    try:
        if   system == "Darwin":  subprocess.run(["open", path])
        elif system == "Windows": os.startfile(path)
        else:                     subprocess.run(["xdg-open", path])
        print(f"🖼️  이미지 뷰어 실행: {path}")
    except Exception as e:
        print(f"⚠️  자동 열기 실패: {e}\n   직접 열어주세요: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="프레임 추출 결과 프리뷰 생성기 (v6.0 — OpenCV 전용)")
    parser.add_argument("--id",   help="video_id (없으면 최신 영상 자동 탐지)")
    parser.add_argument("--open", action="store_true", help="생성 후 이미지 뷰어 자동 실행")
    args     = parser.parse_args()
    video_id = args.id or _get_latest_video_id()
    if not video_id:
        print("❌ 분석된 영상이 없습니다. 먼저 파이프라인을 실행하세요.")
    else:
        out = make_preview(video_id)
        if out:
            print(f"✅ 프리뷰 저장: {out}")
            if args.open: open_preview(out)
            else: print(f"\n💡 뷰어로 열려면: python preprocessing.py --open")