"""
Phase 1: 전처리 모듈 (v7.2) — 타이포그래피 쇼츠(검정 배경+전면 텍스트) 프레임 추출 버그 수정
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[v7.2 핵심 변경 사항 — 타이포그래피 쇼츠 4종 버그 수정]

  ① ROI 범위 확장: SUBTITLE_TOP 0.60 → 0.10
     - 기존: 화면 하단 40%만 스캔 → 중앙/상단 텍스트 완전 누락
     - 수정: 화면 전체(10~95%) 스캔 → 전면 텍스트 / 중앙 텍스트 / 하단 자막 모두 커버
     - 영향: _has_subtitle_text, _compute_phash, _sharpness_of_roi 전부 자동 적용

  ② 자막 3-신호 필터 신호B 완화: AND → bright AND (dark OR bright*2)
     - 기존: bright(>200) ≥ 8% AND dark(<55) ≥ 8% 동시 충족 필수
     - 문제: 검정 배경 영상은 dark_ratio=0.8+ 이지만 ROI가 넓어져
             bright_ratio(글자만)가 8%에 미달 → 신호B 항상 실패
     - 수정: bright 기준만 충족하면 dark 조건은 선택적으로 완화

  ③ 블랙 프레임 필터 이중 조건: mean < 8 AND max < 30
     - 기존: cv2.mean < 8 단독 → 글자가 몇 픽셀만 있어도 평균이 낮아 스킵
     - 수정: max < 30 도 함께 충족해야 스킵 → 글자 있는 프레임 보존

  ④ 극단 쇼트(<10초) 자막 판정 비율 완화: sub_ratio 0.12 → 0.05
     - 4초 영상에서 샘플 12장 중 1장(8%)만 감지돼도 자막형으로 판정
     - 기존 12% 기준에서는 2장 이상 필요 → 전부 무자막형 오판

[v7.1 버그 수정 유지] calibrate 인트로 샘플링 버그 수정

[기획서 핵심 설계 원칙 — 코드에 직접 반영]

  Nf = 25   쇼츠 정보 밀도 물리적 상한
            60s ÷ 2.4s/컷 ≈ 25 (SBD 알고리즘 + 쇼츠 편집 트렌드 근거)
            → Phase A candidates 상한으로 실제 적용

  Ncap = 15  API 비용·정확도 최적 배치 크기
            GPT-4o mini 512×512 타일 과금 기준 최적점
            "Lost in the Middle" (Liu et al., 2023) — 15장이 컨텍스트 망각 임계
            → N_final = min(Nf_scaled, Ncap) 최종 상한

  N_final = min(Nf_scaled, Ncap)
            Nf_scaled = Nf × (duration / 60)  — 60초 초과 영상에서 Nf 비례 확장
            SLA: 어떤 영상이 들어와도 T_total ≤ 10초 보장

[v7.0 핵심 변경 사항]

  ① 전 구간 영상 지원 (≤10s / ≤20s / ≤30s / ≤60s / ≤120s / >120s)
  ② Nf 상한을 Phase A candidates에 실제 적용
  ③ 긴 영상 균등 분포 보장 (temporal spread)
  ④ Phase A 후보 없을 때 Fallback 강화
  ⑤ 기존 v6.x 모든 기능 유지

[파이프라인 흐름]
  Step 0  : calculate_dynamic_settings → 구간별 파라미터 결정
  Step 0b : _detect_has_subtitles     → 자막형 / 무자막형 판정
  Phase A : grab루프 → diff 후보 생성 → Nf_scaled 상한 적용 → 시간대별 균등 선별
  Phase B : [자막형]   자막 필터 → 선명도 보정 → pHash 중복 제거
            [무자막형] 선명도 보정 → pHash 중복 제거
  Step C  : N_final 미달 시 보충 (자막형: 구간 재스캔 / 무자막형: 균등 보충)
  저장    : 병렬 JPEG 저장 → LMM(Phase 2)으로 전달

[역할 분담]
  OpenCV (본 모듈): 픽셀 변화 감지, pHash 중복 제거, 키프레임 추출
  LMM (Phase 2)  : 자막 텍스트 해석 + 의미론적 맥락 판별

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
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from config import ROI_CONFIG, MOCK_MODE
from models import VideoMetadata, PreprocessingResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── 디렉토리 ──────────────────────────────────────────────────────────────────
BASE_DIR  = "downloads"
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
FRAME_DIR = os.path.join(BASE_DIR, "frames")
PREVIEW_DIR = os.path.join(BASE_DIR, "previews")

# ══════════════════════════════════════════════════════════════════════════════
# 기획서 핵심 수치 — 절대 변경 금지 (설계 근거 포함)
# ══════════════════════════════════════════════════════════════════════════════

# Nf = 25 : 쇼츠 정보 밀도 물리적 상한선
#   근거: 60s ÷ 2.4s/컷 ≈ 25 (SBD 알고리즘 + 쇼츠 편집 트렌드)
#   용도: Phase A candidates 상한 (Nf_scaled = Nf × duration/60 으로 긴 영상 확장)
Nf   = 25

# Ncap = 15 : API 비용·정확도 최적 배치 크기 (N_final 절대 상한)
#   근거: GPT-4o mini 512×512 타일 과금 최적점
#         "Lost in the Middle" (Liu et al., 2023) — 컨텍스트 망각 임계
#   용도: N_final = min(Nf_scaled, Ncap) 의 최종 상한
Ncap = 15

MAX_FRAMES = Ncap   # N_final = min(Nf_scaled, Ncap) → 최대 15
MIN_FRAMES = 3      # 어떤 영상도 최소 3장은 보장

# ── ROI 설정 — 자막 영역 ─────────────────────────────────────────────────────
# [v7.2 수정] 타이포그래피 쇼츠(전면 텍스트) 대응: 상단을 0.10으로 확장
# 기존 0.60~0.92는 하단 자막 전용 → 화면 중앙 글자 완전 누락
# 0.10~0.95 로 넓혀 전면 텍스트 / 중앙 텍스트 / 하단 자막 모두 커버
SUBTITLE_TOP    = 0.10   # 자막 ROI 상단 (화면의 10% 지점) ← 0.60에서 확장
SUBTITLE_BOTTOM = 0.95   # 자막 ROI 하단 (화면의 95% 지점) ← 0.92에서 소폭 확장

# ── 변화량 감지 기본 임계값 ───────────────────────────────────────────────────
# calibrate_noise_threshold() 가 영상별로 자동 보정하므로 이 값은 fallback
CHANGE_THRESHOLD = 6.0

# ── 블랙 프레임 필터 ──────────────────────────────────────────────────────────
# [v7.2 수정] 검정 배경+흰 글자 타이포그래피 쇼츠는 ROI 평균 밝기가 낮아
# BLACK_FRAME_THRESHOLD=8 에 걸려 유효 프레임이 모두 스킵됨.
# 글자가 있는 프레임은 ROI 최대 픽셀이 높으므로, mean < 8 이면 max도 확인.
BLACK_FRAME_THRESHOLD     = 8
BLACK_FRAME_MAX_THRESHOLD = 30   # mean < 8 이어도 max > 30이면 글자 있는 프레임

# ── pHash — 유사도 기반 중복 제거 (기획서 "Nf 필터") ─────────────────────────
PHASH_SIZE        = 8    # 8×8 = 64비트 해시
PHASH_HAMMING_MAX = 8    # 해밍 거리 ≤ 8 → 동일 자막으로 판정, 제거

# ── 자막 존재 3-신호 필터 ─────────────────────────────────────────────────────
SUBTITLE_EDGE_DENSITY_MIN  = 0.06   # 신호A: Canny 엣지 밀도 하한 (6%)
SUBTITLE_BRIGHT_RATIO_MIN  = 0.08   # 신호B: 밝은 픽셀(>200) 비율 하한
SUBTITLE_DARK_RATIO_MIN    = 0.08   # 신호B: 어두운 픽셀(<55) 비율 하한
SUBTITLE_HRUN_DENSITY_MIN  = 0.04   # 신호C: 수평 연속런 밀도 하한
SUBTITLE_SIGNAL_MIN_PASS   = 2      # 3개 신호 중 2개 이상 통과 시 자막 있음

# ── 선명도 보정 — bounce/fade-in 대응 (v6.3) ─────────────────────────────────
SHARPNESS_SCAN_RADIUS  = 4     # 후보 기준 ±4프레임 스캔
SHARPNESS_MIN_VARIANCE = 30.0  # Laplacian 분산 하한 (이하면 경고)

# ── 병렬 저장 / JPEG 품질 ─────────────────────────────────────────────────────
PARALLEL_SAVE_WORKERS = 4
FRAME_EXT             = ".jpg"
FRAME_JPEG_QUALITY    = 92


# ══════════════════════════════════════════════════════════════════════════════
# 데이터클래스
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class KeyframeEvent:
    """LMM으로 전달되는 키프레임 이벤트 (텍스트 해석은 LMM 담당)"""
    frame_index  : int
    timestamp_sec: float
    timestamp_str: str
    diff_score   : float
    phash        : int  = 0
    image_path   : str  = ""


# ══════════════════════════════════════════════════════════════════════════════
# 순수 함수 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def _sec_to_hhmmss(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _compute_phash(frame: np.ndarray, hash_size: int = PHASH_SIZE) -> int:
    """순수 OpenCV DCT 기반 pHash — 자막 ROI 유사도 측정"""
    roi_y0 = int(frame.shape[0] * SUBTITLE_TOP)
    roi_y1 = int(frame.shape[0] * SUBTITLE_BOTTOM)
    roi    = frame[roi_y0:roi_y1, :]
    gray   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4),
                         interpolation=cv2.INTER_AREA).astype(np.float32)
    dct     = cv2.dct(resized)
    dct_low = dct[:hash_size, :hash_size]
    avg     = np.mean(dct_low)
    bits    = (dct_low > avg).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count('1')


def _has_subtitle_text(frame: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    """
    OCR 없이 3가지 OpenCV 신호로 자막 존재 여부 판단.
      A) 엣지 밀도 (Canny)  — 글자 윤곽선
      B) 고대비 픽셀 공존   — 흰 글씨 + 검은 테두리
      C) 수평 연속 런       — 글자 획의 수평 연속성
    2/3 통과 시 True 반환.
    """
    vid_h  = frame.shape[0]
    roi_y0 = int(vid_h * SUBTITLE_TOP)
    roi_y1 = int(vid_h * SUBTITLE_BOTTOM)
    roi    = frame[roi_y0:roi_y1, :]
    gray   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    total  = gray.size
    sigs   = {}

    # 신호 A: 엣지 밀도
    clahe      = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray_eq    = clahe.apply(gray)
    edges      = cv2.Canny(gray_eq, 40, 120)
    edge_dens  = float(np.count_nonzero(edges)) / total
    sigs["edge_density"] = edge_dens
    pass_A = edge_dens >= SUBTITLE_EDGE_DENSITY_MIN

    # 신호 B: 고대비 픽셀 공존
    # [v7.2 수정] 검정 배경+흰 글자 영상: dark_ratio는 거의 항상 높으나
    # bright_ratio(글자 픽셀)가 ROI 전체 대비 낮아 fail → OR 조건으로 완화
    bright = float(np.sum(gray > 200)) / total
    dark   = float(np.sum(gray < 55))  / total
    sigs["bright_ratio"] = bright
    sigs["dark_ratio"]   = dark
    # 흰 글자 존재(bright 기준 충족) AND (검정 배경 OR 극단적 밝은 글자)
    pass_B = bright >= SUBTITLE_BRIGHT_RATIO_MIN and (
        dark >= SUBTITLE_DARK_RATIO_MIN or bright >= SUBTITLE_BRIGHT_RATIO_MIN * 2
    )

    # 신호 C: 수평 연속 런
    _, bw = cv2.threshold(gray_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hrun, MIN_RUN = 0, 5
    for row in bw:
        in_run, run_len = False, 0
        for px in row:
            if px > 0:
                run_len += 1; in_run = True
            else:
                if in_run and run_len >= MIN_RUN:
                    hrun += run_len
                run_len = 0; in_run = False
        if in_run and run_len >= MIN_RUN:
            hrun += run_len
    hrun_dens = float(hrun) / total
    sigs["hrun_density"] = hrun_dens
    pass_C = hrun_dens >= SUBTITLE_HRUN_DENSITY_MIN

    passed   = sum([pass_A, pass_B, pass_C])
    has_text = passed >= SUBTITLE_SIGNAL_MIN_PASS
    logger.debug(
        f"  [자막필터] 엣지={edge_dens:.3f}({'✓' if pass_A else '✗'}) "
        f"밝기={bright:.3f}/어둠={dark:.3f}({'✓' if pass_B else '✗'}) "
        f"수평런={hrun_dens:.3f}({'✓' if pass_C else '✗'}) "
        f"→ {passed}/3 → {'있음' if has_text else '없음'}"
    )
    return has_text, sigs


def _sharpness_of_roi(frame: np.ndarray) -> float:
    """자막 ROI의 Laplacian 분산 (클수록 선명)"""
    vid_h  = frame.shape[0]
    roi_y0 = int(vid_h * SUBTITLE_TOP)
    roi_y1 = int(vid_h * SUBTITLE_BOTTOM)
    roi    = frame[roi_y0:roi_y1, :]
    gray   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _pick_sharpest_frame(
    cap          : cv2.VideoCapture,
    center_fidx  : int,
    center_frame : np.ndarray,
    total_frames : int,
    radius       : int = SHARPNESS_SCAN_RADIUS,
) -> Tuple[int, np.ndarray, float]:
    """
    center_fidx ±radius 범위에서 Laplacian 분산이 가장 높은 프레임 반환.
    bounce/fade-in 자막의 "착지 완료" 순간을 선택하기 위함.
    """
    best_fidx      = center_fidx
    best_frame     = center_frame
    best_sharpness = _sharpness_of_roi(center_frame)

    for fidx in range(max(0, center_fidx - radius),
                      min(total_frames - 1, center_fidx + radius) + 1):
        if fidx == center_fidx:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, fr = cap.read()
        if not ret:
            continue
        s = _sharpness_of_roi(fr)
        if s > best_sharpness:
            best_sharpness, best_fidx, best_frame = s, fidx, fr

    return best_fidx, best_frame, best_sharpness


# ══════════════════════════════════════════════════════════════════════════════
# 동적 파라미터 계산 — 영상 길이 6구간 분기
# ══════════════════════════════════════════════════════════════════════════════

def calculate_dynamic_settings(video_duration: float) -> Dict[str, Any]:
    """
    영상 길이에 따라 6구간으로 분기해 파라미터를 결정한다.

    [설계 철학]
      - 짧은 영상(≤10s): 모든 파라미터를 "촘촘하게" — 놓치는 자막 없도록
      - 쇼츠(≤60s)    : 기획서 Nf=25, Ncap=15 원칙 그대로
      - 긴 영상(>60s) : Nf를 duration 비례로 확장 (Nf_scaled = Nf × duration/60)
                        단, target_frames는 Ncap=15 절대 상한 유지 (SLA 보장)
                        시간대별 균등 슬롯 강제 (앞부분 쏠림 방지)

    Returns dict keys:
      DYNAMIC_TARGET      int    최종 목표 프레임 수 (≤ Ncap=15)
      NF_SCALED           int    Phase A candidates 상한
      CHANGE_THRESHOLD    float  diff 임계값
      DIFF_STEP_DIVISOR   int    fps ÷ 이 값 = DIFF_STEP
      SKIP_SEC            float  영상 앞부분 스킵 초
      SUBTITLE_SAMPLE_N   int    자막 감지 샘플 수
      SUBTITLE_HIT_RATIO  float  자막형 판정 비율 하한
      TEMPORAL_SPREAD     bool   시간대별 균등 슬롯 강제 여부 (긴 영상)
      TIME_SLOTS          int    균등 분할 슬롯 수 (TEMPORAL_SPREAD=True 시)
    """
    d = video_duration

    if d < 1.0:
        # ── 1초 미만 초극단 클립 — 스킵 없이 전 구간 스캔 ───────────────
        target        = MIN_FRAMES
        nf_scaled     = Nf
        thresh        = 1.0
        step_div      = 8                        # fps//8 → 최대한 촘촘하게
        skip          = 0.0                      # 스킵 없음
        sub_sample    = max(5, int(d * 10))
        sub_ratio     = 0.10
        spread        = False
        slots         = 0

    elif d <= 10:
        # ── 극단적 쇼트 클립 ──────────────────────────────────────────────
        target        = max(MIN_FRAMES, min(MAX_FRAMES, int(d * 1.5)))
        nf_scaled     = Nf                      # 상한 그대로 (짧으니 후보가 적음)
        thresh        = 2.0
        step_div      = 6                        # fps//6 ≈ 5fps마다
        skip          = 0.15
        sub_sample    = max(10, int(d * 3))
        sub_ratio     = 0.05   # [v7.2] 0.12 → 0.05: 4초 영상(샘플≈12)에서 1장만 감지돼도 자막형
        spread        = False
        slots         = 0

    elif d <= 20:
        target        = max(5, min(MAX_FRAMES, int(d / 3)))
        nf_scaled     = Nf
        thresh        = 3.0
        step_div      = 4
        skip          = 0.3
        sub_sample    = 15
        sub_ratio     = 0.18
        spread        = False
        slots         = 0

    elif d <= 30:
        target        = max(5, min(MAX_FRAMES, int(d / 3)))
        nf_scaled     = Nf
        thresh        = 4.5
        step_div      = 3
        skip          = 0.5
        sub_sample    = 20
        sub_ratio     = 0.22
        spread        = False
        slots         = 0

    elif d <= 60:
        # ── 기획서 핵심 구간 — Nf=25, Ncap=15 원칙 ──────────────────────
        target        = MAX_FRAMES              # 15 고정
        nf_scaled     = Nf                      # 25 고정
        thresh        = 6.0
        step_div      = 3
        skip          = 0.5
        sub_sample    = 20
        sub_ratio     = 0.25
        spread        = False
        slots         = 0

    elif d <= 120:
        # ── 중장편 (60~120초) — Nf 비례 확장, 균등 분포 강제 ─────────────
        nf_scaled     = min(50, int(Nf * d / 60))   # 최대 50개 후보
        target        = MAX_FRAMES              # Ncap=15 유지 (SLA)
        thresh        = 6.0
        step_div      = 3
        skip          = 1.0
        sub_sample    = 25
        sub_ratio     = 0.25
        spread        = True
        slots         = target                  # 15개 슬롯으로 균등 분할

    else:
        # ── 장편 (120초+) — 공격적 균등 분포, fps 적응형 스캔 ─────────────
        nf_scaled     = min(100, int(Nf * d / 60))  # 최대 100개 후보
        target        = MAX_FRAMES              # Ncap=15 유지 (SLA)
        thresh        = 6.0
        step_div      = 2                        # fps//2 → 0.5초마다
        skip          = 2.0
        sub_sample    = 30
        sub_ratio     = 0.25
        spread        = True
        slots         = target

    logger.info(
        f"🎯 동적 설정 [{d:.1f}초] → "
        f"target={target}장 | Nf_scaled={nf_scaled} | "
        f"diff≥{thresh} | step=fps//{step_div} | skip={skip}s | "
        f"자막샘플={sub_sample}({sub_ratio*100:.0f}%) | "
        f"균등분포={'ON('+str(slots)+'슬롯)' if spread else 'OFF'}"
    )
    return {
        "DYNAMIC_TARGET"    : target,
        "NF_SCALED"         : nf_scaled,
        "CHANGE_THRESHOLD"  : thresh,
        "DIFF_STEP_DIVISOR" : step_div,
        "SKIP_SEC"          : skip,
        "SUBTITLE_SAMPLE_N" : sub_sample,
        "SUBTITLE_HIT_RATIO": sub_ratio,
        "TEMPORAL_SPREAD"   : spread,
        "TIME_SLOTS"        : slots,
    }


def calibrate_noise_threshold(
    cap          : cv2.VideoCapture,
    skip_frames  : int = 0,
    sample_count : int = 6,
) -> float:
    """
    영상 ROI absdiff 분포로 노이즈 기준값 측정.

    [v7.1 버그 수정]
    기존: cap.set(..., 0) → 항상 인트로(f=0)부터 샘플링
    문제: 인트로가 정적 화면이면 diff가 극히 낮아 baseline이 낮게 잡힘
          → calibrated = max(1.5, 낮은값) = 1.5 (하한)
          → 인트로 미세변화도 후보 등록 → 초반 가짜 후보 폭발
          → Nf_scaled 상한에 잘려 자막 구간 프레임 누락
    수정: skip_frames 이후부터 샘플링 → 실제 영상 콘텐츠 기반 노이즈 측정

    짧은 영상: grab_interval을 영상 길이에 맞게 줄여 샘플 고갈 방지.
    """
    diffs      = []
    prev_small = None
    orig_pos   = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    vid_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    y0, y1     = int(vid_h * SUBTITLE_TOP), int(vid_h * SUBTITLE_BOTTOM)
    total_f    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # skip_frames 이후 구간에서 샘플링 (인트로 제외)
    sample_start = min(skip_frames, max(0, total_f - 1))
    usable_f     = max(1, total_f - sample_start)
    grab_interval = max(1, min(5, usable_f // max(1, sample_count * 3)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, sample_start)  # ← 핵심 수정: 0이 아닌 skip_frames부터
    grabbed = 0
    while len(diffs) < sample_count:
        if not cap.grab():
            break
        grabbed += 1
        if grabbed % grab_interval != 0:
            continue
        ret, frame = cap.retrieve()
        if not ret:
            continue
        small = cv2.resize(
            cv2.cvtColor(frame[y0:y1, :], cv2.COLOR_BGR2GRAY),
            (120, 30), interpolation=cv2.INTER_AREA
        )
        if prev_small is not None:
            diffs.append(float(np.mean(cv2.absdiff(small, prev_small))))
        prev_small = small

    cap.set(cv2.CAP_PROP_POS_FRAMES, orig_pos)
    if not diffs:
        logger.info(f"🔬 노이즈 샘플 없음(skip={sample_start}f~) → 기본 {CHANGE_THRESHOLD}")
        return CHANGE_THRESHOLD

    baseline   = float(np.mean(sorted(diffs)[:max(1, int(len(diffs) * 0.75))]))
    calibrated = max(1.5, min(7.0, baseline * 1.5))
    logger.info(
        f"🔬 노이즈 보정: skip={sample_start}f~ | "
        f"baseline={baseline:.2f} → {calibrated:.2f} (n={len(diffs)})"
    )
    return calibrated


def _apply_temporal_spread(
    candidates  : List[Tuple[int, float]],
    total_frames: int,
    skip_frames : int,
    n_slots     : int,
) -> List[Tuple[int, float]]:
    """
    긴 영상에서 candidates가 앞부분에 몰리는 현상 방지.
    영상을 n_slots개 시간 구간으로 나눠 각 구간에서 diff_score 1위 후보만 유지.

    Args:
        candidates  : (frame_idx, diff_score) 리스트
        total_frames: 영상 총 프레임 수
        skip_frames : 인트로 스킵 프레임 수
        n_slots     : 시간 구간 수 (보통 target_frames와 동일)

    Returns:
        균등 분포된 candidates (최대 n_slots개)
    """
    if not candidates or n_slots <= 0:
        return candidates

    usable    = total_frames - skip_frames
    slot_size = max(1, usable // n_slots)
    slots: Dict[int, Tuple[int, float]] = {}   # slot_id → best (fidx, diff)

    for fidx, diff in candidates:
        slot_id = min((fidx - skip_frames) // slot_size, n_slots - 1)
        if slot_id not in slots or diff > slots[slot_id][1]:
            slots[slot_id] = (fidx, diff)

    result = sorted(slots.values(), key=lambda x: x[0])
    logger.info(
        f"  [균등 분포] {len(candidates)}개 후보 → "
        f"{n_slots}슬롯 분할 → {len(result)}개 선별"
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 메인 클래스
# ══════════════════════════════════════════════════════════════════════════════

class VideoPreprocessor:

    def __init__(self):
        os.makedirs(VIDEO_DIR, exist_ok=True)
        os.makedirs(FRAME_DIR, exist_ok=True)
        os.makedirs(PREVIEW_DIR, exist_ok=True)
        logger.info("VideoPreprocessor v7.2 초기화")
        logger.info(f"  Nf={Nf} | Ncap={Ncap} | N_final=min(Nf_scaled, {Ncap})")
        logger.info(f"  자막 ROI: {int(SUBTITLE_TOP*100)}~{int(SUBTITLE_BOTTOM*100)}%")
        logger.info(f"  MOCK={MOCK_MODE}")

    # ── 1. 메타데이터 ─────────────────────────────────────────────────────────

    def parse_metadata(self, video_url: str) -> VideoMetadata:
        if MOCK_MODE:
            return self._mock_parse_metadata(video_url)
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True,
                                    'skip_download': True}) as ydl:
                logger.info(f"🔍 메타데이터 수집: {video_url}")
                info = ydl.extract_info(video_url, download=False)
                return VideoMetadata(
                    video_id      = info.get('id', ''),
                    title         = info.get('title', ''),
                    description   = (info.get('description', '') or '')[:500],
                    duration      = info.get('duration', 0) or 0,
                    view_count    = info.get('view_count', 0) or 0,
                    upload_date   = info.get('upload_date', '') or '',
                    channel_name  = info.get('uploader', '') or info.get('channel', ''),
                    thumbnail_url = info.get('thumbnail', ''),
                )
        except Exception as e:
            logger.error(f"❌ 메타데이터 수집 실패: {e}")
            return self._mock_parse_metadata(video_url)

    def _mock_parse_metadata(self, video_url: str) -> VideoMetadata:
        fake = random.choice([
            {"title": "🔥충격🔥 이것만 알면 100만원!", "description": "돈 버는 비법.",
             "duration": 58, "view_count": 1250000, "channel": "돈버는법채널"},
            {"title": "Python 변수와 함수 완벽 마스터", "description": "파이썬 기초.",
             "duration": 55, "view_count": 45000,   "channel": "코딩교육채널"},
        ])
        vid_id = video_url.split('/')[-1] if '/' in video_url \
                 else f"mock_{random.randint(1000,9999)}"
        return VideoMetadata(
            video_id=vid_id, title=fake["title"], description=fake["description"],
            duration=fake["duration"], view_count=fake["view_count"],
            upload_date="2024-01-15", channel_name=fake["channel"],
            thumbnail_url=f"https://img.youtube.com/vi/{vid_id}/maxresdefault.jpg",
        )

    # ── 2. 다운로드 ───────────────────────────────────────────────────────────

    def download_video(self, video_url: str, video_id: str) -> Optional[str]:
        if MOCK_MODE:
            logger.info("🎭 Mock 모드: 다운로드 생략")
            return None
        video_path = os.path.join(VIDEO_DIR, f"{video_id}.mp4")
        if os.path.exists(video_path):
            logger.info(f"♻️ 기존 영상 재사용: {video_path}")
            return video_path
        try:
            import yt_dlp
            logger.info(f"⬇️ 다운로드: {video_url}")
            with yt_dlp.YoutubeDL({'quiet': False, 'no_warnings': True,
                                    'format': 'best[ext=mp4]/best',
                                    'outtmpl': video_path}) as ydl:
                ydl.download([video_url])
            if os.path.exists(video_path):
                logger.info(f"✅ 다운로드 완료: {os.path.getsize(video_path)/1024/1024:.1f}MB")
                return video_path
            logger.error("❌ 다운로드 후 파일 없음")
            return None
        except Exception as e:
            logger.error(f"❌ 다운로드 실패: {e}")
            return None

    # ── 3. 자막 영상 여부 판별 ───────────────────────────────────────────────

    def _detect_has_subtitles(
        self,
        cap         : cv2.VideoCapture,
        total_frames: int,
        fps         : float,
        sample_n    : int   = 20,
        hit_ratio   : float = 0.25,
    ) -> bool:
        """
        균등 샘플 sample_n장으로 자막 존재 비율 투표.
        hit_ratio 이상이면 "자막형" → 자막 필터 적용.
        짧은 영상은 sample_n/hit_ratio를 완화해 오판정 방지.
        """
        orig_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        actual_n = min(sample_n, total_frames)
        step     = max(1, total_frames // actual_n)
        hits     = 0

        for i in range(actual_n):
            fidx = min(i * step, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue
            has_text, _ = _has_subtitle_text(frame)
            if has_text:
                hits += 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, orig_pos)
        ratio    = hits / max(1, actual_n)
        detected = ratio >= hit_ratio
        logger.info(
            f"🔎 자막 판별: {hits}/{actual_n} ({ratio*100:.0f}%) "
            f"기준≥{hit_ratio*100:.0f}% → "
            f"{'자막형 ✓' if detected else '무자막형'}"
        )
        return detected

    # ── 4. 메인 프레임 추출 파이프라인 ───────────────────────────────────────

    def extract_keyframes(self, video_path: str, video_id: str) -> List[KeyframeEvent]:
        """
        v7.0 — 6구간 동적 파라미터 + Nf 상한 실적용 + 시간대별 균등 분포

        [Phase A] grab 루프 → diff 후보 생성 → Nf_scaled 상한 적용
                  → (긴 영상) 시간대별 균등 슬롯 선별
        [Phase B] 자막 필터(자막형) → 선명도 보정 → pHash 중복 제거
        [Step C]  N_final 미달 시 보충 → 병렬 JPEG 저장
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
            fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
            duration     = total_frames / fps
            vid_h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            roi_y0       = int(vid_h * SUBTITLE_TOP)
            roi_y1       = int(vid_h * SUBTITLE_BOTTOM)

            logger.info(f"🎬 {total_frames}f / {fps:.1f}fps / {duration:.1f}초")

            # ── Step 0: 동적 파라미터 결정 ────────────────────────────────
            s             = calculate_dynamic_settings(duration)
            target_frames = s["DYNAMIC_TARGET"]
            nf_scaled     = s["NF_SCALED"]
            DIFF_STEP     = max(2, int(fps / s["DIFF_STEP_DIVISOR"]))
            skip_sec    = s["SKIP_SEC"]
            skip_frames = 0 if skip_sec == 0.0 else max(1, int(fps * skip_sec))  # 1초 미만은 skip 없음

            # calibrate는 반드시 skip_frames 이후 구간으로 측정 (v7.1 버그 수정)
            # 이유: 인트로(f=0~skip)는 정적 화면이 많아 baseline이 낮게 잡혀
            #       threshold가 1.5(하한)로 고정되어 가짜 초반 후보가 폭발함
            diff_threshold = min(
                calibrate_noise_threshold(cap, skip_frames=skip_frames),
                s["CHANGE_THRESHOLD"]
            )
            temporal_spread = s["TEMPORAL_SPREAD"]
            time_slots      = s["TIME_SLOTS"]

            logger.info(
                f"📌 DIFF_STEP={DIFF_STEP}f | skip={skip_frames}f | "
                f"diff≥{diff_threshold:.2f} | Nf_scaled={nf_scaled} | 목표={target_frames}장"
            )

            # ── Step 0b: 자막형 / 무자막형 판별 ──────────────────────────
            has_subtitles = self._detect_has_subtitles(
                cap, total_frames, fps,
                sample_n  = s["SUBTITLE_SAMPLE_N"],
                hit_ratio = s["SUBTITLE_HIT_RATIO"],
            )
            logger.info(f"📋 모드: {'자막형' if has_subtitles else '무자막형'}")

            # ── Phase A: diff 기반 후보 생성 ──────────────────────────────
            candidates: List[Tuple[int, float]] = []
            prev_small  = None
            frame_idx   = skip_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
            t0 = time.time()

            while frame_idx < total_frames:
                # DIFF_STEP-1 프레임은 grab만 (디코딩 없음 → 고속)
                for _ in range(DIFF_STEP - 1):
                    if frame_idx >= total_frames or not cap.grab():
                        frame_idx = total_frames
                        break
                    frame_idx += 1

                if frame_idx >= total_frames:
                    break

                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                roi_gray = cv2.cvtColor(frame[roi_y0:roi_y1, :], cv2.COLOR_BGR2GRAY)
                if cv2.mean(roi_gray)[0] < BLACK_FRAME_THRESHOLD and np.max(roi_gray) < BLACK_FRAME_MAX_THRESHOLD:
                    prev_small = None
                    continue

                curr_small = cv2.resize(roi_gray, (120, 30), interpolation=cv2.INTER_AREA)
                if prev_small is not None:
                    diff = float(np.mean(cv2.absdiff(curr_small, prev_small)))
                    if diff >= diff_threshold:
                        candidates.append((frame_idx, diff))

                prev_small = curr_small

            cap.release()
            logger.info(f"🔎 Phase A: 후보 {len(candidates)}개 ({time.time()-t0:.2f}초)")

            # ── Phase A Fallback: 후보가 없으면 임계값 완화 후 재시도 ─────
            if not candidates:
                logger.warning("⚠️ 후보 없음 → 임계값 50% 완화 후 재시도")
                relaxed_thresh = diff_threshold * 0.5
                cap_fb = cv2.VideoCapture(video_path)
                cap_fb.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
                prev_small, frame_idx = None, skip_frames
                while frame_idx < total_frames:
                    for _ in range(max(1, DIFF_STEP // 2) - 1):
                        if frame_idx >= total_frames or not cap_fb.grab():
                            frame_idx = total_frames; break
                        frame_idx += 1
                    if frame_idx >= total_frames:
                        break
                    ret, frame = cap_fb.read()
                    if not ret:
                        break
                    frame_idx += 1
                    roi_gray = cv2.cvtColor(frame[roi_y0:roi_y1, :], cv2.COLOR_BGR2GRAY)
                    if cv2.mean(roi_gray)[0] < BLACK_FRAME_THRESHOLD and np.max(roi_gray) < BLACK_FRAME_MAX_THRESHOLD:
                        prev_small = None; continue
                    curr_small = cv2.resize(roi_gray, (120, 30), interpolation=cv2.INTER_AREA)
                    if prev_small is not None:
                        diff = float(np.mean(cv2.absdiff(curr_small, prev_small)))
                        if diff >= relaxed_thresh:
                            candidates.append((frame_idx, diff))
                    prev_small = curr_small
                cap_fb.release()
                logger.info(f"  재시도 후 후보: {len(candidates)}개")

            # 그래도 없으면 균등 샘플 fallback
            if not candidates:
                logger.warning("⚠️ 재시도 후에도 후보 없음 → 균등 샘플 fallback")
                step = max(1, total_frames // (target_frames + 1))
                candidates = [
                    (min(skip_frames + i * step, total_frames - 1), 0.0)
                    for i in range(1, target_frames + 1)
                ]

            # ── Nf_scaled 상한 적용 (기획서 "Nv → Nf 필터") ─────────────
            if len(candidates) > nf_scaled:
                candidates.sort(key=lambda x: x[1], reverse=True)
                candidates = candidates[:nf_scaled]
                candidates.sort(key=lambda x: x[0])   # 시간 순 복원
                logger.info(f"  Nf_scaled={nf_scaled} 상한 적용 → {len(candidates)}개")

            # ── 긴 영상 시간대별 균등 분포 적용 ──────────────────────────
            if temporal_spread and time_slots > 0:
                candidates = _apply_temporal_spread(
                    candidates, total_frames, skip_frames, time_slots
                )

            # ── Phase B: 자막 필터 + 선명도 보정 + pHash 중복 제거 ────────
            t1   = time.time()
            cap2 = cv2.VideoCapture(video_path)
            save_targets: List[Tuple[int, float, np.ndarray, int]] = []
            prev_hash: Optional[int] = None

            for cand_fidx, diff_score in candidates:
                if len(save_targets) >= Ncap:
                    break

                cap2.set(cv2.CAP_PROP_POS_FRAMES, cand_fidx)
                ret, frame = cap2.read()
                if not ret:
                    continue

                roi_gray = cv2.cvtColor(frame[roi_y0:roi_y1, :], cv2.COLOR_BGR2GRAY)
                if cv2.mean(roi_gray)[0] < BLACK_FRAME_THRESHOLD and np.max(roi_gray) < BLACK_FRAME_MAX_THRESHOLD:
                    continue

                # 자막형: 자막 필터 적용
                if has_subtitles:
                    ok, sigs = _has_subtitle_text(frame)
                    if not ok:
                        logger.debug(
                            f"  [자막없음 스킵] f={cand_fidx} "
                            f"edge={sigs['edge_density']:.3f}"
                        )
                        continue

                # 선명도 보정 (bounce/fade-in 대응)
                sharp_fidx, frame, sharpness = _pick_sharpest_frame(
                    cap2, cand_fidx, frame, total_frames
                )
                if sharp_fidx != cand_fidx:
                    logger.debug(
                        f"  [선명도 보정] f={cand_fidx}→{sharp_fidx} "
                        f"sharpness={sharpness:.1f}"
                    )
                    roi_gray = cv2.cvtColor(frame[roi_y0:roi_y1, :], cv2.COLOR_BGR2GRAY)
                    if cv2.mean(roi_gray)[0] < BLACK_FRAME_THRESHOLD and np.max(roi_gray) < BLACK_FRAME_MAX_THRESHOLD:
                        continue
                if sharpness < SHARPNESS_MIN_VARIANCE:
                    logger.debug(f"  [선명도 낮음] f={sharp_fidx} s={sharpness:.1f} — 포함")

                # pHash 중복 제거
                curr_hash = _compute_phash(frame)
                if prev_hash is not None:
                    if _hamming_distance(curr_hash, prev_hash) <= PHASH_HAMMING_MAX:
                        logger.debug(f"  [pHash 중복] f={sharp_fidx} → 스킵")
                        continue

                save_targets.append((sharp_fidx, diff_score, frame, curr_hash))
                prev_hash = curr_hash

            cap2.release()
            logger.info(
                f"🗂️ Phase B: {len(save_targets)}/{len(candidates)} 유효 "
                f"({time.time()-t1:.2f}초)"
            )

            # ── Step C: N_final 미달 시 보충 ──────────────────────────────
            if len(save_targets) < target_frames:
                need = target_frames - len(save_targets)
                logger.info(
                    f"🔧 보충: {len(save_targets)}장 → 목표 {target_frames}장 "
                    f"({need}장 부족)"
                )
                used_fidxs = {t[0] for t in save_targets}
                all_hashes = [t[3] for t in save_targets]
                cap3       = cv2.VideoCapture(video_path)

                if has_subtitles:
                    # 기존 프레임 사이 구간을 세밀하게 재스캔
                    sorted_fidxs = sorted(used_fidxs) if used_fidxs \
                                   else [skip_frames]
                    boundaries = (
                        [(skip_frames, sorted_fidxs[0])] +
                        [(sorted_fidxs[i], sorted_fidxs[i+1])
                         for i in range(len(sorted_fidxs)-1)] +
                        [(sorted_fidxs[-1], total_frames)]
                    )
                    fine_step = max(2, DIFF_STEP // 2)
                    extras: List[Tuple[int, float]] = []
                    for seg_s, seg_e in boundaries:
                        f = seg_s + fine_step
                        while f < seg_e and f < total_frames:
                            if f not in used_fidxs:
                                extras.append((f, 0.0))
                            f += fine_step

                    for efidx, _ in extras:
                        if len(save_targets) >= target_frames:
                            break
                        cap3.set(cv2.CAP_PROP_POS_FRAMES, efidx)
                        ret, frame = cap3.read()
                        if not ret:
                            continue
                        roi_gray = cv2.cvtColor(frame[roi_y0:roi_y1, :],
                                                cv2.COLOR_BGR2GRAY)
                        if cv2.mean(roi_gray)[0] < BLACK_FRAME_THRESHOLD and np.max(roi_gray) < BLACK_FRAME_MAX_THRESHOLD:
                            continue
                        ok, _ = _has_subtitle_text(frame)
                        if not ok:
                            continue
                        h = _compute_phash(frame)
                        if any(_hamming_distance(h, eh) <= PHASH_HAMMING_MAX
                               for eh in all_hashes):
                            continue
                        save_targets.append((efidx, 0.0, frame, h))
                        all_hashes.append(h)
                        logger.info(f"  ➕ [자막보충] f={efidx} t={efidx/fps:.1f}s")

                else:
                    # 무자막: 전체 균등 분할 보충
                    step_fill = max(1, total_frames // (target_frames + 1))
                    for i in range(1, target_frames * 3):
                        if len(save_targets) >= target_frames:
                            break
                        ef = min(skip_frames + i * step_fill, total_frames - 1)
                        if ef in used_fidxs:
                            continue
                        cap3.set(cv2.CAP_PROP_POS_FRAMES, ef)
                        ret, frame = cap3.read()
                        if not ret:
                            continue
                        roi_gray = cv2.cvtColor(frame[roi_y0:roi_y1, :],
                                                cv2.COLOR_BGR2GRAY)
                        if cv2.mean(roi_gray)[0] < BLACK_FRAME_THRESHOLD and np.max(roi_gray) < BLACK_FRAME_MAX_THRESHOLD:
                            continue
                        h = _compute_phash(frame)
                        if any(_hamming_distance(h, eh) <= PHASH_HAMMING_MAX
                               for eh in all_hashes):
                            continue
                        save_targets.append((ef, 0.0, frame, h))
                        all_hashes.append(h)
                        used_fidxs.add(ef)
                        logger.info(f"  ➕ [균등보충] f={ef} t={ef/fps:.1f}s")

                cap3.release()
                logger.info(f"✅ 보충 완료: {len(save_targets)}장")

            # 시간 순 정렬 + Ncap 상한
            save_targets.sort(key=lambda x: x[0])
            save_targets = save_targets[:Ncap]

            # ── 병렬 JPEG 저장 ────────────────────────────────────────────
            def _save_one(si, fi, di, fr, ph):
                ts   = fi / fps
                path = self._save_frame(fr, frame_save_dir, si, fi, ts)
                return si, path

            saved_paths: Dict[int, str] = {}
            with ThreadPoolExecutor(max_workers=PARALLEL_SAVE_WORKERS) as ex:
                futs = {
                    ex.submit(_save_one, si, fi, di, fr, ph): si
                    for si, (fi, di, fr, ph) in enumerate(save_targets)
                }
                for fut in as_completed(futs):
                    si, path = fut.result()
                    saved_paths[si] = path

            # ── 이벤트 목록 구성 ──────────────────────────────────────────
            events: List[KeyframeEvent] = []
            for si, (fi, diff, frame, phash) in enumerate(save_targets):
                img_path = saved_paths.get(si, "")
                if not img_path:
                    continue
                ts = fi / fps
                events.append(KeyframeEvent(
                    frame_index   = fi,
                    timestamp_sec = round(ts, 3),
                    timestamp_str = _sec_to_hhmmss(ts),
                    diff_score    = round(diff, 3),
                    phash         = phash,
                    image_path    = img_path,
                ))
                logger.info(f"  ✅ [#{si+1:02d}] t={_sec_to_hhmmss(ts)} | diff={diff:.2f}")

            logger.info(
                f"📊 완료 ({'자막형' if has_subtitles else '무자막형'} | {duration:.0f}초): "
                f"후보 {len(candidates)}개 → 최종 {len(events)}장 "
                f"(T={time.time()-t0:.2f}s)"
            )
            logger.info(f"📤 {len(events)}장 → LMM(Phase 2)")
            return events

        except Exception as e:
            logger.error(f"❌ 키프레임 추출 실패: {e}", exc_info=True)
            return []

    # ── 4. 프레임 저장 (JPEG) ─────────────────────────────────────────────────

    def _save_frame(
        self, frame: np.ndarray, save_dir: str,
        saved_count: int, frame_idx: int, timestamp_sec: float,
    ) -> str:
        roi_gray = cv2.cvtColor(
            frame[int(frame.shape[0]*SUBTITLE_TOP):, :], cv2.COLOR_BGR2GRAY
        )
        if float(cv2.mean(roi_gray)[0]) < BLACK_FRAME_THRESHOLD and np.max(roi_gray) < BLACK_FRAME_MAX_THRESHOLD:
            logger.warning(f"  ⚠️ 검정 프레임 저장 생략 f={frame_idx}")
            return ""

        ts              = _sec_to_hhmmss(timestamp_sec)
        hh, mm, ss_ms   = ts.split(":")
        ss, ms          = ss_ms.split(".")
        filename        = f"frame_{saved_count+1:05d}_{hh}h{mm}m{ss}s{ms}{FRAME_EXT}"
        filepath        = os.path.join(save_dir, filename)

        if not cv2.imwrite(filepath, frame,
                           [cv2.IMWRITE_JPEG_QUALITY, FRAME_JPEG_QUALITY]):
            logger.error(f"❌ 저장 실패: {filepath}")
            return ""

        logger.info(f"  💾 [{saved_count+1:03d}] {filename} | {timestamp_sec:.1f}s")
        return filepath

    # ── 4-b. 기존 인터페이스 호환 ────────────────────────────────────────────

    def _to_base64(self, frame: np.ndarray) -> str:
        _, buf = cv2.imencode('.jpg', frame)
        return base64.b64encode(buf).decode('utf-8')

    def extract_frames(self, video_path: str, video_id: str) -> List[str]:
        """기존 preprocessing.py 호환 래퍼 — base64 리스트 반환"""
        events = self.extract_keyframes(video_path, video_id)
        result = []
        for ev in events:
            if ev.image_path and os.path.exists(ev.image_path):
                with open(ev.image_path, "rb") as f:
                    result.append(base64.b64encode(f.read()).decode("utf-8"))
        return result

    # ── 4-c. ROI / OCR stub ───────────────────────────────────────────────────

    def extract_roi(self, frame) -> Dict[str, np.ndarray]:
        h, w = 100, 200
        return {name: np.zeros((h, w, 3), dtype=np.uint8)
                for name in ROI_CONFIG.keys()}

    def extract_text_ocr(self, roi_regions) -> str:
        return ""

    def calculate_layout_score(self, roi_regions) -> float:
        return 0.0

    # ── 5. Contact Sheet ──────────────────────────────────────────────────────

    def save_contact_sheet(
        self, frame_dir: str, video_id: str,
        metadata: Dict[str, Any] = None,
    ) -> Optional[str]:
        if not os.path.exists(frame_dir):
            return None
        files  = sorted([
            f for f in os.listdir(frame_dir)
            if f.startswith("frame_") and f.endswith((".jpg", ".png"))
        ])
        frames = [cv2.imread(os.path.join(frame_dir, f)) for f in files]
        frames = [f for f in frames if f is not None]
        if not frames:
            return None

        cols = 3
        rows = (len(frames) + cols - 1) // cols
        tw, th, hh, pad = 480, 270, 70, 2
        canvas = np.full((hh + rows*(th+pad), cols*tw, 3), 28, dtype=np.uint8)

        title   = (metadata or {}).get("title", video_id)[:60]
        channel = (metadata or {}).get("channel_name", "")
        cv2.rectangle(canvas, (0,0), (cols*tw, hh), (40,40,40), -1)
        cv2.putText(canvas, f"{title}  |  {channel}  |  {len(frames)}장",
                    (14,26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200,220,255), 1)
        cv2.putText(canvas, f"v7.0 | video_id:{video_id}",
                    (14,54), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140,140,140), 1)

        for idx, fr in enumerate(frames):
            row, col = divmod(idx, cols)
            y0, x0   = hh + row*(th+pad), col*tw
            canvas[y0:y0+th, x0:x0+tw] = cv2.resize(fr, (tw, th))
            cv2.rectangle(canvas, (x0,y0), (x0+tw-1,y0+th-1), (70,70,70), 1)

        out_path = os.path.join(frame_dir, "_contact_sheet.jpg")
        cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
        logger.info(f"🖼️  Contact Sheet: {out_path}")
        return out_path

    # ── 6. 전체 파이프라인 ────────────────────────────────────────────────────

    def process(self, video_url: str) -> PreprocessingResult:
        """전체 전처리 파이프라인 v7.0"""
        start_time     = time.time()
        processing_log = []

        logger.info(f"🚀 전처리 시작: {video_url}")
        processing_log.append(f"전처리 시작: {video_url}")

        # Step 1. 메타데이터
        metadata = self.parse_metadata(video_url)
        processing_log.append(f"✅ 메타데이터: {metadata.title}")
        logger.info(f"📋 {metadata.title} | {metadata.channel_name} | {metadata.duration}s")

        # Step 2. 다운로드
        video_path = self.download_video(video_url, metadata.video_id)
        processing_log.append(
            f"✅ 다운로드: downloads/videos/{metadata.video_id}.mp4"
            if video_path else "⚠️ 다운로드 없음 (Mock 또는 실패)"
        )

        frame_dir = os.path.join(FRAME_DIR, metadata.video_id)

        # Step 3. 키프레임 추출 (v7.0)
        keyframe_events = self.extract_keyframes(video_path, metadata.video_id)

        def _encode(ev: KeyframeEvent) -> str:
            if not ev.image_path or not os.path.exists(ev.image_path):
                return ""
            with open(ev.image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")

        with ThreadPoolExecutor(max_workers=PARALLEL_SAVE_WORKERS) as ex:
            encoded = list(ex.map(_encode, keyframe_events))
        keyframes = [e for e in encoded if e]

        processing_log.append(
            f"✅ 프레임 추출: {len(keyframes)}장 "
            f"(N_final={len(keyframes)} ≤ Ncap={Ncap}, Nf_scaled 적용)"
        )
        processing_log.append("📤 자막 텍스트 추출: LMM(Phase 2) 위임")

        # Step 4. ROI / OCR stub
        mock_frame   = np.zeros((720, 1280, 3), dtype=np.uint8)
        roi_regions  = self.extract_roi(mock_frame)
        ocr_text     = self.extract_text_ocr(roi_regions)
        layout_score = self.calculate_layout_score(roi_regions)

        # 부가 결과물
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
            ocr_text       = ocr_text,
            layout_score   = layout_score,
            roi_data       = {
                "video_path"         : video_path or "",
                "frame_dir"          : frame_dir,
                "frame_count"        : len(keyframes),
                "regions"            : list(roi_regions.keys()),
                "roi_range"          : f"{int(SUBTITLE_TOP*100)}~{int(SUBTITLE_BOTTOM*100)}%",
                "contact_sheet"      : sheet_path or "",
                "version"            : "v7.2",
                "nf"                 : Nf,
                "ncap"               : Ncap,
                "n_final"            : len(keyframes),
                "ocr_engine"         : "none — LMM delegates",
                "extraction_method"  : "opencv_v7.2_fullrange",
                "subtitle_filtered"  : True,
                "sharpness_radius"   : SHARPNESS_SCAN_RADIUS,
                "phash_hamming_max"  : PHASH_HAMMING_MAX,
                "subtitle_filter"    : {
                    "edge_min"  : SUBTITLE_EDGE_DENSITY_MIN,
                    "bright_min": SUBTITLE_BRIGHT_RATIO_MIN,
                    "dark_min"  : SUBTITLE_DARK_RATIO_MIN,
                    "hrun_min"  : SUBTITLE_HRUN_DENSITY_MIN,
                    "min_pass"  : SUBTITLE_SIGNAL_MIN_PASS,
                },
            },
            processing_log = processing_log,
        )


# ══════════════════════════════════════════════════════════════════════════════
# CLI 프리뷰 유틸리티
# ══════════════════════════════════════════════════════════════════════════════

def _get_latest_video_id() -> Optional[str]:
    if not os.path.exists(FRAME_DIR):
        return None
    folders = [f for f in os.listdir(FRAME_DIR)
               if os.path.isdir(os.path.join(FRAME_DIR, f))
               and not f.startswith(".")]
    if not folders:
        return None
    folders.sort(
        key=lambda f: os.path.getmtime(os.path.join(FRAME_DIR, f)),
        reverse=True
    )
    return folders[0]


def make_preview(video_id: str) -> Optional[str]:
    from datetime import datetime
    folder = os.path.join(FRAME_DIR, video_id)
    if not os.path.exists(folder):
        return None
    files  = sorted([f for f in os.listdir(folder)
                     if f.startswith("frame_") and f.endswith((".jpg", ".png"))])
    frames = [(f, cv2.imread(os.path.join(folder, f))) for f in files]
    frames = [(f, img) for f, img in frames if img is not None]
    if not frames:
        return None

    os.makedirs(PREVIEW_DIR, exist_ok=True)
    cols, tw, th, lh, hh, pad = 3, 540, 304, 28, 64, 6
    rows     = (len(frames) + cols - 1) // cols
    canvas_w = cols * (tw + pad) + pad
    canvas_h = hh + rows * (th + lh + pad) + pad
    canvas   = np.full((canvas_h, canvas_w, 3), 22, dtype=np.uint8)

    cv2.rectangle(canvas, (0,0), (canvas_w, hh), (45,45,45), -1)
    cv2.putText(canvas,
                f"v7.0 | {video_id} | {len(frames)}장 | "
                f"Nf={Nf} Ncap={Ncap}",
                (14,30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,220,255), 1)
    cv2.putText(canvas, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                (14,54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130,130,130), 1)

    for idx, (fname, fr) in enumerate(frames):
        row, col = divmod(idx, cols)
        x0 = pad + col * (tw + pad)
        y0 = hh  + pad + row * (th + lh + pad)
        canvas[y0:y0+th, x0:x0+tw] = cv2.resize(fr, (tw, th))
        cv2.rectangle(canvas, (x0-1,y0-1), (x0+tw,y0+th), (70,70,70), 1)
        ly = y0 + th
        cv2.rectangle(canvas, (x0,ly), (x0+tw,ly+lh), (35,35,35), -1)
        cv2.putText(canvas, fname, (x0+6,ly+19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

    out_path = os.path.join(PREVIEW_DIR, f"{video_id}_preview.png")
    if not cv2.imwrite(out_path, canvas):
        return None
    logger.info(f"🖼️  프리뷰: {out_path}")
    return out_path


def open_preview(path: str) -> None:
    import platform, subprocess
    system = platform.system()
    try:
        if   system == "Darwin":  subprocess.run(["open", path])
        elif system == "Windows": os.startfile(path)
        else:                     subprocess.run(["xdg-open", path])
    except Exception as e:
        print(f"⚠️ 자동 열기 실패: {e}\n   직접 열어주세요: {path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="프레임 추출 결과 프리뷰 생성기 (v7.0)"
    )
    parser.add_argument("--id",   help="video_id (없으면 최신 자동 탐지)")
    parser.add_argument("--open", action="store_true", help="생성 후 자동 열기")
    args     = parser.parse_args()
    video_id = args.id or _get_latest_video_id()
    if not video_id:
        print("❌ 분석된 영상이 없습니다.")
    else:
        out = make_preview(video_id)
        if out:
            print(f"✅ 프리뷰: {out}")
            if args.open:
                open_preview(out)
