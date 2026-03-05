"""
중앙 집중식 설정 관리
모든 임계값, API 설정, 모델 파라미터를 여기서 관리
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv  # ← 이 줄 추가

# .env 파일 로드  # ← 이 줄 추가
load_dotenv()      # ← 이 줄 추가

# 신뢰도 임계값 설정
CONFIDENCE_THRESHOLDS = {
    "auto_approve": 0.8,    # 자동 승인
    "pending": 0.5,         # 사람 검토 필요
    "reject": 0.3           # 자동 거부
}

# 콘텐츠 카테고리 정의
CONTENT_CATEGORIES = {
    "C1": "어그로/스팸",
    "C2": "공장형 패턴", 
    "C3": "품질 불량",
    "C4": "무단 도용",
    "C5": "정상 영상"
}
"""
중앙 집중식 설정 관리
모든 임계값, API 설정, 모델 파라미터를 여기서 관리
"""
import os
from typing import Dict, Any
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 🎯 Mock 모드 설정 (API 할당량 이슈로 기본값 True)
MOCK_MODE = True  # True: Mock 로직 사용, False: 실제 API 호출

# 신뢰도 임계값 설정
CONFIDENCE_THRESHOLDS = {
    "auto_approve": 0.8,    # 자동 승인
    "pending": 0.5,         # 사람 검토 필요
    "reject": 0.3           # 자동 거부
}

# 콘텐츠 카테고리 정의
CONTENT_CATEGORIES = {
    "C1": "어그로/스팸",
    "C2": "공장형 패턴", 
    "C3": "품질 불량",
    "C4": "무단 도용",
    "C5": "정상 영상"
}

# GPT-4o 설정 (Mock 모드에서는 사용하지 않음)
GPT4O_CONFIG = {
    "model": "gpt-3.5-turbo",  # API 접근 가능한 모델
    "api_key": os.getenv("OPENAI_API_KEY", "sk-test-key"),
    "base_url": "https://api.openai.com/v1",
    "timeout": 60,
    "max_tokens": 1000,
    "temperature": 0.1
}

# Qwen2.5-VL 설정 (vLLM 서버)
QWEN_CONFIG = {
    "model": "Qwen2.5-VL-7B-Custom",
    "api_key": "EMPTY",
    "base_url": "http://localhost:8000/v1",
    "timeout": 60,
    "max_tokens": 1000,
    "temperature": 0.1
}

# API 재시도 설정
RETRY_CONFIG = {
    "max_retries": 3,
    "retry_delay": 2,  # 초
    "backoff_factor": 1.5
}

# Context Score 가중치 (기획서 6-1항)
CONTEXT_WEIGHTS = {
    "semantic": 0.5,    # 의미적 유사도
    "existence": 0.3,   # 객체 존재 여부  
    "sync": 0.2         # 시공간 동기화
}

# Mock 모드 분류 규칙 (실제 LMM 대신 사용)
MOCK_CLASSIFICATION_RULES = {
    "C1_keywords": ["100만원", "돈버는법", "클릭", "충격", "대박", "속도위반", "🔥", "진짜", "비밀", "무료"],
    "C2_keywords": ["TTS", "템플릿", "반복", "자동생성", "복사", "따라하기"],
    "C3_keywords": ["불일치", "싱크", "오류", "깨짐", "로딩"],
    "C4_keywords": ["무단", "도용", "복제", "표절", "저작권"],
    "normal_indicators": ["교육", "강의", "리뷰", "일상", "요리", "여행", "음악", "게임"]
}

# 프롬프트 템플릿 (실제 API 사용 시)
SYSTEM_PROMPT = """당신은 유튜브 쇼츠 콘텐츠 분석 전문가입니다.
다음 5가지 카테고리로 영상을 분류해주세요:

C1: 어그로/스팸 - 자극적 제목, 사기성 내용, 낚시 콘텐츠
C2: 공장형 패턴 - 반복적 템플릿, TTS 남용, 저품질 대량생산
C3: 품질 불량 - 내용과 제목 불일치, 영상-음성 싱크 오류
C4: 무단 도용 - 타인 콘텐츠 도용, 저작권 침해 의심
C5: 정상 영상 - 고유한 창작물, 명확한 내러티브

반드시 JSON 형태로 응답해주세요:
{
    "c_category": "C1~C5 중 하나",
    "reasoning_log": "판별 근거 상세 설명",
    "confidence_score": 0.0~1.0 사이 값
}"""

# ROI 추출 설정
ROI_CONFIG = {
    "title_region": (0, 0, 100, 20),      # 제목 영역 (%)
    "content_region": (0, 20, 100, 80),   # 콘텐츠 영역 (%)
    "ui_region": (0, 80, 100, 100),       # UI 영역 (%)
}
# GPT-4o 설정
GPT4O_CONFIG = {
    "model": "gpt-4o",
    "api_key": os.getenv("OPENAI_API_KEY", "sk-test-key"),
    "base_url": "https://api.openai.com/v1",
    "timeout": 60,
    "max_tokens": 1000,
    "temperature": 0.1
}

# Qwen2.5-VL 설정 (vLLM 서버)
QWEN_CONFIG = {
    "model": "Qwen2.5-VL-7B-Custom",
    "api_key": "EMPTY",  # vLLM은 API 키 불필요
    "base_url": "http://localhost:8000/v1",
    "timeout": 60,
    "max_tokens": 1000,
    "temperature": 0.1
}

# API 재시도 설정
RETRY_CONFIG = {
    "max_retries": 3,
    "retry_delay": 2,  # 초
    "backoff_factor": 1.5
}

# Context Score 가중치 (기획서 6-1항)
CONTEXT_WEIGHTS = {
    "semantic": 0.5,    # 의미적 유사도
    "existence": 0.3,   # 객체 존재 여부  
    "sync": 0.2         # 시공간 동기화
}

# 프롬프트 템플릿
SYSTEM_PROMPT = """당신은 유튜브 쇼츠 콘텐츠 분석 전문가입니다.
다음 5가지 카테고리로 영상을 분류해주세요:

C1: 어그로/스팸 - 자극적 제목, 사기성 내용, 낚시 콘텐츠
C2: 공장형 패턴 - 반복적 템플릿, TTS 남용, 저품질 대량생산
C3: 품질 불량 - 내용과 제목 불일치, 영상-음성 싱크 오류
C4: 무단 도용 - 타인 콘텐츠 도용, 저작권 침해 의심
C5: 정상 영상 - 고유한 창작물, 명확한 내러티브

반드시 JSON 형태로 응답해주세요:
{
    "c_category": "C1~C5 중 하나",
    "reasoning_log": "판별 근거 상세 설명",
    "confidence_score": 0.0~1.0 사이 값
}"""
# 기존 내용에 추가

# 기획서 6-2항 KPI 목표치 (SOTA 근거)
PERFORMANCE_TARGETS = {
    "CER": {"target": 0.05, "description": "Character Error Rate (5% 이하)"},
    "ROUGE_L": {"target": 0.80, "description": "ROUGE-L Score (0.80 이상)"},
    "BERT_SCORE": {"target": 0.85, "description": "BERTScore (0.85 이상)"},
    "F1_SCORE": {"target": 0.88, "description": "F1-Score (0.88~0.92)"},
    "CONTEXT_SCORE": {"target": 0.75, "description": "통합 Context Score"}
}

# 성능 로깅 설정
PERFORMANCE_LOGGING = {
    "enable_logging": True,
    "log_format": "json",  # json 또는 csv
    "log_directory": "performance_logs",
    "batch_size": 100,     # N개마다 파일 저장
    "retention_days": 30   # 로그 보관 일수
}

# A/B 테스트 설정
AB_TEST_CONFIG = {
    "enable_ab_test": True,
    "test_ratio": 0.5,     # 50:50 비율
    "minimum_samples": 50,  # 최소 샘플 수
    "significance_level": 0.05  # 통계적 유의수준
}
