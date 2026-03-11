🎯 프로젝트 개요
YouTube Shorts의 급속한 성장과 함께 저품질 콘텐츠와 어그로성 영상이 증가하고 있습니다. 본 프로젝트는 AI 기반 멀티모달 분석을 통해 YouTube Shorts 콘텐츠를 자동으로 분류하고 품질을 평가하는 시스템을 구축했습니다.

🎪 해결하고자 하는 문제
어그로성 콘텐츠: 자극적인 제목과 썸네일로 클릭을 유도하는 저품질 영상
공장형 콘텐츠: TTS 음성과 반복적인 패턴으로 대량 생산되는 영상
콘텐츠 도용: 다른 창작자의 영상을 무단으로 재업로드하는 사례
품질 불일치: 제목과 실제 내용이 일치하지 않는 영상

🎯 프로젝트 목표
YouTube Shorts 영상을 5가지 카테고리로 자동 분류
Context Score 기반의 정량적 품질 평가
실시간 분석 및 사용자 친화적 대시보드 제공
HITL(Human-in-the-loop) 워크플로우를 통한 지속적인 품질 개선

✨ 주요 기능
🤖 AI 기반 멀티모달 분석
GPT-4o ↔ Qwen2.5-VL 모델 교체 지원
OCR 텍스트 추출 및 키프레임 분석
의미적 유사도 분석 및 객체 검출
📊 Context Score 품질 평가
기획서 기반 정량적 평가 지표:

Context Score = S_semantic(50%) + O_existence(30%) + A_sync(20%)
S_semantic: 영상과 텍스트 간 의미적 유사도
O_existence: 텍스트 언급 객체의 영상 내 존재 여부
A_sync: 영상과 음성/자막의 시공간 동기화

🎨 완전한 웹 인터페이스
Streamlit 대시보드: 실시간 분석 및 결과 시각화
레이더 차트: Context Score 구성 요소별 시각화
사용자 액션: 채널 추천 안함, 신고, 피드백 등 실제 동작

🗄️ 데이터 관리 시스템
SQLAlchemy ORM: 6개 테이블 완벽 설계
실시간 성능 로깅: JSON/CSV 형태 자동 저장
A/B 테스트: GPT-4o vs Qwen 통계적 비교
지식 증류: GPT-4o → Qwen 자동 학습 데이터 생성

🏗️ 시스템 아키텍처
┌─────────────────────────────────────────────────────────────┐
│                    사용자 인터페이스                          │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Streamlit 대시보드 │    │   크롬 확장앱    │                │
│  │   (Port 8501)   │    │  (개발 예정)     │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                  마이크로서비스 계층                          │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│ │  사용자 API  │ │  관리자 API  │ │크롬확장 API │            │
│ │(Port 8000) │ │(Port 8001) │ │(Port 8002) │            │
│ │영상 분석    │ │데이터 관리   │ │빠른 분석    │            │
│ │피드백 수집   │ │HITL 워크플로우│ │오버레이    │            │
│ └─────────────┘ └─────────────┘ └─────────────┘            │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                     AI 처리 엔진                            │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │           멀티모달 분석 파이프라인                        │ │
│ │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │ │
│ │  │전처리(Phase1)│ │AI분석(Phase2)│ │평가(Phase3) │      │ │
│ │  │• OCR 추출   │ │• GPT-4o     │ │• Context Score│     │ │
│ │  │• 키프레임   │ │• Qwen2.5-VL │ │• SOTA 지표   │     │ │
│ │  │• ROI 검출   │ │• 분류 C1~C5 │ │• 성능 평가   │     │ │
│ │  └─────────────┘ └─────────────┘ └─────────────┘      │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                    데이터 관리 계층                          │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                SQLAlchemy ORM                          │ │
│ │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │ │
│ │ │   Contents  │ │AnalysisResults│ │ValidationLabels│  │ │
│ │ │콘텐츠 메타데이터│ │  분석 결과    │ │  검증 라벨     │  │ │
│ │ └─────────────┘ └─────────────┘ └─────────────┘      │ │
│ │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │ │
│ │ │PerformanceLogs│ │UserFeedback │ │DistillationBatch│  │ │
│ │ │  성능 로그     │ │사용자 피드백 │ │  지식증류       │  │ │
│ │ └─────────────┘ └─────────────┘ └─────────────┘      │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

📂 프로젝트 구youtube_shorts_detector/
├── 📄 core_files/               # 핵심 파이프라인
│   ├── config.py               # 시스템 설정
│   ├── models.py               # 데이터 모델
│   ├── pipeline.py             # 메인 분석 파이프라인
│   ├── preprocessing.py        # 데이터 전처리
│   ├── content_analyzer.py     # 콘텐츠 분석기
│   └── context_score_calculator.py  # Context Score 계산
│
├── 📡 api_servers/             # 마이크로서비스
│   ├── user_api.py            # 사용자 API (포트 8000)
│   ├── admin_api.py           # 관리자 API (포트 8001)  
│   └── chrome_extension_api.py # 크롬 확장 API (포트 8002)
│
├── 🎨 frontend/               # 사용자 인터페이스
│   └── streamlit_dashboard.py # Streamlit 대시보드 (포트 8501)
│
├── 🗄️ database/              # 데이터 관리
│   ├── database_models.py     # SQLAlchemy 모델
│   ├── database_manager.py    # DB 연결 관리
│   └── knowledge_distillation.py # 지식 증류
│
├── 📊 evaluation/             # 성능 평가
│   ├── evaluation_metrics.py  # SOTA 평가 지표
│   ├── performance_logger.py  # 성능 로깅
│   └── ab_test_framework.py   # A/B 테스트
│
├── 👤 workflows/             # 워크플로우
│   └── hitl_workflow.py      # Human-in-the-loop
│
├── 🧪 tests/                # 테스트
│   ├── test_database_integration.py
│   ├── test_ui_integration.py
│   └── simple_test.py
│
├── 📋 docs/                 # 문서
│   ├── README.md           # 프로젝트 설명서
│   ├── requirements.txt    # 패키지 목록
│   └── .gitignore         # Git 제외 파일
│
└── 🔧 deployment/          # 배포 (선택사항)
    ├── docker-compose.yml
    ├── Dockerfile.fastapi
    └── Dockerfile.streamlit

🚀 빠른 시작
⚡ 1. 환경 설정
# 저장소 클론
git clone https://github.com/yourusername/youtube_shorts_detector.git
cd youtube_shorts_detector

# 가상환경 생성 및 활성화
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux  
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt

🔑 2. 환경변수 설정
# .env 파일 생성 (프로젝트 루트)
echo "OPENAI_API_KEY=your-api-key-here" > .env
💡 Mock 모드 (기본값)에서는 API 키 없이도 모든 기능을 체험할 수 있습니다!

🖥️ 3. 서버 실행
터미널 3개를 열어서 각각 실행:
# 터미널 1: 사용자 API
python user_api.py

# 터미널 2: 관리자 API (선택사항)  
python admin_api.py

# 터미널 3: Streamlit 대시보드 (메인)
streamlit run streamlit_dashboard.py

🌐 4. 접속 확인
🎨 메인 대시보드: http://localhost:8501
📊 사용자 API: http://localhost:8000/docs
🛠️ 관리자 API: http://localhost:8001/docs

🚀 4.1 접속 오류 발생 시, 접속방법
1. 웹 인터페이스 실행
streamlit run app.py
2. API 서버 실행
uvicorn main:app --reload --port 8000
3. Chrome 확장 프로그램 설치
Chrome 확장 프로그램 관리 페이지 접속
개발자 모드 활성화
chrome_extension/ 폴더 로드
4. 로컬에서 실행 시
mkdir -p /Users/jungsoomin/Third-Party-semi-Project-2026/youtube_shorts_detector_final/.streamlit
터미널에 입력(숨김폴더 생성)
touch /Users/jungsoomin/Third-Party-semi-Project-2026/youtube_shorts_detector_final/.streamlit/secrets.toml
터미널에 입력 (toml파일만들기)
cd youtube_shorts_detector_final
해서 위치 고정
OPENAI_API_KEY=sk-나키 streamlit run streamlit_dashboard.py
터미널에 입력 해서 api연결

💡 사용법
🎬 영상 분석하기
Streamlit 대시보드 (http://localhost:8501) 접속
YouTube Shorts URL 입력
예시: https://youtube.com/shorts/example
Enter 키 또는 🔍 분석 시작 버튼 클릭
60초 내 분석 결과 확인
📊 상세 리포트 보기 클릭하여 세부 분석 확인

📊 결과 해석
카테고리 분류:
🚫 C1: 어그로/스팸 콘텐츠 (자극적 제목, 허위 정보)
🏭 C2: 공장형 콘텐츠 (TTS 남용, 반복적 패턴)
⚠️ C3: 품질 불량 (제목-내용 불일치, 조악한 편집)
⚖️ C4: 무단 도용 (저작권 침해 의심)
✅ C5: 정상 영상 (양질의 오리지널 콘텐츠)

Context Score:
🎯 0.75 이상: 우수한 품질
🔶 0.50 ~ 0.74: 보통 품질
🔻 0.50 미만: 품질 개선 필요

신뢰도 레벨:
🟢 High (0.8+): 자동 처리 권장
🟡 Medium (0.5~0.8): 사람 검토 권장
🔴 Low (0.5-): 재분석 필요
🎭 Mock 모드 vs 실제 모드

Mock 모드 (기본값):
✅ API 비용: 0원
✅ 처리 속도: 1-2초
✅ 기능: 완전한 시스템 체험
✅ 안정성: 에러 없는 데모

실제 모드:
💸 API 비용: 분석당 약 $0.01~0.05
⏱️ 처리 속도: 30-60초
🎯 기능: 실제 AI 추론 결과
📡 요구사항: OpenAI API 키 필요

전환 방법:

p# config.py에서 변경
MOCK_MODE = False  # True → False로 변경

📚 API 문서
🔌 주요 엔드포인트
POST /analyze - 영상 분석
{
  "video_url": "https://youtube.com/shorts/example",
  "request_source": "web"
}

응답:
{
  "video_id": "video_12345",
  "analysis_result": {
    "category": "C1", 
    "confidence_score": 0.85,
    "status": "AUTO_REJECT",
    "reasoning_log": "어그로성 키워드 감지..."
  },
  "context_score": {
    "context_score": 0.75,
    "s_semantic": 0.80,
    "o_existence": 0.70, 
    "a_sync": 0.75
  },
  "confidence_level": "high",
  "recommended_actions": ["🚫 채널 추천 안 함", "📢 신고하기"],
  "processing_time": 1.234,
  "report_url": "/report/video_12345"
}

POST /feedback - 사용자 피드백
{
  "video_id": "video_12345",
  "action": "like",
  "feedback_text": "분석이 정확해요!",
  "rating": 5
}

GET /health - 서비스 상태 확인
{
  "status": "healthy",
  "timestamp": "2024-03-05T10:30:00",
  "pipeline_status": "ready"
}
📖 전체 API 문서: http://localhost:8000/docs (Swagger UI)

🛠️ 기술 스택
🤖 AI/ML
OpenAI GPT-4o: 멀티모달 콘텐츠 분석
Qwen2.5-VL: 경량 대안 모델
scikit-learn: 성능 평가 지표
NumPy/Pandas: 데이터 처리

🌐 백엔드
FastAPI: 고성능 API 서버
SQLAlchemy: ORM 및 데이터베이스 관리
Alembic: 데이터베이스 마이그레이션
Pydantic: 데이터 검증

🎨 프론트엔드
Streamlit: 인터랙티브 대시보드
Plotly: 데이터 시각화 (레이더 차트 등)
🗄️ 데이터베이스
SQLite: 개발/데모용 (기본값)
PostgreSQL: 프로덕션 지원

🔧 개발/배포
Docker & Docker Compose: 컨테이너화 배포
Uvicorn: ASGI 서버
python-dotenv: 환경변수 관리

📊 성능 지표
🎯 목표 성능 (SOTA 기준)

생각

🔍 YouTube Shorts Detector
AI 기반 유튜브 쇼츠 콘텐츠 품질 분석 및 자동 분류 시스템

Python
FastAPI
Streamlit
SQLAlchemy
License

📋 목차
프로젝트 개요
주요 기능
시스템 아키텍처
빠른 시작
사용법
API 문서
기술 스택
성능 지표
팀 정보
🎯 프로젝트 개요
YouTube Shorts의 급속한 성장과 함께 저품질 콘텐츠와 어그로성 영상이 증가하고 있습니다. 본 프로젝트는 AI 기반 멀티모달 분석을 통해 YouTube Shorts 콘텐츠를 자동으로 분류하고 품질을 평가하는 시스템을 구축했습니다.

🎪 해결하고자 하는 문제
어그로성 콘텐츠: 자극적인 제목과 썸네일로 클릭을 유도하는 저품질 영상
공장형 콘텐츠: TTS 음성과 반복적인 패턴으로 대량 생산되는 영상
콘텐츠 도용: 다른 창작자의 영상을 무단으로 재업로드하는 사례
품질 불일치: 제목과 실제 내용이 일치하지 않는 영상
🎯 프로젝트 목표
YouTube Shorts 영상을 5가지 카테고리로 자동 분류
Context Score 기반의 정량적 품질 평가
실시간 분석 및 사용자 친화적 대시보드 제공
HITL(Human-in-the-loop) 워크플로우를 통한 지속적인 품질 개선
✨ 주요 기능
🤖 AI 기반 멀티모달 분석
GPT-4o ↔ Qwen2.5-VL 모델 교체 지원
OCR 텍스트 추출 및 키프레임 분석
의미적 유사도 분석 및 객체 검출
📊 Context Score 품질 평가
기획서 기반 정량적 평가 지표:

scss
코드 복사
Context Score = S_semantic(50%) + O_existence(30%) + A_sync(20%)
S_semantic: 영상과 텍스트 간 의미적 유사도
O_existence: 텍스트 언급 객체의 영상 내 존재 여부
A_sync: 영상과 음성/자막의 시공간 동기화
🎨 완전한 웹 인터페이스
Streamlit 대시보드: 실시간 분석 및 결과 시각화
레이더 차트: Context Score 구성 요소별 시각화
사용자 액션: 채널 추천 안함, 신고, 피드백 등 실제 동작
🗄️ 데이터 관리 시스템
SQLAlchemy ORM: 6개 테이블 완벽 설계
실시간 성능 로깅: JSON/CSV 형태 자동 저장
A/B 테스트: GPT-4o vs Qwen 통계적 비교
지식 증류: GPT-4o → Qwen 자동 학습 데이터 생성
🏗️ 시스템 아키텍처
scss
코드 복사
┌─────────────────────────────────────────────────────────────┐
│                    사용자 인터페이스                          │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ Streamlit 대시보드 │    │   크롬 확장앱    │                │
│  │   (Port 8501)   │    │  (개발 예정)     │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                  마이크로서비스 계층                          │
│ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│ │  사용자 API  │ │  관리자 API  │ │크롬확장 API │            │
│ │(Port 8000) │ │(Port 8001) │ │(Port 8002) │            │
│ │영상 분석    │ │데이터 관리   │ │빠른 분석    │            │
│ │피드백 수집   │ │HITL 워크플로우│ │오버레이    │            │
│ └─────────────┘ └─────────────┘ └─────────────┘            │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                     AI 처리 엔진                            │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │           멀티모달 분석 파이프라인                        │ │
│ │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │ │
│ │  │전처리(Phase1)│ │AI분석(Phase2)│ │평가(Phase3) │      │ │
│ │  │• OCR 추출   │ │• GPT-4o     │ │• Context Score│     │ │
│ │  │• 키프레임   │ │• Qwen2.5-VL │ │• SOTA 지표   │     │ │
│ │  │• ROI 검출   │ │• 분류 C1~C5 │ │• 성능 평가   │     │ │
│ │  └─────────────┘ └─────────────┘ └─────────────┘      │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────────┐
│                    데이터 관리 계층                          │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │                SQLAlchemy ORM                          │ │
│ │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │ │
│ │ │   Contents  │ │AnalysisResults│ │ValidationLabels│  │ │
│ │ │콘텐츠 메타데이터│ │  분석 결과    │ │  검증 라벨     │  │ │
│ │ └─────────────┘ └─────────────┘ └─────────────┘      │ │
│ │ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │ │
│ │ │PerformanceLogs│ │UserFeedback │ │DistillationBatch│  │ │
│ │ │  성능 로그     │ │사용자 피드백 │ │  지식증류       │  │ │
│ │ └─────────────┘ └─────────────┘ └─────────────┘      │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
📂 프로젝트 구조
python
코드 실행
코드 복사
youtube_shorts_detector/
├── 📄 core_files/               # 핵심 파이프라인
│   ├── config.py               # 시스템 설정
│   ├── models.py               # 데이터 모델
│   ├── pipeline.py             # 메인 분석 파이프라인
│   ├── preprocessing.py        # 데이터 전처리
│   ├── content_analyzer.py     # 콘텐츠 분석기
│   └── context_score_calculator.py  # Context Score 계산
│
├── 📡 api_servers/             # 마이크로서비스
│   ├── user_api.py            # 사용자 API (포트 8000)
│   ├── admin_api.py           # 관리자 API (포트 8001)  
│   └── chrome_extension_api.py # 크롬 확장 API (포트 8002)
│
├── 🎨 frontend/               # 사용자 인터페이스
│   └── streamlit_dashboard.py # Streamlit 대시보드 (포트 8501)
│
├── 🗄️ database/              # 데이터 관리
│   ├── database_models.py     # SQLAlchemy 모델
│   ├── database_manager.py    # DB 연결 관리
│   └── knowledge_distillation.py # 지식 증류
│
├── 📊 evaluation/             # 성능 평가
│   ├── evaluation_metrics.py  # SOTA 평가 지표
│   ├── performance_logger.py  # 성능 로깅
│   └── ab_test_framework.py   # A/B 테스트
│
├── 👤 workflows/             # 워크플로우
│   └── hitl_workflow.py      # Human-in-the-loop
│
├── 🧪 tests/                # 테스트
│   ├── test_database_integration.py
│   ├── test_ui_integration.py
│   └── simple_test.py
│
├── 📋 docs/                 # 문서
│   ├── README.md           # 프로젝트 설명서
│   ├── requirements.txt    # 패키지 목록
│   └── .gitignore         # Git 제외 파일
│
└── 🔧 deployment/          # 배포 (선택사항)
    ├── docker-compose.yml
    ├── Dockerfile.fastapi
    └── Dockerfile.streamlit
🚀 빠른 시작
⚡ 1. 환경 설정
bash
코드 복사
# 저장소 클론
git clone https://github.com/yourusername/youtube_shorts_detector.git
cd youtube_shorts_detector

# 가상환경 생성 및 활성화
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux  
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
🔑 2. 환경변수 설정
bash
코드 복사
# .env 파일 생성 (프로젝트 루트)
echo "OPENAI_API_KEY=your-api-key-here" > .env
💡 Mock 모드 (기본값)에서는 API 키 없이도 모든 기능을 체험할 수 있습니다!

🖥️ 3. 서버 실행
터미널 3개를 열어서 각각 실행:

bash
코드 복사
# 터미널 1: 사용자 API
python user_api.py

# 터미널 2: 관리자 API (선택사항)  
python admin_api.py

# 터미널 3: Streamlit 대시보드 (메인)
streamlit run streamlit_dashboard.py
🌐 4. 접속 확인
🎨 메인 대시보드: http://localhost:8501
📊 사용자 API: http://localhost:8000/docs
🛠️ 관리자 API: http://localhost:8001/docs
💡 사용법
🎬 영상 분석하기
Streamlit 대시보드 (http://localhost:8501) 접속
YouTube Shorts URL 입력
javascript
코드 실행
코드 복사
예시: https://youtube.com/shorts/example
Enter 키 또는 🔍 분석 시작 버튼 클릭
60초 내 분석 결과 확인
📊 상세 리포트 보기 클릭하여 세부 분석 확인
📊 결과 해석
카테고리 분류:
🚫 C1: 어그로/스팸 콘텐츠 (자극적 제목, 허위 정보)
🏭 C2: 공장형 콘텐츠 (TTS 남용, 반복적 패턴)
⚠️ C3: 품질 불량 (제목-내용 불일치, 조악한 편집)
⚖️ C4: 무단 도용 (저작권 침해 의심)
✅ C5: 정상 영상 (양질의 오리지널 콘텐츠)
Context Score:
🎯 0.75 이상: 우수한 품질
🔶 0.50 ~ 0.74: 보통 품질
🔻 0.50 미만: 품질 개선 필요
신뢰도 레벨:
🟢 High (0.8+): 자동 처리 권장
🟡 Medium (0.5~0.8): 사람 검토 권장
🔴 Low (0.5-): 재분석 필요
🎭 Mock 모드 vs 실제 모드
Mock 모드 (기본값):
✅ API 비용: 0원
✅ 처리 속도: 1-2초
✅ 기능: 완전한 시스템 체험
✅ 안정성: 에러 없는 데모
실제 모드:
💸 API 비용: 분석당 약 $0.01~0.05
⏱️ 처리 속도: 30-60초
🎯 기능: 실제 AI 추론 결과
📡 요구사항: OpenAI API 키 필요
전환 방법:

python
코드 실행
코드 복사
# config.py에서 변경
MOCK_MODE = False  # True → False로 변경
📚 API 문서
🔌 주요 엔드포인트
POST /analyze - 영상 분석
json
코드 복사
{
  "video_url": "https://youtube.com/shorts/example",
  "request_source": "web"
}
응답:

json
코드 복사
{
  "video_id": "video_12345",
  "analysis_result": {
    "category": "C1", 
    "confidence_score": 0.85,
    "status": "AUTO_REJECT",
    "reasoning_log": "어그로성 키워드 감지..."
  },
  "context_score": {
    "context_score": 0.75,
    "s_semantic": 0.80,
    "o_existence": 0.70, 
    "a_sync": 0.75
  },
  "confidence_level": "high",
  "recommended_actions": ["🚫 채널 추천 안 함", "📢 신고하기"],
  "processing_time": 1.234,
  "report_url": "/report/video_12345"
}
POST /feedback - 사용자 피드백
json
코드 복사
{
  "video_id": "video_12345",
  "action": "like",
  "feedback_text": "분석이 정확해요!",
  "rating": 5
}
GET /health - 서비스 상태 확인
json
코드 복사
{
  "status": "healthy",
  "timestamp": "2024-03-05T10:30:00",
  "pipeline_status": "ready"
}
📖 전체 API 문서: http://localhost:8000/docs (Swagger UI)

🛠️ 기술 스택
🤖 AI/ML
OpenAI GPT-4o: 멀티모달 콘텐츠 분석
Qwen2.5-VL: 경량 대안 모델
scikit-learn: 성능 평가 지표
NumPy/Pandas: 데이터 처리

🌐 백엔드
FastAPI: 고성능 API 서버
SQLAlchemy: ORM 및 데이터베이스 관리
Alembic: 데이터베이스 마이그레이션
Pydantic: 데이터 검증

🎨 프론트엔드
Streamlit: 인터랙티브 대시보드
Plotly: 데이터 시각화 (레이더 차트 등)

🗄️ 데이터베이스
SQLite: 개발/데모용 (기본값)
PostgreSQL: 프로덕션 지원

🔧 개발/배포
Docker & Docker Compose: 컨테이너화 배포
Uvicorn: ASGI 서버
python-dotenv: 환경변수 관리

📊 성능 지표
🎯 목표 성능 (SOTA 기준)
지표	목표값	현재 상태	설명
CER	≤ 5%	🎯 Mock: 4.5%	Character Error Rate (문자 오류율)
ROUGE-L	≥ 0.80	🎯 Mock: 0.82	텍스트 요약 품질 평가
BERTScore	≥ 0.85	🎯 Mock: 0.86	의미적 유사도 평가
F1-Score	0.88~0.92	🎯 Mock: 0.89	분류 정확도
Context Score	≥ 0.75	🎯 Mock: 0.78	통합 맥락 점수
처리 시간	≤ 60초	⚡ Mock: 1-2초	사용자 응답 시간

📈 실시간 모니터링
성능 로그: 모든 분석 결과 자동 저장 (JSON/CSV)
A/B 테스트: GPT-4o vs Qwen 성능 비교
사용자 피드백: 실시간 만족도 추적
에러 모니터링: 장애 상황 즉시 감지

🧪 테스트
단위 테스트 실행
# 전체 통합 테스트
python test_database_integration.py

# UI 테스트  
python test_ui_integration.py

# 간단한 기능 테스트
python simple_test.py

성능 벤치마크
# A/B 테스트 실행
python ab_test_framework.py

# 성능 지표 확인
python evaluation_metrics.py

🐳 Docker 배포 (선택사항)
전체 시스템 실행

# 모든 서비스 한 번에 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 서비스 중단
docker-compose down

개별 서비스 실행
# Streamlit 대시보드만 실행
docker-compose up streamlit-dashboard

# 사용자 API만 실행  
docker-compose up user-api

🤝 기여하기
개발 참여 방법
Fork the Project
Feature Branch 생성 (git checkout -b feature/AmazingFeature)
Commit 변경사항 (git commit -m 'Add some AmazingFeature')
Push to the Branch (git push origin feature/AmazingFeature)
Pull Request 열기

이슈 리포트
🐛 버그 발견: Issues 탭에서 버그 리포트 작성
💡 기능 제안: Feature Request 템플릿 사용
📚 문서 개선: Documentation 라벨 추가

👥 팀 정보
🎯 프로젝트 팀 (4명)
📋 프로젝트 리더: 시스템 설계, 문서화, 발표 준비
🔧 백엔드 개발자: API 안정성, 성능 최적화
🎨 프론트엔드 개발자: UI/UX 개선, 시각화
🧪 QA 엔지니어: 테스트, 품질 보증

🏗️ 기술 아키텍처 분담
AI/ML Pipeline: GPT-4o, Qwen2.5-VL 멀티모달 분석
Backend Services: FastAPI 마이크로서비스 아키텍처
Frontend: Streamlit 기반 인터랙티브 대시보드
Database: SQLAlchemy ORM + Alembic 마이그레이션
DevOps: Docker Compose 컨테이너 오케스트레이션

📅 개발 일정
기간: 2024년 3월 (11일 완료 목표)
방식: 애자일 스프린트, 매일 스탠드업
목표: 실제 서비스 수준의 MVP 완성

📄 라이선스
이 프로젝트는 MIT License 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.

📞 문의 및 지원
📧 Email: your.email@domain.com
🐙 GitHub: https://github.com/yourusername/youtube_shorts_detector
📱 Issues: 프로젝트의 Issues 탭에서 질문 및 버그 리포트

🙏 감사의 말
OpenAI: GPT-4o API 제공
Alibaba Cloud: Qwen2.5-VL 모델 제공
Streamlit 커뮤니티: 훌륭한 대시보드 프레임워크
FastAPI 팀: 고성능 웹 프레임워크

⭐ Star History
만약 이 프로젝트가 도움이 되었다면 ⭐를 눌러주세요!

