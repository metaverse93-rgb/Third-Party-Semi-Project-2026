"""
Shorts Check - Streamlit 대시보드
기획서 8-3항 구현: Verdict Card, Reasoning View, Metric Gauge
"""

import streamlit as st
from PIL import Image
import io as _io
import base64 as _b64
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import requests
import random
from datetime import datetime, timedelta
import logging

# ─────────────────────────────────────────
# 페이지 설정 (반드시 최상단)
# ─────────────────────────────────────────
# 로고 base64
_LOGO_B64 = "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"

# ── 파비콘: PIL Image 방식 (Streamlit 공식 지원) ──
import base64 as _b64c, io as _ioc
from PIL import Image as _PILI
_fav_img = _PILI.open(_ioc.BytesIO(_b64c.b64decode(_LOGO_B64))).convert("RGBA")

st.set_page_config(
    page_title="Shorts Check",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="collapsed"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _inject_favicon(b64: str):
    """JS로 브라우저 파비콘을 직접 교체 - 가장 확실한 방법"""
    st.components.v1.html(f"""
    <script>
    (function() {{
      var b64 = "{b64}";
      var url = "data:image/png;base64," + b64;
      
      // 기존 favicon 제거
      var links = document.querySelectorAll("link[rel*='icon']");
      links.forEach(function(l) {{ l.parentNode.removeChild(l); }});
      
      // 새 favicon 삽입
      var link = document.createElement('link');
      link.type = 'image/png';
      link.rel = 'shortcut icon';
      link.href = url;
      document.getElementsByTagName('head')[0].appendChild(link);
      
      // 탭 아이콘 즉시 갱신
      var link2 = document.createElement('link');
      link2.type = 'image/png';  
      link2.rel = 'icon';
      link2.href = url;
      document.getElementsByTagName('head')[0].appendChild(link2);
    }})();
    </script>
    """, height=0, scrolling=False)


# ─────────────────────────────────────────
# 전역 CSS 스타일
# ─────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* ── 4. 전체 배경 흰색 ── */
  .stApp {
    background: #ffffff !important;
    color: #1a1a2e;
  }

  /* ── 상단 헤더/툴바 흰색 ── */
  [data-testid="stToolbar"],
  header[data-testid="stHeader"],
  .stDeployButton {
    background: #ffffff !important;
  }
  header[data-testid="stHeader"] {
    background: #ffffff !important;
    border-bottom: 1px solid rgba(0,0,0,0.08) !important;
  }
  [data-testid="stToolbar"] button,
  [data-testid="stToolbar"] a,
  header[data-testid="stHeader"] button,
  header[data-testid="stHeader"] a,
  .stDeployButton button {
    color: #888888 !important;
    background: transparent !important;
    transition: color 0.2s ease !important;
  }
  [data-testid="stToolbar"] button:hover,
  header[data-testid="stHeader"] button:hover,
  .stDeployButton button:hover {
    color: #333333 !important;
    background: rgba(0,0,0,0.05) !important;
  }

  /* ── 2. 헤더 타이틀 가운데 정렬 ── */
  .main-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -1px;
    color: #1a1a2e;
    margin-bottom: 0;
    line-height: 1.1;
    text-align: center;
  }
  .main-title .title-red {
    color: #cc0000;
  }
  .main-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #888888;
    margin-top: 4px;
    letter-spacing: 0.02em;
    text-align: center;
  }
  .main-header-wrap {
    text-align: center;
    margin-bottom: 24px;
  }

  /* ── Verdict 카드 ── */
  .verdict-card {
    border-radius: 16px;
    padding: 28px 32px;
    position: relative;
    overflow: hidden;
    margin-bottom: 8px;
  }
  .verdict-danger  { background: linear-gradient(135deg, rgba(239,68,68,0.10), rgba(185,28,28,0.06)); border-left: 4px solid #ef4444; }
  .verdict-warning { background: linear-gradient(135deg, rgba(245,158,11,0.10), rgba(180,83,9,0.06));  border-left: 4px solid #f59e0b; }
  .verdict-safe    { background: linear-gradient(135deg, rgba(16,185,129,0.10), rgba(5,150,105,0.06));  border-left: 4px solid #10b981; }

  .verdict-category {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    line-height: 1;
    margin: 0;
  }
  .verdict-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.1rem;
    font-weight: 500;
    margin: 6px 0 0 0;
    opacity: 0.85;
  }
  .verdict-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 10px;
  }
  .badge-danger  { background: rgba(239,68,68,0.15);  color: #dc2626; }
  .badge-warning { background: rgba(245,158,11,0.15); color: #d97706; }
  .badge-safe    { background: rgba(16,185,129,0.15); color: #059669; }

  /* ── Metric 카드 ── */
  .metric-card {
    background: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 14px;
    padding: 20px 22px;
    text-align: center;
  }
  .metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1;
    color: #333333;
  }
  .metric-label {
    font-size: 0.78rem;
    color: #888888;
    margin-top: 6px;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .metric-weight {
    font-size: 0.7rem;
    color: #aaaaaa;
    margin-top: 3px;
  }

  /* ── Reasoning 박스 ── */
  .reasoning-box {
    background: #fff8f8;
    border: 1px solid rgba(204,0,0,0.15);
    border-radius: 12px;
    padding: 18px 22px;
    font-size: 0.92rem;
    line-height: 1.75;
    color: #444444;
  }

  /* ── 섹션 헤더 ── */
  .section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #aaaaaa;
    margin: 28px 0 12px 0;
  }

  /* ── 3. 버튼: 빨간색 계열 ── */
  .stButton > button {
    background: #ffffff !important;
    border: 1.5px solid #cc0000 !important;
    color: #cc0000 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    padding: 10px 16px !important;
    transition: all 0.2s ease !important;
    width: 100%;
  }
  .stButton > button:hover,
  .stButton > button:active,
  .stButton > button:focus {
    background: #cc0000 !important;
    border-color: #cc0000 !important;
    color: #ffffff !important;
    transform: translateY(-1px);
  }

  /* ── 3. 분석 submit 버튼도 빨간색 ── */
  [data-testid="stFormSubmitButton"] > button {
    background: #cc0000 !important;
    border: 1.5px solid #cc0000 !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
  }
  [data-testid="stFormSubmitButton"] > button:hover {
    background: #aa0000 !important;
    border-color: #aa0000 !important;
    color: #ffffff !important;
    transform: translateY(-1px);
  }

  /* ── 1. 사이드바 토글 가능하게 (collapsed 상태 스타일) ── */
  [data-testid="stSidebar"] {
    background: #f8f9fa !important;
    border-right: 1px solid #e9ecef !important;
    transition: all 0.3s ease !important;
  }
  [data-testid="stSidebar"][aria-expanded="false"] {
    display: none !important;
  }
  /* 사이드바 토글 버튼 스타일 */
  [data-testid="collapsedControl"] {
    color: #cc0000 !important;
  }
  button[kind="header"] {
    color: #cc0000 !important;
  }

  /* 스크롤바 */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #f0f0f0; }
  ::-webkit-scrollbar-thumb { background: #999999 !important; border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: #777777 !important; }

  /* ── 입력창 ── */
  .stTextInput > div > div > input {
    background: #ffffff !important;
    border: 1.5px solid #dddddd !important;
    border-radius: 10px !important;
    color: #BC8F8F !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  .stTextInput > div > div > input::placeholder {
    color: #aaaaaa !important;
    opacity: 1;
  }
  .stTextInput > div > div > input:focus {
    border-color: #cc0000 !important;
    box-shadow: 0 0 0 2px rgba(204,0,0,0.12) !important;
    outline: none !important;
  }
  /* Enter 힌트 숨김 */
  .stTextInput small,
  .stTextInput [data-testid="InputInstructions"],
  small[data-testid="InputInstructions"] {
    display: none !important;
    height: 0 !important;
  }

  /* ── 상태 칩 ── */
  .status-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 5px 12px;
    border-radius: 20px;
  }
  .chip-online  { background: rgba(16,185,129,0.1); color: #059669; border: 1px solid rgba(16,185,129,0.2); }
  .chip-offline { background: rgba(239,68,68,0.1);  color: #dc2626; border: 1px solid rgba(239,68,68,0.2); }

  /* ── 분리선 ── */
  hr { border-color: #e9ecef !important; }

  /* ── OCR 텍스트 박스 ── */
  .stTextArea textarea {
    background: #f8f9fa !important;
    border: 1px solid #e9ecef !important;
    color: #666666 !important;
    font-size: 0.85rem !important;
    border-radius: 10px !important;
  }

  /* ── 테이블 ── */
  .stDataFrame { border-radius: 12px !important; overflow: hidden; }

  /* ── Spinner ── */
  .stSpinner > div { border-top-color: #cc0000 !important; }

  /* ── 알림 박스 ── */
  .stAlert { border-radius: 10px !important; }

  /* ── form 포커스 아웃라인 제거 ── */
  [data-testid="stForm"]:focus-within {
    outline: none !important;
    border-color: transparent !important;
  }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# 헬퍼 함수들
# ─────────────────────────────────────────

CATEGORY_META = {
    "C1": {"label": "어그로 / 스팸",    "verdict": "danger",  "icon": "🚨", "badge": "HIGH RISK"},
    "C2": {"label": "공장형 패턴",      "verdict": "danger",  "icon": "🏭", "badge": "HIGH RISK"},
    "C3": {"label": "품질 불량",        "verdict": "warning", "icon": "⚠️",  "badge": "REVIEW"},
    "C4": {"label": "무단 도용",        "verdict": "danger",  "icon": "⚖️",  "badge": "HIGH RISK"},
    "C5": {"label": "정상 영상",        "verdict": "safe",    "icon": "✅",  "badge": "SAFE"},
}

ACTION_MAP = {
    "C1": [("👍", "좋아요", "like"), ("🚫", "채널 추천 안 함", "block_channel"), ("📢", "신고하기", "report"), ("💬", "의견 보내기", "feedback")],
    "C2": [("👍", "좋아요", "like"), ("🚫", "채널 추천 안 함", "block_channel"), ("📢", "신고하기", "report"), ("💬", "의견 보내기", "feedback")],
    "C3": [("👍", "좋아요", "like"), ("🚫", "채널 추천 안 함", "block_channel"), ("📢", "신고하기", "report"), ("💬", "의견 보내기", "feedback")],
    "C4": [("👍", "좋아요", "like"), ("🚫", "채널 추천 안 함", "block_channel"), ("📢", "신고하기", "report"), ("💬", "의견 보내기", "feedback")],
    "C5": [("👍", "좋아요", "like"), ("🚫", "채널 추천 안 함", "block_channel"), ("📢", "신고하기", "report"), ("💬", "의견 보내기", "feedback")],
}


def get_verdict_class(category: str) -> str:
    return CATEGORY_META.get(category, {}).get("verdict", "warning")


def _extract_video_id(url: str) -> str:
    import re
    for p in [r'youtube\.com/shorts/([a-zA-Z0-9_-]+)',
              r'youtu\.be/([a-zA-Z0-9_-]+)',
              r'youtube\.com/watch\?v=([a-zA-Z0-9_-]+)']:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return f"video_{abs(hash(url)) % 100000}"


def make_mock_analysis(video_id: str) -> dict:
    """Mock 분석 데이터 (API/DB 없을 때 사용)"""
    cats = ["C1", "C2", "C3", "C4", "C5"]
    cat = random.choice(cats)
    conf = round(random.uniform(0.55, 0.97), 3)
    s = round(random.uniform(0.5, 0.95), 3)
    o = round(random.uniform(0.4, 0.9), 3)
    a = round(random.uniform(0.5, 0.92), 3)
    ctx = round(s * 0.5 + o * 0.3 + a * 0.2, 3)
    reasoning_map = {
        "C1": "제목에 자극적 키워드('충격', '100만원', '🔥')가 다수 감지되었습니다. 영상 내용과 제목 사이의 의미적 유사도가 낮아 낚시성 콘텐츠로 판단됩니다.",
        "C2": "TTS 특유의 일정한 피치·톤 반복성이 감지되었으며 레이아웃 수치(ROI)가 정적 템플릿 패턴과 일치합니다. 공장형 대량 생산 콘텐츠로 분류합니다.",
        "C3": "OCR 추출 자막과 영상 화면 간 의미적 불일치(S_semantic < 0.6)가 확인되었습니다. 자막-화면 싱크 오류가 품질 문제를 시사합니다.",
        "C4": "역이미지 검색 결과 최초 업로드 시점이 다른 채널과 불일치합니다. 원본 영상의 변조(반전·마스킹) 흔적이 전처리 단계에서 탐지되었습니다.",
        "C5": "명확한 기획 의도와 고유한 편집 스타일이 확인되었습니다. 의미적 유사도·객체 일치도·싱크 점수 모두 정상 범위 내에 있어 우수 콘텐츠로 판정합니다.",
    }
    status_map = {"C1": "AUTO_REJECT", "C2": "AUTO_REJECT", "C3": "HUMAN_REVIEW", "C4": "AUTO_REJECT", "C5": "AUTO_APPROVE"}
    return {
        "video_id": video_id,
        "category": cat,
        "confidence_score": conf,
        "reasoning_log": reasoning_map[cat],
        "status": status_map[cat],
        "model_used": "GPT4o (Mock)",
        "processing_time": round(random.uniform(1.2, 4.8), 2),
        "context_score": ctx,
        "s_semantic": s,
        "o_existence": o,
        "a_sync": a,
        "layout_score": round(random.uniform(0.2, 0.9), 2),
        "raw_ocr_text": "[title_region] 🔥충격🔥 이것만 알면 100만원 [content_region] 돈버는법 클릭 지금바로 [ui_region] 좋아요 구독 알림",
        "channel_name": "Mock Channel",
        "view_count": random.randint(1000, 3000000),
        "duration": random.randint(15, 60),
        "created_at": datetime.now().isoformat(),
    }


def get_analysis_data(video_id: str) -> dict:
    """분석 데이터 가져오기: 세션 → API → Mock"""

    # 1. 세션 캐시
    if (st.session_state.get("latest_result") and
            st.session_state.latest_result.get("video_id") == video_id):
        r = st.session_state.latest_result
        return {
            "video_id": video_id,
            "category": r["analysis_result"]["category"],
            "confidence_score": r["analysis_result"]["confidence_score"],
            "reasoning_log": r["analysis_result"]["reasoning_log"],
            "status": r["analysis_result"]["status"],
            "model_used": "GPT4o (Mock)",
            "processing_time": r.get("processing_time", 1.5),
            "context_score": r["context_score"].get("context_score", 0.75),
            "s_semantic": r["context_score"].get("s_semantic", 0.8),
            "o_existence": r["context_score"].get("o_existence", 0.7),
            "a_sync": r["context_score"].get("a_sync", 0.8),
            "layout_score": r["context_score"].get("layout_score", 0.37),
            "raw_ocr_text": "Mock OCR 텍스트",
            "channel_name": "Mock Channel",
            "view_count": 1250000,
            "duration": 58,
            "created_at": datetime.now().isoformat(),
        }

    # 2. DB 연동 시도
    try:
        from database_manager import DatabaseManager
        from database_models import AnalysisResults, Contents
        _dm = DatabaseManager(mock_mode=True)
        session = _dm.get_session()
        try:
            analysis = session.query(AnalysisResults).join(Contents).filter(
                AnalysisResults.video_id == video_id
            ).order_by(AnalysisResults.created_at.desc()).first()
            if analysis and analysis.content:
                return {
                    "video_id": video_id,
                    "category": analysis.c_category,
                    "confidence_score": analysis.confidence_score,
                    "reasoning_log": analysis.reasoning_log,
                    "status": analysis.status.value,
                    "model_used": analysis.model_used,
                    "processing_time": analysis.processing_time,
                    "context_score": analysis.context_score or 0.75,
                    "s_semantic": analysis.s_semantic or 0.8,
                    "o_existence": analysis.o_existence or 0.7,
                    "a_sync": analysis.a_sync or 0.8,
                    "layout_score": analysis.content.layout_score,
                    "raw_ocr_text": analysis.content.raw_ocr_text,
                    "channel_name": analysis.content.channel_name,
                    "view_count": analysis.content.view_count,
                    "duration": analysis.content.duration,
                    "created_at": analysis.created_at.isoformat(),
                }
        finally:
            session.close()
    except Exception:
        pass

    # 3. Mock fallback
    return make_mock_analysis(video_id)


def submit_feedback(video_id: str, action: str, text: str = "") -> bool:
    try:
        r = requests.post("http://localhost:8000/feedback",
                          json={"video_id": video_id, "action": action, "feedback_text": text},
                          timeout=5)
        return r.status_code == 200
    except Exception:
        return False



def send_feedback_email(video_id: str, email: str, ai_category: str,
                        user_category: str, reason: str) -> bool:
    """의견을 이메일로 전송 (smtplib 사용)"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    # 설정: 본인 Gmail 계정으로 변경하세요
    SMTP_SERVER   = "smtp.gmail.com"
    SMTP_PORT     = 587
    SENDER_EMAIL  = st.secrets.get("FEEDBACK_EMAIL", "")       # .streamlit/secrets.toml
    SENDER_PASS   = st.secrets.get("FEEDBACK_EMAIL_PASS", "")  # 앱 비밀번호
    RECEIVER_EMAIL = st.secrets.get("FEEDBACK_RECEIVER", email)

    if not SENDER_EMAIL or not SENDER_PASS:
        # 이메일 설정 없으면 로그만 남기고 성공 처리
        logger.info(f"[FEEDBACK] video={video_id} ai={ai_category} user={user_category} reason={reason}")
        return True

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[Shorts Check] 분류 의견 — {video_id}"
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = RECEIVER_EMAIL

        html_body = f"""
        <html><body style="font-family:sans-serif;color:#333;max-width:600px;margin:auto">
          <h2 style="color:#cc0000">📋 Shorts Check 분류 의견</h2>
          <table style="width:100%;border-collapse:collapse">
            <tr style="background:#fff8f8">
              <td style="padding:10px;border:1px solid #eee;font-weight:600;width:140px">Video ID</td>
              <td style="padding:10px;border:1px solid #eee"><code>{video_id}</code></td>
            </tr>
            <tr>
              <td style="padding:10px;border:1px solid #eee;font-weight:600">AI 분류 결과</td>
              <td style="padding:10px;border:1px solid #eee"><b>{ai_category}</b></td>
            </tr>
            <tr style="background:#fff8f8">
              <td style="padding:10px;border:1px solid #eee;font-weight:600">사용자 제안 분류</td>
              <td style="padding:10px;border:1px solid #eee;color:#cc0000"><b>{user_category}</b></td>
            </tr>
            <tr>
              <td style="padding:10px;border:1px solid #eee;font-weight:600">제출자 이메일</td>
              <td style="padding:10px;border:1px solid #eee">{email}</td>
            </tr>
            <tr style="background:#fff8f8">
              <td style="padding:10px;border:1px solid #eee;font-weight:600">의견 / 이유</td>
              <td style="padding:10px;border:1px solid #eee">{reason}</td>
            </tr>
            <tr>
              <td style="padding:10px;border:1px solid #eee;font-weight:600">제출 시각</td>
              <td style="padding:10px;border:1px solid #eee">{__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</td>
            </tr>
          </table>
          <p style="color:#aaa;font-size:0.8rem;margin-top:20px">Shorts Check 자동 발송 메일입니다.</p>
        </body></html>
        """
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASS)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())

        logger.info(f"이메일 전송 완료: {RECEIVER_EMAIL}")
        return True

    except Exception as e:
        logger.error(f"이메일 전송 실패: {e}")
        return False

def check_api_health() -> bool:
    try:
        r = requests.get("http://localhost:8000/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────
# 차트 함수
# ─────────────────────────────────────────

def render_radar_chart(data: dict):
    fig = go.Figure()
    labels = ["의미적 유사도<br>(S_semantic)", "객체 존재<br>(O_existence)", "시공간 동기화<br>(A_sync)"]
    values = [
        data.get("s_semantic", 0) * 100,
        data.get("o_existence", 0) * 100,
        data.get("a_sync", 0) * 100,
    ]
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    # 배경 기준선
    for threshold, color in [(75, "rgba(16,185,129,0.08)"), (50, "rgba(245,158,11,0.06)")]:
        fig.add_trace(go.Scatterpolar(
            r=[threshold] * 4,
            theta=labels_closed,
            fill="toself",
            fillcolor=color,
            line=dict(color="rgba(255,255,255,0.04)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(124,131,253,0.18)",
        line=dict(color="#7c83fd", width=2.5),
        marker=dict(size=8, color="#7c83fd", symbol="circle"),
        showlegend=False,
        hovertemplate="%{theta}: %{r:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                ticksuffix="%", tickfont=dict(size=10, color="#999999"),
                gridcolor="rgba(0,0,0,0.10)",
                linecolor="rgba(0,0,0,0.12)",
                tickcolor="rgba(0,0,0,0.15)",
                showline=True,
                gridwidth=1,
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#555555"),
                gridcolor="rgba(0,0,0,0.10)",
                linecolor="rgba(0,0,0,0.12)",
                gridwidth=1,
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=30, b=30),
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_confidence_gauge(confidence: float, category: str):
    verdict = get_verdict_class(category)
    color_map = {"danger": "#ef4444", "warning": "#f59e0b", "safe": "#10b981"}
    color = color_map.get(verdict, "#7c83fd")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={"suffix": "%", "font": {"size": 32, "color": color, "family": "Syne"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#999999",
                     "tickfont": {"size": 10, "color": "#999999"}},
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50],  "color": "rgba(239,68,68,0.08)"},
                {"range": [50, 80], "color": "rgba(245,158,11,0.08)"},
                {"range": [80, 100],"color": "rgba(16,185,129,0.08)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.8, "value": confidence * 100},
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=20),
        height=220,
        font={"color": "#555555"},
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_category_bar():
    """카테고리 분포 가로 막대 차트"""
    cat_counts = safe_db_category_dist()

    colors = {"C1": "#ef4444", "C2": "#f97316", "C3": "#f59e0b", "C4": "#8b5cf6", "C5": "#10b981"}
    labels = [f"{k} {CATEGORY_META[k]['label']}" for k in cat_counts]
    values = list(cat_counts.values())
    bar_colors = [colors[k] for k in cat_counts]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=bar_colors,
        marker=dict(cornerradius=6),
        hovertemplate="%{y}: %{x}건<extra></extra>",
        text=values, textposition="outside",
        textfont=dict(size=11, color="#555555"),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=11, color="#555555"),
                   gridcolor="rgba(0,0,0,0.06)"),
        margin=dict(l=10, r=40, t=10, b=10),
        height=220,
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────
# 화면 렌더링: 분석 리포트
# ─────────────────────────────────────────

def show_report(video_id: str):
    data = get_analysis_data(video_id)
    cat = data["category"]
    meta = CATEGORY_META.get(cat, CATEGORY_META["C5"])
    verdict = meta["verdict"]

    # 뒤로가기
    if st.button("← 메인으로 돌아가기", key="back_btn"):
        st.session_state.show_report = False
        st.session_state.video_id = None
        st.rerun()

    st.markdown("<div class='section-header'>분석 리포트</div>", unsafe_allow_html=True)

    # ── Verdict Card + Gauge ──────────────────
    col_card, col_gauge = st.columns([3, 2])

    with col_card:
        st.markdown(f"""
        <div class="verdict-card verdict-{verdict}">
          <div style="display:flex;align-items:flex-start;gap:18px;">
            <div style="font-size:3.5rem;line-height:1">{meta['icon']}</div>
            <div>
              <p class="verdict-category" style="color:{'#fca5a5' if verdict=='danger' else '#fcd34d' if verdict=='warning' else '#6ee7b7'}">{cat}</p>
              <p class="verdict-label">{meta['label']}</p>
              <span class="verdict-badge badge-{verdict}">{meta['badge']}</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # 영상 정보
        st.markdown(f"""
        <div style="display:flex;gap:24px;margin-top:14px;flex-wrap:wrap;">
          <div><span style="color:#3d4263;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em">채널</span>
               <p style="margin:2px 0 0 0;color:#8b90b5;font-size:0.9rem">{data.get('channel_name','—')}</p></div>
          <div><span style="color:#3d4263;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em">조회수</span>
               <p style="margin:2px 0 0 0;color:#8b90b5;font-size:0.9rem">{data.get('view_count',0):,}회</p></div>
          <div><span style="color:#3d4263;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em">길이</span>
               <p style="margin:2px 0 0 0;color:#8b90b5;font-size:0.9rem">{data.get('duration',0)}초</p></div>
          <div><span style="color:#3d4263;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em">모델</span>
               <p style="margin:2px 0 0 0;color:#8b90b5;font-size:0.9rem">{data.get('model_used','—')}</p></div>
          <div><span style="color:#3d4263;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em">처리 시간</span>
               <p style="margin:2px 0 0 0;color:#8b90b5;font-size:0.9rem">{data.get('processing_time',0):.2f}s</p></div>
        </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        st.markdown("<div class='section-header' style='margin-top:0'>AI 확신도 (Confidence)</div>", unsafe_allow_html=True)
        render_confidence_gauge(data["confidence_score"], cat)
        status_color = {"AUTO_APPROVE": "#10b981", "HUMAN_REVIEW": "#f59e0b",
                        "AUTO_REJECT": "#ef4444", "ANALYSIS_FAILED": "#6b7280"}
        s = data.get("status", "HUMAN_REVIEW")
        st.markdown(f"""
        <div style="text-align:center;margin-top:-8px">
          <span style="font-size:0.78rem;color:{status_color.get(s,'#8b90b5')};
                       background:rgba(0,0,0,0.3);padding:4px 14px;border-radius:20px;
                       border:1px solid {status_color.get(s,'#8b90b5')}40">{s}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── AI 판단 근거 ──────────────────────────
    st.markdown("<div class='section-header'>AI 판단 근거 (Reasoning View)</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='reasoning-box'>💬 {data.get('reasoning_log','—')}</div>", unsafe_allow_html=True)

    # ── Context Score ──────────────────────────
    st.markdown("<div class='section-header'>Context Score 상세 분석</div>", unsafe_allow_html=True)

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    scores_info = [
        (col_m1, "S_semantic",   data.get("s_semantic", 0),    "의미적 유사도", "50%"),
        (col_m2, "O_existence",  data.get("o_existence", 0),   "객체 존재 여부", "30%"),
        (col_m3, "A_sync",       data.get("a_sync", 0),        "시공간 동기화", "20%"),
        (col_m4, "Context Score",data.get("context_score", 0), "통합 점수", "최종"),
    ]
    for col, name, val, desc, weight in scores_info:
        bar_w = int(val * 100)
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div class="metric-value">{val:.3f}</div>
              <div class="metric-label">{desc}</div>
              <div class="metric-weight">가중치 {weight}</div>
              <div style="margin-top:10px;background:rgba(255,255,255,0.05);border-radius:4px;height:4px;">
                <div style="width:{bar_w}%;height:4px;border-radius:4px;
                            background:{'#10b981' if val>=0.75 else '#f59e0b' if val>=0.5 else '#ef4444'};
                            transition:width 0.6s ease;"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── 레이더 차트 ───────────────────────────
    col_radar, col_empty = st.columns([2, 1])
    with col_radar:
        render_radar_chart(data)

    # ── 사용자 액션 ───────────────────────────
    st.markdown("<div class='section-header'>사용자 액션</div>", unsafe_allow_html=True)

    # 세션 초기화
    if "show_opinion_form" not in st.session_state:
        st.session_state.show_opinion_form = False
    if "show_report_form" not in st.session_state:
        st.session_state.show_report_form = False
    if "action_result" not in st.session_state:
        st.session_state.action_result = None

    # 버튼 3개 + 의견보내기
    btn_cols = st.columns(4)
    with btn_cols[0]:
        if st.button("👍 좋아요", use_container_width=True, key="act_like"):
            submit_feedback(video_id, "like")
            st.session_state.action_result = ("success", "좋아요가 기록되었습니다!")
            st.session_state.show_opinion_form = False
    with btn_cols[1]:
        if st.button("🚫 채널 추천 안 함", use_container_width=True, key="act_block"):
            submit_feedback(video_id, "block_channel")
            st.session_state.action_result = ("success", "채널 추천 안 함이 기록되었습니다!")
            st.session_state.show_opinion_form = False
    with btn_cols[2]:
        report_label = "📢 신고하기 ▲" if st.session_state.show_report_form else "📢 신고하기 ▼"
        if st.button(report_label, use_container_width=True, key="act_report"):
            st.session_state.show_report_form = not st.session_state.show_report_form
            st.session_state.show_opinion_form = False
            st.session_state.action_result = None
    with btn_cols[3]:
        # 의견보내기: 토글 방식
        btn_label = "✏️ 의견 보내기 ▲" if st.session_state.show_opinion_form else "✏️ 의견 보내기 ▼"
        if st.button(btn_label, use_container_width=True, key="act_opinion"):
            st.session_state.show_opinion_form = not st.session_state.show_opinion_form
            st.session_state.show_report_form = False
            st.session_state.action_result = None

    # 액션 결과 표시
    if st.session_state.action_result:
        msg_type, msg_text = st.session_state.action_result
        if msg_type == "success":
            st.success(msg_text)
        elif msg_type == "error":
            st.error(msg_text)


    # ── 신고하기 확장 폼 ─────────────────────────
    if st.session_state.show_report_form:
        REPORT_CATEGORIES = [
            ("🔞", "성적인 콘텐츠"),
            ("💀", "폭력적 또는 혐오스러운 콘텐츠"),
            ("😡", "증오 또는 악의적인 콘텐츠"),
            ("😰", "괴롭힘 또는 폭력"),
            ("⚠️", "유해하거나 위험한 행위"),
            ("🆘", "자살, 자해 또는 섭식 장애"),
            ("❌", "잘못된 정보"),
            ("👶", "아동 학대"),
            ("💣", "테러 조장"),
            ("🚫", "스팸 또는 혼동을 야기하는 콘텐츠"),
            ("⚖️", "법적 문제"),
        ]

        if "selected_report" not in st.session_state:
            st.session_state.selected_report = None

        # 선택 상태에 따라 pill 버튼 CSS를 동적으로 생성
        pill_css = ""
        for _, lbl in REPORT_CATEGORIES:
            key = f"rpt_{lbl}"
            if st.session_state.selected_report == lbl:
                pill_css += f"""
                div[data-testid="stButton"]:has(button[data-testid="{key}"] ),
                div[data-testid="stButton"] button[key="{key}"] {{
                    background: #cc0000 !important;
                    color: #ffffff !important;
                    border-color: #cc0000 !important;
                }}"""

        st.markdown(f"""
        <style>
          /* 신고 pill 버튼 공통: 둥글게 */
          .report-pill-area div[data-testid="stButton"] > button {{
            border-radius: 50px !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            padding: 8px 10px !important;
            transition: all 0.15s ease !important;
            border: 1.5px solid #dddddd !important;
            background: #ffffff !important;
            color: #333333 !important;
            min-height: 44px !important;
          }}
          .report-pill-area div[data-testid="stButton"] > button:hover {{
            border-color: #cc0000 !important;
            color: #cc0000 !important;
          }}
        </style>
        <div style="background:#fff8f8;border:1.5px solid rgba(204,0,0,0.18);
                    border-radius:14px;padding:20px 24px 4px 24px;margin-top:12px;">
          <p style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
                    color:#cc0000;margin:0 0 2px 0">📢 신고하기</p>
          <p style="font-size:0.85rem;color:#666;margin:0 0 14px 0">어떤 문제인가요?</p>
        </div>
        """, unsafe_allow_html=True)

        # pill 버튼 영역 (클래스로 감싸서 CSS 타겟팅)
        st.markdown('<div class="report-pill-area">', unsafe_allow_html=True)

        items_per_row = 3
        rows = [REPORT_CATEGORIES[i:i+items_per_row]
                for i in range(0, len(REPORT_CATEGORIES), items_per_row)]

        for row in rows:
            r_cols = st.columns(items_per_row)
            for idx, (icon, lbl) in enumerate(row):
                with r_cols[idx]:
                    is_sel = st.session_state.selected_report == lbl
                    # 선택됐으면 버튼 스타일 인라인으로 주입
                    if is_sel:
                        st.markdown(f"""
                        <style>
                          div[data-testid="stButton"]:has(> button p)
                          + div {{ display:none }}
                        </style>
                        """, unsafe_allow_html=True)
                    btn_text = f"{'✓ ' if is_sel else ''}{icon} {lbl}"
                    if st.button(btn_text, key=f"rpt_{lbl}", use_container_width=True):
                        st.session_state.selected_report = None if is_sel else lbl
                        st.rerun()
                    # 선택 항목은 CSS로 빨간색 덮어쓰기
                    if is_sel:
                        st.markdown(f"""
                        <style>
                          div[data-testid="stButton"]:has(button:contains("✓")) > button {{
                            background: #cc0000 !important;
                            color: #ffffff !important;
                            border-color: #cc0000 !important;
                            box-shadow: 0 2px 8px rgba(204,0,0,0.3) !important;
                          }}
                        </style>
                        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        sub_col, can_col = st.columns(2)
        with sub_col:
            can_submit = st.session_state.selected_report is not None
            if st.button("📤 신고 제출하기", key="report_submit",
                         use_container_width=True, disabled=not can_submit):
                selected = st.session_state.selected_report
                submit_feedback(video_id, "report", f"[신고 유형: {selected}]")
                st.session_state.action_result = ("success", f"✅ '{selected}' 유형으로 신고가 접수되었습니다!")
                st.session_state.show_report_form = False
                st.session_state.selected_report = None
                st.rerun()
        with can_col:
            if st.button("✕ 취소", key="report_cancel", use_container_width=True):
                st.session_state.show_report_form = False
                st.session_state.selected_report = None
                st.rerun()

        if not st.session_state.selected_report:
            st.caption("※ 신고 유형을 하나 선택해주세요.")


    # ── 의견 보내기 확장 폼 ──────────────────────
    if st.session_state.show_opinion_form:
        st.markdown("""
        <div style="background:#fff8f8;border:1.5px solid rgba(204,0,0,0.15);
                    border-radius:14px;padding:24px 28px;margin-top:12px">
          <div style="font-family:'Syne',sans-serif;font-size:0.8rem;font-weight:700;
                      letter-spacing:0.1em;text-transform:uppercase;color:#cc0000;
                      margin-bottom:16px">✏️ 분류 의견 제출</div>
        </div>
        """, unsafe_allow_html=True)

        with st.form("opinion_form", clear_on_submit=True):
            st.markdown(
                "<p style='font-size:0.85rem;color:#666;margin-bottom:4px'>"
                f"AI 분류 결과: <b style='color:#cc0000'>{cat} — {CATEGORY_META[cat]['label']}</b></p>",
                unsafe_allow_html=True
            )

            # 카테고리 직접 선택
            cat_options = {
                "C1 — 어그로 / 스팸": "C1",
                "C2 — 공장형 패턴":   "C2",
                "C3 — 품질 불량":     "C3",
                "C4 — 무단 도용":     "C4",
                "C5 — 정상 영상":     "C5",
            }
            selected_label = st.selectbox(
                "올바른 분류라고 생각하시는 카테고리를 선택해주세요",
                options=list(cat_options.keys()),
                index=list(cat_options.values()).index(cat),
                key="opinion_category"
            )
            user_category = cat_options[selected_label]

            # 이유 작성
            reason = st.text_area(
                "이유를 작성해주세요 (선택)",
                placeholder="예: 영상 내용을 직접 봤는데 실제로는 정상적인 교육 콘텐츠였습니다.",
                height=100,
                key="opinion_reason"
            )

            # 이메일 입력
            email_input = st.text_input(
                "이메일 주소 (결과를 이메일로도 받으시려면 입력)",
                placeholder="example@gmail.com",
                key="opinion_email"
            )

            col_submit, col_cancel = st.columns([1, 1])
            with col_submit:
                submitted_opinion = st.form_submit_button("📤 제출하기", use_container_width=True)
            with col_cancel:
                cancel = st.form_submit_button("✕ 취소", use_container_width=True)

            if submitted_opinion:
                if not reason.strip() and user_category == cat:
                    st.warning("AI 분류 결과와 동일하거나 이유를 입력해주세요.")
                else:
                    # DB 피드백 저장
                    feedback_text = f"[사용자 제안: {user_category}] {reason}"
                    submit_feedback(video_id, "opinion", feedback_text)

                    # 이메일 전송
                    email_sent = False
                    if email_input.strip():
                        email_sent = send_feedback_email(
                            video_id=video_id,
                            email=email_input.strip(),
                            ai_category=f"{cat} — {CATEGORY_META[cat]['label']}",
                            user_category=f"{user_category} — {CATEGORY_META[user_category]['label']}",
                            reason=reason or "(이유 미작성)"
                        )

                    if email_input.strip() and email_sent:
                        st.session_state.action_result = ("success", f"의견이 제출되었고 {email_input.strip()}으로 이메일이 전송되었습니다!")
                    elif email_input.strip() and not email_sent:
                        st.session_state.action_result = ("success", "의견이 기록되었습니다! (이메일 설정 필요 — secrets.toml 확인)")
                    else:
                        st.session_state.action_result = ("success", "의견이 기록되었습니다!")

                    st.session_state.show_opinion_form = False
                    st.rerun()

            if cancel:
                st.session_state.show_opinion_form = False
                st.rerun()

    # ── 기술 세부사항 ──────────────────────────
    with st.expander("🔬 기술적 세부사항", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("<span style='color:#3d4263;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em'>OCR 추출 텍스트</span>", unsafe_allow_html=True)
            st.text_area("ocr", data.get("raw_ocr_text", ""), height=120, label_visibility="collapsed")
        with c2:
            st.markdown("<span style='color:#3d4263;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em'>성능 지표</span>", unsafe_allow_html=True)
            st.json({
                "처리 시간": f"{data.get('processing_time', 0):.2f}s",
                "모델": data.get("model_used", "—"),
                "레이아웃 점수": data.get("layout_score", 0),
                "분석 시각": data.get("created_at", "")[:19],
            })


# ─────────────────────────────────────────
# 화면 렌더링: 메인 대시보드
# ─────────────────────────────────────────

def show_main():
    # ── URL 분석 입력 ──────────────────────────
    st.markdown("<div class='section-header'>영상 분석</div>", unsafe_allow_html=True)

    with st.form("analyze_form"):
        url_col, btn_col = st.columns([5, 1])
        with url_col:
            video_url = st.text_input(
                "url", label_visibility="collapsed",
                placeholder="YouTube Shorts URL을 붙여넣으세요  (예: https://youtube.com/shorts/...)"
            )
        with btn_col:
            submitted = st.form_submit_button("CHECK", use_container_width=True)

    if submitted and video_url.strip():
        with st.spinner("🤖 AI 분석 중... (최대 60초)"):
            try:
                r = requests.post(
                    "http://localhost:8000/analyze",
                    json={"video_url": video_url.strip(), "request_source": "streamlit"},
                    timeout=65
                )
                if r.status_code == 200:
                    result = r.json()
                    st.session_state.latest_result = result
                    st.session_state.video_id = result["video_id"]
                    st.session_state.show_report = True
                    st.rerun()
                else:
                    st.error(f"분석 실패: {r.json().get('detail','알 수 없는 오류')}")
            except requests.exceptions.ConnectionError:
                # 서버 없으면 Mock으로 자동 전환
                vid = _extract_video_id(video_url.strip())
                st.session_state.latest_result = {
                    "video_id": vid,
                    "analysis_result": make_mock_analysis(vid),
                    "context_score": {"context_score": 0.75, "s_semantic": 0.8,
                                      "o_existence": 0.7, "a_sync": 0.8, "layout_score": 0.45},
                    "processing_time": 1.8,
                }
                # latest_result를 analysis_result 구조에 맞게 재구성
                mock = make_mock_analysis(vid)
                st.session_state.latest_result = {
                    "video_id": vid,
                    "analysis_result": {
                        "category": mock["category"],
                        "confidence_score": mock["confidence_score"],
                        "reasoning_log": mock["reasoning_log"],
                        "status": mock["status"],
                    },
                    "context_score": {
                        "context_score": mock["context_score"],
                        "s_semantic": mock["s_semantic"],
                        "o_existence": mock["o_existence"],
                        "a_sync": mock["a_sync"],
                        "layout_score": mock["layout_score"],
                    },
                    "processing_time": mock["processing_time"],
                }
                st.session_state.video_id = vid
                st.session_state.show_report = True
                st.rerun()
            except Exception as e:
                st.error(f"오류: {str(e)}")

    elif submitted:
        st.warning("URL을 입력해주세요.")

    # 바로가기: 최근 결과가 있으면 리포트 버튼 표시
    if st.session_state.get("latest_result"):
        vid = st.session_state.latest_result.get("video_id", "")
        cat = st.session_state.latest_result.get("analysis_result", {}).get("category", "?")
        meta = CATEGORY_META.get(cat, CATEGORY_META["C5"])
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);
                    border-radius:12px;padding:14px 20px;margin-top:6px;display:flex;
                    align-items:center;justify-content:space-between;">
          <div style="display:flex;align-items:center;gap:12px;">
            <span style="font-size:1.5rem">{meta['icon']}</span>
            <div>
              <div style="font-size:0.8rem;color:#3d4263">최근 분석 결과</div>
              <div style="font-size:0.9rem;color:#8b90b5">{vid}</div>
            </div>
          </div>
          <span style="font-size:0.78rem;color:#3d4263">{cat} · {meta['label']}</span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("📊 상세 리포트 보기", key="view_report"):
            st.session_state.video_id = vid
            st.session_state.show_report = True
            st.rerun()

    # ── 요약 통계 카드 ──────────────────────────
    st.markdown("<div class='section-header' style='margin-top:32px'>시스템 현황</div>", unsafe_allow_html=True)

    total_c, total_a, total_f, pending = safe_db_counts()

    s1, s2, s3, s4 = st.columns(4)
    for col, val, lbl, icon in [
        (s1, total_c, "분석된 콘텐츠", "📹"),
        (s2, total_a, "분석 결과",     "🎯"),
        (s3, total_f, "사용자 피드백", "💬"),
        (s4, pending, "검토 대기 (HITL)", "👤"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:1.5rem;margin-bottom:6px">{icon}</div>
              <div class="metric-value">{val:,}</div>
              <div class="metric-label">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)




# ─────────────────────────────────────────
# DB 안전 조회 헬퍼
# ─────────────────────────────────────────

def safe_db_counts() -> tuple:
    """DB 카운트를 안전하게 조회. 실패 시 Mock 반환."""
    try:
        from database_manager import DatabaseManager
        from database_models import Contents, AnalysisResults, UserFeedback, ValidationLabels, ReviewStatus
        _dm = DatabaseManager(mock_mode=True)
        session = _dm.get_session()
        try:
            total_c = session.query(Contents).count()
            total_a = session.query(AnalysisResults).count()
            total_f = session.query(UserFeedback).count()
            pending  = session.query(ValidationLabels).filter(
                ValidationLabels.review_status == ReviewStatus.PENDING).count()
            return total_c, total_a, total_f, pending
        finally:
            session.close()
    except Exception:
        return (random.randint(80, 200), random.randint(80, 200),
                random.randint(10, 80),  random.randint(0, 20))


def safe_db_recent_rows() -> list:
    """최근 분석 이력을 안전하게 조회. 실패 시 Mock 반환."""
    try:
        from database_manager import DatabaseManager
        from database_models import AnalysisResults, Contents
        _dm = DatabaseManager(mock_mode=True)
        session = _dm.get_session()
        try:
            rows = session.query(AnalysisResults).join(Contents).order_by(
                AnalysisResults.created_at.desc()).limit(8).all()
            if not rows:
                raise ValueError("empty")
            return [{
                "Video ID": r.video_id,
                "제목": ((r.content.title or "")[:40] + "…") if r.content and r.content.title else "—",
                "카테고리": r.c_category,
                "신뢰도": f"{r.confidence_score:.2f}",
                "상태": r.status.value,
                "분석 시각": r.created_at.strftime("%m/%d %H:%M"),
            } for r in rows]
        finally:
            session.close()
    except Exception:
        mock_rows = []
        for _ in range(6):
            cat = random.choice(["C1", "C2", "C3", "C4", "C5"])
            mock_rows.append({
                "Video ID": f"mock_{random.randint(10000,99999)}",
                "제목": random.choice(["🔥충격 100만원 비법 공개", "Python 기초 강의",
                                       "TTS 자동생성 영상", "맛집 리뷰 솔직 후기", "무단 복사 의심 영상"]),
                "카테고리": cat,
                "신뢰도": f"{random.uniform(0.55, 0.97):.2f}",
                "상태": CATEGORY_META[cat]["badge"],
                "분석 시각": (datetime.now() - timedelta(minutes=random.randint(1, 180))).strftime("%m/%d %H:%M"),
            })
        return mock_rows


def safe_db_category_dist() -> dict:
    """카테고리 분포를 안전하게 조회. 실패 시 Mock 반환."""
    try:
        from database_manager import DatabaseManager
        from database_models import AnalysisResults
        _dm = DatabaseManager(mock_mode=True)
        session = _dm.get_session()
        try:
            cat_counts = {"C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 0}
            results = session.query(AnalysisResults).order_by(
                AnalysisResults.created_at.desc()).limit(100).all()
            for r in results:
                if r.c_category in cat_counts:
                    cat_counts[r.c_category] += 1
            if sum(cat_counts.values()) == 0:
                raise ValueError("empty")
            return cat_counts
        finally:
            session.close()
    except Exception:
        return {k: random.randint(2, 25) for k in ["C1", "C2", "C3", "C4", "C5"]}


# ─────────────────────────────────────────
# 메인 진입점
# ─────────────────────────────────────────

def render_sidebar():
    """사이드바 렌더링 - 반드시 main() 안에서 호출"""
    with st.sidebar:
        st.markdown("""
        <div style="padding:8px 0 20px 0">
          <p style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:700;
                    color:#cc0000;margin:0;letter-spacing:-0.5px">Shorts</p>
          <p style="font-family:'Syne',sans-serif;font-size:1.15rem;font-weight:700;
                    color:#333333;margin:0;letter-spacing:-0.5px">Check</p>
          <p style="font-size:0.72rem;color:#888888;margin-top:4px;letter-spacing:0.06em">
            멀티모달 LMM 기반 부적합 콘텐츠 판별</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # API 상태
        api_ok = check_api_health()
        chip_class = "chip-online" if api_ok else "chip-offline"
        chip_text  = "● API 서버 정상" if api_ok else "● API 서버 오프라인 (Mock)"
        st.markdown(
            f"<span class='status-chip {chip_class}'>{chip_text}</span>",
            unsafe_allow_html=True
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # ── 아코디언 1: 분류 체계 ──
        with st.expander("📋 분류 체계", expanded=False):
            for k, v in CATEGORY_META.items():
                color = {"danger": "#dc2626", "warning": "#d97706", "safe": "#059669"}[v["verdict"]]
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;padding:6px 0;"
                    f"border-bottom:1px solid #f0f0f0;font-size:0.84rem;'>"
                    f"<span style='font-size:1rem'>{v['icon']}</span>"
                    f"<b style='color:{color}'>{k}</b>"
                    f"<span style='color:#888;font-size:0.8rem'>{v['label']}</span></div>",
                    unsafe_allow_html=True
                )

        # ── 아코디언 2: 카테고리 분포 ──
        with st.expander("📊 카테고리 분포", expanded=False):
            cat_counts = safe_db_category_dist()
            total = sum(cat_counts.values()) or 1
            colors = {"C1": "#ef4444", "C2": "#f97316", "C3": "#f59e0b",
                      "C4": "#8b5cf6", "C5": "#10b981"}
            for k, cnt in cat_counts.items():
                pct = cnt / total * 100
                st.markdown(
                    f"<div style='margin-bottom:8px'>"
                    f"<div style='display:flex;justify-content:space-between;"
                    f"font-size:0.8rem;margin-bottom:3px'>"
                    f"<span style='color:#333;font-weight:600'>{CATEGORY_META[k]['icon']} {k}</span>"
                    f"<span style='color:#888'>{cnt}건 ({pct:.0f}%)</span></div>"
                    f"<div style='background:#f0f0f0;border-radius:4px;height:6px'>"
                    f"<div style='background:{colors[k]};width:{pct:.0f}%;"
                    f"height:6px;border-radius:4px;transition:width 0.3s'></div>"
                    f"</div></div>",
                    unsafe_allow_html=True
                )

        # ── 아코디언 3: 최근 분석 이력 ──
        with st.expander("🕒 최근 분석 이력", expanded=False):
            rows = safe_db_recent_rows()

            CAT_STYLE = {
                "C1": ("#fef2f2", "#dc2626", "어그로"),
                "C2": ("#fff7ed", "#ea580c", "공장형"),
                "C3": ("#fefce8", "#ca8a04", "품질불량"),
                "C4": ("#f5f3ff", "#7c3aed", "무단도용"),
                "C5": ("#f0fdf4", "#16a34a", "정상"),
            }

            for row in rows:
                cat_key = row.get("카테고리", "C5")
                bg, fg, cat_label = CAT_STYLE.get(cat_key, ("#f9f9f9", "#999", cat_key))
                conf = float(row.get("신뢰도", 0))
                conf_color = "#16a34a" if conf >= 0.8 else "#d97706" if conf >= 0.6 else "#dc2626"
                conf_bar = int(conf * 100)

                title = row.get("제목", "—")
                title_short = (title[:18] + "…") if len(title) > 18 else title

                st.markdown(f"""
                <div style="
                    background:#ffffff;
                    border:1px solid #f0f0f0;
                    border-left:3px solid {fg};
                    border-radius:8px;
                    padding:10px 12px;
                    margin-bottom:7px;
                ">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">
                    <span style="
                        background:{bg};color:{fg};
                        font-size:0.7rem;font-weight:700;
                        padding:2px 8px;border-radius:10px;
                        letter-spacing:0.03em
                    ">{cat_key} {cat_label}</span>
                    <span style="font-size:0.7rem;color:#bbb">{row.get('분석 시각','—')}</span>
                  </div>
                  <div style="font-size:0.82rem;color:#333;font-weight:500;margin-bottom:6px;
                              white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                    {title_short}
                  </div>
                  <div style="display:flex;align-items:center;gap:8px">
                    <div style="flex:1;height:4px;background:#f0f0f0;border-radius:2px;overflow:hidden">
                      <div style="width:{conf_bar}%;height:100%;background:{conf_color};border-radius:2px"></div>
                    </div>
                    <span style="font-size:0.72rem;font-weight:700;color:{conf_color};min-width:32px;text-align:right">
                      {conf:.0%}
                    </span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<div class='section-header'>설정</div>", unsafe_allow_html=True)
        auto_refresh = st.checkbox("자동 새로고침 (30s)", value=False)
        if auto_refresh:
            import time as _time
            _time.sleep(30)
            st.rerun()

        if st.button("새로고침", use_container_width=True):
            st.rerun()


def main():
    # ── 세션 초기화 ───────────────────────────
    if "show_report" not in st.session_state:
        st.session_state.show_report = False
    if "video_id" not in st.session_state:
        st.session_state.video_id = None

    # ── 사이드바 (main 안에서 호출) ───────────
    render_sidebar()
    _inject_favicon(_LOGO_B64)

    # ── 헤더 ──────────────────────────────────
    st.markdown("""
    <div class="main-header-wrap">
      <h1 class="main-title"><span class="title-red">Shorts</span> Check</h1>
      <p class="main-subtitle">멀티모달 LMM 기반 부적합 콘텐츠 탐지 및 5단계 분류 시스템</p>
    </div>
    """, unsafe_allow_html=True)

    # ── URL 파라미터 확인 ─────────────────────
    try:
        vid_from_url = st.query_params.get("video_id")
    except Exception:
        vid_from_url = None

    # ── 화면 분기 ─────────────────────────────
    video_id = st.session_state.video_id or vid_from_url
    if (st.session_state.show_report or vid_from_url) and video_id:
        show_report(video_id)
    else:
        show_main()


main()
