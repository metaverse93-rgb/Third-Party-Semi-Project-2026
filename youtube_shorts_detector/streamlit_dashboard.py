"""
Streamlit 상세 분석 리포트 대시보드
기획서 8-3항 구현: st.status, st.metric, 사용자 액션 버튼
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import requests
from datetime import datetime
import logging

# 페이지 설정
st.set_page_config(
    page_title="YouTube Shorts Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 기존 모듈들
from database_manager import db_manager
from database_models import Contents, AnalysisResults, UserFeedback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """메인 대시보드"""
    
    # 헤더
    st.title("🔍 YouTube Shorts Detector")
    st.markdown("---")
    
    # 세션 상태 초기화
    if "show_report" not in st.session_state:
        st.session_state.show_report = False
    if "video_id" not in st.session_state:
        st.session_state.video_id = None
    
    # URL 파라미터에서 video_id 가져오기 (수정됨)
    try:
        query_params = st.query_params
        video_id_from_url = query_params.get("video_id", None)
    except AttributeError:
        query_params = st.experimental_get_query_params()
        video_id_from_url = query_params.get("video_id", [None])[0]
    
    # 우선순위: 세션 상태 > URL 파라미터
    video_id = st.session_state.video_id or video_id_from_url
    
    # 뒤로가기 버튼 (리포트 화면에서)
    if st.session_state.show_report and video_id:
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("← 메인으로"):
                st.session_state.show_report = False
                st.session_state.video_id = None
                st.rerun()
    
    # 화면 표시 결정
    if st.session_state.show_report and video_id:
        show_video_analysis_report(video_id)
    elif video_id_from_url:  # URL에서 직접 접근
        show_video_analysis_report(video_id_from_url)
    else:
        show_main_dashboard()
def show_video_analysis_report(video_id: str):
    """특정 영상 분석 리포트 표시"""
    
    st.header(f"📹 영상 분석 리포트")
    st.markdown(f"**Video ID:** `{video_id}`")
    
    # 분석 결과 조회
    analysis_data = get_analysis_data(video_id)
    
    if not analysis_data:
        st.error("분석 데이터를 찾을 수 없습니다.")
        if st.button("🔄 새로 분석하기"):
            st.rerun()
        return
    
    # 기획서 8-3항 구현: st.status로 결과별 색상
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 분석 결과 상태 표시
        status_color = get_status_color(analysis_data["category"])
        
        with st.status(f"분석 결과: {analysis_data['category']}", state=status_color):
            st.write(f"**신뢰도:** {analysis_data['confidence_score']:.2f}")
            st.write(f"**상태:** {analysis_data['status']}")
            st.write(f"**분석 근거:** {analysis_data['reasoning_log']}")
    
    with col2:
        # 영상 정보
        st.info(f"""
        **채널:** {analysis_data.get('channel_name', 'Unknown')}
        **조회수:** {analysis_data.get('view_count', 0):,}회
        **길이:** {analysis_data.get('duration', 0)}초
        """)
    
    # 기획서 8-3항 구현: st.metric으로 3대 맥락 점수 게이지
    st.markdown("### 📊 Context Score 상세 분석")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🧠 의미적 유사도 (S_semantic)",
            value=f"{analysis_data.get('s_semantic', 0.0):.3f}",
            delta=f"가중치: 50%",
            help="영상과 텍스트 간의 의미적 일치도"
        )
    
    with col2:
        st.metric(
            label="👁️ 객체 존재 여부 (O_existence)", 
            value=f"{analysis_data.get('o_existence', 0.0):.3f}",
            delta=f"가중치: 30%",
            help="텍스트에 언급된 객체가 영상에 실제 존재하는지"
        )
    
    with col3:
        st.metric(
            label="🎬 시공간 동기화 (A_sync)",
            value=f"{analysis_data.get('a_sync', 0.0):.3f}",
            delta=f"가중치: 20%",
            help="영상과 음성/자막의 시간적 동기화"
        )
    
    with col4:
        context_score = analysis_data.get('context_score', 0.0)
        st.metric(
            label="🎯 통합 Context Score",
            value=f"{context_score:.3f}",
            delta="최종 점수",
            help="S_semantic(50%) + O_existence(30%) + A_sync(20%)"
        )
    
    # Context Score 시각화
    show_context_score_chart(analysis_data)
    
    # 기획서 8-3항 구현: 사용자 액션 버튼들 (실제 동작)
    st.markdown("### 🎬 사용자 액션")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚫 채널 추천 안 함", use_container_width=True):
            submit_feedback(video_id, "block_channel", "사용자가 채널 추천 거부")
            st.success("피드백이 전송되었습니다!")
    
    with col2:
        if st.button("📢 신고하기", use_container_width=True):
            submit_feedback(video_id, "report", "사용자가 콘텐츠 신고")
            st.success("신고가 접수되었습니다!")
    
    with col3:
        if st.button("💬 의견 보내기", use_container_width=True):
            feedback_text = st.text_input("의견을 입력하세요:", key="feedback_input")
            if feedback_text:
                submit_feedback(video_id, "feedback", feedback_text)
                st.success("의견이 전송되었습니다!")
    
    with col4:
        if st.button("👍 좋아요", use_container_width=True):
            submit_feedback(video_id, "like", "사용자가 콘텐츠 좋아요")
            st.success("좋아요가 반영되었습니다!")
    
    # OCR 텍스트 및 기술적 세부사항
    with st.expander("🔍 기술적 세부사항", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**OCR 추출 텍스트:**")
            st.text_area(
                "extracted_text",
                analysis_data.get('raw_ocr_text', ''),
                height=150,
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("**성능 지표:**")
            st.json({
                "처리 시간": f"{analysis_data.get('processing_time', 0):.2f}초",
                "모델": analysis_data.get('model_used', 'Unknown'),
                "레이아웃 점수": analysis_data.get('layout_score', 0.0),
                "생성 시간": analysis_data.get('created_at', '')
            })

def show_main_dashboard():
    """메인 대시보드 (영상 분석 요청)"""
    
    st.header("🎬 YouTube Shorts 영상 분석")
    
    # Form을 사용해서 Enter 키 지원
    with st.form(key="video_analysis_form"):
        video_url = st.text_input(
            "YouTube Shorts URL을 입력하세요:",
            placeholder="https://youtube.com/shorts/...",
            value=""
        )
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            analyze_button = st.form_submit_button("🔍 분석 시작", use_container_width=True)
        
        with col2:
            st.markdown("*Enter 키를 눌러서 분석하세요*")
    
    # 폼 제출 시 분석 실행
    if analyze_button and video_url.strip():
        with st.spinner("영상 분석 중... (최대 60초)"):
            try:
                # User API 호출
                response = requests.post(
                    "http://localhost:8000/analyze",
                    json={
                        "video_url": video_url.strip(),
                        "request_source": "streamlit"
                    },
                    timeout=65
                )
                
                if response.status_code == 200:
                    result = response.json()
                    st.success("분석이 완료되었습니다!")
                    
                    # 결과 요약 표시
                    st.json(result)
                    
                    # 세션에 결과 저장
                    st.session_state.latest_result = result
                    
                else:
                    error_detail = response.json().get('detail', '알 수 없는 오류') if response.content else '서버 응답 없음'
                    st.error(f"분석 실패: {error_detail}")
                    
            except requests.exceptions.Timeout:
                st.error("분석 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.")
            except requests.exceptions.ConnectionError:
                st.error("API 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.")
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
    
    elif analyze_button and not video_url.strip():
        st.warning("YouTube URL을 입력해주세요!")
    
    # 최근 결과가 있으면 상세 리포트 버튼 표시
    if "latest_result" in st.session_state and st.session_state.latest_result:
        st.markdown("---")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if st.button("📊 상세 리포트 보기", use_container_width=True):
                st.session_state.video_id = st.session_state.latest_result["video_id"]
                st.session_state.show_report = True
                st.rerun()
        
        with col2:
            st.info("위 분석 결과의 상세 리포트를 볼 수 있습니다.")
    
    # 최근 분석 결과 표시
    show_recent_analyses()
    
def get_analysis_data(video_id: str) -> dict:
    """분석 데이터 조회 (세션 우선)"""
    
    # 1. 먼저 세션 상태에서 확인 (최우선)
    if ("latest_result" in st.session_state and 
        st.session_state.latest_result and 
        st.session_state.latest_result.get("video_id") == video_id):
        
        result = st.session_state.latest_result
        
        return {
            "video_id": video_id,
            "category": result["analysis_result"]["category"],
            "confidence_score": result["analysis_result"]["confidence_score"],
            "reasoning_log": result["analysis_result"]["reasoning_log"],
            "status": result["analysis_result"]["status"],
            "model_used": "GPT4o (Mock)",
            "processing_time": result.get("processing_time", 1.5),
            "context_score": result["context_score"].get("context_score", 0.75),
            "s_semantic": result["context_score"].get("s_semantic", 0.8),
            "o_existence": result["context_score"].get("o_existence", 0.7),
            "a_sync": result["context_score"].get("a_sync", 0.8),
            "layout_score": result["context_score"].get("layout_score", 0.37),
            "raw_ocr_text": "Mock OCR 텍스트: TTS로 만든 어그로성 콘텐츠입니다.",
            "channel_name": "Mock Channel",
            "view_count": 1250000,
            "duration": 58,
            "created_at": datetime.now().isoformat()
        }
    
    # 2. 데이터베이스에서 조회 시도
    try:
        with db_manager.get_db_session() as session:
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
                    "raw_ocr_text": analysis.content.raw_ocr_text,
                    "layout_score": analysis.content.layout_score,
                    "channel_name": analysis.content.channel_name,
                    "view_count": analysis.content.view_count,
                    "duration": analysis.content.duration,
                    "created_at": analysis.created_at.isoformat()
                }
                
    except Exception as e:
        logger.error(f"데이터베이스 조회 실패: {e}")
    
    # 3. Fallback: Mock 데이터 생성
    logger.info(f"Mock 데이터로 대체: {video_id}")
    
    return {
        "video_id": video_id,
        "category": "C1",
        "confidence_score": 0.85,
        "reasoning_log": "Mock 분석: 어그로성 키워드가 감지되어 C1으로 분류되었습니다.",
        "status": "AUTO_REJECT",
        "model_used": "GPT4o (Mock)",
        "processing_time": 1.5,
        "context_score": 0.75,
        "s_semantic": 0.80,
        "o_existence": 0.70,
        "a_sync": 0.75,
        "layout_score": 0.37,
        "raw_ocr_text": "Mock OCR 텍스트: 🔥충격🔥 100만원 돈버는법 클릭 지금바로 구독 알림",
        "channel_name": "Mock Channel",
        "view_count": 1250000,
        "duration": 58,
        "created_at": datetime.now().isoformat()
    }
def get_status_color(category: str) -> str:
    """카테고리별 상태 색상 결정"""
    
    color_map = {
        "C1": "error",    # Red - 어그로/스팸
        "C2": "error",    # Red - 공장형 패턴
        "C3": "warning",  # Yellow - 품질 불량
        "C4": "error",    # Red - 무단 도용
        "C5": "complete"  # Green - 정상 영상
    }
    
    return color_map.get(category, "running")

def show_context_score_chart(analysis_data: dict):
    """Context Score 레이더 차트"""
    
    categories = ['의미적 유사도', '객체 존재', '시공간 동기화']
    values = [
        analysis_data.get('s_semantic', 0) * 100,
        analysis_data.get('o_existence', 0) * 100,
        analysis_data.get('a_sync', 0) * 100
    ]
    
    # 레이더 차트 생성
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Context Score',
        line_color='rgb(0, 123, 255)',
        fillcolor='rgba(0, 123, 255, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                ticksuffix='%'
            )),
        showlegend=False,
        title="Context Score 구성 요소",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def show_recent_analyses():
    """최근 분석 결과 표시"""
    
    st.markdown("### 📋 최근 분석 결과")
    
    try:
        with db_manager.get_db_session() as session:
            recent_analyses = session.query(AnalysisResults).join(Contents).order_by(
                AnalysisResults.created_at.desc()
            ).limit(10).all()
            
            if recent_analyses:
                data = []
                for analysis in recent_analyses:
                    data.append({
                        "Video ID": analysis.video_id,
                        "제목": analysis.content.title[:50] + "..." if analysis.content.title else "Unknown",
                        "카테고리": analysis.c_category,
                        "신뢰도": f"{analysis.confidence_score:.2f}",
                        "상태": analysis.status.value,
                        "분석 시간": analysis.created_at.strftime("%Y-%m-%d %H:%M")
                    })
                
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("아직 분석된 영상이 없습니다.")
                
    except Exception as e:
        st.error(f"데이터 조회 실패: {e}")

def submit_feedback(video_id: str, action: str, text: str = ""):
    """피드백 제출"""
    
    try:
        response = requests.post(
            "http://localhost:8000/feedback",
            json={
                "video_id": video_id,
                "action": action,
                "feedback_text": text
            },
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info(f"피드백 제출 성공: {video_id} - {action}")
            return True
        else:
            logger.error(f"피드백 제출 실패: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"피드백 제출 오류: {e}")
        return False

# 사이드바
with st.sidebar:
    st.header("🔧 시스템 정보")
    
    # 서비스 상태 확인
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            st.success("✅ API 서버 정상")
        else:
            st.error("❌ API 서버 오류")
    except:
        st.error("❌ API 서버 연결 실패")
    
    # 통계 정보
    st.markdown("### 📊 실시간 통계")
    
    try:
        with db_manager.get_db_session() as session:
            total_contents = session.query(Contents).count()
            total_analyses = session.query(AnalysisResults).count()
            total_feedback = session.query(UserFeedback).count()
            
            st.metric("총 분석 콘텐츠", f"{total_contents:,}개")
            st.metric("총 분석 결과", f"{total_analyses:,}개") 
            st.metric("사용자 피드백", f"{total_feedback:,}개")
            
    except Exception as e:
        st.error(f"통계 조회 실패: {e}")
    
    # 설정
    st.markdown("### ⚙️ 설정")
    auto_refresh = st.checkbox("자동 새로고침", value=False)
    
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()