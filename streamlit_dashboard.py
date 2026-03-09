"""
Streamlit 대시보드 - 개선된 정밀 점수 표시
Context Score + 정밀 점수 통합 버전
"""
import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="YouTube Shorts Detector",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .score-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4ecdc4;
        margin: 0.5rem 0;
    }
    .result-card {
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

def get_analysis_data(video_id: str):
    """분석 데이터 조회 - DB 직접 접근"""
    try:
        # 1. 먼저 세션 상태에서 확인
        if ("latest_result" in st.session_state and 
            st.session_state.latest_result and 
            st.session_state.latest_result.get("video_id") == video_id):
            return st.session_state.latest_result
        
        # 2. DB에서 직접 조회
        from database_manager import db_manager
        from database_models import AnalysisResults, Contents
        
        with db_manager.get_db_session() as session:
            analysis = session.query(AnalysisResults).filter(
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
                    "precise_scores": analysis.performance_metrics or {
                        "c1_spam_score": 0.2,
                        "c2_pattern_score": 0.3,
                        "c3_context_score": 0.8,
                        "cis_final": 0.65
                    },
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
        "reasoning_log": "Mock 분석 결과입니다. 실제 분석 데이터가 없어 샘플 데이터를 표시합니다.",
        "status": "AUTO_APPROVE",
        "model_used": "GPT4o",
        "processing_time": 2.3,
        "context_score": 0.75,
        "s_semantic": 0.8,
        "o_existence": 0.7,
        "a_sync": 0.75,
        "precise_scores": {
            "c1_spam_score": 0.234,
            "c2_pattern_score": 0.123,
            "c3_context_score": 0.789,
            "cis_final": 0.654
        },
        "raw_ocr_text": "🔥충격🔥 이것만 알면 100만원 번다!! 클릭 지금바로",
        "layout_score": 0.65,
        "channel_name": "테스트 채널",
        "view_count": 125000,
        "duration": 58,
        "created_at": "2024-01-15T10:30:00"
    }

def get_status_color(category: str) -> str:
    """카테고리별 상태 색상 반환"""
    color_map = {
        "C1": "error",     # 빨간색
        "C2": "warning",   # 주황색  
        "C3": "warning",   # 주황색
        "C5": "complete"   # 초록색
    }
    return color_map.get(category, "complete")

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

def show_comprehensive_score_analysis(analysis_data):
    """Context Score + 정밀 점수 통합 분석 결과 표시"""
    
    # 데이터 추출
    precise_scores = analysis_data.get('precise_scores', {})
    c1_score = precise_scores.get('c1_spam_score', 0.0)
    c2_score = precise_scores.get('c2_pattern_score', 0.0) 
    c3_score = precise_scores.get('c3_context_score', 0.0)
    cis_final = precise_scores.get('cis_final', 0.0)
    
    # Context Score 데이터
    s_semantic = analysis_data.get('s_semantic', 0.0)
    o_existence = analysis_data.get('o_existence', 0.0)
    a_sync = analysis_data.get('a_sync', 0.0)
    context_score = analysis_data.get('context_score', 0.0)
    
    # 탭으로 구분해서 표시
    tab1, tab2, tab3 = st.tabs(["🎯 정밀 점수 분석", "📊 Context Score 분석", "🏆 최종 결과"])
    
    # ===== 탭 1: 정밀 점수 분석 =====
    with tab1:
        st.markdown("### 📊 정밀 점수 분석 (기획서 4항)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🚨 C1 스팸 점수", 
                value=f"{c1_score:.3f}",
                delta="어그로/클릭베이트",
                delta_color="inverse",
                help="키워드 위험도 + 의미적 유사도"
            )
        
        with col2:
            st.metric(
                label="🏭 C2 패턴 점수",
                value=f"{c2_score:.3f}", 
                delta="공장형 템플릿",
                delta_color="inverse",
                help="레이아웃 일관성 + 반복 패턴"
            )
        
        with col3:
            st.metric(
                label="🎯 C3 맥락 점수",
                value=f"{c3_score:.3f}",
                delta="콘텐츠 품질", 
                delta_color="normal",
                help="의미적 일치도 + 객체 매칭"
            )
        
        with col4:
            st.metric(
                label="⭐ CIS 최종",
                value=f"{cis_final:.3f}",
                delta="Content Intelligence",
                help="C3 - (0.6×C1 + 0.3×C2)"
            )
        
        # 정밀 점수 공식 설명
        st.markdown("#### 🧮 계산 공식")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **C1 스팸 점수:** {c1_score:.3f}
            - 키워드 위험도 (70%): {c1_score*0.7:.3f}
            - 의미적 유사도 (30%): {c1_score*0.3:.3f}
            """)
            
            st.info(f"""
            **C2 패턴 점수:** {c2_score:.3f}
            - 템플릿 사용도 기반 SSIM 계산
            - 레이아웃 일관성 분석
            """)
        
        with col2:
            st.info(f"""
            **C3 맥락 점수:** {c3_score:.3f}
            - S_semantic (60%): {c3_score*0.6:.3f}
            - O_existence (40%): {c3_score*0.4:.3f}
            """)
            
            st.success(f"""
            **CIS 최종 점수:** {cis_final:.3f}
            = {c3_score:.3f} - (0.6×{c1_score:.3f} + 0.3×{c2_score:.3f})
            = {c3_score:.3f} - {(0.6*c1_score + 0.3*c2_score):.3f}
            """)
        
        # CIS 점수 게이지 차트
        st.markdown("#### ⭐ CIS 점수 시각화")
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = cis_final,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "CIS Final Score"},
            delta = {'reference': 0.5, 'increasing': {'color': "green"}},
            gauge = {
                'axis': {'range': [None, 1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 0.3], 'color': "lightcoral"},
                    {'range': [0.3, 0.5], 'color': "lightyellow"},
                    {'range': [0.5, 0.7], 'color': "lightblue"},
                    {'range': [0.7, 1], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # ===== 탭 2: Context Score 분석 =====  
    with tab2:
        st.markdown("### 📊 Context Score 상세 분석")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🧠 의미적 유사도 (S_semantic)",
                value=f"{s_semantic:.3f}",
                delta="가중치: 50%",
                help="영상과 텍스트 간의 의미적 일치도"
            )
        
        with col2:
            st.metric(
                label="👁️ 객체 존재 여부 (O_existence)", 
                value=f"{o_existence:.3f}",
                delta="가중치: 30%",
                help="텍스트에 언급된 객체가 영상에 실제 존재하는지"
            )
        
        with col3:
            st.metric(
                label="🎬 시공간 동기화 (A_sync)",
                value=f"{a_sync:.3f}",
                delta="가중치: 20%",
                help="영상과 음성/자막의 시간적 동기화"
            )
        
        with col4:
            st.metric(
                label="🎯 통합 Context Score",
                value=f"{context_score:.3f}",
                delta="최종 점수",
                help="S_semantic(50%) + O_existence(30%) + A_sync(20%)"
            )
        
        # Context Score 시각화
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Context Score 구성 비율")
            
            # 레이더 차트
            categories = ['의미적 유사도<br>(50%)', '객체 존재<br>(30%)', '시공간 동기화<br>(20%)']
            values = [s_semantic, o_existence, a_sync]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name='Context Score',
                line_color='rgb(32, 145, 236)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=False,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 구성 요소별 기여도")
            
            # 바 차트
            components = ['의미적 유사도', '객체 존재', '시공간 동기화']
            scores = [s_semantic, o_existence, a_sync]
            weights = [0.5, 0.3, 0.2]
            contributions = [score * weight for score, weight in zip(scores, weights)]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=components,
                y=contributions,
                text=[f'{contrib:.3f}' for contrib in contributions],
                textposition='auto',
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
            ))
            
            fig.update_layout(
                title="Context Score 기여도",
                yaxis_title="기여 점수",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # ===== 탭 3: 최종 결과 =====
    with tab3:
        st.markdown("### 🏆 최종 분류 결과")
        
        category = analysis_data.get('category', 'C5')
        confidence = analysis_data.get('confidence_score', 0.0)
        
        # 카테고리별 색상 및 이모지
        category_info = {
            'C1': {'color': '#ff4444', 'emoji': '🚨', 'name': '어그로/스팸'},
            'C2': {'color': '#ff8800', 'emoji': '🏭', 'name': '공장형 패턴'}, 
            'C3': {'color': '#ffcc00', 'emoji': '⚠️', 'name': '품질 불량'},
            'C5': {'color': '#44ff44', 'emoji': '✅', 'name': '정상 영상'}
        }
        
        info = category_info.get(category, category_info['C5'])
        
        # 메인 결과 카드
        st.markdown(f"""
        <div class="result-card" style="
            background: linear-gradient(135deg, {info['color']}20 0%, {info['color']}10 100%);
            border: 3px solid {info['color']};
        ">
            <h1 style="color: {info['color']}; margin: 0; font-size: 2.5em;">
                {info['emoji']} {category}
            </h1>
            <h2 style="color: {info['color']}; margin: 10px 0;">
                {info['name']}
            </h2>
            <h3 style="margin: 15px 0; color: #666;">
                🎯 신뢰도: {confidence:.1%} | ⭐ CIS 점수: {cis_final:.3f}
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        # 점수 요약 테이블
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 정밀 점수 요약")
            score_df = pd.DataFrame({
                '점수 유형': ['C1 스팸', 'C2 패턴', 'C3 맥락', 'CIS 최종'],
                '점수': [f"{c1_score:.3f}", f"{c2_score:.3f}", f"{c3_score:.3f}", f"{cis_final:.3f}"],
                '상태': [
                    '🔴 높음' if c1_score > 0.6 else '🟡 보통' if c1_score > 0.3 else '🟢 낮음',
                    '🔴 높음' if c2_score > 0.6 else '🟡 보통' if c2_score > 0.3 else '🟢 낮음',
                    '🟢 높음' if c3_score > 0.7 else '🟡 보통' if c3_score > 0.4 else '🔴 낮음',
                    '🟢 양호' if cis_final > 0.5 else '🟡 보통' if cis_final > 0.3 else '🔴 문제'
                ]
            })
            st.dataframe(score_df, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Context Score 요약")
            context_df = pd.DataFrame({
                '구성 요소': ['의미적 유사도', '객체 존재', '시공간 동기화', '통합 점수'],
                '점수': [f"{s_semantic:.3f}", f"{o_existence:.3f}", f"{a_sync:.3f}", f"{context_score:.3f}"],
                '가중치': ['50%', '30%', '20%', '100%']
            })
            st.dataframe(context_df, hide_index=True, use_container_width=True)
        
        # 종합 판단 근거
        st.markdown("#### 💭 종합 판단 근거")
        reasoning = analysis_data.get('reasoning_log', '분석 근거가 없습니다.')
        st.text_area("AI 분석 근거", reasoning, height=150, disabled=True)

def show_analysis_result(video_id):
    """분석 결과 상세 표시"""
    
    # 뒤로가기 버튼
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ 메인으로 돌아가기", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()
    
    st.markdown(f"## 📹 영상 분석 결과: `{video_id}`")
    
    # 분석 결과 조회
    analysis_data = get_analysis_data(video_id)
    
    if not analysis_data:
        st.error("분석 데이터를 찾을 수 없습니다.")
        if st.button("🔄 새로 분석하기"):
            st.session_state.page = "main"
            st.rerun()
        return
    
    # 영상 기본 정보
    with st.expander("📋 영상 정보", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📺 채널", analysis_data.get('channel_name', 'Unknown'))
            st.metric("👁️ 조회수", f"{analysis_data.get('view_count', 0):,}회")
        
        with col2:
            st.metric("⏱️ 길이", f"{analysis_data.get('duration', 0)}초")
            st.metric("🤖 분석 모델", analysis_data.get('model_used', 'GPT4o'))
        
        with col3:
            st.metric("⏰ 처리시간", f"{analysis_data.get('processing_time', 0):.1f}초")
            st.metric("📅 분석일시", analysis_data.get('created_at', '')[:19])
    
    # 🆕 통합 점수 분석 표시
    show_comprehensive_score_analysis(analysis_data)
    
    # 사용자 액션 버튼들
    st.markdown("### 🎬 사용자 액션")
    
    col1, col2 = st.columns(2)
    
    with col1:
        action_type = st.selectbox(
            "액션 선택",
            ["👍 좋아요", "🚫 채널 차단", "📢 콘텐츠 신고"],
            key="action_select"
        )
        
        if st.button("액션 실행", use_container_width=True):
            action_map = {
                "👍 좋아요": "like",
                "🚫 채널 차단": "block_channel", 
                "📢 콘텐츠 신고": "report"
            }
            
            if submit_feedback(video_id, action_map[action_type]):
                st.success(f"{action_type} 처리되었습니다!")
            else:
                st.error("처리 중 오류가 발생했습니다.")
    
    with col2:
        feedback_text = st.text_area("💬 상세 의견", placeholder="분석 결과에 대한 의견을 남겨주세요...", key="feedback_text")
        
        if st.button("의견 제출", use_container_width=True):
            if feedback_text.strip():
                if submit_feedback(video_id, "feedback", feedback_text):
                    st.success("의견이 전송되었습니다!")
                    st.session_state.feedback_text = ""
                else:
                    st.error("전송 중 오류가 발생했습니다.")
            else:
                st.warning("의견을 입력해주세요.")
    
    # OCR 텍스트 및 기술적 세부사항
    with st.expander("🔍 기술적 세부사항", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**OCR 추출 텍스트:**")
            st.text_area(
                "extracted_text",
                analysis_data.get('raw_ocr_text', '텍스트 없음'),
                height=150,
                label_visibility="collapsed"
            )
        
        with col2:
            st.markdown("**성능 지표:**")
            tech_info = {
                "처리 시간": f"{analysis_data.get('processing_time', 0):.2f}초",
                "분석 모델": analysis_data.get('model_used', 'Unknown'),
                "레이아웃 점수": f"{analysis_data.get('layout_score', 0.0):.3f}",
                "상태": analysis_data.get('status', 'Unknown'),
                "생성 시간": analysis_data.get('created_at', '')
            }
            st.json(tech_info)


def get_all_videos():
    """DB에서 전체 영상 목록 조회"""
    try:
        from database_manager import db_manager
        from database_models import AnalysisResults, Contents
        with db_manager.get_db_session() as session:
            results = session.query(AnalysisResults, Contents.title, Contents.channel_name).outerjoin(
                Contents, AnalysisResults.video_id == Contents.video_id
            ).order_by(AnalysisResults.created_at.desc()).all()
            return [
                {
                    "video_id": r.AnalysisResults.video_id,
                    "category": r.AnalysisResults.c_category,
                    "confidence": r.AnalysisResults.confidence_score,
                    "created_at": str(r.AnalysisResults.created_at)[:19] if r.AnalysisResults.created_at else "-",
                    "title": r.title or "-",
                    "channel": r.channel_name or "-"
                }
                for r in results
            ]
    except Exception as e:
        logger.error(f"영상 목록 조회 실패: {e}")
        return []


def delete_video(video_id: str) -> bool:
    """특정 영상 DB에서 삭제"""
    try:
        from database_manager import db_manager
        from database_models import AnalysisResults, Contents
        with db_manager.get_db_session() as session:
            # AnalysisResults 삭제
            deleted = session.query(AnalysisResults).filter(
                AnalysisResults.video_id == video_id
            ).delete()
            # Contents 테이블도 있으면 삭제
            try:
                session.query(Contents).filter(
                    Contents.video_id == video_id
                ).delete()
            except Exception:
                pass
            session.commit()
            return deleted > 0
    except Exception as e:
        logger.error(f"삭제 실패: {e}")
        return False


def show_db_manager():
    """DB 관리 페이지"""
    st.markdown("### 🗄️ DB 영상 목록 관리")

    # 새로고침 버튼
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🔄 새로고침"):
            st.rerun()

    videos = get_all_videos()

    if not videos:
        st.info("저장된 영상 데이터가 없습니다.")
        return

    st.markdown(f"**총 {len(videos)}개** 영상")

    # 카테고리 필터
    categories = ["전체"] + sorted(set(v["category"] for v in videos if v["category"]))
    selected_cat = st.selectbox("카테고리 필터", categories)
    if selected_cat != "전체":
        videos = [v for v in videos if v["category"] == selected_cat]

    # 전체 선택 체크박스
    select_all = st.checkbox("전체 선택")

    # 영상 목록 테이블
    selected_ids = []
    st.markdown("---")

    for v in videos:
        col1, col2, col3, col4, col5, col6 = st.columns([0.5, 2.5, 1.5, 1, 1, 2])
        checked = col1.checkbox("", key=f"chk_{v['video_id']}", value=select_all)
        if checked:
            selected_ids.append(v["video_id"])

        cat_emoji = {"C1": "🚨", "C2": "🏭", "C3": "⚠️", "C5": "✅"}.get(v["category"], "❓")
        col2.markdown(f"**{v['title'][:20]}**" if v['title'] != '-' else f"`{v['video_id']}`")
        col3.markdown(f"`{v['video_id']}`")
        col4.markdown(f"{cat_emoji} **{v['category']}**")
        col5.markdown(f"{v['confidence']:.2f}" if isinstance(v['confidence'], float) else "-")
        col6.markdown(f"{v['created_at']}")

    st.markdown("---")

    # 선택 삭제
    if selected_ids:
        st.warning(f"**{len(selected_ids)}개** 영상이 선택됨")
        if st.button("🗑️ 선택 항목 삭제", type="primary"):
            success_count = 0
            for vid in selected_ids:
                if delete_video(vid):
                    success_count += 1
            st.success(f"✅ {success_count}개 삭제 완료")
            st.rerun()
    else:
        st.info("삭제할 항목을 선택하세요.")

def main():
    """메인 애플리케이션"""
    
    # 헤더
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; margin: 0;">🎬 YouTube Shorts Detector</h1>
        <p style="color: white; margin: 0;">AI 기반 유튜브 쇼츠 콘텐츠 품질 분석</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 사이드바 네비게이션
    with st.sidebar:
        st.markdown("### 📋 메뉴")
        if st.button("🏠 분석 홈", use_container_width=True):
            st.session_state.page = 'main'
            st.rerun()
        if st.button("🗄️ DB 관리", use_container_width=True):
            st.session_state.page = 'db_manager'
            st.rerun()

    # 세션 상태 초기화
    if 'page' not in st.session_state:
        st.session_state.page = 'main'
    
    # URL 파라미터 확인
    if 'video_id' in st.query_params and st.session_state.page == 'main':
        st.session_state.page = 'result'
        st.session_state.current_video_id = st.query_params['video_id']
    
    # 페이지 라우팅
    if st.session_state.page == 'main':
        # 메인 페이지
        col1, col2, col3 = st.columns([1, 3, 1])
        
        with col2:
            st.markdown("### 🔍 YouTube Shorts URL 분석")
            
            video_url = st.text_input(
                "YouTube Shorts URL을 입력하세요:",
                placeholder="https://youtube.com/shorts/example",
                key="video_url_input"
            )
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                analyze_btn = st.button("🔍 분석 시작", use_container_width=True)
            
            with col_btn2:
                if st.button("📊 샘플 데이터", use_container_width=True):
                    st.session_state.page = 'result'
                    st.session_state.current_video_id = 'sample_video_001'
                    st.rerun()
            
            if analyze_btn and video_url:
                with st.spinner('🤖 AI가 영상을 분석 중입니다... (최대 60초)'):
                    try:
                        response = requests.post(
                            "http://localhost:8000/analyze",
                            json={
                                "video_url": video_url,
                                "request_source": "streamlit_dashboard"
                            },
                            timeout=65
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.session_state.page = 'result'
                            st.session_state.current_video_id = result['video_id']
                            st.success("✅ 분석이 완료되었습니다!")
                            st.rerun()
                        else:
                            st.error(f"❌ 분석 실패: {response.json().get('detail', '알 수 없는 오류')}")
                    
                    except requests.exceptions.Timeout:
                        st.error("⏰ 분석 시간이 초과되었습니다. 다시 시도해주세요.")
                    except Exception as e:
                        st.error(f"❌ 오류 발생: {str(e)}")
        
        # 사용법 안내
        with st.expander("📚 사용법 안내", expanded=False):
            st.markdown("""
            **🎯 분석 가능한 URL:**
            - `https://youtube.com/shorts/VIDEO_ID`
            - `https://youtu.be/VIDEO_ID`
            - `https://youtube.com/watch?v=VIDEO_ID`
            
            **📊 분석 결과:**
            - **C1**: 🚨 어그로/스팸 (자극적 제목, 허위 정보)
            - **C2**: 🏭 공장형 콘텐츠 (TTS 남용, 반복 패턴)
            - **C3**: ⚠️ 품질 불량 (제목-내용 불일치)
            - **C5**: ✅ 정상 영상 (양질의 오리지널 콘텐츠)
            
            **⭐ 점수 체계:**
            - **Context Score**: 영상-텍스트 맥락 일치도
            - **정밀 점수**: C1/C2/C3 개별 위험도 분석
            - **CIS 점수**: 종합 콘텐츠 지능 점수
            """)
    
    elif st.session_state.page == 'result':
        # 결과 페이지
        video_id = st.session_state.get('current_video_id', 'unknown')
        show_analysis_result(video_id)

    elif st.session_state.page == 'db_manager':
        # DB 관리 페이지
        show_db_manager()

if __name__ == "__main__":
    main()