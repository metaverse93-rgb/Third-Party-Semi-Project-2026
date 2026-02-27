import streamlit as st
import os
import pandas as pd
from datetime import datetime

# 기존 프로젝트 파일들에서 함수 불러오기
from config import initialize_directories, FINAL_TOTAL_CSV
from step1_filter import run_filter
from step2_download import download_video
from step3_extract import extract_subtitle_frames_and_text
from step4_ocr import run_ocr_analysis 
from step5_analyzer import run_analysis

# 페이지 설정
st.set_page_config(page_title="유튜브 쇼츠 체크", page_icon="🔍")

# CSS로 리포트 디자인 살짝 추가
st.markdown("""
    <style>
    .report-box {
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        background-color: #f0f2f6;
        line-height: 1.6;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎬 유튜브 쇼츠 체크")
st.write("유튜브 링크를 입력하면 AI가 자막과 이미지의 맥락을 분석하여 쇼츠영상을 판별합니다.")

# URL 입력
youtube_url = st.text_input("🔗 분석할 유튜브 URL", placeholder="https://youtube.com/shorts/...")

if st.button("분석 시작"):
    if not youtube_url:
        st.warning("URL을 입력해 주세요.")
    else:
        # 폴더 초기화
        initialize_directories()
        
        with st.spinner("🚀 분석 중... (필터링 -> 다운로드 -> OCR -> AI 분석)"):
            # 1. 필터링 (step1)
            passed, filter_res = run_filter(youtube_url)
            
            if not passed:
                # 1단계에서 탈락한 경우
                final_class = "부적합 (분류 1)"
                basis = filter_res['reason']
                summary = "스팸 필터링 차단"
            else:
                # 2. 영상 처리 (step2 ~ step4)
                video_path = download_video(youtube_url)
                if video_path:
                    video_name = os.path.splitext(os.path.basename(video_path))[0]
                    
                    # 프레임 추출 및 OCR
                    extract_subtitle_frames_and_text(video_path) 
                    ocr_csv_path = run_ocr_analysis(video_name) 
                    
                    if ocr_csv_path and os.path.exists(ocr_csv_path):
                        ocr_list = pd.read_csv(ocr_csv_path)['text'].tolist()
                    else:
                        ocr_list = []

                    # 3. AI 분석 (step5)
                    total_res = run_analysis(video_name, youtube_url, ocr_list)
                    
                    if total_res:
                        final_class = total_res.get('final_class', "분류 미정")
                        basis = total_res.get('basis', "근거 없음")
                        summary = total_res.get('summary', "요약 없음")
                    else:
                        st.error("AI 분석 응답을 받지 못했습니다.")
                        st.stop()
                else:
                    st.error("영상 다운로드에 실패했습니다.")
                    st.stop()

            # 최종 결과 리포트 출력
            st.markdown("---")
            
            # 리포트 제목 (흰색 고정)
            st.markdown("<h3 style='color: #FFFFFF;'>📌 [최종 판별 결과 리포트]</h3>", unsafe_allow_html=True)
            
            # 리포트 박스 스타일 (어두운 배경 + 흰색 글자)
            report_style = """
            <div style="
                background-color: #262730; 
                padding: 25px; 
                border-radius: 10px; 
                border-left: 8px solid #FF4B4B;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                margin-top: 10px;
            ">
                <p style="color: #FFFFFF; font-size: 19px; margin-bottom: 15px; line-height: 1.5;">
                    <span style="font-weight: bold; color: #FFFFFF;">✅ 최종 판별 :</span> {final_class}
                </p>
                <p style="color: #FFFFFF; font-size: 19px; margin-bottom: 15px; line-height: 1.5;">
                    <span style="font-weight: bold; color: #FFFFFF;">📝 판별 근거 :</span> {basis}
                </p>
                <p style="color: #FFFFFF; font-size: 19px; margin-bottom: 0; line-height: 1.5;">
                    <span style="font-weight: bold; color: #FFFFFF;">💡 판별 요약 :</span> "{summary}"
                </p>
            </div>
            """
            
            # 변수 적용하여 출력
            st.markdown(report_style.format(
                final_class=final_class, 
                basis=basis, 
                summary=summary
            ), unsafe_allow_html=True)
            
            st.balloons()

st.sidebar.info("이 시스템은 GPT-4o-mini 멀티모달 모델을 사용합니다.")


#streamlit run d:/project/ytb/shorts_code/shorts_streamlit.py