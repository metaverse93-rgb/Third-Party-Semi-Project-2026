import os
import csv
import pandas as pd
from datetime import datetime
from config import initialize_directories, FINAL_TOTAL_CSV, PATHS
from step1_filter import run_filter
from step2_download import download_video
from step3_extract import extract_subtitle_frames_and_text
from step4_ocr import run_ocr_analysis 
from step5_analyzer import run_analysis

def save_to_final_total(final_class, basis, summary, video_name, url):
    file_exists = os.path.isfile(FINAL_TOTAL_CSV)
    with open(FINAL_TOTAL_CSV, mode='a', newline='', encoding='utf-8-sig') as f:
        # 기존 필드 구조를 유지하면서 저장 (category 칸은 비워둠)
        writer = csv.DictWriter(f, fieldnames=["datetime", "video_name", "final_class", "basis", "summary", "url"])
        if not file_exists: writer.writeheader()
        writer.writerow({
            "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "video_name": video_name,
            "final_class": final_class,
            "basis": basis,
            "summary": summary,
            "url": url
        })

def print_final_report(final_class, basis, summary):
    print("\n" + "="*60)
    print(f"📌 [최종 판별 결과 리포트]")
    print("-" * 60)
    # ✅ 최종 판별 : 분류 X. 명칭 형태로 출력
    print(f"✅ 최종 판별 : {final_class}")
    print(f"📝 판별 근거  : {basis}")
    print(f"💡 판별 요약  : \"{summary}\"")
    print("="*60 + "\n")

def main():
    initialize_directories()
    youtube_url = input("🔗 분석할 유튜브 URL: ").strip()
    
    # 1. 1단계 필터링
    passed, filter_res = run_filter(youtube_url)
    if not passed:
        print_final_report("부적합 (분류 1)", filter_res['reason'], "스팸 필터링 차단")
        return

    # 2. 전처리 (다운로드/추출/OCR)
    video_path = download_video(youtube_url)
    if not video_path: return
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    extract_subtitle_frames_and_text(video_path) 
    ocr_csv_path = run_ocr_analysis(video_name) 
    ocr_list = pd.read_csv(ocr_csv_path)['text'].tolist() if ocr_csv_path else []

    # 3. GPT 멀티모달 분석
    total_res = run_analysis(video_name, youtube_url, ocr_list)
    
    if total_res:
        f_class = total_res.get('final_class', "분류 미정")
        f_basis = total_res.get('basis', "근거 없음")
        f_summary = total_res.get('summary', "요약 없음")

        # 결과 저장 및 리포트 출력
        save_to_final_total(f_class, f_basis, f_summary, video_name, youtube_url)
        print_final_report(f_class, f_basis, f_summary)
    else:
        print("❌ 분석 결과 생성에 실패했습니다.")

if __name__ == "__main__":
    main()