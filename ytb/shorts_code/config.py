import os

BASE_DIR = os.path.normpath(r"D:/project/Third-Party-semi-Project-2026/ytb/")

PATHS = {
    "shorts": os.path.join(BASE_DIR, "shorts"),
    "shorts_frames": os.path.join(BASE_DIR, "shorts_frames"),
    "shorts_ocr": os.path.join(BASE_DIR, "shorts_ocr"),
    "filter_result": os.path.join(BASE_DIR, "filter_result"),
    "final_result": os.path.join(BASE_DIR, "shorts_final"),       # 장면별 상세 CSV 폴더
    "final_total": os.path.join(BASE_DIR, "shorts_final_total"),  # 최종 리포트 CSV 폴더
}

# 파일 경로 정의
FINAL_DETAIL_CSV = os.path.join(PATHS["final_result"], "detailed_scene_analysis.csv")
FINAL_TOTAL_CSV = os.path.join(PATHS["final_total"], "total_analysis_report.csv")
CSV_FILE = os.path.join(PATHS["filter_result"], "filter_results.csv")

def initialize_directories():
    for path in PATHS.values():
        os.makedirs(path, exist_ok=True)

initialize_directories()