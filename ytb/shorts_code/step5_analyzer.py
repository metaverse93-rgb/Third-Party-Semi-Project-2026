import os
import glob
import base64
import csv
import json
import requests
from datetime import datetime
from dotenv import load_dotenv
from config import PATHS, FINAL_DETAIL_CSV

load_dotenv(dotenv_path=r'D:\python\.env')
API_KEY = os.getenv("OPENAI_API_KEY")

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def run_analysis(video_name_only, youtube_url, ocr_results):
    frame_dir = os.path.join(PATHS["shorts_frames"], video_name_only)
    image_paths = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    
    if not image_paths:
        return None

    content_list = [{
        "type": "text", 
        "text": (
            "너는 영상의 시각적 요소와 자막의 의미적 결합을 분석하는 멀티모달 전문가야.\n\n"
            "### [핵심 분석 단계] ###\n"
            "1. 자막 문맥 복원: OCR 데이터에 오타(예: 사링훼)나 일본어 발음(예: 세카이)이 있더라도, 주변 단어와 외국어 원문을 참고해 제작자가 의도한 '정상적인 한국어 의미'로 먼저 변환해.\n"
            "2. 시각적 대조: 복원된 자막의 의미가 현재 화면(이미지)의 인물 동작, 표정, 배경 상황과 논리적으로 일치하는지 비교해.\n"
            "3. 맥락 판정:\n"
            "   - 자막의 오타나 발음 표기 자체는 감점 요인이 아님.\n"
            "   - 오타를 감안하여 해석했음에도 불구하고, 화면 상황과 자막 내용이 전혀 연관성이 없다면 '분류 2. 품질 불량'으로 판정해.\n\n"
            "### [분류 기준] ###\n"
            "- 분류 2. 품질 불량: 자막이 의도하는 상황과 화면의 실제 상황이 어긋날 때 (예: 슬픈 장면인데 자막은 즐거운 내용인 경우 등)\n"
            "- 분류 4. 정상 영상: 자막에 노이즈가 많더라도 해석된 의미가 화면의 퍼포먼스를 적절히 설명할 때\n\n"
            "### [응답 형식] ###\n"
            "반드시 JSON 형식으로 응답해:\n"
            "{\n"
            "  'scenes': [ {'frame': '파일명', 'analysis': '[분류] 오타를 감안해 해석한 자막 의미와 화면 일치도 설명'} ],\n"
            "  'total': {\n"
            "    'final_class': '분류 번호. 명칭',\n"
            "    'basis': '자막 노이즈를 문맥적으로 복원하여 이미지와 대조한 상세 근거',\n"
            "    'summary': '최종 요약'\n"
            "  }\n"
            "}"
        )
    }]

    # 이미지 페이로드 구성 (최대 10개)
    for i, img_p in enumerate(image_paths[:10]):
        base64_image = encode_image(img_p)
        raw_text = ocr_results[i] if i < len(ocr_results) else "자막 없음"
        
        content_list.append({"type": "text", "text": f"Frame: {os.path.basename(img_p)}, OCR Raw Data: {raw_text}"})
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    headers = {"Authorization": f"Bearer {API_KEY}"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": content_list}],
        "response_format": {"type": "json_object"},
        "temperature": 0
    }

    try:
        res = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        parsed = json.loads(res.json()['choices'][0]['message']['content'])
        
        # 상세 결과 저장
        for s in parsed.get('scenes', []):
            with open(FINAL_DETAIL_CSV, mode='a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now(), video_name_only, s['frame'], s['analysis'], youtube_url])
        
        return parsed.get('total')
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")
        return None