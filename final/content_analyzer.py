
"""
멀티모달 콘텐츠 분석기 - 기획서 기반
LLM은 분석만, 분류는 precise_score_calculator가 담당
"""
import json
import re
import time
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI

from config import (
    GPT4O_CONFIG, QWEN_CONFIG, RETRY_CONFIG, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
    MOCK_MODE
    # SPAM_PATTERNS, SPAM_KEYWORDS 제거
)
from models import AnalysisResult, ContentCategory, PreciseScores
from precise_score_calculator import PreciseScoreCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """콘텐츠 분석기 - 기획서 기반 분석 + 알고리즘 분류"""
    
    def __init__(self, mode: str = "GPT4o"):
        self.mode = mode
        self.score_calculator = PreciseScoreCalculator()
        
        if MOCK_MODE:
            logger.info("🎭 Mock 모드로 초기화")
            self.client = None
            self.model_name = "Mock"
        elif mode == "GPT4o":
            self.client = OpenAI(
                api_key=GPT4O_CONFIG["api_key"],
                base_url=GPT4O_CONFIG["base_url"]
            )
            self.model_name = GPT4O_CONFIG["model"]
            logger.info(f"🤖 GPT-4o-mini 모드로 초기화")
        elif mode == "Qwen":
            self.client = OpenAI(
                api_key=QWEN_CONFIG["api_key"],
                base_url=QWEN_CONFIG["base_url"]
            )
            self.model_name = QWEN_CONFIG["model"]
            logger.info(f"🤖 Qwen2.5-VL 모드로 초기화")
        else:
            raise ValueError(f"지원하지 않는 모드: {mode}")
    
    def switch_model(self, new_mode: str):
        """모델 교체"""
        logger.info(f"🔄 모델 교체: {self.mode} → {new_mode}")
        self.__init__(new_mode)

    def analyze(self,
                frames: List[str],
                ocr_text: str,
                layout_score: float = 0.0,
                metadata: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """
        기획서 기반 콘텐츠 분석
        1. LLM이 영상 분석 데이터 제공 (분류 X)
        2. 알고리즘이 C1, C2, C3 점수 계산
        3. CIS 기반 최종 분류
        """
        start_time = time.time()
        
        try:
            metadata = metadata or {}
            
            # ✅ Step 1: LLM 분석 (분류 X, 분석 데이터만)
            if MOCK_MODE:
                analysis_data = self._mock_analysis_data(metadata, ocr_text, frames)
            else:
                analysis_data = self._analyze_with_llm(frames, ocr_text, layout_score, metadata)
            
            # ✅ Step 2: 알고리즘 점수 계산 및 분류
            scores = self.score_calculator.calculate_all_scores(
                analysis_data, ocr_text, frames, metadata
            )
            
            # ✅ Step 3: 결과 구성
            processing_time = time.time() - start_time
            final_category = scores.get('final_category', 'C5')
            confidence_score = self._calculate_confidence(scores)
            
            result = AnalysisResult(
                c_category=ContentCategory(final_category),
                reasoning_log=self._generate_reasoning(scores, analysis_data, final_category),
                confidence_score=confidence_score,
                raw_response=json.dumps(analysis_data, ensure_ascii=False),
                processing_time=processing_time,
                precise_scores=PreciseScores(
                    c1_spam_score=scores.get('c1_spam_score', 0.0),
                    c2_pattern_score=min(1.0, scores.get('c2_pattern_score', 0.0)),
                    c3_context_score=min(1.0, scores.get('c3_context_score', 0.0)),
                    cis_final=scores.get('cis_final', 0.0)
                )
            )

            logger.info(f"✅ 분석 완료: {result.c_category.value} (CIS: {scores.get('cis_final', 0):.3f})")
            return result

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ 분석 실패: {e}")
            return self._create_error_result(str(e), processing_time)

    def _analyze_with_llm(self, frames: List[str], ocr_text: str, layout_score: float, metadata: Dict) -> Dict:
        """LLM으로 분석 데이터 추출 (분류하지 않음)"""
        
        # 메타데이터 준비
        title = metadata.get("title", "")
        description = metadata.get("description", "")
        channel_name = metadata.get("channel_name", "")
        view_count = metadata.get("view_count", 0)
        duration = metadata.get("duration", 0)
        
        # OCR 텍스트 유무에 따라 프롬프트 분기
        has_ocr = bool(ocr_text and ocr_text.strip())
        user_prompt = USER_PROMPT_TEMPLATE.format(
            title=title,
            description=description,
            channel_name=channel_name,
            view_count=view_count,
            duration=duration,
            frame_count=len(frames),
            ocr_text=ocr_text if has_ocr else "(자막 없음 - 이미지 시각 분석만 수행)"
        )

        # OCR 없을 때 이미지 중심 분석 안내 문구 추가
        if not has_ocr:
            user_prompt += "\n\n[안내] 이 영상은 추출된 자막 텍스트가 없습니다. 첨부된 프레임 이미지만을 기반으로 시각적 요소, 레이아웃 패턴, 화면 구성을 분석해주세요."
        
        # 메시지 구성
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        # 이미지 추가 (최대 5개 프레임)
        for i, frame in enumerate(frames[:5]):
            if frame:  # base64 데이터가 있는 경우만
                messages[1]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame}",
                        "detail": "low"
                    }
                })
        
        # API 호출 with 재시도
        for attempt in range(RETRY_CONFIG["max_retries"]):
            try:
                logger.info(f"🤖 LLM 분석 호출 (시도 {attempt + 1}/{RETRY_CONFIG['max_retries']})")
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=GPT4O_CONFIG["max_tokens"],
                    temperature=GPT4O_CONFIG["temperature"]
                )
                
                raw_response = response.choices[0].message.content
                logger.info("📨 LLM 분석 응답 수신 완료")
                return self._parse_analysis_response(raw_response)
                
            except Exception as e:
                logger.warning(f"⚠️ LLM 호출 실패 (시도 {attempt + 1}): {e}")
                if attempt < RETRY_CONFIG["max_retries"] - 1:
                    sleep_time = RETRY_CONFIG["retry_delay"] * (RETRY_CONFIG["backoff_factor"] ** attempt)
                    logger.info(f"⏳ {sleep_time:.1f}초 후 재시도...")
                    time.sleep(sleep_time)
                else:
                    logger.error("❌ 모든 재시도 실패, Mock 데이터로 대체")
                    return self._mock_analysis_data(metadata, ocr_text, frames)

    def _parse_analysis_response(self, raw_response: str) -> Dict:
        """LLM 응답 파싱"""
        try:
            # JSON 추출
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                analysis_data = json.loads(json_str)
                
                # 필수 필드 확인 및 기본값 설정
                required_fields = {
                    "visual_elements": [],
                    "frame_descriptions": [],
                    "layout_consistency": "보통",
                    "layout_analysis": "",
                    "spam_indicators": [],
                    "content_object_matching": {
                        "mentioned_in_text": [],
                        "visible_in_frames": [],
                        "matching_analysis": ""
                    },
                    "action_elements": [],
                    "quality_issues": [],
                    "overall_analysis": ""
                }
                
                for field, default_value in required_fields.items():
                    if field not in analysis_data:
                        analysis_data[field] = default_value
                
                # content_object_matching 내부 필드 확인
                if not isinstance(analysis_data.get("content_object_matching"), dict):
                    analysis_data["content_object_matching"] = required_fields["content_object_matching"]
                
                logger.info("✅ LLM 응답 파싱 성공")
                return analysis_data
            else:
                raise ValueError("JSON 형식을 찾을 수 없음")
                
        except Exception as e:
            logger.error(f"❌ LLM 응답 파싱 실패: {e}")
            logger.info("🔄 Fallback 파싱 시도")
            return self._fallback_parse(raw_response)

    def _fallback_parse(self, raw_response: str) -> Dict:
        """Fallback 파싱 (응답 실패시)"""
        
        # 응답에서 키워드 추출 시도
        visual_keywords = re.findall(r'(차트|그래프|이미지|텍스트|버튼|아이콘)', raw_response)
        
        return {
            "visual_elements": visual_keywords[:5],
            "frame_descriptions": [f"파싱 실패: {raw_response[:100]}..."],
            "layout_consistency": "보통",
            "layout_analysis": "파싱 실패로 분석 불가",
            "spam_indicators": [],  # 스팸 키워드 체크 제거
            "content_object_matching": {
                "mentioned_in_text": [],
                "visible_in_frames": visual_keywords,
                "matching_analysis": "파싱 실패"
            },
            "action_elements": [],
            "quality_issues": ["분석 데이터 파싱 실패"],
            "overall_analysis": f"LLM 응답 파싱 실패. 원본 응답: {raw_response[:200]}..."
        }

    def _mock_analysis_data(self, metadata: Dict, ocr_text: str, frames: List[str]) -> Dict:
        """Mock 분석 데이터 생성 - 실제 제목/OCR 텍스트 기반으로 스팸 지표 탐지"""
        logger.info("🎭 Mock 분석 데이터 생성 (실제 텍스트 기반)")

        title = metadata.get('title', '')
        description = metadata.get('description', '')
        full_text = f"{title} {description} {ocr_text}".lower()

        # ✅ 실제 텍스트 기반 스팸 지표 탐지
        spam_keyword_map = {
            "충격": "과장 표현: 충격",
            "경악": "과장 표현: 경악",
            "실화": "과장 표현: 실화",
            "대박": "과장 표현: 대박",
            "역대급": "과장 표현: 역대급",
            "무료": "무료 혜택 유인",
            "선착순": "희소성 자극: 선착순",
            "카톡": "스팸 연락처: 카카오톡",
            "텔레그램": "스팸 연락처: 텔레그램",
            "수익률": "금전 자극: 수익률",
            "돈버는법": "금전 자극: 돈버는법",
            "100만원": "금전 자극: 고액",
            "이것만 알면": "클릭베이트 문구",
            "비밀 공개": "클릭베이트 문구",
            "지금바로": "긴박감 자극",
            "클릭": "클릭 유도",
        }
        spam_indicators = [
            label for keyword, label in spam_keyword_map.items()
            if keyword in full_text
        ]

        # ✅ 레이아웃 일관성 - 프레임 수 기반 휴리스틱
        if len(frames) >= 5:
            layout_consistency = "높음"
            layout_analysis = f"총 {len(frames)}장의 프레임에서 동일한 자막 위치와 폰트 패턴이 반복되는 것으로 추정"
        elif len(frames) >= 3:
            layout_consistency = "보통"
            layout_analysis = f"총 {len(frames)}장의 프레임에서 일부 레이아웃 일관성 확인"
        else:
            layout_consistency = "낮음"
            layout_analysis = f"총 {len(frames)}장의 프레임 - 다양한 구성으로 템플릿 패턴 없음"

        # ✅ 객체 매칭 - 실제 제목 단어 기반
        title_words = [w for w in title.split() if len(w) > 1]
        mentioned_objects = title_words[:5]
        # 자막(OCR)에도 등장하면 화면에 보이는 것으로 간주
        visible_objects = [w for w in mentioned_objects if w.lower() in ocr_text.lower()]
        if not visible_objects:
            visible_objects = mentioned_objects[:2]

        # ✅ 품질 문제 탐지
        quality_issues = []
        if title and ocr_text and not any(w in ocr_text for w in title.split() if len(w) > 2):
            quality_issues.append("제목 키워드가 자막에서 발견되지 않음 (제목-내용 불일치 의심)")

        # ✅ 동작 요소
        action_keywords = ["움직", "이동", "클릭", "터치", "변화", "전환", "증가", "상승", "하락"]
        action_elements = [f"{kw} 감지" for kw in action_keywords if kw in full_text]
        if not action_elements:
            action_elements = []

        overall = (
            f"제목 '{title[:40]}' 기반 Mock 분석. "
            f"스팸 지표 {len(spam_indicators)}개 탐지. "
            f"레이아웃 일관성: {layout_consistency}. "
            f"품질 이슈: {len(quality_issues)}건."
        )

        return {
            "visual_elements": ["자막 텍스트", "배경 화면"] + (["스팸 키워드 오버레이"] if spam_indicators else []),
            "frame_descriptions": [
                f"프레임 {i+1}: {title[:30]}... 관련 장면" for i in range(min(len(frames), 5))
            ],
            "layout_consistency": layout_consistency,
            "layout_analysis": layout_analysis,
            "spam_indicators": spam_indicators,
            "content_object_matching": {
                "mentioned_in_text": mentioned_objects,
                "visible_in_frames": visible_objects,
            },
            "action_elements": action_elements,
            "quality_issues": quality_issues,
            "overall_analysis": overall
        }

    def _calculate_confidence(self, scores: Dict) -> float:
        """신뢰도 계산"""
        cis_score = abs(scores.get('cis_final', 0))
        
        # CIS 절댓값이 클수록 확신도 높음
        if cis_score >= 0.5:
            confidence = 0.9
        elif cis_score >= 0.3:
            confidence = 0.8
        elif cis_score >= 0.1:
            confidence = 0.7
        else:
            confidence = 0.6
        
        return round(confidence, 3)

    def _generate_reasoning(self, scores: Dict, analysis_data: Dict, category: str) -> str:
        """판단 근거 생성"""
        cis = scores.get('cis_final', 0)
        c1 = scores.get('c1_spam_score', 0)
        c2 = scores.get('c2_pattern_score', 0)  
        c3 = scores.get('c3_context_score', 0)
        
        reasoning = f"🎯 기획서 기반 CIS 분류 시스템\n"
        reasoning += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        reasoning += f"📊 점수 계산 결과:\n"
        reasoning += f"  • C1 (어그로/스팸): {c1:.3f}\n"
        reasoning += f"  • C2 (공장형 패턴): {c2:.3f}\n"
        reasoning += f"  • C3 (맥락 품질): {c3:.3f}\n"
        reasoning += f"  • CIS 최종 점수: {cis:.3f}\n"
        reasoning += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
        # 카테고리별 상세 설명
        if category == "C1":
            spam_indicators = analysis_data.get('spam_indicators', [])
            reasoning += f"🚨 C1 (어그로/스팸) 판정:\n"
            reasoning += f"  • CIS < -0.5이며 C1 점수가 높음\n"
            if spam_indicators:
                reasoning += f"  • 감지된 스팸 지표: {', '.join(spam_indicators[:3])}\n"
        elif category == "C2":
            layout = analysis_data.get('layout_consistency', '보통')
            reasoning += f"🏭 C2 (공장형 패턴) 판정:\n"
            reasoning += f"  • CIS < -0.5이며 C2 점수가 높음\n"
            reasoning += f"  • 레이아웃 일관성: {layout}\n"
        elif category == "C3":
            quality_issues = analysis_data.get('quality_issues', [])
            reasoning += f"⚠️ C3 (품질 불량) 판정:\n"
            reasoning += f"  • CIS < -0.2 또는 맥락 품질 저하\n"
            if quality_issues:
                reasoning += f"  • 품질 문제: {', '.join(quality_issues[:2])}\n"
        else:  # C5
            reasoning += f"✅ C5 (정상 영상) 판정:\n"
            reasoning += f"  • CIS ≥ 0.3 또는 양호한 품질\n"
            reasoning += f"  • 기본적인 품질 기준 충족\n"
        
        # 상세 점수 근거 추가
        if 'detailed_results' in scores:
            details = scores['detailed_results']
            if 'cis_details' in details:
                cis_reasoning = details['cis_details'].get('reasoning', '')
                reasoning += f"\n📝 CIS 계산 근거:\n  {cis_reasoning}\n"
        
        return reasoning

    # _check_spam_patterns 함수 제거
    # _create_spam_result 함수 제거

    def _create_error_result(self, error_msg: str, processing_time: float) -> AnalysisResult:
        """오류 결과 생성"""
        return AnalysisResult(
            c_category=ContentCategory.C5,
            reasoning_log=f"❌ 분석 중 오류 발생\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n오류 내용: {error_msg}\n\n오류 발생시 안전을 위해 정상(C5)으로 분류합니다.",
            confidence_score=0.1,
            error_message=error_msg,
            processing_time=processing_time,
            precise_scores=PreciseScores(
                c1_spam_score=0.0,
                c2_pattern_score=0.0,
                c3_context_score=0.5,
                cis_final=0.5
            )
        )

# 테스트 함수
def test_content_analyzer():
    """콘텐츠 분석기 테스트"""
    print("🧪 기획서 기반 콘텐츠 분석기 테스트")
    print("=" * 60)
    
    # Mock 모드로 테스트
    analyzer = ContentAnalyzer(mode="GPT4o")
    
    # 테스트 데이터
    test_metadata = {
        "title": "건강한 식단으로 체중 관리하기",
        "description": "영양 전문가가 알려주는 건강한 다이어트 방법",
        "channel_name": "건강채널",
        "view_count": 50000,
        "duration": 58
    }
    
    test_frames = ["mock_frame_1", "mock_frame_2", "mock_frame_3"]
    test_ocr = "건강한 식단과 운동으로 체중을 관리하는 방법을 알아보세요."
    
    # 분석 실행
    result = analyzer.analyze(
        frames=test_frames,
        ocr_text=test_ocr,
        layout_score=0.75,
        metadata=test_metadata
    )
    
    print(f"\n📊 분석 결과:")
    print(f"  분류: {result.c_category.value}")
    print(f"  상태: {result.status.value}")
    print(f"  신뢰도: {result.confidence_score:.3f}")
    print(f"  처리 시간: {result.processing_time:.2f}초")
    
    if result.precise_scores:
        print(f"\n📈 정밀 점수:")
        for key, value in result.precise_scores.items():
            print(f"    {key}: {value}")
    
    print(f"\n📝 판단 근거:")
    print(f"{result.reasoning_log}")

if __name__ == "__main__":
    test_content_analyzer()



# """
# 멀티모달 콘텐츠 분석기 - 기획서 기반
# LLM은 분석만, 분류는 precise_score_calculator가 담당
# """
# import json
# import re
# import time
# import logging
# from typing import List, Dict, Any, Optional
# from openai import OpenAI

# from config import (
#     GPT4O_CONFIG, QWEN_CONFIG, RETRY_CONFIG, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
#     MOCK_MODE, SPAM_PATTERNS, SPAM_KEYWORDS
# )
# from models import AnalysisResult, ContentCategory
# from precise_score_calculator import PreciseScoreCalculator

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class ContentAnalyzer:
#     """콘텐츠 분석기 - 기획서 기반 분석 + 알고리즘 분류"""
    
#     def __init__(self, mode: str = "GPT4o"):
#         self.mode = mode
#         self.score_calculator = PreciseScoreCalculator()
        
#         if MOCK_MODE:
#             logger.info("🎭 Mock 모드로 초기화")
#             self.client = None
#             self.model_name = "Mock"
#         elif mode == "GPT4o":
#             self.client = OpenAI(
#                 api_key=GPT4O_CONFIG["api_key"],
#                 base_url=GPT4O_CONFIG["base_url"]
#             )
#             self.model_name = GPT4O_CONFIG["model"]
#             logger.info(f"🤖 GPT-4o-mini 모드로 초기화")
#         elif mode == "Qwen":
#             self.client = OpenAI(
#                 api_key=QWEN_CONFIG["api_key"],
#                 base_url=QWEN_CONFIG["base_url"]
#             )
#             self.model_name = QWEN_CONFIG["model"]
#             logger.info(f"🤖 Qwen2.5-VL 모드로 초기화")
#         else:
#             raise ValueError(f"지원하지 않는 모드: {mode}")
    
#     def switch_model(self, new_mode: str):
#         """모델 교체"""
#         logger.info(f"🔄 모델 교체: {self.mode} → {new_mode}")
#         self.__init__(new_mode)

#     def analyze(self,
#                 frames: List[str],
#                 ocr_text: str,
#                 layout_score: float = 0.0,
#                 metadata: Optional[Dict[str, Any]] = None) -> AnalysisResult:
#         """
#         기획서 기반 콘텐츠 분석
#         1. 스팸 패턴 사전 체크 (즉시 분류)
#         2. LLM이 영상 분석 데이터 제공 (분류 X)
#         3. 알고리즘이 C1, C2, C3 점수 계산
#         4. CIS 기반 최종 분류
#         """
#         start_time = time.time()
        
#         try:
#             metadata = metadata or {}
            
#             # ✅ Step 1: 스팸 패턴 사전 체크 (기존 유지)
#             spam_result = self._check_spam_patterns(metadata, ocr_text)
#             if spam_result:
#                 processing_time = time.time() - start_time
#                 logger.info(f"🚨 스팸 패턴 감지 → 즉시 C1 분류: {spam_result}")
#                 return self._create_spam_result(spam_result, processing_time)

#             # ✅ Step 2: LLM 분석 (분류 X, 분석 데이터만)
#             if MOCK_MODE:
#                 analysis_data = self._mock_analysis_data(metadata, ocr_text, frames)
#             else:
#                 analysis_data = self._analyze_with_llm(frames, ocr_text, layout_score, metadata)
            
#             # ✅ Step 3: 알고리즘 점수 계산 및 분류
#             scores = self.score_calculator.calculate_all_scores(
#                 analysis_data, ocr_text, frames, metadata
#             )
            
#             # ✅ Step 4: 결과 구성
#             processing_time = time.time() - start_time
#             final_category = scores.get('final_category', 'C5')
#             confidence_score = self._calculate_confidence(scores)
            
#             result = AnalysisResult(
#                 c_category=ContentCategory(final_category),
#                 reasoning_log=self._generate_reasoning(scores, analysis_data, final_category),
#                 confidence_score=confidence_score,
#                 raw_response=json.dumps(analysis_data, ensure_ascii=False),
#                 processing_time=processing_time,
#                 precise_scores={
#                     "c1_spam_score": scores.get('c1_spam_score', 0),
#                     "c2_pattern_score": scores.get('c2_pattern_score', 0),
#                     "c3_context_score": scores.get('c3_context_score', 0),
#                     "cis_final": scores.get('cis_final', 0)
#                 }
#             )

#             logger.info(f"✅ 분석 완료: {result.c_category.value} (CIS: {scores.get('cis_final', 0):.3f})")
#             return result

#         except Exception as e:
#             processing_time = time.time() - start_time
#             logger.error(f"❌ 분석 실패: {e}")
#             return self._create_error_result(str(e), processing_time)

#     def _analyze_with_llm(self, frames: List[str], ocr_text: str, layout_score: float, metadata: Dict) -> Dict:
#         """LLM으로 분석 데이터 추출 (분류하지 않음)"""
        
#         # 메타데이터 준비
#         title = metadata.get("title", "")
#         description = metadata.get("description", "")
#         channel_name = metadata.get("channel_name", "")
#         view_count = metadata.get("view_count", 0)
#         duration = metadata.get("duration", 0)
        
#         # 프롬프트 구성
#         user_prompt = USER_PROMPT_TEMPLATE.format(
#             title=title,
#             description=description,
#             channel_name=channel_name,
#             view_count=view_count,
#             duration=duration,
#             frame_count=len(frames)
#         )
        
#         # 메시지 구성
#         messages = [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {
#                 "role": "user", 
#                 "content": [
#                     {"type": "text", "text": user_prompt}
#                 ]
#             }
#         ]
        
#         # 이미지 추가 (최대 5개 프레임)
#         for i, frame in enumerate(frames[:5]):
#             if frame:  # base64 데이터가 있는 경우만
#                 messages[1]["content"].append({
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{frame}",
#                         "detail": "low"
#                     }
#                 })
        
#         # API 호출 with 재시도
#         for attempt in range(RETRY_CONFIG["max_retries"]):
#             try:
#                 logger.info(f"🤖 LLM 분석 호출 (시도 {attempt + 1}/{RETRY_CONFIG['max_retries']})")
                
#                 response = self.client.chat.completions.create(
#                     model=self.model_name,
#                     messages=messages,
#                     max_tokens=GPT4O_CONFIG["max_tokens"],
#                     temperature=GPT4O_CONFIG["temperature"]
#                 )
                
#                 raw_response = response.choices[0].message.content
#                 logger.info("📨 LLM 분석 응답 수신 완료")
#                 return self._parse_analysis_response(raw_response)
                
#             except Exception as e:
#                 logger.warning(f"⚠️ LLM 호출 실패 (시도 {attempt + 1}): {e}")
#                 if attempt < RETRY_CONFIG["max_retries"] - 1:
#                     sleep_time = RETRY_CONFIG["retry_delay"] * (RETRY_CONFIG["backoff_factor"] ** attempt)
#                     logger.info(f"⏳ {sleep_time:.1f}초 후 재시도...")
#                     time.sleep(sleep_time)
#                 else:
#                     logger.error("❌ 모든 재시도 실패, Mock 데이터로 대체")
#                     return self._mock_analysis_data(metadata, ocr_text, frames)

#     def _parse_analysis_response(self, raw_response: str) -> Dict:
#         """LLM 응답 파싱"""
#         try:
#             # JSON 추출
#             json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
#             if json_match:
#                 json_str = json_match.group()
#                 analysis_data = json.loads(json_str)
                
#                 # 필수 필드 확인 및 기본값 설정
#                 required_fields = {
#                     "visual_elements": [],
#                     "frame_descriptions": [],
#                     "layout_consistency": "보통",
#                     "layout_analysis": "",
#                     "spam_indicators": [],
#                     "content_object_matching": {
#                         "mentioned_in_text": [],
#                         "visible_in_frames": [],
#                         "matching_analysis": ""
#                     },
#                     "action_elements": [],
#                     "quality_issues": [],
#                     "overall_analysis": ""
#                 }
                
#                 for field, default_value in required_fields.items():
#                     if field not in analysis_data:
#                         analysis_data[field] = default_value
                
#                 # content_object_matching 내부 필드 확인
#                 if not isinstance(analysis_data.get("content_object_matching"), dict):
#                     analysis_data["content_object_matching"] = required_fields["content_object_matching"]
                
#                 logger.info("✅ LLM 응답 파싱 성공")
#                 return analysis_data
#             else:
#                 raise ValueError("JSON 형식을 찾을 수 없음")
                
#         except Exception as e:
#             logger.error(f"❌ LLM 응답 파싱 실패: {e}")
#             logger.info("🔄 Fallback 파싱 시도")
#             return self._fallback_parse(raw_response)

#     def _fallback_parse(self, raw_response: str) -> Dict:
#         """Fallback 파싱 (응답 실패시)"""
        
#         # 응답에서 키워드 추출 시도
#         visual_keywords = re.findall(r'(차트|그래프|이미지|텍스트|버튼|아이콘)', raw_response)
#         spam_keywords = []
        
#         for keyword in SPAM_KEYWORDS.keys():
#             if keyword in raw_response:
#                 spam_keywords.append(keyword)
        
#         return {
#             "visual_elements": visual_keywords[:5],
#             "frame_descriptions": [f"파싱 실패: {raw_response[:100]}..."],
#             "layout_consistency": "보통",
#             "layout_analysis": "파싱 실패로 분석 불가",
#             "spam_indicators": spam_keywords,
#             "content_object_matching": {
#                 "mentioned_in_text": [],
#                 "visible_in_frames": visual_keywords,
#                 "matching_analysis": "파싱 실패"
#             },
#             "action_elements": [],
#             "quality_issues": ["분석 데이터 파싱 실패"],
#             "overall_analysis": f"LLM 응답 파싱 실패. 원본 응답: {raw_response[:200]}..."
#         }

#     def _mock_analysis_data(self, metadata: Dict, ocr_text: str, frames: List[str]) -> Dict:
#         """Mock 분석 데이터 생성"""
#         logger.info("🎭 Mock 분석 데이터 생성")
        
#         title = metadata.get('title', '')
#         description = metadata.get('description', '')
#         full_text = f"{title} {description} {ocr_text}".lower()
        
#         # Mock 스팸 지표 탐지
#         spam_indicators = []
#         for keyword in SPAM_KEYWORDS.keys():
#             if keyword in full_text:
#                 spam_indicators.append(f"Mock 탐지: {keyword}")
        
#         # Mock 레이아웃 일관성 판단
#         layout_consistency = "높음" if len(frames) > 3 else "보통"
        
#         # Mock 시각 요소
#         visual_elements = ["Mock 텍스트", "Mock 이미지", "Mock 차트"]
        
#         # Mock 객체 매칭
#         title_words = title.split()
#         mock_objects = [word for word in title_words if len(word) > 2][:3]
        
#         return {
#             "visual_elements": visual_elements,
#             "frame_descriptions": [
#                 f"Mock 프레임 {i+1}: {title[:30]}..." for i in range(min(len(frames), 3))
#             ],
#             "layout_consistency": layout_consistency,
#             "layout_analysis": f"Mock 분석: {layout_consistency} 수준의 레이아웃 일관성",
#             "spam_indicators": spam_indicators,
#             "content_object_matching": {
#                 "mentioned_in_text": mock_objects,
#                 "visible_in_frames": mock_objects[:2],  # 일부만 일치
#                 "matching_analysis": "Mock 분석: 부분적 일치"
#             },
#             "action_elements": ["Mock 동작 요소"],
#             "quality_issues": [],
#             "overall_analysis": "Mock 모드에서 생성된 분석 데이터"
#         }

#     def _check_spam_patterns(self, metadata: Dict, ocr_text: str) -> Optional[str]:
#         """스팸 패턴 사전 체크 (즉시 분류용)"""
#         title = metadata.get("title", "")
#         description = metadata.get("description", "")
#         full_text = f"{title} {description} {ocr_text}".lower()
        
#         # 정규식 패턴 체크
#         for pattern in SPAM_PATTERNS:
#             matches = re.findall(pattern, full_text)
#             if matches:
#                 return f"스팸 패턴 매칭: {pattern} → {matches}"
        
#         # 고위험 키워드 체크 (가중치 0.9 이상)
#         high_risk_keywords = [k for k, v in SPAM_KEYWORDS.items() if v >= 0.9]
#         detected_high_risk = [k for k in high_risk_keywords if k in full_text]
        
#         if len(detected_high_risk) >= 2:
#             return f"고위험 키워드 다수 검출: {detected_high_risk}"
        
#         return None

#     def _calculate_confidence(self, scores: Dict) -> float:
#         """신뢰도 계산"""
#         cis_score = abs(scores.get('cis_final', 0))
        
#         # CIS 절댓값이 클수록 확신도 높음
#         if cis_score >= 0.5:
#             confidence = 0.9
#         elif cis_score >= 0.3:
#             confidence = 0.8
#         elif cis_score >= 0.1:
#             confidence = 0.7
#         else:
#             confidence = 0.6
        
#         return round(confidence, 3)

#     def _generate_reasoning(self, scores: Dict, analysis_data: Dict, category: str) -> str:
#         """판단 근거 생성"""
#         cis = scores.get('cis_final', 0)
#         c1 = scores.get('c1_spam_score', 0)
#         c2 = scores.get('c2_pattern_score', 0)  
#         c3 = scores.get('c3_context_score', 0)
        
#         reasoning = f"🎯 기획서 기반 CIS 분류 시스템\n"
#         reasoning += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
#         reasoning += f"📊 점수 계산 결과:\n"
#         reasoning += f"  • C1 (어그로/스팸): {c1:.3f}\n"
#         reasoning += f"  • C2 (공장형 패턴): {c2:.3f}\n"
#         reasoning += f"  • C3 (맥락 품질): {c3:.3f}\n"
#         reasoning += f"  • CIS 최종 점수: {cis:.3f}\n"
#         reasoning += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        
#         # 카테고리별 상세 설명
#         if category == "C1":
#             spam_indicators = analysis_data.get('spam_indicators', [])
#             reasoning += f"🚨 C1 (어그로/스팸) 판정:\n"
#             reasoning += f"  • CIS < -0.5이며 C1 점수가 높음\n"
#             reasoning += f"  • 감지된 스팸 지표: {', '.join(spam_indicators[:3])}\n"
#         elif category == "C2":
#             layout = analysis_data.get('layout_consistency', '보통')
#             reasoning += f"🏭 C2 (공장형 패턴) 판정:\n"
#             reasoning += f"  • CIS < -0.5이며 C2 점수가 높음\n"
#             reasoning += f"  • 레이아웃 일관성: {layout}\n"
#         elif category == "C3":
#             quality_issues = analysis_data.get('quality_issues', [])
#             reasoning += f"⚠️ C3 (품질 불량) 판정:\n"
#             reasoning += f"  • CIS < -0.2 또는 맥락 품질 저하\n"
#             if quality_issues:
#                 reasoning += f"  • 품질 문제: {', '.join(quality_issues[:2])}\n"
#         else:  # C5
#             reasoning += f"✅ C5 (정상 영상) 판정:\n"
#             reasoning += f"  • CIS ≥ 0.3 또는 양호한 품질\n"
#             reasoning += f"  • 기본적인 품질 기준 충족\n"
        
#         # 상세 점수 근거 추가
#         if 'detailed_results' in scores:
#             details = scores['detailed_results']
#             if 'cis_details' in details:
#                 cis_reasoning = details['cis_details'].get('reasoning', '')
#                 reasoning += f"\n📝 CIS 계산 근거:\n  {cis_reasoning}\n"
        
#         return reasoning

#     def _create_spam_result(self, spam_reason: str, processing_time: float) -> AnalysisResult:
#         """스팸 감지 결과 생성"""
#         return AnalysisResult(
#             c_category=ContentCategory.C1,
#             reasoning_log=f"🚨 즉시 스팸 탐지\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n{spam_reason}\n\n이 패턴은 명백한 스팸/어그로 지표로 사전 정의되어 있어 C1으로 분류됩니다.",
#             confidence_score=0.95,
#             raw_response="immediate_spam_detection",
#             processing_time=processing_time,
#             precise_scores={
#                 "c1_spam_score": 2.0,
#                 "c2_pattern_score": 0.0,
#                 "c3_context_score": 0.2,
#                 "cis_final": -1.0
#             }
#         )

#     def _create_error_result(self, error_msg: str, processing_time: float) -> AnalysisResult:
#         """오류 결과 생성"""
#         return AnalysisResult(
#             c_category=ContentCategory.C5,
#             reasoning_log=f"❌ 분석 중 오류 발생\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n오류 내용: {error_msg}\n\n오류 발생시 안전을 위해 정상(C5)으로 분류합니다.",
#             confidence_score=0.1,
#             error_message=error_msg,
#             processing_time=processing_time,
#             precise_scores={
#                 "c1_spam_score": 0.0,
#                 "c2_pattern_score": 0.0,
#                 "c3_context_score": 0.5,
#                 "cis_final": 0.5
#             }
#         )

# # 테스트 함수
# def test_content_analyzer():
#     """콘텐츠 분석기 테스트"""
#     print("🧪 기획서 기반 콘텐츠 분석기 테스트")
#     print("=" * 60)
    
#     # Mock 모드로 테스트
#     analyzer = ContentAnalyzer(mode="GPT4o")
    
#     # 테스트 데이터
#     test_metadata = {
#         "title": "🔥충격🔥 무료 증정! 투자 수익률 200% 보장",
#         "description": "선착순 100명만! 카톡으로 문의하세요",
#         "channel_name": "돈버는법채널",
#         "view_count": 50000,
#         "duration": 58
#     }
    
#     test_frames = ["mock_frame_1", "mock_frame_2", "mock_frame_3"]
#     test_ocr = "무료로 드립니다. 투자 수익률 200% 보장. 카톡 문의 바랍니다."
    
#     # 분석 실행
#     result = analyzer.analyze(
#         frames=test_frames,
#         ocr_text=test_ocr,
#         layout_score=0.75,
#         metadata=test_metadata
#     )
    
#     print(f"\n📊 분석 결과:")
#     print(f"  분류: {result.c_category.value}")
#     print(f"  상태: {result.status.value}")
#     print(f"  신뢰도: {result.confidence_score:.3f}")
#     print(f"  처리 시간: {result.processing_time:.2f}초")
    
#     if result.precise_scores:
#         print(f"\n📈 정밀 점수:")
#         for key, value in result.precise_scores.items():
#             print(f"    {key}: {value}")
    
#     print(f"\n📝 판단 근거:")
#     print(f"{result.reasoning_log}")

# if __name__ == "__main__":
#     test_content_analyzer()