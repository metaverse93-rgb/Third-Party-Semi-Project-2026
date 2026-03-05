"""
멀티모달 LMM 기반 콘텐츠 분석기 (Mock 모드 지원)
GPT-4o ↔ Qwen2.5-VL 무중단 교체 지원
"""
import json
import time
import random
import logging
from typing import Dict, Any, Optional
from openai import OpenAI

from config import *
from models import LLMResponse, AnalysisResult, ContentCategory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentAnalyzer:
    """유튜브 쇼츠 콘텐츠 분석 클래스 (Mock 모드 지원)"""
    
    def __init__(self, mode: str = "GPT4o", mock_mode: bool = None):
        """
        초기화
        Args:
            mode: "GPT4o" 또는 "Qwen" 선택
            mock_mode: None이면 config.MOCK_MODE 사용, True/False로 강제 설정
        """
        self.mode = mode
        self.mock_mode = mock_mode if mock_mode is not None else MOCK_MODE
        
        if not self.mock_mode:
            self.client = self._setup_client()
            self.model_name = self._get_model_name()
            
        logger.info(f"ContentAnalyzer 초기화 완료: {mode} 모드 (Mock: {self.mock_mode})")
    
    def _setup_client(self) -> OpenAI:
        """모드별 OpenAI 클라이언트 설정"""
        if self.mode == "GPT4o":
            config = GPT4O_CONFIG
        elif self.mode == "Qwen":
            config = QWEN_CONFIG
        else:
            raise ValueError(f"지원하지 않는 모드: {self.mode}")
        
        return OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=config["timeout"]
        )
    
    def _get_model_name(self) -> str:
        """모드별 모델명 반환"""
        return GPT4O_CONFIG["model"] if self.mode == "GPT4o" else QWEN_CONFIG["model"]
    
    def analyze(self, 
                frames: list[str], 
                ocr_text: str, 
                layout_score: float = 0.0,
                metadata: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """
        멀티모달 콘텐츠 분석 수행
        """
        start_time = time.time()
        
        try:
            if self.mock_mode:
                # Mock 모드: 로컬 로직 사용
                llm_result = self._mock_analyze(frames, ocr_text, layout_score, metadata)
                raw_response = f"Mock response from {self.mode}"
            else:
                # 실제 API 호출
                raw_response = self._call_llm_with_retry(frames, ocr_text, layout_score, metadata)
                llm_result = self._parse_llm_response(raw_response)
            
            processing_time = time.time() - start_time
            
            # 최종 결과 구성
            return AnalysisResult(
                c_category=llm_result.c_category,
                reasoning_log=llm_result.reasoning_log,
                confidence_score=llm_result.confidence_score,
                raw_response=raw_response,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"분석 실패: {str(e)}")
            
            return AnalysisResult(
                c_category=ContentCategory.C5,  # 기본값
                reasoning_log=f"분석 실패: {str(e)}",
                confidence_score=0.0,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _mock_analyze(self, frames: list[str], ocr_text: str, 
                     layout_score: float, metadata: Dict) -> LLMResponse:
        """
        Mock 분석 로직: 실제 LMM 대신 로컬 규칙 기반 분류
        """
        logger.info("🎭 Mock 분석 수행 중...")
        
        # 시뮬레이션 지연 시간
        time.sleep(random.uniform(0.5, 2.0))
        
        # OCR 텍스트와 메타데이터 기반 분류
        analysis_result = self._classify_by_rules(ocr_text, layout_score, metadata)
        
        return LLMResponse(
            c_category=analysis_result["category"],
            reasoning_log=analysis_result["reasoning"],
            confidence_score=analysis_result["confidence"]
        )
    
    def _classify_by_rules(self, ocr_text: str, layout_score: float, 
                          metadata: Dict) -> Dict:
        """규칙 기반 분류 로직"""
        text_lower = ocr_text.lower()
        title = metadata.get("title", "") if metadata else ""
        
        # C1: 어그로/스팸 검사
        c1_matches = sum(1 for keyword in MOCK_CLASSIFICATION_RULES["C1_keywords"] 
                        if keyword.lower() in text_lower or keyword.lower() in title.lower())
        
        if c1_matches >= 2:
            return {
                "category": ContentCategory.C1,
                "reasoning": f"어그로성 키워드 {c1_matches}개 감지: {ocr_text[:50]}...",
                "confidence": min(0.7 + c1_matches * 0.1, 0.95)
            }
        
        # C2: 공장형 패턴 검사
        c2_matches = sum(1 for keyword in MOCK_CLASSIFICATION_RULES["C2_keywords"] 
                        if keyword.lower() in text_lower)
        
        if c2_matches >= 1 or layout_score < 0.3:
            return {
                "category": ContentCategory.C2,
                "reasoning": f"공장형 패턴 의심 (키워드: {c2_matches}, 레이아웃: {layout_score:.2f})",
                "confidence": 0.75
            }
        
        # C3: 품질 불량 검사
        c3_matches = sum(1 for keyword in MOCK_CLASSIFICATION_RULES["C3_keywords"] 
                        if keyword.lower() in text_lower)
        
        if c3_matches >= 1 or (metadata and len(metadata.get("title", "")) > 50 and len(ocr_text) < 10):
            return {
                "category": ContentCategory.C3,
                "reasoning": f"품질 문제 감지 (제목-내용 불일치 또는 오류 키워드)",
                "confidence": 0.68
            }
        
        # C4: 무단 도용 검사
        c4_matches = sum(1 for keyword in MOCK_CLASSIFICATION_RULES["C4_keywords"] 
                        if keyword.lower() in text_lower)
        
        if c4_matches >= 1:
            return {
                "category": ContentCategory.C4,
                "reasoning": f"저작권 침해 의심 키워드 감지",
                "confidence": 0.82
            }
        
        # C5: 정상 영상 (기본값)
        normal_matches = sum(1 for keyword in MOCK_CLASSIFICATION_RULES["normal_indicators"] 
                            if keyword.lower() in text_lower or keyword.lower() in title.lower())
        
        if normal_matches >= 1 and layout_score > 0.6:
            return {
                "category": ContentCategory.C5,
                "reasoning": f"정상 콘텐츠로 판단 (교육/정보성 키워드 {normal_matches}개, 양호한 레이아웃)",
                "confidence": 0.88
            }
        
        # 기본값: 애매한 경우
        return {
            "category": ContentCategory.C5,
            "reasoning": f"명확한 분류 기준 없음, 기본값으로 정상 분류 (레이아웃: {layout_score:.2f})",
            "confidence": 0.55
        }
    
    def _call_llm_with_retry(self, frames: list[str], ocr_text: str, 
                            layout_score: float, metadata: Dict) -> str:
        """재시도 로직이 포함된 LMM 호출 (실제 API용)"""
        
        user_prompt = self._build_prompt(ocr_text, layout_score, metadata)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": user_prompt}
                ] + [
                    {"type": "image_url", "image_url": {"url": frame}} 
                    for frame in frames[:3]
                ]
            }
        ]
        
        for attempt in range(RETRY_CONFIG["max_retries"]):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=GPT4O_CONFIG["max_tokens"],
                    temperature=GPT4O_CONFIG["temperature"]
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                logger.warning(f"LMM 호출 실패 (시도 {attempt + 1}/{RETRY_CONFIG['max_retries']}): {e}")
                
                if attempt < RETRY_CONFIG["max_retries"] - 1:
                    time.sleep(RETRY_CONFIG["retry_delay"] * (RETRY_CONFIG["backoff_factor"] ** attempt))
                else:
                    raise e
    
    def _build_prompt(self, ocr_text: str, layout_score: float, metadata: Dict) -> str:
        """구조화된 분석 프롬프트 생성"""
        return f"""
다음 유튜브 쇼츠 영상을 분석해주세요:

**OCR 텍스트**: {ocr_text}
**레이아웃 점수**: {layout_score:.2f}
**메타데이터**: {json.dumps(metadata, ensure_ascii=False)}

위 정보와 제공된 이미지를 종합하여 C1~C5 카테고리로 분류하고,
판별 근거와 신뢰도를 JSON 형태로 제공해주세요.
"""
    
    def _parse_llm_response(self, raw_response: str) -> LLMResponse:
        """LMM 응답을 구조화된 객체로 파싱"""
        try:
            json_str = self._extract_json_from_response(raw_response)
            response_dict = json.loads(json_str)
            return LLMResponse(**response_dict)
            
        except Exception as e:
            logger.warning(f"JSON 파싱 실패, fallback 적용: {e}")
            return self._fallback_parse(raw_response)
    
    def _extract_json_from_response(self, response: str) -> str:
        """응답에서 JSON 부분만 추출"""
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            return response[start:end].strip()
        
        start = response.find("{")
        if start != -1:
            brace_count = 0
            for i, char in enumerate(response[start:], start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return response[start:i+1]
        
        return response
    
    def _fallback_parse(self, raw_response: str) -> LLMResponse:
        """파싱 실패 시 키워드 기반 대안 파싱"""
        category = ContentCategory.C5
        for cat in ContentCategory:
            if cat.value in raw_response.upper():
                category = cat
                break
        
        import re
        confidence_match = re.search(r'0\.\d+', raw_response)
        confidence = float(confidence_match.group()) if confidence_match else 0.3
        
        return LLMResponse(
            c_category=category,
            reasoning_log=f"Fallback 파싱: {raw_response[:100]}...",
            confidence_score=confidence
        )

    def switch_model(self, new_mode: str) -> None:
        """실행 중 모델 교체"""
        logger.info(f"모델 교체: {self.mode} → {new_mode}")
        self.mode = new_mode
        if not self.mock_mode:
            self.client = self._setup_client()
            self.model_name = self._get_model_name()