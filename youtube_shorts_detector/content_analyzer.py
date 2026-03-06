"""
멀티모달 콘텐츠 분석기
gpt-4o-mini에 메타데이터 텍스트 + 프레임 이미지(base64) 전송
"""
import json
import time
import random
import logging
import re
from typing import Dict, Any, Optional, List
from openai import OpenAI

from config import (
    GPT4O_CONFIG, QWEN_CONFIG, RETRY_CONFIG,
    MOCK_CLASSIFICATION_RULES, SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE, MOCK_MODE, SPAM_PATTERNS,
)
from models import LLMResponse, AnalysisResult, ContentCategory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """유튜브 쇼츠 콘텐츠 분석 클래스"""

    def __init__(self, mode: str = "GPT4o", mock_mode: bool = None):
        self.mode = mode
        self.mock_mode = mock_mode if mock_mode is not None else MOCK_MODE

        if not self.mock_mode:
            self.client = self._setup_client()
            self.model_name = self._get_model_name()
            logger.info(f"✅ ContentAnalyzer: {mode} ({self.model_name})")
        else:
            logger.info("🎭 ContentAnalyzer: Mock 모드")

    def _setup_client(self) -> OpenAI:
        config = GPT4O_CONFIG if self.mode == "GPT4o" else QWEN_CONFIG
        return OpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
            timeout=config["timeout"]
        )

    def _get_model_name(self) -> str:
        return GPT4O_CONFIG["model"] if self.mode == "GPT4o" else QWEN_CONFIG["model"]

    # =============================================
    # 메인 분석
    # =============================================

    def analyze(self,
                frames: List[str],
                ocr_text: str,
                layout_score: float = 0.0,
                metadata: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """
        콘텐츠 분석
        - frames: base64 인코딩된 프레임 이미지 리스트 (최대 10장)
        - metadata: 제목/설명/채널명 등
        """
        start_time = time.time()

        try:
            # ✅ Step 1: 스팸 패턴 사전 체크 (API 호출 없이 즉시 C1 판단)
            spam_result = self._check_spam_patterns(metadata, ocr_text)
            if spam_result:
                processing_time = time.time() - start_time
                logger.info(f"🚨 스팸 패턴 감지 → 즉시 C1 분류 (API 호출 생략)")
                return AnalysisResult(
                    c_category=ContentCategory.C1,
                    reasoning_log=spam_result,
                    confidence_score=0.95,
                    raw_response="spam_pattern_detected",
                    processing_time=processing_time
                )

            # Step 2: 스팸 아니면 gpt-4o-mini 호출
            if self.mock_mode:
                llm_result = self._mock_analyze(frames, ocr_text, layout_score, metadata)
                raw_response = "Mock response"
            else:
                raw_response = self._call_gpt4o_mini(frames, ocr_text, layout_score, metadata)
                llm_result = self._parse_llm_response(raw_response)

            processing_time = time.time() - start_time
            logger.info(
                f"✅ 분석 완료: {llm_result.c_category.value} "
                f"(신뢰도: {llm_result.confidence_score:.2f}, "
                f"{processing_time:.2f}초)"
            )

            return AnalysisResult(
                c_category=llm_result.c_category,
                reasoning_log=llm_result.reasoning_log,
                confidence_score=llm_result.confidence_score,
                raw_response=raw_response,
                processing_time=processing_time
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ 분석 실패: {e}")
            return AnalysisResult(
                c_category=ContentCategory.C5,
                reasoning_log=f"분석 실패: {str(e)}",
                confidence_score=0.0,
                error_message=str(e),
                processing_time=processing_time
            )

    def _check_spam_patterns(self, metadata: Dict, ocr_text: str) -> Optional[str]:
        """
        ✅ 스팸 패턴 사전 체크
        제목 + 설명 + 자막 텍스트에서 SPAM_PATTERNS 정규식 매칭
        매칭되면 근거 문자열 반환, 없으면 None
        """
        # 검사할 텍스트 합치기
        check_text = " ".join([
            metadata.get("title", "") if metadata else "",
            metadata.get("description", "") if metadata else "",
            ocr_text or ""
        ])

        matched_patterns = []
        for pattern in SPAM_PATTERNS:
            match = re.search(pattern, check_text, re.IGNORECASE)
            if match:
                matched_patterns.append(f"'{match.group()}' 감지 (패턴: {pattern})")

        if matched_patterns:
            return (
                f"스팸 패턴 {len(matched_patterns)}개 감지:\n"
                + "\n".join(f"  - {p}" for p in matched_patterns)
            )
        return None

    # =============================================
    # gpt-4o-mini 호출
    # =============================================

    def _call_gpt4o_mini(self,
                         frames: List[str],
                         ocr_text: str,
                         layout_score: float,
                         metadata: Dict) -> str:
        """
        gpt-4o-mini 멀티모달 호출
        USER_PROMPT_TEMPLATE + base64 프레임 이미지 전송
        """
        # ✅ USER_PROMPT_TEMPLATE으로 유저 프롬프트 구성
        user_text = USER_PROMPT_TEMPLATE.format(
            title=metadata.get("title", "") if metadata else "",
            description=metadata.get("description", "") if metadata else "",
            channel_name=metadata.get("channel_name", "") if metadata else "",
            view_count=f"{metadata.get('view_count', 0):,}" if metadata else "0",
            duration=metadata.get("duration", 0) if metadata else 0,
            frame_count=len(frames)
        )

        # ✅ content 구성: 텍스트 + 이미지 순서로
        content = [{"type": "text", "text": user_text}]

        if frames:
            logger.info(f"🖼️ 프레임 {len(frames)}장 첨부")
            for idx, frame_b64 in enumerate(frames):
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{frame_b64}",
                        "detail": "low"  # 비용 절감 (low: ~85토큰/장)
                    }
                })
                logger.debug(f"  이미지 {idx + 1}/{len(frames)} 첨부")
        else:
            logger.info("ℹ️ 프레임 없음, 텍스트만으로 분석")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content}
        ]

        # 재시도 로직
        for attempt in range(RETRY_CONFIG["max_retries"]):
            try:
                logger.info(
                    f"🤖 gpt-4o-mini 호출 "
                    f"(프레임 {len(frames)}장, "
                    f"시도 {attempt + 1}/{RETRY_CONFIG['max_retries']})"
                )

                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=GPT4O_CONFIG["max_tokens"],
                    temperature=GPT4O_CONFIG["temperature"]
                )

                result = response.choices[0].message.content
                logger.info(f"📨 응답 수신:\n{result}")
                return result

            except Exception as e:
                logger.warning(f"⚠️ API 호출 실패 (시도 {attempt + 1}): {e}")
                if attempt < RETRY_CONFIG["max_retries"] - 1:
                    sleep_time = RETRY_CONFIG["retry_delay"] * (
                        RETRY_CONFIG["backoff_factor"] ** attempt
                    )
                    logger.info(f"⏳ {sleep_time:.1f}초 후 재시도...")
                    time.sleep(sleep_time)
                else:
                    raise e

    # =============================================
    # 응답 파싱
    # =============================================

    def _parse_llm_response(self, raw_response: str) -> LLMResponse:
        """
        gpt-4o-mini 응답 파싱
        새 JSON 필드: c_category, confidence_score, subtitle_detected,
                      subtitle_content, frame_analysis, reasoning_log
        """
        try:
            json_str = self._extract_json(raw_response)
            response_dict = json.loads(json_str)

            # ✅ 추가 필드 로깅
            subtitle_detected = response_dict.get("subtitle_detected", False)
            subtitle_content = response_dict.get("subtitle_content", "자막 없음")
            frame_analysis = response_dict.get("frame_analysis", "")

            logger.info(f"📝 자막 감지: {subtitle_detected}")
            logger.info(f"📝 자막 내용: {subtitle_content}")
            logger.info(f"🖼️ 프레임 분석: {frame_analysis}")

            # reasoning_log에 자막 정보 + 프레임 분석 통합
            reasoning = response_dict.get("reasoning_log", "")
            full_reasoning = (
                f"[자막] {subtitle_content}\n"
                f"[프레임] {frame_analysis}\n"
                f"[판단] {reasoning}"
            )

            return LLMResponse(
                c_category=response_dict["c_category"],
                reasoning_log=full_reasoning,
                confidence_score=float(response_dict["confidence_score"])
            )

        except Exception as e:
            logger.warning(f"⚠️ JSON 파싱 실패, fallback: {e}")
            return self._fallback_parse(raw_response)

    def _extract_json(self, response: str) -> str:
        """응답에서 JSON 추출"""
        # ```json 블록 처리
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            return response[start:end].strip()

        # 중괄호 기반 추출
        start = response.find("{")
        if start != -1:
            brace_count = 0
            for i, char in enumerate(response[start:], start):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return response[start:i + 1]
        return response

    def _fallback_parse(self, raw_response: str) -> LLMResponse:
        """파싱 실패 시 키워드 기반 파싱"""
        category = ContentCategory.C5
        for cat in ContentCategory:
            if cat.value in raw_response.upper():
                category = cat
                break

        confidence_match = re.search(r'0\.\d+', raw_response)
        confidence = float(confidence_match.group()) if confidence_match else 0.3

        return LLMResponse(
            c_category=category,
            reasoning_log=f"Fallback 파싱 적용: {raw_response[:200]}",
            confidence_score=confidence
        )

    # =============================================
    # Mock 분석 (테스트용)
    # =============================================

    def _mock_analyze(self, frames, ocr_text, layout_score, metadata) -> LLMResponse:
        """Mock 분석 (MOCK_MODE=True 일 때)"""
        logger.info("🎭 Mock 분석 수행 중...")
        time.sleep(random.uniform(0.3, 0.8))
        result = self._classify_by_rules(ocr_text, layout_score, metadata)
        return LLMResponse(
            c_category=result["category"],
            reasoning_log=result["reasoning"],
            confidence_score=result["confidence"]
        )

    def _classify_by_rules(self, ocr_text, layout_score, metadata) -> Dict:
        """규칙 기반 분류 (Mock용)"""
        text_lower = ocr_text.lower()
        title = (metadata.get("title", "") if metadata else "").lower()

        c1_matches = sum(
            1 for k in MOCK_CLASSIFICATION_RULES["C1_keywords"]
            if k.lower() in text_lower or k.lower() in title
        )
        if c1_matches >= 2:
            return {
                "category": ContentCategory.C1,
                "reasoning": f"어그로성 키워드 {c1_matches}개 감지",
                "confidence": min(0.7 + c1_matches * 0.1, 0.95)
            }

        c2_matches = sum(
            1 for k in MOCK_CLASSIFICATION_RULES["C2_keywords"]
            if k.lower() in text_lower
        )
        if c2_matches >= 1 or layout_score < 0.3:
            return {
                "category": ContentCategory.C2,
                "reasoning": "공장형 패턴 의심",
                "confidence": 0.75
            }

        return {
            "category": ContentCategory.C5,
            "reasoning": "정상 콘텐츠로 판단",
            "confidence": 0.80
        }

    def switch_model(self, new_mode: str) -> None:
        """모델 교체"""
        logger.info(f"🔄 모델 교체: {self.mode} → {new_mode}")
        self.mode = new_mode
        if not self.mock_mode:
            self.client = self._setup_client()
            self.model_name = self._get_model_name()
