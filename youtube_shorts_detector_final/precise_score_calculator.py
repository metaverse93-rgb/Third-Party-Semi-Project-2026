"""
기획서 기반 정밀 점수 계산기
C1, C2, C3 각 점수를 기획서 공식에 따라 계산하고 CIS로 통합
"""
import re
import base64
import cv2
import numpy as np
import logging
from typing import Dict, List, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim

from config import (
    C1_WEIGHTS, C3_WEIGHTS, CIS_WEIGHTS,
    SPAM_KEYWORDS, SPAM_PATTERNS, SPAM_REFERENCE_SENTENCES,
    SSIM_THRESHOLD, CIS_CLASSIFICATION_THRESHOLDS, DETAILED_CLASSIFICATION
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreciseScoreCalculator:
    """기획서 기반 정밀 점수 계산 클래스"""
    
    def __init__(self):
        """초기화"""
        logger.info("🧮 PreciseScoreCalculator 초기화 (기획서 공식 기반)")
        
        # Sentence Transformer 모델 로드
        try:
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("✅ SentenceTransformer 모델 로드 완료")
        except Exception as e:
            logger.warning(f"⚠️ SentenceTransformer 로드 실패, Mock 모드로 전환: {e}")
            self.sentence_model = None
    
    def calculate_c1_score(self, analysis_data: Dict, ocr_text: str, metadata: Dict) -> Dict[str, Any]:
        """
        기획서 2-2-가: Score_C1 = (w1a · Σ Risk_keyword) + (w1b · Sim_semantic)
        어그로/스팸 점수 계산
        """
        logger.info("📊 C1 점수 계산 시작 (어그로/스팸)")
        
        # 전체 텍스트 구성
        title = metadata.get('title', '')
        description = metadata.get('description', '')
        full_text = f"{title} {description} {ocr_text}".lower()
        
        # 1️⃣ Σ Risk_keyword 계산 (w1a 가중치)
        keyword_risk_score = 0.0
        detected_keywords = []
        
        # 정적 키워드 점수
        for keyword, weight in SPAM_KEYWORDS.items():
            if keyword in full_text:
                keyword_risk_score += weight
                detected_keywords.append(f"{keyword}({weight})")
        
        # 정규식 패턴 점수
        for pattern in SPAM_PATTERNS:
            matches = re.findall(pattern, full_text)
            if matches:
                pattern_score = len(matches) * 0.5
                keyword_risk_score += pattern_score
                detected_keywords.extend([f"패턴:{m}" for m in matches])
        
        # LLM 발견 스팸 지표 추가 (높은 가중치)
        spam_indicators = analysis_data.get('spam_indicators', [])
        for indicator in spam_indicators:
            keyword_risk_score += 0.8
            detected_keywords.append(f"LLM:{indicator}")
        
        # 2️⃣ Sim_semantic 계산 (w1b 가중치)
        semantic_score = 0.0
        if self.sentence_model and full_text.strip():
            try:
                # 입력 텍스트 임베딩
                text_embedding = self.sentence_model.encode([full_text])
                
                # 스팸 참조 문장들과 유사도 계산
                spam_embeddings = self.sentence_model.encode(SPAM_REFERENCE_SENTENCES)
                similarities = cosine_similarity(text_embedding, spam_embeddings)
                semantic_score = float(np.max(similarities))
                
            except Exception as e:
                logger.warning(f"⚠️ 의미 유사도 계산 실패: {e}")
                semantic_score = 0.0
        else:
            # Mock 모드 또는 모델 없을 때
            semantic_score = self._mock_semantic_similarity(full_text)
        
        # 3️⃣ keyword_risk_score 정규화 (0~1)
        # 최대 기대값: 고위험 키워드 3개(2.7) + LLM 지표 2개(1.6) = 4.3 수준
        MAX_KEYWORD_SCORE = 4.0
        keyword_risk_normalized = min(1.0, keyword_risk_score / MAX_KEYWORD_SCORE)

        # 4️⃣ C1 점수 계산: Score_C1 = (w1a · Σ Risk_keyword) + (w1b · Sim_semantic)
        score_c1 = min(1.0,
            (C1_WEIGHTS["keyword"] * keyword_risk_normalized) +
            (C1_WEIGHTS["semantic"] * semantic_score)
        )
        
        result = {
            'score': float(score_c1),
            'keyword_score': float(keyword_risk_score),
            'keyword_score_normalized': float(keyword_risk_normalized),
            'semantic_score': float(semantic_score),
            'detected_keywords': detected_keywords,
            'reasoning': (
                f"키워드 위험도: {keyword_risk_score:.3f} → 정규화: {keyword_risk_normalized:.3f} × {C1_WEIGHTS['keyword']} "
                f"+ 의미 유사도: {semantic_score:.3f} × {C1_WEIGHTS['semantic']} = {score_c1:.3f}"
            )
        }
        
        logger.info(f"📈 C1 점수: {score_c1:.3f} (키워드:{keyword_risk_score:.3f}→{keyword_risk_normalized:.3f}, 의미:{semantic_score:.3f})")
        return result
    
    def calculate_c2_score(self, analysis_data: Dict, frames: List[str]) -> Dict[str, Any]:
        """
        기획서 2-2-나: Score_C2 = (1 / T-1) · Σ SSIM(Rt, Rt+1)
        실제 프레임 이미지 간 SSIM 계산으로 공장형 패턴 판별
        frames: base64 인코딩된 이미지 리스트
        """
        logger.info("🏭 C2 점수 계산 시작 (실제 SSIM 계산)")

        layout_consistency = analysis_data.get('layout_consistency', '보통')
        layout_analysis = analysis_data.get('layout_analysis', '')

        # 실제 SSIM 계산 시도
        ssim_score = self._calculate_real_ssim(frames)

        if ssim_score is not None:
            score_c2 = float(ssim_score)
            method = "실제 SSIM"
        else:
            # 프레임 계산 실패시 LLM 판단 기반 fallback
            logger.warning("⚠️ 실제 SSIM 계산 실패 → LLM 판단 기반 fallback")
            consistency_to_ssim = {
                '높음': 0.80,
                '보통': 0.60,
                '낮음': 0.35
            }
            base_ssim = consistency_to_ssim.get(layout_consistency, 0.60)
            pattern_keywords = ['템플릿', '반복', '일정한', '고정', '동일', '자동']
            pattern_bonus = sum(0.02 for keyword in pattern_keywords if keyword in layout_analysis)
            score_c2 = min(1.0, base_ssim + pattern_bonus)
            method = "LLM fallback"

        result = {
            'score': score_c2,
            'layout_consistency': layout_consistency,
            'is_factory_pattern': score_c2 >= SSIM_THRESHOLD,
            'method': method,
            'reasoning': f"[{method}] C2 점수: {score_c2:.3f} (공장형 임계값: {SSIM_THRESHOLD})"
        }

        logger.info(f"🏭 C2 점수: {score_c2:.3f} [{method}] (공장형 판정: {result['is_factory_pattern']})")
        return result

    def _calculate_real_ssim(self, frames: List[str]) -> float:
        """
        기획서 공식: Score_C2 = (1 / T-1) · Σ SSIM(Rt, Rt+1)
        base64 프레임들을 디코딩해서 인접 프레임 간 SSIM 평균 계산
        """
        if not frames or len(frames) < 2:
            logger.info("🏭 프레임 2장 미만 → SSIM 계산 불가, None 반환")
            return None

        try:
            decoded = []
            for i, frame_b64 in enumerate(frames):
                if not frame_b64:
                    continue
                img_bytes = base64.b64decode(frame_b64)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                # 비교를 위해 동일 크기로 리사이즈 (320x180)
                img = cv2.resize(img, (320, 180))
                decoded.append(img)

            if len(decoded) < 2:
                return None

            # 기획서 공식: (1/T-1) · Σ SSIM(Rt, Rt+1)
            ssim_values = []
            for i in range(len(decoded) - 1):
                score, _ = ssim(decoded[i], decoded[i + 1], full=True)
                ssim_values.append(score)
                logger.debug(f"  SSIM frame{i+1}↔frame{i+2}: {score:.3f}")

            avg_ssim = float(np.mean(ssim_values))
            logger.info(f"🏭 실제 SSIM 계산 완료: {len(ssim_values)}쌍 평균 = {avg_ssim:.3f}")
            return avg_ssim

        except Exception as e:
            logger.warning(f"⚠️ SSIM 계산 오류: {e}")
            return None
    
    def calculate_c3_score(self, analysis_data: Dict, ocr_text: str) -> Dict[str, Any]:
        """
        기획서 2-2-다: Context_Score_C3 = (ws · S) + (wo · O) + (wa · A)
        품질/맥락 일치 점수 계산
        """
        logger.info("🎯 C3 점수 계산 시작 (품질/맥락)")
        
        # S: 의미적 일치도 (BERTScore 기반)
        s_semantic = self._calculate_semantic_alignment(analysis_data, ocr_text)
        
        # O: 객체 존재 여부
        o_existence = self._calculate_object_existence(analysis_data)
        
        # A: 액션 싱크 (동작 일치도)
        a_sync = self._calculate_action_sync(analysis_data, ocr_text)
        
        # C3 점수 계산: Context_Score_C3 = (ws · S) + (wo · O) + (wa · A)
        context_score_c3 = (
            C3_WEIGHTS["semantic"] * s_semantic +
            C3_WEIGHTS["existence"] * o_existence +
            C3_WEIGHTS["sync"] * a_sync
        )
        
        result = {
            'score': float(context_score_c3),
            'semantic_score': float(s_semantic),
            'existence_score': float(o_existence),
            'sync_score': float(a_sync),
            'reasoning': f"의미일치: {s_semantic:.3f} × {C3_WEIGHTS['semantic']} + 객체일치: {o_existence:.3f} × {C3_WEIGHTS['existence']} + 액션싱크: {a_sync:.3f} × {C3_WEIGHTS['sync']} = {context_score_c3:.3f}"
        }
        
        logger.info(f"🎯 C3 점수: {context_score_c3:.3f} (S:{s_semantic:.3f}, O:{o_existence:.3f}, A:{a_sync:.3f})")
        return result
    
    def calculate_cis_final(self, c1_result: Dict, c2_result: Dict, c3_result: Dict) -> Dict[str, Any]:
        """
        기획서 4항: CIS_Final = Context_Score_C3 − (α · Score_C1 + β · Score_C2)
        통합 맥락 점수 계산
        """
        logger.info("🎯 CIS 최종 점수 계산")
        
        c1_score = c1_result['score']
        c2_score = c2_result['score']
        c3_score = c3_result['score']
        
        # 페널티 계산
        c1_penalty = CIS_WEIGHTS["alpha"] * c1_score
        c2_penalty = CIS_WEIGHTS["beta"] * c2_score
        total_penalty = c1_penalty + c2_penalty
        
        # CIS 최종 점수
        cis_final = c3_score - total_penalty
        
        result = {
            'cis_score': float(cis_final),
            'base_quality': float(c3_score),
            'c1_penalty': float(c1_penalty),
            'c2_penalty': float(c2_penalty),
            'total_penalty': float(total_penalty),
            'reasoning': f"CIS = C3({c3_score:.3f}) - (α×C1: {c1_penalty:.3f} + β×C2: {c2_penalty:.3f}) = {cis_final:.3f}"
        }
        
        logger.info(f"🎯 CIS 최종: {cis_final:.3f} = 기본품질({c3_score:.3f}) - 페널티({total_penalty:.3f})")
        return result
    
    def classify_by_cis(self, c1_result: Dict, c2_result: Dict, c3_result: Dict, cis_result: Dict) -> str:
        """
        CIS 점수 기반 최종 분류
    
        """
        cis_score = cis_result['cis_score']
        c1_score = c1_result['score']
        c2_score = c2_result['score']

        logger.info(f"🏷️ CIS 기반 분류 시작 (CIS: {cis_score:.3f})")

        if cis_score >= CIS_CLASSIFICATION_THRESHOLDS["positive"]:
            logger.info(f"→ C5 (정상): CIS {cis_score:.3f} ≥ {CIS_CLASSIFICATION_THRESHOLDS['positive']}")
            return "C5"

        else:
            # CIS < 0.2 → C1/C2 임계값으로 판별, 미달 시 C3
            if c1_score >= DETAILED_CLASSIFICATION["C1_min_score"]:
                logger.info(f"→ C1 (어그로/스팸): CIS {cis_score:.3f} < 0.3, C1({c1_score:.3f}) ≥ {DETAILED_CLASSIFICATION['C1_min_score']}")
                return "C1"
            elif c2_score >= DETAILED_CLASSIFICATION["C2_min_score"]:
                logger.info(f"→ C2 (공장형): CIS {cis_score:.3f} < 0.3, C2({c2_score:.3f}) ≥ {DETAILED_CLASSIFICATION['C2_min_score']}")
                return "C2"
            else:
                logger.info(f"→ C3 (품질불량): CIS {cis_score:.3f} < 0.3, C1/C2 임계값 미달 (C1:{c1_score:.3f}, C2:{c2_score:.3f})")
                return "C3"
    
    def calculate_all_scores(self, analysis_data: Dict, ocr_text: str, frames: List[str], metadata: Dict) -> Dict[str, Any]:
        """모든 점수를 한번에 계산하고 분류까지 수행"""
        logger.info("🧮 전체 정밀 점수 계산 시작")
        
        try:
            # 각 점수 계산
            c1_result = self.calculate_c1_score(analysis_data, ocr_text, metadata)
            c2_result = self.calculate_c2_score(analysis_data, frames)
            c3_result = self.calculate_c3_score(analysis_data, ocr_text)
            cis_result = self.calculate_cis_final(c1_result, c2_result, c3_result)
            
            # 최종 분류
            final_category = self.classify_by_cis(c1_result, c2_result, c3_result, cis_result)
            
            # 통합 결과
            scores = {
                "c1_spam_score": c1_result['score'],
                "c2_pattern_score": c2_result['score'],
                "c3_context_score": c3_result['score'],
                "cis_final": cis_result['cis_score'],
                "final_category": final_category,
                "detailed_results": {
                    "c1_details": c1_result,
                    "c2_details": c2_result,
                    "c3_details": c3_result,
                    "cis_details": cis_result
                }
            }
            
            logger.info(f"✅ 전체 점수 계산 완료: {final_category} (CIS: {cis_result['cis_score']:.3f})")
            return scores
            
        except Exception as e:
            logger.error(f"❌ 점수 계산 실패: {e}")
            return {
                "c1_spam_score": 0.0,
                "c2_pattern_score": 0.0,
                "c3_context_score": 0.5,
                "cis_final": 0.5,
                "final_category": "C5",
                "error": str(e)
            }
    
    # =============================================
    # 헬퍼 메서드들
    # =============================================
    
    def _calculate_semantic_alignment(self, analysis_data: Dict, ocr_text: str) -> float:
        """의미적 일치도 계산 (BERTScore 기반)
        
        OCR 텍스트가 없는 경우: frame_descriptions와 overall_analysis 기반으로
        시각적 일관성 점수를 산출하여 이미지만으로도 분석 가능하게 처리
        """
        frame_descriptions = analysis_data.get('frame_descriptions', [])
        has_ocr = bool(ocr_text and ocr_text.strip())

        if not frame_descriptions:
            return 0.5  # 프레임 묘사 자체가 없으면 기본값

        # OCR 없음: 이미지 기반 분석
        if not has_ocr:
            overall = analysis_data.get('overall_analysis', '')
            quality_issues = analysis_data.get('quality_issues', [])
            spam_indicators = analysis_data.get('spam_indicators', [])

            base = 0.75                                            # 0.6 → 0.75 (텍스트 없는 정상 영상 고려)
            base += min(len(frame_descriptions) * 0.02, 0.10)     # 프레임 묘사 수
            base += 0.05 if overall else 0.0                      # overall_analysis 존재
            base -= len(quality_issues) * 0.05                    # 품질 문제 패널티
            base -= len(spam_indicators) * 0.03                   # 스팸 지표 패널티
            score = round(max(0.0, min(1.0, base)), 3)
            logger.info(f"이미지 기반 S_semantic (OCR 없음): {score}")
            return score

        # OCR 있음: 프레임 묘사 <-> OCR 텍스트 유사도
        if self.sentence_model:
            try:
                combined_desc = ' '.join(frame_descriptions)
                desc_embedding = self.sentence_model.encode([combined_desc])
                text_embedding = self.sentence_model.encode([ocr_text])
                similarity = cosine_similarity(desc_embedding, text_embedding)[0][0]
                return float(similarity)
            except Exception as e:
                logger.warning(f"의미 일치도 계산 실패: {e}")

        return self._mock_semantic_alignment(frame_descriptions, ocr_text)
    
    def _calculate_object_existence(self, analysis_data: Dict) -> float:
        """객체 존재 일치도 계산"""
        content_matching = analysis_data.get('content_object_matching', {})
        mentioned_objects = content_matching.get('mentioned_in_text', [])
        visible_objects = content_matching.get('visible_in_frames', [])
        
        if not mentioned_objects:
            return 0.8  # 언급된 객체가 없으면 높은 기본값
        
        mentioned_set = set([obj.lower().strip() for obj in mentioned_objects])
        visible_set = set([obj.lower().strip() for obj in visible_objects])
        
        # 교집합 비율 계산
        intersection = len(mentioned_set & visible_set)
        union = len(mentioned_set | visible_set)
        
        if union == 0:
            return 0.8
        
        # Jaccard 유사도
        jaccard_score = intersection / len(mentioned_set)
        return float(jaccard_score)
    
    def _calculate_action_sync(self, analysis_data: Dict, ocr_text: str) -> float:
        """액션 싱크 계산 (동작 일치도)
        
        OCR 없는 경우: 프레임 묘사와 action_elements만으로 판단
        """
        action_keywords = ['움직', '동작', '행동', '진행', '변화', '이동', '클릭', '터치', '누르', '돌리']
        has_ocr = bool(ocr_text and ocr_text.strip())

        action_elements = analysis_data.get('action_elements', [])
        frame_descriptions = analysis_data.get('frame_descriptions', [])
        desc_text = ' '.join(frame_descriptions + action_elements)
        desc_has_action = any(keyword in desc_text for keyword in action_keywords + ['변화', '움직'])

        # OCR 없음: 이미지(프레임 묘사)만으로 동작 여부 판단
        if not has_ocr:
            if action_elements:
                return 0.8  # 동작 요소가 이미지에서 확인됨
            elif desc_has_action:
                return 0.7  # 프레임 묘사에서 동작 감지
            else:
                return 0.9  # 정적 콘텐츠 (이미지만 있는 경우 패널티 없음)

        # OCR 있음: OCR <-> 프레임 동작 일치도
        ocr_has_action = any(keyword in ocr_text for keyword in action_keywords)
        if ocr_has_action and desc_has_action:
            return 1.0  # 완전 일치
        elif not ocr_has_action and not desc_has_action:
            return 0.9  # 둘 다 정적 (양호)
        else:
            return 0.4  # 불일치
    
    def _mock_semantic_similarity(self, text: str) -> float:
        """Mock 의미 유사도 계산"""
        spam_words = ['무료', '증정', '선착순', '카톡', '텔레그램', '수익률', '투자', '돈', '충격', '대박']
        text_lower = text.lower()
        
        spam_count = sum(1 for word in spam_words if word in text_lower)
        return min(0.95, spam_count * 0.15)
    
    def _mock_semantic_alignment(self, descriptions: List[str], ocr_text: str) -> float:
        """Mock 의미 일치도 계산"""
        if not descriptions or not ocr_text:
            return 0.5
        
        desc_words = set(' '.join(descriptions).lower().split())
        ocr_words = set(ocr_text.lower().split())
        
        if not ocr_words:
            return 0.5
        
        # 단어 겹침 비율
        overlap = len(desc_words & ocr_words)
        total = len(ocr_words)
        
        return min(0.95, (overlap / total) * 1.2)


# 테스트 함수
def test_precise_calculator():
    """정밀 점수 계산기 테스트"""
    print("🧪 기획서 기반 정밀 점수 계산기 테스트")
    print("=" * 60)
    
    calculator = PreciseScoreCalculator()
    
    # 샘플 데이터 (어그로 영상)
    sample_analysis = {
        "visual_elements": ["자극적인 썸네일", "강렬한 텍스트", "화살표"],
        "frame_descriptions": [
            "첫 번째 프레임: 충격적인 제목과 함께 돈 이미지",
            "두 번째 프레임: 투자 차트가 급상승하는 모습",
            "세 번째 프레임: 카카오톡 메시지 화면"
        ],
        "layout_consistency": "높음",
        "layout_analysis": "모든 프레임에서 동일한 폰트와 색상을 사용하며 템플릿 기반으로 보임",
        "spam_indicators": ["무료 증정", "선착순 100명", "카톡 문의"],
        "content_object_matching": {
            "mentioned_in_text": ["투자", "수익률", "돈", "차트"],
            "visible_in_frames": ["차트", "그래프", "카카오톡"],
            "matching_analysis": "제목에서 언급한 투자와 차트가 실제로 화면에 보임"
        },
        "action_elements": ["차트 움직임", "메시지 전송"],
        "quality_issues": [],
        "overall_analysis": "전형적인 투자 유혹 어그로 콘텐츠로 보임"
    }
    
    sample_metadata = {
        "title": "🔥무료 증정🔥 투자 수익률 200% 보장! 선착순 100명만",
        "description": "카톡으로 문의하세요. 텔레그램 방에서 실시간 정보 공유"
    }
    
    scores = calculator.calculate_all_scores(
        analysis_data=sample_analysis,
        ocr_text="투자하면 수익률 200% 보장합니다. 카톡 문의 바랍니다.",
        frames=["frame1", "frame2", "frame3"],
        metadata=sample_metadata
    )
    
    print("\n📊 계산 결과:")
    print(f"  최종 분류: {scores['final_category']}")
    print(f"  C1 점수: {scores['c1_spam_score']:.3f}")
    print(f"  C2 점수: {scores['c2_pattern_score']:.3f}")
    print(f"  C3 점수: {scores['c3_context_score']:.3f}")
    print(f"  CIS 점수: {scores['cis_final']:.3f}")
    
    if 'detailed_results' in scores:
        print(f"\n📝 상세 근거:")
        print(f"  C1: {scores['detailed_results']['c1_details']['reasoning']}")
        print(f"  C2: {scores['detailed_results']['c2_details']['reasoning']}")
        print(f"  C3: {scores['detailed_results']['c3_details']['reasoning']}")
        print(f"  CIS: {scores['detailed_results']['cis_details']['reasoning']}")

if __name__ == "__main__":
    test_precise_calculator()