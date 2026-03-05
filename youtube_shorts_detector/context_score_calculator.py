"""
Context Score 계산 엔진
기획서 6-1항 수식: Context Score = (S_semantic × 0.5) + (O_existence × 0.3) + (A_sync × 0.2)
"""
import numpy as np
import logging
import time
import random
from typing import Dict, List, Tuple, Any
from config import CONTEXT_WEIGHTS, MOCK_MODE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextScoreCalculator:
    """Context Score 계산 클래스"""
    
    def __init__(self, mock_mode: bool = None):
        """
        초기화
        Args:
            mock_mode: None이면 config.MOCK_MODE 사용
        """
        self.mock_mode = mock_mode if mock_mode is not None else MOCK_MODE
        logger.info(f"ContextScoreCalculator 초기화 (Mock: {self.mock_mode})")
    
    def calculate_context_score(self, 
                               frames: List[str], 
                               ocr_text: str, 
                               metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        통합 Context Score 계산
        
        Args:
            frames: 키프레임 이미지 리스트
            ocr_text: OCR 추출 텍스트
            metadata: 영상 메타데이터
            
        Returns:
            Dict: 각 하위 점수와 최종 Context Score
        """
        logger.info("🧮 Context Score 계산 시작")
        start_time = time.time()
        
        # 1. S_semantic: 의미적 유사도 계산
        s_semantic = self.calculate_semantic_similarity(frames, ocr_text, metadata)
        
        # 2. O_existence: 객체 존재 여부 계산
        o_existence = self.calculate_object_existence(frames, ocr_text)
        
        # 3. A_sync: 시공간 동기화 계산
        a_sync = self.calculate_sync_score(frames, ocr_text, metadata)
        
        # 4. 최종 Context Score 계산 (기획서 수식)
        context_score = (
            s_semantic * CONTEXT_WEIGHTS["semantic"] +
            o_existence * CONTEXT_WEIGHTS["existence"] + 
            a_sync * CONTEXT_WEIGHTS["sync"]
        )
        
        processing_time = time.time() - start_time
        
        result = {
            "s_semantic": round(s_semantic, 3),
            "o_existence": round(o_existence, 3),
            "a_sync": round(a_sync, 3),
            "context_score": round(context_score, 3),
            "processing_time": round(processing_time, 3),
            "weights_used": CONTEXT_WEIGHTS
        }
        
        logger.info(f"✅ Context Score 계산 완료: {context_score:.3f} ({processing_time:.2f}초)")
        return result
    
    def calculate_semantic_similarity(self, 
                                    frames: List[str], 
                                    ocr_text: str, 
                                    metadata: Dict) -> float:
        """
        S_semantic: 의미적 유사도 계산 (50% 가중치)
        실제 구현 시 CLIP/BLIP-2 기반 Cosine Similarity 측정
        """
        if self.mock_mode:
            return self._mock_semantic_similarity(frames, ocr_text, metadata)
        
        # 실제 CLIP 구현 예시 (placeholder)
        try:
            # import clip
            # model, preprocess = clip.load("ViT-B/32")
            
            # 실제 CLIP 기반 유사도 계산 로직
            # 1. 이미지 인코딩
            # 2. 텍스트 인코딩  
            # 3. 코사인 유사도 계산
            
            logger.info("🔍 CLIP 기반 의미적 유사도 계산 (실제 구현 대기)")
            return 0.75  # placeholder
            
        except ImportError:
            logger.warning("CLIP 모델 미설치, Mock 모드로 전환")
            return self._mock_semantic_similarity(frames, ocr_text, metadata)
    
    def _mock_semantic_similarity(self, frames: List[str], ocr_text: str, metadata: Dict) -> float:
        """Mock 의미적 유사도 계산"""
        logger.info("🎭 Mock 의미적 유사도 계산")
        
        # 텍스트 길이와 메타데이터 기반 가상 점수
        text_quality = min(len(ocr_text) / 100, 1.0)  # 텍스트 풍부도
        title_relevance = 0.8 if metadata.get("title") and len(metadata["title"]) > 10 else 0.4
        
        # 프레임 수 고려
        frame_coverage = min(len(frames) / 3, 1.0)  # 3개 이상이면 1.0
        
        # 가중 평균 + 랜덤 노이즈
        base_score = (text_quality * 0.4 + title_relevance * 0.4 + frame_coverage * 0.2)
        noise = random.uniform(-0.1, 0.1)
        
        return max(0.0, min(1.0, base_score + noise))
    
    def calculate_object_existence(self, frames: List[str], ocr_text: str) -> float:
        """
        O_existence: 객체 존재 여부 계산 (30% 가중치)
        실제 구현 시 Grounding DINO 기반 객체 탐지
        """
        if self.mock_mode:
            return self._mock_object_existence(frames, ocr_text)
        
        # 실제 Grounding 구현 예시
        try:
            # from groundingdino.util.inference import Model
            
            # 실제 객체 탐지 및 Grounding 로직
            # 1. OCR 텍스트에서 명사 추출
            # 2. 프레임에서 해당 객체 탐지
            # 3. Recall 점수 계산
            
            logger.info("🎯 Grounding 기반 객체 존재 여부 계산 (실제 구현 대기)")
            return 0.65  # placeholder
            
        except ImportError:
            logger.warning("Grounding 모델 미설치, Mock 모드로 전환")
            return self._mock_object_existence(frames, ocr_text)
    
    def _mock_object_existence(self, frames: List[str], ocr_text: str) -> float:
        """Mock 객체 존재 여부 계산"""
        logger.info("🎭 Mock 객체 존재 여부 계산")
        
        # 텍스트에서 명사성 키워드 추출 (간단한 휴리스틱)
        object_keywords = ['사람', '자동차', '음식', '건물', '동물', '제품', '화면', '버튼', 'click', 'button']
        
        detected_objects = sum(1 for keyword in object_keywords 
                             if keyword.lower() in ocr_text.lower())
        
        # 프레임 수와 탐지된 객체 수 기반 점수
        frame_score = min(len(frames) / 5, 1.0)  # 5개 이상이면 1.0
        object_score = min(detected_objects / 3, 1.0)  # 3개 이상이면 1.0
        
        return (frame_score * 0.6 + object_score * 0.4)
    
    def calculate_sync_score(self, frames: List[str], ocr_text: str, metadata: Dict) -> float:
        """
        A_sync: 시공간 동기화 계산 (20% 가중치)
        실제 구현 시 SyncNet 기반 영상-음성 동기화 측정
        """
        if self.mock_mode:
            return self._mock_sync_score(frames, ocr_text, metadata)
        
        # 실제 SyncNet 구현 예시
        try:
            # from syncnet import SyncNetModel
            
            # 실제 동기화 점수 계산 로직
            # 1. 영상 프레임에서 립싱크 탐지
            # 2. 오디오 스펙트로그램 분석
            # 3. 시간축 정렬 점수 계산
            
            logger.info("🎬 SyncNet 기반 시공간 동기화 계산 (실제 구현 대기)")
            return 0.80  # placeholder
            
        except ImportError:
            logger.warning("SyncNet 모델 미설치, Mock 모드로 전환")
            return self._mock_sync_score(frames, ocr_text, metadata)
    
    def _mock_sync_score(self, frames: List[str], ocr_text: str, metadata: Dict) -> float:
        """Mock 시공간 동기화 점수"""
        logger.info("🎭 Mock 시공간 동기화 계산")
        
        # 영상 길이와 텍스트 밀도 기반 휴리스틱
        duration = metadata.get("duration", 60)
        text_density = len(ocr_text) / max(duration, 1)  # 초당 텍스트 량
        
        # 적절한 텍스트 밀도 (1-3자/초)가 좋은 동기화
        if 1.0 <= text_density <= 3.0:
            base_score = 0.9
        elif 0.5 <= text_density < 1.0 or 3.0 < text_density <= 5.0:
            base_score = 0.7
        else:
            base_score = 0.5
        
        # 프레임 연속성 고려 (프레임이 많을수록 연속성 높음)
        continuity_bonus = min(len(frames) / 10, 0.2)  # 최대 0.2 보너스
        
        return min(1.0, base_score + continuity_bonus)
    
    def batch_calculate(self, batch_data: List[Dict]) -> List[Dict]:
        """배치 Context Score 계산"""
        logger.info(f"📊 배치 Context Score 계산: {len(batch_data)}개 샘플")
        
        results = []
        for i, data in enumerate(batch_data):
            logger.info(f"처리 중: {i+1}/{len(batch_data)}")
            
            result = self.calculate_context_score(
                frames=data.get("frames", []),
                ocr_text=data.get("ocr_text", ""),
                metadata=data.get("metadata", {})
            )
            
            result["sample_id"] = data.get("sample_id", f"sample_{i}")
            results.append(result)
        
        return results