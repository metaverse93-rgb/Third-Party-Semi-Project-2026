"""
SQLAlchemy 데이터베이스 모델 정의
기획서 7항 데이터 아키텍처 구현
"""
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, Boolean, ForeignKey, JSON, Enum
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from enum import Enum as PyEnum

Base = declarative_base()

class ContentStatus(PyEnum):
    """콘텐츠 상태 열거형"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class ReviewStatus(PyEnum):
    """검토 상태 열거형"""
    PENDING = "pending"
    APPROVED = "approved" 
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"

class AnalysisResultStatus(PyEnum):
    """분석 결과 상태"""
    AUTO_APPROVE = "AUTO_APPROVE"
    HUMAN_REVIEW = "HUMAN_REVIEW"
    AUTO_REJECT = "AUTO_REJECT"
    ANALYSIS_FAILED = "ANALYSIS_FAILED"

class Contents(Base):
    """
    Contents 테이블 (기획서 7항)
    Phase 1 데이터 저장 및 인덱싱 최적화
    """
    __tablename__ = "contents"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(String(255), unique=True, nullable=False, index=True)
    url = Column(Text, nullable=False)
    title = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    channel_name = Column(String(255), nullable=True)
    duration = Column(Integer, nullable=True)  # 초
    view_count = Column(Integer, nullable=True)
    upload_date = Column(String(50), nullable=True)
    
    # Phase 1 데이터
    raw_ocr_text = Column(Text, nullable=True)
    layout_score = Column(Float, nullable=True)
    keyframes = Column(JSON, nullable=True)  # 키프레임 경로 리스트
    roi_data = Column(JSON, nullable=True)   # ROI 추출 데이터
    
    # 메타 정보
    status = Column(Enum(ContentStatus), default=ContentStatus.PENDING)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # 관계 설정
    analysis_results = relationship("AnalysisResults", back_populates="content", cascade="all, delete-orphan")
    validation_labels = relationship("ValidationLabels", back_populates="content", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Content(video_id={self.video_id}, title={self.title[:50]})>"

class AnalysisResults(Base):
    """
    Analysis_Results 테이블 (기획서 7항)
    Phase 2 추론 결과 및 상세 근거 아카이빙
    """
    __tablename__ = "analysis_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    result_id = Column(String(255), unique=True, nullable=False, index=True)
    video_id = Column(String(255), ForeignKey("contents.video_id"), nullable=False)
    
    # 분석 결과
    c_category = Column(String(10), nullable=False)  # C1~C5
    reasoning_log = Column(Text, nullable=False)
    confidence_score = Column(Float, nullable=False)
    status = Column(Enum(AnalysisResultStatus), nullable=False)
    
    # 모델 정보
    model_used = Column(String(50), nullable=False)  # GPT4o, Qwen 등
    model_version = Column(String(50), nullable=True)
    processing_time = Column(Float, nullable=True)
    
    # Context Score 상세
    context_score = Column(Float, nullable=True)
    s_semantic = Column(Float, nullable=True)
    o_existence = Column(Float, nullable=True) 
    a_sync = Column(Float, nullable=True)
    
    # 성능 지표
    performance_metrics = Column(JSON, nullable=True)  # CER, ROUGE-L, BERTScore 등
    raw_response = Column(Text, nullable=True)
    
    # A/B 테스트 정보
    ab_test_group = Column(String(10), nullable=True)  # A, B
    
    # 메타 정보
    created_at = Column(DateTime, default=func.now())
    is_used_for_training = Column(Boolean, default=False)  # 학습 데이터 사용 여부
    
    # 관계 설정
    content = relationship("Contents", back_populates="analysis_results")
    validation_label = relationship("ValidationLabels", uselist=False, back_populates="analysis_result")
    
    def __repr__(self):
        return f"<AnalysisResult(result_id={self.result_id}, category={self.c_category})>"

class ValidationLabels(Base):
    """
    Validation_Labels 테이블 (기획서 7항)
    지식 증류 및 HITL을 위한 학습 데이터셋 관리
    """
    __tablename__ = "validation_labels"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    label_id = Column(String(255), unique=True, nullable=False, index=True)
    video_id = Column(String(255), ForeignKey("contents.video_id"), nullable=False)
    result_id = Column(String(255), ForeignKey("analysis_results.result_id"), nullable=True)
    
    # 검증 데이터
    ground_truth_category = Column(String(10), nullable=True)  # 실제 정답
    human_reviewer_id = Column(String(100), nullable=True)
    review_status = Column(Enum(ReviewStatus), default=ReviewStatus.PENDING)
    review_comments = Column(Text, nullable=True)
    
    # 특수 플래그
    is_hard_example = Column(Boolean, default=False)
    is_consensus_required = Column(Boolean, default=False)  # 다중 검토자 필요
    difficulty_score = Column(Float, nullable=True)  # 난이도 점수
    
    # 학습 데이터 메타정보
    data_quality_score = Column(Float, nullable=True)
    is_synthetic = Column(Boolean, default=False)  # 합성 데이터 여부
    source_model = Column(String(50), nullable=True)  # 데이터 생성 모델
    
    # 메타 정보
    created_at = Column(DateTime, default=func.now())
    reviewed_at = Column(DateTime, nullable=True)
    last_updated_by = Column(String(100), nullable=True)
    
    # 관계 설정
    content = relationship("Contents", back_populates="validation_labels")
    analysis_result = relationship("AnalysisResults", back_populates="validation_label")
    
    def __repr__(self):
        return f"<ValidationLabel(label_id={self.label_id}, is_hard_example={self.is_hard_example})>"

class PerformanceLogs(Base):
    """
    Performance_Logs 테이블
    실시간 성능 추적 및 모니터링
    """
    __tablename__ = "performance_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    log_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # 연결 정보
    video_id = Column(String(255), nullable=False)
    result_id = Column(String(255), nullable=True)
    
    # 성능 지표
    cer_score = Column(Float, nullable=True)
    rouge_l_score = Column(Float, nullable=True)
    bert_score = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    context_score = Column(Float, nullable=True)
    
    # 시스템 성능
    processing_time = Column(Float, nullable=True)
    memory_usage = Column(Float, nullable=True)
    cpu_usage = Column(Float, nullable=True)
    
    # 모델 정보
    model_name = Column(String(50), nullable=False)
    model_version = Column(String(50), nullable=True)
    
    # 배치 정보
    batch_id = Column(String(255), nullable=True)
    experiment_id = Column(String(255), nullable=True)
    
    # 메타 정보
    timestamp = Column(DateTime, default=func.now())
    environment = Column(String(20), default="production")  # production, staging, dev
    
    def __repr__(self):
        return f"<PerformanceLog(log_id={self.log_id}, model={self.model_name})>"

class UserFeedback(Base):
    """
    User_Feedback 테이블
    사용자 피드백 수집 및 관리
    """
    __tablename__ = "user_feedback"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    feedback_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # 연결 정보
    video_id = Column(String(255), nullable=False)
    result_id = Column(String(255), nullable=True)
    
    # 피드백 내용
    user_action = Column(String(50), nullable=False)  # like, dislike, report, etc.
    feedback_type = Column(String(50), nullable=False)  # quality, accuracy, relevance
    feedback_text = Column(Text, nullable=True)
    rating = Column(Integer, nullable=True)  # 1-5 점수
    
    # 사용자 정보
    user_id = Column(String(255), nullable=True)  # 익명 가능
    session_id = Column(String(255), nullable=True)
    user_agent = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)  # IPv6 지원
    
    # 추가 메타데이터
    feedback_context = Column(JSON, nullable=True)  # 피드백 상황 정보
    is_verified = Column(Boolean, default=False)    # 검증된 피드백
    sentiment_score = Column(Float, nullable=True)   # 감정 점수
    
    # 메타 정보
    created_at = Column(DateTime, default=func.now())
    processed_at = Column(DateTime, nullable=True)
    is_processed = Column(Boolean, default=False)
    
    def __repr__(self):
        return f"<UserFeedback(feedback_id={self.feedback_id}, action={self.user_action})>"

class DistillationBatches(Base):
    """
    지식 증류 배치 관리
    """
    __tablename__ = "distillation_batches"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(255), unique=True, nullable=False, index=True)
    
    # 배치 정보
    source_model = Column(String(50), nullable=False)  # GPT4o
    target_model = Column(String(50), nullable=False)  # Qwen
    total_samples = Column(Integer, nullable=False)
    high_confidence_samples = Column(Integer, nullable=False)  # >= 0.9
    medium_confidence_samples = Column(Integer, nullable=False)  # 0.5-0.9
    
    # 품질 메트릭
    avg_confidence_score = Column(Float, nullable=True)
    data_quality_score = Column(Float, nullable=True)
    category_distribution = Column(JSON, nullable=True)
    
    # 배치 상태
    status = Column(String(20), default="preparing")  # preparing, ready, training, completed, failed
    file_path = Column(String(500), nullable=True)    # 생성된 데이터셋 경로
    
    # 메타 정보
    created_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime, nullable=True)
    created_by = Column(String(100), nullable=True)
    
    def __repr__(self):
        return f"<DistillationBatch(batch_id={self.batch_id}, samples={self.total_samples})>"