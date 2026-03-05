"""
데이터베이스 연결 및 세션 관리
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from contextlib import contextmanager
import logging

from database_models import Base
from config import MOCK_MODE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """데이터베이스 연결 관리자"""
    
    def __init__(self, mock_mode: bool = None):
        self.mock_mode = mock_mode if mock_mode is not None else MOCK_MODE
        
        if self.mock_mode:
            # SQLite 인메모리 DB 사용 (테스트용)
            self.database_url = "sqlite:///:memory:"
            self.engine = create_engine(
                self.database_url,
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
                echo=False
            )
        else:
            # 실제 PostgreSQL 사용
            self.database_url = self._get_database_url()
            self.engine = create_engine(self.database_url, echo=False)
        
        # 세션 팩토리 생성
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # 테이블 생성 (Mock 모드에서)
        if self.mock_mode:
            self.create_tables()
        
        logger.info(f"DatabaseManager 초기화 (Mock: {self.mock_mode})")
    
    def _get_database_url(self) -> str:
        """실제 데이터베이스 URL 구성"""
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "youtube_shorts_detector")
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD", "password")
        
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    
    def create_tables(self):
        """테이블 생성"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("데이터베이스 테이블 생성 완료")
    
    def drop_tables(self):
        """테이블 삭제 (주의: 개발용)"""
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("모든 테이블이 삭제되었습니다")
    
    @contextmanager
    def get_db_session(self):
        """데이터베이스 세션 컨텍스트 매니저"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"데이터베이스 오류: {e}")
            raise
        finally:
            session.close()
    
    def get_session(self) -> Session:
        """새 세션 반환 (직접 관리용)"""
        return self.SessionLocal()

# 전역 인스턴스
db_manager = DatabaseManager()

def get_db():
    """FastAPI 의존성 주입용"""
    with db_manager.get_db_session() as session:
        yield session