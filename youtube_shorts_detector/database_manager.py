"""
데이터베이스 연결 및 세션 관리
- MOCK_MODE=True: SQLite 인메모리 (테스트용)
- MOCK_MODE=False: SQLite 파일 DB (개발용) → PostgreSQL 준비되면 교체
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

# SQLite 파일 DB 경로 (MOCK_MODE=False 일 때 사용)
SQLITE_DB_PATH = "/Users/jungsoomin/Third-Party-semi-Project-2026/youtube_shorts_detector.db"

class DatabaseManager:
    """데이터베이스 연결 관리자"""

    def __init__(self, mock_mode: bool = None):
        self.mock_mode = mock_mode if mock_mode is not None else MOCK_MODE

        if self.mock_mode:
            # ✅ SQLite 인메모리 DB (테스트용)
            self.database_url = "sqlite:///:memory:"
            self.engine = create_engine(
                self.database_url,
                poolclass=StaticPool,
                connect_args={"check_same_thread": False},
                echo=False
            )
            logger.info("🗃️ SQLite 인메모리 DB 사용 (Mock 모드)")
        else:
            # ✅ PostgreSQL 환경변수가 있으면 PostgreSQL, 없으면 SQLite 파일 DB
            pg_host = os.getenv("DB_HOST", "")

            if pg_host:
                # PostgreSQL 사용
                self.database_url = self._get_postgres_url()
                try:
                    self.engine = create_engine(self.database_url, echo=False)
                    logger.info(f"🐘 PostgreSQL DB 연결: {pg_host}")
                except Exception as e:
                    logger.warning(f"⚠️ PostgreSQL 연결 실패, SQLite로 대체: {e}")
                    self.database_url = f"sqlite:///{SQLITE_DB_PATH}"
                    self.engine = create_engine(
                        self.database_url,
                        connect_args={"check_same_thread": False},
                        echo=False
                    )
            else:
                # ✅ SQLite 파일 DB 사용 (PostgreSQL 미설정 시 기본값)
                self.database_url = f"sqlite:///{SQLITE_DB_PATH}"
                self.engine = create_engine(
                    self.database_url,
                    connect_args={"check_same_thread": False},
                    echo=False
                )
                logger.info(f"🗃️ SQLite 파일 DB 사용: {SQLITE_DB_PATH}")

        # 세션 팩토리 생성
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

        # 테이블 생성 (항상 실행)
        self.create_tables()

        logger.info(f"✅ DatabaseManager 초기화 완료 (Mock: {self.mock_mode})")

    def _get_postgres_url(self) -> str:
        """PostgreSQL 연결 URL 구성"""
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "youtube_shorts_detector")
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD", "password")
        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    def create_tables(self):
        """테이블 생성"""
        Base.metadata.create_all(bind=self.engine)
        logger.info("📋 데이터베이스 테이블 생성 완료")

    def drop_tables(self):
        """테이블 삭제 (주의: 개발용)"""
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("⚠️ 모든 테이블이 삭제되었습니다")

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
