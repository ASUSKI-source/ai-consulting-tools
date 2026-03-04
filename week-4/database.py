"""Database configuration and ORM models for the Week 4 analyzer app."""

# --- Imports ---
# Core SQLAlchemy, datetime, and environment helpers.
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    Text,
    DateTime,
    Boolean,
    inspect,
    text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os


# --- Database setup ---
# Configure the database engine, session factory, and declarative base.
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./analyzer.db")

# Railway PostgreSQL URLs use the deprecated 'postgres://' scheme; SQLAlchemy
# expects 'postgresql://', so normalize it here if needed.
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Use SQLite-specific connect_args only for SQLite URLs; for PostgreSQL and
# other engines, do not pass SQLite-only options.
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},  # needed for SQLite only
    )
else:
    engine = create_engine(DATABASE_URL)

print(f'Database: {"SQLite" if "sqlite" in DATABASE_URL else "PostgreSQL"}')

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- ORM model: Analysis ---
# Stores each AI market analysis along with pricing and token usage details.
class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(20), nullable=False, index=True)
    asset_type = Column(String(10), nullable=False)  # 'stock' or 'crypto'
    company_name = Column(String(100))
    current_price = Column(Float)
    rsi = Column(Float)
    sma_5 = Column(Float)
    sma_20 = Column(Float)
    momentum_5d = Column(Float)
    momentum_30d = Column(Float)
    market_cap = Column(Float)
    pe_ratio = Column(Float)
    week_high_52 = Column(Float)
    week_low_52 = Column(Float)
    ai_analysis = Column(Text, nullable=False)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    estimated_cost = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)  # for user-added notes


# --- ORM model: WatchlistItem ---
# Tracks assets the user has added to their personal watchlist.
class WatchlistItem(Base):
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(20), nullable=False, unique=True)
    asset_type = Column(String(10), nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


# --- Schema management helpers ---
# Utilities to create tables and provide a scoped database session.
def create_tables() -> None:
    """Create all database tables defined on the Base metadata."""
    Base.metadata.create_all(bind=engine)

    # Simple, best-effort migration for new Analysis columns on existing databases.
    try:
        inspector = inspect(engine)
        columns = {col["name"] for col in inspector.get_columns("analyses")}
        with engine.begin() as conn:
            if "pe_ratio" not in columns:
                conn.execute(text("ALTER TABLE analyses ADD COLUMN pe_ratio FLOAT"))
            if "week_high_52" not in columns:
                conn.execute(text("ALTER TABLE analyses ADD COLUMN week_high_52 FLOAT"))
            if "week_low_52" not in columns:
                conn.execute(text("ALTER TABLE analyses ADD COLUMN week_low_52 FLOAT"))
    except Exception:
        # In dev environments it's safe to ignore migration errors; the app will
        # still function, but some history fields may be missing for older rows.
        pass


def get_db():
    """Yield a database session and ensure it is closed after use."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    # When run directly, create tables and print a confirmation message.
    create_tables()
    print("Database tables created successfully.")

