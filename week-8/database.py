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
    ForeignKey,
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


# --- ORM model: User ---
# User accounts for authentication and ownership of analyses/documents.
#   id: Primary key.
#   email: Unique login identifier; indexed for lookups.
#   hashed_password: Bcrypt/argon2 hash; never store plain text.
#   full_name: Display name (optional).
#   is_active: Soft-disable account without deleting.
#   created_at: Account creation time (UTC).
#   last_login: Most recent login (UTC); null until first login.
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)


# --- ORM model: Analysis ---
# Stores each AI market analysis along with pricing and token usage details.
class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # nullable for existing rows
    ticker = Column(
        String(80), nullable=False, index=True
    )  # 80 allows "asset1 vs asset2" for comparisons
    asset_type = Column(
        String(10), nullable=False
    )  # 'stock', 'crypto', or 'comparison'
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
    tags = Column(String(200))  # comma-separated tags e.g. 'bullish,high-rsi,tech'

# --- ORM model: WatchlistItem ---
# Tracks assets the user has added to their personal watchlist.
class WatchlistItem(Base):
    __tablename__ = "watchlist"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(20), nullable=False, unique=True)
    asset_type = Column(String(10), nullable=False)
    added_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


# --- ORM model: Document ---
# Tracks each indexed document used for RAG (many documents per collection_group).
class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    filename = Column(String(255), nullable=False)
    collection_name = Column(String(120), nullable=False)
    collection_group = Column(String(100), default="default", index=True)
    chunks_indexed = Column(Integer, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)


class CollectionGroup(Base):
    __tablename__ = "collection_groups"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    document_count = Column(Integer, default=0)


# --- ORM model: QAHistory ---
# Stores question–answer interactions for RAG, grouped by collection.
class QAHistory(Base):
    __tablename__ = "qa_history"

    id = Column(Integer, primary_key=True, index=True)
    collection_name = Column(String(120), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    found_relevant_context = Column(Boolean, default=True)
    estimated_cost = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


# --- Schema management helpers ---
# Utilities to create tables and provide a scoped database session.
def create_tables() -> None:
    """Create all database tables defined on the Base metadata."""
    # Week 6 schema update — drops Week 5 test data
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    # Simple, best-effort migration for new Analysis columns on existing databases.
    try:
        inspector = inspect(engine)
        columns = {col["name"] for col in inspector.get_columns("analyses")}
        with engine.begin() as conn:
            if "pe_ratio" not in columns:
                conn.execute(text("ALTER TABLE analyses ADD COLUMN pe_ratio FLOAT"))
            if "week_high_52" not in columns:
                conn.execute(
                    text("ALTER TABLE analyses ADD COLUMN week_high_52 FLOAT")
                )
            if "week_low_52" not in columns:
                conn.execute(
                    text("ALTER TABLE analyses ADD COLUMN week_low_52 FLOAT")
                )
            # Widen ticker for comparison labels (PostgreSQL only; SQLite is typeless).
            if engine.dialect.name == "postgresql":
                conn.execute(
                    text("ALTER TABLE analyses ALTER COLUMN ticker TYPE VARCHAR(80)")
                )
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

