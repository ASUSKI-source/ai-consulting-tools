"""Authentication module: password hashing, JWT creation/validation, and FastAPI dependencies.

Flow:
  1. User registers or logs in with email + plain password.
  2. Plain password is hashed (bcrypt) for storage, or checked against stored hash at login.
  3. On successful login, a JWT access token is created (payload includes 'sub' = user id, 'exp' = expiry).
  4. Client sends token in Authorization: Bearer <token> on protected requests.
  5. get_current_user (or get_optional_user) decodes the token, loads the User from DB, and enforces is_active.
"""

from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from database import get_db, User
import os


# --- Constants ---
SECRET_KEY = os.getenv(
    "SECRET_KEY",
    "change-this-in-production-use-a-long-random-string",
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7


# --- Setup ---
# Bcrypt for one-way password hashing; OAuth2 bearer token from Authorization header.
# auto_error=False so endpoints can use get_optional_user and receive None when no token is sent.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)


# --- Password hashing (registration & login) ---

def hash_password(password: str) -> str:
    """Hash a plain password using bcrypt. One-way — cannot be reversed."""
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """Check if a plain password matches a bcrypt hash."""
    return pwd_context.verify(plain, hashed)


# --- JWT creation and validation ---

def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Build a JWT access token. Encodes the given data dict plus an 'exp' claim (expiry time in UTC).
    Caller typically passes data={'sub': str(user.id)} so the token identifies the user."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    )
    to_encode["exp"] = expire
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and verify a JWT; raises HTTP 401 if the token is invalid or expired."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# --- FastAPI dependencies: resolve token -> User ---

def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Require a valid Bearer token and return the corresponding User.
    Used as a dependency on protected routes; returns 401 if missing/invalid token or user not found, 403 if disabled."""
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(token)
    user_id = payload.get("sub")  # 'sub' is JWT standard for subject
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled",
        )
    return user


def get_optional_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Resolve Bearer token to User when present; return None when no token or invalid/disabled user.
    Use on endpoints that work with or without auth (e.g. list analyses: all vs only mine)."""
    if token is None:
        return None
    try:
        return get_current_user(token, db)
    except HTTPException:
        return None
