import time
from collections import defaultdict
from typing import Dict, List

from fastapi import HTTPException

# In-memory store: {identifier: [timestamp1, timestamp2, ...]}
request_log: Dict[str, List[float]] = defaultdict(list)

RATE_LIMIT = 20        # max requests
WINDOW_SECONDS = 3600  # per hour


def check_rate_limit(identifier: str) -> None:
    """Check if identifier has exceeded the rate limit.

    Raises HTTPException 429 if limit exceeded.
    identifier: user_id for authenticated users, IP address for anonymous.
    """
    now = time.time()
    window_start = now - WINDOW_SECONDS

    # Remove timestamps outside the current window
    request_log[identifier] = [t for t in request_log[identifier] if t > window_start]

    if len(request_log[identifier]) >= RATE_LIMIT:
        oldest = request_log[identifier][0]
        reset_in = int(WINDOW_SECONDS - (now - oldest))
        message = (
            f"You have made {RATE_LIMIT} requests this hour. "
            f"Please wait {reset_in // 60} minutes before trying again."
        )
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "limit": RATE_LIMIT,
                "window": "1 hour",
                "reset_in_seconds": reset_in,
                "message": message,
            },
        )

    # Record this request
    request_log[identifier].append(now)


def get_rate_limit_status(identifier: str) -> dict:
    """Return current rate limit status for an identifier."""
    now = time.time()
    window_start = now - WINDOW_SECONDS
    recent = [t for t in request_log[identifier] if t > window_start]
    return {
        "requests_made": len(recent),
        "requests_remaining": max(0, RATE_LIMIT - len(recent)),
        "limit": RATE_LIMIT,
        "window": "1 hour",
    }

