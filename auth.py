# backend/auth.py
"""Authentication helpers: password hashing and privileged access codes.

Passwords are stored as bcrypt hashes. Older accounts were stored as unsalted
SHA-256 hex digests; verify_password() still accepts those so existing users can
log in, and needs_rehash() flags them so the caller can transparently upgrade
the stored hash to bcrypt on the next successful login.

Privileged codes (admin signup, clinician access) come from the environment, not
source — see .env. They are intentionally empty by default so a missing/blank
code never grants elevated access.
"""
import os
import re
import time
import hashlib
import bcrypt
import jwt
from dotenv import load_dotenv

load_dotenv()

# Codes that grant elevated roles. Blank by default => access denied.
ADMIN_SIGNUP_CODE = os.getenv("ADMIN_SIGNUP_CODE", "")
CLINICIAN_KEY = os.getenv("CLINICIAN_SPECIAL_CODE", "")

# Secret used to sign session tokens. MUST be set in the environment; blank => no
# tokens can be issued or verified (fail closed).
SESSION_SECRET = os.getenv("SESSION_SECRET", "")
TOKEN_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(60 * 60 * 12)))  # 12h
_JWT_ALG = "HS256"


def create_access_token(email: str, roles) -> str:
    """Mint a signed session token carrying the user's email and roles."""
    if not SESSION_SECRET:
        raise RuntimeError("SESSION_SECRET is not configured; cannot issue tokens.")
    now = int(time.time())
    payload = {
        "sub": email,
        "roles": list(roles or []),
        "iat": now,
        "exp": now + TOKEN_TTL_SECONDS,
    }
    return jwt.encode(payload, SESSION_SECRET, algorithm=_JWT_ALG)


def decode_access_token(token: str):
    """Return the token payload if valid and unexpired, else None."""
    if not SESSION_SECRET or not token:
        return None
    try:
        return jwt.decode(token, SESSION_SECRET, algorithms=[_JWT_ALG])
    except jwt.PyJWTError:
        return None

# A bcrypt-free legacy hash: 64 hex chars (unsalted SHA-256).
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$", re.IGNORECASE)


def _pw_bytes(plain: str) -> bytes:
    """Encode a password for bcrypt, which only considers the first 72 bytes."""
    return plain.encode("utf-8")[:72]


def hash_password(plain: str) -> str:
    """Return a salted bcrypt hash suitable for storage."""
    return bcrypt.hashpw(_pw_bytes(plain), bcrypt.gensalt()).decode("utf-8")


def _is_legacy_sha256(stored: str) -> bool:
    return bool(stored) and bool(_SHA256_RE.match(stored))


def verify_password(plain: str, stored: str) -> bool:
    """Check a password against either a bcrypt hash or a legacy SHA-256 digest."""
    if not stored:
        return False
    if _is_legacy_sha256(stored):
        return hashlib.sha256(plain.encode("utf-8")).hexdigest() == stored
    try:
        return bcrypt.checkpw(_pw_bytes(plain), stored.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def needs_rehash(stored: str) -> bool:
    """True when the stored hash should be upgraded to current bcrypt on login."""
    return _is_legacy_sha256(stored)


def verify_clinician(email: str, code: str):
    """Authorize a clinician based on the configured access code."""
    if CLINICIAN_KEY and code == CLINICIAN_KEY:
        return {"status": "authorized", "role": "psychiatrist"}
    return {"status": "unauthorized", "role": "user"}
