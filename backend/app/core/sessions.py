"""Sign an identity into a short-lived token, and read one back.

HMAC-SHA256 from the standard library, for the same reasons `passwords.py` uses
scrypt from it: no new dependency, and the token is self-describing, so the
scheme can change without a new version silently accepting old tokens.

The token carries its own expiry and is verified by signature, so there is no
session table to keep or clean up. The cost is that a single token cannot be
revoked before it expires — hence the short lifetime. Changing SESSION_SECRET
invalidates every token at once, which is the blunt instrument that replaces
per-session revocation.
"""
import hmac
from base64 import urlsafe_b64decode, urlsafe_b64encode
from hashlib import sha256
from time import time

from app.core.config import session_secret

# Bumped if the payload layout changes, so an old token fails the scheme check
# rather than being misread by new code.
SCHEME = "v1"

# One working day: long enough that an officer is not signed out mid-review,
# short enough to bound a leaked token, given it cannot be revoked individually.
LIFETIME_SECONDS = 8 * 60 * 60


class InvalidSession(ValueError):
    """The token is malformed, was tampered with, or has expired."""


def issue(email: str, lifetime_seconds: int = LIFETIME_SECONDS) -> tuple[str, int]:
    """Mint a token for this identity, with the epoch second it expires at."""
    expires_at = int(time()) + lifetime_seconds
    payload = f"{email}:{expires_at}"
    return f"{SCHEME}.{_encode(payload.encode())}.{_encode(_sign(payload))}", expires_at


def identity_from(token: str) -> str:
    """The email a valid token carries. Raises InvalidSession for anything else."""
    try:
        scheme, payload_part, signature_part = token.split(".")
        payload = _decode(payload_part).decode()
        signature = _decode(signature_part)
    except (AttributeError, ValueError, UnicodeDecodeError) as exception:
        raise InvalidSession("Malformed session token") from exception

    if scheme != SCHEME:
        raise InvalidSession(f"Unknown session token scheme {scheme!r}")

    # Checked before anything in the payload is believed, so a caller cannot
    # choose their own identity or push their own expiry out.
    if not hmac.compare_digest(signature, _sign(payload)):
        raise InvalidSession("Session token signature does not match")

    # rpartition, not split: the expiry is the last field, and an email is not
    # guaranteed to be free of the separator.
    email, _, expires_at = payload.rpartition(":")
    if int(expires_at) <= time():
        raise InvalidSession("Session has expired. Sign in again.")
    return email


def _sign(payload: str) -> bytes:
    return hmac.new(session_secret().encode(), payload.encode(), sha256).digest()


def _encode(raw: bytes) -> str:
    # Padding stripped so a token carries no '=' and stays safe in a header.
    return urlsafe_b64encode(raw).decode().rstrip("=")


def _decode(text: str) -> bytes:
    return urlsafe_b64decode(text + "=" * (-len(text) % 4))
