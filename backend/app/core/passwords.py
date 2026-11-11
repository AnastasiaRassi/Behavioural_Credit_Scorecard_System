"""Password hashing. The only module that handles a plaintext password.

scrypt from the standard library rather than bcrypt or argon2: it is memory-hard,
needs no new dependency, and keeps the stored form self-describing, so the cost
can be raised later without invalidating existing hashes.

Hashing a password is not authentication. This exists so that `users.password_hash`
can never hold a readable credential when one is set.
"""
import hmac
from hashlib import scrypt
from secrets import token_bytes

SALT_BYTES = 16
DIGEST_BYTES = 32

# Interactive-login cost from RFC 7914. 128 * n * r is 16 MB of working memory,
# which is what makes a stolen hash expensive to attack in bulk.
COST = {"n": 2**14, "r": 8, "p": 1}

SCHEME = "scrypt"


def hash_password(password: str) -> str:
    """Hash a password into a storable `scheme$salt$digest` string."""
    if not password:
        raise ValueError("Refusing to hash an empty password")
    salt = token_bytes(SALT_BYTES)
    return f"{SCHEME}${salt.hex()}${_derive(password, salt).hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Check a password against a stored hash, without leaking timing."""
    try:
        scheme, salt, digest = stored.split("$")
        if scheme != SCHEME:
            return False
        expected = bytes.fromhex(digest)
        computed = _derive(password, bytes.fromhex(salt))
    except (AttributeError, ValueError):
        # A malformed stored value fails closed rather than raising into a caller
        # that would read an exception as "not my problem".
        return False
    return hmac.compare_digest(computed, expected)


def _derive(password: str, salt: bytes) -> bytes:
    return scrypt(
        password.encode("utf-8"), salt=salt, dklen=DIGEST_BYTES, **COST
    )
