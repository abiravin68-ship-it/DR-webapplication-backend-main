import os
import base64
import binascii
import json
import threading
from typing import Optional, Union, Any, Dict

from Crypto.Cipher import AES
from Crypto.Protocol.KDF import PBKDF2
from Crypto.Hash import SHA256
from Crypto.Random import get_random_bytes

_KEY_LEN = 32
_NONCE_LEN = 12
_TAG_LEN = 16
_PBKDF2_ITERS = int(os.getenv("PBKDF2_ITERATIONS", "200000"))


class EncryptionError(Exception):
    pass


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def _parse_key_material(key_str: str) -> bytes:
    k = (key_str or "").strip()
    if not k:
        raise EncryptionError("ENCRYPTION_KEY is empty")

    if k.lower().startswith("b64:"):
        raw = _b64d(k[4:])
        if len(raw) != _KEY_LEN:
            raise EncryptionError("Base64 key must decode to 32 bytes for AES-256")
        return raw

    if k.lower().startswith("hex:"):
        raw = bytes.fromhex(k[4:])
        if len(raw) != _KEY_LEN:
            raise EncryptionError("Hex key must decode to 32 bytes for AES-256")
        return raw

    try:
        raw = _b64d(k)
        if len(raw) == _KEY_LEN:
            return raw
    except Exception:
        pass

    try:
        raw = bytes.fromhex(k)
        if len(raw) == _KEY_LEN:
            return raw
    except (ValueError, binascii.Error):
        pass

    return k.encode("utf-8")


def _derive_key_from_passphrase(passphrase: bytes, salt: bytes) -> bytes:
    return PBKDF2(
        passphrase,
        salt,
        dkLen=_KEY_LEN,
        count=_PBKDF2_ITERS,
        hmac_hash_module=SHA256,
    )


class AESEncryptor:
    def __init__(self, key_material: bytes, salt: Optional[bytes] = None, require_salt_for_passphrase: bool = False):
        self._key_material = key_material
        self._static_salt = salt
        self._require_salt_for_passphrase = require_salt_for_passphrase

    def _get_key_and_salt(self, salt_override: Optional[bytes]) -> (bytes, bytes):
        km = self._key_material
        if len(km) == _KEY_LEN:
            use_salt = salt_override or self._static_salt or b""
            return km, use_salt

        use_salt = salt_override or self._static_salt
        if use_salt is None:
            if self._require_salt_for_passphrase:
                raise EncryptionError("Passphrase mode requires ENCRYPTION_SALT_B64")
            use_salt = get_random_bytes(16)
        key = _derive_key_from_passphrase(km, use_salt)
        return key, use_salt

    def encrypt_bytes(self, plaintext: bytes, aad: Optional[bytes] = None) -> str:
        if not isinstance(plaintext, (bytes, bytearray)):
            raise EncryptionError("encrypt_bytes expects bytes")

        key, salt = self._get_key_and_salt(salt_override=None)
        nonce = get_random_bytes(_NONCE_LEN)
        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)

        if aad:
            cipher.update(aad)

        ciphertext, tag = cipher.encrypt_and_digest(bytes(plaintext))

        salt_out = salt if len(salt) == 16 else b""
        if len(salt_out) not in (0, 16):
            raise EncryptionError("Salt must be 16 bytes (or empty for raw-key mode)")

        blob = salt_out.ljust(16, b"\x00") + nonce + tag + ciphertext
        return _b64e(blob)

    def decrypt_bytes(self, token_b64: str, aad: Optional[bytes] = None) -> bytes:
        try:
            blob = _b64d(token_b64)
        except Exception as e:
            raise EncryptionError(f"Invalid base64 token: {e}")

        min_len = 16 + _NONCE_LEN + _TAG_LEN
        if len(blob) < min_len:
            raise EncryptionError("Token too short")

        salt_16 = blob[:16]
        nonce = blob[16:16 + _NONCE_LEN]
        tag = blob[16 + _NONCE_LEN:16 + _NONCE_LEN + _TAG_LEN]
        ciphertext = blob[16 + _NONCE_LEN + _TAG_LEN:]

        salt = None if salt_16 == b"\x00" * 16 else salt_16
        key, _ = self._get_key_and_salt(salt_override=salt)

        cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
        if aad:
            cipher.update(aad)

        try:
            return cipher.decrypt_and_verify(ciphertext, tag)
        except Exception as e:
            raise EncryptionError(f"Decryption failed: {e}")

    def encrypt_text(self, text: str, aad: Optional[bytes] = None) -> str:
        if not isinstance(text, str):
            raise EncryptionError("encrypt_text expects str")
        return self.encrypt_bytes(text.encode("utf-8"), aad=aad)

    def decrypt_text(self, token_b64: str, aad: Optional[bytes] = None) -> str:
        pt = self.decrypt_bytes(token_b64, aad=aad)
        try:
            return pt.decode("utf-8")
        except UnicodeDecodeError as e:
            raise EncryptionError(f"Decrypted bytes are not valid UTF-8: {e}")

    def encrypt_json(self, obj: Union[Dict[str, Any], list], aad: Optional[bytes] = None) -> str:
        s = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
        return self.encrypt_text(s, aad=aad)

    def decrypt_json(self, token_b64: str, aad: Optional[bytes] = None) -> Union[Dict[str, Any], list]:
        s = self.decrypt_text(token_b64, aad=aad)
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            raise EncryptionError(f"Decrypted text is not valid JSON: {e}")


_encryptor_singleton: Optional[AESEncryptor] = None
_encryptor_lock = threading.Lock()


def get_encryptor() -> Optional[AESEncryptor]:
    global _encryptor_singleton
    with _encryptor_lock:
        if _encryptor_singleton is not None:
            return _encryptor_singleton

        key_env = os.getenv("ENCRYPTION_KEY", "").strip()
        if not key_env:
            return None

        key_material = _parse_key_material(key_env)

        salt_b64 = os.getenv("ENCRYPTION_SALT_B64", "").strip()
        salt: Optional[bytes] = None
        if salt_b64:
            try:
                salt = _b64d(salt_b64)
            except Exception as e:
                raise EncryptionError(f"Invalid ENCRYPTION_SALT_B64: {e}")
            if len(salt) != 16:
                raise EncryptionError("ENCRYPTION_SALT_B64 must decode to 16 bytes")

        require_salt = (len(key_material) != _KEY_LEN)
        _encryptor_singleton = AESEncryptor(
            key_material=key_material,
            salt=salt,
            require_salt_for_passphrase=require_salt,
        )
        return _encryptor_singleton


def generate_key_b64() -> str:
    return "b64:" + _b64e(get_random_bytes(_KEY_LEN))


def generate_salt_b64() -> str:
    return _b64e(get_random_bytes(16))

class AESEncryption(AESEncryptor):
    pass
