"""
OceanMaster v13.2 — Shared Security Module
=============================================
統一安全邏輯: rate limiter, NaN/Inf sanitizer, session token (HMAC).

所有 web/API surface 共用此模組，避免邏輯不一致。

[v15.3-refactor] 從 web_server.py + api/app.py 抽取
"""

import hashlib
import hmac
import math
import os
import secrets
import time as _time
from collections import OrderedDict
from typing import Any

# ═══════════════════════════════════════════════════
# Sliding-window Rate Limiter
# ═══════════════════════════════════════════════════
_RATE_LIMIT = int(os.environ.get("OCEANMASTER_RATE_LIMIT", "100"))
_RATE_WINDOW = 60  # seconds
_MAX_IPS = 10000   # prevent unbounded dict growth (memory DoS)
_request_log: OrderedDict = OrderedDict()  # ip -> [timestamps]


def rate_check(ip: str) -> bool:
    """
    Sliding-window rate limiter. Returns True if request is allowed.

    - 100 requests / 60s per IP (configurable via OCEANMASTER_RATE_LIMIT)
    - OrderedDict + _MAX_IPS cap to prevent memory DoS
    """
    now = _time.time()

    # Evict stale IPs when approaching capacity
    if len(_request_log) > _MAX_IPS:
        cutoff = now - _RATE_WINDOW * 2
        stale_keys = [k for k, v in _request_log.items() if not v or v[-1] < cutoff]
        for k in stale_keys[:max(1, len(stale_keys) // 2)]:
            _request_log.pop(k, None)
        # Hard evict oldest if still over limit
        while len(_request_log) > _MAX_IPS:
            _request_log.popitem(last=False)

    timestamps = _request_log.get(ip, [])
    # Prune entries outside current window
    timestamps = [t for t in timestamps if now - t < _RATE_WINDOW]
    if len(timestamps) >= _RATE_LIMIT:
        _request_log[ip] = timestamps
        return False
    timestamps.append(now)
    _request_log[ip] = timestamps
    return True


# ═══════════════════════════════════════════════════
# NaN/Inf Sanitizer — prevent JSON serialization crashes
# ═══════════════════════════════════════════════════

def sanitize_for_json(obj: Any) -> Any:
    """遞歸清除 dict/list 中的 NaN/Inf → None"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


# ═══════════════════════════════════════════════════
# HMAC Session Token — 不暴露 master API key 到 HTML
# ═══════════════════════════════════════════════════
_DEFAULT_SESSION_TTL = 86400  # 24 hours


def generate_session_token(
    secret: str,
    ttl: int = _DEFAULT_SESSION_TTL,
) -> str:
    """生成 HMAC session token (TTL-aware), 注入到 dashboard HTML 取代 master key"""
    ts = str(int(_time.time()))
    sig = hmac.new(secret.encode(), ts.encode(), hashlib.sha256).hexdigest()[:32]
    return f"ses_{ts}_{sig}"


def verify_session_token(
    token: str,
    secret: str,
    ttl: int = _DEFAULT_SESSION_TTL,
) -> bool:
    """驗證 session token 是否有效 (未過期 + 簽章正確)"""
    if not token or not token.startswith("ses_"):
        return False
    parts = token.split("_", 2)
    if len(parts) != 3:
        return False
    ts_str, sig = parts[1], parts[2]
    try:
        ts = int(ts_str)
    except ValueError:
        return False
    if _time.time() - ts > ttl:
        return False
    expected = hmac.new(secret.encode(), ts_str.encode(), hashlib.sha256).hexdigest()[:32]
    return hmac.compare_digest(sig, expected)
