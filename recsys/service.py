"""Serve recommendations with the Week 6 metrics and the Week 7 rate limiter.

  GET /health                       -> {"status": "ok"}
  GET /recommend/{user_id}?k=10     -> {"user_id": int, "items": [raw movie ids], "fallback": bool}
                                       k must be 1..100 (else 422)
  GET /metrics                      -> Prometheus text incl. recommend_latency_seconds (histogram) and
                                       recommend_fallback_total (counter)
Rate limiting exactly as in serve.app (X-API-Key or client host; 429 + Retry-After).
"""
from __future__ import annotations

from typing import Any


def create_recsys_app(recommender: Any, rate_limiter: Any = None):
    raise NotImplementedError
