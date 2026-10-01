"""FastAPI app. Run it for real with:

    uvicorn serve.app:app_from_env --factory --port 8000      (reads FORGE_CKPT, FORGE_DEVICE env vars)

Endpoints (the tests call exactly these):
  GET  /health            -> {"status": "ok"}
  POST /generate          body {"prompt": str (1..2000 chars), "max_new_tokens": int (1..512, default 64),
                                "temperature": float (>0, <=5, default 1.0), "top_k": int|null (>=1)}
                          -> {"text": <continuation>, "latency_ms": float}
                          Goes through a DynamicBatcher (engine.generate_batch runs per batch).
                          Invalid body -> 422 (let pydantic do it).
  POST /generate/stream   same body -> text/event-stream: one `data: {"token": "..."}\\n\\n` per piece,
                          then `data: [DONE]\\n\\n`. (Not batched.)
  GET  /metrics           -> Prometheus text (content-type text/plain) with at least:
                          http_requests_total (counter), request_latency_seconds (histogram),
                          batch_size (histogram, one observation per processed batch).
Week 7: if ``rate_limiter`` is given, every request except /health and /metrics first calls
  rate_limiter.allow(key) where key = the X-API-Key header, else the client host. Denied -> 429 with
  a Retry-After header.
The batcher must start and stop with the app (FastAPI lifespan).
"""
from __future__ import annotations

from typing import Any

from serve.engine import Engine


def create_app(engine: Engine, max_batch_size: int = 8, max_wait_ms: float = 5.0, rate_limiter: Any = None):
    raise NotImplementedError


def app_from_env():
    """Factory for uvicorn: GPTEngine(os.environ["FORGE_CKPT"], os.environ.get("FORGE_DEVICE", "cpu"))."""
    raise NotImplementedError
