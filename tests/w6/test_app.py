"""Week 6 autograder: FastAPI app (with a fake engine) + GPTEngine batching correctness."""
import json
import re
import threading
import time

import pytest
from fastapi.testclient import TestClient

from serve.app import create_app
from serve.engine import GenerateRequest


class FakeEngine:
    def __init__(self, delay=0.0):
        self.batches = []
        self.delay = delay

    def generate_batch(self, reqs):
        self.batches.append(len(reqs))
        time.sleep(self.delay)
        return [r.prompt[::-1] * 2 for r in reqs]

    def stream(self, req):
        yield from req.prompt.upper()


def test_health_and_generate():
    eng = FakeEngine()
    with TestClient(create_app(eng)) as c:
        assert c.get("/health").json() == {"status": "ok"}
        r = c.post("/generate", json={"prompt": "abc"})
        assert r.status_code == 200
        body = r.json()
        assert body["text"] == "cbacba" and body["latency_ms"] >= 0


@pytest.mark.parametrize("bad", [{}, {"prompt": ""}, {"prompt": "x", "max_new_tokens": 0},
                                 {"prompt": "x", "max_new_tokens": 513}, {"prompt": "x", "temperature": 0},
                                 {"prompt": "x", "top_k": 0}, {"prompt": "x" * 2001}])
def test_validation(bad):
    with TestClient(create_app(FakeEngine())) as c:
        assert c.post("/generate", json=bad).status_code == 422


def test_requests_are_batched():
    eng = FakeEngine(delay=0.05)
    results = {}
    with TestClient(create_app(eng, max_batch_size=8, max_wait_ms=20)) as c:
        def call(i):
            results[i] = c.post("/generate", json={"prompt": f"p{i:02d}"}).json()["text"]
        ts = [threading.Thread(target=call, args=(i,)) for i in range(16)]
        [t.start() for t in ts]
        [t.join() for t in ts]
    assert results == {i: f"p{i:02d}"[::-1] * 2 for i in range(16)}
    assert max(eng.batches) > 1, f"concurrent requests were never batched: {eng.batches}"


def test_streaming_sse():
    with TestClient(create_app(FakeEngine())) as c:
        with c.stream("POST", "/generate/stream", json={"prompt": "hey"}) as r:
            assert r.headers["content-type"].startswith("text/event-stream")
            events = [line[6:] for line in r.iter_lines() if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    assert [json.loads(e)["token"] for e in events[:-1]] == ["H", "E", "Y"]


def test_metrics_endpoint():
    with TestClient(create_app(FakeEngine())) as c:
        for _ in range(3):
            c.post("/generate", json={"prompt": "abc"})
        text = c.get("/metrics").text
    assert re.search(r"^http_requests_total [3-9]", text, re.M)
    assert "# TYPE request_latency_seconds histogram" in text
    assert re.search(r"^batch_size_count [1-9]", text, re.M)


# ------------------------------------------------------------------------------ real engine, tiny model
@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    import torch

    from forge.gpt.model import GPT, GPTConfig
    from forge.gpt.tokenizer import CharTokenizer
    from forge.gpt.train import save_checkpoint
    tok = CharTokenizer.from_text("abcdefghijklmnopqrstuvwxyz ")
    torch.manual_seed(0)
    path = str(tmp_path_factory.mktemp("ck") / "ck.pt")
    save_checkpoint(path, GPT(GPTConfig(tok.vocab_size, 32, 2, 2, 32)).eval(), tok)
    return path


def test_gpt_engine_batch_equals_individual(tiny_ckpt):
    from serve.engine import GPTEngine
    eng = GPTEngine(tiny_ckpt)
    reqs = [GenerateRequest("hello", 12, 1.0, 1), GenerateRequest("abc", 12, 1.0, 1),
            GenerateRequest("world", 12, 1.0, 1), GenerateRequest("xy", 5, 1.0, 1), GenerateRequest("hello", 12, 1.0, 1)]
    batch = eng.generate_batch(reqs)
    single = [eng.generate_batch([r])[0] for r in reqs]
    assert batch == single
    assert [len(t) for t in batch] == [12, 12, 12, 5, 12], "continuations only, max_new_tokens chars (char model)"


def test_gpt_engine_stream_matches_batch(tiny_ckpt):
    from serve.engine import GPTEngine
    eng = GPTEngine(tiny_ckpt)
    req = GenerateRequest("abc", 40, 1.0, 1)  # 3 + 40 > block_size 32
    assert "".join(eng.stream(req)) == eng.generate_batch([req])[0]
