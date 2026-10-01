"""Week 6 autograder: experiment tracking + ops deliverables."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_mlflow_tracking(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    from forge.gpt.train import TrainConfig, train
    data = tmp_path / "t.txt"
    data.write_text("abcabcabd " * 300, encoding="utf-8")
    uri = "sqlite:///" + (tmp_path / "mlflow.db").as_posix()  # file-store backend is deprecated in MLflow 3
    mlflow.set_tracking_uri(uri)
    mlflow.create_experiment("w6-test", artifact_location=(tmp_path / "artifacts").as_uri())
    cfg = TrainConfig(data_path=str(data), block_size=16, n_layer=1, n_head=1, n_embd=16, batch_size=8,
                      max_steps=20, eval_interval=10, eval_iters=2, device="cpu", out_dir=str(tmp_path / "ck"),
                      mlflow=True, mlflow_tracking_uri=uri, mlflow_experiment="w6-test")
    train(cfg)
    runs = mlflow.search_runs(experiment_names=["w6-test"], output_format="list")
    assert len(runs) == 1
    run = runs[0]
    assert run.data.params["max_steps"] == "20" and run.data.params["n_embd"] == "16"
    assert "val_loss" in run.data.metrics and "train_loss" in run.data.metrics
    hist = mlflow.MlflowClient().get_metric_history(run.info.run_id, "val_loss")
    assert len(hist) >= 2, "log val_loss at every eval, with step="
    arts = [a.path for a in mlflow.MlflowClient().list_artifacts(run.info.run_id)]
    assert any(a.endswith(".pt") for a in arts), "log the checkpoint as an artifact"


def test_dockerfile():
    df = ROOT / "Dockerfile"
    assert df.exists(), "write a Dockerfile for the server"
    text = df.read_text()
    assert text.count("FROM ") >= 2, "use a multi-stage build"
    assert "HEALTHCHECK" in text and "/health" in text
    assert "uvicorn" in text


def test_dvc_tracks_dataset():
    assert (ROOT / ".dvc").is_dir(), "run: dvc init"
    assert (ROOT / "data" / "shakespeare.txt.dvc").exists(), "run: dvc add data/shakespeare.txt"


def test_load_test_results():
    assert (ROOT / "loadtest" / "locustfile.py").exists()
    res = ROOT / "loadtest" / "results.json"
    assert res.exists(), "record your load test numbers in loadtest/results.json"
    r = json.loads(res.read_text())
    for k in ["rps_batch1", "rps_batched", "p50_ms_batched", "p99_ms_batched", "max_batch_size", "max_wait_ms"]:
        assert k in r, f"missing {k}"
    assert r["rps_batched"] >= 3 * r["rps_batch1"], "dynamic batching should buy >= 3x throughput on the GPU"
