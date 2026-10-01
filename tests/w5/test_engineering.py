"""Week 5 autograder: packaging, CLI, typing, linting, CI, property tests, performance."""
import ast
import importlib.metadata
import json
import shutil
import subprocess
import sys
import sysconfig
import tomllib
from pathlib import Path

import pytest

import forge
from tests.helpers import need_data

ROOT = Path(__file__).resolve().parents[2]


def run(*cmd):
    return subprocess.run([sys.executable, "-m", *cmd], cwd=ROOT, capture_output=True, text=True)


# ------------------------------------------------------------------------------- packaging
def test_pyproject_declares_console_script():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert cfg["project"].get("scripts", {}).get("forge") == "forge.cli:main"


def test_installed_editable():
    try:
        v = importlib.metadata.version("forge-ml")
    except importlib.metadata.PackageNotFoundError:
        pytest.fail('not installed -- run: pip install -e ".[dev]"')
    assert v == forge.__version__
    exe = shutil.which("forge") or shutil.which("forge", path=sysconfig.get_path("scripts"))
    assert exe, "the `forge` console script is not on PATH"


# ------------------------------------------------------------------------------- CLI
def test_cli_version(capsys):
    from forge.cli import main
    assert main(["--version"]) == 0
    assert forge.__version__ in capsys.readouterr().out


def test_cli_usage_error_returns_2():
    from forge.cli import main
    assert main(["no-such-command"]) == 2


def test_cli_generate(tmp_path, capsys):
    import torch

    from forge.cli import main
    from forge.gpt.model import GPT, GPTConfig
    from forge.gpt.tokenizer import CharTokenizer
    from forge.gpt.train import save_checkpoint
    tok = CharTokenizer.from_text("abcdefgh ")
    torch.manual_seed(0)
    ck = str(tmp_path / "ck.pt")
    save_checkpoint(ck, GPT(GPTConfig(tok.vocab_size, 16, 1, 1, 8)).eval(), tok)
    assert main(["generate", "--ckpt", ck, "--prompt", "abc", "--max-new-tokens", "7", "--top-k", "1"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("abc") and len(out) == 10


def test_cli_train_mnist(capsys):
    need_data("mnist.npz")
    from forge.cli import main
    assert main(["train-mnist", "--epochs", "1", "--limit", "5000"]) == 0
    out = capsys.readouterr().out
    assert "test accuracy:" in out
    acc = float(out.split("test accuracy:")[1].split()[0])
    assert acc > 0.85


# ------------------------------------------------------------------------------- static analysis
def test_mypy_strict_core():
    r = run("mypy", "--strict", "forge/autograd.py", "forge/optim.py", "forge/nn")
    assert r.returncode == 0, r.stdout[-3000:]


def test_ruff_clean():
    r = run("ruff", "check", "forge", "pyeng")
    assert r.returncode == 0, r.stdout[-3000:]


def test_pyproject_has_tool_config():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert "mypy" in cfg.get("tool", {}) and "ruff" in cfg.get("tool", {}), "add [tool.mypy] and [tool.ruff]"


# ------------------------------------------------------------------------------- CI + tests you write
def test_ci_workflow():
    wf = ROOT / ".github" / "workflows" / "ci.yml"
    assert wf.exists(), "create .github/workflows/ci.yml"
    text = wf.read_text()
    for needed in ["pytest", "ruff", "mypy", "pip install"]:
        assert needed in text, f"CI should run {needed}"


def test_you_wrote_property_tests():
    tree = ast.parse((ROOT / "tests" / "w5" / "test_properties.py").read_text())
    given = [f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
             and any("given" in ast.unparse(d) for d in f.decorator_list)]
    assert len(given) >= 5, f"write at least 5 hypothesis properties (found {len(given)})"


# ------------------------------------------------------------------------------- performance
def test_conv_block_performance():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    from bench_conv import RESULTS, measure
    ms = measure()
    assert RESULTS.exists(), "record your baseline FIRST: python benchmarks/bench_conv.py --record baseline"
    data = json.loads(RESULTS.read_text())
    assert {"baseline_ms", "optimized_ms"} <= set(data)
    assert ms < 400, f"conv block fwd+bwd {ms:.0f} ms -- target < 400 ms (reference solution: ~150 ms)"
    writeup = ROOT / "docs" / "w5_perf.md"
    assert writeup.exists() and len(writeup.read_text().split()) > 150, "write docs/w5_perf.md (template: docs/templates/perf.md)"
