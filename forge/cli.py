"""Week 5 -- the `forge` command-line tool (argparse, stdlib only).

    forge --version
    forge train-mnist --epochs 1 --limit 5000 [--cnn]      prints "test accuracy: 0.9xxx"
    forge train-gpt [--max-steps N] [--device cpu|cuda]     runs forge.gpt.train with overrides
    forge generate --ckpt PATH --prompt TEXT [--max-new-tokens 100] [--temperature 1.0]
                   [--top-k K] [--seed 0]                   prints prompt + continuation

main() returns an exit code (0 success, 2 for usage errors -- argparse's default) instead of calling
sys.exit itself, so it's testable. Register it in pyproject.toml as the `forge` console script.
"""
from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())
