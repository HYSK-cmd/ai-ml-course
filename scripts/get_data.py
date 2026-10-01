"""Download every dataset the course uses into ./data (gitignored).

    python scripts/get_data.py                 # everything (~30 MB)
    python scripts/get_data.py mnist movielens # just some

This is plumbing, not course material, so it's provided.
"""
import gzip
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
MNIST = "https://ossci-datasets.s3.amazonaws.com/mnist/"
SHAKESPEARE = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
MOVIELENS = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def fetch(url: str) -> bytes:
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "forge-course"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def mnist() -> None:
    out = DATA / "mnist.npz"
    if out.exists():
        return
    parts = {}
    for key, name, offset in [("X_train", "train-images-idx3-ubyte.gz", 16), ("y_train", "train-labels-idx1-ubyte.gz", 8),
                              ("X_test", "t10k-images-idx3-ubyte.gz", 16), ("y_test", "t10k-labels-idx1-ubyte.gz", 8)]:
        raw = np.frombuffer(gzip.decompress(fetch(MNIST + name)), dtype=np.uint8, offset=offset)
        parts[key] = raw.reshape(-1, 784) if key.startswith("X") else raw
    np.savez_compressed(out, **parts)  # X_*: uint8 (N, 784) in [0, 255]; y_*: uint8 labels


def shakespeare() -> None:
    out = DATA / "shakespeare.txt"
    if not out.exists():
        out.write_bytes(fetch(SHAKESPEARE))


def california() -> None:
    out = DATA / "california.npz"
    if out.exists():
        return
    from sklearn.datasets import fetch_california_housing  # dev extra
    print("  downloading California housing via scikit-learn")
    d = fetch_california_housing(data_home=str(DATA / "sklearn"))
    np.savez_compressed(out, X=d.data, y=d.target, feature_names=np.array(d.feature_names))


def movielens() -> None:
    if (DATA / "ml-1m" / "ratings.dat").exists():
        return
    zipfile.ZipFile(io.BytesIO(fetch(MOVIELENS))).extractall(DATA)  # -> data/ml-1m/{ratings,movies,users}.dat


ALL = {"mnist": mnist, "shakespeare": shakespeare, "california": california, "movielens": movielens}

if __name__ == "__main__":
    DATA.mkdir(exist_ok=True)
    for name in sys.argv[1:] or ALL:
        print(f"[{name}]")
        ALL[name]()
    print("done ->", DATA)
