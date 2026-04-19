"""
Serialization Benchmark: Pickle vs numpy.tobytes vs PyArrow
============================================================
Compares read/write throughput and memory overhead for three serialization
strategies used to store EEG patches in LMDB.

EEG Patch Shape: (channels=32, samples=200)  → 6,400 float32 values = 25.6 KB per patch
Test corpus: 10,000 patches (~256 MB total)

Run:
    uv run python benchmarks/bench_serialization.py
"""

import io
import os
import pickle
import struct
import tempfile
import time

import lmdb
import numpy as np

# ── optional: pyarrow ──────────────────────────────────────────────────────────
try:
    import pyarrow as pa

    HAS_ARROW = True
except ImportError:
    HAS_ARROW = False
    print("[WARN] pyarrow not installed. Install with: uv pip install pyarrow")

# ── Config ─────────────────────────────────────────────────────────────────────
N_PATCHES = 10_000
PATCH_SHAPE = (32, 200)  # (channels, samples)
BATCH_SIZE = 1_000  # read all N_PATCHES in batches of this size
LMDB_MAP_SIZE = 2 * 1024**3  # 2 GB map


def make_patches(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    return [rng.uniform(-80, 80, size=PATCH_SHAPE).astype(np.float32) for _ in range(n)]


# ── Serializers ────────────────────────────────────────────────────────────────


def ser_pickle(arr: np.ndarray) -> bytes:
    return pickle.dumps(arr, protocol=4)


def deser_pickle(buf: bytes) -> np.ndarray:
    return pickle.loads(buf)


def ser_numpy(arr: np.ndarray) -> bytes:
    # Header: 2 uint32 dims + raw bytes
    header = struct.pack("II", *arr.shape)
    return header + arr.tobytes()


def deser_numpy(buf: bytes) -> np.ndarray:
    shape = struct.unpack("II", buf[:8])
    return np.frombuffer(buf[8:], dtype=np.float32).reshape(shape)


def ser_arrow(arr: np.ndarray) -> bytes:
    tensor = pa.Tensor.from_numpy(arr)
    sink = pa.BufferOutputStream()
    pa.ipc.write_tensor(tensor, sink)
    return sink.getvalue().to_pybytes()


def deser_arrow(buf: bytes) -> np.ndarray:
    reader = pa.BufferReader(buf)
    tensor = pa.ipc.read_tensor(reader)
    return tensor.to_pydict() if False else tensor.to_numpy()  # zero-copy view


# ── Benchmark harness ──────────────────────────────────────────────────────────


def bench(name: str, patches: list[np.ndarray], serialize_fn, deserialize_fn) -> dict:
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "bench.lmdb")
        env = lmdb.open(db_path, map_size=LMDB_MAP_SIZE)

        # ── WRITE ──
        t0 = time.perf_counter()
        txn = env.begin(write=True)
        for i, patch in enumerate(patches):
            txn.put(f"k{i}".encode(), serialize_fn(patch))
            if i % BATCH_SIZE == BATCH_SIZE - 1:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
        write_s = time.perf_counter() - t0

        # On-disk size
        db_size_mb = os.path.getsize(os.path.join(db_path, "data.mdb")) / 1024**2

        # ── READ ──
        t0 = time.perf_counter()
        with env.begin(write=False) as txn:
            for i in range(N_PATCHES):
                buf = txn.get(f"k{i}".encode())
                arr = deserialize_fn(buf)
        read_s = time.perf_counter() - t0

        env.close()

    result = {
        "name": name,
        "write_s": write_s,
        "read_s": read_s,
        "write_throughput_MB_s": (N_PATCHES * 25.6e-3) / write_s,
        "read_throughput_MB_s": (N_PATCHES * 25.6e-3) / read_s,
        "db_size_mb": db_size_mb,
    }
    return result


def print_results(results: list[dict]) -> None:
    header = f"{'Method':<16} {'Write MB/s':>12} {'Read MB/s':>12} {'DB Size MB':>12}"
    print("\n" + "=" * 56)
    print(header)
    print("-" * 56)
    baseline_read = results[0]["read_throughput_MB_s"]
    for r in results:
        speedup = r["read_throughput_MB_s"] / baseline_read
        print(
            f"{r['name']:<16} "
            f"{r['write_throughput_MB_s']:>12.1f} "
            f"{r['read_throughput_MB_s']:>12.1f} "
            f"{r['db_size_mb']:>12.1f}"
            f"   (read {speedup:.2f}x vs pickle)"
        )
    print("=" * 56 + "\n")


if __name__ == "__main__":
    print(f"Generating {N_PATCHES:,} EEG patches {PATCH_SHAPE} float32 ...")
    patches = make_patches(N_PATCHES)

    results = []
    results.append(bench("pickle", patches, ser_pickle, deser_pickle))
    print(
        f"  [pickle]  write={results[-1]['write_s']:.2f}s  read={results[-1]['read_s']:.2f}s"
    )

    results.append(bench("numpy.tobytes", patches, ser_numpy, deser_numpy))
    print(
        f"  [numpy]   write={results[-1]['write_s']:.2f}s  read={results[-1]['read_s']:.2f}s"
    )

    if HAS_ARROW:
        results.append(bench("pyarrow", patches, ser_arrow, deser_arrow))
        print(
            f"  [arrow]   write={results[-1]['write_s']:.2f}s  read={results[-1]['read_s']:.2f}s"
        )
    else:
        print("  [arrow]   SKIPPED (install pyarrow)")

    print_results(results)
