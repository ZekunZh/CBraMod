# CBraMod Codebase Review — Data Pipeline, Preprocessing & Code Quality

> [!NOTE]
> This review covers all files in the original CBraMod repository. File links point to the local workspace copy.

---

## 1. Preprocessing Pipeline

The preprocessing scripts live in [preprocessing/](file:///home/zekun/projects/0-jobs/SigmaNova/CBraMod/preprocessing) — one file per dataset (12 scripts). There is **no shared preprocessing framework**; each script is a standalone, top-level script.

### ✅ Pros

| Aspect | Detail |
|---|---|
| **Correct signal processing** | Resampling to 200 Hz <sup>[1]</sup>, bandpass filtering (0.3–75 Hz) <sup>[2]</sup>, 60 Hz notch filter <sup>[3]</sup> are all applied consistently for the TUH-based datasets (TUEG, TUAB, TUEV). |
| **Amplitude gating** | The pretraining script ([preprocessing_tueg_for_pretraining.py:L84](file:///home/zekun/projects/0-jobs/SigmaNova/CBraMod/preprocessing/preprocessing_tueg_for_pretraining.py#L84)) rejects patches with `max(abs) >= 100`, a simple but effective artifact rejection strategy. |
| **LMDB for pretraining** | Using LMDB for the large-scale pretraining dataset (TUEG) is a sound choice — it provides memory-mapped, read-optimized storage. |
| **Multiprocessing for TUAB** | [preprocessing_tuab.py:L220](file:///home/zekun/projects/0-jobs/SigmaNova/CBraMod/preprocessing/preprocessing_tuab.py#L220) uses `multiprocessing.Pool(24)` for parallel EDF processing — one of very few scalability-aware decisions. |

### ❌ Cons

#### 1.1 — Hardcoded paths everywhere

Preprocessing scripts contain **hardcoded absolute paths** baked into the source code as top-level variables (e.g., `root = "/data/datasets/BigDownstream/TUAB/edf"`).

> [!CAUTION]
> It is ok-ish if the pre-processing script is only used once isolated. However, if we have many data sources and regularly update them, it's far better to use config files to store this information safely across environments.

#### 1.2 — No shared preprocessing abstraction

Each of the 12 scripts reimplements the same pattern:

1. List files → split into train/val/test
2. Load raw data (`.mat`, `.edf`, `.pkl`)
3. Resample + filter
4. Reshape into `(channels, patches, 200)`
5. Write to LMDB or pickle

There is **zero code reuse** across scripts. The TUAB and TUEV scripts both implement the identical bipolar re-referencing (16 TCP channels) independently, creating ~60 duplicated lines per script.

#### 1.3 — Inconsistent storage formats across datasets

| Dataset | Storage Format | Access Pattern |
|---|---|---|
| TUEG (pretraining) | LMDB | Keyed lookup |
| SHU, FACED, BCIC-IV-2a, SEEDV, etc. | LMDB | Keyed lookup |
| TUAB | Individual `.pkl` files | Filesystem walk |
| TUEV | Individual `.pkl` files | Filesystem walk |
| ISRUC | Individual `.npy` files | Filesystem walk |

**Inconsistent pre-processing leads to fragmented dataset loading during pre-training and finetuning.** Because of this fragmentation, every dataset needs its own `Dataset` class with a completely different I/O pattern.

#### 1.4 — Bad Coding Patterns (Bare exceptions & shell injection)

The codebase utilizes multiple unsafe patterns that should be fused and corrected:

1. **Bare `except:` clauses:** In `preprocessing_tuab.py:L127`, a bare `except:` catches *everything* (including `KeyboardInterrupt`, `MemoryError`) and silently logs to a flat text file without tracebacks, making debugging nearly impossible.
2. **`os.system("cp ...")`:** In `preprocessing_tuev.py:L230`, it uses `os.system` for file copies. This introduces shell injection vulnerabilities (filenames with spaces), is non-portable to Windows, ignores exit codes, and is dramatically slower than Python's native `shutil.copy2()`.

#### 1.7 — No data integrity validation

- No checksums or hash verification after writing to LMDB. *(Graceful solution: Compute an MD5 or SHA256 string hash of the NumPy array bytes before pickling, and store it in the LMDB as part of a metadata dictionary alongside the array. During `Dataset` loading, you can optionally re-hash the bytes and assert it matches to guarantee no disk corruption occurred).*
- No schema validation on loaded `.mat` / `.edf` files.
- No assertion on expected shapes, channel counts, or sampling rates.
- If a `.mat` file is corrupted or has a different structure, the script crashes with an opaque `KeyError`.

#### 1.8 — Non-reproducible splits

Train/val/test splits are based on `sorted(os.listdir(...))`, which is filesystem-order dependent. The root cause is Unicode normalization: Linux uses **NFC** (Canonical Decomposition, followed by Canonical Composition) while macOS uses **NFD** (Canonical Decomposition). This means a filename with an accented character like `é` is stored as a different byte sequence on each OS, causing `sorted()` to produce a different order silently.

---

## 2. Data Loading Pipeline (`datasets/`)

The dataset loaders live in [datasets/](file:///home/zekun/projects/0-jobs/SigmaNova/CBraMod/datasets) — one `CustomDataset` + one `LoadDataset` per downstream task (13 files + 1 pretraining).

### ✅ Pros

| Aspect | Detail |
|---|---|
| **Standard PyTorch Dataset** | All loaders correctly extend `torch.utils.data.Dataset` with `__len__` / `__getitem__`. |
| **LMDB for pretraining** | [PretrainingDataset](file:///home/zekun/projects/0-jobs/SigmaNova/CBraMod/datasets/pretraining_dataset.py) uses `lmdb` with `readonly=True, lock=False` — correct flags for read-heavy workloads. |
| **Custom collate functions** | Each dataset defines a `collate` function that handles the numpy→tensor conversion, keeping `__getitem__` lightweight. |

### ❌ Cons

#### 2.1 — Magic number normalization: `data / 100`

This appears across **every single dataset**:

```python
# shu_dataset.py:L31
return data/100, label

# tuev_dataset.py:L32
return data/100, label

# pretrain_trainer.py:L64
x = x.to(self.device)/100
```

> [!IMPORTANT]
> The division by `/ 100` acts as a hardcoded normalizer because the pre-training script explicitly drops any patches where `amp > 100` before creating the LMDB. Therefore, this acts as a pseudo min-max scaler capping data roughly to the range `[-1, 1]`. 
> However, using a magic number without documentation is harmful for training on different datasets, and exceptionally dangerous for **inference**. During real-world inference, if the model encounters an outlier artifact `> 100` $\mu V$, there is no defined behavior specified in the model (should it be rejected? should it be clipped to 100?). A standard **z-score normalization** (subtracting the mathematical `mean` and dividing by the `standard deviation` of the dataset, effectively centering data at 0 with a variance of 1) or an explicit computationally guarded `torch.clamp` is fundamentally safer and more robust.

#### 2.2 — Every `CustomDataset` class is named `CustomDataset`

All 13 downstream dataset files define a class called `CustomDataset`. This causes:
- **Name collisions** if you attempt to import more than one.
- Impossible to distinguish datasets in stack traces, profilers, or debuggers.

#### 2.3 — LMDB handles are never closed

```python
# pretraining_dataset.py:L15
self.db = lmdb.open(dataset_dir, readonly=True, lock=False, readahead=True, meminit=False)
```

The `db` handle is opened in `__init__` but there is **no `__del__`, no context manager, and no explicit `close()`**. The only close call is in `pretrain_main.py:L65` (`pretrained_dataset.db.close()`) — but this is fragile and will leak if the program exits early.

#### 2.4 — `readahead=True` on LMDB for random access

```python
self.db = lmdb.open(dataset_dir, readonly=True, lock=False, readahead=True, meminit=False)
```

When using `DataLoader` with `shuffle=True`, access is **random**, not sequential. `readahead=True` triggers OS read-ahead on the backing file, which will **waste I/O bandwidth** and pollute the page cache with data that won't be used soon. The correct setting for random access is `readahead=False`.

#### 2.5 — Individual pickle `open()` per sample (TUEV)

```python
# tuev_dataset.py:L27
data_dict = pickle.load(open(os.path.join(self.data_dir, file), "rb"))
```

> [!WARNING]
> This opens a new file descriptor **for every sample, on every access**, and **never closes it** (no `with` statement). With a DataLoader using `num_workers > 0`, this will leak file descriptors and eventually crash with `OSError: [Errno 24] Too many open files`.

#### 2.6 — Validation/test DataLoaders have `shuffle=True` (SHU)

```python
# shu_dataset.py:L62
'val': DataLoader(..., shuffle=True),
# shu_dataset.py:L67
'test': DataLoader(..., shuffle=True),
```

Shuffling validation and test sets has **no benefit** and makes results non-deterministic. Several other datasets (TUEV, TUAB) correctly set `shuffle=False` for val/test, showing this is an oversight.

#### 2.7 — No `num_workers` in downstream DataLoaders

None of the downstream `LoadDataset.get_data_loader()` methods set `num_workers`. This defaults to `0` (main process), meaning **all data loading is single-threaded and blocking** during training. Only the pretraining loader in `pretrain_main.py` sets `num_workers=8`.

#### 2.8 — `collate` does unnecessary numpy round-trip

```python
def collate(self, batch):
    x_data = np.array([x[0] for x in batch])
    y_label = np.array([x[1] for x in batch])
    return to_tensor(x_data), to_tensor(y_label)
```

This converts a list of numpy arrays → one big numpy array → `torch.from_numpy()`. PyTorch's default collate function (`default_collate`) already handles this efficiently using `torch.stack()`. The manual collate adds complexity without benefit.

#### 2.9 — ISRUC relies on `os.listdir` ordering for seq/label pairing

```python
# isruc_dataset.py:L88
for seq_fname, label_fname in zip(seq_fnames, label_fnames):
```

`os.listdir()` returns files in **arbitrary order**. If the sequence files and label files don't happen to sort in the same order, samples will be **paired with wrong labels** — a silent correctness bug.

---

## 3. Training Code

### ✅ Pros

| Aspect | Detail |
|---|---|
| **Multiple LR scheduler options** | The pretraining trainer supports 5 different schedulers. |
| **Gradient clipping** | Both trainers apply `clip_grad_norm_`, important for transformer training stability. |
| **Best-model checkpointing** | Finetuning saves only the best model based on validation metrics via `copy.deepcopy`. |
| **Comprehensive metrics** | The evaluator correctly uses balanced accuracy, Cohen's kappa, weighted F1, ROC-AUC, PR-AUC, R², RMSE, and Pearson correlation. |

### ❌ Cons

#### 3.1 — No logging framework

All output goes through `print()`. No structured logging, no log levels, no file output. Impossible to analyze training history programmatically.

#### 3.2 — Hardcoded `device_ids` for DataParallel

```python
# pretrain_trainer.py:L20
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]
```

Assumes exactly 8 GPUs. Will crash on machines with fewer GPUs.

#### 3.3 — `DataParallel` instead of `DistributedDataParallel`

`nn.DataParallel` has well-documented GIL bottlenecks and poor scaling beyond 2–4 GPUs. The paper mentions using 8 GPUs for pretraining — `DistributedDataParallel` should have been used.

#### 3.4 — No early stopping

The finetuning trainer runs for the full `params.epochs` regardless of whether validation metrics have plateaued. No patience mechanism.

#### 3.5 — `.cuda()` hardcoded instead of `.to(device)`

```python
# finetune_trainer.py:L21
self.model = model.cuda()
# finetune_evaluator.py:L19
x = x.cuda()
```

This hardcodes CUDA GPU 0 and prevents CPU-only execution, MPS (Apple Silicon) support, or multi-GPU device selection.

#### 3.6 — `best_model_states` can be `None`

```python
# finetune_trainer.py:L125
self.model.load_state_dict(self.best_model_states)
```

If validation never improves (e.g., `kappa` is always ≤ 0), `self.best_model_states` remains `None`, and this line will crash with `AttributeError`.

#### 3.7 — Massive if/elif chain for dataset dispatch

```python
# finetune_main.py:L63-L140
if params.downstream_dataset == 'FACED':
    ...
elif params.downstream_dataset == 'SEED-V':
    ...
# ... 11 more elif blocks ...
```

13 near-identical if/elif blocks. This should be a registry pattern or at minimum a dictionary lookup.

---

## 4. Model Architecture Code

### ✅ Pros

| Aspect | Detail |
|---|---|
| **Clean criss-cross attention** | The spatial/temporal split in [criss_cross_transformer.py](file:///home/zekun/projects/0-jobs/SigmaNova/CBraMod/models/criss_cross_transformer.py) is well-implemented — `d_model` is split in half, spatial and temporal attention are applied in parallel, then concatenated. |
| **Spectral embedding** | The FFT-based spectral branch in [PatchEmbedding](file:///home/zekun/projects/0-jobs/SigmaNova/CBraMod/models/cbramod.py#L35) adds frequency-domain information alongside the CNN-based temporal embedding. |
| **Convolutional positional encoding** | Using depthwise conv2d for position encoding is more flexible than fixed sinusoidal encoding and handles variable sequence lengths better. |

### ❌ Cons

#### 4.1 — 13 nearly-identical model wrapper files

Each downstream model file (e.g., [model_for_shu.py](file:///home/zekun/projects/0-jobs/SigmaNova/CBraMod/models/model_for_shu.py)) is a copy-paste of the same pattern with **only the channel/patch dimensions changed** in the `nn.Linear` layers. The differences are:

| Model | Linear input size | Output dim |
|---|---|---|
| SHU | `32 * 4 * 200` | 1 |
| FACED | `32 * 10 * 200` | 9 |
| SEEDV | `62 * 10 * 200` | 5 |
| TUAB | `16 * 10 * 200` | 1 |

This is ~60 lines × 13 files = **~780 lines of duplicated code** that should be one parameterized class.

#### 4.2 — Weight initialization mismatch

```python
def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
```

Kaiming init with `nonlinearity='relu'` is applied, but the model uses **GELU** activations everywhere. The gain factor is wrong. For GELU, `nonlinearity='leaky_relu'` with a small negative slope is a closer approximation, or Xavier init is more appropriate.

---

## 5. Overall Code Quality

| Dimension | Rating | Notes |
|---|---|---|
| **Type hints** | ❌ None | No type annotations on any function except the copy-pasted PyTorch transformer code. |
| **Docstrings** | ❌ None | Zero docstrings in any authored code. Only the copy-pasted `signaltools.py` and transformer code have docstrings. |
| **Testing** | ❌ None | No test files, no test directory, no CI. |
| **Error handling** | ❌ Poor | Bare `except:` clauses, no custom exceptions, no input validation. |
| **Configuration** | ⚠️ Poor | `argparse` for training scripts, but hardcoded paths in preprocessing. Moreover, `argparse` is very poor for extending to new model parameters: the only way to understand a given run's hyperparameters is to read its raw shell command, which is not readable or reusable. The elegant alternative is **YAML or Hydra configs** to declare common shared config blocks for preprocessing, dataset splitting, and model hyperparams, and to automatically store the exact config as an artifact alongside each experiment run. |
| **Linting** | ❌ None | Unused imports (`random`, `signal`, `torch`, `os`), duplicate imports, commented-out code left everywhere. |
| **Reproducibility** | ⚠️ Partial | Seeds are set but splits depend on filesystem ordering. |
| **Code reuse** | ❌ Very low | Massive copy-paste across 13 dataset loaders, 13 model wrappers, and 12 preprocessing scripts. |

---

## 6. Scalability Concerns

### 6.1 — LMDB single-writer bottleneck during preprocessing

All LMDB writes happen **one transaction per sample**:

```python
for i, (sample, label) in enumerate(zip(eeg_, labels)):
    txn = db.begin(write=True)
    txn.put(key=sample_key.encode(), value=pickle.dumps(data_dict))
    txn.commit()
```

Each `begin(write=True)` + `commit()` flushes to disk. For 1M+ pretraining samples, this creates **1M+ fsync operations**. Batching writes (committing every N samples) would be orders of magnitude faster:

```python
# Instead of opening a new transaction per sample:
txn = db.begin(write=True)
for i, sample in enumerate(samples):
    sample_key = f"{file_name}_{i}"
    txn.put(sample_key.encode(), pickle.dumps(sample))
    # Flush every 1000 writes to bound memory usage
    if i % 1000 == 999:
        txn.commit()
        txn = db.begin(write=True)
txn.commit()  # Final flush for remaining samples
```

**Multi-reader bottleneck at large-scale training (>1 GB/s IO):** LMDB supports multiple simultaneous readers via MVCC (Multi-Version Concurrency Control) on a *single node*, but its fundamental design is local and memory-mapped. It was not built for distributed multi-node training. Key limitations:

| Scenario | LMDB Behaviour |
|---|---|
| Multiple readers, single node, same `.mdb` | ✅ Safe — MVCC allows concurrent reads without locking |
| Multiple GPU workers, same node, same `.mdb` | ✅ OK — `lock=False, readonly=True` works |
| Multiple nodes reading same `.mdb` over NFS | ❌ **Dangerous** — memory-map semantics break over NFS; file locking is unreliable |
| Single `.mdb` file, >1 GB/s sustained read | ❌ **Bottleneck** — single file saturates disk IOPS; no parallel striping |

**Solutions for large-scale distributed training:**
1. **Shard LMDB:** Split the single `.mdb` into N smaller files (`shard-00.mdb`, …). Each DDP node reads only its assigned shards locally — avoids the single-file IOPS ceiling.
2. **Move to WebDataset TAR shards** (see §6.3) — inherently distributed, cloud-native, and designed for >1 GB/s parallel streaming from object storage.
3. **Use a distributed store** (e.g., [AIStore](https://github.com/NVIDIA/aistore) or [Petastorm](https://github.com/uber/petastorm)) for enterprise-scale setups where data lives on shared network storage.

### 6.2 — Pickle serialization overhead

Every LMDB value is `pickle.dumps(numpy_array)`. Pickle adds Python-specific metadata and is **not zero-copy**. For numerical arrays, the more efficient alternatives are:

- **`numpy.tobytes()` + shape metadata:** Store the raw C-contiguous byte buffer and reconstruct with `np.frombuffer(buf, dtype=np.float32).reshape(shape)`. Avoids all pickle overhead entirely.
- **Apache Arrow IPC format:** Arrow is a language-agnostic, columnar in-memory format. Each EEG patch is serialised as a `pa.Tensor` via `pa.ipc.write_tensor()`. Advantages:
  1. **Zero-copy reads:** Arrow buffers can be memory-mapped directly into NumPy/PyTorch without deserialization overhead.
  2. **Cross-language compatibility:** The same LMDB database can be read by a Go or Rust data loader.
  3. **Schema validation:** Explicit Arrow schema (channel count, sample rate, dtype) is asserted on every read, catching silent disk corruption.
- **Parquet (via Arrow):** Best suited for *tabular metadata* (file paths, labels, subject IDs), not raw patch storage. Best practice: **Arrow IPC for patches + Parquet for metadata**.

**Real benchmark on this machine** — 10,000 EEG patches (32 × 200 float32, ~256 MB total), stored in LMDB with batched writes ([`benchmarks/bench_serialization.py`](file:///home/zekun/projects/0-jobs/SigmaNova/CBraMod/benchmarks/bench_serialization.py)):

| Method | Write MB/s | Read MB/s | DB Size MB | Read speedup |
|---|---|---|---|---|
| `pickle` (baseline) | 284.8 | 2961.0 | 273.9 | 1.00× |
| `numpy.tobytes` | 481.4 | 3817.1 | 273.9 | **1.29×** |
| `pyarrow` (IPC) | 567.2 | 3979.4 | 273.9 | **1.34×** |

> [!NOTE]
> At this patch size (~25 KB), the absolute read bandwidth is already very high (3–4 GB/s from NVMe cache). The ~1.3× gain may appear modest in isolation—but at large scale (3,000-subject HBN corpus, 1.9 TB), this compounds to meaningful GPU-stall reduction since Arrow also eliminates Python GC pressure from pickle object allocation.

### 6.3 — No support for streaming / out-of-core datasets

The pretraining dataset loads **all keys into memory** at init:

```python
self.keys = pickle.loads(txn.get('__keys__'.encode()))
```

For 1M+ keys, this is a ~50 MB list of strings held in RAM permanently. There's also no support for distributed data loading across nodes.

**WebDataset Streaming** is the modern solution. Instead of an LMDB key-value store, data is stored in **sharded TAR archives** (e.g., `shard-000000.tar`, `shard-000001.tar`), where each file inside contains one sample. PyTorch's `webdataset` library streams these TAR files sequentially, **never loading the full dataset into RAM**:

```python
import webdataset as wds

dataset = (
    wds.WebDataset("s3://my-bucket/eeg-shards/shard-{000000..001024}.tar")
    .decode()  # auto-decode .npy, .json, .cls files inside the TAR
    .to_tuple("eeg.npy", "label.cls")
    .map_tuple(preprocess_fn, int)
    .batched(64)
)
```

Key advantages for EEG foundation model pretraining:
- **Streaming from cloud (S3/GCS):** Workers fetch shards on-the-fly; the full 1TB dataset never lives on local disk.
- **Perfect for multi-node DDP:** Each DDP rank processes a disjoint shard subset, with no shared-state coordination.
- **No key manifest:** The TAR format makes shards self-contained; there is no `__keys__` blob to load into RAM.
- **Exact epoch control:** Shuffle within a shard buffer (`shardshuffle=True`) gives approximate epoch semantics without full-dataset random access.

### 6.4 — `DataParallel` with `num_workers=8` only

The pretraining DataLoader uses only 8 workers regardless of the number of GPUs or CPU cores. On an 8-GPU machine with 64+ cores, this is a severe data-loading bottleneck.

### 6.5 — No mixed-precision training

Neither the pretraining nor finetuning trainers use `torch.cuda.amp` (automatic mixed precision). For a 12-layer transformer, FP16/BF16 training would roughly **halve VRAM usage** and **2× throughput** on modern GPUs.

### 6.6 — Full model `state_dict` deep-copied for checkpointing

```python
self.best_model_states = copy.deepcopy(self.model.state_dict())
```

For a model with ~15M parameters, this allocates ~60 MB of CPU RAM per checkpoint. If checkpointing is frequent, this can add up. Saving directly to disk would be more memory-efficient.

### 6.7 — No support for heterogeneous dataset formats

As the project scales to incorporate more EEG corpora, a critical structural gap emerges: each new dataset source arrives in a completely different format.

#### Dataset Format Landscape

| Source | Format | Scale | Notes |
|---|---|---|---|
| CBraMod (SHU-MI, TUEG) | `.mdb` (LMDB) | ~tens of GB | Custom preprocessed |
| TUAB / TUEV | `.pkl` per sample | ~hundreds of GB | Temple University EDF-derived |
| **HBN Dataset** | **BIDS (.bdf/.set)** | **3,000+ subjects, ~1.9 TB** | Rich metadata, 128-ch EGI, 500 Hz |

#### What is a BDF File?

**BDF (BioSemi Data Format)** is a 24-bit extension of the European Data Format (EDF), used natively by BioSemi EEG amplifier systems. Key properties:
- **Bit depth:** 24-bit ADC (vs EDF's 16-bit). Higher dynamic range, but 24-bit integers are non-standard in most CPU architectures, requiring software conversion to 32-bit on read.
- **Structure:** Binary file with a fixed-length ASCII header (subject info, channel labels, sampling rates) followed by interleaved signal data records.
- **Reading in Python (MNE):**
  ```python
  import mne
  raw = mne.io.read_raw_bdf("sub-01_task-RestingState_eeg.bdf", preload=True)
  # Access the numpy array: shape (n_channels, n_samples)
  data, times = raw.get_data(return_times=True)
  ```
- **Writing:** MNE does not natively write BDF. Use `pyedflib` to export processed data back to BDF/EDF format.

#### Unifying Heterogeneous Formats: Options

The fundamental challenge is converting formats with different channel counts (19-ch TUH vs 128-ch HBN), sampling rates (200 Hz processed vs 500 Hz raw), and metadata schemas into a single training-ready corpus.

| Approach | Description | Pros | Cons |
|---|---|---|---|
| **LMDB shards (per-dataset)** | Continue current approach; one `.mdb` per dataset | Simple; works today | No unified access; multi-reader ceiling; NFS-unsafe |
| **One file per subject per task** | `.npz` or `.pt` file per subject | Simple; easy partial loading | Millions of small files; bad for sequential I/O |
| **WebDataset TAR shards** | Pack N subjects per `.tar` shard | Cloud-native; DDP-ready; no RAM manifest | Requires re-packaging all existing data |
| **HDF5 (`.h5`)** | Single hierarchical file; supports partial reads via h5py | Excellent for exploratory analysis | Poor random-access under DataLoader; GIL-locked reads |
| **Zarr + FSSpec** | Chunked N-dimensional arrays; cloud-native (S3/GCS) | True zero-copy cloud reads; supports 128-ch data natively | Ecosystem less mature than WebDataset for DL training |

**Recommendation for a unified EEG pretraining pipeline:**

```
Raw Data (BDF/EDF/MAT)
       ↓  [common MNE preprocessing pipeline]
Normalized .npz per recording (channels reshaped, resampled to 200 Hz)
       ↓  [patch extraction + metadata Parquet table]
WebDataset TAR shards (patches .npy + metadata .json per sample)
       ↓  [wds.WebDataset streaming loader]
DDP Training
```

This architecture decouples format-specific parsing (MNE handles BDF/EDF/SET/MAT) from storage (TAR shards) and training (WebDataset), making adding new datasets like HBN trivial: write one MNE-based adapter, run the patcher, and append new shards without touching the training code.

---

## 7. Recommended Remediation Plan

As an ML engineer, here is what I would implement, ordered by **impact × effort**:

### 🔴 Critical (correctness / reliability)

1. **Fix ISRUC seq/label pairing** — Sort both file lists explicitly to guarantee correct alignment:
   ```python
   seq_fnames = sorted(os.listdir(subject_seq))
   label_fnames = sorted(os.listdir(subject_label))
   ```

2. **Fix TUEV file descriptor leak** — Replace bare `open()` with `with` statement:
   ```python
   with open(os.path.join(self.data_dir, file), "rb") as f:
       data_dict = pickle.load(f)
   ```

3. **Guard `best_model_states`** — Initialize to the model's initial state, or add an explicit check before `load_state_dict`.

4. **Remove bare `except:`** — Catch specific exceptions (`ValueError`, `KeyError`) and log full tracebacks.

### 🟡 High priority (performance / scalability)

5. **Add AMP (mixed-precision training)** — Wrap forward pass in `torch.cuda.amp.autocast()` and use `GradScaler`. Estimated 1.5–2× training speedup.

6. **Switch to `DistributedDataParallel`** — Replace `DataParallel` with DDP + `torchrun` launcher. Critical for pretraining at scale.

7. **Batch LMDB writes** — Commit every N samples instead of every single sample:
   ```python
   txn = db.begin(write=True)
   for i, sample in enumerate(samples):
       txn.put(key=..., value=...)
       if i % 1000 == 0:
           txn.commit()
           txn = db.begin(write=True)
   txn.commit()
   ```

8. **Set `readahead=False`** on LMDB for random-access DataLoaders.

9. **Add `num_workers` and `pin_memory=True`** to all downstream DataLoaders.

### 🟢 Medium priority (maintainability / extensibility)

10. **Unify preprocessing into a configurable pipeline** — Create a base `Preprocessor` class with shared filtering/resampling/patching logic. Use a YAML config for paths, channel mappings, and split ratios. Each dataset overrides only dataset-specific parsing.

11. **Create a single parameterized downstream model** — Replace 13 model files with:
    ```python
    class DownstreamModel(nn.Module):
        def __init__(self, backbone, n_channels, n_patches, d_model, n_classes, classifier_type):
            ...
    ```

12. **Replace the if/elif chain with a dataset/model registry**:
    ```python
    DATASET_REGISTRY = {
        "SHU-MI": (shu_dataset.LoadDataset, model_for_shu.Model, "binary"),
        ...
    }
    ```

13. **Replace `print()` with `loguru`** — Structured logging with file rotation, as per project conventions.

14. **Replace `os.path` with `pathlib`** — As per project GEMINI.md conventions.

15. **Add type hints and docstrings** to all public functions.

16. **Parameterize the normalization constant** — Replace `/ 100` with a configurable normalization strategy (z-score, min-max, or a dataset-specific constant stored in metadata).

17. **Add data integrity checks** — Assert expected shapes, channel counts, and sampling rates after loading. Store checksums in LMDB metadata.

### 🔵 Nice-to-have (future-proofing)

18. **WebDataset / Mosaic StreamingDataset** for pretraining — enables streaming from cloud storage and distributed sharding.

19. **Weights & Biases / MLflow integration** — Replace ad-hoc `print()` logging + manual model saving with experiment tracking.

20. **Add `pytest` test suite** — At minimum, test that each preprocessing script produces expected output shapes, and that each dataset loader returns correctly shaped tensors.

---

## Summary

The CBraMod codebase reads like **high-quality research prototype code**: the core model architecture and signal processing are sound, but the engineering infrastructure around it — preprocessing, data loading, configuration, error handling, and scalability — is at a "make it work for our specific lab machines" level. The dominant anti-pattern is **copy-paste duplication** (~780 lines of model wrappers, ~500 lines of dataset loaders, ~1000 lines of preprocessing), which makes the codebase fragile and resistant to extension. The most impactful single change would be to **unify the dataset/model/preprocessing code into registry-based, parameterized abstractions** — this would cut the codebase by roughly 40% while making it trivial to add new downstream tasks.

**[1] Resampling to 200 Hz & Nyquist:** To accurately capture a signal without aliasing, you must sample at least twice the maximum frequency (Nyquist-Shannon Theorem). A 200 Hz sampling rate creates a 100 Hz Nyquist limit, which perfectly covers all standard clinical EEG bands: Delta (0.5–4 Hz), Theta (4–8 Hz), Alpha (8–13 Hz), Beta (13–30 Hz), and Gamma (30–100 Hz).

**[2] Bandpass (0.3–75 Hz):** This is the standard clinical window. The 0.3 Hz high-pass removes slow-drift artifacts (sweat, electrode polarization). The 75 Hz low-pass attenuates high-frequency muscle noise (EMG) and non-cortical hardware artifacts.

**[3] 60 Hz Notch Filter:** Mandatory for the TUH dataset (recorded at Temple University, Philadelphia). It acts as a surgical band-stop filter explicitly targeting the North American alternating current (AC) power line frequency to eliminate electrical room interference without destroying surrounding neural signals.
