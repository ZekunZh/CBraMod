# SHU-MI Dataset Observations & Summary

*(Note: The contents of the specific Jupyter interactive session "Interactive-1" could not be read directly as it is a transient VSCode window not saved to disk. However, based on our investigation so far and the data excerpts you provided, here is a comprehensive summary.)*

## 1. Data Shape and Structure
The standard raw shape extracted from the processing pipeline yields:
**`EEG data shape: (100, 32, 1000)`**

This breaks down as follows:
- **Batches/Trials (`100`)**: Number of segmented trial instances/windows per physical file.
- **Channels (`32`)**: Number of EEG electrodes recording the signals.
- **Timepoints (`1000`)**: The number of sampling points per trial before resampling.

During the preprocessing phase (in `preprocessing_shu.py`), these tensors are resampled along the time axis to `800` data points (`scipy.signal.resample(eeg, 800, axis=2)`) and reshaped to `(100, 32, 4, 200)` to fit the target patched-sequence architecture for the CBraMod Transformer.

## 2. Event Labels (Motor Imagery)
The raw extracted labels array (`[1 1 2 1 1 2 ...]`) correlates directly to the `task-motorimagery_events.json` metadata provided by the dataset authors:
- **`1`**: Imagining the movement of the **Left Hand**.
- **`2`**: Imagining the movement of the **Right Hand**.

The codebase then transforms these labels to `[0, 1]` logic via `label - 1` for standard binary cross-entropy classification in deep learning frameworks like PyTorch.

## 3. Storage and Preprocessing Scale
- **Dataset Size:** Contains 125 individual `.mat` sequence files uniformly spanning `train` (75), `val` (25), and `test` (25).
- **Storage Strategy:** Handled entirely by flattening to a zero-copy LMDB database.
- **Important Bottleneck Flag:** 
  Due to the aggregation of all 125 sessions, the total dataset size far exceeded the standard `110MB` mapping configuration originally defined in the code. To prevent `MDB_MAP_FULL` crashes without wasting physical RAM, the `map_size` was effectively raised to an OS-safe `1TB` virtual ceiling constraint, allowing the DB to freely expand to its natural ~2-3GB limit on disk.
