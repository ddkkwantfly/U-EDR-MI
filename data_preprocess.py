from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio
import mne


# =========================
# Config
# =========================
@dataclass
class PreprocessConfig:
    data_dir: Path = Path("Data")
    label_dir: Path = Path("label")
    out_dir: Path = Path("processed_final")

    # choose: 2.0 or 1.5
    win_len_s: float = 1.0

    # MI bandpass
    l_freq: float = 8.0
    h_freq: float = 30.0

    pick_eeg_only: bool = True

    # event code for trial start in BCI2a (usually 768)
    trial_start_code: int = 768

    # training output format
    out_format: str = "eegnet"   # "eegnet" => (N, 1, C, T) ; "plain" => (N, C, T)

    save_float32: bool = True
    seed: int = 42

    # split policy by TRIAL index (avoid leakage)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # If cue events after 768 contain extra non-MI types, keep only top-K frequent cue ids
    keep_top_k_cue: int = 4


# def default_offsets(win_len_s: float) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
#     """
#     IC: within [2, 6]s
#     NC: within [0, 2]s
#     """
#     if abs(win_len_s - 2.0) < 1e-6:
#         ic_offsets = (2.0, 3.0, 4.0)    # [2-4], [3-5], [4-6]
#         nc_offsets = (0.0,)             # [0-2]
#     elif abs(win_len_s - 1.5) < 1e-6:
#         ic_offsets = (2.0, 2.75, 3.5, 4.25)  # all inside [2,6]
#         nc_offsets = (0.0, 0.5)              # both inside [0,2]
#     else:
#         raise ValueError("win_len_s must be 2.0 or 1.5")
#     return ic_offsets, nc_offsets

def default_offsets(win_len_s: float, stride_s: float = 0.5) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """
    IC: inside [2,6]s
    NC: inside [0,2]s

    Offsets are the window START times.
    """
    def make_offsets(t0: float, t1: float) -> Tuple[float, ...]:
        # windows must fit fully: start <= t1 - win_len_s
        last = t1 - win_len_s
        xs = []
        x = t0
        while x <= last + 1e-9:
            xs.append(round(float(x), 6))
            x += stride_s
        return tuple(xs)

    if win_len_s in (2.0, 1.5, 1.0):
        ic_offsets = make_offsets(2.0, 6.0)
        nc_offsets = make_offsets(0.0, 2.0)
    else:
        raise ValueError("win_len_s must be 2.0, 1.5, or 1.0")
    return ic_offsets, nc_offsets

# =========================
# IO helpers
# =========================
def find_mat_label_vector(mat_path: Path) -> np.ndarray:
    """
    BCI2a labels in .mat are usually under 'classlabel' (len 288, values 1..4).
    We'll search robustly.
    """
    mat = sio.loadmat(str(mat_path))
    keys = [k for k in mat.keys() if not k.startswith("__")]

    for k in ["classlabel", "labels", "label", "y", "Y"]:
        if k in mat:
            return np.asarray(mat[k]).squeeze().astype(int)

    # fallback: find vector length 288
    for k in keys:
        v = np.asarray(mat[k])
        if v.size == 288:
            return v.squeeze().astype(int)
        if v.ndim == 2 and (v.shape == (288, 1) or v.shape == (1, 288)):
            return v.squeeze().astype(int)

    raise RuntimeError(f"Cannot find labels in {mat_path}. Keys={keys}")


def read_gdf(gdf_path: Path, cfg: PreprocessConfig) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=True, stim_channel="auto", verbose="ERROR")
    if cfg.pick_eeg_only:
        picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude="bads")
        raw.pick(picks)
    raw.filter(cfg.l_freq, cfg.h_freq, fir_design="firwin", verbose="ERROR")
    return raw


def events_from_raw(raw: mne.io.BaseRaw) -> Tuple[np.ndarray, Dict[str, int]]:
    return mne.events_from_annotations(raw, verbose="ERROR")


def resolve_event_id(event_id: Dict[str, int], code: int) -> int:
    """
    event_id: annotation description -> integer id used in events[:,2]
    Most often keys are numeric strings like "768".
    """
    if str(code) in event_id:
        return event_id[str(code)]
    # fallback: sometimes integer id itself matches code even if key isn't numeric
    for _k, v in event_id.items():
        if v == code:
            return v
    
    raise KeyError(f"Event code {code} not found. Keys sample: {list(event_id.keys())[:30]}")


# =========================
# Trial alignment (robust: do NOT assume 769-772 exist)
# =========================
def align_trials_by_next_event(
    events: np.ndarray,
    trial_start_eid: int,
) -> Tuple[List[int], List[int]]:
    """
    For each trial start (eid==trial_start_eid, usually 768),
    take the first following event (eid != 768) before next trial start as the cue marker.

    Returns:
      trial_starts: start_sample per trial
      cue_eids:     cue eid per trial (for debugging/filtering only)
    """
    events = events[np.argsort(events[:, 0])]
    ts_idx = np.where(events[:, 2] == trial_start_eid)[0]
    if len(ts_idx) == 0:
        return [], []

    trial_starts: List[int] = []
    cue_eids: List[int] = []

    for k, ti in enumerate(ts_idx):
        s0 = int(events[ti, 0])
        s1 = int(events[ts_idx[k + 1], 0]) if k + 1 < len(ts_idx) else int(events[-1, 0]) + 1

        seg = events[(events[:, 0] > s0) & (events[:, 0] < s1)]
        seg = seg[seg[:, 2] != trial_start_eid]  # drop 768 itself
        if len(seg) == 0:
            continue

        cue = int(seg[0, 2])
        trial_starts.append(s0)
        cue_eids.append(cue)

    # deduplicate consecutive duplicates (rare, but can happen)
    dedup_starts: List[int] = []
    dedup_cues: List[int] = []
    last = None
    for s, c in zip(trial_starts, cue_eids):
        if last is None or s != last:
            dedup_starts.append(s)
            dedup_cues.append(c)
        last = s

    return dedup_starts, dedup_cues


def filter_top_k_cues(
    trial_starts: List[int],
    cue_eids: List[int],
    k: int = 4,
) -> Tuple[List[int], List[int]]:
    """
    If cue_eids include extra non-MI event types, keep only the top-k most frequent cue ids.
    """
    if k <= 0:
        return trial_starts, cue_eids
    arr = np.asarray(cue_eids, dtype=int)
    vals, cnts = np.unique(arr, return_counts=True)
    if len(vals) <= k:
        return trial_starts, cue_eids
    topk = set(vals[np.argsort(cnts)[-k:]].tolist())
    keep_idx = [i for i, c in enumerate(cue_eids) if c in topk]
    return [trial_starts[i] for i in keep_idx], [cue_eids[i] for i in keep_idx]


def extract_windows(raw: mne.io.BaseRaw, start_sample: int, offsets_s: Tuple[float, ...], win_len_s: float) -> np.ndarray:
    sfreq = float(raw.info["sfreq"])
    win_samp = int(round(win_len_s * sfreq))
    xs: List[np.ndarray] = []
    for off in offsets_s:
        s0 = start_sample + int(round(off * sfreq))
        s1 = s0 + win_samp
        x = raw.get_data(start=s0, stop=s1)  # (C, T)
        if x.shape[1] != win_samp:
            continue
        xs.append(x)
    if not xs:
        c = raw.get_data().shape[0]
        return np.zeros((0, c, win_samp), dtype=np.float32)
    return np.stack(xs, axis=0)  # (n_win, C, T)


# =========================
# Core preprocessing
# =========================
def preprocess_one_file(gdf_path: Path, mat_path: Path, cfg: PreprocessConfig) -> Dict:
    raw = read_gdf(gdf_path, cfg)
    events, event_id = events_from_raw(raw)

    trial_start_eid = resolve_event_id(event_id, cfg.trial_start_code)

    # labels from mat define MI trials order: 1..4 (len 288)
    y_trial = find_mat_label_vector(mat_path)
    y_trial = np.clip(y_trial.astype(int), 1, 4)

    # align trials robustly
    trial_starts, cue_eids = align_trials_by_next_event(events, trial_start_eid)

    # filter to top-K cue types (MI has 4 cues)
    trial_starts, cue_eids = filter_top_k_cues(trial_starts, cue_eids, k=cfg.keep_top_k_cue)
    

    # ===== DEBUG: cue_eid -> annotation key (what does "7" mean?) =====
    inv = {v: k for k, v in event_id.items()}  # reverse map: int->str
    print("[DEBUG] cue_eids unique:", sorted(set(cue_eids)))
    print("[DEBUG] inv_event_id for cue ids:", {cid: inv.get(cid, None) for cid in sorted(set(cue_eids))})
    print("[DEBUG] sample event_id items:", list(event_id.items())[:30])

    # ===== DEBUG: check spacing of trial starts =====
    ds = np.diff(np.array(trial_starts, dtype=int))
    if ds.size > 0:
        print(f"[DEBUG-DS] {gdf_path.stem}: head={ds[:10].tolist()} median={float(np.median(ds))} min={int(ds.min())} max={int(ds.max())}")

    # strict align length
    n = min(len(trial_starts), len(y_trial))
    trial_starts = trial_starts[:n]
    cue_eids = cue_eids[:n]
    y_trial = y_trial[:n]

    ic_offsets, nc_offsets = default_offsets(cfg.win_len_s, stride_s = 0.5)

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    trial_index_list: List[np.ndarray] = []

    for i, s in enumerate(trial_starts):
        # IC windows (label 1..4)
        X_ic = extract_windows(raw, s, ic_offsets, cfg.win_len_s)
        if X_ic.shape[0] > 0:
            X_list.append(X_ic)
            y_list.append(np.full((X_ic.shape[0],), y_trial[i], dtype=int))
            trial_index_list.append(np.full((X_ic.shape[0],), i, dtype=int))

        # NC windows (label 0) from baseline [0,2]
        X_nc = extract_windows(raw, s, nc_offsets, cfg.win_len_s)
        if X_nc.shape[0] > 0:
            X_list.append(X_nc)
            y_list.append(np.zeros((X_nc.shape[0],), dtype=int))
            trial_index_list.append(np.full((X_nc.shape[0],), i, dtype=int))

    if not X_list:
        raise RuntimeError(f"No windows extracted from {gdf_path.name}")

    X = np.concatenate(X_list, axis=0)  # (N, C, T)
    y = np.concatenate(y_list, axis=0)  # (N,)
    trial_idx = np.concatenate(trial_index_list, axis=0)  # (N,)

    if cfg.save_float32:
        X = X.astype(np.float32)

    if cfg.out_format == "eegnet":
        X = X[:, None, :, :]  # (N, 1, C, T)

    # meta info
    cue_unique = sorted(list(set(cue_eids)))
    meta = dict(
        gdf=str(gdf_path),
        mat=str(mat_path),
        sfreq=float(raw.info["sfreq"]),
        ch_names=list(raw.info["ch_names"]),
        win_len_s=float(cfg.win_len_s),
        ic_offsets_s=list(ic_offsets),
        nc_offsets_s=list(nc_offsets),
        filter=[cfg.l_freq, cfg.h_freq],
        out_format=cfg.out_format,
        n_trials=int(n),
        cue_eid_unique=cue_unique,
        X_shape=list(X.shape),
        y_counts={str(k): int((y == k).sum()) for k in sorted(set(y.tolist()))},
        event_id_keys_sample=list(event_id.keys())[:30],
    )

    print(f"[DEBUG] {gdf_path.stem}: n_trials={n}, cue_unique={cue_unique}")

    return {"X": X, "y": y, "trial_idx": trial_idx, "meta": meta}


def split_by_trial_indices(trial_indices: np.ndarray, cfg: PreprocessConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split windows by unique trial index to avoid leakage.
    """
    rng = np.random.default_rng(cfg.seed)
    uniq = np.unique(trial_indices)
    rng.shuffle(uniq)

    n = len(uniq)
    n_train = int(round(cfg.train_ratio * n))
    n_val = int(round(cfg.val_ratio * n))
    n_test = n - n_train - n_val

    train_trials = set(uniq[:n_train].tolist())
    val_trials = set(uniq[n_train:n_train + n_val].tolist())
    test_trials = set(uniq[n_train + n_val:].tolist())

    train_mask = np.isin(trial_indices, list(train_trials))
    val_mask = np.isin(trial_indices, list(val_trials))
    test_mask = np.isin(trial_indices, list(test_trials))

    return train_mask, val_mask, test_mask


def save_split_npz(out_base: Path, X: np.ndarray, y: np.ndarray, meta: Dict, prefix: str) -> None:
    out_path = out_base / f"{prefix}.npz"
    np.savez_compressed(out_path, X=X, y=y, meta=json.dumps(meta))
    print(f"  saved: {out_path} | X={X.shape} y={y.shape}")


def merge_splits(folder: Path) -> None:
    """
    Merge all subjects/runs within this folder into ALL_train/val/test.npz
    """
    for split in ["train", "val", "test"]:
        files = sorted(folder.glob(f"*_{split}.npz"))
        if not files:
            continue
        Xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        for f in files:
            pack = np.load(f, allow_pickle=True)
            Xs.append(pack["X"])
            ys.append(pack["y"])
        X = np.concatenate(Xs, axis=0)
        y = np.concatenate(ys, axis=0)
        out = folder / f"ALL_{split}.npz"
        np.savez_compressed(out, X=X, y=y)
        print(f"[MERGE] {out} | X={X.shape} y={y.shape}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--win", type=float, default=2.0, choices=[2.0, 1.5], help="window length in seconds")
    parser.add_argument("--out_format", type=str, default="eegnet", choices=["eegnet", "plain"])
    parser.add_argument("--out_dir", type=str, default="processed_final")
    parser.add_argument("--merge", action="store_true", help="merge all per-file splits into ALL_train/val/test.npz")
    args = parser.parse_args()

    cfg = PreprocessConfig(win_len_s=float(args.win), out_format=args.out_format, out_dir=Path(args.out_dir))
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    gdfs = sorted(cfg.data_dir.glob("A??[ET].gdf"))
    if not gdfs:
        raise FileNotFoundError(f"No .gdf found in {cfg.data_dir.resolve()}")

    base = cfg.out_dir / f"win{cfg.win_len_s:.1f}s"
    base.mkdir(parents=True, exist_ok=True)

    for gdf_path in gdfs:
        stem = gdf_path.stem  # A01T / A01E
        mat_path = cfg.label_dir / f"{stem}.mat"
        if not mat_path.exists():
            print(f"[SKIP] {stem}: label not found: {mat_path}")
            continue

        print(f"[PROCESS] {stem} | win={cfg.win_len_s}s | format={cfg.out_format}")
        pack = preprocess_one_file(gdf_path, mat_path, cfg)
        X, y, trial_idx, meta = pack["X"], pack["y"], pack["trial_idx"], pack["meta"]

        # split by trial index
        train_mask, val_mask, test_mask = split_by_trial_indices(trial_idx, cfg)

        # save per file
        save_split_npz(base, X[train_mask], y[train_mask], meta, prefix=f"{stem}_train")
        save_split_npz(base, X[val_mask], y[val_mask], meta, prefix=f"{stem}_val")
        save_split_npz(base, X[test_mask], y[test_mask], meta, prefix=f"{stem}_test")

    if args.merge:
        merge_splits(base)

    print("Done.")


if __name__ == "__main__":
    main()
