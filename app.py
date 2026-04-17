from __future__ import annotations

import csv
from datetime import datetime
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile

import mne
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from mne.datasets import eegbci
from mne.decoding import CSP
from scipy.signal import spectrogram, welch
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import joblib

try:
    import h5py
except Exception:
    h5py = None


app = Flask(__name__)

DATA_ROOT = Path.home() / "mne_data" / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"
MI_RUNS = [4, 8, 12]
PREFERRED_CHANNELS = ["FCz", "C3", "Cz", "C4", "CP3", "CPz", "CP4", "Pz"]
BANDS = {
    "Alpha (8-12 Hz)": (8, 12),
    "Beta (13-30 Hz)": (13, 30),
    "Gamma (31-45 Hz)": (31, 45),
}
CONFIG_PATH = Path(__file__).with_name("config.json")
MODEL_DIR = Path(__file__).with_name("saved_models")
MODEL_PATH = MODEL_DIR / "latest_model.joblib"
REPORT_PATH = MODEL_DIR / "latest_report.json"

APP_STATE = {
    "source": "none",
    "dataset": None,
    "summary": {"status": "No EEG dataset loaded."},
    "logs": [],
    "recent_runs": [],
    "last_payload": None,
    "saved_model_available": False,
    "pipeline_results": {"baseline": None, "gan_augmented": None},
    "augmentation": None,
}


DEFAULT_CONFIG = {
    "filter_band": [8.0, 30.0],
    "epoch_window": [1.0, 4.0],
    "csp_components": 4,
    "test_size": 0.3,
    "random_seed": 42,
    "preferred_classifier": "SVM",
    "subject_default": 1,
}


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        return {**DEFAULT_CONFIG, **loaded}
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(DEFAULT_CONFIG, handle, indent=2)
    return DEFAULT_CONFIG.copy()


CONFIG = _load_config()
MODEL_DIR.mkdir(exist_ok=True)


def _load_local_module(module_name: str, relative_path: str):
    module_path = Path(__file__).with_name("app") / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load local module: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


GAN_AUGMENTATION = _load_local_module("neurosense_gan_augmentation", "gan_augmentation.py")
COMPARISON = _load_local_module("neurosense_comparison", "comparison.py")
train_gan = GAN_AUGMENTATION.train_gan
generate_synthetic_data = GAN_AUGMENTATION.generate_synthetic_data
augment_dataset = GAN_AUGMENTATION.augment_dataset
compare_models = COMPARISON.compare_models


def _log(level: str, message: str) -> None:
    APP_STATE["logs"].append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "level": level.upper(),
            "message": message,
        }
    )
    APP_STATE["logs"] = APP_STATE["logs"][-12:]


def _start_run_log() -> None:
    APP_STATE["logs"] = []
    _log("info", "Session initialized.")
    _log("info", f"Configuration loaded with random seed {CONFIG['random_seed']}.")


def _dataset_family(source: str, subject: str) -> str:
    text = f"{source} {subject}".lower()
    if "physionet" in text or "eegbci" in text or ".edf" in text:
        return "PhysioNet EEGBCI"
    if ".mat" in text or "matlab" in text or "kaggle" in text:
        return "MATLAB EEG Session"
    if "synthetic" in text or "demo" in text:
        return "Synthetic Demo"
    return "Unified EEG Session"


def _subject_dir(subject: int) -> Path:
    return DATA_ROOT / f"S{subject:03d}"


def _safe_cv_splits(y: np.ndarray) -> int:
    counts = np.bincount(y)
    valid = counts[counts > 0]
    if valid.size == 0:
        return 2
    return max(2, min(5, int(valid.min())))


def _pick_channels(raw: mne.io.BaseRaw) -> list[str]:
    available = [name for name in PREFERRED_CHANNELS if name in raw.ch_names]
    if len(available) >= 6:
        return available[:8]
    eeg_channels = mne.pick_types(raw.info, eeg=True, exclude="bads")
    return [raw.ch_names[idx] for idx in eeg_channels[:8]]


def _raw_to_waveform(raw: mne.io.BaseRaw, duration: float = 8.0) -> dict:
    channels = _pick_channels(raw)
    picks = [raw.ch_names.index(name) for name in channels]
    stop = min(int(raw.info["sfreq"] * duration), raw.n_times)
    times = raw.times[:stop]
    segment = raw.get_data(picks=picks, start=0, stop=stop) * 1e6
    offsets = np.arange(len(channels))[::-1] * 160.0
    traces = []
    palette = ["#22d3ee", "#8b5cf6", "#22c55e", "#f59e0b", "#3b82f6", "#ec4899", "#14b8a6", "#f97316"]
    for idx, name in enumerate(channels):
        traces.append(
            {
                "name": name,
                "x": np.round(times, 4).tolist(),
                "y": np.round(segment[idx] + offsets[idx], 3).tolist(),
                "color": palette[idx % len(palette)],
            }
        )
    return {"traces": traces, "offsets": offsets.tolist()}


def _epochs_to_waveform(epochs: mne.Epochs) -> dict:
    sample = epochs.get_data()[0]
    channels = epochs.ch_names[: min(8, len(epochs.ch_names))]
    segment = sample[: len(channels)] * 1e6
    times = np.arange(segment.shape[-1]) / float(epochs.info["sfreq"])
    offsets = np.arange(len(channels))[::-1] * 160.0
    palette = ["#22d3ee", "#8b5cf6", "#22c55e", "#f59e0b", "#3b82f6", "#ec4899", "#14b8a6", "#f97316"]
    traces = []
    for idx, name in enumerate(channels):
        traces.append(
            {
                "name": name,
                "x": np.round(times, 4).tolist(),
                "y": np.round(segment[idx] + offsets[idx], 3).tolist(),
                "color": palette[idx % len(palette)],
            }
        )
    return {"traces": traces, "offsets": offsets.tolist()}


def _normalize_trials(trials: np.ndarray, n_channels: int) -> np.ndarray:
    arr = np.asarray(trials, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError("Expected 2D or 3D trial data.")
    if arr.shape[0] == n_channels:
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.shape[1] == n_channels:
        pass
    elif arr.shape[2] == n_channels:
        arr = np.transpose(arr, (0, 2, 1))
    else:
        raise ValueError("Could not infer channel axis from trial array.")
    return arr


def _as_plain_dict(value, prefix: str = "") -> dict[str, np.ndarray]:
    flat = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).startswith("__"):
                continue
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_as_plain_dict(item, next_prefix))
        return flat
    if hasattr(value, "_fieldnames"):
        for key in getattr(value, "_fieldnames", []):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_as_plain_dict(getattr(value, key), next_prefix))
        return flat
    if isinstance(value, np.ndarray) and value.dtype == object:
        for index, item in np.ndenumerate(value):
            next_prefix = f"{prefix}[{','.join(map(str, index))}]" if prefix else str(index[0])
            flat.update(_as_plain_dict(item, next_prefix))
        return flat
    if prefix:
        flat[prefix] = np.asarray(value)
    return flat


def _epochs_from_arrays(trials: np.ndarray, labels: np.ndarray, sfreq: float, ch_names: list[str], source_name: str) -> mne.Epochs:
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    trials = _normalize_trials(trials, len(ch_names))
    labels = np.asarray(labels).astype(int).reshape(-1)
    if trials.shape[0] != labels.shape[0]:
        raise ValueError(f"{source_name}: trial count does not match label count.")
    events = np.column_stack([np.arange(len(labels)), np.zeros(len(labels), dtype=int), labels + 2])
    epochs = mne.EpochsArray(
        trials,
        info,
        events=events,
        event_id={"Left Hand Movement": 2, "Right Hand Movement": 3},
        tmin=0.0,
        verbose="ERROR",
    )
    epochs.set_montage("standard_1005", on_missing="ignore")
    return epochs


def _combine_epochs(items: list[tuple[str, mne.Epochs, mne.io.BaseRaw | None]]) -> dict:
    epochs_list = [item[1] for item in items]
    combined = mne.concatenate_epochs(epochs_list, verbose="ERROR") if len(epochs_list) > 1 else epochs_list[0]
    representative_raw = next((item[2] for item in items if item[2] is not None), None)
    source_names = [item[0] for item in items]
    return {
        "epochs": combined,
        "raw": representative_raw,
        "source_names": source_names,
    }


def _load_signal_file(file_path: Path) -> mne.io.BaseRaw:
    suffix = file_path.suffix.lower()
    if suffix == ".edf":
        return mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")
    raise ValueError(f"Unsupported signal file: {file_path.name}")


def _prepare_motor_imagery_epochs(raw: mne.io.BaseRaw) -> mne.Epochs:
    eegbci.standardize(raw)
    raw.set_montage("standard_1005", on_missing="ignore")
    raw.filter(CONFIG["filter_band"][0], CONFIG["filter_band"][1], fir_design="firwin", verbose="ERROR")
    events, event_map = mne.events_from_annotations(raw, verbose="ERROR")
    if not event_map:
        raise ValueError("No event annotations were found for motor imagery extraction.")

    left_keys = {"t1", "769", "left", "left_hand", "left hand"}
    right_keys = {"t2", "770", "right", "right_hand", "right hand"}
    remapped = []
    for name, code in event_map.items():
        key = name.strip().lower()
        if key in left_keys:
            remapped.append((name, code, 2))
        if key in right_keys:
            remapped.append((name, code, 3))
    if not remapped:
        raise ValueError("No left/right motor imagery events were detected in this recording.")

    left_codes = {item[1] for item in remapped if item[2] == 2}
    right_codes = {item[1] for item in remapped if item[2] == 3}
    selected_events = []
    for event in events:
        if event[2] in left_codes:
            selected_events.append([event[0], event[1], 2])
        elif event[2] in right_codes:
            selected_events.append([event[0], event[1], 3])
    if not selected_events:
        raise ValueError("The file contained annotations, but none matched left/right motor imagery trials.")

    picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude="bads")
    return mne.Epochs(
        raw,
        np.asarray(selected_events, dtype=int),
        event_id={"Left Hand Movement": 2, "Right Hand Movement": 3},
        tmin=CONFIG["epoch_window"][0],
        tmax=CONFIG["epoch_window"][1],
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )


def _label_to_binary(values: np.ndarray) -> np.ndarray:
    labels = np.asarray(values).astype(int).reshape(-1)
    unique = sorted({int(item) for item in labels.tolist()})
    if set(unique).issubset({0, 1}):
        return labels
    if set(unique).issubset({1, 2}):
        return labels - 1
    if 769 in unique or 770 in unique:
        mapped = []
        for item in labels:
            if int(item) == 769:
                mapped.append(0)
            elif int(item) == 770:
                mapped.append(1)
        return np.asarray(mapped, dtype=int)
    raise ValueError("Only left/right motor imagery labels are currently supported.")


def _extract_from_mat_dict(data: dict, source_name: str) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    clean = {}
    for key, value in data.items():
        if str(key).startswith("__"):
            continue
        clean.update(_as_plain_dict(value, str(key)))

    imagery_left_key = next((key for key in clean if "imagery_left" in key.lower()), None)
    imagery_right_key = next((key for key in clean if "imagery_right" in key.lower()), None)
    srate_key = next((key for key in clean if any(term in key.lower() for term in ["srate", "sfreq", "fs"])), None)
    if imagery_left_key and imagery_right_key and srate_key:
        left = np.asarray(clean[imagery_left_key])
        right = np.asarray(clean[imagery_right_key])
        n_channels = left.shape[0] if left.ndim == 3 else left.shape[1]
        sfreq = float(np.asarray(clean[srate_key]).reshape(-1)[0])
        ch_names = [f"EEG {idx + 1:02d}" for idx in range(n_channels)]
        left_trials = _normalize_trials(left, n_channels)
        right_trials = _normalize_trials(right, n_channels)
        trials = np.concatenate([left_trials, right_trials], axis=0)
        labels = np.concatenate([np.zeros(len(left_trials), dtype=int), np.ones(len(right_trials), dtype=int)])
        return trials, labels, sfreq, ch_names

    arrays = [(key, np.asarray(value)) for key, value in clean.items()]
    label_candidates = []
    trial_candidates = []
    channel_candidates = []
    sfreq = 160.0

    for key, arr in arrays:
        key_lower = key.lower()
        if arr.size == 0:
            continue
        if any(term in key_lower for term in ["fs", "srate", "sfreq"]) and arr.size <= 4:
            sfreq = float(arr.reshape(-1)[0])
        if arr.ndim in {1, 2} and any(term in key_lower for term in ["label", "class", "target", "y", "marker"]):
            label_candidates.append((key, arr.reshape(-1)))
        elif arr.ndim == 3:
            trial_candidates.append((key, arr))
        elif arr.ndim == 2 and min(arr.shape) <= 128 and max(arr.shape) >= 64:
            channel_candidates.append((key, arr))

    candidate_trials = None
    candidate_labels = None
    for label_key, labels in label_candidates:
        label_count = labels.shape[0]
        for trial_key, trials in trial_candidates:
            if label_count in trials.shape:
                candidate_trials = trials
                candidate_labels = labels
                break
        if candidate_trials is not None:
            break

    if candidate_trials is None and channel_candidates and label_candidates:
        for label_key, labels in label_candidates:
            label_count = labels.shape[0]
            for trial_key, samples in channel_candidates:
                if samples.ndim == 2 and samples.shape[1] % label_count == 0:
                    samples_per_trial = samples.shape[1] // label_count
                    candidate_trials = samples.reshape(samples.shape[0], label_count, samples_per_trial).transpose(1, 0, 2)
                    candidate_labels = labels
                    break
            if candidate_trials is not None:
                break

    if candidate_trials is None:
        grouped = {}
        for key, value in clean.items():
            if "." not in key:
                continue
            prefix, suffix = key.rsplit(".", 1)
            grouped.setdefault(prefix, {})[suffix.lower()] = np.asarray(value)
        for prefix, item in grouped.items():
            if {"x", "trial", "y"} <= item.keys():
                signal = np.asarray(item["x"], dtype=np.float64)
                onsets = np.asarray(item["trial"]).reshape(-1).astype(int)
                labels = _label_to_binary(np.asarray(item["y"]).reshape(-1))
                if signal.ndim != 2 or onsets.size == 0 or labels.size == 0 or labels.size != onsets.size:
                    continue
                local_sfreq = float(np.asarray(item.get("fs", sfreq)).reshape(-1)[0])
                if signal.shape[0] < signal.shape[1]:
                    signal = signal.T
                n_samples, n_channels = signal.shape
                diffs = np.diff(np.sort(onsets))
                valid_diffs = diffs[diffs > 0]
                default_window = int(local_sfreq * 4)
                window = int(valid_diffs.min()) if valid_diffs.size else default_window
                window = max(int(local_sfreq * 2), min(window, default_window))
                extracted = []
                filtered_labels = []
                for onset, label in zip(onsets, labels):
                    start = int(onset)
                    stop = start + window
                    if start < 0 or stop > n_samples:
                        continue
                    extracted.append(signal[start:stop].T)
                    filtered_labels.append(int(label))
                if extracted:
                    ch_names = [f"EEG {idx + 1:02d}" for idx in range(n_channels)]
                    return np.asarray(extracted), np.asarray(filtered_labels), local_sfreq, ch_names

    if candidate_trials is None or candidate_labels is None:
        available_keys = ", ".join(list(clean.keys())[:8])
        raise ValueError(
            f"{source_name}: could not infer left/right trials and labels from MAT data. "
            f"Expected arrays like imagery_left/imagery_right or a 3D trial array plus labels. "
            f"Detected keys: {available_keys}"
        )

    label_count = np.asarray(candidate_labels).reshape(-1).shape[0]
    if candidate_trials.shape[0] == label_count:
        n_channels = candidate_trials.shape[1]
    elif candidate_trials.shape[1] == label_count:
        n_channels = candidate_trials.shape[0]
    elif candidate_trials.shape[2] == label_count:
        n_channels = candidate_trials.shape[1]
    else:
        n_channels = candidate_trials.shape[1]
    ch_names = [f"EEG {idx + 1:02d}" for idx in range(n_channels)]
    return candidate_trials, _label_to_binary(candidate_labels), sfreq, ch_names


def _load_mat_epochs(file_path: Path) -> tuple[mne.Epochs, None]:
    source_name = file_path.name
    try:
        data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        trials, labels, sfreq, ch_names = _extract_from_mat_dict(data, source_name)
        return _epochs_from_arrays(trials, labels, sfreq, ch_names, source_name), None
    except NotImplementedError:
        if h5py is None:
            raise ValueError(f"{source_name}: MATLAB v7.3 file detected, but h5py is not installed.")
        with h5py.File(file_path, "r") as handle:
            data = {key: np.array(handle[key]) for key in handle.keys()}
        trials, labels, sfreq, ch_names = _extract_from_mat_dict(data, source_name)
        return _epochs_from_arrays(trials, labels, sfreq, ch_names, source_name), None


def _load_dataset_from_path(file_path: Path) -> tuple[mne.Epochs, mne.io.BaseRaw | None]:
    suffix = file_path.suffix.lower()
    if suffix == ".edf":
        raw = _load_signal_file(file_path)
        return _prepare_motor_imagery_epochs(raw), raw
    if suffix == ".mat":
        return _load_mat_epochs(file_path)
    raise ValueError(f"Unsupported dataset format: {file_path.name}")

def _diagnostics(
    raw: mne.io.BaseRaw | None,
    epochs: mne.Epochs,
    results: dict,
    source: str,
    active_pipeline: str,
    comparison: dict | None = None,
    augmentation_summary: dict | None = None,
) -> dict:
    channels = _pick_channels(raw) if raw is not None else epochs.ch_names[: min(8, len(epochs.ch_names))]
    epoch_data = epochs.get_data()
    channel_rms = np.sqrt(np.mean(epoch_data[:, : len(channels), :] ** 2, axis=(0, 2))) * 1e6
    left_count = int(np.sum(epochs.events[:, -1] == 2))
    right_count = int(np.sum(epochs.events[:, -1] == 3))
    sampling_rate = float(raw.info["sfreq"]) if raw is not None else float(epochs.info["sfreq"])
    total_channels = len(raw.ch_names) if raw is not None else len(epochs.ch_names)
    pipeline_label = "GAN-Augmented" if active_pipeline == "gan_augmented" else "Baseline"

    session_cards = [
        {"label": "Acquisition Source", "value": source, "accent": "cyan"},
        {"label": "Analyzed Epochs", "value": str(len(epochs)), "accent": "blue"},
        {"label": "Active Pipeline", "value": pipeline_label, "accent": "green" if active_pipeline == "gan_augmented" else "blue"},
        {"label": "Sampling Rate", "value": f"{sampling_rate:.1f} Hz", "accent": "green"},
        {"label": "EEG Channels", "value": str(total_channels), "accent": "amber"},
    ]
    if augmentation_summary:
        session_cards.append({"label": "Synthetic Epochs", "value": str(augmentation_summary["synthetic_epochs"]), "accent": "green"})
    pipeline = [
        {"title": "Signal Intake", "body": "PhysioNet EDF sessions or MATLAB trial arrays are normalized into one EEG analysis workspace."},
        {"title": "Preprocessing", "body": "Bandpass filtering from 8-30 Hz plus epoch extraction isolates motor imagery activity before learning begins."},
        {"title": "Baseline Branch", "body": "The baseline model sends real EEG epochs directly into CSP feature extraction and a linear SVM classifier."},
        {"title": "GAN Branch", "body": "The augmented branch trains a GAN on real EEG epochs, generates synthetic trials, and merges them with real data before CSP and SVM."},
        {"title": "Comparative Review", "body": "Both pipelines are evaluated with the same held-out target labels so the dashboard can report measurable gains or regressions."},
    ]
    channel_cards = [
        {
            "name": name,
            "rms": round(float(rms), 2),
            "focus": "Motor strip" if name in {"C3", "C4", "Cz", "CP3", "CP4", "FCz"} else "Support channel",
        }
        for name, rms in zip(channels, channel_rms)
    ]
    notes = [
        f"Class balance remains stable with {left_count} left-hand and {right_count} right-hand epochs in the active session.",
        "The EEG monitor emphasizes central and fronto-central electrodes to mimic a clinical motor imagery review layout.",
        f"The active {pipeline_label.lower()} pipeline predicts {results['predicted_class']} with {results['confidence']:.1f}% confidence.",
        f"Validation performance is {results['accuracy']:.1f}% accuracy with {results['precision']:.1f}% precision and {results['recall']:.1f}% recall.",
    ]
    if comparison:
        notes.append(
            f"Comparison shows {comparison['best_model']} as the best model with an accuracy delta of {comparison['accuracy_difference']:+.1f}%."
        )
    else:
        notes.append("Run both pipelines to unlock side-by-side performance comparison and best-model highlighting.")
    if augmentation_summary:
        notes.append(
            f"The GAN branch added {augmentation_summary['synthetic_epochs']} synthetic epochs to the training pool after class-wise generation."
        )
        if augmentation_summary.get("backend") != "torch":
            notes.append("PyTorch GAN was unavailable in this environment, so NeuroSense used a statistical EEG augmentation fallback to keep the comparison workflow operational.")
    footer = {
        "body": "This system compares classical and augmentation-enhanced EEG decoding so the dashboard reads like a research benchmarking workspace rather than a single-model demo.",
    }
    family = _dataset_family(source, "")
    wizard = {
        "family": family,
        "headline": "Dataset Import Wizard",
        "description": "The system inspects incoming EDF or MAT files, maps them into a common EEG epoch structure, and routes the data into both baseline and GAN-ready evaluation branches.",
        "steps": [
            {
                "title": "Format Detection",
                "body": "Identify whether the session arrives as EDF signal recordings or MATLAB trial arrays and route it into the correct ingestion path.",
            },
            {
                "title": "Signal Normalization",
                "body": "Standardize sampling metadata, channel structure, and left-vs-right motor imagery labels into one shared representation.",
            },
            {
                "title": "Dual Pipeline Execution",
                "body": "Run the baseline CSP plus SVM branch, then optionally trigger GAN augmentation for comparative evaluation on the same dataset.",
            },
        ],
        "supported_formats": ["EDF", "MAT"],
        "recommendation": "EDF mode is best when you want event-aware raw recordings, while MAT mode is best when you already have segmented MATLAB trials ready for direct dual-pipeline classification.",
    }
    benefits = [
        {
            "title": "Why It Is Useful",
            "body": "It turns raw or pre-segmented EEG motor imagery data into a readable comparison dashboard that highlights the impact of data augmentation.",
        },
        {
            "title": "Main Purpose",
            "body": "It is built to classify imagined left-hand versus right-hand movement while contrasting baseline decoding against GAN-enhanced decoding.",
        },
        {
            "title": "Practical Advantage",
            "body": "Using both EDF and MAT support makes the system easier to use with lab recordings, benchmark datasets, and augmentation-driven research workflows.",
        },
    ]
    architecture = [
        {"title": "Data Acquisition", "body": "Load PhysioNet EDF sessions or MATLAB trial files as the entry point for motor imagery analysis."},
        {"title": "Preprocessing", "body": "Apply bandpass filtering, epoch extraction, and channel normalization before either model branch is executed."},
        {"title": "Baseline Pipeline", "body": "Run CSP feature extraction on real EEG epochs and classify the resulting features with a linear SVM."},
        {"title": "GAN-Augmented Pipeline", "body": "Train a fully connected GAN on the training epochs, synthesize new EEG trials, and retrain CSP plus SVM on the augmented set."},
        {"title": "Comparison Layer", "body": "Measure baseline versus GAN accuracy, precision, recall, confidence, and confusion matrices side by side."},
        {"title": "Visualization Platform", "body": "Present waveforms, spectral views, CSP maps, augmentation traces, logs, and exportable results in an interactive dashboard."},
    ]
    return {
        "session_cards": session_cards,
        "pipeline": pipeline,
        "channel_cards": channel_cards,
        "notes": notes,
        "footer": footer,
        "wizard": wizard,
        "benefits": benefits,
        "architecture": architecture,
    }


def _epoch_analytics(epochs: mne.Epochs) -> dict:
    data = epochs.get_data()
    sample = data[0]
    sfreq = float(epochs.info["sfreq"])
    channel_names = epochs.ch_names[: min(8, len(epochs.ch_names))]
    mean_channels = sample[: len(channel_names)]

    nperseg = min(128, mean_channels[0].shape[-1])
    noverlap = min(96, max(0, nperseg - 1))
    freq_bins, time_bins, spec = spectrogram(mean_channels[0], fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    freq_mask = freq_bins <= 45
    psd_freqs, psd = welch(sample, fs=sfreq, nperseg=min(256, sample.shape[-1]), axis=-1)
    band_power = []
    for label, (low, high) in BANDS.items():
        mask = (psd_freqs >= low) & (psd_freqs <= high)
        value = float(psd[:, mask].mean()) * 1e12 if np.any(mask) else 0.0
        band_power.append({"band": label, "value": round(value, 2)})

    return {
        "spectrogram": {
            "x": np.round(time_bins, 3).tolist(),
            "y": np.round(freq_bins[freq_mask], 2).tolist(),
            "z": np.round(spec[freq_mask] * 1e12, 3).tolist(),
            "bands": [{"label": "Mu", "range": [8, 12]}, {"label": "Beta", "range": [13, 30]}],
        },
        "band_power": band_power,
        "epoch_snapshot": {
            "channels": channel_names,
            "values": np.round(mean_channels, 4).tolist(),
        },
    }


def _prepare_epochs(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, mne.Epochs]:
    return raw, _prepare_motor_imagery_epochs(raw)


def _pipeline_title(pipeline_key: str) -> str:
    return "GAN-Augmented" if pipeline_key == "gan_augmented" else "Baseline"


def _svm_classifier() -> SVC:
    return SVC(kernel="linear", probability=True, random_state=CONFIG["random_seed"])


def _evaluate_classifier(name: str, classifier, X_train_csp, X_test_csp, y_train, y_test) -> dict:
    classifier.fit(X_train_csp, y_train)
    y_pred = classifier.predict(X_test_csp)
    probabilities = classifier.predict_proba(X_test_csp)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1], zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities[:, 1]) if len(np.unique(y_test)) == 2 else 0.0
    return {
        "name": name,
        "model": classifier,
        "predictions": y_pred,
        "probabilities": probabilities,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "f1": float(np.mean(f1)),
        "roc_auc": float(roc_auc),
        "per_class": [
            {"label": "Left", "precision": round(float(precision[0]) * 100, 1), "recall": round(float(recall[0]) * 100, 1), "f1": round(float(f1[0]) * 100, 1)},
            {"label": "Right", "precision": round(float(precision[1]) * 100, 1), "recall": round(float(recall[1]) * 100, 1), "f1": round(float(f1[1]) * 100, 1)},
        ],
    }


def _prepare_evaluation_bundle(train_epochs: mne.Epochs, test_epochs: mne.Epochs | None = None) -> dict:
    X = train_epochs.get_data()
    y = train_epochs.events[:, -1] - 2
    if test_epochs is None:
        class_counts = np.bincount(y)
        valid_counts = class_counts[class_counts > 0]
        if valid_counts.size and valid_counts.min() < 3:
            X_train, X_test, y_train, y_test = X, X, y, y
            evaluation_note = "Small-sample evaluation"
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=CONFIG["test_size"],
                random_state=CONFIG["random_seed"],
                stratify=y,
            )
            evaluation_note = "Random hold-out evaluation"
    else:
        X_train = X
        y_train = y
        X_test = test_epochs.get_data()
        y_test = test_epochs.events[:, -1] - 2
        evaluation_note = "Cross-subject evaluation"

    return {
        "X_full": X,
        "y_full": y,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "cv_splits": _safe_cv_splits(y),
        "evaluation_note": evaluation_note,
        "real_train_count": int(len(y_train)),
        "test_count": int(len(y_test)),
    }


def _evaluate_pipeline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_full: np.ndarray,
    y_full: np.ndarray,
    cv_splits: int,
    evaluation_note: str,
    train_epochs: mne.Epochs,
    pipeline_key: str,
    real_train_count: int,
    synthetic_count: int = 0,
) -> dict:
    csp = CSP(n_components=CONFIG["csp_components"], reg=None, log=True, norm_trace=False)
    X_train_csp = csp.fit_transform(X_train, y_train)
    X_test_csp = csp.transform(X_test)

    evaluation = _evaluate_classifier("SVM", _svm_classifier(), X_train_csp, X_test_csp, y_train, y_test)

    if evaluation_note == "Small-sample evaluation":
        cv_scores = np.asarray([evaluation["accuracy"]])
    else:
        pipeline = Pipeline(
            [
                ("csp", CSP(n_components=CONFIG["csp_components"], reg=None, log=True, norm_trace=False)),
                ("clf", _svm_classifier()),
            ]
        )
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=CONFIG["random_seed"])
        cv_scores = cross_val_score(pipeline, X_full, y_full, cv=cv, scoring="accuracy")

    sample_index = int(np.argmax(evaluation["probabilities"].max(axis=1)))
    predicted_label = int(np.argmax(evaluation["probabilities"][sample_index]))
    confidence = float(evaluation["probabilities"][sample_index][predicted_label])
    matrix = confusion_matrix(y_test, evaluation["predictions"], labels=[0, 1]).tolist()

    pattern_channels = train_epochs.ch_names[: min(8, len(train_epochs.ch_names))]
    csp_patterns = np.asarray(csp.patterns_)[: CONFIG["csp_components"], : len(pattern_channels)]

    artifact_summary = {
        "bad_channels": len(train_epochs.info.get("bads", [])),
        "dropped_epochs": int(np.sum([1 for item in train_epochs.drop_log if item])),
        "retained_epochs": int(len(y_train)),
        "real_epochs": int(real_train_count),
        "synthetic_epochs": int(synthetic_count),
        "test_epochs": int(len(y_test)),
    }

    joblib.dump(
        {
            "csp": csp,
            "classifier": evaluation["model"],
            "classifier_name": "SVM",
            "pipeline_name": pipeline_key,
            "config": CONFIG,
        },
        MODEL_PATH,
    )
    APP_STATE["saved_model_available"] = True

    return {
        "pipeline_key": pipeline_key,
        "pipeline_label": _pipeline_title(pipeline_key),
        "predicted_class": ["Left Hand Movement", "Right Hand Movement"][predicted_label],
        "confidence": round(confidence * 100, 1),
        "accuracy": round(float(evaluation["accuracy"]) * 100, 1),
        "cross_validation": round(float(cv_scores.mean()) * 100, 1),
        "precision": round(float(evaluation["precision"]) * 100, 1),
        "recall": round(float(evaluation["recall"]) * 100, 1),
        "f1_score": round(float(evaluation["f1"]) * 100, 1),
        "roc_auc": round(float(evaluation["roc_auc"]) * 100, 1),
        "confusion_matrix": matrix,
        "class_labels": ["Left", "Right"],
        "per_class": evaluation["per_class"],
        "benchmark": [],
        "selected_classifier": "SVM",
        "evaluation_note": evaluation_note,
        "csp_patterns": {
            "channels": pattern_channels,
            "components": [f"CSP {idx + 1}" for idx in range(csp_patterns.shape[0])],
            "values": np.round(csp_patterns, 3).tolist(),
        },
        "artifact_summary": artifact_summary,
    }


def _run_baseline_pipeline(train_epochs: mne.Epochs, test_epochs: mne.Epochs | None = None) -> dict:
    bundle = _prepare_evaluation_bundle(train_epochs, test_epochs=test_epochs)
    return _evaluate_pipeline(
        bundle["X_train"],
        bundle["y_train"],
        bundle["X_test"],
        bundle["y_test"],
        bundle["X_full"],
        bundle["y_full"],
        bundle["cv_splits"],
        bundle["evaluation_note"],
        train_epochs,
        pipeline_key="baseline",
        real_train_count=bundle["real_train_count"],
        synthetic_count=0,
    )


def _build_real_vs_synthetic_waveform(real_data: np.ndarray, synthetic_data: np.ndarray, epochs: mne.Epochs) -> dict:
    preferred = [name for name in ["C3", "Cz", "C4"] if name in epochs.ch_names]
    channels = preferred if preferred else epochs.ch_names[: min(3, len(epochs.ch_names))]
    picks = [epochs.ch_names.index(name) for name in channels]
    times = np.round(epochs.times, 4).tolist()
    real_mean = real_data[:, picks, :].mean(axis=0) * 1e6
    synthetic_mean = synthetic_data[:, picks, :].mean(axis=0) * 1e6
    offsets = np.arange(len(channels))[::-1] * 140.0
    traces = []
    for idx, name in enumerate(channels):
        traces.append(
            {
                "name": f"Real {name}",
                "x": times,
                "y": np.round(real_mean[idx] + offsets[idx], 3).tolist(),
                "color": "#3b82f6",
                "dash": "solid",
            }
        )
        traces.append(
            {
                "name": f"Synthetic {name}",
                "x": times,
                "y": np.round(synthetic_mean[idx] + offsets[idx], 3).tolist(),
                "color": "#22c55e",
                "dash": "dash",
            }
        )
    return {
        "traces": traces,
        "offsets": offsets.tolist(),
        "ticktext": channels,
    }


def _run_gan_augmented_pipeline(train_epochs: mne.Epochs, test_epochs: mne.Epochs | None = None) -> tuple[dict, dict]:
    bundle = _prepare_evaluation_bundle(train_epochs, test_epochs=test_epochs)
    real_train = bundle["X_train"]
    y_train = bundle["y_train"]

    synthetic_batches = []
    synthetic_labels = []
    class_breakdown = []
    loss_summary = []
    label_names = {0: "Left", 1: "Right"}

    for label in sorted(np.unique(y_train).tolist()):
        class_epochs = real_train[y_train == label]
        gan_model = train_gan(class_epochs)
        class_synthetic = generate_synthetic_data(gan_model, len(class_epochs))
        synthetic_batches.append(class_synthetic)
        synthetic_labels.append(np.full(len(class_synthetic), int(label), dtype=int))
        class_breakdown.append(
            {
                "label": label_names.get(int(label), str(label)),
                "real": int(len(class_epochs)),
                "synthetic": int(len(class_synthetic)),
            }
        )
        loss_summary.append(
            {
                "label": label_names.get(int(label), str(label)),
                "generator": round(float(gan_model.history["generator_loss"][-1]), 4),
                "discriminator": round(float(gan_model.history["discriminator_loss"][-1]), 4),
            }
        )
        if gan_model.backend != "torch":
            if len(class_epochs) < 4:
                _log("warning", "Small class sample detected. Using fallback EEG augmentation for the augmented pipeline.")
            else:
                _log("warning", "PyTorch GAN import failed on this machine. Using fallback EEG augmentation for the augmented pipeline.")

    synthetic_data = np.concatenate(synthetic_batches, axis=0)
    synthetic_y = np.concatenate(synthetic_labels, axis=0)
    augmented_train = augment_dataset(real_train, synthetic_data)
    augmented_labels = np.concatenate([y_train, synthetic_y], axis=0)

    results = _evaluate_pipeline(
        augmented_train,
        augmented_labels,
        bundle["X_test"],
        bundle["y_test"],
        bundle["X_full"],
        bundle["y_full"],
        bundle["cv_splits"],
        bundle["evaluation_note"],
        train_epochs,
        pipeline_key="gan_augmented",
        real_train_count=bundle["real_train_count"],
        synthetic_count=int(len(synthetic_y)),
    )
    augmentation_summary = {
        "available": True,
        "backend": gan_model.backend if synthetic_batches else "torch",
        "synthetic_epochs": int(len(synthetic_y)),
        "training_real_epochs": int(len(y_train)),
        "combined_training_epochs": int(len(augmented_labels)),
        "class_breakdown": class_breakdown,
        "loss_summary": loss_summary,
        "waveform": _build_real_vs_synthetic_waveform(real_train, synthetic_data, train_epochs),
    }
    return results, augmentation_summary


def _pipeline_benchmark(baseline_results: dict | None, gan_results: dict | None, active_results: dict) -> list[dict]:
    items = []
    if baseline_results:
        items.append(
            {
                "name": "Baseline",
                "accuracy": baseline_results["accuracy"],
                "precision": baseline_results["precision"],
                "recall": baseline_results["recall"],
                "f1": baseline_results["f1_score"],
            }
        )
    if gan_results:
        items.append(
            {
                "name": "GAN-Augmented",
                "accuracy": gan_results["accuracy"],
                "precision": gan_results["precision"],
                "recall": gan_results["recall"],
                "f1": gan_results["f1_score"],
            }
        )
    if not items:
        items.append(
            {
                "name": active_results["pipeline_label"],
                "accuracy": active_results["accuracy"],
                "precision": active_results["precision"],
                "recall": active_results["recall"],
                "f1": active_results["f1_score"],
            }
        )
    return items


def _empty_comparison(active_results: dict | None = None) -> dict:
    return {
        "available": False,
        "accuracy_difference": 0.0,
        "precision_difference": 0.0,
        "recall_difference": 0.0,
        "improvement_percentage": 0.0,
        "best_model": active_results["pipeline_label"] if active_results else "Awaiting Comparison",
    }


def _remember_dataset(
    raw: mne.io.BaseRaw | None,
    epochs: mne.Epochs,
    source: str,
    subject: str,
    train_subjects: list[int] | None = None,
    test_subject: int | None = None,
    file_count: int = 1,
    active_format: str = "EDF",
    test_epochs: mne.Epochs | None = None,
) -> None:
    APP_STATE["source"] = source
    APP_STATE["dataset"] = {
        "raw": raw,
        "epochs": epochs,
        "source": source,
        "subject": subject,
        "train_subjects": train_subjects or [],
        "test_subject": test_subject,
        "file_count": file_count,
        "active_format": active_format,
        "test_epochs": test_epochs,
    }
    APP_STATE["pipeline_results"] = {"baseline": None, "gan_augmented": None}
    APP_STATE["augmentation"] = None


def _build_payload(
    raw: mne.io.BaseRaw | None,
    epochs: mne.Epochs,
    source: str,
    subject: str,
    train_subjects: list[int] | None = None,
    test_subject: int | None = None,
    file_count: int = 1,
    active_format: str = "EDF",
    active_pipeline: str = "baseline",
    baseline_results: dict | None = None,
    gan_results: dict | None = None,
    augmentation_info: dict | None = None,
) -> dict:
    active_results = gan_results if active_pipeline == "gan_augmented" and gan_results else baseline_results
    if active_results is None:
        raise ValueError("No pipeline results are available for the active dataset.")

    comparison = _empty_comparison(active_results)
    if baseline_results and gan_results:
        comparison = {"available": True, **compare_models(baseline_results, gan_results)}

    results = {**active_results, "benchmark": _pipeline_benchmark(baseline_results, gan_results, active_results)}
    _log("success", f"{results['pipeline_label']} pipeline completed with {results['accuracy']}% accuracy.")
    _log("success", f"Model saved to {MODEL_PATH.name}.")
    counts = epochs.events[:, -1]
    left_count = int(np.sum(counts == 2))
    right_count = int(np.sum(counts == 3))
    waveform = _raw_to_waveform(raw) if raw is not None else _epochs_to_waveform(epochs)
    analytics = _epoch_analytics(epochs)
    details = _diagnostics(
        raw,
        epochs,
        results,
        source,
        active_pipeline=active_pipeline,
        comparison=comparison if comparison["available"] else None,
        augmentation_summary=augmentation_info,
    )
    summary = {
        "subject": subject,
        "status": f"{subject} | {left_count} left + {right_count} right epochs",
        "source": source,
        "sampling_rate": round(float(epochs.info["sfreq"]), 1),
        "channels": len(raw.ch_names) if raw is not None else len(epochs.ch_names),
        "epochs": len(epochs),
        "file_count": file_count,
        "active_format": active_format,
        "train_subjects": train_subjects or [],
        "test_subject": test_subject,
        "active_pipeline": results["pipeline_label"],
        "comparison_ready": comparison["available"],
    }
    baseline_accuracy = f"{baseline_results['accuracy']:.1f}%" if baseline_results else "Pending"
    gan_accuracy = f"{gan_results['accuracy']:.1f}%" if gan_results else "Pending"
    model_settings = {
        "Filter Band": f"{CONFIG['filter_band'][0]}-{CONFIG['filter_band'][1]} Hz",
        "Epoch Window": f"{CONFIG['epoch_window'][0]} s to {CONFIG['epoch_window'][1]} s",
        "CSP Components": str(CONFIG["csp_components"]),
        "Classifier": "Linear SVM",
        "Active Pipeline": results["pipeline_label"],
        "Baseline Accuracy": baseline_accuracy,
        "GAN Accuracy": gan_accuracy,
        "Augmentation Backend": augmentation_info["backend"].title() if augmentation_info else "Inactive",
        "Train/Test Split": f"{int((1 - CONFIG['test_size']) * 100)}/{int(CONFIG['test_size'] * 100)}",
        "Random Seed": str(CONFIG["random_seed"]),
    }
    synthetic_epochs = augmentation_info["synthetic_epochs"] if augmentation_info else 0
    training_summary = [
        {"label": "Files Used", "value": str(file_count)},
        {"label": "Real Epochs", "value": str(len(epochs))},
        {"label": "Synthetic Epochs", "value": str(synthetic_epochs)},
        {"label": "Class Balance", "value": f"{left_count} left / {right_count} right"},
        {"label": "Active Format", "value": active_format},
        {"label": "Comparison Ready", "value": "Yes" if comparison["available"] else "No"},
    ]
    about_dataset = [
        {"mode": "EDF Mode", "body": "EDF mode works with raw EEG recordings that already contain event annotations for left and right motor imagery."},
        {"mode": "MAT Mode", "body": "MAT mode works with MATLAB trial arrays and labels, making it ideal for pre-segmented EEG sessions from research workflows."},
        {"mode": "Dual Pipeline Mode", "body": "The dashboard can benchmark a baseline CSP plus SVM branch against a GAN-augmented CSP plus SVM branch."},
    ]
    APP_STATE["recent_runs"] = (
        [
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "source": source,
                "format": active_format,
                "accuracy": results["accuracy"],
                "classifier": results["pipeline_label"],
                "subject": subject,
            }
        ]
        + APP_STATE["recent_runs"]
    )[:6]
    payload = {
        "summary": summary,
        "waveform": waveform,
        "analytics": analytics,
        "results": results,
        "pipeline_results": {"baseline": baseline_results, "gan_augmented": gan_results},
        "comparison": comparison,
        "augmentation": augmentation_info
        or {
            "available": False,
            "backend": "none",
            "synthetic_epochs": 0,
            "training_real_epochs": 0,
            "combined_training_epochs": 0,
            "class_breakdown": [],
            "loss_summary": [],
            "waveform": {"traces": [], "offsets": [], "ticktext": []},
        },
        "details": details,
        "model_settings": model_settings,
        "training_summary": training_summary,
        "about_dataset": about_dataset,
        "logs": APP_STATE["logs"],
        "recent_runs": APP_STATE["recent_runs"],
        "project_identity": {
            "Project": "NeuroSense",
            "Student": "Final Year Research Team",
            "Guide": "Academic Project Guide",
            "Department": "Electronics / Biomedical Engineering",
            "Institution": "Engineering Institute",
        },
        "footer_meta": {
            "version": "v2.0",
            "mode": active_format,
            "line": "NeuroSense EEG Motor Imagery Intelligence Dashboard",
        },
        "mat_help": {
            "headline": "MAT Upload Help",
            "examples": [
                "imagery_left + imagery_right + srate",
                "data (trials x channels x samples) + labels + sfreq",
                "session.data + session.labels + session.sfreq",
            ],
        },
    }
    APP_STATE["source"] = source
    APP_STATE["summary"] = summary
    APP_STATE["last_payload"] = payload
    REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _build_payload_from_state(active_pipeline: str | None = None) -> dict:
    dataset = APP_STATE.get("dataset")
    if not dataset:
        raise ValueError("Load PhysioNet data or upload EEG data before running the model.")
    selected_pipeline = active_pipeline or ("gan_augmented" if APP_STATE["pipeline_results"]["gan_augmented"] else "baseline")
    return _build_payload(
        dataset["raw"],
        dataset["epochs"],
        source=dataset["source"],
        subject=dataset["subject"],
        train_subjects=dataset["train_subjects"],
        test_subject=dataset["test_subject"],
        file_count=dataset["file_count"],
        active_format=dataset["active_format"],
        active_pipeline=selected_pipeline,
        baseline_results=APP_STATE["pipeline_results"]["baseline"],
        gan_results=APP_STATE["pipeline_results"]["gan_augmented"],
        augmentation_info=APP_STATE["augmentation"],
    )


def load_physionet_subject(subject: int = 1) -> dict:
    _start_run_log()
    _log("info", f"EDF mode selected for subject S{subject:03d}.")
    paths = [str(_subject_dir(subject) / f"S{subject:03d}R{run:02d}.edf") for run in MI_RUNS]
    if not all(Path(path).exists() for path in paths):
        _log("info", "Local PhysioNet files missing. Requesting MNE download.")
        paths = [str(Path(item)) for item in eegbci.load_data(subject, runs=MI_RUNS)]
    _log("success", f"Loaded {len(paths)} EDF files.")
    raws = [mne.io.read_raw_edf(path, preload=True, verbose="ERROR") for path in paths]
    raw = mne.concatenate_raws(raws)
    raw, epochs = _prepare_epochs(raw)
    _log("success", f"Extracted {len(epochs)} epochs from subject S{subject:03d}.")
    _remember_dataset(raw, epochs, source="PhysioNet EEGBCI", subject=f"Subject {subject:03d}", train_subjects=[subject], active_format="EDF", file_count=len(paths))
    _log("info", "Running baseline CSP plus SVM pipeline.")
    APP_STATE["pipeline_results"]["baseline"] = _run_baseline_pipeline(epochs)
    return _build_payload_from_state("baseline")


def _load_subject_epochs(subject: int) -> tuple[mne.io.BaseRaw, mne.Epochs]:
    paths = [str(_subject_dir(subject) / f"S{subject:03d}R{run:02d}.edf") for run in MI_RUNS]
    if not all(Path(path).exists() for path in paths):
        paths = [str(Path(item)) for item in eegbci.load_data(subject, runs=MI_RUNS)]
    raws = [mne.io.read_raw_edf(path, preload=True, verbose="ERROR") for path in paths]
    raw = mne.concatenate_raws(raws)
    return _prepare_epochs(raw)


def load_physionet_multi_subject(train_subjects: list[int], test_subject: int) -> dict:
    _start_run_log()
    _log("info", f"Multi-subject EDF mode selected. Train: {train_subjects}, Test: S{test_subject:03d}.")
    train_items = []
    file_count = 0
    for subject in train_subjects:
        raw, epochs = _load_subject_epochs(subject)
        file_count += len(MI_RUNS)
        train_items.append((f"S{subject:03d}", epochs, raw))
        _log("success", f"Prepared training subject S{subject:03d} with {len(epochs)} epochs.")
    combined = _combine_epochs(train_items)
    _test_raw, test_epochs = _load_subject_epochs(test_subject)
    file_count += len(MI_RUNS)
    _log("success", f"Prepared held-out test subject S{test_subject:03d} with {len(test_epochs)} epochs.")
    subject_label = f"Train {', '.join([f'S{s:03d}' for s in train_subjects])} | Test S{test_subject:03d}"
    _remember_dataset(
        combined["raw"],
        combined["epochs"],
        source="PhysioNet EEGBCI",
        subject=subject_label,
        train_subjects=train_subjects,
        test_subject=test_subject,
        active_format="EDF",
        file_count=file_count,
        test_epochs=test_epochs,
    )
    _log("info", "Running baseline CSP plus SVM pipeline.")
    APP_STATE["pipeline_results"]["baseline"] = _run_baseline_pipeline(combined["epochs"], test_epochs=test_epochs)
    return _build_payload_from_state("baseline")


def generate_synthetic_dataset() -> dict:
    _start_run_log()
    _log("info", "Generating synthetic EEG demo session.")
    sfreq = 160.0
    seconds = 36
    samples = int(sfreq * seconds)
    ch_names = ["FCz", "C3", "Cz", "C4", "CP3", "CPz", "CP4", "Pz"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    times = np.arange(samples) / sfreq
    rng = np.random.default_rng(42)
    data = []
    for idx, name in enumerate(ch_names):
        phase = idx * 0.3
        carrier = 12e-6 * np.sin(2 * np.pi * 10 * times + phase)
        beta = 8e-6 * np.sin(2 * np.pi * 22 * times + phase / 2)
        modulation = np.where((times % 8) < 4, 1.2, 0.7)
        asymmetry = 1.25 if name in {"C4", "CP4"} else 0.8 if name in {"C3", "CP3"} else 1.0
        noise = rng.normal(scale=2.5e-6, size=samples)
        data.append((carrier + beta * modulation * asymmetry + noise).astype(np.float64))
    raw = mne.io.RawArray(np.vstack(data), info, verbose="ERROR")
    raw.set_montage("standard_1005", on_missing="ignore")

    annotations = []
    for start in np.arange(2, seconds - 4, 4):
        label = "T1" if int(start / 4) % 2 == 0 else "T2"
        annotations.append((start, 3.0, label))
    raw.set_annotations(
        mne.Annotations(
            onset=[item[0] for item in annotations],
            duration=[item[1] for item in annotations],
            description=[item[2] for item in annotations],
        )
    )
    raw, epochs = _prepare_epochs(raw)
    _log("success", f"Synthetic session created with {len(epochs)} epochs.")
    _remember_dataset(raw, epochs, source="Synthetic EEG Generator", subject="Demo Subject", active_format="EDF")
    _log("info", "Running baseline CSP plus SVM pipeline.")
    APP_STATE["pipeline_results"]["baseline"] = _run_baseline_pipeline(epochs)
    return _build_payload_from_state("baseline")


def load_uploaded_bundle(files) -> dict:
    _start_run_log()
    dataset_items = []
    temp_paths = []
    try:
        for file_storage in files:
            suffix = Path(file_storage.filename or "").suffix.lower()
            if not suffix:
                continue
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                file_storage.save(tmp.name)
                temp_path = Path(tmp.name)
                temp_paths.append(temp_path)

            epochs, raw = _load_dataset_from_path(temp_path)
            dataset_items.append((file_storage.filename, epochs, raw))
            _log("success", f"Accepted {file_storage.filename} with {len(epochs)} epochs.")

        if not dataset_items:
            raise ValueError("Upload at least one supported file: EDF or MAT.")

        combined = _combine_epochs(dataset_items)
        source = " + ".join(sorted({Path(name).suffix.lower().lstrip('.') or "data" for name, _, _ in dataset_items}))
        subject = f"Uploaded Bundle ({len(dataset_items)} file{'s' if len(dataset_items) != 1 else ''})"
        active_format = "MAT" if all(Path(name).suffix.lower() == ".mat" for name, _, _ in dataset_items) else "EDF"
        _remember_dataset(
            combined["raw"],
            combined["epochs"],
            source=f"User Uploads [{source.upper()}]",
            subject=subject,
            active_format=active_format,
            file_count=len(dataset_items),
        )
        _log("info", "Running baseline CSP plus SVM pipeline.")
        APP_STATE["pipeline_results"]["baseline"] = _run_baseline_pipeline(combined["epochs"])
        return _build_payload_from_state("baseline")
    finally:
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)


def _run_baseline_on_current_dataset() -> dict:
    dataset = APP_STATE.get("dataset")
    if not dataset:
        raise ValueError("Load PhysioNet data or generate synthetic EEG first.")
    train_epochs = dataset["epochs"].copy().load_data()
    test_epochs = dataset["test_epochs"].copy().load_data() if dataset["test_epochs"] is not None else None
    _log("info", "Executing baseline pipeline: EEG -> CSP -> SVM.")
    APP_STATE["pipeline_results"]["baseline"] = _run_baseline_pipeline(train_epochs, test_epochs=test_epochs)
    return _build_payload_from_state("baseline")


def _run_gan_on_current_dataset() -> dict:
    dataset = APP_STATE.get("dataset")
    if not dataset:
        raise ValueError("Load PhysioNet data or generate synthetic EEG first.")
    train_epochs = dataset["epochs"].copy().load_data()
    test_epochs = dataset["test_epochs"].copy().load_data() if dataset["test_epochs"] is not None else None
    if APP_STATE["pipeline_results"]["baseline"] is None:
        _log("info", "Baseline results missing. Running baseline first for a fair comparison.")
        APP_STATE["pipeline_results"]["baseline"] = _run_baseline_pipeline(train_epochs, test_epochs=test_epochs)
    _log("info", "Executing GAN-augmented pipeline: EEG -> GAN -> CSP -> SVM.")
    gan_results, augmentation = _run_gan_augmented_pipeline(train_epochs, test_epochs=test_epochs)
    APP_STATE["pipeline_results"]["gan_augmented"] = gan_results
    APP_STATE["augmentation"] = augmentation
    return _build_payload_from_state("gan_augmented")


def _compare_current_models() -> dict:
    baseline_results = APP_STATE["pipeline_results"]["baseline"]
    gan_results = APP_STATE["pipeline_results"]["gan_augmented"]
    if not baseline_results or not gan_results:
        raise ValueError("Run both the baseline model and the GAN-augmented model before comparing results.")
    comparison = compare_models(baseline_results, gan_results)
    _log("info", "Compiling dual-pipeline comparison metrics.")
    _log("success", f"Best model identified: {comparison['best_model']}.")
    preferred_pipeline = "gan_augmented" if comparison["best_model"] == "GAN-Augmented" else "baseline"
    return _build_payload_from_state(preferred_pipeline)


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    return jsonify(APP_STATE["summary"])


@app.post("/api/load-physionet")
def api_load_physionet():
    body = request.get_json(silent=True) or {}
    subject = int(body.get("subject", CONFIG["subject_default"]))
    format_mode = str(body.get("format_mode", "edf")).lower()
    train_subjects = [int(item) for item in body.get("train_subjects", [subject])]
    test_subject = body.get("test_subject")
    try:
        if format_mode == "mat":
            return jsonify({"ok": False, "error": "MAT mode expects uploaded MATLAB files. Use the MAT upload option to load data."}), 400
        if test_subject is not None:
            return jsonify({"ok": True, "payload": load_physionet_multi_subject(train_subjects, int(test_subject))})
        return jsonify({"ok": True, "payload": load_physionet_subject(subject)})
    except Exception as exc:
        return jsonify({"ok": False, "error": f"PhysioNet loading failed: {exc}"}), 500


@app.post("/api/generate-synthetic")
def api_generate_synthetic():
    try:
        return jsonify({"ok": True, "payload": generate_synthetic_dataset()})
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Synthetic data generation failed: {exc}"}), 500


@app.post("/api/demo-mode")
def api_demo_mode():
    try:
        try:
            payload = load_physionet_subject(1)
        except Exception:
            payload = generate_synthetic_dataset()
        return jsonify({"ok": True, "payload": payload})
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Demo mode failed: {exc}"}), 500


@app.post("/api/upload-datasets")
def api_upload_datasets():
    files = [item for item in request.files.getlist("files") if item and item.filename]
    if not files:
        return jsonify({"ok": False, "error": "Upload at least one EDF or MAT file."}), 400
    try:
        return jsonify({"ok": True, "payload": load_uploaded_bundle(files)})
    except Exception as exc:
        message = f"Dataset upload failed: {exc}"
        if any(Path(item.filename).suffix.lower() == ".mat" for item in files):
            message += " | MAT examples: imagery_left + imagery_right + srate, or data + labels + sfreq, or session.data + session.labels + session.sfreq."
        return jsonify({"ok": False, "error": message}), 500


@app.post("/api/run-classification")
def api_run_classification():
    try:
        _start_run_log()
        return jsonify({"ok": True, "payload": _run_baseline_on_current_dataset()})
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Classification failed: {exc}"}), 500


@app.post("/api/run-baseline-model")
def api_run_baseline_model():
    try:
        _start_run_log()
        return jsonify({"ok": True, "payload": _run_baseline_on_current_dataset()})
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Baseline model failed: {exc}"}), 500


@app.post("/api/run-gan-augmented-model")
def api_run_gan_augmented_model():
    try:
        _start_run_log()
        return jsonify({"ok": True, "payload": _run_gan_on_current_dataset()})
    except Exception as exc:
        return jsonify({"ok": False, "error": f"GAN-augmented model failed: {exc}"}), 500


@app.post("/api/compare-models")
def api_compare_models():
    try:
        _start_run_log()
        return jsonify({"ok": True, "payload": _compare_current_models()})
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Model comparison failed: {exc}"}), 500


@app.post("/api/reset-session")
def api_reset_session():
    APP_STATE["source"] = "none"
    APP_STATE["dataset"] = None
    APP_STATE["summary"] = {"status": "No EEG dataset loaded."}
    APP_STATE["logs"] = []
    APP_STATE["last_payload"] = None
    APP_STATE["pipeline_results"] = {"baseline": None, "gan_augmented": None}
    APP_STATE["augmentation"] = None
    APP_STATE["recent_runs"] = []
    APP_STATE["saved_model_available"] = False
    return jsonify({"ok": True})


@app.get("/api/export-results.csv")
def api_export_results():
    payload = APP_STATE.get("last_payload")
    if not payload:
        return jsonify({"ok": False, "error": "Run a session before exporting results."}), 400
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Section", "Metric", "Value"])
    for model_name, results in payload.get("pipeline_results", {}).items():
        if not results:
            continue
        label = "GAN-Augmented" if model_name == "gan_augmented" else "Baseline"
        writer.writerow([label, "Accuracy", results["accuracy"]])
        writer.writerow([label, "Confidence", results["confidence"]])
        writer.writerow([label, "Precision", results["precision"]])
        writer.writerow([label, "Recall", results["recall"]])
        writer.writerow([label, "F1 Score", results["f1_score"]])
        writer.writerow([label, "ROC AUC", results["roc_auc"]])
        writer.writerow([label, "Cross Validation", results["cross_validation"]])
        writer.writerow([label, "Predicted Class", results["predicted_class"]])
        writer.writerow([label, "Confusion Matrix", ""])
        for row in results["confusion_matrix"]:
            writer.writerow([label, "Matrix Row", " | ".join(str(item) for item in row)])
    comparison = payload.get("comparison", {})
    if comparison.get("available"):
        writer.writerow(["Comparison", "Best Model", comparison["best_model"]])
        writer.writerow(["Comparison", "Accuracy Difference", comparison["accuracy_difference"]])
        writer.writerow(["Comparison", "Precision Difference", comparison["precision_difference"]])
        writer.writerow(["Comparison", "Recall Difference", comparison["recall_difference"]])
        writer.writerow(["Comparison", "Improvement Percentage", comparison["improvement_percentage"]])
    elif payload.get("results"):
        writer.writerow(["Active Pipeline", "Accuracy", payload["results"]["accuracy"]])
        writer.writerow(["Active Pipeline", "Confidence", payload["results"]["confidence"]])
        writer.writerow(["Active Pipeline", "Predicted Class", payload["results"]["predicted_class"]])
        writer.writerow(["Active Pipeline", "Confusion Matrix", ""])
        for row in payload["results"]["confusion_matrix"]:
            writer.writerow(["Active Pipeline", "Matrix Row", " | ".join(str(item) for item in row)])
    data = io.BytesIO(buffer.getvalue().encode("utf-8"))
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="neurosense_results.csv")


@app.get("/api/load-saved-report")
def api_load_saved_report():
    if not REPORT_PATH.exists():
        return jsonify({"ok": False, "error": "No saved report is available yet."}), 404
    payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    APP_STATE["last_payload"] = payload
    APP_STATE["summary"] = payload.get("summary", APP_STATE["summary"])
    APP_STATE["source"] = payload.get("summary", {}).get("source", APP_STATE["source"])
    APP_STATE["recent_runs"] = payload.get("recent_runs", [])
    APP_STATE["pipeline_results"] = payload.get("pipeline_results", {"baseline": None, "gan_augmented": None})
    APP_STATE["augmentation"] = payload.get("augmentation")
    return jsonify({"ok": True, "payload": payload})


if __name__ == "__main__":
    app.run(debug=True)
