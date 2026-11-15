"""Shared helpers for the multispeaker tools: paths and reading one recording as mono float samples."""
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # multispeaker/
DATA = ROOT / "data"
RECORDINGS = DATA / "recordings"
HALLWAY = ROOT / "hallway"
SR = 48000


def n_channels(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                          "stream=channels", "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return int(out.strip())


def load_mono(path, sr=SR):
    """Samples of a recording as float32 in [-1, 1]. A stereo file is mixed to mono by the plain average
    of its two channels (ffmpeg's own mono mix adds gain and can clip)."""
    ch = n_channels(path)
    if ch not in (1, 2):
        raise ValueError(f"{path}: {ch} channels, expected 1 or 2")
    raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", str(path), "-map", "0:a:0", "-f", "f32le",
                          "-ac", str(ch), "-ar", str(sr), "-"], capture_output=True).stdout
    x = np.frombuffer(raw, np.float32)
    if ch == 2:
        x = x.reshape(-1, 2).astype(np.float64).mean(axis=1).astype(np.float32)
    return x
