"""Time-sync the recordings of one multi-device session from the 3 claps made at the start, all devices together.

For every recording in data/recordings/ it finds the 3-clap pattern in the first 150 s (the detector is
feasibility/tools/find_claps.py), then refines each device's offset to the reference device by
cross-correlating the clap waveforms (30 ms windows around the onsets, high-passed at 1.5 kHz). The offset
is the median over the three claps; the spread over the claps is reported (a clap that disagrees by more
than about 1 ms is a weak correlation, the median ignores it).

Writes hallway/sync_offsets.json:  file_time[device] = file_time[reference] + offset_s[device].
Devices were together for the claps, so the sound's travel time between them is ignored (about 1 ms per
34 cm of separation). Clock drift over the session is NOT corrected (there were no end claps).

Usage: sync_recordings.py [--ref iphoneair] [--out hallway/sync_offsets.json]
"""
import argparse
import json
import sys

import numpy as np
from scipy.signal import butter, correlate, correlation_lags, sosfiltfilt

import msio
sys.path.insert(0, str(msio.ROOT.parent / "feasibility" / "tools"))
import find_claps  # noqa: E402  (clap detector; importing it loads the default preset, which is harmless here)

SR = msio.SR
WINDOW_S = 150.0            # where to look for the start claps
HALF_S = 0.03               # half-width of the clap waveform window
MAXLAG_S = 0.012            # largest offset residual accepted from the cross-correlation


def find_start_claps(x):
    cand = find_claps.candidates(x[: int(WINDOW_S * SR)], SR)
    tri = find_claps.pick_triple(cand)
    if not tri:
        raise RuntimeError("no 3-clap pattern found")
    return sorted(tri)


def residual_lag(ya, ta, yb, tb):
    """How much later (s) the clap is in b than the onset times ta, tb say, by cross-correlating the waveforms."""
    n = int(HALF_S * SR)
    a, b = ya[int(ta * SR) - n:int(ta * SR) + n], yb[int(tb * SR) - n:int(tb * SR) + n]
    cc = correlate(b, a, "full")
    lags = correlation_lags(len(b), len(a), "full")
    ok = np.abs(lags) <= int(MAXLAG_S * SR)
    i = np.flatnonzero(ok)[np.argmax(cc[ok])]
    y0, y1, y2 = cc[i - 1], cc[i], cc[i + 1]
    return (lags[i] + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) / SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="iphoneair")
    ap.add_argument("--out", default=str(msio.HALLWAY / "sync_offsets.json"))
    args = ap.parse_args()
    files = {p.stem: p for p in sorted(msio.RECORDINGS.glob("*.m4a"))}
    if args.ref not in files:
        sys.exit(f"reference {args.ref} not in {msio.RECORDINGS}")
    hp = butter(4, 1500, "highpass", fs=SR, output="sos")
    claps, filt, dur = {}, {}, {}
    for name, path in files.items():
        x = msio.load_mono(path)
        dur[name] = len(x) / SR
        claps[name] = [t for t, _, _ in find_start_claps(x)]
        gaps = np.diff(claps[name])
        print(f"{name:10s} {dur[name]:8.1f} s  claps at {', '.join(f'{t:.3f}' for t in claps[name])}  gaps {gaps.round(3)}")
        filt[name] = sosfiltfilt(hp, x[: int(40 * SR)].astype(np.float64))
    offsets, spread = {args.ref: 0.0}, {args.ref: 0.0}
    for name in files:
        if name == args.ref:
            continue
        per = [claps[name][k] - claps[args.ref][k]
               + residual_lag(filt[args.ref], claps[args.ref][k], filt[name], claps[name][k]) for k in range(3)]
        offsets[name] = round(float(np.median(per)), 4)
        spread[name] = round(float((max(per) - min(per)) * 1e3), 2)
        print(f"{name:10s} offset {offsets[name]:.4f} s   per clap {np.round(per, 4).tolist()}  spread {spread[name]} ms")
    json.dump(dict(
        description=("Start-clap time sync of the recordings in data/recordings/. file_time[device] = file_time[reference] "
                     "+ offset_s[device]: a sound at reference file time t is at t + offset in the other file. Devices were "
                     "together for the claps, so travel time between them is ignored. No drift correction."),
        reference=args.ref, offset_s=offsets,
        offset_spread_over_3_claps_ms=spread,
        start_claps_file_time_s={k: [round(t, 3) for t in v] for k, v in claps.items()},
        durations_s={k: round(v, 3) for k, v in dur.items()}), open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
