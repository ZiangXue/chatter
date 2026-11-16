"""Time-sync the camera video to the phone recordings from the 3 start claps heard in its audio track.

Same method as sync_recordings.py (3-clap detector, waveform cross-correlation, median over the claps),
with the video's audio as one more device. Also checks the three mid-session claps (about 977-979 s):
if the video and the reference device agree on them to a few ms, drift between them is small.

Writes hallway/video_sync.json:  video_time = iphoneair_file_time + offset_s  (a sound at reference
file time t is at video time t + offset_s). Video frame k is at time k / fps of the video (ffprobe
frame timestamps are used by track_video.py).

Usage: sync_video.py [--video data/Video.mov] [--ref iphoneair]
"""
import argparse
import json
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

import msio
import sync_recordings as sr_

SR = msio.SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=str(msio.DATA / "Video.mov"))
    ap.add_argument("--ref", default="iphoneair")
    ap.add_argument("--out", default=str(msio.HALLWAY / "video_sync.json"))
    args = ap.parse_args()
    v = msio.load_mono(args.video)
    r = msio.load_mono(msio.RECORDINGS / f"{args.ref}.m4a")
    print(f"video audio {len(v) / SR:.2f} s, {args.ref} {len(r) / SR:.2f} s")
    cv_, cr = [t for t, _, _ in sr_.find_start_claps(v)], [t for t, _, _ in sr_.find_start_claps(r)]
    print("video claps", np.round(cv_, 3), " ref claps", np.round(cr, 3), " gaps", np.diff(cv_).round(3), np.diff(cr).round(3))
    hp = butter(4, 1500, "highpass", fs=SR, output="sos")
    fv, fr = sosfiltfilt(hp, v[: 40 * SR].astype(np.float64)), sosfiltfilt(hp, r[: 40 * SR].astype(np.float64))
    per = [cv_[k] - cr[k] + sr_.residual_lag(fr, cr[k], fv, cv_[k]) for k in range(3)]
    off = float(np.median(per))
    print(f"offset {off:.4f} s, per clap {np.round(per, 4).tolist()}, spread {(max(per) - min(per)) * 1e3:.2f} ms")
    # mid-session claps: where does each see them?  (search near the known time, offset applied)
    mid = []
    for t in (976.99, 978.22, 979.44):
        a = int((t - 0.3) * SR); b = int((t + 0.3) * SR)
        ya = sosfiltfilt(hp, r[a:b].astype(np.float64))
        yb = sosfiltfilt(hp, v[int(a + off * SR):int(b + off * SR)].astype(np.float64))
        i, j = int(np.argmax(np.abs(ya))), int(np.argmax(np.abs(yb)))      # loudest peak of each, then waveform xcorr
        mid.append(sr_.residual_lag(ya, i / SR, yb, j / SR) + (j - i) / SR)
    print("mid-session claps: video minus reference-expected, ms:", np.round(np.array(mid) * 1e3, 2).tolist())
    json.dump(dict(
        description=("video_time = reference_file_time + offset_s; from the 3 start claps heard in the video's audio track, "
                     "median of the 3. mid_session_residual_ms: where the video hears the mid-session claps relative to this "
                     "offset (drift check; includes the arrival-time differences between camera and reference phone)."),
        reference=args.ref, offset_s=round(off, 4),
        offset_per_clap_s=[round(p, 4) for p in per], spread_ms=round((max(per) - min(per)) * 1e3, 2),
        video_claps_s=[round(t, 3) for t in cv_], reference_claps_s=[round(t, 3) for t in cr],
        mid_session_residual_ms=[round(m * 1e3, 2) for m in mid], video_duration_s=round(len(v) / SR, 3)),
        open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
