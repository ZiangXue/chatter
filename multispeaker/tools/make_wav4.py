"""Make the clock-synced multi-channel WAV from the original recordings and hallway/sync_offsets.json.

Channel c, sample i  =  recording c, sample i + round(offset * 48000); time zero is the start of the
reference device's file. The WAV ends where the shortest recording (or the cut you give with --end) ends.
Output: 16-bit, 48 kHz, one channel per device in --order. A stereo recording is mixed to mono (plain
average of left and right). Offsets are rounded to whole samples (at most 10 microseconds of error). Clock
drift is not corrected.

The WAV is about 2.3x the size of the recordings, so it is not committed: run this to get it.

Usage: make_wav4.py --end Macbook:1519.45 [--order iphone8,ipad,Macbook,iphoneair] [--out data/audio_4ch.wav]
  --end DEVICE:SECONDS   cut the WAV where that device's recording stops being usable (seconds in that device's file)
"""
import argparse
import json

import numpy as np
from scipy.io import wavfile

import msio

SR = msio.SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", default="iphone8,ipad,Macbook,iphoneair")
    ap.add_argument("--end", required=True, help="DEVICE:SECONDS in that device's own file")
    ap.add_argument("--offsets", default=str(msio.HALLWAY / "sync_offsets.json"))
    ap.add_argument("--out", default=str(msio.DATA / "audio_4ch.wav"))
    ap.add_argument("--meta", default=str(msio.HALLWAY / "audio_4ch.json"))
    args = ap.parse_args()
    order = args.order.split(",")
    sync = json.load(open(args.offsets))
    off = sync["offset_s"]
    end_dev, end_s = args.end.split(":")
    end_s = float(end_s)
    ends_ref = {d: sync["durations_s"][d] - off[d] for d in order}            # where each file ends, in reference time
    t_end = min(min(ends_ref.values()), end_s - off[end_dev])
    n = int(round(t_end * SR))
    shifts = {d: int(round(off[d] * SR)) for d in order}
    print(f"length {n / SR:.3f} s ({n} samples); sample shifts {shifts}")
    out = np.zeros((n, len(order)), np.int16)
    for c, d in enumerate(order):
        x = msio.load_mono(msio.RECORDINGS / f"{d}.m4a")[shifts[d]:shifts[d] + n]
        assert len(x) == n, (d, len(x), n)
        out[:, c] = np.clip(np.rint(x * 32768.0), -32768, 32767).astype(np.int16)
        print(f"  {d:10s} peak {np.abs(x).max():.3f}  clipped samples {int((np.abs(out[:, c]) >= 32767).sum())}")
    wavfile.write(args.out, SR, out)
    try:
        positions = json.load(open(msio.HALLWAY / "devices.json"))["positions_m"]
    except OSError:
        positions = None
    json.dump(dict(file=args.out.split("/")[-1], sample_rate=SR, n_samples=n, duration_s=n / SR, channels=order,
                   note=("time zero = start of the reference file; channel sample i = device file sample i + sample_shift "
                         "(offsets from the start claps, no drift correction); stereo recordings averaged to mono"),
                   reference=sync["reference"], sample_shifts=shifts, offset_s={d: off[d] for d in order},
                   end=f"{end_dev} file time {end_s} s (where that recording stopped being usable)",
                   positions_m=positions), open(args.meta, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
