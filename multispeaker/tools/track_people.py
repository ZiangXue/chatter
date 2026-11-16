"""Where are the two speakers on the floor over time? Video -> pose detections -> speakers A and B -> floor positions.

Stage 1 (--detect, slow, ~20 min): runs the YOLO pose model (feasibility/models/yolo11m-pose.pt) on every
STRIDE-th frame of data/Video.mov and caches every detected person (box, 17 keypoints, thigh colour) in
data/video_detections.pkl. Frames are mirrored in the file: they are flipped back and scaled to the
1920x1080 frame of hallway/grid_hallway.json, so the floor homography applies unchanged.
Stage 2 (default, fast): reads the cache and writes the distributable files
  hallway/positions.csv        one row per processed video frame (10 fps)
  hallway/positions.json       what the columns mean: time base, frame, speakers, method, quality
Speakers are told apart by the colour of their trousers (k-means, 2 clusters, on the thigh colour of every
detection): A = the darker trousers, B = the lighter. The files contain no images and no identities.

Position of a person = mean of the ankle keypoints (confidence >= 0.35), lowered to the floor by 4.5 % of the
person's ankle-to-nose height (ankle joints are about 8 cm above the floor), through the floor homography.
Raw positions are kept; "_s" columns are smoothed (Savitzky-Golay, 1 s, per unbroken stretch, gaps up to 0.6 s
filled).

Time base: t_wav_s = video time - video_sync.json offset = time in data/audio_4ch.wav (zero = start of the iPhone
Air file, which is where make_wav4.py starts the WAV). Only 0 <= t_wav_s <= length of the WAV is processed.

Usage: track_people.py --detect [--stride 6] [--device mps]     then     track_people.py
"""
import argparse
import json
import pickle
import subprocess
import time

import cv2
import numpy as np
from scipy.signal import savgol_filter

import msio

IN = 0.0254
VIDEO = msio.DATA / "Video.mov"
CACHE = msio.DATA / "video_detections.pkl"
MODEL = msio.ROOT.parent / "feasibility" / "models" / "yolo11m-pose.pt"
ANKLE_CONF = 0.35
ANKLE_LIFT_M, BODY_M = 0.08, 1.75           # ankle joint height above the floor; ankle-to-nose height of an adult
GAP_FILL_S, SEG_BREAK_S, SMOOTH_S = 0.6, 1.0, 1.0
GATE_M = 1.0                                # a person does not move more than this between processed frames (10 fps)
NOSE, LHIP, RHIP, LKNEE, RKNEE, LANK, RANK = 0, 11, 12, 13, 14, 15, 16


def frame_times(path):
    out = subprocess.run(["ffprobe", "-v", "quiet", "-select_streams", "v", "-show_entries", "packet=pts_time",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True).stdout
    return np.sort(np.array([float(x.strip(",")) for x in out.split() if x.strip(",")]))


def thigh_lab(img, kxy, kc):
    """Median Lab colour of the thighs (between hip and knee), or None if not visible."""
    pts = []
    for h, k in ((LHIP, LKNEE), (RHIP, RKNEE)):
        if kc[h] > 0.4 and kc[k] > 0.4:
            L = np.linalg.norm(kxy[h] - kxy[k])
            for s in np.linspace(0.3, 0.7, 5):
                c = kxy[h] + (kxy[k] - kxy[h]) * s
                r = max(2, int(0.1 * L))
                x0, y0 = int(c[0]) - r, int(c[1]) - r
                if x0 >= 0 and y0 >= 0 and x0 + 2 * r < img.shape[1] and y0 + 2 * r < img.shape[0]:
                    pts.append(img[y0:y0 + 2 * r, x0:x0 + 2 * r].reshape(-1, 3))
    if not pts:
        return None
    return np.median(cv2.cvtColor(np.concatenate(pts)[None].astype(np.uint8), cv2.COLOR_BGR2Lab)[0], axis=0)


def detect(args):
    from ultralytics import YOLO
    sync = json.load(open(msio.HALLWAY / "video_sync.json"))
    t_end = json.load(open(msio.HALLWAY / "audio_4ch.json"))["duration_s"] + sync["offset_s"]
    times = frame_times(VIDEO)
    model = YOLO(str(MODEL))
    cap = cv2.VideoCapture(str(VIDEO))
    n_dec = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{len(times)} packets over {times[-1]:.1f} s, decoder {n_dec} frames; processing video 0-{t_end:.1f} s, stride {args.stride}", flush=True)
    rows, i, t0 = [], -1, time.time()
    while True:
        if not cap.grab():
            break
        i += 1
        if i >= len(times) or times[i] > t_end:
            break
        if i % args.stride:
            continue
        ok, fr = cap.retrieve()
        if not ok:
            continue
        fr = cv2.flip(fr, 1)                                   # the file is mirrored
        res = model.predict(fr, imgsz=1280, conf=0.25, device=args.device, verbose=False)[0]
        people = []
        if res.keypoints is not None and len(res.boxes):
            kxy, kc = res.keypoints.xy.cpu().numpy(), res.keypoints.conf.cpu().numpy()
            for b, bc, p, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), kxy, kc):
                people.append(dict(box=b, conf=float(bc), kxy=p, kc=c, lab=thigh_lab(fr, p, c)))
        rows.append(dict(i=i, t_video=float(times[i]), people=people))
        if len(rows) % 500 == 0:
            el = time.time() - t0
            print(f"  {len(rows)} frames, video {times[i]:6.1f}/{t_end:.0f} s, {el:.0f} s elapsed, ~{el * (t_end / times[i] - 1):.0f} s left", flush=True)
    pickle.dump(dict(rows=rows, stride=args.stride, frame_size=fr.shape[1::-1]), open(CACHE, "wb"))
    print("wrote", CACHE, len(rows), "frames")


def to_floor(H, pts_px):
    return cv2.perspectiveTransform(np.asarray(pts_px, float).reshape(1, -1, 2), H)[0] * IN


def person_foot(p, scale, H):
    """Floor position (m) from the ankles of one detection, in the 1080p unmirrored frame; None if no ankle."""
    a = [i for i in (LANK, RANK) if p["kc"][i] >= ANKLE_CONF]
    if not a:
        return None
    ank = p["kxy"][a].mean(axis=0)
    if p["kc"][NOSE] >= 0.35:                                   # lower the ankle joint to the floor
        h_px = np.linalg.norm(p["kxy"][NOSE] - ank)
        ank = ank + np.array([0, ANKLE_LIFT_M / BODY_M * h_px])
    return to_floor(H, [ank * scale])[0]


def associate(args):
    D = pickle.load(open(CACHE, "rb"))
    sync = json.load(open(msio.HALLWAY / "video_sync.json"))
    H = np.array(json.load(open(msio.HALLWAY / "grid_hallway.json"))["H_image_px_to_floor_in"])
    wav_len = json.load(open(msio.HALLWAY / "audio_4ch.json"))["duration_s"]
    W0 = D["frame_size"][0]
    scale = 1920.0 / W0
    rows = [r for r in D["rows"] if 0 <= r["t_video"] - sync["offset_s"] <= wav_len]
    # --- who is who: 2-means on the thigh colour (L, a, b) of all detections that have ankles and a colour
    labs = np.array([p["lab"] for r in rows for p in r["people"] if p["lab"] is not None and p["conf"] > 0.6
                     and (p["kc"][[LANK, RANK]] >= ANKLE_CONF).any()])
    cen = labs[np.argsort(labs[:, 0])[[len(labs) // 10, -len(labs) // 10]]].astype(float)
    for _ in range(30):
        lab_ = np.argmin(((labs[:, None] - cen[None]) ** 2).sum(-1), axis=1)
        cen = np.array([labs[lab_ == k].mean(0) for k in range(2)])
    order = np.argsort(cen[:, 0])                               # A = darker trousers
    cen = cen[order]
    print("thigh colour (Lab) A (dark):", cen[0].round(1), " B (light):", cen[1].round(1),
          " cluster sizes", np.bincount(lab_, minlength=2)[order])
    sep = np.linalg.norm(cen[0] - cen[1])
    # --- per frame: assign detections to A/B
    n = len(rows)
    t = np.array([r["t_video"] for r in rows]) - sync["offset_s"]
    X = np.full((n, 2, 2), np.nan)
    q = np.zeros((n, 2), int)
    last = [None, None]; last_t = [-9.0, -9.0]
    for k, r in enumerate(rows):
        cands = []
        for p in r["people"]:
            if p["conf"] < 0.4:
                continue
            f = person_foot(p, scale, H)
            if f is None:
                continue
            col = (np.linalg.norm(p["lab"] - cen, axis=1) / sep) if p["lab"] is not None else np.array([0.5, 0.5])
            cands.append((f, col, int((p["kc"][[LANK, RANK]] >= ANKLE_CONF).sum()), p["conf"]))
        # best assignment of up to 2 candidates to 2 speakers (colour distance + distance from last position)
        best = None
        for a in ([(None, None)] if not cands else [(i, j) for i in [None] + list(range(len(cands))) for j in [None] + list(range(len(cands))) if i is None or i != j]):
            cost = 0.0
            for s, ci in enumerate(a):
                if ci is None:
                    cost += 1.0                                 # leaving a speaker unassigned costs something
                    continue
                f, col, _, _ = cands[ci]
                cost += col[s]
                if last[s] is not None and t[k] - last_t[s] < 2.0:
                    d = np.linalg.norm(f - last[s])
                    cost += 2.0 * max(0.0, d - GATE_M) + 0.3 * d
            if best is None or cost < best[0]:
                best = (cost, a)
        for s, ci in enumerate(best[1]):
            if ci is not None:
                f, col, na, _ = cands[ci]
                if col[s] > 0.9 and last[s] is None:
                    continue
                X[k, s], q[k, s] = f, na
                last[s], last_t[s] = f, t[k]
    # --- smooth per speaker
    fps = 60.0 / D["stride"]
    S = np.full_like(X, np.nan)
    status = np.full((n, 2), "lost", dtype=object)
    for s in range(2):
        Xs = X[:, s].copy()
        good = ~np.isnan(Xs[:, 0])
        status[good, s] = "tracked"
        idx = np.flatnonzero(good)
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a and t[b] - t[a] <= GAP_FILL_S:
                for c in range(2):
                    Xs[a + 1:b, c] = np.interp(t[a + 1:b], [t[a], t[b]], [Xs[a, c], Xs[b, c]])
                status[a + 1:b, s] = "interp"
        have = ~np.isnan(Xs[:, 0])
        win = max(5, int(round(SMOOTH_S * fps)) | 1)
        start = None
        for j in range(n + 1):
            on = j < n and have[j] and (start is None or t[j] - t[j - 1] <= SEG_BREAK_S)
            if on and start is None:
                start = j
            elif not on and start is not None:
                m = j - start
                w = min(win, m if m % 2 else m - 1)
                S[start:j, s] = savgol_filter(Xs[start:j], w, 2, axis=0) if (m >= 5 and w >= 5) else Xs[start:j]
                start = j if (j < n and have[j]) else None
    speakers = "AB"
    with open(msio.HALLWAY / "positions.csv", "w") as f:
        f.write("t_wav_s," + ",".join(f"{c}_{v}" for c in speakers for v in ("x_m", "y_m", "x_s_m", "y_s_m", "status", "n_ankles")) + "\n")
        for k in range(n):
            cells = [f"{t[k]:.3f}"]
            for s in range(2):
                fm = lambda v: "" if np.isnan(v) else f"{v:.3f}"
                cells += [fm(X[k, s, 0]), fm(X[k, s, 1]), fm(S[k, s, 0]), fm(S[k, s, 1]), status[k, s], str(q[k, s]) if q[k, s] else ""]
            f.write(",".join(cells) + "\n")
    cover = {c: round(float((status[:, s] != "lost").mean()), 3) for s, c in enumerate(speakers)}
    json.dump(dict(
        description="Floor positions of the two speakers from the fixed camera, one row per processed video frame.",
        time_base="t_wav_s = seconds in data/audio_4ch.wav (zero = start of the iPhone Air file) = video time - video_sync offset_s",
        video_offset_s=sync["offset_s"], sample_rate_hz=fps, frame="floor frame of devices.json: x across the hall from the left wall, "
        "y along the hall away from the camera, metres (x range 0 to 3.04)",
        columns=dict(x_m_y_m="raw position, blank if not detected", x_s_m_y_s_m="smoothed (Savitzky-Golay 1 s, gaps <= 0.6 s filled)",
                     status="tracked | interp (gap filled) | lost", n_ankles="ankle keypoints used, 1 or 2"),
        speakers=dict(A="the person with the darker trousers", B="the person with the lighter trousers"),
        coverage=cover, n_rows=n, method="YOLO11m-pose ankles -> floor homography; speakers separated by thigh colour",
        caveats=("Ankle-based. Floor resolution is about 3 cm/px at 5 m and 6 cm/px at 9 m in 1080p (twice that in this 720p video), "
                 "so expect errors of 0.1 m near the camera and more at the far end, plus the grid's own 0.9 in. "
                 "A person whose feet are cut off by the bottom of the frame (very close to the camera) is 'lost'. "
                 "Speakers are told apart by trouser colour, so a wrong swap is possible when the two stand close together.")),
        open(msio.HALLWAY / "positions.json", "w"), indent=1)
    print("coverage", cover, "rows", n, "wrote hallway/positions.csv, positions.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect", action="store_true")
    ap.add_argument("--stride", type=int, default=6, help="process every Nth frame (6 = 10 fps)")
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    detect(a) if a.detect else associate(a)
