"""Picture of the floor area each speaker covers: trajectory coloured by time, and where they stood (occupancy).
Reads hallway/positions.csv and devices.json; writes hallway/tracks.png (no people in it, shareable).
Usage: plot_tracks.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import msio

rows = [l.rstrip("\n").split(",") for l in open(msio.HALLWAY / "positions.csv")][1:]
f = lambda i: np.array([float(r[i]) if r[i] else np.nan for r in rows])
t = f(0)
P = {"A": (f(3), f(4)), "B": (f(9), f(10))}
dev = json.load(open(msio.HALLWAY / "devices.json"))["positions_m"]
W = 3.0353
fig, ax = plt.subplots(2, 2, figsize=(10, 13))
for k, (c, (x, y)) in enumerate(P.items()):
    a = ax[0, k]
    sc = a.scatter(x, y, c=t / 60, s=2, cmap="viridis")
    a.plot(x, y, color="0.7", lw=0.2, zorder=0)
    a.set_title(f"Speaker {c}: path (colour = minutes)")
    h = ax[1, k]
    ok = ~np.isnan(x)
    im = h.hist2d(x[ok], y[ok], bins=[np.arange(0, W + 0.1, 0.1), np.arange(0, 6.01, 0.1)], cmap="magma_r", cmin=1)
    h.set_title(f"Speaker {c}: time spent ({ok.mean() * t.max() / 60:.0f} min tracked)")
    for aa in (a, h):
        aa.set_xlim(-0.1, W + 0.1); aa.set_ylim(-0.3, 6.2); aa.set_aspect("equal")
        aa.axvline(0, color="k", lw=2); aa.axvline(W, color="k", lw=2)
        for n, (dx, dy) in dev.items():
            aa.plot(dx, dy, "s", color="C3", ms=7); aa.annotate(n, (dx, dy), (4, 4), textcoords="offset points", fontsize=7, color="C3")
        aa.set_xlabel("x across the hall (m)"); aa.set_ylabel("y along the hall, away from the camera (m)")
fig.colorbar(sc, ax=ax[0, :], shrink=0.5, label="minutes"); fig.colorbar(im[3], ax=ax[1, :], shrink=0.5, label="samples per 10 cm cell (10 per s)")
fig.savefig(msio.HALLWAY / "tracks.png", dpi=90, bbox_inches="tight")
