# Multispeaker: hallway session

Two people talking in a hallway, recorded by four devices in the corners, plus a short camera clip for the
floor grid. Same idea as `../feasibility/`, one room, more speakers.

## Layout

| Path | What | In git |
|---|---|---|
| `data/recordings/` | the 4 original recordings (`iphone8`, `ipad`, `Macbook`, `iphoneair`, `.m4a`, about 25 min, 48 kHz) | no (279 MB, voices) |
| `data/Session_Thumbnail.mov` | 17 s camera clip, 4K, fixed camera (the only video we have) | no (faces) |
| `data/audio_4ch.wav` | the synced 4-channel WAV, rebuilt by `tools/make_wav4.py` | no (581 MB) |
| `hallway/sync_offsets.json` | clock offsets of the devices from the start claps | yes |
| `hallway/audio_4ch.json` | what is in the WAV: channel order, sample shifts, length, device positions | yes |
| `hallway/grid_hallway.json` | homography, camera pixel -> floor position | yes |
| `hallway/devices.json` | where the four devices are, in the floor frame (metres) | yes |
| `hallway/grid_inputs.json` | lines fitted in the camera frame that the grid is built from (input of `fit_hallway_grid.py`) | yes |
| `hallway/frames/` | `sample_frame_1080.jpg`, `grid.jpg`, `grid_devices.jpg`; `work/` holds the intermediate pictures | no (people) |
| `tools/` | `sync_recordings.py`, `make_wav4.py`, `fit_hallway_grid.py` (+ `msio.py`, shared helpers) | yes |

Run the tools from `tools/` with the venv Python (`/Users/ziangxue/.venvs/chatter-feas/bin/python`).

## Getting the WAV from the recordings only (e.g. a partner without the WAV)

1. Put the 4 original `.m4a` files, unchanged and with these names, in `data/recordings/`. Check them with
   `cd data/recordings && shasum -c ../../hallway/recordings.sha256`.
2. Needs `ffmpeg` and `ffprobe` on the PATH, and Python with `numpy` and `scipy`.
3. `cd tools && python make_wav4.py --end Macbook:1519.45` writes `data/audio_4ch.wav`. It only needs
   `hallway/sync_offsets.json` (in git); the clip, the frames and the grid are not needed.

`sync_recordings.py` (re-deriving the offsets) also needs opencv and `../feasibility/tools/find_claps.py`.

## Rebuild everything from the recordings

```
python sync_recordings.py                    # claps -> hallway/sync_offsets.json
python make_wav4.py --end Macbook:1519.45    # -> data/audio_4ch.wav, hallway/audio_4ch.json
python fit_hallway_grid.py                   # grid_inputs.json -> grid, devices, pictures
```

## Floor frame

X across the hall from the left wall (the right wall is at 3.04 m), Y along the hall away from the camera, metres.
Origin = the left door edge where it meets the left floor edge = the iPhone 8.

| Device | Position (X, Y) m | |
|---|---|---|
| iphone8 | (0.00, 0.00) | origin |
| ipad | (3.04, 0.00) | right wall, same across-hall line |
| Macbook | (2.72, 5.41) | the laptop on the floor in front of the trash can (+-0.15 m) |
| iphoneair | (0.05, 5.41) | left wall, across from the MacBook (X is a guess, +-0.15 m) |

The grid was fitted from tape measurements (hall width 9'11"-10', 17'1" from the origin to the bottle/laptop line,
3 ft, 3'11" and 33 in between points along the right wall); it fits them to 0.9 in RMS. The fitted lines in
`grid_inputs.json` came from edge fits on the sample frame and were checked against the bottle and laptop, which
land on the same across-hall line without being used in the fit. Camera estimate: focal length about 1400 px,
roll about 1.5 degrees.

## Sync notes

- Start claps (all devices together) give offsets to the iPhone Air with 0.02 to 2 ms agreement over the three
  claps. The MacBook's middle clap is a 2 ms outlier (the median ignores it).
- **No end claps.** Drift is not corrected. A speech-loudness check over 20-1380 s gives at most -15 ppm for the
  MacBook, -7 ppm for the iPad, -1 ppm for the iPhone 8; this is confounded with the speakers moving, so it is an
  upper bound. Irrelevant for loudness features over 1 s windows, relevant for timing methods.
- **Mid-session claps** at about 976.99, 978.22 and 979.44 s (WAV time) are heard by all four devices. The first-clap
  arrival times spread over 22 ms (MacBook earliest), more than the geometry allows between the MacBook and the
  iPhone 8 (6.06 m = 17.7 ms), so they carry drift information. Where the claps were made is not recorded.
- **The MacBook recording died** at 25:19.45 of its file (exact digital zeros from 1519.5 s). The WAV is cut there:
  0 to 1514.253 s of iPhone Air time. The other three recordings run longer.
- The **iPhone Air recorded stereo** (left/right correlate 0.65-0.71, right 1.5 dB quieter, zero lag); the WAV has the
  plain average as one channel. The original is untouched in `data/recordings/` if left only or right only is wanted.
- **Clipping:** the MacBook has 664 full-scale samples in the WAV, at the start claps, the mid-session claps and a few
  loud events near the laptop (24.9, 30.1, 30.6, 977-1009 s). Drop the windows around them when training.

## Sharing

`data/` is not in git (size, voices and faces). Send `data/recordings/` (and the clip if needed) through a shared
drive link; the rest comes from git, and the WAV is rebuilt with the commands above.
