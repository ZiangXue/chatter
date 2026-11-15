"""Floor grid of the hallway camera: pixel -> floor position, from tape measurements and lines fitted in the image.

Input: hallway/grid_inputs.json, the lines fitted by hand-guided edge fits on frames/sample_frame_1080.jpg
(1920x1080): the left and right floor edges (x = a*y + b), the across-hall vanishing point, the five
"across the hall" lines (perpendicular to the floor edges; the 4 on the right wall plus the one through the
water bottle and the laptop), and the left door edge. Measurements (inches) are the constants below.

Method: along the right floor edge, 1/(y - y_vanish) is linear in real distance. A least-squares fit of that
line to the measured distances gives the position of every across-line along the hall. A homography
(pixel -> floor inches) is then fitted to the 12 corner points of those lines (X across from the left wall,
Y along the hall away from the camera, origin = the left door edge meeting the left floor edge = iPhone 8).

Writes hallway/grid_hallway.json (homography), hallway/devices.json (device positions in metres) and
two pictures in hallway/frames/: grid.jpg (0.5 m grid over the frame) and grid_devices.jpg (with devices).
Prints the residuals of the fit (about 1 inch RMS).

Usage: fit_hallway_grid.py
"""
import json

import cv2
import numpy as np
from scipy.optimize import least_squares

import msio

IN = 0.0254
W_IN = 119.5                    # hall width: 9'11" to 10'
ORIGIN_TO_LINE5_IN = 205.0      # 17'1": from the origin (phone corner) to the line through the bottle and laptop
D_4_TO_3, D_4_TO_1, D_1_TO_2 = 36.0, 47.0, 33.0     # along the right wall: line 4 to 3 (3 ft), 4 to 1 (3'11"), 1 to 2 (33 in)
LAPTOP_FRONT_PX = (1114, 733)   # front edge of the laptop (the MacBook) on the floor, in the 1080p frame
LAPTOP_CENTRE_BACK_M = 0.15     # footprint centre is about this much behind the front edge
AIR_X_M = 0.05                  # iPhone Air: left wall, across from the MacBook (X is a guess, +-0.15 m)

H_DIR = msio.HALLWAY
J = json.load(open(H_DIR / "grid_inputs.json"))
aL, bL = J["left_floor_edge"]["x_at_y"]
aR, bR = J["right_floor_edge"]["x_at_y"]
v2 = np.array(J["across_hall_vp_px"])
yv = J["intersections_px"]["green_x_purple"][1]
A = J["across_lines"]
org_L = np.array(J["intersections_px"]["door_edge_x_green"])


def right_end(px, py):
    """Where the across-line through a left-wall point meets the right floor edge."""
    m = (v2[1] - py) / (v2[0] - px)
    x = (aR * (py - m * px) + bR) / (1 - aR * m)
    return np.array([x, py + m * (x - px)])


org_R = right_end(*org_L)
yp = {"o": org_R[1], "3": A["3"]["start"][1], "4": A["4"]["start"][1], "1": A["1"]["start"][1],
      "2": A["2"]["start"][1], "5": A["5"]["right_edge_end"][1]}


def residual(p):
    alpha, c, y3 = p
    Y = {"o": 0, "3": y3, "4": y3 + D_4_TO_3, "1": y3 + D_4_TO_3 + D_4_TO_1, "2": y3 + D_4_TO_3 + D_4_TO_1 + D_1_TO_2,
         "5": ORIGIN_TO_LINE5_IN}
    return np.array([(1 / (yp[k] - yv) - c) / alpha - Y[k] for k in Y]), Y


fit = least_squares(lambda p: residual(p)[0], [1.1e-5, 1 / (yp["o"] - yv), 70.0])
r, Yk = residual(fit.x)
print("distances along the hall: residual RMS %.2f in (per point %s)" % (np.sqrt(np.mean(r ** 2)), np.round(r, 1)))

src, dst = [], []


def add(p, X, Y):
    src.append(p)
    dst.append((X, Y))


add(tuple(org_L), 0, 0)
add(tuple(org_R), W_IN, 0)
add(tuple(A["5"]["start"]), 0, ORIGIN_TO_LINE5_IN)
add(tuple(A["5"]["right_edge_end"]), W_IN, ORIGIN_TO_LINE5_IN)
for k in "3412":
    add(tuple(A[k]["start"]), W_IN, Yk[k])
    add(tuple(A[k]["left_wall_end"]), 0, Yk[k])
src, dst = np.array(src, float), np.array(dst, float)
H, _ = cv2.findHomography(src, dst, 0)
err = np.hypot(*(cv2.perspectiveTransform(src[None], H)[0] - dst).T)
print("homography: residual RMS %.2f in, max %.2f in over %d points" % (np.sqrt(np.mean(err ** 2)), err.max(), len(err)))
Hi = np.linalg.inv(H)


def to_floor_m(p):
    return cv2.perspectiveTransform(np.array(p, float).reshape(1, -1, 2), H)[0] * IN


# device positions (metres)
lap = to_floor_m([LAPTOP_FRONT_PX])[0]
mac_y = round(float(lap[1]) + LAPTOP_CENTRE_BACK_M, 2)
devices = {"iphone8": (0.0, 0.0), "ipad": (round(W_IN * IN, 4), 0.0), "Macbook": (round(float(lap[0]), 2), mac_y),
           "iphoneair": (AIR_X_M, mac_y)}
json.dump(dict(
    frame="grid_hallway.json: X across the hall from the left wall, Y along the hall away from the camera, metres",
    positions_m=devices,
    notes=dict(iphone8="origin (left wall corner)", ipad="right wall, same across-hall line as the origin: X = hall width",
               Macbook="the laptop on the floor in front of the trash can: footprint centre, +-0.15 m",
               iphoneair="left wall, across from the MacBook (same Y); X is a guess, +-0.15 m")),
    open(H_DIR / "devices.json", "w"), indent=1)
json.dump(dict(
    frame="X across the hall from the left wall, Y along the hall away from the camera; origin = phone corner "
          "(left door edge meets left floor edge); units inches",
    H_image_px_to_floor_in=H.tolist(), width_in=W_IN, image="frames/sample_frame_1080.jpg (1920x1080)",
    measurements=dict(width_in=W_IN, origin_to_line5_in=ORIGIN_TO_LINE5_IN, line4_to_line3_in=D_4_TO_3,
                      line4_to_line1_in=D_4_TO_1, line1_to_line2_in=D_1_TO_2),
    residual_rms_in=float(np.sqrt(np.mean(err ** 2)))), open(H_DIR / "grid_hallway.json", "w"), indent=1)

# pictures
im = cv2.imread(str(H_DIR / "frames" / "sample_frame_1080.jpg"))
Hh, Ww = im.shape[:2]
PAD = 350
C = np.full((Hh + PAD, Ww, 3), 255, np.uint8)
C[:Hh] = im
Wm = W_IN * IN
ys = np.arange(-0.5, 8.01, 0.5)
xs = list(np.arange(0, Wm, 0.5)) + [Wm]


def seg(a, b, th):
    pts = np.array([a + (b - a) * t for t in np.linspace(0, 1, 60)]) / IN
    q = cv2.perspectiveTransform(pts[None], Hi)[0]
    cv2.polylines(C, [q.astype(np.int32)], False, (0, 255, 255) if th > 1 else (0, 200, 200), th, cv2.LINE_AA)


for X in xs:
    seg(np.array([X, ys[0]]), np.array([X, ys[-1]]), 2 if abs(X % 1) < 1e-6 else 1)
for Y in ys:
    seg(np.array([0, Y]), np.array([Wm, Y]), 2 if abs(Y % 1) < 1e-6 else 1)
for Y in ys:
    p = cv2.perspectiveTransform(np.array([[[0, Y / IN]]], float), Hi)[0, 0]
    if 0 < p[1] < Hh + PAD - 5 and 0 < p[0] < Ww:
        cv2.putText(C, "%.1f" % Y, (int(p[0]) - 48, int(p[1]) + 5), 0, 0.55, (0, 0, 0) if p[1] > Hh else (255, 255, 255), 1)
for X in xs:
    p = cv2.perspectiveTransform(np.array([[[X / IN, 0]]], float), Hi)[0, 0]
    if 0 < p[1] < Hh + PAD - 5:
        cv2.putText(C, "%.1f" % X, (int(p[0]) - 12, int(p[1]) + 22), 0, 0.55, (0, 0, 0) if p[1] > Hh else (255, 255, 255), 1)
cv2.imwrite(str(H_DIR / "frames" / "grid.jpg"), C)
col = {"iphone8": (0, 0, 255), "ipad": (255, 0, 0), "Macbook": (0, 160, 0), "iphoneair": (255, 0, 255)}
for n, (X, Y) in devices.items():
    p = cv2.perspectiveTransform(np.array([[[X / IN, Y / IN]]], float), Hi)[0, 0]
    cv2.circle(C, (int(p[0]), int(p[1])), 14, col[n], -1)
    cv2.circle(C, (int(p[0]), int(p[1])), 14, (255, 255, 255), 2)
    cv2.putText(C, n, (int(p[0]) + (16 if n != "iphoneair" else -150), int(p[1]) - 14), 0, 0.8, col[n], 2)
cv2.imwrite(str(H_DIR / "frames" / "grid_devices.jpg"), C)
print("devices (m):", devices)
