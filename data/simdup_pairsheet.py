#!/usr/bin/env python3
"""쌍 하나를 '정렬해서' 보여 주는 판정판.

A · B 를 그냥 나란히 놓으면 고정 구도라 다 비슷해 보인다. 그래서
B 를 A 좌표계로 워프해 겹친 뒤, 차이 영상까지 함께 낸다.

  1) A (노란 윤곽 = A 의 결함)
  2) B 를 A 에 맞춰 워프 (빨간 윤곽 = B 의 결함)
  3) A 와 워프된 B 의 차이 — 같은 부품이면 결함 자리만 빼고 어둡다
  4) 결함 부위 확대 (A / 워프 B 겹침)

데이터는 읽기만 한다.
"""
import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

ROOT = os.environ.get("VISION_UNITS", str(Path(__file__).resolve().parents[1] / "workdir" / "units_raw"))   # VISION 원본을 단위별로 묶은 폴더


def load_coco(unit):
    with open(os.path.join(ROOT, unit, "annotations.json")) as f:
        c = json.load(f)
    byid = {im["id"]: im for im in c["images"]}
    segs = defaultdict(list)
    for a in c["annotations"]:
        fn = os.path.basename(byid[a["image_id"]]["file_name"])
        if isinstance(a.get("segmentation"), list):
            segs[fn].append(a["segmentation"])
    return segs


def polys_of(segs, name, scale):
    out = []
    for seg in segs.get(name, []):
        for part in seg:
            out.append(np.int32(np.array(part, np.float32).reshape(-1, 2) * scale))
    return out


def warp_polys(polys, H):
    out = []
    for p in polys:
        q = cv2.perspectiveTransform(p.astype(np.float32).reshape(-1, 1, 2), H)
        out.append(np.int32(q.reshape(-1, 2)))
    return out


def label(img, text, color=(255, 255, 255)):
    bar = np.full((24, img.shape[1], 3), 32, np.uint8)
    cv2.putText(bar, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def build(unit, a_name, b_name, side, out_path, nfeat=3000):
    segs = load_coco(unit)
    pa = os.path.join(ROOT, unit, "images", a_name)
    pb = os.path.join(ROOT, unit, "images", b_name)
    A = cv2.imread(pa)
    B = cv2.imread(pb)
    if A is None or B is None:
        return None
    sa = side / max(A.shape[:2])
    sb = side / max(B.shape[:2])
    A = cv2.resize(A, (int(A.shape[1] * sa), int(A.shape[0] * sa)))
    B = cv2.resize(B, (int(B.shape[1] * sb), int(B.shape[0] * sb)))
    ga, gb = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY), cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=nfeat)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    H = None
    if da is not None and db is not None:
        knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(db, da, k=2)
        good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) >= 8:
            src = np.float32([kb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([ka[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    hA, wA = ga.shape
    if H is None:
        Bw = cv2.resize(B, (wA, hA))
        pb_w = []
        note = "[ALIGN FAILED]"
    else:
        Bw = cv2.warpPerspective(B, H, (wA, hA))
        pb_w = warp_polys(polys_of(segs, b_name, sb), H)
        note = ""

    pa_p = polys_of(segs, a_name, sa)
    th = max(1, wA // 500)
    vA = A.copy()
    cv2.polylines(vA, pa_p, True, (0, 255, 255), th)
    vB = Bw.copy()
    cv2.polylines(vB, pb_w, True, (0, 0, 255), th)

    diff = cv2.absdiff(cv2.cvtColor(A, cv2.COLOR_BGR2GRAY), cv2.cvtColor(Bw, cv2.COLOR_BGR2GRAY))
    diff = cv2.applyColorMap(cv2.convertScaleAbs(diff, alpha=3.0), cv2.COLORMAP_INFERNO)

    both = A.copy()
    cv2.polylines(both, pa_p, True, (0, 255, 255), th)
    cv2.polylines(both, pb_w, True, (0, 0, 255), th)
    allp = pa_p + pb_w
    if allp:
        v = np.vstack(allp)
        cx, cy = int(v[:, 0].mean()), int(v[:, 1].mean())
        r = int(max(40, max(v[:, 0].ptp(), v[:, 1].ptp()) * 0.9 + 40))
        x0, y0 = max(0, cx - r), max(0, cy - r)
        zoom = both[y0:min(hA, cy + r), x0:min(wA, cx + r)]
    else:
        zoom = both
    if zoom.size == 0:
        zoom = both
    zoom = cv2.resize(zoom, (wA, int(zoom.shape[0] * wA / zoom.shape[1])))

    panes = [label(vA, f"A {a_name} (yellow = A defect)"),
             label(vB, f"B {b_name} warped to A (red = B defect) {note}", (0, 200, 255)),
             label(diff, "DIFF (dark = same part)"),
             label(zoom, "ZOOM  yellow=A  red=B")]
    h = max(p.shape[0] for p in panes)
    panes = [np.vstack([p, np.full((h - p.shape[0], p.shape[1], 3), 32, np.uint8)])
             if p.shape[0] < h else p for p in panes]
    sep = np.full((h, 4, 3), 200, np.uint8)
    out = np.hstack([panes[0], sep, panes[1], sep, panes[2], sep, panes[3]])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", required=True)
    ap.add_argument("--pair", nargs=2, required=True, metavar=("A", "B"))
    ap.add_argument("--side", type=int, default=560)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    a, b = (x if x.endswith(".jpg") else x + ".jpg" for x in args.pair)
    p = build(args.unit, a, b, args.side, args.out)
    print(p or "실패")


if __name__ == "__main__":
    main()
