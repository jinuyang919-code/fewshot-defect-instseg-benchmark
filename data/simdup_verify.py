#!/usr/bin/env python3
"""후보쌍 2차 검증 — '정렬이 되는가' 만으로는 부족해서 '결함까지 겹치는가' 를 본다.

1차(simdup_scan.py)는 ORB 내부점으로 후보를 잡는다. 그런데 나사산처럼
**무늬가 반복되는 단위**에서는 서로 다른 부품끼리도 무늬가 맞아 내부점이 많이 나온다
(09_Screw-defect 에서 117장이 한 덩어리로 묶이는 오탐을 확인했다, 2026-09-27).

그래서 B 를 A 좌표계로 워프한 뒤 다음을 잰다.

  ncc        겹치는 영역의 정규화 상관 — 같은 부품이면 미세 무늬까지 일치한다
  ncc_bg     결함 영역을 뺀 나머지의 상관 — 결함 차이에 휘둘리지 않는 배경 일치도
  mask_iou   B 의 결함 마스크를 워프해 A 의 결함 마스크와 겹친 비율
  mask_cov   작은 쪽 기준 겹침 비율 (라벨 수가 다른 경우 대비)

'같은 부품 재촬영' 은 ncc 와 mask 쪽이 **함께** 높다.
'다른 부품인데 무늬만 반복' 은 ncc 가 높아도 mask 가 낮다.

데이터는 읽기만 한다.
"""
import argparse
import csv
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


def mask_of(segs, name, shape, scale):
    m = np.zeros(shape, np.uint8)
    polys = []
    for seg in segs.get(name, []):
        for part in seg:
            pts = np.array(part, np.float32).reshape(-1, 2) * scale
            polys.append(np.int32(pts))
    if polys:
        cv2.fillPoly(m, polys, 1)
    return m


def gray(path, side):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None, 1.0
    h, w = im.shape
    s = side / max(h, w)
    return cv2.resize(im, (int(round(w * s)), int(round(h * s)))), s


def ncc(a, b, valid):
    v = valid > 0
    if v.sum() < 500:
        return float("nan")
    x = a[v].astype(np.float32)
    y = b[v].astype(np.float32)
    x -= x.mean()
    y -= y.mean()
    d = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x.dot(y) / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-inliers", type=int, default=60)
    ap.add_argument("--side", type=int, default=768)
    ap.add_argument("--nfeat", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--units", nargs="*", default=None)
    args = ap.parse_args()
    cv2.setNumThreads(args.threads)

    cand = defaultdict(list)
    with open(args.pairs, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if int(r["내부점"]) >= args.min_inliers:
                if args.units and r["단위"] not in args.units:
                    continue
                cand[r["단위"]].append(r)

    orb = cv2.ORB_create(nfeatures=args.nfeat)
    out = []
    for unit, rows in sorted(cand.items()):
        segs = load_coco(unit)
        cache = {}

        def get(name):
            if name not in cache:
                g, s = gray(os.path.join(ROOT, unit, "images", name), args.side)
                kp, des = orb.detectAndCompute(g, None) if g is not None else ([], None)
                cache[name] = {
                    "g": g, "s": s,
                    "pts": np.float32([k.pt for k in kp]).reshape(-1, 1, 2) if kp else None,
                    "des": des,
                    "m": mask_of(segs, name, g.shape, s) if g is not None else None,
                }
            return cache[name]

        for r in rows:
            A, B = get(r["A"]), get(r["B"])
            rec = dict(r)
            rec.update({"ncc": "", "ncc_bg": "", "ncc_fg": "", "mae_fg": "",
                        "fg_ratio": "", "mask_iou": "", "mask_cov": "",
                        "iou_d": "", "cov_d": ""})
            if A["g"] is None or B["g"] is None or A["des"] is None or B["des"] is None:
                out.append(rec)
                continue
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            knn = bf.knnMatch(B["des"], A["des"], k=2)
            good = [m for m, n in (p for p in knn if len(p) == 2)
                    if m.distance < 0.75 * n.distance]
            if len(good) < 8:
                out.append(rec)
                continue
            src = np.float32([B["pts"][m.queryIdx][0] for m in good]).reshape(-1, 1, 2)
            dst = np.float32([A["pts"][m.trainIdx][0] for m in good]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if H is None:
                out.append(rec)
                continue
            hA, wA = A["g"].shape
            warp = cv2.warpPerspective(B["g"], H, (wA, hA))
            cover = cv2.warpPerspective(np.ones_like(B["g"]), H, (wA, hA))
            mB = cv2.warpPerspective(B["m"], H, (wA, hA), flags=cv2.INTER_NEAREST)
            mA = A["m"]

            # 검은 배경·고정 배경이 넓으면 전역 ncc 가 부풀려진다.
            # A 의 밝은 물체 영역(Otsu)으로 한정한 지표를 따로 낸다.
            _, fg = cv2.threshold(A["g"], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            fg = cv2.dilate(fg, np.ones((9, 9), np.uint8))
            fgv = (fg > 0) & (cover > 0)
            rec["fg_ratio"] = round(float(fgv.mean()), 4)
            rec["ncc_fg"] = round(ncc(A["g"], warp, fgv.astype(np.uint8)), 4)
            d = cv2.absdiff(A["g"], warp).astype(np.float32)
            base = float(A["g"][fgv].mean()) if fgv.sum() else 0.0
            rec["mae_fg"] = round(float(d[fgv].mean() / base), 4) if base > 0 else ""
            rec["ncc"] = round(ncc(A["g"], warp, cover), 4)
            bg = (cover > 0) & (mA == 0) & (mB == 0)
            rec["ncc_bg"] = round(ncc(A["g"], warp, bg.astype(np.uint8)), 4)
            # 가는 결함(머리카락 같은 균열)은 몇 픽셀만 어긋나도 IoU 가 0 이 된다.
            # 2026-09-27 07_Wood-pits 000011↔000055 에서 확인. 부풀린 뒤에도 잰다.
            k = np.ones((11, 11), np.uint8)
            dA, dB = cv2.dilate(mA, k), cv2.dilate(mB, k)
            di = int(((dA > 0) & (dB > 0)).sum())
            du = int(((dA > 0) | (dB > 0)).sum())
            ds = max(1, min(int((dA > 0).sum()), int((dB > 0).sum())))
            rec["iou_d"] = round(di / du, 4) if du else ""
            rec["cov_d"] = round(di / ds, 4)
            inter = int(((mA > 0) & (mB > 0)).sum())
            union = int(((mA > 0) | (mB > 0)).sum())
            small = max(1, min(int((mA > 0).sum()), int((mB > 0).sum())))
            rec["mask_iou"] = round(inter / union, 4) if union else ""
            rec["mask_cov"] = round(inter / small, 4)
            out.append(rec)
        print(f"{unit}: 검증 {len(rows)}쌍", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"\n{len(out)}쌍 → {args.out}")


if __name__ == "__main__":
    main()
