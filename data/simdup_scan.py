#!/usr/bin/env python3
"""이미지 수준 유사(중복) 후보 탐지 — 단위 안에서만 비교.

폴리곤 좌표 기준(`generated/dup_candidates.csv`)이 놓치는
'각도만 살짝 · 크기만 살짝 · 마스크만 살짝' 쌍을 잡기 위한 것이다.

두 신호를 함께 기록한다.
  dhash   전역 해시 거리 — 거의 동일한 사진에 강하다
  orb     ORB 키포인트 + RANSAC 호모그래피 내부점 — 회전·확대에 강하다
          같은 표면(가공 무늬·결)이면 내부점이 많이 남는다.
          고정 구도라 배경은 원래 비슷하므로, 전역 밝기 상관이 아니라
          이 내부점 수를 판별 신호로 쓴다.

호모그래피에서 추정 배율·회전각도 함께 남긴다 ('크기만 살짝 바꿘 것' 확인용).

데이터는 읽기만 한다. 출력은 CSV 하나.
"""
import argparse
import csv
import glob
import itertools
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = os.environ.get("VISION_UNITS", str(Path(__file__).resolve().parents[1] / "workdir" / "units_raw"))   # VISION 원본을 단위별로 묶은 폴더
EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def unit_list():
    return sorted(
        d for d in os.listdir(ROOT)
        if os.path.isdir(os.path.join(ROOT, d, "images"))
    )


def dhash(gray, size=8):
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = np.packbits(diff.flatten())
    return bits


def hamming(a, b):
    return int(np.unpackbits(a ^ b).sum())


def load_unit(unit, side):
    """단위의 이미지를 회색조로 읽어 긴 변을 side 로 맞춘다 (원본은 건드리지 않음)."""
    paths = sorted(
        p for p in glob.glob(os.path.join(ROOT, unit, "images", "*"))
        if p.lower().endswith(EXTS)
    )
    out = []
    for p in paths:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if im is None:
            print(f"  [경고] 읽기 실패: {p}", file=sys.stderr)
            continue
        h, w = im.shape
        s = side / max(h, w)
        g = cv2.resize(im, (int(round(w * s)), int(round(h * s))),
                       interpolation=cv2.INTER_AREA)
        out.append((os.path.basename(p), (h, w), g))
    return out


def describe(images, nfeat):
    orb = cv2.ORB_create(nfeatures=nfeat)
    feats = []
    for name, shape, g in images:
        kp, des = orb.detectAndCompute(g, None)
        pts = np.float32([k.pt for k in kp]).reshape(-1, 1, 2) if kp else None
        feats.append({"name": name, "shape": shape, "hash": dhash(g),
                      "pts": pts, "des": des, "n": 0 if des is None else len(des)})
    return feats


def match_pair(a, b, ratio=0.75, reproj=4.0):
    """good match 수, RANSAC 내부점 수, 추정 배율·회전각."""
    if a["des"] is None or b["des"] is None:
        return 0, 0, float("nan"), float("nan")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = bf.knnMatch(a["des"], b["des"], k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
    if len(good) < 4:
        return len(good), 0, float("nan"), float("nan")
    src = np.float32([a["pts"][m.queryIdx][0] for m in good]).reshape(-1, 1, 2)
    dst = np.float32([b["pts"][m.trainIdx][0] for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, reproj)
    if H is None or mask is None:
        return len(good), 0, float("nan"), float("nan")
    inl = int(mask.sum())
    A = H[:2, :2]
    det = float(np.linalg.det(A))
    scale = float(np.sqrt(abs(det))) if det != 0 else float("nan")
    rot = float(np.degrees(np.arctan2(A[1, 0], A[0, 0])))
    return len(good), inl, scale, rot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", nargs="*", default=None, help="기본: 전체 27단위")
    ap.add_argument("--side", type=int, default=512, help="비교용 축소 크기 (원본 불변)")
    ap.add_argument("--nfeat", type=int, default=1500)
    ap.add_argument("--min-inliers", type=int, default=15,
                    help="이 값 미만 · dhash 도 멀면 기록하지 않는다")
    ap.add_argument("--max-dhash", type=int, default=16,
                    help="이 값 이하면 내부점이 적어도 기록한다")
    ap.add_argument("--threads", type=int, default=8, help="OpenCV 스레드 상한 (공용 서버)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cv2.setNumThreads(args.threads)

    units = args.units or unit_list()
    rows = []
    for ui, unit in enumerate(units, 1):
        t0 = time.time()
        imgs = load_unit(unit, args.side)
        feats = describe(imgs, args.nfeat)
        n = len(feats)
        kept = 0
        for i, j in itertools.combinations(range(n), 2):
            a, b = feats[i], feats[j]
            hd = hamming(a["hash"], b["hash"])
            good, inl, scale, rot = match_pair(a, b)
            denom = max(1, good)
            if inl >= args.min_inliers or hd <= args.max_dhash:
                rows.append({
                    "단위": unit,
                    "A": a["name"], "B": b["name"],
                    "A크기": f'{a["shape"][1]}x{a["shape"][0]}',
                    "B크기": f'{b["shape"][1]}x{b["shape"][0]}',
                    "dhash거리": hd,
                    "good": good,
                    "내부점": inl,
                    "내부점비": round(inl / denom, 4),
                    "특징A": a["n"], "특징B": b["n"],
                    "추정배율": "" if scale != scale else round(scale, 4),
                    "추정회전": "" if rot != rot else round(rot, 2),
                })
                kept += 1
        print(f"[{ui}/{len(units)}] {unit}: 이미지 {n} · 쌍 {n*(n-1)//2} "
              f"· 기록 {kept} · {time.time()-t0:.1f}s", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["단위", "A", "B", "A크기", "B크기", "dhash거리",
                            "good", "내부점", "내부점비", "추정배율", "추정회전"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["단위"], -r["내부점"])))
    print(f"\n기록 {len(rows)}쌍 → {args.out}")


if __name__ == "__main__":
    main()
