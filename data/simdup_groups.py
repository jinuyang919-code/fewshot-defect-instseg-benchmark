#!/usr/bin/env python3
"""유사쌍 → 덩어리(연결요소) 묶기 · 육안 판정용 대조 시트 생성 · 제거 영향 계산.

입력:  simdup_scan.py 가 만든 쌍 CSV
출력:  - 덩어리 목록 CSV
       - review/<단위>/g<번호>_*.jpg  대조 시트 (전체 + 결함 확대, 마스크 겹침)
       - 표준출력에 단위별 영향 (덩어리마다 1장만 남기면 몇 장 되는지)

데이터셋은 읽기만 한다. 아무것도 지우지 않는다.
대조 시트의 확대 패널은 **검토용 파생물**이며 데이터셋 이미지를 자르는 것이 아니다.
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


def load_pairs(path, min_inliers, min_ratio, max_dhash):
    """임계를 넘는 쌍만 남긴다. (내부점 기준 OR dhash 기준)"""
    keep = defaultdict(list)
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            inl = int(r["내부점"])
            ratio = float(r["내부점비"] or 0)
            hd = int(r["dhash거리"])
            if (inl >= min_inliers and ratio >= min_ratio) or hd <= max_dhash:
                keep[r["단위"]].append(r)
    return keep


def components(pairs):
    """이름 기준 union-find → 덩어리."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for r in pairs:
        union(r["A"], r["B"])
    groups = defaultdict(set)
    for name in parent:
        groups[find(name)].add(name)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def unit_images(unit):
    d = os.path.join(ROOT, unit, "images")
    return sorted(os.listdir(d))


def load_coco(unit):
    """파일명 -> [폴리곤...] · 이미지 크기."""
    p = os.path.join(ROOT, unit, "annotations.json")
    with open(p) as f:
        c = json.load(f)
    byid = {im["id"]: im for im in c["images"]}
    segs = defaultdict(list)
    for a in c["annotations"]:
        fn = byid[a["image_id"]]["file_name"]
        seg = a.get("segmentation")
        if isinstance(seg, list):
            segs[os.path.basename(fn)].append(seg)
    return segs


def panel(unit, name, segs, width, zoom_box):
    """위: 전체(마스크 윤곽) · 아래: 결함 확대. 둘 다 width 폭으로 맞춘다."""
    p = os.path.join(ROOT, unit, "images", name)
    im = cv2.imread(p)
    if im is None:
        return None
    vis = im.copy()
    polys = []
    for seg in segs.get(name, []):
        for part in seg:
            pts = np.int32(np.array(part).reshape(-1, 2))
            polys.append(pts)
    cv2.polylines(vis, polys, True, (0, 255, 255), max(2, im.shape[1] // 400))

    def fit(img, w):
        s = w / img.shape[1]
        return cv2.resize(img, (w, max(1, int(round(img.shape[0] * s)))))

    # 윗줄은 라벨 없는 원본. 라벨을 그려 놓으면 선이 먼저 눈에 들어와
    # 같은 사진인지 판단이 흐려진다 (2026-09-27 확인).
    top = fit(im, width)

    if polys:
        allp = np.vstack(polys)
        x0, y0 = allp[:, 0].min(), allp[:, 1].min()
        x1, y1 = allp[:, 0].max(), allp[:, 1].max()
    else:
        x0 = y0 = 0
        x1, y1 = im.shape[1], im.shape[0]
    # 덩어리 안에서 같은 배율을 쓰려면 같은 크기 창을 쓴다
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    bw, bh = zoom_box
    x0 = max(0, min(im.shape[1] - bw, cx - bw // 2))
    y0 = max(0, min(im.shape[0] - bh, cy - bh // 2))
    crop = vis[y0:y0 + bh, x0:x0 + bw]
    if crop.size == 0:
        crop = vis
    bot = fit(crop, width)

    label = np.full((26, width, 3), 32, np.uint8)
    cv2.putText(label, f"{name}  n={len(segs.get(name, []))}  [top=no label]", (6, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    gap = np.full((4, width, 3), 32, np.uint8)
    return np.vstack([label, top, gap, bot])


def sheet(unit, names, segs, width, out_path):
    # 확대 창 크기 = 덩어리 안 결함 박스의 최대치 (양쪽 배율 동일)
    boxes = []
    for n in names:
        polys = [np.int32(np.array(part).reshape(-1, 2))
                 for seg in segs.get(n, []) for part in seg]
        if polys:
            allp = np.vstack(polys)
            boxes.append((allp[:, 0].ptp() + 1, allp[:, 1].ptp() + 1))
    if boxes:
        side = int(max(max(b) for b in boxes) * 1.6) + 60
        bw = bh = max(256, side)          # 정사각 창 — 덩어리 안 배율 동일
    else:
        bw = bh = 400
    panels = [panel(unit, n, segs, width, (bw, bh)) for n in names]
    panels = [p for p in panels if p is not None]
    if not panels:
        return False
    h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < h:
            p = np.vstack([p, np.full((h - p.shape[0], p.shape[1], 3), 32, np.uint8)])
        padded.append(p)
        padded.append(np.full((h, 5, 3), 200, np.uint8))
    out = np.hstack(padded[:-1])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--min-inliers", type=int, default=100)
    ap.add_argument("--min-ratio", type=float, default=0.25)
    ap.add_argument("--max-dhash", type=int, default=6)
    ap.add_argument("--width", type=int, default=330, help="시트 패널 폭")
    ap.add_argument("--per-sheet", type=int, default=4, help="한 시트에 넣는 장수")
    ap.add_argument("--units", nargs="*", default=None)
    ap.add_argument("--no-sheets", action="store_true")
    args = ap.parse_args()

    kept = load_pairs(args.pairs, args.min_inliers, args.min_ratio, args.max_dhash)
    os.makedirs(args.outdir, exist_ok=True)
    grows = []
    print(f"{'단위':<26}{'전체':>5}{'덩어리':>7}{'묶인장':>7}{'1장만남기면':>11}{'제거':>6}")
    tot_all = tot_keep = 0
    for unit in sorted(set(list(kept) + (args.units or []))):
        if args.units and unit not in args.units:
            continue
        allimg = unit_images(unit)
        comps = components(kept.get(unit, []))
        grouped = sum(len(c) for c in comps)
        remain = len(allimg) - grouped + len(comps)
        tot_all += len(allimg)
        tot_keep += remain
        flag = " *" if remain < 10 else ""
        print(f"{unit:<26}{len(allimg):>5}{len(comps):>7}{grouped:>7}{remain:>11}"
              f"{grouped - len(comps):>6}{flag}")
        segs = load_coco(unit) if not args.no_sheets else {}
        for gi, comp in enumerate(sorted(comps, key=lambda c: (-len(c), c[0])), 1):
            gid = f"g{gi:03d}"
            for row_name in comp:
                grows.append({"단위": unit, "덩어리": gid, "장수": len(comp),
                              "이미지": row_name, "판정": "", "메모": ""})
            if args.no_sheets:
                continue
            pages = [comp[i:i + args.per_sheet]
                     for i in range(0, len(comp), args.per_sheet)]
            for pi, page in enumerate(pages, 1):
                suffix = "" if len(pages) == 1 else f"_p{pi}"
                ids = "-".join(n.split(".")[0] for n in page)
                out = os.path.join(args.outdir, "review", unit,
                                   f"{gid}_{len(comp)}장{suffix}_{ids}.jpg")
                sheet(unit, page, segs, args.width, out)
    print(f"{'합계':<26}{tot_all:>5}{'':>7}{'':>7}{tot_keep:>11}{tot_all - tot_keep:>6}")

    gp = os.path.join(args.outdir, "groups.csv")
    with open(gp, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["단위", "덩어리", "장수", "이미지", "판정", "메모"])
        w.writeheader()
        w.writerows(grows)
    print(f"\n덩어리표 → {gp}")
    if not args.no_sheets:
        print(f"대조 시트 → {os.path.join(args.outdir, 'review')}")


if __name__ == "__main__":
    main()
