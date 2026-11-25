#!/usr/bin/env python3
"""덩어리 만들기 — 완전 연결(clique) 방식.

사슬 방식(A~B, B~C 면 한 덩어리)은 서로 확인되지 않은 장을 같은 덩어리에 넣는다.
2026-09-27 `07_Wood-pits` 에서 10쌍 중 5쌍만 통과했는데 5장이 한 덩어리가 됐고,
진우님이 "11↔55, 18↔41 은 같은데 왜 넷을 묶었나" 로 짚었다. 그래서 바꿨다.

**한 덩어리 안의 모든 쌍이 규칙을 통과해야 한다.**

큰 덩어리부터 욕심껏 고른다 (근사 최대 클리크). 한 장은 한 덩어리에만 들어간다.
데이터는 읽기만 한다.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simdup_groups import load_coco, sheet, unit_images  # noqa: E402


def f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def passes(r, iou_key, iou_min, mae_max):
    return f(r[iou_key]) > iou_min and f(r["mae_fg"], 9) < mae_max


def cliques(adj, strength):
    """근사 최대 클리크를 반복해 뽑는다. 한 정점은 한 번만 쓴다."""
    left = set(adj)
    out = []
    while left:
        # 남은 정점 중 이웃이 가장 많은 것에서 시작
        seed = max(left, key=lambda v: len(adj[v] & left))
        if not (adj[seed] & left):
            left.discard(seed)
            continue
        clique = {seed}
        cand = adj[seed] & left
        while cand:
            # 현재 덩어리 전원과 이어져 있고 근거가 가장 센 것부터
            nxt = max(cand, key=lambda v: sum(strength.get(frozenset((v, u)), 0)
                                              for u in clique))
            clique.add(nxt)
            cand &= adj[nxt]
        if len(clique) > 1:
            out.append(sorted(clique))
        left -= clique
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--iou-key", default="iou_d", help="iou_d(부풀린) 또는 mask_iou")
    ap.add_argument("--iou", type=float, default=0.05)
    ap.add_argument("--mae", type=float, default=0.15)
    ap.add_argument("--per-sheet", type=int, default=4)
    ap.add_argument("--width", type=int, default=330)
    ap.add_argument("--units", nargs="*", default=None)
    ap.add_argument("--no-sheets", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.verify, encoding="utf-8-sig")))
    if args.iou_key not in rows[0]:
        raise SystemExit(f"검증표에 '{args.iou_key}' 칸이 없다")

    byunit = defaultdict(list)
    for r in rows:
        if r.get("mae_fg") and passes(r, args.iou_key, args.iou, args.mae):
            if args.units and r["단위"] not in args.units:
                continue
            byunit[r["단위"]].append(r)

    grows, near = [], []
    print(f"{'단위':<26}{'전체':>5}{'통과쌍':>7}{'덩어리':>7}{'묶인장':>7}{'1장만':>7}{'제거':>6}")
    tot_all = tot_keep = 0
    for unit in sorted(byunit):
        adj = defaultdict(set)
        strength = {}
        for r in byunit[unit]:
            a, b = r["A"], r["B"]
            adj[a].add(b)
            adj[b].add(a)
            strength[frozenset((a, b))] = f(r[args.iou_key])
        comps = cliques(adj, strength)
        allimg = unit_images(unit)
        grouped = sum(len(c) for c in comps)
        keep = len(allimg) - grouped + len(comps)
        tot_all += len(allimg)
        tot_keep += keep
        print(f"{unit:<26}{len(allimg):>5}{len(byunit[unit]):>7}{len(comps):>7}"
              f"{grouped:>7}{keep:>7}{grouped - len(comps):>6}")
        # 한 변만 모자라 덩어리에서 빠진 장 — 그 한 쌍만 보면 합칠 수 있다
        for ci, comp in enumerate(comps, 1):
            cs = set(comp)
            for v in adj:
                if v in cs:
                    continue
                miss = [u for u in comp if u not in adj[v]]
                if len(miss) == 1:
                    near.append({"단위": unit, "덩어리": f"g{ci:03d}",
                                 "덩어리장수": len(comp), "빠진장": v,
                                 "확인안된쌍": f"{v} ↔ {miss[0]}",
                                 "판정": "", "메모": ""})

        segs = {} if args.no_sheets else load_coco(unit)
        for gi, comp in enumerate(sorted(comps, key=lambda c: (-len(c), c[0])), 1):
            gid = f"g{gi:03d}"
            for n in comp:
                grows.append({"단위": unit, "덩어리": gid, "장수": len(comp),
                              "이미지": n, "판정": "", "메모": ""})
            if args.no_sheets:
                continue
            pages = [comp[i:i + args.per_sheet]
                     for i in range(0, len(comp), args.per_sheet)]
            for pi, page in enumerate(pages, 1):
                sfx = "" if len(pages) == 1 else f"_p{pi}"
                ids = "-".join(n.split(".")[0] for n in page)
                sheet(unit, page, segs, args.width,
                      os.path.join(args.outdir, "review", unit,
                                   f"{gid}_{len(comp)}장{sfx}_{ids}.jpg"))
    print(f"{'합계':<26}{tot_all:>5}{'':>7}{'':>7}{'':>7}{tot_keep:>7}{tot_all - tot_keep:>6}")

    os.makedirs(args.outdir, exist_ok=True)
    p = os.path.join(args.outdir, "groups.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=["단위", "덩어리", "장수", "이미지", "판정", "메모"])
        w.writeheader()
        w.writerows(grows)
    print(f"\n덩어리표 → {p}")
    if near:
        np_ = os.path.join(args.outdir, "붙임후보.csv")
        with open(np_, "w", newline="", encoding="utf-8-sig") as fp:
            w = csv.DictWriter(fp, fieldnames=list(near[0].keys()))
            w.writeheader()
            w.writerows(near)
        print(f"한 변만 모자란 장 {len(near)}건 → {np_}")


if __name__ == "__main__":
    main()
