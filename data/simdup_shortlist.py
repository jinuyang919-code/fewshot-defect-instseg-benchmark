#!/usr/bin/env python3
"""2차 검증 결과 → 육안 판정 대상 추리기 + 판정판 생성 + 판정표 만들기.

규칙(잠정)은 라벨 5쌍으로만 맞춘 것이다. **경계 부근은 반드시 눈으로 본다.**

  통과:  mask_iou > --iou  AND  mae_fg < --mae
  경계:  통과하지 못했지만 둘 중 하나가 아깝게 걸친 쌍 (--margin 배수 안)

판정판은 `simdup_pairsheet.py` 의 형식(원본 A · A 에 정렬한 B · 차이 · 결함 겹침)이다.
데이터는 읽기만 한다.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simdup_pairsheet import build  # noqa: E402


def f(x, d=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--iou", type=float, default=0.05)
    ap.add_argument("--mae", type=float, default=0.15)
    ap.add_argument("--margin", type=float, default=2.0,
                    help="경계 표본 범위 배수 (iou 는 1/margin 배, mae 는 margin 배까지)")
    ap.add_argument("--max-border", type=int, default=12, help="단위당 경계 표본 상한")
    ap.add_argument("--units", nargs="*", default=None)
    ap.add_argument("--no-sheets", action="store_true")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.verify, encoding="utf-8-sig")))
    byunit = defaultdict(list)
    for r in rows:
        if r.get("mae_fg") in (None, "", "nan"):
            continue
        if args.units and r["단위"] not in args.units:
            continue
        byunit[r["단위"]].append(r)

    table = []
    print(f"{'단위':<26}{'검증':>6}{'통과':>6}{'경계':>6}{'판정판':>7}")
    for unit in sorted(byunit):
        rs = byunit[unit]
        passed, border = [], []
        for r in rs:
            iou, mae = f(r["mask_iou"]), f(r["mae_fg"], 9)
            if iou > args.iou and mae < args.mae:
                passed.append(r)
            elif iou > args.iou / args.margin and mae < args.mae * args.margin:
                border.append(r)
        border.sort(key=lambda r: (-f(r["mask_iou"]), f(r["mae_fg"], 9)))
        border = border[:args.max_border]
        made = 0
        for kind, group in (("통과", passed), ("경계", border)):
            for r in group:
                a, b = r["A"], r["B"]
                name = (f"{kind}_iou{f(r['mask_iou']):.2f}_mae{f(r['mae_fg']):.2f}"
                        f"_{a.split('.')[0]}__{b.split('.')[0]}.jpg")
                out = os.path.join(args.outdir, "판정판", unit, name)
                if not args.no_sheets and build(unit, a, b, 560, out):
                    made += 1
                table.append({
                    "단위": unit, "A": a, "B": b, "구분": kind,
                    "내부점": r["내부점"], "mask_iou": r["mask_iou"],
                    "mask_cov": r["mask_cov"], "mae_fg": r["mae_fg"],
                    "ncc_fg": r["ncc_fg"],
                    "판정판": os.path.relpath(out, args.outdir),
                    "판정": "", "메모": "",
                })
        print(f"{unit:<26}{len(rs):>6}{len(passed):>6}{len(border):>6}{made:>7}")

    os.makedirs(args.outdir, exist_ok=True)
    p = os.path.join(args.outdir, "판정표.csv")
    with open(p, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=["단위", "A", "B", "구분", "내부점", "mask_iou",
                                           "mask_cov", "mae_fg", "ncc_fg", "판정판",
                                           "판정", "메모"])
        w.writeheader()
        w.writerows(table)
    print(f"\n판정 대상 {len(table)}쌍 → {p}")


if __name__ == "__main__":
    main()
