#!/usr/bin/env python3
"""본 실험 결과 집계 → 본실험_결과표.csv (실행별) · 화면 요약.

평균은 '단위 안에서 Round 3개 평균 → 단위들 평균' 순서로 낸다.
Round 3개가 다 끝난 단위만 평균에 넣는다 (끝나지 않은 모델은 단위 수를 같이 적는다).
"""
import csv, json
from pathlib import Path
from statistics import mean, pstdev

import os
HERE = Path(os.environ.get("EXP_ROOT", Path(__file__).resolve().parents[1] / "workdir"))
P = [100, 300, 1000, 3000]
MODELS = ["mask_rcnn", "htc", "mask2former", "maskdino", "fastinst", "solov2", "transfiner", "yolo11_seg"]
NAME = dict(mask_rcnn="Mask R-CNN", htc="HTC", mask2former="Mask2Former", maskdino="Mask DINO",
            fastinst="FastInst", solov2="SOLOv2", transfiner="Mask Transfiner", yolo11_seg="YOLO11-seg")
SPECIAL = {"23_PCB_1-missing_hole": "†", "02_Casting-Inclusoes-C-1": "‡", "17_Cable-thunderbolt-A": "‡", "21_Console-Scratch-A": "‡"}

d, rows = {}, []
# Detectron2 계열은 조건을 맞춘 재실행(02)만 쓴다. 01 은 SGD · RandomFlip 으로 돈 것이라 다른 5개와 나란히 놓을 수 없다.
# 01 을 보고 싶으면: python3 collect_main.py --legacy
import sys
LEGACY = "--legacy" in sys.argv
D2_MODELS = {"maskdino", "fastinst", "transfiner"}
def _files():
    for f in sorted(HERE.glob("runs/*/*/0[12]/result.json")):
        m, a = f.parts[-3], f.parts[-2]
        want = "01" if (LEGACY or m not in D2_MODELS) else "02"
        if a == want:
            yield f
for f in _files():
    try:
        r = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        continue
    unit, k = f.parts[-4].rsplit("__R", 1)
    m = f.parts[-3]
    cp = {c["updates"]: c for c in r.get("checkpoints", [])}
    rows.append(dict(model=m, unit=unit, round=int(k), status=r.get("status"),
                     **{f"AP@{p}": round(cp[p]["mask_AP"] * 100, 2) if p in cp else "" for p in P},
                     **{f"AP50@{p}": round(cp[p]["mask_AP50"] * 100, 2) if p in cp else "" for p in P},
                     train_s=round(r.get("train_seconds") or 0), peak_mib=round(r.get("peak_gpu_memory_mib") or 0)))
    if r.get("status") == "passed":
        d[(m, unit, int(k))] = {p: cp[p]["mask_AP"] * 100 for p in P}
with (Path(__file__).resolve().parents[1] / "results" / "main_results.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

units = sorted({u for _, u, _ in d})
def full(m): return [u for u in units if all((m, u, k) in d for k in (1, 2, 3))]
def uavg(m, u, p): return mean(d[(m, u, k)][p] for k in (1, 2, 3))

print(f"완료 {len(d)} / 624" + ("  [--legacy: D2 계열은 01 (SGD · 증강 있음) — 다른 모델과 조건이 다르다]" if LEGACY else "  [D2 계열은 조건을 맞춘 02 만 집계]") + "\n")
print("[26단위 평균 mask AP ×100 — Round 3개 평균]")
print(f"{'모델':<17}{'완료':>7}" + "".join(f"{p:>8}" for p in P) + "   최고시점  PCB제외@최고  Round편차")
tab = []
for m in MODELS:
    fu = full(m)
    if not fu:
        print(f"{NAME[m]:<17}{sum(1 for x in d if x[0]==m):>4}/78"); continue
    row = [mean(uavg(m, u, p) for u in fu) for p in P]
    b = P[row.index(max(row))]
    nopcb = mean(uavg(m, u, b) for u in fu if u != "23_PCB_1-missing_hole")
    sd = mean(pstdev([d[(m, u, k)][b] for k in (1, 2, 3)]) for u in fu)
    tab.append((m, row, b))
    print(f"{NAME[m]:<17}{sum(1 for x in d if x[0]==m):>4}/78" + "".join(f"{v:8.2f}" for v in row)
          + f"   {b:>6}  {nopcb:10.2f}  {sd:8.2f}" + ("" if len(fu) == 26 else f"   ← 단위 {len(fu)}개만"))

done = [m for m in MODELS if len(full(m)) == 26]
if done:
    print(f"\n[단위별 — 각 모델의 최고 시점 기준 · 26단위 다 끝난 모델만: {', '.join(NAME[m] for m in done)}]")
    best = {m: next(b for mm, _, b in tab if mm == m) for m in done}
    print(f"{'단위':<30}{'평가장':>5}" + "".join(f"{NAME[m][:11]:>12}" for m in done))
    for u in units:
        nv = json.loads((HERE / "data" / f"{u}__R1" / "manifest.json").read_text())["val_count"]
        print(f"{(u + SPECIAL.get(u, ''))[:29]:<30}{nv:>5}" + "".join(f"{uavg(m, u, best[m]):12.2f}" for m in done))
    print("  † 같은 사진에 결함만 합성한 단위   ‡ 평가 사진이 5장뿐인 단위")
