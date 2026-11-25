#!/usr/bin/env python3
"""그림 2 — 에폭별 26개 데이터셋 평균 mask AP (높이 줄인 판, 2026-10-26).

make_fig2_8pt.py 와 같고 높이만 3.8cm → 3.3cm (Word 에서 3쪽 맞추기). 축 아래 눈금·축 이름 자리(pt)는 그대로 두고 그래프 칸만 줄였다.

아래는 원래 설명.

교수님 13차 요청: 인쇄 폭 8cm 에서 글씨 8pt 이상, 범례 대신 오른쪽에 600 에폭 값 순서로 이름 (끝점과 연한 점선으로 연결), y축 "mask AP".
실제 인쇄 크기로 그리므로 fontsize 가 곧 지면 pt 다. 수치는 본실험_결과표.csv 에서 직접 읽는다 (표 1 과 같은 값).
이전 판(두 줄 범례, 7pt)은 make_fig1.py → fig1_budget.png 에 그대로 있다.
실행: env/pilot_legacy/bin/python make_fig2_8pt_short.py  ->  fig2_budget_8pt_short.png (600 dpi)
"""
import csv
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "results" / "main_results.csv"
OUT = HERE.parent / "assets" / "fig2_epochs.png"
P = [100, 300, 1000, 3000]                    # 업데이트 수. 5장 · batch 1 → 5 업데이트 = 1 에폭
W_IN, H_IN = 8.0 / 2.54, 3.3 / 2.54     # 4.6cm → 3.8cm (10-20) → 3.3cm (10-26, Word 3쪽 맞춤)
FS = 8
# 색 · 마커는 make_fig1.py 와 같다 (모델마다 고정)
SERIES = [("htc", "HTC", "#2a78d6", "o"), ("mask2former", "Mask2Former", "#eb6834", "s"),
          ("mask_rcnn", "Mask R-CNN", "#1baf7a", "^"), ("yolo11_seg", "YOLO11-seg", "#eda100", "D"),
          ("solov2", "SOLOv2", "#e87ba4", "v"),
          ("maskdino", "Mask DINO", "#008300", "P"), ("transfiner", "Mask Transfiner", "#4a3aa7", "X"),
          ("fastinst", "FastInst", "#e34948", "*")]

plt.rcParams.update({"font.family": "Liberation Sans", "font.size": FS})
rows = [r for r in csv.DictReader(CSV.open(encoding="utf-8")) if r["status"] == "passed"]
units = sorted({r["unit"] for r in rows})
assert len(units) == 26

Y = {}
for key, *_ in SERIES:
    assert sum(1 for r in rows if r["model"] == key) == 26 * 3, key
    Y[key] = [mean(mean(float(r[f"AP@{p}"]) for r in rows if r["model"] == key and r["unit"] == u) for u in units)
              for p in P]

fig = plt.figure(figsize=(W_IN, H_IN))
AX_B = 0.235 * 3.8 / 3.3                # 아래 여백(눈금 · "Epochs") pt 는 3.8cm 판과 같게
AX_H = 1 - AX_B - 0.02 * 3.8 / 3.3      # 위 여백도 pt 로 같게
ax = fig.add_axes([0.105, AX_B, 0.555, AX_H])
top = max(max(v) for v in Y.values())
for key, name, color, mk in SERIES:
    ax.plot(range(4), Y[key], color=color, marker=mk, markersize=4.2, linewidth=1.3,
            markeredgecolor="white", markeredgewidth=0.5, clip_on=False, zorder=3)
ax.set_xticks(range(4)); ax.set_xticklabels([str(p // 5) for p in P])
ax.set_xlabel("Epochs", fontsize=FS, labelpad=1.5); ax.set_ylabel("mask AP", fontsize=FS, labelpad=2)
ax.set_ylim(0, 11); ax.set_yticks([0, 2, 4, 6, 8, 10]); ax.set_xlim(-0.15, 3.0)
ax.tick_params(labelsize=FS, length=2.5, width=0.5, colors="#3a3a38", pad=1.5)
ax.grid(axis="y", color="#e6e5e0", linewidth=0.5); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#9a9994"); ax.spines[s].set_linewidth(0.6)

# 선 끝 이름: 600 에폭 값 순서대로, 겹치지 않게 아래위로 벌린다 (글자 높이 ≈ 0.95 AP 단위)
GAP = 9.2 / (H_IN * 72 * AX_H / 11)          # 이름 한 줄 높이 9.2pt 를 AP 단위로 (y축 0~11)
order = sorted(SERIES, key=lambda s: -Y[s[0]][3])
pos = [Y[s[0]][3] for s in order]
for _ in range(500):                           # 겹치는 이웃끼리 반씩 밀어낸다 (제자리에서 가장 덜 움직이게)
    for i in range(len(pos) - 1):
        d = GAP - (pos[i] - pos[i + 1])
        if d > 0:
            pos[i] += d / 2; pos[i + 1] -= d / 2
    pos[0] = min(pos[0], 10.55)                # 맨 위 이름이 그림 위로 잘리지 않게 (높이를 줄인 판에서 추가)
    pos[-1] = max(pos[-1], -0.25)              # 맨 아래 이름은 축 바로 아래까지 허용 (3.8cm 판: 0.35)
assert all(pos[i] - pos[i + 1] > GAP * 0.97 for i in range(len(pos) - 1)), pos   # 이름끼리 겹치지 않았는지
for (key, name, color, mk), yl in zip(order, pos):
    # 600 끝점 → 이름: 연한 점선 (데이터 선이 600 뒤로 이어져 보이지 않게 끝점에서 조금 띄우고 옅게)
    ax.plot([3.07, 3.24], [Y[key][3], yl], color=color, alpha=0.4, lw=0.6, ls=(0, (1.2, 1.2)),
            clip_on=False, zorder=2)
    ax.plot([3.30], [yl], marker=mk, color=color, markersize=4.0, markeredgecolor="white",
            markeredgewidth=0.4, clip_on=False, zorder=4)
    ax.text(3.40, yl, name, fontsize=FS, va="center", ha="left", color="#1a1a18")

fig.savefig(OUT, dpi=600, facecolor="white")
for key, name, *_ in SERIES:
    print(f"{name:16s}", " ".join(f"{v:.2f}" for v in Y[key]))
print(OUT)
