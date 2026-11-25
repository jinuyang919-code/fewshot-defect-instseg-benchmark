#!/usr/bin/env python3
"""그림 3 — 학습이 상대적으로 잘 된 결함 / 거의 안 된 결함 예시 (심사용 v2 판, 2026-10-20).

기준: 표(데이터셋별 mask AP, 20 에폭)의 데이터셋별 최고값(8개 모델 중 최고) 8 이상 / 8 미만.
8.3 다음 값이 5.6 이라 그 사이로 나눴다 (12개 / 14개). 예시는 묶음마다 4장 (PCB 는 합성이라 뺌).
사진은 make_fig_pipeline_v2.py 의 tile() 과 같다 (결함 면적이 가장 큰 장, 결함 둘레 4:3, 빨간 선 = 정답 마스크).
실제 인쇄 크기(8cm)로 그리므로 fontsize 가 곧 지면 pt.
실행: env/pilot_legacy/bin/python make_fig3_examples.py  ->  fig3_examples.png (600 dpi)
"""
import csv
import os
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

import make_fig_pipeline_v2 as P          # tile(), 데이터 경로 (import 할 때 그림 1 도 다시 그려진다)

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "results", "main_results.csv")
OUT = os.path.join(HERE, "..", "assets", "fig3_examples.png")
W_PT = 8.0 / 2.54 * 72
FS, FS_T = 8.0, 8.5
FONT = "Liberation Sans"
INK, SUB = "#222222", "#555555"
EP = "AP@100"                              # 20 에폭
THR = 8.0
# 이름 두 줄: 부품 / 결함 종류 (본문 "같은 케이블(Cable)이라도 thunderbolt와 break" 와 같은 나눔)
GOOD = [("18_Cable-thunderbolt-B", "Cable", "thunderbolt B"), ("13_Cable-break-A", "Cable", "break A"),
        ("11_Cylinder-Porosity", "Cylinder", "Porosity"), ("07_Wood-pits", "Wood", "Pits")]
POOR = [("24_Electronics-damage-A", "Electronics", "Damage A"), ("26_Lens-Tear-A", "Lens", "Tear A"),
        ("05_Hemisphere-B", "Hemisphere", "B"), ("21_Console-Scratch-A", "Console", "Scratch A")]

rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8")) if r["status"] == "passed"]
models = sorted({r["model"] for r in rows})
assert len(models) == 8


def best(u):
    return max(round(mean(float(r[EP]) for r in rows if r["model"] == m and r["unit"] == u), 1) for m in models)


units = sorted({r["unit"] for r in rows})
B = {u: best(u) for u in units}
assert sum(B[u] >= THR for u in units) == 12 and sum(B[u] < THR for u in units) == 14
assert all(B[u] >= THR for u, *_ in GOOD) and all(B[u] < THR for u, *_ in POOR)

plt.rcParams.update({"font.family": FONT})
fig = plt.figure(figsize=(W_PT / 72, 3))
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
R = fig.canvas.get_renderer()


def width(s, fs=FS, weight="normal"):
    t = ax.text(0, 0, s, fontsize=fs, weight=weight)
    w = t.get_window_extent(R).width * 72 / fig.dpi
    t.remove()
    return w


M_, GAP = 1.0, 3.0
TW = (W_PT - 2 * M_ - 3 * GAP) / 4
TH = TW * 3 / 4
y = -M_
for title, items in ((f"Best mask AP ≥ {THR:g}  (12 datasets)", GOOD), (f"Best mask AP < {THR:g}  (14 datasets)", POOR)):
    ax.text(M_, y, title, fontsize=FS_T, weight="bold", color=INK, va="top")
    y -= 12
    for k, (u, part, var) in enumerate(items):
        x = M_ + k * (TW + GAP)
        ax.imshow(P.tile(u), extent=(x, x + TW, y - TH, y), zorder=3, interpolation="lanczos")
        ax.add_patch(Rectangle((x, y - TH), TW, TH, fill=False, ec="#888888", lw=0.3, zorder=4))
        s = f"{B[u]:.1f}"
        bw = width(s, weight="bold") + 3
        ax.add_patch(FancyBboxPatch((x + 1.2, y - 1.2 - 9.5), bw, 9.5, boxstyle="round,pad=0,rounding_size=1.5",
                                    fc="white", ec="none", alpha=0.9, zorder=5))
        ax.text(x + 1.2 + bw / 2, y - 1.2 - 4.75, s, fontsize=FS, weight="bold", color=INK,
                ha="center", va="center_baseline", zorder=6)
        ax.text(x + TW / 2, y - TH - 1.0, part, fontsize=FS, color=SUB, ha="center", va="top")
        ax.text(x + TW / 2, y - TH - 10.0, var, fontsize=FS, color=SUB, ha="center", va="top")
    for _, part, var in items:                    # 이름이 사진 폭(+틈)을 넘지 않게
        assert max(width(part), width(var)) < TW + GAP - 1, (part, var)
    y -= TH + 21
y += 3
H_PT = -y + M_
fig.set_size_inches(W_PT / 72, H_PT / 72)
ax.set_xlim(0, W_PT); ax.set_ylim(-H_PT, 0)
fig.savefig(OUT, dpi=600, facecolor="white")
print(OUT, f"{W_PT / 72 * 2.54:.2f} x {H_PT / 72 * 2.54:.2f} cm", {u.split('_', 1)[1]: B[u] for u, *_ in GOOD + POOR})
