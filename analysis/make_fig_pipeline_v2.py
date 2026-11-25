#!/usr/bin/env python3
"""그림 1 — 실험 절차 (심사용 v2 판, 2026-10-20).

교수님 13차 요청에 맞춘 판:
  · 인쇄 폭 8cm 에서 글씨 8pt 이상 (좌표 1 단위 = 1pt 로 그려서 fontsize 가 곧 지면 pt)
  · 위쪽: 26개 데이터셋마다 대표 사진 1장 (결함 면적이 가장 큰 장, 결함 둘레를 같은 4:3 비율로 잘라 같은 크기로)
    빨간 선은 어노테이션 마스크 경계. 자르는 것은 그림에 보여 주기 위한 것이고 학습 데이터와는 무관하다
  · 부품 묶음 · 데이터셋 수는 표 2 와 같은 순서 (결과표에서 계산)
  · 접근 방식 이름은 본문 2장의 다섯 갈래를 영어로 (그림 안 글자는 영문)
  · "at each checkpoint …" 같은 긴 설명은 뺌 (캡션 · 본문에 있음)
모델 칩 색은 그림 2(make_fig2_8pt.py)의 선 색과 같다.
이전 판: make_fig_pipeline_1col.py (한 단 폭, 사진 5장, 영문)
실행: env/pilot_legacy/bin/python make_fig_pipeline_v2.py  ->  fig_pipeline_v2.png (600 dpi)
"""
import json
import os
from collections import OrderedDict

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from pycocotools import mask as M

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("UNITS_DEDUP", os.path.join(HERE, "..", "workdir", "units_dedup"))
OUT = os.path.join(HERE, "..", "assets", "fig1_pipeline.png")
W_PT = 8.0 / 2.54 * 72          # 8cm
FS, FS_T = 8.0, 8.5             # 본문 글씨 · 칸 제목 (pt)
KO, EN = "Liberation Sans", "Liberation Sans"    # 영문판 (한글판: make_fig_pipeline_v2_한글판.py)
INK, SUB = "#222222", "#555555"
TRAIN, TEST = "#5B4BB7", "#3A8A87"

# 표 2 와 같은 부품 순서 · 데이터셋 묶음 (build_draft.py 의 PARTS 순서. 수는 아래 assert 로 확인)
GROUPS = OrderedDict([
    ("Cable-thunderbolt", ["17_Cable-thunderbolt-A", "18_Cable-thunderbolt-B", "19_Cable-thunderbolt-G"]),
    ("PCB", ["23_PCB_1-missing_hole"]),
    ("Cylinder", ["10_Cylinder-Chip", "11_Cylinder-Porosity", "12_Cylinder-RCS"]),
    ("Screw", ["09_Screw-defect"]),
    ("Cable-break", ["13_Cable-break-A", "14_Cable-break-C", "15_Cable-break-D", "16_Cable-break-E"]),
    ("Wood", ["06_Wood-impurities", "07_Wood-pits"]),
    ("Electronics", ["24_Electronics-damage-A"]),
    ("Capacitor", ["08_Capacitor"]),
    ("Casting", ["01_Casting-Inclusoes-A", "02_Casting-Inclusoes-C-1", "03_Casting-Rechupe-D"]),
    ("Hemisphere", ["04_Hemisphere-A+C+D", "05_Hemisphere-B"]),
    ("Lens", ["26_Lens-Tear-A", "27_Lens-Tear-B"]),
    ("Console", ["20_Console-Collision-B", "21_Console-Scratch-A", "22_Console-Scratch-B"]),
])
# 실험에 쓴 26개 = 결과표의 단위 (정제 데이터셋 폴더에는 제외된 25_Electronics-damage-B 도 있다)
import csv
CSV = os.path.join(HERE, "..", "results", "main_results.csv")
UNITS = sorted({r["unit"] for r in csv.DictReader(open(CSV, encoding="utf-8")) if r["status"] == "passed"})
assert sorted(u for us in GROUPS.values() for u in us) == UNITS and len(UNITS) == 26, UNITS

MODELS = [("Two-stage", [("Mask R-CNN", "#1baf7a"), ("HTC", "#2a78d6")]),
          ("Refinement", [("Mask Transfiner", "#4a3aa7")]),
          ("Query-based", [("Mask2Former", "#eb6834"), ("Mask DINO", "#008300"), ("FastInst", "#e34948")]),
          ("Box-free", [("SOLOv2", "#e87ba4")]),
          ("Real-time", [("YOLO11-seg", "#eda100")])]

TW, TH = 21.2, 15.9             # 사진 한 칸 (pt, 4:3)
# 화면 녹화 워터마크(BANDICAM)가 위쪽 띠에 찍힌 단위 — 그 띠를 빼고 자른다
WATERMARK_TOP = {"13_Cable-break-A": 0.07, "17_Cable-thunderbolt-A": 0.07}
TILE_PX = (240, 180)


def tile(unit):
    """결함 면적이 가장 큰 장에서 결함 둘레를 4:3 으로 잘라 정답 경계를 그린다."""
    a = json.load(open(os.path.join(DATA, unit, "annotations.json")))
    anns = {}
    for x in a["annotations"]:
        anns.setdefault(x["image_id"], []).append(x)
    im = max(a["images"], key=lambda i: sum(x["area"] for x in anns.get(i["id"], [])))
    img = cv2.imread(os.path.join(DATA, unit, "images", im["file_name"]))
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for x in anns[im["id"]]:
        s = x["segmentation"]
        rle = M.frPyObjects(s, h, w) if isinstance(s, list) else s
        mask |= M.decode(M.merge(rle) if isinstance(rle, list) else rle).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    cw = max(bw * 1.6, bh * 1.6 * 4 / 3, w * 0.25)          # 결함이 보일 만큼 확대하되 너무 좁지 않게
    top = int(h * WATERMARK_TOP.get(unit, 0))
    cw = min(cw, w, (h - top) * 4 / 3); ch = cw * 3 / 4
    x0 = int(np.clip(cx - cw / 2, 0, w - cw)); y0 = int(np.clip(cy - ch / 2, top, h - ch))
    x1, y1 = x0 + int(cw), y0 + int(ch)
    crop = cv2.resize(img[y0:y1, x0:x1], TILE_PX, interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask[y0:y1, x0:x1], TILE_PX, interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, cs, -1, (40, 40, 230), 3)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


fig = plt.figure(figsize=(W_PT / 72, 4))
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
R = fig.canvas.get_renderer()


def width(s, fs=FS, font=KO, weight="normal"):
    t = ax.text(0, 0, s, fontsize=fs, family=font, weight=weight)
    w = t.get_window_extent(R).width * 72 / fig.dpi
    t.remove()
    return w


def text(x, y, s, fs=FS, font=KO, weight="normal", color=INK, **kw):
    ax.text(x, y, s, fontsize=fs, family=font, weight=weight, color=color, **kw)


def panel(x, y_top, w, h, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y_top - h), w, h, boxstyle="round,pad=0,rounding_size=3",
                                fc=fc, ec=ec, lw=0.7, zorder=1))


def arrow(p, q, color=INK):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.8, color=color,
                                 zorder=5, shrinkA=0, shrinkB=0))


M_ = 2.0                                   # 바깥 여백
y = -M_
# ── 1. 데이터셋 ─────────────────────────────────────────────
text(M_, y, "VISION dataset", fs=FS_T, weight="bold", va="top")
text(M_ + width("VISION dataset", FS_T, weight="bold") + 4, y - 0.4,
     "12 parts, 26 datasets, 1,125 images", color=SUB, va="top")
y -= 10.5
GAP_T, GAP_G = 0.8, 2.6
slots = []
for g, us in GROUPS.items():
    tw = len(us) * TW + (len(us) - 1) * GAP_T
    slots.append((g, us, max(tw, width(g)), tw))
rows, cur, cw = [], [], 0
for s in slots:                            # 한 줄에 들어갈 만큼씩
    need = s[2] + (GAP_G if cur else 0)
    if cur and cw + need > W_PT - 1.0:          # 사진 줄은 바깥 여백을 0.5pt 까지 쓴다
        rows.append(cur); cur, cw = [], 0; need = s[2]
    cur.append(s); cw += need
rows.append(cur)
for row in rows:
    tot = sum(s[2] for s in row) + GAP_G * (len(row) - 1)
    x = (W_PT - tot) / 2
    for g, us, sw, tw in row:
        tx = x + (sw - tw) / 2
        for u in us:
            ax.imshow(tile(u), extent=(tx, tx + TW, y - TH, y), zorder=3, interpolation="lanczos")
            ax.add_patch(Rectangle((tx, y - TH), TW, TH, fill=False, ec="#888888", lw=0.3, zorder=4))
            tx += TW + GAP_T
        text(x + sw / 2, y - TH - 0.8, g, color=SUB, ha="center", va="top")
        x += sw + GAP_G
    y -= TH + 10.2
y_data_bottom = y + 1.5

# ── 2. 퓨샷 분할 | 학습 · 평가 ────────────────────────────────
y -= 2
WS = 104.0
x_s, x_t = M_, M_ + WS + 7
WT = W_PT - M_ - x_t
SQ, SQG = 5.4, 1.3
rows_ = {"R1": [0, 3, 6, 9, 11], "R2": [1, 3, 5, 8, 10], "R3": [2, 4, 5, 7, 11]}
H2 = 66.0
panel(x_s, y, WS, H2, "#EEF5E4", "#9CC26E")
panel(x_t, y, WT, H2, "#DCE8F8", "#5B8FD4")
text(x_s + WS / 2, y - 3, "Few-shot split", fs=FS_T, weight="bold", ha="center", va="top")
text(x_s + WS / 2, y - 12.5, "3 rounds per dataset", color=SUB, ha="center", va="top")
for k, (lab, tr) in enumerate(rows_.items()):
    yy = y - 27 - k * 8.0
    text(x_s + 5, yy, lab, font=EN, weight="bold", color=SUB, va="center")
    for j in range(12):
        on = j in tr
        ax.add_patch(Rectangle((x_s + 18 + j * (SQ + SQG), yy - SQ / 2), SQ, SQ, fc=TRAIN if on else "white",
                               ec=TRAIN if on else TEST, lw=0.5, zorder=3))
ly = y - 51.5
ax.add_patch(Rectangle((x_s + 6, ly - SQ / 2), SQ, SQ, fc=TRAIN, ec=TRAIN, zorder=3))
text(x_s + 6 + SQ + 2.5, ly, "train: 5", va="center")
xl = x_s + 6 + SQ + 2.5 + width("train: 5") + 6
ax.add_patch(Rectangle((xl, ly - SQ / 2), SQ, SQ, fc="white", ec=TEST, lw=0.5, zorder=3))
text(xl + SQ + 2.5, ly, "test: the rest", va="center")
text(x_s + WS / 2, ly - 8.6, "no validation set", color=SUB, ha="center", va="center")

text(x_t + WT / 2, y - 3, "Training & evaluation", fs=FS_T, weight="bold", ha="center", va="top")
for k, s in enumerate(["AdamW, lr 1e-4 (constant)", "batch 1, long side 640 px", "no augmentation"]):
    text(x_t + WT / 2, y - 13 - k * 9.6, s, ha="center", va="top")
yl = y - 53.5
xs = [x_t + WT * f for f in (0.18, 0.39, 0.60, 0.81)]
ax.plot([x_t + 7, x_t + WT - 5], [yl, yl], color=INK, lw=0.7, zorder=3)
arrow((x_t + WT - 6, yl), (x_t + WT - 3.5, yl))
for xx, e in zip(xs, ["20", "60", "200", "600"]):
    ax.add_patch(Circle((xx, yl), 2.0, fc="#FBE9E9", ec="#DD7777", lw=0.7, zorder=4))
    text(xx, yl + 3.2, e, font=EN, weight="bold", ha="center", va="bottom")
text(x_t + WT / 2, yl - 3.5, "mask AP at each epoch", color=SUB, ha="center", va="top")
for w_ in (width("mask AP at each epoch"), width("AdamW, lr 1e-4 (constant)"), width("batch 1, long side 640 px")):
    assert w_ < WT - 4, w_
arrow((x_s + WS + 0.5, y - H2 / 2), (x_t - 0.5, y - H2 / 2), color=TEST)
arrow((W_PT / 2, y_data_bottom), (W_PT / 2, y + 0.2), color=SUB)
y -= H2 + 4.5

# ── 3. 8개 모델 ──────────────────────────────────────────────
H3 = 11 + 9.1 * len(MODELS) + 1.5
panel(M_, y, W_PT - 2 * M_, H3, "#F4F4F2", "#8A8A8A")
arrow((x_t + WT / 2, y + 0.2), (x_t + WT / 2, y + 5.3), color=INK)
text(W_PT / 2, y - 3, "8 models (COCO-pretrained)", fs=FS_T, weight="bold", ha="center", va="top")
LW = max(width(g) for g, _ in MODELS)
for k, (g, ms) in enumerate(MODELS):
    yy = y - 16.0 - k * 9.1
    text(M_ + 4, yy, g, color=SUB, va="center")
    x = M_ + 4 + LW + 4
    for name, c in ms:
        cw_ = width(name, font=EN) + 4.5
        ax.add_patch(FancyBboxPatch((x, yy - 4.2), cw_, 8.4, boxstyle="round,pad=0,rounding_size=2.5",
                                    fc=c, ec="none", zorder=3))
        text(x + cw_ / 2, yy, name, font=EN, color=INK if c == "#eda100" else "white", ha="center",
             va="center_baseline", zorder=4)
        x += cw_ + 2
    assert x < W_PT - M_ - 2, (g, x)
y -= H3 + M_

H_PT = -y
fig.set_size_inches(W_PT / 72, H_PT / 72)
ax.set_xlim(0, W_PT); ax.set_ylim(-H_PT, 0)
fig.savefig(OUT, dpi=600, facecolor="white")
print(OUT, f"{W_PT / 72 * 2.54:.2f} x {H_PT / 72 * 2.54:.2f} cm", "rows:", [[s[0] for s in r] for r in rows])
