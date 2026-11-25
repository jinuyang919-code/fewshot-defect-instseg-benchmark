#!/usr/bin/env python3
"""판정 결과를 반영한 정제 데이터셋을 만든다.

규칙 (2026-09-27 결정)
  · 판정 **가** 인 덩어리는 **파일명이 가장 빠른 한 장만 남기고** 나머지를 뺀다
  · 판정 **나** · 덩어리에 안 묶인 장은 전부 남긴다
  · split 은 남은 장으로 **새로 뽑는다** (Round 1·2·3 · 단위마다 train 5장, 나머지 test)

원본(`03_data/instance/Experiment_Dataset`)은 **읽기만** 한다. 새 폴더에 복사본을 만든다.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import shutil
from collections import defaultdict

SRC = os.environ.get("VISION_UNITS", str(Path(__file__).resolve().parents[1] / "workdir" / "units_raw"))   # VISION 원본을 단위별로 묶은 폴더


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-per-round", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--min-images", type=int, default=10,
                    help="이 수 미만이면 '단위 탈락' 으로 표시만 한다 (빼지는 않는다)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # 1) 판정 읽기 → 뺄 장 정하기
    groups = defaultdict(lambda: {"imgs": [], "v": "", "memo": ""})
    for r in csv.DictReader(open(args.groups, encoding="utf-8-sig")):
        k = (r["단위"], r["덩어리"])
        groups[k]["imgs"].append(r["이미지"])
        if r["판정"]:
            groups[k]["v"], groups[k]["memo"] = r["판정"], r["메모"]

    drop, reason = defaultdict(set), {}
    for (unit, gid), g in groups.items():
        if g["v"] != "가":
            continue
        imgs = sorted(g["imgs"])
        keep = imgs[0]                      # 파일명이 가장 빠른 것
        for n in imgs[1:]:
            drop[unit].add(n)
            reason[(unit, n)] = (gid, keep)

    units = sorted(d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC, d, "images")))
    manifest, summary = [], []
    for unit in units:
        allimg = sorted(os.listdir(os.path.join(SRC, unit, "images")))
        kept = [n for n in allimg if n not in drop[unit]]
        summary.append((unit, len(allimg), len(allimg) - len(kept), len(kept)))
        for n in allimg:
            if n in drop[unit]:
                gid, keeper = reason[(unit, n)]
                manifest.append({"단위": unit, "이미지": n, "상태": "제거",
                                 "덩어리": gid, "사유": f"{keeper} 와 중복(가) — 파일명이 늦어 제외"})
            else:
                gid = next((g for (u, g), v in groups.items()
                            if u == unit and n in v["imgs"]), "")
                manifest.append({"단위": unit, "이미지": n, "상태": "남김",
                                 "덩어리": gid, "사유": ""})

        if args.dry_run:
            continue

        # 2) 이미지 복사
        dst = os.path.join(args.out, unit)
        os.makedirs(os.path.join(dst, "images"), exist_ok=True)
        os.makedirs(os.path.join(dst, "splits"), exist_ok=True)
        for n in kept:
            s = os.path.join(SRC, unit, "images", n)
            d = os.path.join(dst, "images", n)
            if not os.path.exists(d):
                shutil.copy2(s, d)

        # 3) annotations.json 걸러내기 (id 는 원본 그대로 둔다 — 추적 가능하게)
        with open(os.path.join(SRC, unit, "annotations.json")) as f:
            c = json.load(f)
        keepset = set(kept)
        imgs = [im for im in c["images"] if os.path.basename(im["file_name"]) in keepset]
        ids = {im["id"] for im in imgs}
        anns = [a for a in c["annotations"] if a["image_id"] in ids]
        c["images"], c["annotations"] = imgs, anns
        c.setdefault("info", {})
        c["info"]["dedup"] = {
            "source": "03_data/instance/Experiment_Dataset",
            "rule": "판정 '가' 덩어리는 파일명이 가장 빠른 한 장만 남김",
            "removed": len(allimg) - len(kept),
            "date": "2026-09-27",
        }
        with open(os.path.join(dst, "annotations.json"), "w") as f:
            json.dump(c, f)

        # 4) split 새로 뽑기 — 단위·Round 별 고정 시드, Round 간 train 조합이 겹치지 않게
        pool = sorted(kept)
        used = []
        for rd in range(1, args.rounds + 1):
            rng = random.Random(f"{unit}|round{rd}|2026-09-27")
            ok = None
            for _ in range(200):
                cand = sorted(rng.sample(pool, min(args.train_per_round, len(pool))))
                if cand not in used:
                    ok = cand
                    break
            ok = ok or sorted(rng.sample(pool, min(args.train_per_round, len(pool))))
            used.append(ok)
            test = [n for n in pool if n not in set(ok)]
            for name, rows in (("train", ok), ("test", test)):
                with open(os.path.join(dst, "splits", f"round{rd}_{name}.txt"), "w") as f:
                    f.write("\n".join(rows) + "\n")

    # 5) 요약·매니페스트
    if not args.dry_run:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "제거_매니페스트.csv"), "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["단위", "이미지", "상태", "덩어리", "사유"])
            w.writeheader()
            w.writerows(manifest)

    print(f"{'단위':<26}{'원본':>5}{'제거':>5}{'남음':>5}")
    t = [0, 0, 0]
    for unit, a, d, k in summary:
        flag = "  🔴 10장 미만" if k < args.min_images else ""
        print(f"{unit:<26}{a:>5}{d:>5}{k:>5}{flag}")
        t[0] += a
        t[1] += d
        t[2] += k
    print(f"{'합계':<26}{t[0]:>5}{t[1]:>5}{t[2]:>5}")
    if not args.dry_run:
        print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
