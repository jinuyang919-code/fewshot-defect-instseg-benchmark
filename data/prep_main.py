#!/usr/bin/env python3
"""본 실험 데이터 준비 — 26단위 × Round 1~3.

정제 데이터셋(`build_dedup_dataset.py` 의 결과)에서 단위·Round 마다 train(5장)/val(나머지 전부)을 만든다.
이미지는 심볼릭 링크다. 아무것도 복사·수정하지 않는다.

  data/<단위>__R<k>/train.json · val.json · train/ · val/ · manifest.json

제외 단위는 정제 데이터셋 폴더의 `제외단위.txt` 를 따른다 (10장 미만 → 25_Electronics-damage-B).
"""
import hashlib
import json
import os
from pathlib import Path

HERE = Path(os.environ.get("EXP_ROOT", Path(__file__).resolve().parents[1] / "workdir"))   # 학습·평가 묶음과 결과가 쌓이는 곳
SRC = Path(os.environ.get("UNITS_DEDUP", HERE / "units_dedup"))                              # build_dedup_dataset.py 의 결과
ROUNDS = (1, 2, 3)


def units():
    ex = {l.split("\t")[0] for l in (SRC / "제외단위.txt").read_text().splitlines() if l.strip()}
    return [p.name for p in sorted(SRC.iterdir()) if p.is_dir() and p.name[:2].isdigit() and p.name not in ex]


def build(unit, k):
    su = SRC / unit
    coco = json.loads((su / "annotations.json").read_text())
    train = (su / f"splits/round{k}_train.txt").read_text().split()
    val = (su / f"splits/round{k}_test.txt").read_text().split()
    names_all = {os.path.basename(im["file_name"]) for im in coco["images"]}
    assert len(train) == 5 and not set(train) & set(val), (unit, k)
    assert set(train) | set(val) == names_all, f"{unit} R{k}: 분할이 전체와 다르다"
    out = HERE / "data" / f"{unit}__R{k}"
    out.mkdir(parents=True, exist_ok=True)
    stat = {}
    for split, names in (("train", train), ("val", val)):
        want = set(names)
        images = [im for im in coco["images"] if os.path.basename(im["file_name"]) in want]
        assert len(images) == len(want), f"{unit} R{k} {split}"
        ids = {im["id"] for im in images}
        anns = [a for a in coco["annotations"] if a["image_id"] in ids]
        (out / f"{split}.json").write_text(json.dumps(
            dict(info=coco.get("info", {}), licenses=[], images=images,
                 annotations=anns, categories=coco["categories"])))
        d = out / split
        d.mkdir(exist_ok=True)
        for im in images:
            n = os.path.basename(im["file_name"])
            link = d / n
            if not link.exists():
                link.symlink_to(su / "images" / n)
        stat[split] = (len(images), len(anns))
    (out / "manifest.json").write_text(json.dumps(dict(
        unit=unit, round=k, source=str(su), train=sorted(train), val_count=len(val),
        train_annotations=stat["train"][1], val_annotations=stat["val"][1],
        categories=[(c["id"], c["name"]) for c in coco["categories"]],
        annotation_sha256=hashlib.sha256((su / "annotations.json").read_bytes()).hexdigest()),
        indent=2, ensure_ascii=False))
    return stat


if __name__ == "__main__":
    os.umask(0o002)
    us = units()
    tot_val = 0
    for u in us:
        row = []
        for k in ROUNDS:
            s = build(u, k)
            row.append(f"R{k} 학습 {s['train'][0]}장/{s['train'][1]}개 · 평가 {s['val'][0]}장")
            if k == 1:
                tot_val += s["val"][0]
        print(f"{u:<34}" + " | ".join(row))
    print(f"\n단위 {len(us)}개 × Round {len(ROUNDS)} = {len(us) * len(ROUNDS)} 묶음 · 단위당 평가 장수 합 {tot_val}")
