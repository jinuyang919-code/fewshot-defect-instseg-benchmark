"""세 실행기(run_mmdet · run_d2 · run_modern)가 같이 쓰는 평가·시드 도구.

학습 예산 실험(2026-10-03)에서 '3,000 업데이트를 한 번 돌리며 100·300·1,000·3,000
시점마다 평가'하려고 뺐다. 세 실행기 모두 학습률 스케줄러 없이 고정 학습률이라
중간 시점의 모델은 처음부터 그만큼만 학습한 모델과 같다.
"""
import contextlib
import io
import json
import random


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_eval_at(text, updates):
    """'100,300,1000' → [100, 300, 1000, updates]. 비우면 [updates] (기존 동작)."""
    pts = sorted({int(x) for x in (text or "").split(",") if x.strip()} | {updates})
    return [p for p in pts if 0 < p <= updates]


def coco_segm(preds, val_json):
    """COCO mask AP. 예측이 없으면 0 으로 계산한다."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO(str(val_json))
        if preds:
            # loadRes 는 넘긴 dict 에 numpy 정수(area 등)를 써 넣는다. 원본을 지키려고 사본을 준다.
            pred = truth.loadRes([dict(p) for p in preds])
        else:
            pred = COCO()
            pred.dataset = dict(images=truth.dataset["images"],
                                categories=truth.dataset["categories"], annotations=[])
            pred.createIndex()
        ev = COCOeval(truth, pred, "segm")
        ev.params.imgIds = sorted(truth.imgs)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return dict(mask_AP=float(ev.stats[0]), mask_AP50=float(ev.stats[1]),
                mask_AP75=float(ev.stats[2]))
