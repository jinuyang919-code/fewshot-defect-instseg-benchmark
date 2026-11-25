#!/usr/bin/env python3
"""YOLO11-seg · EoMT 를 정제 데이터셋에서 돌린다 (env/pilot_modern).

초기 실행 시험용 실행기를 일반화했다.
  · 단위마다 클래스 수가 다르다 (04_Hemisphere 3클래스)
  · 업데이트 수를 인자로 받는다
  · 평가는 test 전부

⚠ YOLO 라벨은 인스턴스당 폴리곤 하나만 쓸 수 있다. 우리 데이터에는 조각이 여러 개인
어노테이션이 있다 (13_Cable-break-A 는 학습 5장 전부). **조각마다 한 줄씩 쓴다** —
학습 라벨만 달라지고 평가 정답(COCO json)은 그대로다. 결과에 이 점을 적어야 한다.
"""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 데이터·결과 폴더의 뿌리. 기본은 이 폴더(선별 실험). 본 실험은 EXP_ROOT 로 다른 폴더를 가리킨다.
ROOT = Path(os.environ.get("EXP_ROOT", HERE))
PILOT = Path(os.environ.get("THIRD_PARTY", Path(__file__).resolve().parents[1] / "third_party"))   # 외부 구현체와 사전학습 가중치 (README 참고)
REPOS = PILOT / "repos"


def encode(mask, image_id, score, cat_id):
    import numpy as np
    from pycocotools import mask as mask_util
    rle = mask_util.encode(np.asfortranarray(mask.astype("uint8")))
    rle["counts"] = rle["counts"].decode()
    return dict(image_id=image_id, category_id=cat_id, segmentation=rle, score=float(score))


def run_yolo(data, out, result, updates, cats, seed=20260927, merge=False, eval_at=None, save=None):
    import torch
    import yaml
    from ultralytics import YOLO, settings
    settings.update({"sync": False, "wandb": False, "mlflow": False,
                     "clearml": False, "comet": False})
    catid = [c["id"] for c in cats]
    # merge=False (선별·1차 예산 실험): 여러 조각 어노테이션을 조각마다 한 줄로 쓴다.
    #   → YOLO 는 조각을 별개의 결함으로 배우는데 채점은 합친 결함 기준이라 YOLO 에 불리하다.
    # merge=True (2026-10-04~): Ultralytics 의 COCO 변환과 같은 방식(merge_multi_segment)으로
    #   조각들을 가는 선으로 이어 **인스턴스당 폴리곤 하나**로 만든다.
    # 라벨 폴더를 따로 써서 두 방식의 캐시가 섞이지 않게 한다.
    root = data / ("yolo_merged" if merge else "yolo")
    if merge:
        import numpy as np
        from ultralytics.data.converter import merge_multi_segment
    parts_split = 0
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        coco = json.loads((data / f"{split}.json").read_text())
        idx = {c["id"]: i for i, c in enumerate(cats)}
        for im in coco["images"]:
            n = os.path.basename(im["file_name"])
            link = root / "images" / split / n
            if not link.exists():
                link.symlink_to(data / split / n)
            lines = []
            for a in coco["annotations"]:
                if a["image_id"] != im["id"]:
                    continue
                if len(a["segmentation"]) > 1:
                    parts_split += 1
                if merge and len(a["segmentation"]) > 1:
                    pts = np.concatenate(merge_multi_segment(a["segmentation"]), axis=0)
                    pts = pts / np.array([im["width"], im["height"]])
                    lines.append(f"{idx[a['category_id']]} " +
                                 " ".join(f"{v:.6f}" for v in pts.reshape(-1)))
                    continue
                for xy in a["segmentation"]:            # 조각마다 한 줄
                    coords = [f"{v/(im['width'] if i%2==0 else im['height']):.6f}"
                              for i, v in enumerate(xy)]
                    lines.append(f"{idx[a['category_id']]} " + " ".join(coords))
            (root / "labels" / split / Path(n).with_suffix(".txt")).write_text(
                "\n".join(lines) + "\n")
    df = root / "data.yaml"
    df.write_text(yaml.safe_dump(dict(path=str(root), train="images/train", val="images/val",
                                      names={i: c["name"] for i, c in enumerate(cats)})))
    n_train = len(json.loads((data / "train.json").read_text())["images"])
    epochs = max(1, round(updates / n_train))
    result.update(stage="train", checkpoint="yolo11n-seg.pt", variant="YOLO11n-seg",
                  max_side=640, batch_size=1, epochs=epochs, seed=seed,
                  label_mode="merge_multi_segment" if merge else "split_parts",
                  note=(f"다중조각 어노 {parts_split}건을 이어서 인스턴스당 폴리곤 하나로 기록(학습 라벨만)"
                        if merge else
                        f"다중조각 어노 {parts_split}건을 조각별 줄로 기록(학습 라벨만)"))
    images = json.loads((data / "val.json").read_text())["images"]

    def predict_all(weights):
        m = YOLO(str(weights))
        preds = []
        t1 = time.time()
        for im in images:
            n = os.path.basename(im["file_name"])
            p = m.predict(str(data / "val" / n), imgsz=640, device=0, conf=0.001,
                          max_det=100, retina_masks=True, verbose=False)[0]
            if p.masks is not None:
                for mk, sc, c in zip(p.masks.data, p.boxes.conf, p.boxes.cls):
                    preds.append(encode(mk.cpu().numpy() > 0.5, im["id"], sc, catid[int(c)]))
        torch.cuda.synchronize()
        return preds, time.time() - t1

    model = YOLO(str(PILOT / "cache/yolo11n-seg.pt"))
    # 중간 평가 (본 실험, 2026-10-05). 학습률이 고정(lrf=1.0 · 워밍업 0)이라 중간 시점의 모델은
    # 그만큼만 학습한 모델과 같다. 에폭이 끝나 last.pt 가 저장된 직후(on_fit_epoch_end)에 그 가중치로 평가한다.
    mid = sorted(e for e in (eval_at or []) if e < updates)
    clock = dict(t0=time.time(), train=0.0)
    if mid:
        from common_eval import coco_segm
        assert all(e % n_train == 0 for e in mid), f"eval_at 은 {n_train} 의 배수여야 한다: {mid}"
        result["checkpoints"] = []

        def on_fit_epoch_end(trainer):
            done = (trainer.epoch + 1) * n_train
            if done not in mid:
                return
            torch.cuda.synchronize()
            clock["train"] += time.time() - clock["t0"]
            result.update(updates_completed=done, train_seconds=clock["train"], stage=f"evaluate@{done}")
            if save:
                save()
            preds, inf = predict_all(trainer.last)
            result["checkpoints"].append(dict(updates=done, train_seconds=clock["train"],
                                              inference_seconds=inf, predictions=len(preds),
                                              **coco_segm(preds, data / "val.json")))
            result["stage"] = "train"
            if save:
                save()
            clock["t0"] = time.time()

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    t0 = time.time()
    model.train(data=str(df), epochs=epochs, batch=1, nbs=1, imgsz=640, device=0, workers=0,
                project=str(out), name="train", exist_ok=True, seed=seed,
                deterministic=False, pretrained=True, optimizer="AdamW", lr0=1e-4, lrf=1.0,
                warmup_epochs=0, amp=False, mosaic=0, mixup=0, copy_paste=0, degrees=0,
                translate=0, scale=0, shear=0, perspective=0, fliplr=0, flipud=0,
                hsv_h=0, hsv_s=0, hsv_v=0, rect=True, plots=False, save=True, val=False)
    torch.cuda.synchronize()
    train_sec = (clock["train"] + time.time() - clock["t0"]) if mid else (time.time() - t0)
    result.update(train_seconds=train_sec, updates_completed=epochs * n_train, stage="inference")
    preds, inf = predict_all(out / "train/weights/last.pt")
    result["inference_seconds"] = inf
    if mid:
        from common_eval import coco_segm
        result["checkpoints"].append(dict(updates=epochs * n_train, train_seconds=train_sec,
                                          inference_seconds=inf, predictions=len(preds),
                                          **coco_segm(preds, data / "val.json")))
    return preds, images


def run_eomt(data, out, result, updates, cats, seed=20260927, eval_at=None, save=None):
    import sys
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from pycocotools.coco import COCO
    sys.path.insert(0, str(REPOS / "eomt"))
    from models.vit import ViT
    from models.eomt import EoMT
    from training.mask_classification_loss import MaskClassificationLoss

    from common_eval import seed_all, coco_segm
    seed_all(seed)                       # 헤드 초기화가 시드를 따른다 (데이터 순서는 고정)
    eval_at = eval_at or [updates]
    result.update(seed=seed, eval_at=eval_at, checkpoints=[])
    side = 630
    ncls = len(cats)
    catid = [c["id"] for c in cats]
    idx = {c["id"]: i for i, c in enumerate(cats)}
    result.update(stage="train", checkpoint="timm/vit_small_patch14_dinov2",
                  initialization="DINOv2 backbone; segmentation heads initialized",
                  max_side=side, batch_size=1, optimizer="AdamW lr=1e-4")
    model = EoMT(ViT((side, side), patch_size=14, backbone_name="vit_small_patch14_dinov2"),
                 num_classes=ncls, num_q=100).cuda()
    crit = MaskClassificationLoss(num_points=12544, oversample_ratio=3.,
                                  importance_sample_ratio=.75, mask_coefficient=5.,
                                  dice_coefficient=5., class_coefficient=2.,
                                  num_labels=ncls, no_object_coefficient=.1).cuda()
    coco = COCO(str(data / "train.json"))

    def load(im, split):
        n = os.path.basename(im["file_name"])
        x = torch.from_numpy(np.array(Image.open(data / split / n).convert("RGB")))
        x = x.permute(2, 0, 1).float()[None] / 255
        h, w = im["height"], im["width"]
        rh, rw = round(h * side / max(h, w)), round(w * side / max(h, w))
        x = F.interpolate(x, size=(rh, rw), mode="bilinear", align_corners=False)
        return F.pad(x, (0, side - rw, 0, side - rh)).cuda(), (rh, rw)

    train = []
    for im in coco.imgs.values():
        x, (rh, rw) = load(im, "train")
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[im["id"]]))
        m = torch.from_numpy(np.stack([coco.annToMask(a) for a in anns])).float()[None]
        m = F.interpolate(m, size=(rh, rw), mode="nearest")[0]
        m = F.pad(m, (0, side - rw, 0, side - rh)).cuda()
        lab = torch.tensor([idx[a["category_id"]] for a in anns], device="cuda")
        train.append((x, [dict(masks=m, labels=lab)]))

    images = json.loads((data / "val.json").read_text())["images"]

    def predict_all():
        model.eval()
        preds = []
        t1 = time.time()
        with torch.no_grad():
            for im in images:
                x, (rh, rw) = load(im, "val")
                ml, cl = model(x)
                m = F.interpolate(ml[-1], (side, side), mode="bilinear",
                                  align_corners=False)[..., :rh, :rw]
                m = F.interpolate(m, (im["height"], im["width"]), mode="bilinear",
                                  align_corners=False)[0]
                masks = m > 0
                q = (m.sigmoid() * masks).flatten(1).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
                prob = cl[-1][0].softmax(-1)[:, :ncls]          # 마지막 칸은 no-object
                score, label = (prob * q[:, None]).max(-1)
                for mask, s, l in zip(masks, score, label):
                    preds.append(encode(mask.cpu().numpy(), im["id"], s, catid[int(l)]))
        torch.cuda.synchronize()
        model.train()
        return preds, time.time() - t1

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    model.train()
    t0 = time.time()
    train_sec = 0.0
    hist = []
    preds = []
    for step in range(updates):
        x, t = train[step % len(train)]
        ml, cl = model(x)
        losses = {}
        for layer, (m, c) in enumerate(zip(ml, cl)):
            losses.update({f"{k}_{layer}": v for k, v in crit(m, t, c).items()})
        loss = crit.loss_total(losses, lambda *a, **k: None)
        assert torch.isfinite(loss).item(), "nonfinite loss"
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        hist.append(float(loss.detach()))
        result["updates_completed"] = step + 1
        if step + 1 in eval_at:
            torch.cuda.synchronize()
            train_sec += time.time() - t0
            result.update(train_seconds=train_sec, stage=f"evaluate@{step + 1}")
            if save:
                save()
            preds, inf_sec = predict_all()
            met = coco_segm(preds, data / "val.json")
            result["checkpoints"].append(dict(updates=step + 1, train_seconds=train_sec,
                                              inference_seconds=inf_sec,
                                              predictions=len(preds), **met))
            result["inference_seconds"] = inf_sec
            if save:
                save()
            t0 = time.time()
    (out / "losses.json").write_text(json.dumps(hist))
    return preds, images


def run_rfdetr(data, out, result, updates, cats):
    """RF-DETR Seg. 9/22 파일럿은 AMP 가 켜진 채로 돌아 V100(Volta)에서 backward 가 실패했다
    (`GET was unable to find an engine...`). Volta 는 bf16 하드웨어 지원이 없다. **amp=False 로 끈다.**"""
    import torch
    from rfdetr import RFDETRSegSmall
    catid = [c["id"] for c in cats]
    root = data / "rfdetr"
    # RF-DETR 은 매 에폭 끝에 `valid` 를 통째로 평가한다. 04_Hemisphere 는 val 이 206장이라
    # 학습 3분 + 내부 평가 150분이 되어 timeout 에 걸렸다 (2026-10-01 확인).
    # **우리가 보고하는 AP 는 학습이 끝난 뒤 model.predict + COCOeval 로 따로 낸다.**
    # 내부 평가 결과는 쓰지 않고(run_test=False · early_stopping=False · use_ema=False
    # 라서 체크포인트 선택에도 안 쓰인다), 그래서 `valid` 를 train 과 같게 둔다.
    # 학습 자체와 우리 수치는 바뀌지 않는다.
    for dest, src in (("train", "train"), ("valid", "train"), ("test", "val")):
        folder = root / dest
        folder.mkdir(parents=True, exist_ok=True)
        coco = json.loads((data / f"{src}.json").read_text())
        (folder / "_annotations.coco.json").write_text(json.dumps(coco))
        for im in coco["images"]:
            n = os.path.basename(im["file_name"])
            link = folder / n
            if not link.exists():
                link.symlink_to(data / src / n)
    n_train = len(json.loads((data / "train.json").read_text())["images"])
    epochs = max(1, round(updates / n_train))
    result.update(stage="train", checkpoint="rf-detr-seg-small.pt", variant="RFDETRSegSmall",
                  max_side=384, batch_size=1, epochs=epochs, note="amp=False 로 강제 (Volta)")
    old = os.getcwd()
    os.chdir(PILOT / "cache")
    try:
        model = RFDETRSegSmall(device="cuda")
    finally:
        os.chdir(old)
    t0 = time.time()
    model.train(dataset_dir=str(root), dataset_file="roboflow", epochs=epochs, batch_size=1,
                grad_accum_steps=1, lr=1e-4, lr_encoder=1e-4, resolution=384, num_workers=0,
                output_dir=str(out / "train"), multi_scale=False, expanded_scales=False,
                do_random_resize_via_padding=False, aug_config={}, use_ema=False, wandb=False,
                tensorboard=False, run_test=False, early_stopping=False, seed=20260927,
                amp=False)
    torch.cuda.synchronize()
    result.update(train_seconds=time.time() - t0, updates_completed=epochs * n_train,
                  stage="inference")
    preds = []
    images = json.loads((data / "val.json").read_text())["images"]
    t0 = time.time()
    for im in images:
        n = os.path.basename(im["file_name"])
        pr = model.predict(str(data / "val" / n), threshold=0.001)
        if pr.mask is not None:
            for m, s, c in zip(pr.mask, pr.confidence, pr.class_id):
                i = int(c) - 1                      # roboflow 형식은 0번이 배경
                if 0 <= i < len(catid):
                    preds.append(encode(m, im["id"], s, catid[i]))
    torch.cuda.synchronize()
    result["inference_seconds"] = time.time() - t0
    return preds, images


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=["yolo11_seg", "eomt", "rf_detr_seg"])
    p.add_argument("unit")
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--attempt", default="01")
    # 학습 예산 실험용 (2026-10-03). 지금은 EoMT 만 지원한다.
    # YOLO11·RF-DETR 는 자체 학습률 스케줄이 있어 중간 평가가 '그만큼만 학습한 모델'과 같지 않다.
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument("--eval-at", default="")
    p.add_argument("--yolo-merge", action="store_true",
                   help="YOLO: 여러 조각 어노테이션을 하나의 폴리곤으로 잇는다 (기본은 조각별 줄 — 옛 결과 재현용)")
    args = p.parse_args()

    data = ROOT / "data" / args.unit
    out = ROOT / "runs" / args.unit / args.model / args.attempt
    out.mkdir(parents=True, exist_ok=True)
    result = dict(model=args.model, unit=args.unit, status="running", stage="import",
                  updates_target=args.updates, started=time.time())

    def save():
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False,
                                                    indent=2, default=str),
                                         encoding="utf-8")
    save()
    try:
        cats = json.loads((data / "train.json").read_text())["categories"]
        result["classes"] = [c["name"] for c in cats]
        fn = {"yolo11_seg": run_yolo, "eomt": run_eomt, "rf_detr_seg": run_rfdetr}[args.model]
        if args.model == "eomt":
            from common_eval import parse_eval_at
            preds, images = fn(data, out, result, args.updates, cats, seed=args.seed,
                               eval_at=parse_eval_at(args.eval_at, args.updates), save=save)
        elif args.model == "yolo11_seg":
            # --eval-at 를 주면 그 시점마다 평가한다 (본 실험). 안 주면 마지막에 한 번 (선별·예산 실험).
            from common_eval import parse_eval_at
            preds, images = fn(data, out, result, args.updates, cats, seed=args.seed,
                               merge=args.yolo_merge, save=save,
                               eval_at=parse_eval_at(args.eval_at, args.updates) if args.eval_at else None)
        else:
            assert not args.eval_at and args.seed == 20260927, \
                f"{args.model} 는 --seed/--eval-at 를 지원하지 않는다"
            preds, images = fn(data, out, result, args.updates, cats)
        result.update(evaluated_images=len(images), predictions=len(preds), stage="evaluate")
        save()

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        (out / "predictions.json").write_text(json.dumps(preds))
        truth = COCO(str(data / "val.json"))
        if preds:
            pred = truth.loadRes(preds)
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
        result.update(mask_AP=float(ev.stats[0]), mask_AP50=float(ev.stats[1]),
                      mask_AP75=float(ev.stats[2]), status="passed", stage="complete")
    except Exception as e:
        result.update(status="failed", error=repr(e), traceback=traceback.format_exc())
    finally:
        try:
            import torch
            result["peak_gpu_memory_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
            result["torch"] = torch.__version__
        except Exception:
            pass
        result["elapsed_seconds"] = time.time() - result["started"]
        save()
    print(f"{args.unit} {args.model}: {result['status']} AP={result.get('mask_AP')} "
          f"{result['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
