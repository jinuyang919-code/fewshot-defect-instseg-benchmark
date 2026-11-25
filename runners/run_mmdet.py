#!/usr/bin/env python3
"""MMDetection 모델 하나를 단위 하나에서 돌린다 (선별용).

초기 실행 시험용 실행기를 일반화했다.
  · 데이터: 정제 데이터셋 (`prep.py` 가 만든 `data/<단위>/`)
  · 클래스 수를 단위마다 맞춘다 (04_Hemisphere 는 3클래스)
  · 업데이트 수를 인자로 받는다
  · 평가는 test **전부** (5장이 아니라)

실패해도 `result.json` 에 사유를 남긴다. 실패도 결과다.
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
REPOS = Path(os.environ.get("THIRD_PARTY", Path(__file__).resolve().parents[1] / "third_party")) / "repos"   # 외부 구현체 (README 참고)

CONFIGS = {
    "mask_rcnn": "mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py",
    "cascade_mask_rcnn": "cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py",
    "htc": "htc/htc_r50_fpn_1x_coco.py",
    "ms_rcnn": "ms_rcnn/ms-rcnn_r50-caffe_fpn_1x_coco.py",
    "pointrend": "point_rend/point-rend_r50-caffe_fpn_ms-1x_coco.py",
    "yolact": "yolact/yolact_r50_1xb8-55e_coco.py",
    "solov2": "solov2/solov2_r50_fpn_1x_coco.py",
    "condinst": "condinst/condinst_r50_fpn_ms-poly-90k_coco_instance.py",
    "rtmdet_ins": "rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py",
    "queryinst": "queryinst/queryinst_r50_fpn_1x_coco.py",
    "mask2former": "mask2former/mask2former_r50_8xb2-lsj-50e_coco.py",
}


def set_classes(value, n):
    if isinstance(value, dict):
        for k, v in value.items():
            if k in ("num_classes", "num_things_classes"):
                value[k] = n
            elif k == "num_stuff_classes":
                value[k] = 0
            else:
                set_classes(v, n)
    elif isinstance(value, (list, tuple)):
        for v in value:
            set_classes(v, n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=CONFIGS)
    p.add_argument("unit")
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--attempt", default="01")
    # 학습 예산 실험용 (2026-10-03). 안 주면 기존과 같다: 시드 20260927, 마지막에 한 번 평가.
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument("--eval-at", default="", help="예: 100,300,1000 (마지막 업데이트는 항상 평가)")
    args = p.parse_args()

    data = ROOT / "data" / args.unit
    out = ROOT / "runs" / args.unit / args.model / args.attempt
    out.mkdir(parents=True, exist_ok=True)
    result = dict(model=args.model, unit=args.unit, status="running", stage="import",
                  updates_target=args.updates, started=time.time())

    def save():
        # encoding 을 꼭 지정한다. 지정하지 않으면 계정의 로캘을 따르는데, ASCII 로캘(LANG=C)인 계정에서는
        # 한글(경로·오류 메시지)이 든 결과를 못 써서 **원래 오류가 가려진다** (2026-10-05 실제로 발생).
        (out / "result.json").write_text(json.dumps(result, ensure_ascii=False,
                                                    indent=2, default=str), encoding="utf-8")
    save()
    try:
        import numpy as np
        import torch
        import yaml
        from mmengine.config import Config
        from mmengine.runner import Runner
        from mmdet.apis import inference_detector
        from pycocotools import mask as mask_util

        torch.set_num_threads(4)
        assert torch.cuda.is_available(), "CUDA unavailable"
        cats = json.loads((data / "train.json").read_text())["categories"]
        names = tuple(c["name"] for c in cats)
        catid = [c["id"] for c in cats]
        result.update(stage="configure", classes=list(names))
        save()

        path = REPOS / "mmdetection/configs" / CONFIGS[args.model]
        cfg = Config.fromfile(str(path))
        set_classes(cfg.model, len(names))
        if args.model == "htc":
            # 우리 데이터에는 stuff(semantic) 라벨이 없다. semantic 분기를 뗀다.
            # without-semantic 설정은 공개 체크포인트가 없어서, 표준 설정 + 분기 제거로 간다.
            cfg.model.roi_head.pop("semantic_head", None)
            cfg.model.roi_head.pop("semantic_roi_extractor", None)
            cfg.model.roi_head.pop("semantic_fusion", None)
        if args.model == "mask2former":
            cfg.model.panoptic_head.loss_cls.class_weight = [1.0] * len(names) + [0.1]

        # RTMDet-Ins 는 특징 맵이 32의 배수여야 한다. keep_ratio 로 줄이면 31 같은
        # 크기가 나와 neck 에서 터진다 (13_Cable-break-A 에서 확인, 2026-09-27).
        # **이 모델에만** 패딩을 넣는다. 정규 실험에서는 전 모델에 통일해서 넣을 것.
        pad = [dict(type="Pad", size_divisor=32)] if args.model == "rtmdet_ins" else []
        train_pipe = [dict(type="LoadImageFromFile"),
                      dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
                      dict(type="Resize", scale=(640, 640), keep_ratio=True),
                      *pad, dict(type="PackDetInputs")]
        # rtmdet_ins 는 입력이 32의 배수여야 해서 평가 쪽도 맞춰야 한다.
        # 마스크가 먼저 로드돼야 Resize·Pad 가 함께 적용된다.
        if args.model == "rtmdet_ins":
            val_pipe = [dict(type="LoadImageFromFile"),
                        dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
                        dict(type="Resize", scale=(640, 640), keep_ratio=True),
                        *pad,
                        dict(type="PackDetInputs",
                             meta_keys=("img_id", "img_path", "ori_shape", "img_shape",
                                        "scale_factor"))]
        else:
            val_pipe = [dict(type="LoadImageFromFile"),
                    dict(type="Resize", scale=(640, 640), keep_ratio=True),
                    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
                    # 평가 파이프라인은 Resize 뒤에 LoadAnnotations 가 와서 마스크가
                    # 원본 크기다. 여기에 Pad 를 넣으면 이미지와 크기가 어긋나 터진다
                    # (04_Hemisphere 에서 확인, 2026-09-27). 패딩은 학습에만 넣는다.
                    dict(type="PackDetInputs",
                         meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"))]

        def loader(split, train):
            return dict(batch_size=1, num_workers=0, persistent_workers=False,
                        sampler=dict(type="InfiniteSampler" if train else "DefaultSampler",
                                     shuffle=train),
                        dataset=dict(type="CocoDataset", data_root=str(data),
                                     ann_file=f"{split}.json", metainfo=dict(classes=names),
                                     data_prefix=dict(img=f"{split}/"),
                                     filter_cfg=dict(filter_empty_gt=False, min_size=0),
                                     test_mode=not train,
                                     pipeline=train_pipe if train else val_pipe))

        cfg.train_dataloader = loader("train", True)
        cfg.val_dataloader = cfg.test_dataloader = loader("val", False)
        cfg.train_cfg = dict(type="IterBasedTrainLoop", max_iters=args.updates,
                             val_interval=args.updates + 1)
        cfg.val_cfg = dict(type="ValLoop")
        cfg.test_cfg = dict(type="TestLoop")
        cfg.val_evaluator = cfg.test_evaluator = dict(
            type="CocoMetric", ann_file=str(data / "val.json"), metric="segm", format_only=False)
        cfg.param_scheduler = []
        cfg.optim_wrapper = dict(type="OptimWrapper",
                                 optimizer=dict(type="AdamW", lr=1e-4, weight_decay=1e-4),
                                 clip_grad=dict(max_norm=1.0, norm_type=2))
        cfg.default_hooks = dict(
            timer=dict(type="IterTimerHook"),
            logger=dict(type="LoggerHook", interval=50),
            param_scheduler=dict(type="ParamSchedulerHook"),
            checkpoint=dict(type="CheckpointHook", by_epoch=False,
                            interval=args.updates, max_keep_ckpts=1))
        cfg.custom_hooks = []
        cfg.visualizer = None
        cfg.auto_scale_lr = dict(enable=False)
        cfg.work_dir = str(out)
        cfg.randomness = dict(seed=args.seed, deterministic=False)
        cfg.env_cfg = dict(cudnn_benchmark=False,
                           mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
                           dist_cfg=dict(backend="nccl"))
        cfg.log_processor = dict(type="LogProcessor", window_size=50, by_epoch=False)

        # 공식 COCO 체크포인트를 그대로 쓴다. 클래스 수가 다른 예측 층만 새로 초기화된다.
        meta = yaml.safe_load((path.parent / "metafile.yml").read_text())
        m = [x for x in meta.get("Models", []) if x.get("Config", "").endswith(CONFIGS[args.model])]
        assert m and m[0].get("Weights"), f"체크포인트 없음: {path.name}"
        cfg.load_from = m[0]["Weights"]
        result.update(config=str(path), checkpoint=cfg.load_from,
                      optimizer="AdamW lr=1e-4", batch_size=1, max_side=640, stage="build")
        cfg.dump(str(out / "config.py"))
        save()

        from mmengine.hooks import Hook
        from common_eval import parse_eval_at, coco_segm
        eval_at = parse_eval_at(args.eval_at, args.updates)
        result.update(seed=args.seed, eval_at=eval_at, checkpoints=[])
        images = json.loads((data / "val.json").read_text())["images"]
        last = {}

        def predict_all(model):
            model.eval()
            preds = []
            t1 = time.time()
            with torch.no_grad():
                for im in images:
                    n = os.path.basename(im["file_name"])
                    r = inference_detector(model, str(data / "val" / n)).pred_instances.cpu()
                    for mask, score, label in zip(r.masks, r.scores, r.labels):
                        rle = mask_util.encode(np.asfortranarray(mask.numpy().astype("uint8")))
                        rle["counts"] = rle["counts"].decode()
                        preds.append(dict(image_id=im["id"], category_id=catid[int(label)],
                                          segmentation=rle, score=float(score)))
            torch.cuda.synchronize()
            model.train()
            return preds, time.time() - t1

        clock = dict(t0=None, train=0.0)

        class EvalAt(Hook):
            """eval_at 에 든 업데이트 수를 막 마쳤을 때 val 전부를 평가한다.
            IterBasedTrainLoop 는 after_train_iter 를 부른 뒤에 iter 를 올리므로 batch_idx+1 이 완료 수다."""
            priority = "LOWEST"

            def before_train(self, runner):
                clock["t0"] = time.time()

            def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
                done = batch_idx + 1
                if done not in eval_at:
                    return
                torch.cuda.synchronize()
                clock["train"] += time.time() - clock["t0"]
                result.update(updates_completed=done, train_seconds=clock["train"],
                              stage=f"evaluate@{done}")
                save()
                preds, inf_sec = predict_all(runner.model)
                met = coco_segm(preds, data / "val.json")
                result["checkpoints"].append(dict(updates=done, train_seconds=clock["train"],
                                                  inference_seconds=inf_sec,
                                                  predictions=len(preds), **met))
                last["preds"] = preds
                result["stage"] = "train"
                save()
                clock["t0"] = time.time()

        runner = Runner.from_cfg(cfg)
        runner.model.cfg = cfg                  # inference_detector 가 평가 파이프라인을 읽는다
        runner.register_hook(EvalAt())
        result["stage"] = "train"
        save()
        runner.train()
        (out / "predictions.json").write_text(json.dumps(last.get("preds", [])))
        cp = result["checkpoints"][-1]
        assert cp["updates"] == args.updates, f"마지막 평가가 {cp['updates']} 에서 멈췄다"
        result.update(inference_seconds=cp["inference_seconds"], evaluated_images=len(images),
                      predictions=cp["predictions"], mask_AP=cp["mask_AP"],
                      mask_AP50=cp["mask_AP50"], mask_AP75=cp["mask_AP75"],
                      status="passed", stage="complete")
    except Exception as e:
        result.update(status="failed", error=repr(e), traceback=traceback.format_exc())
    finally:
        try:
            import torch
            result["peak_gpu_memory_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
            result["torch"] = torch.__version__
            result["cuda"] = torch.version.cuda
        except Exception:
            pass
        result["elapsed_seconds"] = time.time() - result["started"]
        save()
    print(f"{args.unit} {args.model}: {result['status']} "
          f"AP={result.get('mask_AP')} {result['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
