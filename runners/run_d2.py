#!/usr/bin/env python3
"""Detectron2 계열 모델을 정제 데이터셋에서 돌린다 (env/pilot_legacy · detectron2 0.6).

모델마다 저장소·설정·클래스 수 키가 달라서 아래 MODELS 표에 적어 둔다.
새 모델을 추가하려면 표에 한 줄 넣으면 된다.

실패해도 사유를 `result.json` 에 남긴다. 실패도 결과다.
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 데이터·결과 폴더의 뿌리. 기본은 이 폴더(선별 실험). 본 실험은 EXP_ROOT 로 다른 폴더를 가리킨다.
ROOT = Path(os.environ.get("EXP_ROOT", HERE))

# 참고 (2026-10-12 확인): Detectron2 계열의 컴파일된 확장 모듈(_C.so · MultiScaleDeformableAttention.so)에는
# 컴파일할 때 쓴 홈 디렉터리 경로가 RUNPATH 로 박혀 있다. 그래도 홈이 없어도 돈다 —
# 필요한 libcudart.so.11.0 은 torch 가 자체로 갖고 있고(torch/lib, SONAME 동일) torch 가 항상 먼저 올라오기 때문이다.
# 홈 경로를 없는 경로로 바꾼 사본으로 직접 불러와 확인했다. 그래서 여기서 따로 미리 올리지 않는다.
PILOT = Path(os.environ.get("THIRD_PARTY", Path(__file__).resolve().parents[1] / "third_party"))   # 외부 구현체와 사전학습 가중치 (README 참고)
REPOS = PILOT / "repos"
CACHE = PILOT / "cache"

MODELS = {
    "sparseinst": dict(
        repo="SparseInst",
        config="configs/sparse_inst_r50_giam.yaml",
        weights=str(CACHE / "sparse_inst_r50_giam.pth"),
        setup="from sparseinst import add_sparse_inst_config as _add",
        num_classes_key="MODEL.SPARSE_INST.DECODER.NUM_CLASSES",
        mask_format="bitmask",
    ),
    "fastinst": dict(
        repo="FastInst",
        config="configs/coco/instance-segmentation/fastinst_R50_ppm-fpn_x3_640.yaml",
        weights=str(CACHE / "fastinst_R50_ppm-fpn_x3_640_37.9.pth"),
        setup="from fastinst import add_fastinst_config as _add",
        num_classes_key="MODEL.SEM_SEG_HEAD.NUM_CLASSES",
        mapper="from fastinst.data.dataset_mappers.fastinst_instance_dataset_mapper import FastInstInstanceDatasetMapper as _M",
        mask_format="bitmask",
    ),
    "maskdino": dict(
        repo="MaskDINO",
        config="configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048_bitmask.yaml",
        weights=str(CACHE / "maskdino_r50.pth"),
        setup="from maskdino import add_maskdino_config as _add",
        num_classes_key="MODEL.SEM_SEG_HEAD.NUM_CLASSES",
        extra_path="maskdino/modeling/pixel_decoder/ops",   # 직접 컴파일한 CUDA 연산
        mapper="from maskdino.data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper as _M",
        mask_format="bitmask",
    ),
    "transfiner": dict(
        # 자체 detectron2 사본을 쓴다. repo 경로를 sys.path 맨 앞에 넣어야 한다.
        repo="transfiner",
        config="configs/transfiner/mask_rcnn_R_50_FPN_1x.yaml",
        weights=str(CACHE / "transfiner_r50_1x.pth"),
        setup="_add = lambda cfg: None",
        num_classes_key=None,
    ),
    "blendmask": dict(
        # AdelaiDet: ml_nms.cu 의 THC(삭제된 API)를 현행 API 로 고쳐 빌드했다 (2026-10-01).
        # COCO 가중치는 원 배포처(cloudstor)가 죽어서 HuggingFace 미러를 쓴다.
        repo="AdelaiDet",
        config="configs/BlendMask/550_R_50_3x.yaml",
        weights=str(CACHE / "blendmask_550_R_50_3x.pth"),
        setup="_add = lambda cfg: None",
        get_cfg="from adet.config import get_cfg as _get",
        num_classes_key="MODEL.FCOS.NUM_CLASSES",
        mask_format="bitmask",
        # 우리 데이터엔 stuff 라벨이 없다. basis 모듈의 의미분할 보조 손실을 끈다.
        overrides={"MODEL.BASIS_MODULE.LOSS_ON": False},
    ),
    "diffusioninst": dict(
        # COCO 가중치가 Baidu 단독이라 못 받는다(2026-09-28 확인). 대신 **DiffusionDet 의
        # COCO 가중치**를 올리고 마스크 분기는 새로 초기화한다 (EoMT 와 같은 방식).
        # → 다른 모델과 사전학습 조건이 다르다는 점을 결과에 반드시 적을 것.
        repo="DiffusionInst",
        config="configs/diffinst.coco.res50.yaml",
        weights=str(CACHE / "diffdet_coco_res50.pth"),
        setup="from diffusioninst import add_diffusioninst_config as _add",
        num_classes_key="MODEL.DiffusionInst.NUM_CLASSES",
        mask_format="bitmask",
    ),
    "panoptic_deeplab": dict(
        repo=None,                       # detectron2 projects 내장
        project="Panoptic-DeepLab",
        panoptic=True,                   # make_panoptic.py 가 만든 파놉틱 정답을 쓴다
        mapper="from panoptic_deeplab import PanopticDeeplabDatasetMapper as _M",
        overrides={"MODEL.PANOPTIC_DEEPLAB.PREDICT_INSTANCES": True},
        config="configs/COCO-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv.yaml",
        weights="https://dl.fbaipublicfiles.com/detectron2/PanopticDeepLab/COCO-PanopticSegmentation/panoptic_deeplab_R_52_os16_mg124_poly_200k_bs64_crop_640_640_coco_dsconv/model_final_5e6da2.pkl",
        setup="from panoptic_deeplab import add_panoptic_deeplab_config as _add",
        num_classes_key=None,
    ),
}


def set_key(cfg, dotted, value):
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        node = getattr(node, p)
    setattr(node, parts[-1], value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", choices=MODELS)
    p.add_argument("unit")
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--attempt", default="01")
    # 학습 예산 실험용 (2026-10-03). 안 주면 기존과 같다: 시드 20260927, 마지막에 한 번 평가.
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument("--eval-at", default="", help="예: 100,300,1000 (마지막 업데이트는 항상 평가)")
    # 2026-10-12: MMDet 실행기와 학습 조건을 맞춘다.
    #   지금까지(이 옵션 없이) Detectron2 계열은 SGD(momentum 0.9) · RandomFlip(+저장소 매퍼의 LSJ)로 돌았다.
    #   `detectron2.solver.build_optimizer` 가 cfg 와 무관하게 항상 SGD 를 만든다는 것을 놓쳤다.
    #   MMDet 계열은 AdamW(lr 1e-4, wd 1e-4) · 증강 없음이다. 그래서 두 계열의 비교가 공정하지 않았다.
    # --matched 를 주면: AdamW(lr 1e-4, wd 1e-4, 전 파라미터 같은 lr) · 증강 없음(긴 변 640 크기 조절만).
    p.add_argument("--check-config", action="store_true",
                   help="GPU 를 쓰지 않고 모델·옵티마이저·증강만 만들어 실제 적용된 조건을 출력하고 끝낸다")
    p.add_argument("--matched", action="store_true",
                   help="MMDet 실행기와 같은 조건: AdamW lr=1e-4 wd=1e-4 · 증강 없음")
    args = p.parse_args()

    spec = MODELS[args.model]
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
        import numpy as np
        import torch
        from pycocotools import mask as mask_util

        if spec["repo"]:
            repo = REPOS / spec["repo"]
        else:
            import detectron2
            repo = Path(detectron2.__file__).resolve().parents[1] / "projects" / spec["project"]
            if not repo.exists():
                repo = REPOS / "detectron2/projects" / spec["project"]
        # 구버전 detectron2(Mask Transfiner 동봉본)는 Pillow 10 에서 사라진 상수를 쓴다.
        from PIL import Image as _PIL
        for old_name, new_name in (("LINEAR", "BILINEAR"), ("CUBIC", "BICUBIC"),
                                   ("ANTIALIAS", "LANCZOS")):
            if not hasattr(_PIL, old_name):
                setattr(_PIL, old_name, getattr(_PIL, new_name))

        # NumPy 1.24 에서 삭제된 별칭 (구 detectron2 가 쓴다)
        for alias, real in (("bool", bool), ("int", int), ("float", float),
                            ("object", object), ("str", str)):
            if not hasattr(np, alias):
                setattr(np, alias, real)

        sys.path.insert(0, str(repo))
        if spec.get("extra_path"):
            sys.path.insert(0, str(repo / spec["extra_path"]))
        ns = {}
        exec(spec["setup"], ns)
        add_cfg = ns["_add"]

        from detectron2.config import get_cfg
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.data.datasets import register_coco_instances
        from detectron2.data import build_detection_train_loader
        from detectron2.solver import build_optimizer
        from detectron2.modeling import build_model
        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.engine import DefaultPredictor

        cats = json.loads((data / "train.json").read_text())["categories"]
        catid = [c["id"] for c in cats]
        names = [f"{args.unit}_{s}" for s in ("train", "val")]
        for n, split in zip(names, ("train", "val")):
            if n in DatasetCatalog.list():
                DatasetCatalog.remove(n)
                MetadataCatalog.remove(n)
            if spec.get("panoptic"):
                recs = json.loads((data / f"panoptic_{split}.json").read_text())
                DatasetCatalog.register(n, lambda r=recs: r)
                md = MetadataCatalog.get(n)
                md.set(
                    thing_classes=[c["name"] for c in cats],
                    stuff_classes=["background"] + [c["name"] for c in cats],
                    thing_dataset_id_to_contiguous_id={c["id"]: i for i, c in enumerate(cats)},
                    stuff_dataset_id_to_contiguous_id={0: 0, **{c["id"]: i + 1
                                                                for i, c in enumerate(cats)}},
                    ignore_label=255, label_divisor=1000,
                    json_file=str(data / f"{split}.json"), image_root=str(data / split),
                    evaluator_type="coco_panoptic_seg",
                )
            else:
                register_coco_instances(n, {}, str(data / f"{split}.json"), str(data / split))

        result["stage"] = "configure"
        save()
        if spec.get("get_cfg"):
            # 저장소가 자체 get_cfg 로 기본값을 등록하는 경우 (AdelaiDet 등)
            gns = {}
            exec(spec["get_cfg"], gns)
            cfg = gns["_get"]()
        else:
            cfg = get_cfg()
            add_cfg(cfg)
        # 저장소마다 이 Detectron2 판에 없는 키를 쓴다 (예: FastInst 의 RESNETS.STEM_TYPE).
        # 병합할 때만 새 키를 허용한다.
        cfg.set_new_allowed(True)
        cfg.merge_from_file(str(repo / spec["config"]))
        cfg.set_new_allowed(False)
        cfg.DATASETS.TRAIN = (names[0],)
        cfg.DATASETS.TEST = (names[1],)
        cfg.DATALOADER.NUM_WORKERS = 0
        cfg.SOLVER.IMS_PER_BATCH = 1
        cfg.SOLVER.MAX_ITER = args.updates
        cfg.SOLVER.BASE_LR = 1e-4
        cfg.SOLVER.CHECKPOINT_PERIOD = args.updates + 1
        cfg.SOLVER.WARMUP_ITERS = 0
        # 저장소마다 자체 gradient clip 타입을 쓴다(D2 0.6 미지원). 우리가 직접 자른다.
        if hasattr(cfg.SOLVER, "CLIP_GRADIENTS"):
            cfg.SOLVER.CLIP_GRADIENTS.ENABLED = False
        cfg.INPUT.MIN_SIZE_TRAIN = (640,)
        cfg.INPUT.MAX_SIZE_TRAIN = 640
        cfg.INPUT.MIN_SIZE_TEST = 640
        cfg.INPUT.MAX_SIZE_TEST = 640
        cfg.INPUT.CROP.ENABLED = False
        cfg.MODEL.WEIGHTS = spec["weights"]
        cfg.MODEL.MASK_ON = True      # 기본 DatasetMapper 가 마스크를 읽게 한다
        # 저장소마다 기대하는 마스크 형식이 다르다 (Transfiner 는 폴리곤을 쓴다)
        cfg.INPUT.MASK_FORMAT = spec.get("mask_format", "polygon")
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(cats)
        if spec.get("panoptic"):
            cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = len(cats) + 1   # background(stuff) 포함
            cfg.MODEL.PANOPTIC_DEEPLAB.STUFF_AREA = 0
        if spec["num_classes_key"]:
            set_key(cfg, spec["num_classes_key"], len(cats))
        # 점수 문턱을 전 모델 공통으로 낮춘다.
        # BlendMask 는 FCOS 헤드의 기본 문턱 0.05 때문에 300 업데이트 수준의 모델에서
        # **예측을 0개** 내놓았다 (2026-10-01 확인). 다른 모델들은 문턱 없이 상위 100개를
        # 내놓으므로 조건이 어긋났다. COCO AP 는 낮은 점수의 예측이 있어도 손해가 없으니
        # 전부 0.001 로 맞춘다. 설정에 없는 키는 건드리지 않는다.
        for key in ("MODEL.ROI_HEADS.SCORE_THRESH_TEST", "MODEL.FCOS.INFERENCE_TH_TEST",
                    "MODEL.RETINANET.SCORE_THRESH_TEST"):
            node, leaf = cfg, None
            try:
                *parts, leaf = key.split(".")
                for q in parts:
                    node = node[q]
                if leaf in node:
                    node[leaf] = 0.001
            except Exception:
                pass
        for k, v in (spec.get("overrides") or {}).items():
            set_key(cfg, k, v)
        if args.matched:
            cfg.INPUT.RANDOM_FLIP = "none"
            # Mask DINO 의 LSJ 매퍼는 IMAGE_SIZE · MIN/MAX_SCALE 로 크기를 흔든다.
            # 1.0 으로 고정하고 캔버스를 640 으로 두면 '긴 변 640 으로 줄이고 남는 곳을 채우기'만 한다.
            for k, v in (("IMAGE_SIZE", 640), ("MIN_SCALE", 1.0), ("MAX_SCALE", 1.0), ("COLOR_AUG_SSD", False)):
                if k in cfg.INPUT:
                    cfg.INPUT[k] = v
        if args.check_config:
            cfg.MODEL.DEVICE = "cpu"
        cfg.OUTPUT_DIR = str(out)
        cfg.SEED = args.seed
        (out / "config.yaml").write_text(cfg.dump())
        result.update(config=spec["config"], checkpoint=spec["weights"],
                      optimizer="(build 후 기록)", matched=bool(args.matched), batch_size=1, max_side=640,
                      classes=[c["name"] for c in cats], stage="build")
        save()

        from common_eval import seed_all, parse_eval_at, coco_segm
        seed_all(args.seed)
        eval_at = parse_eval_at(args.eval_at, args.updates)
        result.update(seed=args.seed, eval_at=eval_at)
        model = build_model(cfg)
        DetectionCheckpointer(model, save_dir=str(out)).resume_or_load(cfg.MODEL.WEIGHTS,
                                                                      resume=False)
        if args.matched:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        else:
            opt = build_optimizer(cfg, model)          # D2 0.6: cfg 와 무관하게 SGD
        g0 = opt.param_groups[0]
        result["optimizer"] = (f"{type(opt).__name__} lr={g0['lr']:g} wd={g0.get('weight_decay', 0):g}"
                               + (f" momentum={g0['momentum']:g}" if "momentum" in g0 else "")
                               + f" · 파라미터 그룹 {len(opt.param_groups)}개")
        if spec.get("mapper"):
            mns = {}
            exec(spec["mapper"], mns)
            # 매퍼마다 생성 인자가 다르다 (cfg, is_train) 또는 (cfg)
            try:
                mapper_obj = mns["_M"](cfg, True)
            except TypeError:
                mapper_obj = mns["_M"](cfg)
            from detectron2.data import transforms as _T
            for attr in ("tfm_gens", "augmentations", "augmentation"):
                augs = getattr(mapper_obj, attr, None)
                if augs is None:
                    continue
                if args.matched:
                    # FastInst 매퍼는 설정과 무관하게 RandomFlip 을 넣는다. 직접 뺀다.
                    kept = [a for a in augs if not isinstance(a, (_T.RandomFlip, _T.RandomCrop))]
                    try:
                        setattr(mapper_obj, attr, type(augs)(kept) if not isinstance(augs, list) else kept)
                    except Exception:
                        setattr(mapper_obj, attr, kept)
                    augs = getattr(mapper_obj, attr)
                result["train_augs"] = [repr(a)[:120] for a in augs]
                if args.matched:
                    assert not any(isinstance(a, _T.RandomFlip) for a in augs), "RandomFlip 이 남아 있다"
            loader = build_detection_train_loader(cfg, mapper=mapper_obj)
        else:
            result["train_augs"] = [f"기본 DatasetMapper · ResizeShortestEdge({cfg.INPUT.MIN_SIZE_TRAIN}, max {cfg.INPUT.MAX_SIZE_TRAIN}) · "
                                    f"RANDOM_FLIP={cfg.INPUT.RANDOM_FLIP} · CROP={cfg.INPUT.CROP.ENABLED}"]
            loader = build_detection_train_loader(cfg)
        save()
        if args.check_config:
            result.update(status="checked", stage="check-config")
            print(f"[점검] {args.model} matched={args.matched}")
            print("  옵티마이저:", result["optimizer"])
            print("  학습 증강 :", result["train_augs"])
            print("  평가 크기 :", cfg.INPUT.MIN_SIZE_TEST, "/", cfg.INPUT.MAX_SIZE_TEST)
            return
        model.train()
        result["stage"] = "train"
        save()
        # 일부 모델은 손실 기록에 EventStorage 컨텍스트를 요구한다 (Mask Transfiner 등)
        from detectron2.utils.events import EventStorage
        import cv2
        images = json.loads((data / "val.json").read_text())["images"]
        holder = {}

        def predict_all():
            """지금 학습 중인 모델로 val 전부를 예측한다.
            DefaultPredictor 의 전처리를 그대로 쓰되, 모델만 학습 중인 것으로 바꿔 끼운다."""
            if "p" not in holder:
                holder["p"] = DefaultPredictor(cfg)
                holder["p"].model = model           # 사전학습 사본은 버리고 학습 모델을 쓴다
                torch.cuda.empty_cache()
            predictor = holder["p"]
            model.eval()
            preds = []
            t1 = time.time()
            for im in images:
                n = os.path.basename(im["file_name"])
                img = cv2.imread(str(data / "val" / n))
                # Panoptic-DeepLab 은 검출된 중심이 없으면 instances 키 자체가 없다
                o = predictor(img)
                if "instances" not in o:
                    continue
                inst = o["instances"].to("cpu")
                if not inst.has("pred_masks"):
                    continue
                for m, s, c in zip(inst.pred_masks, inst.scores, inst.pred_classes):
                    rle = mask_util.encode(np.asfortranarray(m.numpy().astype("uint8")))
                    rle["counts"] = rle["counts"].decode()
                    preds.append(dict(image_id=im["id"], category_id=catid[int(c)],
                                      segmentation=rle, score=float(s)))
            torch.cuda.synchronize()
            model.train()
            return preds, time.time() - t1

        result["checkpoints"] = []
        train_sec = 0.0
        done = 0
        with EventStorage(0):
            t0 = time.time()
            for it, batch in zip(range(args.updates), loader):
                losses = model(batch)
                loss = sum(v for v in losses.values() if torch.is_tensor(v))
                assert torch.isfinite(loss).item(), "손실이 발산했다"
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                done = it + 1
                if done in eval_at:
                    torch.cuda.synchronize()
                    train_sec += time.time() - t0
                    result.update(updates_completed=done, train_seconds=train_sec,
                                  stage=f"evaluate@{done}")
                    save()
                    with torch.no_grad():
                        preds, inf_sec = predict_all()
                    met = coco_segm(preds, data / "val.json")
                    result["checkpoints"].append(dict(updates=done, train_seconds=train_sec,
                                                      inference_seconds=inf_sec,
                                                      predictions=len(preds), **met))
                    result["stage"] = "train"
                    save()
                    t0 = time.time()
        torch.save(model.state_dict(), out / "model_final.pth")
        (out / "predictions.json").write_text(json.dumps(preds))
        last = result["checkpoints"][-1]
        result.update(inference_seconds=last["inference_seconds"], evaluated_images=len(images),
                      predictions=last["predictions"], mask_AP=last["mask_AP"],
                      mask_AP50=last["mask_AP50"], mask_AP75=last["mask_AP75"],
                      status="passed", stage="complete")
    except Exception as e:
        result.update(status="failed", error=repr(e), traceback=traceback.format_exc())
    finally:
        try:
            import torch
            result["peak_gpu_memory_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
        except Exception:
            pass
        result["elapsed_seconds"] = time.time() - result["started"]
        save()
    print(f"{args.unit} {args.model}: {result['status']} AP={result.get('mask_AP')} "
          f"{result['elapsed_seconds']:.0f}s")


if __name__ == "__main__":
    main()
