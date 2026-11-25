#!/usr/bin/env python3
"""본 실험 관리 스크립트 — 모델 8개 × 26단위 × Round 3 = 624회.

쓰는 법 (자세한 건 README.md):
    python3 main_driver.py --who A --gpus 0,1,2,3
    python3 main_driver.py --who B --gpus 4,5,6,7
    python3 main_driver.py --who A --gpus 0 --dry-run      # 무엇을 돌릴지만 본다

조건은 전 모델 공통이다.
  · 학습 5장 · batch 1 · 긴 변 640 · lr 1e-4 고정 · 3,000 업데이트
  · 100 · 300 · 1,000 · 3,000 시점마다 val 전부 평가 (COCO mask AP)
  · Round 1~3 · 시드 20260927 하나
**이 값들을 사람마다 다르게 주면 안 된다.** 그래서 명령줄 인자로 열어 두지 않았다.

사람 없이 버티게 만들었다.
  · GPU 마다 한 번에 하나. 작업은 STAGGER 초 간격으로 하나씩 켠다
  · 실패하면 기록을 01_fail<n> 으로 옮기고 한 번 더 시도한다
  · 이미 성공한 작업은 건너뛴다 → 서버가 꺼져도 같은 명령을 다시 치면 이어서 간다
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HERE = Path(os.environ.get("EXP_ROOT", REPO / "workdir"))   # data/ · runs/ · logs/ 가 여기에 생긴다
RUNNERS = REPO / "runners"                                   # 실행기는 선별 실험과 같은 파일을 쓴다
# 파이썬 환경 두 개: MMDetection · Detectron2 용(legacy), Ultralytics 용(modern). 설치는 README 참고
PY_LEGACY = Path(os.environ.get("PY_LEGACY", "python"))
PY_MODERN = Path(os.environ.get("PY_MODERN", "python"))

# ---- 고정 조건 (바꾸지 말 것) -------------------------------------------------
UPDATES = 3000
EVAL_AT = "100,300,1000"            # + 마지막(3000)은 항상 평가
SEED = 20260927
ROUNDS = (1, 2, 3)
ATTEMPT = "01"
# 2026-10-12: Detectron2 계열 3개는 01 에서 SGD · RandomFlip(+LSJ)로 돌았다 (MMDet 계열은 AdamW · 증강 없음).
# 조건을 맞춘 재실행은 02 에 쓴다 (--matched: AdamW lr 1e-4 wd 1e-4 · 증강 없음). 01 은 지우지 않는다.
ATTEMPT_MATCHED = "02"


def attempt_of(model):
    return ATTEMPT_MATCHED if model in D2 else ATTEMPT
# ---- 분담 (두 사람이 모델 4개씩) ---------------------------------------------
WHO = {
    "A": ["maskdino", "htc", "mask_rcnn", "solov2"],
    "B": ["mask2former", "transfiner", "fastinst", "yolo11_seg"],
    # 조건을 맞춘 재실행 (Detectron2 계열만). 누가 돌려도 된다.
    "rerun": ["maskdino", "transfiner", "fastinst"],
}
MMDET = {"mask_rcnn", "htc", "mask2former", "solov2"}
D2 = {"maskdino", "fastinst", "transfiner"}
MODERN = {"yolo11_seg"}
# 오래 걸리는 것부터 돌리려고 쓰는 대략값 (선별 실험 Cylinder 실측: 300 업데이트 학습 초, 장당 추론 초)
COST = {"mask_rcnn": (53, 0.06), "htc": (81, 0.09), "mask2former": (163, 0.67), "solov2": (52, 0.05),
        "maskdino": (263, 1.9), "fastinst": (79, 1.7), "transfiner": (127, 1.3), "yolo11_seg": (112, 1.0)}
STAGGER = 120
# 원본 이미지가 아주 큰 단위(3840×2748 등)에서 Mask DINO · FastInst 는 평가 때 GPU 메모리를 혼자 ~30GB 쓴다
# (본 실험 01 실측: 20GB 를 넘은 실행이 Mask DINO 15회 · FastInst 12회, 최대 30.1GB. V100 은 32GB).
# 이 작업은 GPU 한 장을 혼자 쓰게 한다. --per-gpu 2 로 옆에 다른 작업을 올리면 메모리가 넘친다.
HEAVY_UNITS = {"20_Console-Collision-B", "21_Console-Scratch-A", "22_Console-Scratch-B",
               "23_PCB_1-missing_hole", "27_Lens-Tear-B"}
HEAVY_MODELS = {"maskdino", "fastinst"}


def heavy(job):
    return job[0] in HEAVY_MODELS and job[1] in HEAVY_UNITS
TIMEOUT = 12 * 3600
RETRIES = 1


def units():
    return sorted({p.name.rsplit("__R", 1)[0] for p in (HERE / "data").iterdir() if "__R" in p.name})


def run_dir(model, unit, k):
    return HERE / "runs" / f"{unit}__R{k}" / model / attempt_of(model)


def state(job):
    try:
        return json.loads((run_dir(*job) / "result.json").read_text(encoding="utf-8"))
    except Exception:
        return None


def command(model, unit, k):
    if model in MMDET:
        py, script, extra = PY_LEGACY, "run_mmdet.py", []
    elif model in D2:
        py, script, extra = PY_LEGACY, "run_d2.py", ["--matched"]
    else:
        py, script, extra = PY_MODERN, "run_modern.py", ["--yolo-merge"]
    return ["nice", "-n", "10", "timeout", str(TIMEOUT), str(py), str(RUNNERS / script),
            model, f"{unit}__R{k}", "--updates", str(UPDATES), "--eval-at", EVAL_AT,
            "--seed", str(SEED), "--attempt", attempt_of(model)] + extra


def my_active():
    """지금 돌고 있는 내 실행기 프로세스 {(모델, 단위, Round): (pid, gpu)}.
    드라이버를 바꿔 띄울 때, 앞 드라이버가 켜 둔 작업을 다시 잡지 않으려고 본다."""
    me, out = os.getuid(), {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            if d.stat().st_uid != me or not (d / "comm").read_text().startswith("python"):
                continue
            a = [x.decode("utf-8", "replace") for x in (d / "cmdline").read_bytes().split(b"\0")]
            env = dict(x.split("=", 1) for x in (d / "environ").read_bytes().decode("utf-8", "replace").split("\0") if "=" in x)
        except OSError:
            continue
        if env.get("EXP_ROOT") != str(HERE) or not any(x.endswith(("run_mmdet.py", "run_d2.py", "run_modern.py")) for x in a):
            continue
        i = next(k for k, x in enumerate(a) if x.endswith(".py"))
        unit, k = a[i + 2].rsplit("__R", 1)
        try:
            out[(a[i + 1], unit, int(k))] = (int(d.name), int(env.get("CUDA_VISIBLE_DEVICES", "-1")))
        except ValueError:
            pass
    return out


def cost(job):
    model, unit, k = job
    nval = json.loads((HERE / "data" / f"{unit}__R{k}" / "manifest.json").read_text())["val_count"]
    tr, inf = COST[model]
    return tr * 10 + inf * nval * 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--who", required=True, choices=list(WHO) + ["all"])
    ap.add_argument("--gpus", required=True, help="쓸 GPU 번호. 예: 0,1,2,3")
    ap.add_argument("--per-gpu", type=int, default=1,
                    help="GPU 한 장에 동시에 올릴 작업 수. batch 1 이라 GPU 사용률이 10~40%% 에 그친다 "
                         "(2026-10-05 실측). 2 로 주면 남는 GPU 를 쓴다. 조건은 바뀌지 않는다")
    ap.add_argument("--shortest-first", action="store_true",
                    help="짧은 작업부터 돌린다 (쓸 수 있는 시간이 짧을 때. 기본은 오래 걸리는 것부터)")
    ap.add_argument("--stagger", type=int, default=STAGGER, help="작업을 켜는 간격(초)")
    ap.add_argument("--dry-run", action="store_true", help="돌리지 않고 무엇이 남았는지만 본다")
    args = ap.parse_args()
    os.umask(0o002)                               # 두 사람이 같은 폴더에 쓴다 → 그룹 쓰기 허용
    gpus = [int(g) for g in args.gpus.split(",")]
    models = [m for w in (WHO if args.who == "all" else [args.who]) for m in WHO[w]]
    log_path = HERE / f"main_driver_{args.who}.log"
    pid_path = HERE / f"main_driver_{args.who}.pid"

    def log(msg):
        line = f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        if not args.dry_run:
            with log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    jobs = [(m, u, k) for m in models for u in units() for k in ROUNDS]
    orphans = {} if args.dry_run else {j: v for j, v in my_active().items() if j in set(jobs)}
    sgn = 1 if args.shortest_first else -1
    todo = sorted((j for j in jobs if (state(j) or {}).get("status") != "passed" and j not in orphans),
                  key=lambda j: ((heavy(j), cost(j)) if args.shortest_first else (0, -cost(j))))   # 짧은 것부터일 때 무거운 작업은 뒤로
    if args.dry_run:
        print(f"{args.who}: 모델 {models}")
        print(f"전체 {len(jobs)}개 · 끝난 것 {len(jobs) - len(todo)}개 · 남은 것 {len(todo)}개 · GPU {gpus}")
        for m in models:
            d = sum(1 for j in jobs if j[0] == m and (state(j) or {}).get("status") == "passed")
            print(f"  {m:<14}{d:>3}/78")
        print(f"  GPU 를 혼자 써야 하는 무거운 작업: {sum(heavy(j) for j in todo)}개 (나머지는 GPU당 {args.per_gpu}개씩)")
        print(f"대략 남은 시간: {sum(cost(j) for j in todo) * 0.7 / 3600 / len(gpus):.0f}시간 "
              f"(GPU 가 비어 있을 때. 다른 작업과 나눠 쓰면 2~4배)")
        return

    if pid_path.exists():
        old = pid_path.read_text().strip()
        if old.isdigit() and Path(f"/proc/{old}").exists():
            sys.exit(f"이미 돌고 있다 (PID {old}). 같은 사람 몫을 두 번 띄우면 안 된다.")
    pid_path.write_text(str(os.getpid()))
    (HERE / "logs").mkdir(exist_ok=True)

    log(f"시작 — {args.who} · 모델 {models} · 전체 {len(jobs)}개 중 남은 것 {len(todo)}개 · GPU {gpus} · GPU당 {args.per_gpu}개")
    for j, (pid, g) in orphans.items():
        log(f"이미 돌고 있어 그대로 둠: {j[0]} / {j[1]} / R{j[2]} (PID {pid}, GPU{g})")
    running, tries, done, failed, last, swept = {}, {}, 0, [], 0.0, False
    env0 = dict(os.environ, EXP_ROOT=str(HERE), OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    while todo or running or orphans:
        orphans = {j: v for j, v in orphans.items() if Path(f"/proc/{v[0]}").exists()}
        if not todo and not running and not orphans and not swept:
            # 앞 드라이버가 켜 둔 작업이 실패로 끝났을 수 있다. 한 번 더 훑는다.
            swept = True
            todo = sorted((j for j in jobs if (state(j) or {}).get("status") != "passed" and j not in failed),
                          key=lambda j: -cost(j))
            if todo:
                log(f"다시 훑음 — 끝나지 않은 작업 {len(todo)}개를 대기열에 넣음")
        for slot in [(g, i) for i in range(args.per_gpu) for g in gpus]:
            g = slot[0]
            used = sum(1 for s_ in running if s_[0] == g) + sum(1 for v in orphans.values() if v[1] == g)
            if slot in running or used >= args.per_gpu or not todo or time.time() - last < args.stagger:
                continue
            if any(heavy(v[1]) for s_, v in running.items() if s_[0] == g) or any(heavy(j) and v[1] == g for j, v in orphans.items()):
                continue                           # 이 GPU 는 무거운 작업이 혼자 쓰는 중
            if used == 0:
                job = todo.pop(0)                  # 빈 GPU: 무엇이든 올린다
            else:
                k = next((i_ for i_, j in enumerate(todo) if not heavy(j)), None)
                if k is None:
                    continue                       # 남은 게 무거운 작업뿐이면 GPU 가 빌 때까지 기다린다
                job = todo.pop(k)
            d = run_dir(*job)
            if d.exists():                         # 실패·중단 기록은 지우지 않고 옮긴다
                n = 1
                while (d.parent / f"{d.name}_fail{n}").exists():
                    n += 1
                shutil.move(str(d), str(d.parent / f"{d.name}_fail{n}"))
            lf = (HERE / "logs" / f"{job[1]}__R{job[2]}__{job[0]}.log").open("w")
            proc = subprocess.Popen(command(*job), cwd=RUNNERS, env=dict(env0, CUDA_VISIBLE_DEVICES=str(g)),
                                    stdout=lf, stderr=subprocess.STDOUT)
            running[slot] = (proc, job, time.time())
            last = time.time()
            log(f"GPU{g} 시작 {job[0]} / {job[1]} / R{job[2]}  (대기 {len(todo)})")
        time.sleep(15)
        for slot, (proc, job, t0) in list(running.items()):
            if proc.poll() is None:
                continue
            del running[slot]
            g = slot[0]
            s = state(job) or {}
            name = f"{job[0]} / {job[1]} / R{job[2]}"
            if s.get("status") == "passed":
                done += 1
                cps = " ".join(f"{c['updates']}:{c['mask_AP'] * 100:.2f}" for c in s.get("checkpoints", []))
                log(f"GPU{g} 성공 {name} {(time.time() - t0) / 60:.0f}분 | {cps}")
            else:
                tries[job] = tries.get(job, 0) + 1
                why = (s.get("error") or f"종료코드 {proc.returncode} (124 = 시간 초과)")[:200]
                if tries[job] <= RETRIES:
                    log(f"GPU{g} 실패 {name} [{s.get('stage')}] {why} → 다시 시도")
                    todo.append(job)
                else:
                    log(f"GPU{g} 최종 실패 {name} [{s.get('stage')}] {why}")
                    failed.append(job)
    log(f"전부 끝 — 이번에 성공 {done} · 최종 실패 {len(failed)}: {failed}")
    pid_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
