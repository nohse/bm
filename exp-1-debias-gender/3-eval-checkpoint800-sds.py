#!/usr/bin/env python
# coding=utf-8
# ============================================================================
# 3-eval-checkpoint800-sds.py
#
# 하나의 스크립트로 아래 세 가지를 수행한다:
#   1) 지정한 run 폴더들의 ckpts/checkpoint-800 가중치를 로드한다.
#      - 별도 .pth export(2-export-checkpoint.py) 없이,
#        1-main 의 eval 과 '똑같은 방식'인 accelerator.load_state 로 EMA 가중치를
#        그대로 로드한다(= 2-export 로 뽑는 text_encoder_lora_EMA.pth 와 동일한 가중치).
#   2) 1-main-gender-sgd_dmscr_h_gen_check.py 의 eval 을 '정확히 동일하게' 실행한다.
#      - eval 이 100% 동일함을 보장하기 위해, 그 파일의 main()/evaluate_process 를
#        그대로 import 해서 eval-only 모드(--eval_at_step0 --eval_only)로 돌린다.
#   3) 각 run 폴더의 eval_results.json '800' 항목에 기존 headline 지표
#        - eval_EMA_gender_gap_abs_mnet
#        - eval_EMA_Clip-I / eval_EMA_Clip-T / eval_EMA_DINO-I
#      와 함께 SDS 로 계산한 gap
#        - eval_EMA_gender_gap_abs_sds
#      를 저장한다. (이 5개 지표는 1-main 의 evaluate_process 가 name=="EMA" 일 때
#       이미 저장하도록 되어 있고, 여기서는 그 코드를 그대로 재사용한다.)
#
# ---------------------------------------------------------------------------
# 왜 이렇게 동작하는가 (재현성 관련 핵심):
#   * JSON 저장 경로는 evaluate_process 가 resume_from_checkpoint 로부터 유도한다:
#       .../<run>/ckpts/checkpoint-800  ->  .../<run>/eval_results.json
#     즉, 각 run 원본 폴더의 eval_results.json 이 in-place 로 '800' 키에 업데이트된다.
#   * region_mask_mode 는 SDS gap 계산에 '실제로' 영향을 준다(woman/man 토큰 어텐션
#     마스크가 SDS MSE 에 곱해져 확률/예측이 달라짐). 이 run 들은 region-attn 으로
#     학습되었으므로 eval 도 반드시 region_mask_mode='attn' 으로 맞춘다.
#     (1-main 의 argparse 기본값은 'none' 이라 그대로 두면 SDS gap 이 달라진다.)
#   * hBlk / hForm / hNorm / wImg / wRealFace / skip 등은 '학습 전용' 손실 옵션이라
#     eval 결과(위 5개 지표)에는 영향이 없다. 여기서는 eval 이미지 폴더 이름을
#     원본과 비슷하게 맞추기 위해 폴더명에서 몇 개만 파싱해 넣지만, 값이 무엇이든
#     지표 계산에는 영향을 주지 않는다.
#   * eval 은 매번 새 난수로 val 이미지를 생성하므로(과거 step-800 학습중 eval 과
#     RNG 상태가 다름), gap 값은 통계적으로 매우 비슷하지만 비트 단위로 동일하지는
#     않다. 이는 'eval 절차를 동일하게' 재현하는 것으로, 원리상 불가피하다.
#   * 각 checkpoint 는 별도 subprocess 로 실행한다(Accelerator/CUDA/wandb 상태 격리).
#
# 사용법:
#   # 유저가 지목한 3개 run 의 checkpoint-800 을 순차 eval:
#   python 3-eval-checkpoint800-sds.py
#
#   # 특정 GPU 로:
#   CUDA_VISIBLE_DEVICES=0 python 3-eval-checkpoint800-sds.py
#
#   # 다른 step / 다른 run 폴더로:
#   python 3-eval-checkpoint800-sds.py --step 800 --runs <folder1> <folder2> ...
#
#   # 무엇을 돌릴지 미리 확인만:
#   python 3-eval-checkpoint800-sds.py --dry-run
#
# 사전 준비: 1-main 을 돌릴 수 있는 동일 환경(파이썬 패키지 + ../data/* + SD1.5 캐시).
#            이 스크립트는 1-main 을 import 하므로 1-main 이 import 가능한 환경이어야 한다.
# ============================================================================

import os
import sys
import json
import argparse
import subprocess
import importlib.util
from pathlib import Path

# 이 스크립트가 있는 디렉터리(= exp-1-debias-gender). 1-main 의 상대 경로(../data/...)와
# 모듈 import 가 이 CWD 를 전제로 하므로 항상 여기로 이동한다.
SCRIPT_DIR = Path(__file__).resolve().parent

# eval 을 재사용할 원본 파일(유저가 지정한 그 파일)
DEFAULT_MAIN_FILE = "1-main-gender-sgd_dmscr_h_gen_check.py"

# 유저가 지목한 3개 run 폴더(outputs/gender_aaai 아래 상대경로). --runs 로 덮어쓸 수 있다.
DEFAULT_RUNS_ROOT = "./outputs/gender_aaai"
DEFAULT_RUNS = [
    "20260703-1001_gender_aaai_region-attn_skip-0pct_wImg-8.0_hBlk-up+mid_hForm-raw_hNorm-std_wRealFace-0.0_Th-0.2_lr-5e-05",
    "20260704-0143_gender_aaai_region-attn_skip-0pct_wImg-8.0_hBlk-up+mid_hForm-raw_hNorm-std_wRealFace-4.0_Th-0.2_lr-5e-05",
    "20260704-1029_gender_aaai_region-attn_skip-0pct_wImg-8.0_hBlk-down+mid_hForm-raw_hNorm-std_wRealFace-4.0_Th-0.2_lr-5e-05",
]

# eval 이미지(그리드/attmap)를 담을 출력 루트. 원본 gender_aaai 폴더를 오염시키지 않도록
# 기본적으로 별도 폴더에 쌓는다. JSON 은 어차피 원본 run 폴더에 저장된다.
DEFAULT_OUTPUT_DIR = "./outputs_eval800"

# eval 이 최종적으로 JSON 에 남기는 5개 headline 지표 키(요약 출력을 위해 사용)
METRIC_KEYS = [
    "eval_EMA_gender_gap_abs_mnet",
    "eval_EMA_gender_gap_abs_sds",
    "eval_EMA_Clip-I",
    "eval_EMA_Clip-T",
    "eval_EMA_DINO-I",
]


# ---------------------------------------------------------------------------
# 폴더명 -> 학습 knob 파싱 (eval 지표엔 영향 없음, 폴더명/추적용 편의)
# ---------------------------------------------------------------------------
def parse_knobs_from_run_name(run_name: str) -> dict:
    """run 폴더명에서 몇몇 knob 을 추출한다. 실패해도 무해(그냥 기본값 사용)."""
    import re
    knobs = {}
    m = re.search(r"region-([a-zA-Z]+)", run_name)
    if m:
        knobs["region_mask_mode"] = m.group(1)
    m = re.search(r"skip-(\d+)pct", run_name)
    if m:
        knobs["skip_final_steps_pct"] = float(m.group(1))
    m = re.search(r"wImg-([0-9.]+)", run_name)
    if m:
        knobs["weight_loss_img"] = float(m.group(1))
    m = re.search(r"wRealFace-([0-9.]+)", run_name)
    if m:
        knobs["weight_loss_face_realistic"] = float(m.group(1))
    m = re.search(r"Th-([0-9.]+)", run_name)
    if m:
        knobs["uncertainty_threshold"] = float(m.group(1))
    m = re.search(r"lr-([0-9eE.+-]+)", run_name)
    if m:
        try:
            knobs["learning_rate"] = float(m.group(1))
        except ValueError:
            pass
    return knobs


# ---------------------------------------------------------------------------
# 단일 checkpoint eval (subprocess 내부에서 실행되는 경로)
#   1-main 의 main() 을 그대로 호출한다 -> eval 이 '정확히 동일'함을 보장.
# ---------------------------------------------------------------------------
def run_single_eval(checkpoint: str, output_dir: str, val_images_per_prompt_gpu: int,
                    region_mask_mode: str, main_file: str) -> int:
    checkpoint = str(Path(checkpoint).resolve())
    if not os.path.isdir(checkpoint):
        print(f"[single-eval][ERROR] checkpoint 폴더가 없습니다: {checkpoint}", flush=True)
        return 2

    # 1-main 의 상대 경로(../data/...)가 풀리도록 CWD 를 스크립트 폴더로.
    os.chdir(SCRIPT_DIR)

    main_path = (SCRIPT_DIR / main_file).resolve()
    if not main_path.exists():
        print(f"[single-eval][ERROR] main 파일이 없습니다: {main_path}", flush=True)
        return 2

    # 하이픈 포함 파일명이라 일반 import 불가 -> 파일 경로로 로드.
    print(f"[single-eval] import: {main_path}", flush=True)
    spec = importlib.util.spec_from_file_location("main_gender_eval_mod", str(main_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 기본값으로 args 네임스페이스 생성(빈 CLI) 후, eval-only 로 오버라이드.
    # parse_args([]) 는 --config 를 안 읽고 순수 기본값 + 후처리(target_male_ratio 등)만 적용.
    args = mod.parse_args([])

    # --- eval 대상/모드 ---
    args.resume_from_checkpoint = checkpoint
    args.eval_at_step0 = True     # 시작 직후 1회 평가
    args.eval_only = True         # 평가만 하고 학습 없이 종료
    args.proj_name = "gender_aaai"
    args.output_dir = str(Path(output_dir).resolve())

    # --- eval 지표에 실제로 영향을 주는 유일한 knob: region_mask_mode ---
    #     (이 run 들은 region-attn 으로 학습 -> SDS gap 재현 위해 'attn' 필수)
    args.region_mask_mode = region_mask_mode

    # --- val 이미지 개수: 1-main 과 동일 기본값(60). eval 은 max(val_per_gpu, ceil(60/nproc)).
    #     단일 프로세스이므로 프롬프트당 60장 생성.
    args.val_images_per_prompt_GPU = int(val_images_per_prompt_gpu)

    # --- 폴더명(추적용) 을 원본과 비슷하게: 학습 knob 은 eval 결과에 영향 없음 ---
    run_name = Path(checkpoint).parent.parent.name  # .../<run>/ckpts/checkpoint-800 -> <run>
    for k, v in parse_knobs_from_run_name(run_name).items():
        # region_mask_mode 는 위에서 명시적으로 세팅했으니 덮어쓰지 않도록 그대로 둔다(동일 값일 것).
        setattr(args, k, v)

    # SDS-eval 은 1-main 에서 항상 켜져 있으므로(enable_sds_eval=True), 별도 설정 불필요.
    # -> evaluate_process 가 gender_gap_abs_sds 를 계산해 JSON 에 저장한다.

    print(f"[single-eval] run={run_name}", flush=True)
    print(f"[single-eval] checkpoint={checkpoint}", flush=True)
    print(f"[single-eval] region_mask_mode={args.region_mask_mode} "
          f"val_images_per_prompt_GPU={args.val_images_per_prompt_GPU}", flush=True)
    print(f"[single-eval] eval 이미지 출력 -> {args.output_dir} (JSON 은 원본 run 폴더에 저장)", flush=True)

    mod.main(args)  # eval_only 라 평가 1회 후 return

    # JSON 이 원본 run 폴더에 잘 갱신됐는지 확인.
    results_path = Path(checkpoint).parent.parent / "eval_results.json"
    step = os.path.basename(checkpoint).split("-")[1]
    if results_path.exists():
        try:
            data = json.load(open(results_path))
            entry = data.get(str(int(step)), {})
            print(f"[single-eval] 저장됨 {results_path} [step {int(step)}]: "
                  f"{json.dumps(entry, ensure_ascii=False)}", flush=True)
        except Exception as e:
            print(f"[single-eval][WARN] eval_results.json 파싱 실패: {e}", flush=True)
    else:
        print(f"[single-eval][WARN] eval_results.json 이 생성되지 않았습니다: {results_path}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# 드라이버: 여러 run 을 순회하며 checkpoint 마다 subprocess 로 single-eval 실행
# ---------------------------------------------------------------------------
def resolve_checkpoints(runs_root: str, runs, step: int):
    ckpts = []
    for run in runs:
        run_path = Path(run)
        if not run_path.is_absolute() and not run_path.exists():
            run_path = SCRIPT_DIR / runs_root / run
        ckpt = run_path / "ckpts" / f"checkpoint-{step}"
        ckpts.append((run, ckpt))
    return ckpts


def main():
    parser = argparse.ArgumentParser(
        description="checkpoint-800 을 1-main 의 eval 로 재평가하고 headline+SDS gap 을 eval_results.json 에 저장.")
    parser.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT,
                        help="run 폴더들이 들어있는 루트(상대경로면 스크립트 폴더 기준).")
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS,
                        help="평가할 run 폴더명(또는 절대경로) 목록.")
    parser.add_argument("--step", type=int, default=800, help="평가할 체크포인트 step(기본 800).")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR,
                        help="eval 이미지(그리드/attmap) 출력 루트. JSON 은 원본 run 폴더에 저장됨.")
    parser.add_argument("--val_images_per_prompt_GPU", type=int, default=60,
                        help="프롬프트당 생성할 val 이미지 수(1-main 기본값 60).")
    parser.add_argument("--region_mask_mode", default="attn", choices=["none", "face", "attn"],
                        help="SDS gap 재현을 위해 학습과 동일하게(기본 attn).")
    parser.add_argument("--main-file", default=DEFAULT_MAIN_FILE, help="eval 을 재사용할 1-main 파일명.")
    parser.add_argument("--wandb-mode", default="offline",
                        help="WANDB_MODE 환경변수(기본 offline; online 로 바꾸면 실제 wandb 로그).")
    parser.add_argument("--dry-run", action="store_true", help="무엇을 돌릴지만 출력하고 종료.")

    # 내부용(드라이버가 subprocess 로 자기 자신을 호출할 때만 사용)
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    # ---------------- single-eval 모드(subprocess 내부) ----------------
    if args.single:
        rc = run_single_eval(
            checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            val_images_per_prompt_gpu=args.val_images_per_prompt_GPU,
            region_mask_mode=args.region_mask_mode,
            main_file=args.main_file,
        )
        sys.exit(rc)

    # ---------------- 드라이버 모드 ----------------
    os.chdir(SCRIPT_DIR)
    ckpts = resolve_checkpoints(args.runs_root, args.runs, args.step)

    print("=" * 78)
    print(f"[driver] step={args.step}  region_mask_mode={args.region_mask_mode}  "
          f"val/prompt={args.val_images_per_prompt_GPU}  WANDB_MODE={args.wandb_mode}")
    print(f"[driver] main-file(eval 재사용) = {args.main_file}")
    print(f"[driver] eval 이미지 출력 루트 = {args.output_dir}")
    print(f"[driver] 대상 {len(ckpts)} 개 checkpoint:")
    missing = []
    for run, ckpt in ckpts:
        exists = ckpt.is_dir()
        print(f"    [{'OK ' if exists else 'MISS'}] {ckpt}")
        if not exists:
            missing.append(str(ckpt))
    print("=" * 78, flush=True)

    if missing:
        print("[driver][ERROR] 아래 checkpoint 가 없습니다. 경로를 확인하세요:")
        for m in missing:
            print(f"    {m}")
        sys.exit(2)

    if args.dry_run:
        print("[driver] --dry-run: 실제 eval 은 실행하지 않았습니다.")
        return

    env = dict(os.environ)
    env["WANDB_MODE"] = args.wandb_mode

    failures = []
    for i, (run, ckpt) in enumerate(ckpts, 1):
        print("\n" + "#" * 78)
        print(f"[driver] ({i}/{len(ckpts)}) EVAL 시작: {run}")
        print("#" * 78, flush=True)
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            "--single",
            "--checkpoint", str(ckpt),
            "--output_dir", args.output_dir,
            "--val_images_per_prompt_GPU", str(args.val_images_per_prompt_GPU),
            "--region_mask_mode", args.region_mask_mode,
            "--main-file", args.main_file,
        ]
        proc = subprocess.run(cmd, env=env, cwd=str(SCRIPT_DIR))
        if proc.returncode != 0:
            print(f"[driver][WARN] ({i}/{len(ckpts)}) 비정상 종료(rc={proc.returncode}): {run}", flush=True)
            failures.append(run)

    # ---------------- 최종 요약: 각 run 의 eval_results.json '{step}' 항목 ----------------
    print("\n" + "=" * 78)
    print(f"[driver] 요약 (eval_results.json 의 '{args.step}' 항목)")
    print("=" * 78)
    for run, ckpt in ckpts:
        results_path = ckpt.parent.parent / "eval_results.json"
        line = {}
        if results_path.exists():
            try:
                data = json.load(open(results_path))
                line = data.get(str(args.step), {})
            except Exception as e:
                line = {"_error": f"파싱 실패: {e}"}
        print(f"\n[{run}]")
        print(f"  {results_path}")
        if line:
            for k in METRIC_KEYS:
                if k in line:
                    print(f"    {k:32s} = {line[k]}")
            if "eval_EMA_gender_gap_abs_sds" not in line:
                print("    [WARN] eval_EMA_gender_gap_abs_sds 없음 — SDS eval 은 완료되지 않았을 수 있음")
            extra = {k: v for k, v in line.items() if k not in METRIC_KEYS}
            if extra:
                print(f"    (기타: {json.dumps(extra, ensure_ascii=False)})")
        else:
            print("    (항목 없음 — eval 실패 가능)")

    if failures:
        print("\n[driver] 실패한 run:")
        for f in failures:
            print(f"    {f}")
        sys.exit(1)
    print("\n[driver] 완료.")


if __name__ == "__main__":
    main()
