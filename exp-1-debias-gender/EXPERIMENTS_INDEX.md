# 실험 폴더 정리 인덱스

이 문서는 `exp-1-debias-gender/` 아래에 흩어져 있던 **일회성 분석 실험 결과물**들을 주제별 폴더로 묶은 정리 인덱스입니다.
(작성: 2026-07-07)

## 정리 원칙

- **묶은 것**: 일회성 분석/시각화 스크립트 + 그 출력 폴더 + 실행 로그를 한 세트로 묶음 → 각 폴더가 "스크립트 + 입력 + 출력 + 로그"로 자기완결(self-contained)되게.
- **건드리지 않은 것 (실험 실행에 중요)**: 학습·평가·`check_*` 메인 스크립트, `run_*.sh`, `configs/`, 그리고 파이프라인이 직접 쓰는 출력들. (아래 [건드리지 않은 것](#건드리지-않은-것-실험-실행에-중요) 참고)
- 이 디렉터리는 git 저장소지만, 이동은 커밋하지 않고 `mv`로만 했습니다. 되돌리려면 [이동 매핑](#부록-이동-매핑-되돌리기용) 표를 역으로 실행하면 됩니다.

> ⚠️ **재실행 시 공통 주의**: 묶인 분석 스크립트들은 대부분 출력 경로를 **상대경로**로 씁니다(예: `./attmap_gen_vs_score_out`). 따라서 재실행할 때는 반드시 **해당 폴더 안에서(CWD = 그 폴더)** 실행해야 합니다. 또한 일부 폴더의 **로그/JSON 내부에는 이동 전 옛날 평면(flat) 경로가 하드코딩**되어 있어 현재 레이아웃과 일치하지 않습니다(특히 `blur_analysis/`).

---

## 📁 attmap_experiments/

**한 줄 요약**: SDS 채점 시점의 cross-attention map(attmap)을 eps-잔차 오차의 공간 가중치로 쓸 때, 그게 실제로 gender debiasing에 영향을 주는가?
**기간**: 2026-06-30 ~ 07-07

| 실험 | 스크립트 | 핵심 결과 |
|---|---|---|
| **gen_vs_score** | `attmap_gen_vs_score_experiment.py` | attmap 추출 축(WHERE: GEN='face'토큰 denoising vs SCORE=woman/man 재노이즈 / LAYER: ALL·MID·RES16)을 분리. **맵 모양은 크게 다른데**(GEN_MID eff_support=55.8=얼굴 집중, GEN_ALL=4011≈균일) **gender CE는 모든 가중치·무마스크에서 사실상 동일**(gender_gap −0.200 고정, ambiguous_frac=0). → 깨끗한 얼굴에선 attmap 가중이 SDS 성별 결정을 바꾸지 못함. |
| **gradient_focus** | `attmap_gradient_focus_experiment.py` | CE 값은 안 바뀌어도 fair-loss gradient가 이미지 "어디를" 편집하는지 재가중. 얼굴 집중 맵이 gradient를 얼굴에 **약간 더** 집중(center_frac 0.629 > 0.595 > nomask 0.567). 효과는 실재하나 작음. ⛓️ **gen_vs_score 출력을 입력으로 재사용**. |
| **scale** | `attmap_scale_experiment.py` | `check.py`(w_hat·res²) vs `att.py`(w_hat²·res²) **한 줄 차이** 격리. w_hat 제곱은 SDS 오차를 ~1/eff_support(**약 3800배**) 축소. fp16에선 언더플로까지 겹쳐 att/check 비 = fp16 3.2e-5 vs fp32 2.7e-4. → **att.py 방식이 SDS 분류기가 죽는 유력한 원인**. |
| **threshold_mask** | `attmap_threshold_mask_visualization.py` | SDS attention map을 임계값으로 얼굴에 국소화할 수 있는지 시각화. 맵이 64×64에서 **거의 균일**(eff_support 3838, frac 0.937)이라 절대 임계값으론 국소화 불가. 4개 `out_*` 폴더는 **임계값 리스트만 다름**(`out_50`=0.2/0.4/0.6, `_th015_025`=0.15/0.2/0.25, `_th01_015_02_025`=0.1~0.25; `out_smoke`=2장 스모크 테스트). |

**의존성**: 4개 스크립트 모두 상대 out_dir → CWD=`attmap_experiments/`에서 실행. gradient_focus는 gen_vs_score를 먼저 돌려야 함. 공통 model=SD-v1.5, seed=5991.
**주의**: gen_vs_score/gradient_focus는 깨끗한 얼굴 10장(및 5장 서브셋) 기준 → 학습 분포(직업 프롬프트)가 아님. `out_smoke`는 실제 결과 아님. 주석/문서 한국어.

---

## 📁 blur_analysis/

**한 줄 요약**: gender-debias 파인튜닝이 진짜 성별 균형으로 "공정"을 달성하는지, 아니면 **얼굴을 흐리게/열화시켜** SDS 성별 분류기가 구분 못 하게 만들어 공정을 "가짜로" 달성하는지 진단. 정규화 레시피(**ckpt1800**: DAL10+SCR-norm+SRR, wImg1/wReal4) vs 원본 DAL-only(**ckpt2000**: wImg0/wReal0) 비교.
**기간**: 2026-06-29 ~ 07-01

| 하위 폴더 / 실험 | 핵심 결과 |
|---|---|
| **ckpt1800 repro_exp** (full25=eval, skip50=train) | realistic-SDS가 흐린 이미지에서 오히려 **낮아짐(블러 보상)**: ori 0.0748→ft 0.0371, Laplacian-var 2981→524, male_ratio 0.68→0.40. **s_f≈s_m** = 분류기가 성별 구분 불가. |
| **ckpt1800 clean + metrics** | n=100: clipT 0.4051→0.4003(**거의 유지**), realistic-SDS 0.0636→0.0342, lapvar 756→487. 정체성 대체로 보존(ClipI 0.568, DINOI 0.494). → "품질 유지·CLIP-T 평탄·블러 보상·성별 붕괴". |
| **ckpt1800 프롬프트/타임스텝 강건성** | 블러 보상 효과가 프롬프트 문구 3종·모든 t 밴드에서 유지(overall ori 0.1823 vs ft 0.1327, 27% 갭). |
| **ckpt2000 clean** (원본 DAL-only) | 부드러운 블러가 아니라 **파국적 붕괴**: ft clipT 0.21~0.33, lapvar **상승**(노이즈/아티팩트), DINOI 0.00~0.04(원본과 거의 무관). |
| **ckpt2000 sds_attn_analysis** | production 경로(attn-masked) SDS 분류기+CE. ft는 **남성 편향 심함**(pred man 0.76, gap 0.525), s_f≈s_m. |
| **ckpt2000 sds_ce_redo** | CE 재계산으로 남성 편향 확인(male_ratio 0.46→0.82). 낮은 CE는 공정이 아니라 "확신에 찬 남성 예측". |
| **blur_analysis_compare** | 두 체크포인트 직접 비교. **ckpt1800**=완만한 블러(lapvar↓)로 붕괴 회피, **ckpt2000**=노이즈(lapvar↑) 붕괴. 하지만 **둘 다 s_f≈s_m으로 공정을 가짜로 달성**하고 실제론 남성 편향. |

**의존성**: CWD=`exp-1-debias-gender`(=`blur_analysis/`의 부모)에서 실행. repro는 학습 체크포인트(`--ckpt`)와 export된 EMA(`--ema`, `2-export-checkpoint.py` 산출)를 필요로 함. 외부 모델: SD-v1.5, open_clip ViT-bigG-14, dinov2_vitg14.
> ⚠️ **경로 불일치**: 로그/JSON 내부가 이동 전 옛 경로(`blur_analysis_ckpt1800/`, `.../scratchpad/full/` 등)를 참조 → 현재 레이아웃과 안 맞음. `render_annot.py`에는 옛 폴더명이 하드코딩되어 있음.
**주의**: 두 실행이 결과 출력 후 크래시(ckpt1800 `metrics_results.json` 미기록 → 수치는 `metrics/run.log`에만; ckpt2000 clean은 이미지만 있고 `clean_results.json` 없음 → 완성 수치는 `blur_analysis_compare`에). repro는 `region_mask_mode='none'`(비마스크)라 절대 SDS 크기가 wandb와 다름(방향만 신뢰). "낮은 CE/균형처럼 보이는" 수치는 분류기 붕괴(s_f≈s_m) 때문이니 **male_ratio·gap이 믿을 지표**.

---

## 📁 blip_experiments/

**한 줄 요약**: 어떤 BLIP 프롬프트 문구("face"/"photo"/"person")와 입력(얼굴 크롭 vs 전체 이미지) 조합이 **성별 미결정률(undecided)**을 최소화하는가.
**기간**: 2026-06-25

| 실험 | 입력 / 프롬프트 | 미결정률 | 비고 |
|---|---|---|---|
| exp3_fullimg_photo | 전체이미지 + "photo" | **10%** | ✅ **BEST** |
| exp2_facecrop_photo | 얼굴크롭 + "photo" | 12% | 'face'→'photo'만 바꿔 미결정 절반↓ |
| exp1_facecrop_face | 얼굴크롭 + "face" | 31% | 현재 파이프라인 기본 설정 |
| exp4_facecrop_person | 얼굴크롭 + "person" | 46% | 'person' 문구가 더 나쁨 |
| exp5_fullimg_person | 전체이미지 + "person" | 62% | ❌ **최악** |

**스크립트**: exp1–3 = `blip_undecided_experiment.py`, exp4–5 = `blip_person_prompt.py`.
**의존성**: `blip_person_prompt.py`(exp4/5)는 `OUT='blip_exp_out'` 상대경로로 exp1–3의 산출물(crops, images, exp3 json)을 **재사용** → exp1–3을 먼저 돌려야 하고 CWD=`blip_experiments/`.
**주의**: 정답 성별이 없는 **미결정률** 실험(정확도 아님). N=50, SD-v1.5. `summary.json`엔 exp1–3만 기록됨(exp4/5는 각자 json).

---

## 📁 clip_t_occupation/

**한 줄 요약**: 한 debias 런의 두 체크포인트(step-400 vs step-1200)에 대해 **직업별 CLIP-T**(프롬프트-이미지 정렬/충실도)를 측정·비교하고, 각 체크포인트 평균 기준으로 직업을 상/하위로 분할.
**기간**: 2026-07-07

- 전체 CLIP-T: **0.3385(step-400) → 0.3564(step-1200)**, 50개 중 **41개 직업 개선**(9개 퇴보).
- 최대 상승: electrical/electronics repairer(+0.070), industrial machinery mechanic(+0.069). 최대 하락: ticket taker(−0.021).
- `split_by_mean`: 각 체크포인트 자기 평균 기준 분할(400→25/25, 1200→22/28). 최상위 'sewer', 최하위 'blaster'(두 체크포인트 공통).

**스크립트**: 집계 = `clip_t_by_occupation_summary.py`(폴더 안). **생성기 `clip_t_by_occupation_eval.py`는 부모 레벨에 그대로 둠**(~190KB 학습/평가 메인 스크립트).
**의존성**: `summary.py`는 `clip_t_by_occupation_out/`를 상대경로로 읽음 → CWD=`clip_t_occupation/`.
> ⚠️ **주의**: CLIP-T는 **프롬프트 충실도** 지표지 성별 공정 지표가 아님(400→1200 상승은 "프롬프트를 더 잘 따른다"일 뿐). `split_by_mean_*.csv`를 만든 스크립트는 리포에 없어 그 분할은 체크인된 코드만으론 재현 불가.

---

## 📁 claude_analysis/ *(기존 폴더, 그대로 유지 — 이미 자기완결)*

**한 줄 요약**: SD1.5 TE-LoRA gender debiasing이 왜 gender gap은 못 줄이면서 화질만 떨어뜨리는지에 대한 디버깅 조사(손실별 gradient 분해 + attn region-masking 왜곡 테스트).

- **`grad_decomp.py`**: fair gradient가 전체의 **98.7–99.8% 지배**(1/tau=1e4 증폭), realistic-face grad는 **fp16 언더플로로 정확히 0**. z-score 모드는 fair 비중을 ~99%→~91%로 낮춤.
- **`attmap_check.py`**: attmap 유효면적 **91%**(거의 균일, 얼굴 전용 아님), region=attn ≈ region=none(corr 0.98) → **attmap은 no-op(red herring)**, "얼굴 전용"으로 보인 건 min-max+pow(0.5) 시각화 아티팩트.
- **결론(`INVESTIGATION_SUMMARY.md`, 한국어)**: "이득 없이 비용만" — z-score=약한 debias(품질 유지) vs fixed-tau=강한 fair(gap 0.23이나 품질 파괴, ClipI 0.52). **sweet spot 없음**. 권장 근본 수정 = SDS 자가분류기를 CLIP/mnet로 교체 + scale-invariant margin/ranking loss.
> ⚠️ `grad_decomp.py`는 학습 스크립트 경로(`1-main-gender-sgd_dmscr_h_gen_check.py`)를 하드코딩해 AST로 함수를 추출 → 그 파일을 옮기거나 함수명을 바꾸면 깨짐. `attmap_check.py`의 OUT은 다른 세션 scratchpad 경로로 하드코딩되어 있음(리포의 `attmap_out/`은 사후 복사본).

---

## 📁 zscore_grad_logs/ *(그대로 유지 — 아래 이유)*

**한 줄 요약**: 고정 tau 대신 **z-score divisor**(배치 std of g = sds_m − sds_f)가 fairness 분류기 포화를 푸는지를, per-step gradient 분해 로깅으로 조사한 산출물.

- `grad_log_main_zscore.jsonl`(1.6MB, `_meta` 1줄 + 1319 step). z-scoring이 포화 해소: 같은 sds에서 `probs_male_zscore`는 중간값, `probs_male_tau_ref`는 0/1로 포화. `loss_fair`가 더 이상 ln2에 고정되지 않음.
> 이 폴더는 **부모에 남겨둔 메인 스크립트 `check_zscore.py`의 기본 출력**(`--grad_log_dir ./zscore_grad_logs`)입니다. 스크립트가 재실행 시 부모 레벨에 이 폴더를 다시 만들고(append 모드), 이 폴더는 "실험 실행에 중요"에 해당하므로 **옮기지 않고 그대로 두었습니다**.

---

## 건드리지 않은 것 (실험 실행에 중요)

아래는 요청대로 **이동/변경하지 않았습니다**:

- **메인 학습/평가/check 스크립트**: 모든 `*-main-*.py`, `1-main-*.py`, `main_check.py`, `check_*.py`(예: `check_last_cos.py`, `check_scoring_cos.py`, `check_scoring_mse.py`, `check_zscore.py`), `chekc_SCRclip*.py`, `clip_t_by_occupation_eval.py`, `2-export-checkpoint.py`, `3-eval-checkpoint800-sds.py`
- **실행 셸 스크립트**: 모든 `run_*.sh`
- **설정**: `configs/`
- **파이프라인 출력(라이브)**: `outputs/`, `logs_sweep/`, `outputs_eval800/`, `zscore_grad_logs/`
- **원본 문서**: `README.md`(업스트림 프로젝트 파이프라인 문서), `profile_comparison_*`, 상위 학습 로그(`gender_dmscr_h_gen*.log`, `sweep_all.log`)

> `run_*.sh`가 호출하는 것: `run_block_ablation_*` → `1-main-gender-sgd_dmscr_h_gen_check_block_ablation.py`, `run_blip_loss_sweep*` → `1-main-gender-sgd_dmscr_h_gen_check_blip.py`, `run_hloss_last_vs_all_600` → `check_last_cos.py` + block_ablation, `run_main_gender_600_x3` → `06291-main-gender-sgd_dmscr_h_gen_check_att.py` 등. 모두 `outputs/`·`logs_sweep/`·`configs/_sweep`에 씀 → 그래서 건드리지 않음.

---

## 부록: 이동 매핑 (되돌리기용)

각 폴더 안으로 아래 항목들을 옮겼습니다. 되돌리려면 이 표를 역으로(`mv <새폴더>/<항목> .`) 실행하세요.

| 새 폴더 | 옮긴 항목 |
|---|---|
| `attmap_experiments/` | `attmap_gen_vs_score_experiment.py`, `attmap_gen_vs_score_out/`, `attmap_gen_vs_score_run.log`, `attmap_gradient_focus_experiment.py`, `attmap_gradient_focus_out/`, `attmap_gradient_focus_run.log`, `attmap_scale_experiment.py`, `attmap_scale_out/`, `attmap_scale_run.log`, `attmap_threshold_mask_visualization.py`, `attmap_threshold_mask_out_50/`, `attmap_threshold_mask_out_50_th015_025/`, `attmap_threshold_mask_out_50_th01_015_02_025/`, `attmap_threshold_mask_out_smoke/`, `attmap_threshold_mask_th01_run.log` |
| `blur_analysis/` | `blur_analysis_ckpt1800(new DAL10 + SCR(norm) +SRR)/`, `blur_analysis_ckpt2000(원래 DAL만)/`, `blur_analysis_compare/` |
| `blip_experiments/` | `blip_undecided_experiment.py`, `blip_person_prompt.py`, `blip_exp_out/`, `blip_exp_out.log` |
| `clip_t_occupation/` | `clip_t_by_occupation_out/`, `clip_t_by_occupation_summary.py` |

*(`claude_analysis/`와 `zscore_grad_logs/`는 원래 폴더 그대로 — 이동 없음.)*
