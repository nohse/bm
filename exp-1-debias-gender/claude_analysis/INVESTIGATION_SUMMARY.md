# finetune-fair-diffusion 성별 디바이싱 — 조사 요약 (외부 검토용)

> SD1.5 기반 text-encoder LoRA로 직업 프롬프트의 성별 분포를 50/50으로 맞추되 화질을 보존하려는 실험.
> 아래는 진단·측정·수정 시도 전체 요약. 수치는 모두 실제 로그/실측이며, 재현 스크립트 위치는 §5.

---

## 0. 세팅

- 코드: SD1.5, **text-encoder LoRA만 학습** (`train_text_encoder=True, train_unet=False`), rank 50, fp16 mixed precision, **AdamW**(파일명 sgd지만 실제 AdamW), lr 5e-5.
- 목표: occupation 프롬프트("A photo of the face of a {occ}, a person") 생성 분포를 50% 남 / 50% 여로, **화질(정체성) 보존**하며.
- 손실 3종 합:
  ```
  loss = loss_fair + weight_loss_img · dyn · loss_hspace + weight_loss_face_realistic · loss_realistic
  ```
  - **loss_fair** = `CE(logits, target)`, `logits = [-sds_f/tau, -sds_m/tau]`, tau=1e-4. sds_f/sds_m = "a photo of a woman"/"a photo of a man" 프롬프트로 SD1.5가 noised latent를 복원할 때의 **eps-MSE**(SDS). 즉 **SD1.5 자기 자신을 성별 분류기로** 사용.
  - **loss_hspace** = fine vs 원본 모델의 mid-block h-space([N,1280,8,8]) MSE (이미지 보존).
  - **loss_realistic** = "a photo of a realistic face"로의 raw SDS error.
- 타겟: 배치를 SDS male-prob로 **rank 정렬 → 상위 50% 남 / 하위 50% 여 강제 배정**, 중앙(불확실) 샘플은 binomial-cdf uncertainty>0.2면 마스킹(target=-1).
- `region_mask_mode=attn`: SDS error를 woman/man 토큰 **cross-attention map**으로 가중.
- **평가 지표(gap)**: `gender_gap_abs_mnet` = MobileNet(CelebA) 성별 분류기 기준 |남비율−여비율|. **손실이 쓰는 SDS 분류기와 다른 분류기.**

---

## 1. 핵심 진단 (한 줄): "비용은 내는데 이득이 없다"

여러 세팅에서 **gender_gap은 잘 안 떨어지고(≈0.5~0.7) 화질은 하락**. 원인은 단일하지 않고 **여러 병이 겹침**. 아래 카탈로그.

---

## 2. 문제 카탈로그

### 2A. 원래(fixed-tau) 문제 — 측정으로 확인

1. **SDS 분류기가 약함 / `loss_fair`가 ln2에 고정**
   - 600x3 로그: `train_loss_fair = 0.69189 ≈ ln2(0.6931)`, 학습 내내 평평. → SDS softmax ≈ 0.5, 성별 판별 신호 미약.
   - (정정) 초기엔 "어디서나 p≈0.5로 죽었다"고 봤으나, 프롬프트별로는 **자주 확신함**(probs 0/1). ln2는 **양극단(확신-정답 CE≈0 + 강제-오답 스파이크 + 마스킹된 중앙)의 평균**. 즉 "죽은 분류기"보다 "강제 오답 + 약한 push"가 더 정확한 서술.
2. **fair 경사가 전체의 98~99.8% 지배** (grad decomposition 실측)
   - `logits=-sds/tau` → `d(CE)/d(sds)=(p−y)·(1/tau)`, **1/tau=1e4 증폭**.
   - 수식 재구성(실측 부합): `‖g_fair‖ = (1/tau)·|p−y|·‖∂g/∂θ‖`, `‖∂g/∂θ‖≈0.03`(성별신호 물리민감도), `‖∂hspace/∂θ‖≈1.15`. → hspace가 물리적으론 38배 세게 연결됐지만 tau의 25000배 계수특혜로 fair가 **~650배** 압도.
3. **fair 경사가 0-or-스파이크 (양극단)**: 씨앗값 3개에서 `‖g_fair‖ = {0, 93, 301}`. 확신-정답 배치→0(img 100%), 강제-오답 샘플→300 스파이크.
4. **realistic-face 경사 = 정확히 0** (fp16 언더플로우): 1/tau 증폭이 없어 fp16 바닥 아래로 소멸. → `wRealFace`가 켜져도 **무효**.
5. **tau가 이중 역할**: forward 온도(판별) + backward 1/tau(경사 크기). **한 손잡이로 둘 다 못 잡음**.
6. **rank 강제 50/50 타겟**: 쏠린 배치(간호사=여 등)에서 명백한 성별을 반대로 강제 → 큰 off-manifold 편집 유발.
7. **skip_final_steps_pct=50**: 학습 이미지가 흐릿한 x0-jump. 게다가 **타겟은 skip=0(선명), 손실은 skip=50(흐림)** 이미지 분포 불일치. (최근 run은 skip=0으로 바뀜.)
8. **hspace 보존이 너무 거침**: 8×8 mid-block, flip 샘플에서 0.2배 다운웨이트, 별도 trajectory 비교 → 고주파(화질) 보존 못 함.
9. **AdamW**: 절대 경사크기는 정규화됨. 그래서 1/tau는 **스텝 크기가 아니라 방향 지배**로 나타남. hspace가 fair에 방향에서 밀림.
10. **`max_grad_norm=100` 파싱만 되고 미적용** (clip_grad_norm_ 호출 없음).
11. **프록시 불일치**: 손실=SDS 분류기, 평가=MobileNet/CLIP. 경계가 다름.

### 2B. z-score 수정 (설계 + 논의)

- 아이디어: 고정 tau 대신 **배치 std(g)** 로 나눔(scale-only z-score, `g=sds_m−sds_f`). = "tau 자리에 배치 std".
- 결론 (논의):
  - z-score는 **forward 살림 + per-sample 스케일 공평 + 스파이크 완화**. ✅
  - 하지만 **loss별 균형은 단독으론 불충분**. `1/std` 증폭이 여전히 경사에 실림.
  - **std < tau이면 오히려 폭증**(divide-by-small). 저분산/동질 배치에서 위험.
  - **fixed 1:1 경사 정규화는 나쁨** — `(p−y)` 적응성(맞으면 0)을 죽임. → 대신 **loss weight**(적응성 보존) 또는 **clip**.
  - 근본적으론 tau/std 어느 것도 divide-by-small 병 있음 → **margin/ranking loss**(스케일 불변·bounded·적응적)가 가장 robust.

### 2C. z-score 실측 결과 (check_zscore.py 학습)

- **share_fair: 99% → 27%(wImg2) / 40%(wImg1)** — 과지배는 고쳐짐. ✅
- **그러나 `loss_fair`는 여전히 ≈ln2, `|p−0.5|`=0.08** (원본 tau는 0.32). → z-score가 **과정규화**.
  - 원인: divisor(no-grad skip0 24샘플 배치 std ≈6.7e-4)가 **청크 내부 g 스프레드(1.6e-4)보다 ~4배 큼** → 확률 0.5로 압축.
  - 배치가 자주 쏠림(60스텝 중 27스텝 `|g_mean|>g_std`) + scale-only(센터링 없음) → 한쪽으로 몰림.
- 우려하던 저분산 폭증은 이번 run엔 없었음(divisor 항상 > tau).

### 2D. loss/gap 안 떨어지는 문제 (eval)

- **z-score run**: gap이 0.5~0.6에서 안 떨어짐, 화질은 하락.
  - wImg2: gap 0.598→0.591→0.591, ClipI 0.712→0.628.
  - wImg1(fair 상대강화): gap 0.55→**0.425**→0.53→0.496(노이지), ClipI 0.676→**0.481**(급락).
- **fixed-tau skip0(baseline)**: gap **0.233**(강한 디바이싱) but ClipI **0.520**(화질 파괴).
- → **트레이드오프 양 끝**: z-score=약한 fair(화질 보존, 약한 디바이싱) / fixed-tau=강한 fair(디바이싱, 화질 파괴). 스윗스팟 없음.
- **두 후보 천장** (미확정):
  - (A) fair 약함(z-score 과정규화, loss~ln2).
  - (B) **프록시 불일치**: per-prompt SDS gap ≈0.35 (n=12 샘플링 바닥 0.23에 근접) vs mnet gap ≈0.5 (n=60 바닥 0.10에서 멂). → "손실은 SDS를 거의 맞췄는데 지표 mnet은 안 따라옴" 의심. **단 train(SDS) vs eval(mnet) 다른 이미지셋이라 confound.**
- **결정적 테스트(아직 미실행)**: 같은 eval 이미지에서 SDS-gap vs mnet-gap. → **이 코드는 SDS gap을 계산은 했는데 저장을 안 하고 버리고 있었음**. 그래서 `eval_EMA_gender_gap_abs_sds` 로깅을 3개 파일에 추가함(§5). 다음 eval에서 A/B 판정 가능.

### 2E. attmap 문제 (실측으로 두 가설 반박)

- 사용자 가설: (1) attmap이 얼굴만 잡고 머리 제외 → SDS 이상, (2) attmap 경사가 이상하게 전체로 흘러 이상한 이미지.
- **실측(attmap_check.py, doctor 얼굴 8장):**
  - attmap **유효면적(IPR) = 91%** (거의 균일), top20% 질량 0.30. → **얼굴만이 아니라 사람 전체(머리 포함)에 거의 균일**.
  - **region=attn ≈ region=none** (신호 상관 **0.98**, 부호일치 **1.0**).
  - attention은 **완전 detach**됨(no_grad+detach) → 상수 weight.
- **결론:**
  - **가설1 반박**: "얼굴만" 인상은 **시각화 artifact**(min-max + `.pow(0.5)` 감마). 실제 weight는 균일.
  - **가설2 메커니즘 비활성**: attmap이 균일+detach라 attn ≈ none. 특별한 국소-경사 왜곡 없음.
  - **진짜 문제 = no-op**: 모든 cross-attn 레이어 + 15 timestep 평균이라 너무 diffuse. region masking이 사실상 작동 안 함. (제대로 focus하면 SNR↑ 가능성은 있음 = 개선 여지.)
- → **attmap은 gap 안 떨어지는 직접 주범 아님(red herring).**

---

## 3. 실험 & 증거 (측정값 요약)

| 실험 | 방법/스크립트 | 핵심 수치 |
|---|---|---|
| loss_fair 고정 확인 | 600x3 로그 | `train_loss_fair=0.69189≈ln2`, gap 0.69~0.71 flat, ClipI 0.855→0.81 |
| **경사 분해** (fixed tau) | `claude_analysis/grad_decomp.py` (AST로 실제 loss fn 추출, 1스텝 재현) | share_fair 98.7~99.8%, ‖g_fair‖={0,93,301}, ‖g_img‖~1, realistic grad=**0**, cos(fair,img)~0.1 |
| 경사 재구성 | 수식 | ‖g_fair‖=(1/tau)|p−y|‖∂g/∂θ‖, ‖∂g/∂θ‖≈0.03, ‖∂hspace/∂θ‖≈1.15, 비율~650 |
| z-score adaptive tau | grad_decomp.py (tau_mode) | fair 99.4%→91.5%, ‖g_fair‖ 188→12.5, |p−0.5| 포화 해제 |
| **z-score 학습 진단** | `zscore_grad_logs/grad_log_main_zscore.jsonl` (check_zscore.py가 매 10스텝 기록) | share_fair 27~42%, loss_fair≈0.66≈ln2, |p−0.5|z=0.08, divisor 6.7e-4, chunk g std 1.6e-4(4× mismatch) |
| eval gap/화질 | `outputs/gender_aaai/*/eval_results.json` | z-score gap 0.5~0.6/화질 하락 vs fixed-tau gap 0.23/화질 0.52 |
| per-prompt SDS gap | grad log 재분석 | SDS gap ≈0.35 (바닥 0.23) vs mnet gap ≈0.5 (바닥 0.10) |
| **attmap 집중도** | `claude_analysis/attmap_check.py` + `attmap_out/attmap_overlay.png` | IPR 91%(거의 균일), attn≈none corr 0.98 |

---

## 4. 권장 수정 (우선순위)

1. **[결정적] eval SDS-gap 로깅** (이미 3개 파일에 추가함) → 다음 eval에서 **A(fair 약함) vs B(프록시 불일치)** 판정.
   - SDS 낮은데 mnet 높다 → **프록시 확정** → 분류기를 **CLIP/mnet으로 교체**(가장 근본).
   - 둘 다 높다 → **fair 약함** → 아래 3~5로 샤프닝.
2. **fair 샤프닝**: 센터링 `(g−mean)/std` + divisor를 **grad 배치 스케일**로(현 4× 과대 해소) + **num_eps↑**(g 노이즈↓).
3. **경사 균형**: fixed 1:1 정규화 금지(적응성 파괴). **loss weight 조절** 또는 **clip**(스파이크만 컷).
4. **fp32** (보존/realistic 항 언더플로우 방지) + `max_grad_norm` 실제 적용.
5. **(대안) margin/ranking loss**: 타겟이 이미 rank 기반 → 스케일불변·bounded·적응적. tau/std 문제 전체 제거.
6. **attmap**: no-op 상태 → 고칠거면 고해상도 cross-attn만+낮은 t+sharpen, 아니면 신경 안 써도 무방(red herring).
7. **rank 강제 50/50 → soft/OT 타겟** (확신 샘플 억지 flip 방지).

---

## 5. 파일 위치

### 코드 (repo: `/workspace/finetune-fair-diffusion/exp-1-debias-gender/`)
- **원본 학습**: `1-main-gender-sgd_dmscr_h_gen_check.py`
- **z-score+경사로깅 버전**: `check_zscore.py` (원본 복사 + ①scale-only z-score(gradient-pass tau 자리에 배치 std 주입, ~L3640) ②매 N스텝 per-loss 경사분해 JSONL 기록 ③eval SDS-gap 로깅). 플래그: `--use_zscore_logit`(기본 True), `--grad_log_dir`, `--grad_log_every`(기본 10).
- **block ablation**: `1-main-gender-sgd_dmscr_h_gen_check_block_ablation.py`
- 위 3개 파일 모두 **eval SDS-gap 로깅 추가됨**(`eval_EMA_gender_gap_abs_sds`).
- 주요 함수 위치(원본 기준): `sds_logits_from_images`~L2360, `generate_image_w_gradient`~L1990, `generate_dynamic_targets`~L2861, `CrossAttnCapture`~L164, 손실 조립~L3715, `_normalize_attmap_for_vis`~L365.

### 내 실험 스크립트 (repo로 복사됨: `claude_analysis/`)
- `grad_decomp.py` — 경사 분해(실제 loss fn을 AST로 추출, 1 학습스텝 재현, fair/img/face 경사 norm·share·cos). fixed vs z-score(adaptive tau) 비교 포함.
- `attmap_check.py` — attmap 집중도(IPR/top-k) + attn vs none 신호 비교 + 오버레이 저장. (`import grad_decomp`로 모델 재사용.)
- `attmap_out/attmap_overlay.png` — 이미지|attmap 오버레이 그리드.

### 데이터/출력 (repo)
- **경사 로그**: `zscore_grad_logs/grad_log_main_zscore.jsonl` (첫 줄 `_meta` = 설정; 이후 매 레코드에 grad norm/share/cos, g/std, probs(zscore & tau), targets, loss들). 여러 run이 `_meta`로 구분됨(재시작마다).
- **eval 결과**: `outputs/gender_aaai/20260704-0137_..._wImg-2.0_.../eval_results.json`, `20260704-1110_..._wImg-1.0_.../eval_results.json` (step→{gender_gap_abs_mnet, Clip-I, Clip-T, DINO-I}).
- **fixed-tau baseline eval**: `outputs/gender-debias-text-encoder/BS-24_wImg-2-..._skip-0_.../eval_results.json`, `..._skip-50_.../`.
- **원래 600x3 로그**: `logs_sweep/600x3_rep{1,2,3}.log`.
- **configs**: `configs/debias-text-encoder2.yaml`(현 run), `debias-text-encoder4_600.yaml` 등.
- **wandb**: `outputs/wandb/run-*` (train_gender_gap_abs / _sds 등, 바이너리).

### 실행 예 (현 z-score run)
```
accelerate launch --config_file configs/accelerate_config2.yaml \
  check_zscore.py --config configs/debias-text-encoder2.yaml \
  --region_mask_mode attn --skip_final_steps_pct 0 \
  --weight_loss_face_realistic 0 --weight_loss_img 1 \
  --use_zscore_logit true --grad_log_dir ./zscore_grad_logs --save_attmaps
```

---

## 6. 미해결 & 다음 스텝

- **최우선**: eval SDS-gap을 재서 **약함(A) vs 프록시(B)** 판정. (로깅은 추가됨, run 재시작 필요.)
- B면 → 분류기를 CLIP/mnet으로 (근본 해결). A면 → §4의 2~5로 fair 샤프닝.
- 어느 경우든 **off-manifold 화질 벽**(fair 강화 시 화질 하락)은 남는 문제 → 분류기 정렬 + 깨끗한 신호가 함께 가야 스윗스팟(gap↓ & 화질↑) 가능.
- 미확정으로 남긴 것: 프록시 불일치의 정확한 크기(같은 이미지 SDS-vs-mnet 필요), attmap을 focus시키면 SNR이 실제로 오르는지.
