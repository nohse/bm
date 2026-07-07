# attmap: generation vs scoring 추출 비교 실험

스크립트: [`../attmap_gen_vs_score_experiment.py`](../attmap_gen_vs_score_experiment.py)
재현: `CUDA_VISIBLE_DEVICES=1 python attmap_gen_vs_score_experiment.py --n_images 10 --out_dir ./attmap_gen_vs_score_out`

## 무엇을 뽑았나
사람 얼굴 10개(base SD1.5, prompt = `"a photo of the face of a person"`)를 생성하고,
각 이미지마다 attmap 을 **두 개의 독립 축**으로 뽑아서 비교한다.

- **축1 — 어디서 뽑나**
  - `GEN`  : 생성(denoising) 하면서 생성 프롬프트의 **`face` 토큰** attention (= 예전 "generation 하면서 뺀" 방식)
  - `SCORE`: 채점(SDS 재노이징 t∈[400,800]) 하면서 **`woman`/`man` 토큰** attention 평균 (= 지금 코드 `sds_logits_from_images` 방식)
- **축2 — 어느 layer**
  - `ALL`  : 모든 cross-attn layer 평균 (현재 `CrossAttnCapture` 방식)
  - `MID`  : `mid_block` cross-attn 만 (예전 `_install_mid_recorders`/`_attn_weight_for_prompt` 방식, 8×8)
  - `RES16`: 16×16 해상도 cross-attn 만 (prompt-to-prompt 에서 가장 의미있는 해상도)

그리고 그 attmap 을 **가중치 w** 로 써서 SDS residual error 를 곱→평균→
`logits=[-sds_f/τ, -sds_m/τ]`(col0=woman, col1=man) → softmax → `p(woman)/p(man)` → CE 로 변환한다.
**동일한 이미지 / eps / eps_pred / 재노이징을 쓰고 weight map 만 바꿔서** gender logit 이 어떻게 변하는지 본다.

## 파일
| 파일 | 내용 |
|---|---|
| `faces_montage.png` | 생성된 얼굴 10개 요약 |
| `ALL_panels.png` | 10개 전부의 비교 패널(세로로 stack) |
| `panels/panel_XX.png` | 이미지별 패널. 1행=overlay(GEN MID / SCORE MID / GEN ALL / SCORE ALL), 2행=heat(MID/RES16/ALL), 3행=`w×err_woman`, `w×err_man`, 하단=각 weight 방식의 `p(man)` |
| `images/img_XX.png` | 생성 원본 512² |
| `maps/attmaps.pt` | 모든 attmap 텐서(`GEN`/`SCORE` × `ALL/MID/RES16`, [N,h,w]) — 직접 재사용용 |
| `summary.json` | config, map spread 진단, CE 집계, per-image 수치 |

## 핵심 결과 (이번 실행)

### 1) map 자체는 방식마다 "모양"이 다르다
`effective_support`(작을수록 얼굴에 집중, `frac`=전체 픽셀 대비 비율):

| map | eff_support | frac | max/mean |
|---|---|---|---|
| GEN_MID | 55.8 (8×8 기준) | 0.87 | 1.92 |
| GEN_RES16 | 238 | 0.93 | 1.95 |
| GEN_ALL | 4011 | **0.98 (거의 균일)** | 1.55 |
| SCORE_MID | 3527 | 0.86 | 2.44 |
| SCORE_RES16 | 3816 | 0.93 | 1.85 |
| SCORE_ALL | 3866 | **0.94 (거의 균일)** | 2.25 |

→ `MID`/`RES16` 는 얼굴 근처 blob, `ALL` 은 화면 전체로 퍼진 거의 균일 map. (패널의 heat 행에서 눈으로 확인 가능)

### 2) 그런데 CE(성별 판정)는 어느 방식이든 **거의 동일**하다
| weight | mean_max_prob(↑또렷) | ambiguous(0.35~0.65) | gender_gap |
|---|---|---|---|
| GEN_MID | 0.955 | 0.000 | −0.20 |
| GEN_ALL | 0.945 | 0.000 | −0.20 |
| SCORE_MID | 0.954 | 0.000 | −0.20 |
| SCORE_ALL | 0.951 | 0.000 | −0.20 |
| **nomask** | 0.947 | 0.000 | −0.20 |

per-image `p(man)` 도 방식 간 차이가 거의 없다(예: img8 = GEN_MID 0.207 / SCORE_MID 0.232 / nomask 0.292 — 전부 "woman", 같은 대소관계).

## 해석 (요약)
**깨끗한 얼굴 이미지에서는, attmap 가중(generation이든 scoring이든, 어느 layer든)이 SDS 성별 판정/CE 에 거의 영향을 주지 않는다** —
`nomask`(마스크 없음)와도 사실상 같다. 즉 woman↔man residual 차이는 "얼굴 위치"에 국한돼 있지 않고 공간적으로 퍼져 있어서,
얼굴에 마스킹해도 적분값(sds_f, sds_m)이 거의 안 변한다. τ=1e-4 라 최종 확률은 `(sds_m−sds_f)/τ` 로 결정되는데 이 gap 이 weight 에 둔감하다.

→ "generation vs scoring attmap" 자체는 당신이 본 품질/애매함 차이의 **직접 원인이 아닐 가능성이 높다.**
   더 자세한 근본원인(ŵ vs ŵ² reduction, fp16, τ, gradient 경로 등) 분석은 상위 답변 참고.
