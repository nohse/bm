import base64, json, os

OUT = "/workspace/bm/exp-1-debias-gender/scr_softmask_out"
R = json.load(open(f"{OUT}/results.json"))


def img(name):
    b = base64.b64encode(open(f"{OUT}/opt_{name}", "rb").read()).decode()
    return f"data:image/png;base64,{b}"


F1, F2, F3, F4 = (img(f"fig{i}_{n}.png") for i, n in
                  [(1, "mask_grid"), (2, "budget_consistency"), (3, "design_space"), (4, "caveat_control")])

CSS = """
:root{
  --paper:#f5f7f9; --paper-2:#ffffff; --ink:#161b22; --ink-2:#3d4753; --muted:#6b7480;
  --rule:#dde3e9; --rule-2:#eef2f5;
  --released:#b93a2c;   /* the released / damped end of the mask colormap */
  --preserved:#1f6296;  /* the preserved end */
  --ok:#12796a;
  --warn:#a4690a;
  --chip:#e9eff4;
  --serif: "Iowan Old Style","Charter","Palatino Linotype",Palatino,Georgia,serif;
  --sans: system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
  --mono: ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
}
@media (prefers-color-scheme: dark){
  :root{
    --paper:#12161b; --paper-2:#171d24; --ink:#e6ebf0; --ink-2:#b8c2cc; --muted:#8b96a2;
    --rule:#2a333d; --rule-2:#20272e;
    --released:#e0705f; --preserved:#6aa8d8; --ok:#3fbfa8; --warn:#d8a04a; --chip:#212a33;
  }
}
:root[data-theme="dark"]{
  --paper:#12161b; --paper-2:#171d24; --ink:#e6ebf0; --ink-2:#b8c2cc; --muted:#8b96a2;
  --rule:#2a333d; --rule-2:#20272e;
  --released:#e0705f; --preserved:#6aa8d8; --ok:#3fbfa8; --warn:#d8a04a; --chip:#212a33;
}
:root[data-theme="light"]{
  --paper:#f5f7f9; --paper-2:#ffffff; --ink:#161b22; --ink-2:#3d4753; --muted:#6b7480;
  --rule:#dde3e9; --rule-2:#eef2f5;
  --released:#b93a2c; --preserved:#1f6296; --ok:#12796a; --warn:#a4690a; --chip:#e9eff4;
}
*{box-sizing:border-box}
body{background:var(--paper);color:var(--ink);font-family:var(--sans);line-height:1.65;
  -webkit-font-smoothing:antialiased;margin:0;padding:0 20px 90px;}
.wrap{max-width:730px;margin:0 auto}
.bleed{max-width:1120px;margin:34px auto}

header{padding:58px 0 26px;border-bottom:1px solid var(--rule);margin-bottom:34px}
.eyebrow{font-family:var(--mono);font-size:11.5px;letter-spacing:.13em;text-transform:uppercase;
  color:var(--muted);margin:0 0 14px}
h1{font-family:var(--serif);font-weight:600;font-size:clamp(28px,4.4vw,40px);line-height:1.16;
  margin:0 0 16px;text-wrap:balance;letter-spacing:-.01em}
.dek{font-size:18px;color:var(--ink-2);margin:0;text-wrap:pretty}
.meta{font-family:var(--mono);font-size:12px;color:var(--muted);margin-top:20px}

h2{font-family:var(--serif);font-weight:600;font-size:24px;margin:52px 0 6px;letter-spacing:-.01em;
  text-wrap:balance}
h2 .num{font-family:var(--mono);font-size:12px;color:var(--muted);display:block;margin-bottom:6px;
  letter-spacing:.1em}
h3{font-size:15px;font-weight:650;margin:30px 0 4px;color:var(--ink)}
p{margin:12px 0}
code{font-family:var(--mono);font-size:.885em;background:var(--chip);padding:1px 5px;border-radius:3px}
strong{font-weight:650}
em{font-style:normal;font-weight:650;color:var(--released)}

.verdict{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:1px;
  background:var(--rule);border:1px solid var(--rule);border-radius:5px;overflow:hidden;margin:30px 0}
.v{background:var(--paper-2);padding:15px 16px}
.v .k{font-family:var(--mono);font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted)}
.v .n{font-family:var(--mono);font-size:23px;font-weight:600;margin-top:5px;
  font-variant-numeric:tabular-nums}
.v .s{font-size:12.5px;color:var(--ink-2);margin-top:2px}
.v.bad .n{color:var(--released)} .v.good .n{color:var(--preserved)} .v.neu .n{color:var(--ok)}

figure{margin:0}
figure img{width:100%;height:auto;display:block;border:1px solid var(--rule);border-radius:5px;
  background:var(--paper-2)}
figcaption{font-size:13px;color:var(--muted);margin-top:10px;max-width:730px;
  margin-left:auto;margin-right:auto;text-align:left}
figcaption b{color:var(--ink-2);font-weight:650}

.tw{overflow-x:auto;margin:20px 0;border:1px solid var(--rule);border-radius:5px;background:var(--paper-2)}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:9px 13px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--rule-2)}
th:first-child,td:first-child{text-align:left;white-space:normal}
thead th{font-family:var(--mono);font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;
  color:var(--muted);background:var(--chip);border-bottom:1px solid var(--rule)}
tbody tr:last-child td{border-bottom:none}
td.num{font-family:var(--mono)}
tr.base td{background:color-mix(in srgb,var(--released) 8%,transparent)}
tr.rec td{background:color-mix(in srgb,var(--preserved) 10%,transparent);font-weight:600}
.tag{font-family:var(--mono);font-size:10px;letter-spacing:.05em;padding:2px 6px;border-radius:3px;
  text-transform:uppercase;vertical-align:middle;margin-left:6px}
.tag.cur{background:color-mix(in srgb,var(--released) 18%,transparent);color:var(--released)}
.tag.rec{background:color-mix(in srgb,var(--preserved) 18%,transparent);color:var(--preserved)}
.tag.no{background:var(--chip);color:var(--muted)}

.callout{border-left:2px solid var(--released);padding:2px 0 2px 18px;margin:26px 0;color:var(--ink-2)}
.callout.blue{border-left-color:var(--preserved)}
.callout.ok{border-left-color:var(--ok)}
.callout p:first-child{margin-top:0} .callout p:last-child{margin-bottom:0}

pre{background:var(--paper-2);border:1px solid var(--rule);border-radius:5px;padding:14px 16px;
  overflow-x:auto;font-family:var(--mono);font-size:12.5px;line-height:1.6;color:var(--ink-2)}
pre b{color:var(--ink);font-weight:650}

ul{padding-left:20px;margin:12px 0} li{margin:7px 0}
hr{border:none;border-top:1px solid var(--rule);margin:52px 0}
.foot{font-size:13px;color:var(--muted);margin-top:44px}
"""

HTML = f"""<title>SCR gradient gate: hard threshold to soft mass mask</title>
<style>{CSS}</style>
<div class="wrap">
<header>
  <p class="eyebrow">Toy experiment &middot; 50 real scoring-time attention maps &middot; CPU</p>
  <h1>SCR 경사 게이트: 하드 임계값을 부드러운 &ldquo;누적 질량&rdquo; 마스크로</h1>
  <p class="dek">요청하신 두 가지(softmax&nbsp;temperature, 누적확률&nbsp;top-p)를 실제 attention map 50장 위에서
  구현·측정했습니다. 결론부터: <b>진짜 문제는 마스크가 &ldquo;딱딱해서&rdquo;가 아니라, 임계값을 <em>attention 값</em>에
  걸어서 샘플마다 예산이 통제되지 않는 것</b>이었습니다.</p>
  <p class="meta">bm/exp-1-debias-gender &middot; 1-main-errorDAL,SCR,SRR_person_truncated_hspace_nodetector.py L3121-3128</p>
</header>

<div class="verdict">
  <div class="v bad"><div class="k">현재 게이트 감쇠 면적</div><div class="n">25&ndash;90%</div>
    <div class="s">평균 67%, CV 0.19</div></div>
  <div class="v bad"><div class="k">푸는 attention mass</div><div class="n">0.39&ndash;0.92</div>
    <div class="s">사실상 p가 흔들리는 top-p</div></div>
  <div class="v good"><div class="k">권장 게이트 CV</div><div class="n">0.031</div>
    <div class="s">6.2배 감소</div></div>
  <div class="v neu"><div class="k">wSCR 재튜닝</div><div class="n">불필요</div>
    <div class="s">&lt;m&gt; 0.463 &rarr; 0.466</div></div>
</div>

<h2><span class="num">01</span>현재 게이트의 정체</h2>
<p>학습 코드는 <code>minmax(common_attn) &ge; 0.15</code>인 픽셀의 SCR 경사를 <code>&times;0.2</code>로 죽입니다.
그런데 이 gender cross-attention map은 <strong>거의 균일합니다</strong> — 정규화 엔트로피 0.996 (1.0 = 완전 균일),
peak/mean 비율 2.18. 그런 맵에 <em>값</em> 기준 임계값을 걸면 어떤 영역이 잡힐지는 사실상 그 샘플의 대비가 결정합니다.</p>
<p>50장에서 측정한 결과, 이 게이트는 <strong>사람 영역 마스크가 아니라 이미지의 평균 67%에 걸친 거의 전역적인
&times;0.2 감쇠</strong>이며, 그 범위가 25%에서 90%까지 흔들립니다. 푸는 gender-attention mass로 환산하면 0.39&ndash;0.92.
즉 <strong>이미 top-p 게이트인데, p가 통제되지 않는 상태</strong>입니다.</p>

<div class="bleed"><figure>
  <img src="{F2}" alt="per-sample released energy and released attention mass">
  <figcaption><b>왼쪽</b>: 점 하나가 이미지 한 장. 현재 게이트가 푸는 경사 에너지 λ가 샘플마다 5배 흔들립니다(CV 0.191).
  질량 기준 게이트는 그렇지 않습니다(CV 0.03&ndash;0.04). <b>오른쪽</b>: 현재 게이트가 푸는 attention mass의 분포 —
  평균 0.75이지만 0.39에서 0.92까지 퍼져 있습니다.</figcaption>
</figure></div>

<h2><span class="num">02</span>요청하신 두 방법, 있는 그대로 평가</h2>

<h3>방법 1 — softmax temperature (τ &lt; 1) <span class="tag no">구조적으로 불가</span></h3>
<p>픽셀 softmax는 합이 1이라 값이 ~10⁻⁴ 수준이므로 마스크로 쓸 수 없습니다. 최대값으로 정규화하는 게 유일한 해석인데,
min-max 정규화된 <code>g</code>에 대해 그것은 정확히 <code>s = exp((g−1)/τ)</code>입니다 (peak에서 s=1 → 감쇠 깊이는 정확히 0.2 유지).</p>
<p>문제는 이 함수가 <code>g</code>에 대해 <strong>볼록(convex)</strong>이고 양끝이 고정되어 있어 항상 <code>s ≤ g</code>라는 것입니다. 따라서</p>
<div class="callout">
  <p><strong>mean(s) ≤ mean(g) = 0.257</strong> — 어떤 τ에서도. 현재 게이트는 mean(s) = 0.672를 씁니다.</p>
  <p>즉 <strong>value-domain softmax 마스크는 τ를 아무리 키워도, 깊이를 1.0까지 키워도 현재 게이트의 감쇠 총량에
  도달할 수 없습니다.</strong> 요청하신 τ=0.25에서는 λ=0.041로 현재(0.537)의 <strong>1/13</strong>, 사실상 no-op입니다.
  그리고 고정 예산에서 비교하면 보존 대비 P는 1.27&ndash;1.38 — 거의 국소화가 없습니다.</p>
</div>
<p>덧붙여 <strong>floor 함정</strong>이 있습니다: 보정 없이 쓰면 배경 전체가 <code>exp(−1/τ)</code>만큼 감쇠됩니다
(τ=0.5 → 13.5%, τ=1.0 → 36.8%). 반드시 floor-correction이 필요합니다.</p>

<h3>방법 2 — 누적확률 top-p 50% + 내부 soft <span class="tag rec">방향은 정확함</span></h3>
<p>파라미터를 <em>값</em>이 아니라 <em>질량</em>으로 잡는다는 게 핵심이고, 이건 정확히 맞는 진단입니다. 질량 기준이므로
샘플의 peakiness와 무관하게 예산이 고정됩니다(CV 0.19 → 0.04).</p>
<p>다만 <strong>p=0.5는 현재보다 훨씬 작습니다</strong>. 현재 게이트가 실제로 푸는 건 mass의 약 0.75입니다.
p=0.5에 soft ramp까지 넣으면 λ=0.065로 현재의 1/8이 됩니다 — 배경은 아주 깨끗해지지만(R_bg 0.012 vs 0.339)
얼굴을 푸는 힘도 0.161로 무너져 debias 압력이 사라집니다.</p>

<h2><span class="num">03</span>왜 그런가 — 세 가지 원리</h2>
<p><strong>(1) 세 가지 &ldquo;스케일&rdquo;은 서로 다릅니다.</strong> 감쇠 <em>깊이</em>(peak에서 ×0.2), 감쇠 <em>총량</em>
λ=mean(1−m), 그리고 <em>배분</em>. 요청하신 &ldquo;크기가 비슷해야 한다&rdquo;는 깊이로 읽으면 모든 방법이 자동 충족(peak s=1),
총량으로 읽으면 peaky 마스크는 원리상 도달 불가입니다(λ ≤ mean(s)).</p>
<p><strong>(2) 모든 마스크는 같은 맵의 단조 변환</strong>이라 픽셀 <em>랭킹</em>이 동일합니다 (AUC 0.913으로 전부 같음).
따라서 설계 문제는 &ldquo;어디를 고르나&rdquo;가 아니라 <strong>&ldquo;정해진 예산을 랭킹 위에 어떻게 배분하나&rdquo;</strong>뿐입니다.</p>
<p><strong>(3) 고정 예산에서는 bang-bang(하드)이 최적입니다.</strong> 픽셀당 상한이 있는 선형 배분 문제라 상위 픽셀부터
가득 채우는 게 LP 최적해입니다. 실제로 τ를 키울수록 selectivity가 단조 감소합니다(3.31 → 2.54).
<strong>soft는 국소화를 사주지 않습니다. 대신 안정성을 사줍니다.</strong></p>

<div class="callout blue">
  <p>그래서 실제 개선의 크기는 이렇게 갈립니다 — 전역 SCR 세기 &lt;m&gt;=0.463을 고정한 채(즉 <code>weight_loss_scr</code>를
  건드리지 않은 채) 보존 대비 P = m_bg/m_face를 재면:</p>
  <p><strong>value→mass 전환</strong>: P 2.81 → 2.69 (≈동일), 예산 CV <strong>0.193 → 0.036</strong>.<br>
  <strong>깊이(factor2) 조절</strong>: factor2 0.2 → 0.1 → 0.0에서 P <strong>2.7 → 4.2 → 7.3</strong>.</p>
  <p>즉 <strong>국소화 품질의 진짜 레버는 soft/hard가 아니라 factor2(깊이)</strong>였습니다. 이건 요청 범위 밖이지만
  가장 큰 발견이라 남겨둡니다.</p>
</div>

<div class="bleed"><figure>
  <img src="{F3}" alt="design space: depth, softness, stability">
  <figcaption><b>왼쪽</b>: 전역 SCR 세기를 고정했을 때의 보존 대비. 게이트 종류(가로축)는 거의 영향이 없고,
  깊이(factor2, 색)가 지배합니다. <b>가운데</b>: temperature를 키우면 국소화가 단조로 나빠집니다.
  <b>오른쪽</b>: 반대로 attention 소스를 바꿔치기(ALL/MID/RES16 블록)했을 때 게이트가 얼마나 흔들리는가 —
  soft할수록 안정적입니다(0.736 → 0.494, 33% 개선).</figcaption>
</figure></div>

<h2><span class="num">04</span>권장 게이트 — 두 아이디어를 한 게이트의 두 축으로</h2>
<p>두 제안은 사실 하나의 게이트의 서로 다른 두 knob입니다. 픽셀의 <strong>누적 attention 질량 C</strong>
(peak에서 0, 최약 픽셀에서 1)를 좌표로 쓰면:</p>
<pre><b>s = sigmoid((p_mid − C) / τ)</b>   , C=0 → s=1, C=1 → s=0 이 되도록 재정규화
m = 1 − (1 − factor2) · s

  p_mid  ← 방법 2: 어느 <b>누적확률</b>에서 자를지 (예산을 샘플 무관하게 고정)
  τ      ← 방법 1: 경계를 얼마나 <b>부드럽게</b> (τ→0 = 하드 nucleus)</pre>
<p>temperature를 <strong>값 도메인이 아니라 질량 도메인</strong>에 거는 것이 핵심입니다. 값 도메인에서는
face와 background의 min-max 값이 거의 안 벌어져(mean g = 0.26) 아무리 sharpening 해도 힘이 안 실립니다.</p>

<div class="tw"><table>
<thead><tr><th>게이트</th><th>λ (푸는 총량)</th><th>λ CV</th><th>R_face ↑</th><th>R_bg ↓</th>
<th>P = m_bg/m_face ↑</th><th>wSCR 보정</th></tr></thead>
<tbody>
<tr class="base"><td>hard minmax≥0.15 <span class="tag cur">현재</span></td><td class="num">0.537</td>
  <td class="num">0.193</td><td class="num">0.765</td><td class="num">0.339</td><td class="num">2.81</td><td class="num">1.00</td></tr>
<tr class="rec"><td>mass_sigmoid p=0.80 τ=0.12 <span class="tag rec">드롭인</span></td><td class="num">0.534</td>
  <td class="num">0.031</td><td class="num">0.740</td><td class="num">0.397</td><td class="num">2.3</td><td class="num">0.99</td></tr>
<tr><td>mass_sigmoid p=0.65 τ=0.12 <span class="tag rec">더 선택적</span></td><td class="num">0.436</td>
  <td class="num">0.043</td><td class="num">0.684</td><td class="num">0.274</td><td class="num">2.6</td><td class="num">0.82</td></tr>
<tr><td>mass_sigmoid p=0.50 τ=0.12</td><td class="num">0.331</td><td class="num">0.057</td>
  <td class="num">0.593</td><td class="num">0.164</td><td class="num">4.0</td><td class="num">0.69</td></tr>
<tr><td>softmax τ=0.25 (요청 그대로) <span class="tag no">no-op</span></td><td class="num">0.041</td>
  <td class="num">0.359</td><td class="num">0.075</td><td class="num">0.015</td><td class="num">1.06</td><td class="num">0.48</td></tr>
<tr><td>top-p 0.5 soft (요청 그대로)</td><td class="num">0.065</td><td class="num">0.273</td>
  <td class="num">0.161</td><td class="num">0.012</td><td class="num">1.18</td><td class="num">0.50</td></tr>
</tbody></table></div>
<p style="font-size:13px;color:var(--muted)">R_face = 얼굴 영역에서 풀린 경사 비율(debias 압력, 높을수록 좋음),
R_bg = 배경에서 풀린 비율(fidelity 비용, 낮을수록 좋음). wSCR 보정 = 평균 SCR 세기를 현재와 같게 맞추려면
<code>weight_loss_scr</code>에 곱할 값.</p>

<div class="callout ok">
  <p><strong>드롭인 설정</strong>: <code>--scr_mask_mode mass_sigmoid --scr_mask_pmid 0.80 --scr_mask_tau 0.12</code></p>
  <p>평균 SCR 세기 &lt;m&gt;이 0.463 → 0.466이라 <code>weight_loss_scr</code>를 <strong>건드릴 필요가 없습니다</strong>.
  바뀌는 건 오직 마스크 <em>모양</em>뿐이라 깨끗한 A/B가 됩니다. 예산 변동은 6.2배 줄고(CV 0.193 → 0.031),
  attention 소스 교란에 24% 더 안정적입니다.</p>
</div>

<div class="bleed"><figure>
  <img src="{F1}" alt="mask grid">
  <figcaption>샘플별로 게이트가 실제 경사에 무슨 짓을 하는지. 행은 <b>현재 게이트의 감쇠 면적 순</b>으로 정렬 —
  25%(위)에서 85%(아래)까지 통제 없이 흔들리는 게 보입니다. 방법 1·2를 요청하신 파라미터 그대로 쓰면
  λ가 0.02&ndash;0.08로 사실상 아무것도 안 합니다(거의 전부 파란색 = 보존). 권장 게이트는 모든 행에서
  λ≈0.42&ndash;0.47로 일정하고, 붉은 영역이 인물 실루엣을 따라갑니다.</figcaption>
</figure></div>

<h2><span class="num">05</span>적용 방법</h2>
<p>새 변형 파일을 만들어 뒀습니다 (기존 파일 무수정).
<code>1-main-errorDAL,SCR,SRR_person_truncated_hspace_nodetector_softgate.py</code></p>
<pre>--scr_mask_mode {{hard, softmax, topp, mass_sigmoid}}   <b>기본 hard = 현재 동작과 bit-identical (검증됨)</b>
--scr_mask_pmid 0.80        누적 attention 질량 컷
--scr_mask_tau  0.12        경계 부드러움 (질량 단위)</pre>
<p>검증한 것들:</p>
<ul>
<li><strong>hard 모드 = 원본과 완전히 동일</strong> (max|diff| = 0.0, 실제 파일에서 import해 대조)</li>
<li><strong>no-face 샘플 안전</strong>: <code>common_attn</code>이 0으로 채워진 샘플에서 네 모드 모두 s=0 (감쇠 없음), NaN 없음.
  ← softmax는 floor 보정이 없으면 <em>무결점 샘플 전체를 감쇠</em>시키고, top-p는 0/0 NaN이 납니다. 둘 다 막았습니다.</li>
<li><strong>SRR 경로 무변경</strong>: <code>residual_gender_and_realism</code>이 쓰는 <code>attn_gate_thr</code> 하드 마스크는
  손대지 않았습니다(함수 바디 바이트 동일). 새 flag는 SCR 훅에서만 읽힙니다.</li>
<li><strong>run 이름에 항상 태깅</strong>: <code>_scrMask-mass_sigmoid-p0.8-t0.12</code></li>
<li><strong>gate 진단 로깅 추가</strong>: 게이트는 backward hook이라 <code>loss_SCR</code> <em>값</em>은 모든 arm에서 동일하게
  찍힙니다. wandb에서 arm을 구분하려면 경사 쪽 지표가 필요해서 <code>scr_grad_mask_mean</code>(=&lt;m&gt;)와
  <code>scr_release_frac</code>을 추가했습니다.</li>
</ul>

<h2><span class="num">06</span>정직한 한계</h2>
<div class="bleed"><figure>
  <img src="{F4}" alt="center prior control">
  <figcaption>대조군. gender attention이 얼굴 픽셀을 가려내는 AUC는 0.913인데, <b>attention을 아예 무시하는
  중앙 가우시안 prior</b>가 0.904입니다. 셔플 대조군이 0.498(=우연)이므로 지표 자체는 정상입니다.</figcaption>
</figure></div>
<ul>
<li><strong>이 데이터에서는 attention map이 &ldquo;얼굴은 가운데 있다&rdquo;는 prior를 이기지 못합니다.</strong>
  insightface 검출 임계값을 0.5/0.3/0.15로 바꿔가며(n=34/40/45) 재봐도 AUC 차이는 +0.009&ndash;+0.033이고
  paired sign test는 모두 유의하지 않습니다(p=0.23&ndash;0.88, 이미지별 승률 38&ndash;53%).
  다만 이 토이 프롬프트가 &ldquo;a photo of the face of a person&rdquo;(정중앙 클로즈업)이라 <em>center prior에게 최대로 유리한
  조건</em>입니다. 실제 학습은 occupation 프롬프트라 구도가 다양하므로, 이 음성 결과를 그대로 확대해석하면 안 됩니다.
  <strong>실제 프롬프트에서 이 대조를 다시 돌려야 합니다.</strong></li>
<li>50장, 프롬프트 1개, seed 1개. 그리고 이 맵들은 <strong>frozen SD1.5</strong>에서 뽑힌 것이라, LoRA가 학습되며
  드리프트한 뒤의 맵 통계는 확인되지 않았습니다.</li>
<li>모든 지표는 마스크 공간의 프록시입니다. 최종 판정은 실제 학습의 FD / gender-gap / CLIP-I로만 가능합니다.
  (GPU가 이 세션에서 죽어 있어 end-to-end는 못 돌렸습니다.)</li>
<li>face box를 &ldquo;정답 영역&rdquo;으로 썼습니다. SCR release가 노려야 하는 건 gender identity 영역이므로 대체로 타당하지만,
  머리카락·목·어깨를 올바르게 포함하는 마스크를 페널티 주는 편향이 있습니다.</li>
</ul>

<hr>
<p class="foot">재현: <code>python scr_softmask_faces.py</code> → <code>python scr_softmask_experiment.py</code> →
<code>python scr_softmask_figs.py</code> (모두 CPU). 원자료 <code>scr_softmask_out/results.json</code>.
마스크 구현 <code>scr_softmask_lib.py</code>. 학습 파일 패치 생성기 <code>make_softgate_variant.py</code>.</p>
</div>
"""

path = "/tmp/claude-0/-workspace/ed8ef1ea-99b6-4f85-b7e2-88a8cc89e078/scratchpad/scr_gate_report.html"
open(path, "w").write(HTML)
print("wrote", path, f"{os.path.getsize(path)/1e6:.2f} MB")
