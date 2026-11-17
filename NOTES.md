# Notes — model, validation, and risks (v2)

Entry for make.fun `gta6-twitch-ccu-forecast-01`. The program is
`main.py` (Python 3.13, stdlib only). This file records how the model
was chosen, with the numbers, so the choice can be checked rather
than taken on trust.

**v2 changes vs the submitted v1** (see §7 for the full story):

1. Two new predictors: a **console-only indicator** (1 if the launch
   is console-only, 0 otherwise) and **log channels at hour 1**
   (clipped). Both are fitted per hour alongside the v1 predictors.
2. A **bug fix in the regime branch**: v1 classified a launch as
   simultaneous whenever the word "simultaneous" appeared anywhere
   in the case notes — including the phrase "no simultaneous launch".
   That misclassified God of War Ragnarök (practice case p02) and
   made v1 train on all 37 comparables for it, sharp-step ones
   included. v2 strips negated phrases before testing for an
   affirmed simultaneous unlock.
3. The console-only indicator is read from the case notes by a
   small parser (`target_console_only`); comparables use their
   `platforms` field. Both are documented in §4 and §8.

## 1. What the data looks like

Each case ships 38 comparables in the practice (42 in the final):
past big launches with hourly Twitch viewers from 14 days before to
3 days after their release day. One of them is not a launch at all —
the GTA VI category's own pre-release history ("…category before
release") — and is never used as a training curve.

The comparables split into two regimes, visible in their own release
day:

- **Gradual / rolling-midnight** (25 of 37 usable curves): no single
  hour multiplies viewers by more than ~2.7x (max one-hour log jump
  < 1.0). The game unlocks region by region; hour 1 already contains
  New Zealand / Australia / Asia / Europe, and the US unlocks add
  viewers through the morning. GTA VI and all four practice launches
  are of this kind (their case notes say "rolling midnight local").
- **Sharp-step / simultaneous-global** (12 of 37): the audience
  multiplies several-fold in one hour at the single global unlock
  (e.g. Hollow Knight: Silksong: hour 1 = 164 viewers, peak 347,773;
  Black Myth: Wukong: 4,941 → 454,807). Hour 1 is pre-unlock hype
  and says almost nothing about the peak.

Pooled across all 37 launches, the median log-ratio to hour 1 has a
standard deviation of ~2.1 log units by the evening — unusable. The
single most informative fact is hour 1 itself: the correlation
between log(hour-1 viewers) and log(viewers at hour 18 / hour 1) is
about −0.91. A launch that drew 800k in hour 1 stays near flat; one
that drew 400 in hour 1 can draw 1,000x later.

**Console-only launches rise much more than PC-included ones.**
Of the 25 gradual comparables, exactly 8 are console-only (no PC or
Stadia in `platforms`): Ghost of Yōtei, Saros, Zelda: Tears of the
Kingdom, Death Stranding 2, Final Fantasy VII Rebirth, The Last of
Us Part II, Ghost of Tsushima, Stellar Blade. Their median peak is
2.66x hour 1 (at hour 19); the 17 multi-platform gradual
comparables' median peak is only 1.26x (at hour 18). All four
practice launches are console-only, and GTA VI is console-only
(PS5 + Xbox Series X|S, no PC until 2027). v1 trained on the pooled
gradual set, so its forecasts sat between the two shapes — too low
in the evening for a console-only launch. That is the main reason
v1's practice score was bad (see §7).

## 2. Validation method — and a correction to the practice proxy

### The practice answers are NOT in the kit

An early assumption in this v2 work was that each practice game
appears as a comparable in the other practice cases' inputs, so its
true hours 2–24 curve could be scored locally. That is **false**,
verified directly:

- All four cases (p00–p03) ship the **same** 38 comparables; none is
  Wolverine, Red Dead Redemption 2, God of War Ragnarök, or
  Marvel's Spider-Man 2 (byte-identical comparable lists across all
  four files).
- `competition.yaml` and `RULES.txt` both state: *"No case holds
  another case's answer: the comparables in a practice case leave
  out every practice launch."* The kit's `input/NOTICE.txt` repeats
  this and adds that the final case holds them all. The dev phase
  grades on a sealed practice instance (`input/`, `reference/`);
  `reference/` is not in the kit, and the harness tarball contains
  only grader/source code, no answer data.

The kit's NOTICE also asks not to scrape or bulk-download
SullyGnome (the source of the counts). The practice answers were
therefore **not** fetched from SullyGnome, and **no measured
practice loss for v1 or v2 exists in this work**. Any claim of a
v2 practice loss < 0.20 would be invented; this file does not make
one.

### What was used instead: stratified leave-one-out on the comparables

**Leave-one-out cross-validation on the comparables.** Each
comparable launch in turn is treated as a pseudo-case: its hour-1
print is taken as known, hours 2–24 are predicted from the *other*
comparables only, and the prediction is scored with the contest's
pinball loss on the log scale (mean over hours 2–24 × 5 quantiles),
exactly as the grader scores. Pinball here is reported on the
grader's scale (2 × standard pinball); the v1 numbers below
reproduce v1's NOTES exactly, which confirms the harness is
faithful to v1.

Three LOO populations are reported, because they answer different
questions:

- **Gradual launches (n = 25)** — the population GTA VI belongs to.
  This is the population the model was selected on.
- **Console-only gradual launches (n = 8)** — the population GTA VI
  and all four practice launches belong to. Small (n = 8), so it is
  a secondary signal, not the selection metric on its own.
- **All launches (n = 37)** — reported for transparency; it mixes in
  the simultaneous regime, which the chosen model deliberately does
  not train on for a rolling target.

With n = 25, differences in mean pinball below ~0.01 are within
noise (paired per-case differences have SD ~0.06). The model was
therefore picked on a combination of the LOO numbers *and*
principle (smooth coefficients, clipped features, no over-fitting
to one launch), not on the smallest decimal.

**The shipped code was validated end-to-end**, not just the
experiment scripts: a harness builds synthetic cases from the
comparables (one target held out, its notes written from its true
platforms and unlock, comparables as raw case dicts) and calls the
shipped `forecast_case` directly. It reproduces the experiment
numbers exactly (gradual 0.165, all 0.339, console-only 0.148),
so the shipped code is the evaluated code.

## 3. Models tried (LOO pinball loss, mean / median, grader scale)

### v1 models (carried over; see v1 NOTES for the full table)

| model | gradual (n=25) | all (n=37) |
|---|---|---|
| House baseline (starter: persistence, σ = 0.15·√(h−1)) | 0.278 / 0.253 | 0.930 / 0.325 |
| Pooled empirical quantiles of log-ratio per hour | 0.232 / 0.211 | 0.714 / 0.408 |
| Log-log regression on log(hour 1) | 0.183 / 0.167 | 0.439 / 0.350 |
| **v1 chosen: regression + pre-day feature + smoothing + spread ×1.15** | **0.178 / 0.152** | **0.343 / 0.244** |
| Regression trained on **all** comparables (no regime split) | 0.325 / 0.293 | — |

### v2 candidates (all LOO, same populations)

| model | gradual | console-only (n=8) | all (n=37) |
|---|---|---|---|
| v1 chosen (baseline for comparison) | 0.178 / 0.152 | 0.166 / 0.152 | 0.343 / 0.244 |
| Kernel-weighted log-ratio quantiles, 1D log hour-1, bw 0.5 | 0.200 | 0.215 | — |
| Kernel-weighted, 2D (log hour-1 + pre-day ratio), bw 0.5/0.5 | 0.200 | 0.207 | — |
| Local-linear (kernel-weighted OLS), bw 0.5 | 0.182 | 0.174 | — |
| Ensemble 50/50 regression + kernel (log-median blend) | 0.184 | 0.182 | — |
| Ensemble 70/30 regression + kernel | — | 0.173 | — |
| Pooled empirical quantiles (gradual) | 0.232 | 0.278 | — |
| Regression trained on all comparables (no regime split) | 0.332 | 0.324 | — |
| **v1 + console-only indicator** | **0.176 / 0.162** | **0.153 / 0.140** | — |
| v1 + console-only indicator, console curves weighted 3× | 0.173 | 0.157 | — |
| v1 + console-only indicator, train only on console-only when target is console-only | 0.193 | 0.214 | — |
| v1 + console-only indicator, early-access-contaminated excluded | 0.186 | 0.152 | — |
| v1 + console-only indicator + post-2023 indicator | 0.186 | 0.151 | — |
| v1 + log channels at hour 1 (no indicator) | 0.181 | 0.164 | — |
| **v2 chosen: v1 + console-only indicator + log channels at hour 1** | **0.165 / 0.126** | **0.148 / 0.123** | **0.339 / 0.206** |
| v2 chosen, spread ×1.3 instead of ×1.15 | 0.165 | 0.150 | — |
| v2 chosen, console curves weighted 3× | 0.163 | 0.151 | — |
| v2 chosen, spread ×1.0 / ×1.3 / ×1.5 / ×1.8 / ×2.2 (gradual) | 0.179 / 0.179 / 0.179 / 0.184 / 0.212 | 0.165 / 0.173 / — / 0.185 / — | — |

Findings:

- **Kernel / neighbour methods do not beat the regression.** The
  kernel quantiles are worse than v1 on every LOO population
  (gradual 0.200 vs 0.178; console-only 0.215 vs 0.166), and
  catastrophic on individual console launches (Saros 0.362 vs
  v1 0.152; Ghost of Yōtei 0.325 vs 0.178). The top leaderboard
  practice scores (~0.097–0.11) are therefore not explained by a
  kernel model; the #1 leaderboard entry ("ccu curve memory",
  loss 0.00201, typical miss 0%, coverage 1.0) looks like
  memorised practice answers, not a general model.
- **The console-only indicator helps on both populations**
  (gradual 0.178 → 0.176; console-only 0.166 → 0.153) and
  **log channels helps only in combination with it** (alone:
  gradual 0.181, console-only 0.164). log(channels) and
  log(viewers-per-channel) give identical results, because
  log(vpc) = log(p1) − log(channels) and log(p1) is already a
  regressor — they span the same space. Channels were kept
  because they are what case files report at hour 1.
- **Widening the spread monotonically hurts LOO** (×1.0: 0.179,
  ×1.15: 0.178, ×1.5: 0.184, ×2.2: 0.212 on gradual), so the
  spread stays at ×1.15 even though v1's practice coverage was
  0.804 — widening cannot be justified from LOO alone.
- Training only on console-only curves when the target is
  console-only is bad (console-only 0.214): n = 7 training curves
  is too few. Weighting console curves 3× is a wash (gradual
  0.163, console-only 0.151) and was not adopted — simpler is
  better at this sample size.
- Per-hour bias correction does nothing (OLS with an intercept
  already centres residuals).

### Per-case LOO loss, v1 vs v2 (gradual launches, grader scale)

| launch | hour 1 | v1 | v2 | Δ |
|---|---|---|---|---|
| ELDEN RING | 875,155 | 0.097 | 0.084 | −0.013 |
| The Last of Us Part II | 365,019 | 0.115 | 0.084 | −0.031 |
| Elden Ring Nightreign | 319,909 | 0.113 | 0.088 | −0.025 |
| Ghost of Tsushima | 131,910 | 0.147 | 0.090 | −0.057 |
| Call of Duty: Modern Warfare III | 114,417 | 0.135 | 0.100 | −0.035 |
| Final Fantasy VII Rebirth | 53,526 | 0.111 | 0.105 | −0.006 |
| Stellar Blade | 27,057 | 0.128 | 0.110 | −0.018 |
| Assassin's Creed Shadows | 146,993 | 0.240 | 0.110 | −0.130 |
| Call of Duty: Black Ops 7 | 79,207 | 0.096 | 0.112 | +0.016 |
| Cyberpunk 2077 | 851,999 | 0.124 | 0.118 | −0.006 |
| Saros | 6,400 | 0.152 | 0.123 | −0.029 |
| Call of Duty: Modern Warfare 4 | 59,162 | 0.115 | 0.125 | +0.010 |
| Ghost of Yōtei | 27,206 | 0.178 | 0.126 | −0.052 |
| DOOM: The Dark Ages | 50,351 | 0.259 | 0.144 | −0.115 |
| 007 First Light | 127,800 | 0.244 | 0.152 | −0.092 |
| Hogwarts Legacy | 387,358 | 0.119 | 0.170 | +0.051 |
| Forza Horizon 5 | 43,895 | 0.266 | 0.195 | −0.071 |
| Borderlands 4 | 133,568 | 0.253 | 0.204 | −0.049 |
| Diablo IV | 533,455 | 0.196 | 0.206 | +0.010 |
| Marvel Rivals | 165,939 | 0.182 | 0.222 | +0.040 |
| Forza Horizon 6 | 26,312 | 0.149 | 0.222 | +0.073 |
| Zelda: Tears of the Kingdom | 129,948 | 0.323 | 0.232 | −0.091 |
| Death Stranding 2 | 42,221 | 0.170 | 0.315 | **+0.145** |
| Starfield | 156,828 | 0.284 | 0.346 | +0.062 |
| Call of Duty: Black Ops 6 | 139,097 | 0.260 | 0.351 | **+0.091** |
| **Mean** | | **0.178** | **0.165** | **−0.013** |
| **Median** | | **0.152** | **0.126** | **−0.026** |

v2 improves on 16/25 gradual launches and worsens on 9/25; on the
console-only subset it improves on 7/8 (only The Last of Us Part II
worsens, 0.115 → 0.147, and Death Stranding 2 is the worst
worsening overall). Paired differences on gradual: mean −0.013,
SD ~0.064 — the mean gain is **not statistically strong on its
own**; the median gain (−0.026) and the console-only gain (−0.018)
are the stronger signals. Coverage and typical miss, gradual:
v1 0.890 / 28.0% → v2 0.847 / 22.4%; console-only:
v1 0.924 / 28.1% → v2 0.875 / 19.5%.

### If the target were simultaneous instead

On the 12 sharp-step pseudo-cases, v2 scores 0.699 vs v1's 0.687
(baseline 2.287) — slightly worse, because the new features are
fitted on the gradual subset and the simultaneous branch trains on
all comparables. Sharp-step is not GTA VI's regime, and the
all-launches LOO is still better overall (0.339 vs 0.343).

### Extrapolation check (GTA VI may exceed every comparable)

Trained only on gradual launches with hour 1 ≤ 200,000 and tested
on the 6 gradual launches with hour 1 > 200,000: v1 regression
only 0.196, regression + pre-day feature + smoothing 0.151. The
pre-day feature is what carries the extrapolation — it is kept,
clipped, so an out-of-range value cannot dominate. Channels are
also clipped (log channels to [1.8, 10.0], the observed range
6–21,700), so a record-breaking GTA VI hour 1 cannot blow up the
log-channels term.

## 4. The chosen v2 model

Per hour h = 2..24, OLS on the gradual comparables:

```
log(viewers_h) = a_h + b_h·log(p1) + c_h·log(pre24max / p1)
               + d_h·console_only + e_h·log(channels at hour 1)
```

- `p1` = hour-1 viewers; `pre24max` = peak viewers in the 24 hours
  before release (clipped ratio to [0.02, 8]); `console_only` = 1 if
  the launch has no PC (and no Stadia) platform, else 0;
  `log(channels at hour 1)` clipped to [1.8, 10.0].
- Coefficients are smoothed across neighbouring hours (0.25/0.5/
  0.25). Spread = 1.15 × per-hour residual SD, floored at 0.18 and
  capped at 0.90 log units; quantiles are normal quantiles of the
  median log forecast.
- Feature fallback: if the full five-feature fit cannot be made at
  any hour (a feature near-constant, or too few curves), the last
  feature is dropped for every hour and the fit retried, down to
  log(p1) alone — so all hours always use the same feature set.
- If the target's channels are missing, the training set's median
  log-channels is used. If too few gradual curves are usable, the
  fit trains on all comparables; if no fit is possible, the house
  baseline is used.

Smoothed coefficients, fitted on all 25 gradual comparables
(console indicator / log-channels shown; log(p1) slope and
pre-feature are as in v1, slope ~1.06 at hour 2 falling to ~0.65
in the evening):

| hour | console d_h | log-channels e_h |
|---|---|---|
| 2 | −0.06 | +0.07 |
| 3 | +0.06 | +0.20 |
| 4 | +0.31 | +0.35 |
| 5 | +0.60 | +0.42 |
| 6 | +0.80 | +0.45 |
| 7 | +0.84 | +0.47 |
| 8 | +0.77 | +0.38 |
| 9 | +0.67 | +0.30 |
| 10 | +0.59 | +0.25 |
| 11 | +0.54 | +0.22 |
| 12 | +0.55 | +0.20 |
| 13 | +0.60 | +0.19 |
| 14 | +0.67 | +0.18 |
| 15 | +0.69 | +0.16 |
| 16 | +0.62 | +0.15 |
| 17 | +0.54 | +0.14 |
| 18 | +0.52 | +0.16 |
| 19 | +0.49 | +0.18 |
| 20 | +0.49 | +0.20 |
| 21 | +0.54 | +0.22 |
| 22 | +0.50 | +0.24 |
| 23 | +0.39 | +0.26 |
| 24 | +0.34 | +0.27 |

Note the console and log-channels coefficients are larger in the
combined fit than in the indicator-only fit — console, log(p1) and
log(channels) are correlated (log(channels) vs log(p1) r = 0.833),
so individual coefficients are unstable even though predictions
improve. Do not read a causal story into any single coefficient.

### How console-only is determined

- **Comparables:** from the `platforms` field — console-only iff
  neither "pc" nor "stadia" appears (case-insensitive).
- **Target:** the case notes have no `platforms` field, so
  `target_console_only(notes)` reads only the release sentence
  (text from "released" to the next ";", up to 160 chars):
  "console only" / "no pc" / "without pc" → True; a console token
  (ps5/ps4/xbox/switch/playstation) followed by "only" → True; a
  standalone "pc" (word-boundary match, so "PC," counts) → False,
  unless it is followed by a year ("pc 2019-11", "pc 2027"), which
  means a later port → True; any console token and no PC → True;
  anything unclear defaults to False (conservative — no console
  boost). Verified on all four practice cases and on GTA VI's
  description in competition.yaml ("on PlayStation 5 and Xbox
  Series X|S only (no PC until 2027)"): all parse True. A PC launch
  ("released … on PC, PS5, Xbox Series") parses False.

### Regime branch (bug fix)

v1 took the simultaneous branch whenever the word "simultaneous"
appeared anywhere in the notes. The notes for God of War
Ragnarök (practice case p02) say **"no simultaneous launch"** —
a rolling launch — and v1 trained on all 37 comparables for it
(including the 12 sharp-step ones). v2 strips negated phrases
("no simultaneous", "not simultaneous", "no single global",
"not a simultaneous", "never simultaneous") before testing for an
affirmed simultaneous unlock. The data-driven backstop is
unchanged: pre-day-peak / hour 1 < 0.10 with no rolling wording
⇒ simultaneous.

## 5. Fitted shape (for a sanity read)

For a mid-size hour 1 (~100k) the v2 median forecast dips to
~0.8x at hours 3–5, recovers to ~1.0x by hours 6–12, and peaks at
~2.0–2.3x in hours 19–21 for a console-only launch — the US
unlocks (05:00Z East, 08:00Z West) followed by evening prime time
in Europe/US. For a small hour 1 (~25k) the peak ratio is near
3.5x; for a very large hour 1 (~500k) it is near 1.0x.

## 6. Practice runs (sanity only — answers are not in the kit)

```
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

- All 4 practice cases produce valid output (23 rows, hours 2–24,
  5 finite non-decreasing quantiles each, > 0 and ≤ 1e9).
- v1 vs v2 median peak ratios (hour-1 print in brackets):
  | case | hour 1 | v1 peak | v2 peak |
  |---|---|---|---|
  | p00 Wolverine | 23,357 | 2.92× @ h20 | 3.54× @ h21 |
  | p01 Red Dead Redemption 2 | 116,550 | 1.64× @ h20 | 2.30× @ h21 |
  | p02 God of War Ragnarök | 269,517 | 0.89× @ h2 (flat/U) | 1.82× @ h19 |
  | p03 Spider-Man 2 | 119,087 | 1.64× @ h20 | 2.15× @ h20 |
  The v2 peaks now sit where the console-only comparables'
  nearest neighbours sit (Ghost of Tsushima 2.25×, Black Ops 6
  2.74×, Stellar Blade 3.40×, Ghost of Yōtei 3.62×), while v1's
  p02 forecast peaked at hour 2 — a shape no console-only
  comparable has.
- Runtime: ~0.19 s for all four cases together (~0.05 s per case),
  far under the 8 s per-case cap, on 1 CPU.
- Determinism: two runs write byte-identical files (`diff -r`
  clean). No randomness, no clock, no network; inputs are sorted.
- Shape checks (automated on every output): exactly one `hours`
  key, 23 rows, hours 2–24 each once, each row exactly
  `{"hour", "q"}` with five finite numbers, each > 0 and ≤ 1e9,
  non-decreasing.
- Edge cases tested with synthetic inputs: a single case file only;
  no comparables at all (falls back to the baseline, still valid);
  hour 1 = 100 and hour 1 = 2,000,000 with 50,000 channels
  (finite, non-decreasing, ≤ 1e9 quantiles — the log-channels
  clip holds); hour 1 with channels missing (falls back to the
  training median log-channels); notes rewritten to a simultaneous
  unlock (takes the all-comparables branch, wider spread); notes
  without a "released" sentence (parses as not console-only).

## 7. Why v1 scored 0.289 on practice — what this v2 can and cannot claim

v1's public leaderboard practice loss was 0.289 (typical miss
59.74%, coverage 0.804, rank 3261/3524), while its gradual LOO was
0.178. Two mechanisms explain most of the gap:

1. **The p02 regime bug (§4).** v1 forecast God of War Ragnarök
   with the simultaneous branch: a flat/U shape peaking at hour 2
   (0.89× hour 1) for a game whose console-only peers peak at
   1.3–2.7×. One of four practice cases scored with a
   fundamentally wrong shape is worth roughly 0.05–0.10 of mean
   practice loss on its own.
2. **The console-only shape gap (§1).** v1's pooled gradual fit
   under-predicts the evening rise of console-only launches
   (v1 peaks 1.35–2.92× vs console-only comparables' typical
   2.66×). v2's console indicator + log channels moves the
   practice peaks to 1.82–3.54×, in line with the console-only
   comparables.

What v2 **cannot** claim: a measured practice loss. The practice
answers are not in the kit (§2), so the proxy evaluation the v2
work was asked to build does not exist. The honest evidence for
v2 is the stratified LOO (§3): gradual 0.165 vs v1 0.178,
console-only 0.148 vs 0.166, all-launches 0.339 vs 0.343 — plus
the p02 bug fix, which is a correctness fix, not a tuning choice.
The practice gap is only partly explained by the LOO gains
(~0.013–0.018), so v2's true practice gain is uncertain and could
be smaller than hoped. The remaining gap, if any, would point at
something the comparables do not contain — e.g. the practice being
scored on SullyGnome hourly averages while the final is scored on
minute-mean Twitch reads, or GTA VI's scale being outside the
comparables' range.

## 8. Risks and known weaknesses

- **The practice-score gap is not fully explained.** See §7. v2
  is better on every LOO population that matters, but the LOO
  gain (−0.013 mean on gradual) is modest and not statistically
  strong (paired SD ~0.06, n = 25). Do not over-claim.
- **Multicollinearity.** console-only, log(p1) and log(channels)
  are correlated (r = 0.833 for log channels vs log(p1)), so the
  fitted coefficients are unstable (§4). The predictions are
  better in LOO, but a target with an unusual combination
  (console-only with very many channels, or multi-platform with
  very few) could be pushed the wrong way. The log-channels clip
  [1.8, 10.0] bounds the damage.
- **Worst LOO worsenings.** Death Stranding 2 (0.170 → 0.315),
  Black Ops 6 (0.260 → 0.351), Forza Horizon 6 (0.149 → 0.222),
  Starfield (0.284 → 0.346). These are launches whose evening
  rise is smaller than their console/multi-platform peers would
  suggest; the console boost overshoots them.
- **Notes-parser fragility.** The console-only flag and the regime
  branch both read the case notes. If the final case's notes phrase
  the platforms differently (e.g. "PS5 and Xbox Series X|S" with no
  "only" and no "no PC"), the parser falls through to the
  console-token rule and still returns True; if it named a PC
  launch window without a year, it would return False and the
  forecast would lose the console boost (the safe direction — it
  degrades to v1, not to nonsense). The simultaneous branch has
  the same exposure; the pre-day-peak ratio is its backstop.
- **Hour 1 outside the training range.** The largest gradual
  hour 1 in training is 875,155 (ELDEN RING). GTA VI could exceed
  that, and its channels could exceed the observed 21,700. The
  log-log fit extrapolates sub-linearly (slope ~0.65 late), which
  is the conservative direction, and the extrapolation check in
  §3 supports it — but a record-breaking launch is, by definition,
  the case the comparables say least about.
- **Hour 1 very low.** If hour 1 came in tiny (say < 1,000) the fit
  predicts a many-fold rise, as it should for a rolling launch whose
  early regions are small — but if the true cause were a delayed
  unlock or a recording gap, the forecast would overshoot badly.
  The clipped pre-day feature damps this only partly.
- **Missing comparables / sparse data.** If the final case shipped
  far fewer comparables, or with many missing hours, the fit degrades
  to the baseline fallback. The program requires ≥ 20 of 24 hours
  per curve and ≥ 10 gradual curves before trusting the gradual fit.
- **Practice ≠ final.** The practice is scored on SullyGnome hourly
  averages; the final on minute-mean Twitch reads. The house notes
  the two measures are close but not the same, and the final's
  history is minute reads (denser) where the practice's is hourly.
  The pre-day feature is computed from `history_hourly` in both,
  so the measure is consistent within each case.
- **Early access.** GTA VI has no early access (per the rules).
  Early-access-contaminated comparables (Diablo IV, Hogwarts
  Legacy, Starfield) are in the training set and are the model's
  worst cases; they pull the late-day slope slightly up. Excluding
  them on the "early access" text cue did not improve LOO, so they
  were kept.
- **Small validation set.** All selection above rests on 25 gradual
  launches (37 total). A single unusual launch moves the mean by
  ~0.004. The model is deliberately simple — five coefficients per
  hour, smoothed — to avoid fitting noise in that small sample.
- **v2 is submitted; v5 is not.** The submitted entry is v2 (repo
  BurningHead69/gta6-twitch-ccu-forecast, commit 4fb443a,
  submission 01a1e959-aa45-77e4-98d5-3609e23cb939, practice loss
  0.180, rank 1237/4861). This directory (v5) has no repo and no
  submission; see §10 before submitting from it.

## 10. v5: the `streams` adjustment (2026-11-17) — a bounded, unvalidated gamble

> **v5 verdict: v5 is v2 plus a small streams adjustment that cannot
> be validated on practice.** The user explicitly chose this gamble
> after v3 and v4 both failed to beat v2. When `streams` is absent
> (every practice case, and every LOO pseudo-case) v5 is
> byte-identical to v2: LOO gradual 0.1653 / median 0.1258,
> console-only 0.1481 / 0.1164, all 0.3386, sharp 0.6995, high-p1
> 0.1426 — reproduced exactly with the v4 LOO harness pointed at
> this directory. The streams effect itself has NO measured
> validation anywhere: it could help or hurt the final. Do not
> present v5 as "better than v2" — present it as "v2 plus an
> unvalidated bet".

### 10.1 Schema audit — what `streams` actually is

Documented identically in RULES.txt ("Your input"), competition.yaml
§"The case", the starter's docstring, and the house-forecast source
in the kit's harness tarball (`grader/house-forecast-twitch-ccu.ts`,
`streams: Array<{login: string; viewers: number; language: string |
null; started_at: string | null}> | null`):

- The category's **top 20 live streams at the last valid read
  before the checkpoint** (end of hour 1), each with exactly four
  fields: `login` (string), `viewers` (number), `language`
  (string or null), `started_at` (ISO 8601 string or null).
- **Null in every practice case** (verified in p00–p03).
- No follower counts, no display names, no game/title, no
  per-stream channel data. `language` and `started_at` may be null
  per stream.
- The case-builder itself is not in the kit's harness tarball (that
  tarball is the platform source, not the recorder); the four-field
  schema above is the documented guarantee, and the code parses
  defensively against anything else.

### 10.2 Hunt for validation data in the kit — none exists

- Comparables carry only aggregate hourly `{viewers, channels}` —
  no per-stream or channel-distribution data at any hour. The
  channel count at hour 1 is already a v2 feature (log-channels);
  nothing in the kit says how hour-1 viewers were distributed
  across streams, so the concentration hypothesis cannot be tested
  on comparables.
- `history` / `history_hourly` are aggregate too. The GTA VI
  category's own pre-release comparable is aggregate. `levels`,
  `checkpoint`, `day_start`, `forecast` carry no stream data.
- `categories_hourly` is also null in practice and is not used.
- Conclusion: there is no test in the kit that can turn this gamble
  into evidence. The only checks possible are the ones in §10.4:
  LOO equality when streams are absent, and synthetic direction /
  cap / fallback tests.

### 10.3 The adjustment (features, signs, caps)

Applied on top of the fitted v2 median `mu` and spread `sd`, per
hour, only when `streams` is a list with >= 5 entries whose
`viewers` is a finite positive number (other entries ignored;
`started_at` must parse as ISO 8601 to count for age):

| feature | mechanism and sign | formula | cap / window |
|---|---|---|---|
| Viewer-weighted mean stream age at the checkpoint | At 01:00Z the live top streams are the early wave: NZ/AU/Asia streamers 10–14 h into their day, EU/UK 1–2 h in, US not yet unlocked (05:00Z/08:00Z). High weighted age = the top of the category is still the Asia-Pacific wave, near the end of its day, winding down before the US arrives → hours 2–5 sag. Low age = the EU/UK wave dominates, its streamers just started launch sessions → hours 2–5 hold up. | shift = −0.02 × (age − 5 h), age clipped to [0, 16] h | \|shift\| ≤ 0.10 log units; full weight hours 2–5, fading to 0 at hour 9 |
| Herfindahl concentration of top-20 viewer shares | Concentration is idiosyncratic risk, not level: one big streamer ending moves the category total; a diffuse audience self-averages. Touches only the spread. | spread mult = 1 + 0.5 × (HHI − 0.10), HHI = Σ share² | mult clipped to [0.90, 1.20]; full weight hours 2–8, fading to 1 at hour 12 |

Deliberately unused: `language` — no clean sign (a high English
share at 01:00Z is UK/NZ/AU, not the US evening wave; a high
Spanish share could be LATAM, whose midnight is 03:00–05:00Z, or
Spain at 23:00Z — the sign flips by region, so it is not used).
Also deliberately not used: top-20 coverage (Σ top-20 / p1) — it
double-counts concentration, and the last-minute snapshot vs the
hour-mean print makes it noisy.

Failure modes guarded against: streams present with a different
schema (unknown fields ignored; `viewers` must be a finite positive
number; booleans rejected; `started_at` unparseable → that stream
drops out of the age average, HHI still computed); a single huge
stream dominating hour 1 (HHI drives the spread to at most ×1.20,
median untouched by concentration); streams started days earlier
(24/7 or pre-launch waiting streams — age clipped at 16 h);
`checkpoint` missing or unparseable (age neutral, HHI still
computed); fewer than 5 usable entries (whole adjustment off).
When the regression itself cannot be fitted, v5 falls back to the
house baseline exactly as v2 does (no streams adjustment there).

Known counter-reading (documented, not resolved): a stream that
has already lasted 10+ h is a survivor — marathon launch streamers
may be *more* likely to keep going than a fresh stream. The
region-mix reading above is the basis for the sign, and the 0.10
cap is what bounds the damage if the counter-reading is right.

### 10.4 Validation performed

- **LOO, streams absent:** gradual 0.1653 / 0.1258 (n=25),
  console-only 0.1481 / 0.1164 (n=8), all 0.3386 / 0.2060 (n=37),
  sharp 0.6995 (n=12), high-p1 0.1426 — identical to v2.
- **Practice outputs byte-identical to v2** (streams None in
  p00–p03); deterministic byte-identical reruns; ~0.19 s for all
  four cases (~0.05 s/case).
- **Synthetic tests** (p00 copied, streams attached; v5 output
  compared with v2 on the same files; median shift = log ratio of
  medians, spread mult = ratio of log(q95/q05) widths):

| scenario | median shift h2 / h5 / h8 / h9 | spread mult h2 / h8 / h12 | verdict |
|---|---|---|---|
| young EU-heavy (ages 0.2–1 h, near-equal viewers) | +0.090 / +0.090 / +0.022 / 0 | 0.978 / 0.978 / 1.000 | intended direction, within caps |
| old APAC-heavy (ages 10–14 h) | −0.100 / −0.100 / −0.025 / 0 | 0.978 / 0.978 / 1.000 | capped at −0.10 as designed |
| concentrated (one stream 60% of top-20) | +0.080 / +0.080 / +0.020 / 0 | 1.141 / 1.141 / 1.000 | spread widens, median from age only |
| diffuse (20 equal streams) | +0.060 / +0.060 / +0.015 / 0 | 0.975 / 0.975 / 1.000 | narrower spread as designed |
| extreme (one stream 95%, age 16 h) | −0.100 / −0.100 / −0.025 / 0 | 1.200 / 1.200 / 1.000 | both caps bind, output still valid |
| streams None / [] / "junk" / no-viewers rows / only 3 entries | 0 / 0 / 0 / 0 | 1.000 everywhere | byte-identical to v2 |
| started_at garbage / null for all | median 0 | 0.975 | age neutral, HHI still applied |
| checkpoint key deleted | median 0 | 0.978 | age neutral, HHI still applied |

All synthetic outputs valid (23 rows, hours 2–24, 5 finite
non-decreasing quantiles each, > 0, ≤ 1e9). No synthetic case
exceeded the ±0.10 median cap or the [0.90, 1.20] spread range.
