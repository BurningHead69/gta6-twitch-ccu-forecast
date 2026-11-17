# Notes — model, validation, and risks (v6; v2's notes carried over, plus §9)

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
- **v7 is not yet submitted.** This directory has no repo and no
  submission; the current final entry on make.fun is v5 (repo
  BurningHead69/gta6-twitch-ccu-forecast, commit f37bc5c7,
  submission 01a1e9de-8475-761c-b635-9078dfc3f1e1). v6 was never
  submitted. See §10 before submitting from this directory.

## 9. v6 investigation (2026-11-17) — the kitchen-sink combination, adopted for the practice push

> **v6 verdict: the kitchen-sink combination is adopted, with the
> EA detection switched from tier label to text mention.** The
> combination was first measured by v3/v4 and never shipped; v6
> implements it in `main.py` and reproduces its LOO numbers
> exactly with a fresh harness (`~/workspace/makefun/v6-exp/loo6.py`,
> which imports the shipped `forecast_case` and builds synthetic
> pseudo-cases from the comparables in `p00.in`, one held out at
> a time, notes in case-note style, regime from the curve's own
> step — the same harness design as v3/v4). The user explicitly
> asked for another push toward the top 1000 of the practice
> leaderboard (rank 1000 needs practice loss ~0.167; v2/v5 sit
> at 0.180, rank ~1417) and was told the gain is modest and
> uncertain. **This directory is not yet submitted**; the
> submitted entry on make.fun is v5 (streams gamble, commit
> f37bc5c7, submission 01a1e9de-8475-761c-b635-9078dfc3f1e1),
> before that v2 (commit 4fb443a, submission
> 01a1e959-aa45-77e4-98d5-3609e23cb939, practice loss 0.180).
> v6 contains **no** streams layer: streams are null in
> practice, and v6 is aimed at the practice leaderboard.

### The combination, exactly as implemented

Per hour, weighted least squares with a ridge penalty, on the
same five v2 features (intercept, log hour-1, clipped
log(pre24max / hour 1), console-only indicator, clipped log
channels at hour 1), coefficients smoothed 0.25/0.5/0.25 across
hours, spread 1.15x residual SD (floor 0.18, cap 0.90), same
regime split, negation fix, and feature-drop fallback as v2:

1. **Early-access down-weighting.** Training curves flagged as
   early-access are weighted 0.5 in the per-hour fit (all other
   curves 1.0). Two detectors were tested on the comparables'
   `unlock` text:
   - **text** (shipped): any affirmed "early access" /
     "early-access" mention, negations ("no early access",
     "not early access", "never early access", "no
     early-access", "without early access") stripped first.
     Flags 11 of 38 comparables: Hogwarts Legacy, Call of Duty:
     Black Ops 6, Call of Duty: Modern Warfare III, 007 First
     Light, Forza Horizon 6, Starfield, Diablo IV, Assassin's
     Creed Shadows, DOOM: The Dark Ages, Death Stranding 2: On
     the Beach, Forza Horizon 5.
   - **tier**: only the literal "Tier: early-access-contaminated"
     label. Flags 9 (the text set minus Black Ops 6 and Modern
     Warfare III).
2. **Ridge λ = 0.5**, added to the diagonal of the weighted
   normal equations for every coefficient except the intercept
   (raw, unstandardised — exactly as in the v4 experiment code
   that produced the reference numbers).
3. **Spread modulation × exp(−2 · (target pre_slope48 − mean
   train pre_slope48))**, applied after the spread floor/cap
   with no re-clipping. pre_slope48 is the OLS slope of log
   viewers on hour index over the 48 hours before day_start
   (≥ 6 usable hourly rows, else 0.0 for a training curve and
   modulation skipped for a target with no usable rows). The
   training mean is over the same regime-filtered training set
   the fit used. Residual SDs are computed from *unweighted*
   residuals of the smoothed fit, as in v4.

### Reproduced LOO numbers (fresh harness, grader scale)

Baseline v2 reproduced exactly first: gradual **0.1653 /
0.1258**, console-only **0.1481 / 0.1164**, all **0.3386 /
0.2060**, sharp 0.6995; high-hour-1 test (train on gradual
hour-1 ≤ 200k, test the 6 gradual launches with hour-1 >
200k) **0.1426** — the harness is faithful.

| variant (all: + ridge 0.5 + spread −2 unless noted) | gradual mean / median | console-only mean / median | all mean / median | sharp | high-p1 |
|---|---|---|---|---|---|
| v2 | 0.1653 / 0.1258 | 0.1481 / 0.1164 | 0.3386 / 0.2060 | 0.6995 | 0.1426 |
| **EA tier w0.5 (the v4 reference combo)** | **0.1598 / 0.1332** | **0.1437 / 0.1146** | **0.3304 / 0.1902** | 0.6856 | **0.1334** |
| **EA text w0.5 (shipped v6)** | **0.1585 / 0.1258** | **0.1435 / 0.1171** | **0.3302 / 0.1997** | 0.6877 | **0.1375** |
| EA tier w0.7 | 0.1602 / 0.1251 | 0.1443 / 0.1141 | 0.3310 / 0.1967 | 0.6868 | 0.1350 |
| EA text w0.7 | 0.1596 / 0.1244 | 0.1443 / 0.1151 | 0.3310 / 0.2027 | 0.6880 | — |
| EA tier w0.5 + ridge 1.0 | 0.1631 / 0.1319 | 0.1494 / 0.1172 | 0.3309 / 0.1975 | 0.6806 | — |
| EA tier w0.5 + spread −1.5 | 0.1597 / 0.1328 | 0.1441 / 0.1150 | 0.3306 / 0.1896 | 0.6866 | — |
| EA tier w0.5 + spread −2.5 | 0.1602 / 0.1335 | 0.1434 / 0.1142 | 0.3303 / 0.1916 | 0.6847 | — |
| EA tier w0.5, no spread | 0.1605 / 0.1322 | 0.1461 / 0.1164 | 0.3325 / 0.1942 | 0.6908 | — |
| EA tier w0.5 only (no ridge, no spread) | 0.1623 / 0.1273 | 0.1473 / 0.1227 | 0.3363 / 0.1857 | 0.6986 | — |

The tier combination reproduces v4's numbers **exactly**
(0.1598 / 0.1332, 0.1437 / 0.1146, 0.3304 / 0.1902, high-p1
0.1334, paired t −0.92, improved 14/25) — the shipped
implementation is a faithful port of the experiment code, not
a re-derivation.

### Why text was shipped instead of tier

- Gradual mean: text **0.1585** vs tier 0.1598 (v2: 0.1653).
- Gradual median: text **0.1258, equal to v2's**; tier 0.1332
  is worse than v2's. Text is the only variant tested whose
  median does not collapse (the w0.7 variants are 0.1244–0.1251
  but their means are worse, 0.1596–0.1602).
- Paired gradual vs v2: text mean diff **−0.0068, SE 0.0055,
  t = −1.24**, improved 15/25, worsened 10/25; tier −0.0055,
  t = −0.92, 14/25. Console-only paired: text −0.0046
  (improved 4/8), tier −0.0045 (4/8).
- All-launches mean: text 0.3302 (best tested), tier 0.3304.
- High-hour-1 test: text 0.1375 vs tier 0.1334 vs v2 0.1426 —
  tier is better here, but both beat v2, and text wins the
  other three populations. The selection criterion was LOO mean
  *and* high-hour-1 with the median intact; text is the best
  overall compromise, tier the best on high-hour-1 alone.
- Coverage and typical miss (gradual; typical miss = median
  |median forecast − truth| / truth over all gradual
  case-hours): v2 0.847 / 19.6% → v6-text **0.875 / 19.1%**.
  (v2's published typical miss of 22.4% used a slightly
  different aggregation in the v2 experiment code; the
  v2→v6 comparison here is computed with identical code, so
  the delta is the meaningful part.)

### Per-case LOO, v6-text vs v2 (gradual launches)

| launch | v2 | v6-text | diff |
|---|---|---|---|
| Marvel Rivals | 0.222 | 0.134 | −0.088 |
| Death Stranding 2 | 0.315 | 0.257 | −0.058 |
| Forza Horizon 6 | 0.222 | 0.173 | −0.049 |
| Starfield | 0.346 | 0.305 | −0.041 |
| Call of Duty: Black Ops 7 | 0.112 | 0.094 | −0.018 |
| Hogwarts Legacy | 0.170 | 0.153 | −0.017 |
| The Last of Us Part II | 0.084 | 0.069 | −0.015 |
| Call of Duty: Black Ops 6 | 0.351 | 0.337 | −0.014 |
| Cyberpunk 2077 | 0.118 | 0.110 | −0.008 |
| Diablo IV | 0.206 | 0.200 | −0.006 |
| Ghost of Yōtei | 0.126 | 0.121 | −0.004 |
| Elden Ring Nightreign | 0.088 | 0.084 | −0.004 |
| Call of Duty: Modern Warfare 4 | 0.125 | 0.122 | −0.003 |
| Stellar Blade | 0.110 | 0.107 | −0.003 |
| ELDEN RING | 0.084 | 0.083 | −0.001 |
| Saros | 0.123 | 0.126 | +0.003 |
| Ghost of Tsushima | 0.090 | 0.095 | +0.005 |
| Call of Duty: Modern Warfare III | 0.100 | 0.106 | +0.007 |
| FINAL FANTASY VII REBIRTH | 0.105 | 0.113 | +0.007 |
| 007 First Light | 0.152 | 0.167 | +0.015 |
| Assassin's Creed Shadows | 0.110 | 0.125 | +0.015 |
| DOOM: The Dark Ages | 0.144 | 0.167 | +0.023 |
| Borderlands 4 | 0.204 | 0.227 | +0.024 |
| Zelda: Tears of the Kingdom | 0.232 | 0.260 | +0.028 |
| Forza Horizon 5 | 0.195 | 0.227 | +0.032 |

The gains concentrate in v2's worst, early-access-contaminated
cases (Marvel Rivals, Death Stranding 2, Forza Horizon 6,
Starfield) — the mechanism the down-weighting targets — and
the worst losses are small (+0.032 at most) and spread across
mid-tier launches. No single case dominates the mean gain the
way Death Stranding 2 did for the v3 EA indicator.

### Honest warning — read before submitting

- The gain is modest and uncertain: gradual mean −0.0068 with
  SE 0.0055 (t = −1.24) on n = 25. That is better than v3/v4's
  best combos (t ≈ −1.0) and the median is intact, but it is
  still within the range where a different 25 launches could
  flip the sign. The practice improvement is **not
  guaranteed**: v6's practice loss is expected to be near
  0.180 − 0.007 ≈ 0.173 if LOO transfers one-to-one, which
  would put it around rank ~1000 — but LOO-to-practice
  transfer has already surprised us once (v1: LOO 0.178,
  practice 0.289), and the practice set is only 4 launches.
- The ridge (λ = 0.5, raw, unstandardised) is nearly
  negligible next to X′X entries of ~10³; its contribution is
  ~0.001–0.002 of the gain. The spread modulation is worth
  ~0.001–0.002 (no-spread: 0.1605 vs 0.1598 tier). The bulk of
  the gain is the EA down-weighting.
- The EA text detector depends on the case authors' research
  notes in the comparables' `unlock` field. If the final case's
  comparables phrase early access differently, fewer or more
  curves get down-weighted; the effect is bounded (weights
  0.5–1.0) and the model degrades toward v2, not toward
  nonsense.
- v6 does not include v5's streams layer. If v6 is submitted
  it supersedes v5, and the final case loses the streams
  adjustment (which was itself an unvalidated gamble). That
  trade-off — a measured practice gain vs an unvalidated final
  gamble — is the user's call, not the model's.

## 10. v7: the merge — v6's model plus v5's streams layer (2026-11-17)

> **v7 verdict: v7 is v6 plus v5's streams layer, nothing else.**
> With streams absent (every practice case, every LOO
> pseudo-case) v7 is byte-identical to v6: LOO gradual
> 0.1585 / 0.1258, console-only 0.1435 / 0.1171, all 0.3302 /
> 0.1997, sharp 0.6877, high-hour-1 0.1375 — reproduced with
> the v6 harness (`~/workspace/makefun/v6-exp/loo6.py`) pointed
> at v7's shipped `forecast_case`, per-case loss difference
> vs v6 exactly 0.000000 on all 37 launches. The streams
> effect itself is unvalidated (see v5's §10, carried over
> below in outline); v7's practice loss is expected to equal
> v6's, because streams are null in practice. **This
> directory is not yet submitted**; the current final entry
> on make.fun is v5 (streams gamble, commit f37bc5c7,
> submission 01a1e9de-8475-761c-b635-9078dfc3f1e1). v6 was
> never submitted.

### What was merged, exactly

`main.py` is v6's file with three changes, and no change to
any fitting code:

1. v5's streams block copied verbatim: the `STREAMS_*`
   constants and `streams_adjustment(case)` (viewer-weighted
   mean stream age → median shift, max ±0.10 log units,
   full weight hours 2–5 fading to 0 at hour 9; Herfindahl
   concentration of the top-20 shares → spread multiplier
   1 + 0.5 × (HHI − 0.10) clipped to [0.90, 1.20], full
   weight hours 2–8 fading to 1 at hour 12; `language`
   unused; {} when streams are None, empty, malformed, or
   have fewer than 5 usable entries).
2. The hour loop computes `streams_adj` once and applies it
   **last**, after v6's spread modulation, exactly as v5
   applied it after v2's spread: `mu += shift`, then the
   spread step below. The baseline fallback is untouched
   (no streams adjustment there, as in v5).
3. The docstring gains a v7 section; nothing else in the
   model description changed.

### The one interaction: the spread, and the clipping rule

v6's spread multiplier is unbounded (`spread_mult =
exp(−2 · (target pre_slope48 − mean train pre_slope48))`,
applied after the floor/cap with no re-clipping); v5's
streams multiplier is in [0.90, 1.20]. The two multiply,
so their product is unbounded in principle. The merge
bounds it as follows, per hour:

- Let `sd_base` be v2's base spread: the floored/capped
  `1.15 × residual SD` *before* v6's modulation.
- v6 alone: `sd = sd_base × spread_mult`.
- v7, when the streams spread layer is active at that hour
  (`smult != 1.0`): `sd = sd_base × clip(spread_mult ×
  smult, 0.90, 1.20)`. The FINAL multiplier relative to
  `sd_base` is therefore always in [0.90, 1.20] wherever
  the streams layer touches the spread.
- v7, when streams are absent, or `smult == 1.0` at that
  hour (hours ≥ 12, or HHI exactly 0.10): `sd` is v6's
  exactly. The clip is a property of the streams layer, so
  it only engages where that layer is active — this is
  what makes v7 byte-identical to v6 with streams absent
  (verification 1) and LOO-identical (verification 3).
  v5's absolute re-clip to [0.18, 0.90] is *not* applied
  in v7; it is replaced by this relative clip, so in
  principle the final sd can reach 0.162–1.08 (in practice
  `sd_base` is ~0.21–0.42, so the point is moot).

Worked example (p00, measured): v6's `spread_mult` is
0.814 (the target's pre-48h slope is above the training
mean, so v6 narrows the spread). The streams multiplier
for a diffuse top-20 is 0.975, so the unclipped product
would be 0.794 — below 0.90 — and the clip binds:
final multiplier 0.900, i.e. v7's spread is 1.106× v6's
in hours 2–8. For a concentrated top-20 (smult 1.134)
the product is 0.923, inside the range, no clipping, and
v7's spread ratio vs v6 is 1.134 — exactly v5's. For the
extreme case (smult 1.20) the product is 0.977, ratio
1.200, again exactly v5's. Consequence worth stating
plainly: because all four practice cases have v6
`spread_mult` below 0.90 (p00 0.814, p01 0.872, p02 0.845,
p03 0.939), the streams layer can only *widen* v7's
spread relative to v6 when it is active — it can never
narrow it below 0.90× base. That is the clipping rule
doing its job, not a bug: the combined multiplier is
bounded on both sides.

### Verification performed

**1. Streams None: v7 byte-identical to v6.** `main.py`
run on all 4 kit practice inputs for both directories;
`diff -r` clean on all four `.out` files. Determinism:
a second v7 run writes byte-identical files.

**2. Synthetic streams tests** (p00 copied, streams
attached; v7 compared with v6 on the same files; v5
compared with v2 on the same files, for the "same
direction and caps" check; median shift = log ratio of
medians, spread ratio = ratio of log(q95/q05) widths,
final multiplier = v7 effective sd / `sd_base`, where
`sd_base` is measured by running v7 with `SPREAD_A = 0`
and streams None):

| scenario | v7 median shift h2 / h5 / h8 / h9 | v5 median shift (same) | v7 spread ratio vs v6 h2 / h8 / h12 | v5 spread ratio vs v2 | v7 final mult vs base h2 / h5 / h8 / h11 |
|---|---|---|---|---|---|
| young EU-heavy (ages 0.2–1 h, near-equal) | +0.091 / +0.091 / +0.023 / 0 | +0.091 / +0.091 / +0.023 / 0 | 1.106 / 1.106 / 1.000 | 0.976 / 0.976 / 1.000 | 0.900 / 0.900 / 0.900 / 0.900 (clip binds low) |
| old APAC-heavy (ages 10–14 h) | −0.100 / −0.100 / −0.025 / 0 | −0.100 / −0.100 / −0.025 / 0 | 1.106 / 1.106 / 1.000 | 0.976 / 0.976 / 1.000 | 0.900 (clip binds low) |
| concentrated (one stream 60%, age 1 h) | +0.080 / +0.080 / +0.020 / 0 | +0.080 / +0.080 / +0.020 / 0 | 1.134 / 1.134 / 1.000 | 1.134 / 1.134 / 1.000 | 0.923 / 0.923 / 0.923 / 0.900 |
| diffuse (20 equal, age 2 h) | +0.060 / +0.060 / +0.015 / 0 | +0.060 / +0.060 / +0.015 / 0 | 1.106 / 1.106 / 1.000 | 0.975 / 0.975 / 1.000 | 0.900 (clip binds low) |
| extreme (one stream 95%, age 16 h) | −0.100 / −0.100 / −0.025 / 0 | −0.100 / −0.100 / −0.025 / 0 | 1.200 / 1.200 / 1.000 | 1.200 / 1.200 / 1.000 | 0.977 / 0.977 / 0.977 / 0.900 |
| streams None / [] / "junk" / no-viewers rows / only 3 entries | 0 | 0 | 1.000 everywhere | 1.000 | byte-identical to v6 |
| started_at garbage / null for all | median 0 | median 0 | HHI still applied (spread differs, as in v5) | same | — |
| checkpoint key deleted | median 0 | median 0 | 1.106 (HHI applied, clip binds) | 0.978 | — |

Median shifts match v5's **exactly** in every scenario
(the +0.091 vs the +0.090 in v5's §10 table is quantile
rounding to 2 decimals; both v5 and v7 measure +0.091 on
these payloads). Spread ratios vs base match v5's
exactly wherever the combined clip does not bind
(concentrated, extreme, and hour 12 everywhere); where
it binds (young, old, diffuse on p00) the ratio is 1.106
instead of ~0.976 — the final multiplier is 0.900
instead of ~0.794, which is the clipping rule working
as specified, and the final multiplier is within
[0.90, 1.20] in every hour where the streams layer is
active (measured final multipliers equal the exact
clipped products to < 0.01). No synthetic case exceeded
the ±0.10 median cap; all synthetic outputs valid.

**3. LOO with streams absent = v6's published numbers.**
Fresh harness (`~/workspace/makefun/v6-exp/loo6.py`
imported, pointed at v7's shipped `forecast_case`):

| model | gradual mean / median | console-only mean / median | all mean / median | sharp | high-p1 |
|---|---|---|---|---|---|
| v6 reproduced | 0.1585 / 0.1258 | 0.1435 / 0.1171 | 0.3302 / 0.1997 | 0.6877 | 0.1375 |
| **v7** | **0.1585 / 0.1258** | **0.1435 / 0.1171** | **0.3302 / 0.1997** | **0.6877** | **0.1375** |

Per-case loss difference v7 − v6: max 0.000000 across
all 37 launches. (The harness first reproduced v6's
published numbers exactly, so the comparison is against
a faithful baseline, not a stale table.)

**4. Validity, determinism, runtime.** All 4 practice
outputs valid (23 rows, hours 2–24, 5 finite
non-decreasing quantiles each, > 0 and ≤ 1e9);
byte-identical reruns; ~0.21 s for all four cases
(~0.05 s/case), far under the 8 s per-case cap.
Stdlib only, deterministic, offline, `main.py` at repo
root.

### Risks (v7-specific, on top of §8 and §9)

- **The streams gamble is unchanged by the merge.** The
  streams layer is still unvalidated (null in practice,
  no validation data in the kit — see v5 §10.2); v7's
  practice loss is expected to equal v6's, and the
  streams effect only shows up on the final case.
  Present v7 as "v6 plus an unvalidated bet", never as
  "better than v6".
- **The clip can widen the spread vs v6.** When v6's
  `spread_mult` is below 0.90 (all four practice cases),
  the combined clip raises v7's spread above v6's by up
  to ~10.6% in hours 2–8 whenever streams are present
  and diffuse/young/old. That is the specified bound
  working as designed, but it means v7's final-case
  spread is never narrower than 0.90× base while the
  streams layer is active — v6's narrowing is partly
  undone. If the final case's `spread_mult` is above
  1.20 the clip binds the other way instead.
- **The clip engages only where `smult != 1.0`.** At
  hours ≥ 12 (spread faded out) and when HHI is exactly
  0.10, the spread is v6's exactly, including v6's
  unbounded modulation. That is deliberate: the clip
  bounds the *interaction*, not v6.
- **Defensive parsing is inherited from v5 unchanged.**
  A streams payload with an unexpected schema is ignored
  field by field (unknown fields, non-numeric viewers,
  unparseable `started_at`); fewer than 5 usable entries
  turns the whole layer off. If the final case ships a
  streams list whose `viewers` are the *hour-mean* print
  rather than the last-read snapshot, the age/concentration
  reading is still the documented one — the code cannot
  tell the difference, and the caps bound the damage.
