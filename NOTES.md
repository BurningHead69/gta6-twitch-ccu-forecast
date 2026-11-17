# Notes — model, validation, and risks

Entry for make.fun `gta6-twitch-ccu-forecast-01`. The program is
`main.py` (Python 3.13, stdlib only). This file records how the model
was chosen, with the numbers, so the choice can be checked rather
than taken on trust.

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

## 2. Validation method

**Leave-one-out cross-validation on the comparables.** Each
comparable launch in turn is treated as a pseudo-case: its hour-1
print is taken as known, hours 2–24 are predicted from the *other*
comparables only, and the prediction is scored with the contest's
pinball loss on the log scale (mean over hours 2–24 × 5 quantiles),
exactly as the grader scores. This is the only honest local check
available: **the practice answers are not in the kit**, so no
practice loss for this entry can be computed locally. The house
baseline's published practice loss is ~0.387 (typical miss ~82.8%,
90% coverage ~0.783).

Two LOO populations are reported, because they answer different
questions:

- **Gradual launches (n = 25)** — the population GTA VI belongs to.
  This is the population the model was selected on.
- **All launches (n = 37)** — reported for transparency; it mixes in
  the simultaneous regime, which the chosen model deliberately does
  not train on for a rolling target.

With n = 25, differences in mean pinball below ~0.01 are within
noise (per-case losses vary by ~0.1). The model was therefore picked
on a combination of the LOO numbers *and* principle (smooth
coefficients, clipped features, no over-fitting to one launch), not
on the smallest decimal.

## 3. Models tried (LOO pinball loss, mean / median)

### All launches (n = 37)

| model | mean | median |
|---|---|---|
| House baseline (starter: persistence, σ = 0.15·√(h−1)) | 0.930 | 0.325 |
| Pooled empirical quantiles of log-ratio per hour | 0.714 | 0.408 |
| Pooled normal (mean/SD of log-ratio per hour) | 0.776 | 0.558 |
| Log-log regression on log(hour 1) | 0.439 | 0.350 |
| Log-log regression, quadratic in log(hour 1) | 0.401 | 0.265 |
| k-nearest neighbours in log(hour 1), k = 8 | 0.418 | 0.283 |
| Kernel-weighted quantiles, bandwidth 0.5 (log10) | 0.402 | 0.251 |
| 5 bins by log(hour 1), empirical quantiles | 0.416 | 0.270 |

No pooled model is competitive: the two regimes must be separated
first.

### Gradual launches only (n = 25) — the relevant population

| model | mean | median |
|---|---|---|
| House baseline | 0.278 | 0.253 |
| Pooled empirical quantiles of log-ratio per hour | 0.232 | 0.211 |
| Log-log regression on log(hour 1) | 0.183 | 0.167 |
| Log-log regression, quadratic | 0.184 | 0.176 |
| kNN k = 8 | 0.218 | 0.217 |
| Kernel-weighted quantiles, bw 0.5 | 0.203 | 0.183 |
| 5 bins, empirical quantiles | 0.245 | 0.206 |
| Regression, empirical residual quantiles | 0.187 | 0.168 |
| Regression trained on **all** comparables (no regime split) | 0.325 | 0.293 |
| Regression trained on all, step-weighted (1 / 0.3 / 0.1) | 0.235 | 0.235 |

The regime split is the single biggest win: training the regression
on gradual launches only (0.183) beats training it on all launches
(0.325) on gradual pseudo-cases. Soft step-weighting (0.235) is
worse than the hard split, so the hard split was kept.

### Refinements of the gradual regression (n = 25)

| variant | mean | median | 90% coverage | typical miss |
|---|---|---|---|---|
| Regression only | 0.1830 | 0.1672 | 0.852 | 35.3% |
| + pre-day feature log(pre24max / hour 1), clipped | 0.1800 | 0.1531 | 0.830 | 33.8% |
| + smoothing coefficients across hours | 0.1793 | 0.1492 | 0.830 | 33.6% |
| + spread × 1.1 | 0.1782 | 0.1513 | 0.866 | 33.6% |
| **Chosen: + spread × 1.15** | **0.1781** | **0.1524** | **0.890** | **33.6%** |
| + spread × 1.2 | 0.1783 | 0.1536 | 0.901 | 33.6% |
| Empirical residual quantiles instead of normal | 0.1820 | 0.1516 | 0.802 | 33.2% |
| Blend median 30% toward kernel-kNN | 0.1861 | 0.1890 | — | — |
| Train on step < 1.1 (adds RE Requiem, Monster Hunter Wilds) | 0.1932 | 0.1539 | 0.960 | 34.5% |

The pre-day feature, smoothing, and spread multiplier each help by
only ~0.001–0.003, but they help consistently (including in the
extrapolation check below) and are principled, so all three were
kept. Spread ×1.15 was chosen over ×1.2: identical pinball
(0.1781 vs 0.1783) with coverage 0.89, closest to the honest 0.90
without over-widening. Channels at hour 1 and viewers-per-channel
were tried as extra predictors and dropped: their correlation with
the regression residuals is |r| < 0.22 and they did not improve LOO.

### Per-case LOO loss, chosen model vs baseline (gradual launches)

| launch | hour 1 | chosen | baseline |
|---|---|---|---|
| Saros | 6,400 | 0.152 | 0.601 |
| Forza Horizon 6 | 26,312 | 0.149 | 0.410 |
| Stellar Blade | 27,057 | 0.128 | 0.372 |
| Ghost of Yōtei | 27,206 | 0.178 | 0.473 |
| Death Stranding 2 | 42,221 | 0.170 | 0.176 |
| Forza Horizon 5 | 43,895 | 0.266 | 0.325 |
| DOOM: The Dark Ages | 50,351 | 0.259 | 0.256 |
| Final Fantasy VII Rebirth | 53,526 | 0.111 | 0.173 |
| Call of Duty: Modern Warfare 4 | 59,162 | 0.115 | 0.236 |
| Call of Duty: Black Ops 7 | 79,207 | 0.096 | 0.173 |
| Call of Duty: Modern Warfare III | 114,417 | 0.135 | 0.152 |
| 007 First Light | 127,800 | 0.244 | 0.348 |
| Zelda: Tears of the Kingdom | 129,948 | 0.323 | 0.309 |
| Ghost of Tsushima | 131,910 | 0.147 | 0.228 |
| Borderlands 4 | 133,568 | 0.253 | 0.253 |
| Call of Duty: Black Ops 6 | 139,097 | 0.260 | 0.262 |
| Assassin's Creed Shadows | 146,993 | 0.240 | 0.345 |
| Starfield | 156,828 | 0.284 | 0.291 |
| Marvel Rivals | 165,939 | 0.182 | 0.210 |
| Elden Ring Nightreign | 319,909 | 0.113 | 0.252 |
| The Last of Us Part II | 365,019 | 0.115 | 0.247 |
| Hogwarts Legacy | 387,358 | 0.119 | 0.288 |
| Diablo IV | 533,455 | 0.196 | 0.125 |
| Cyberpunk 2077 | 851,999 | 0.124 | 0.230 |
| ELDEN RING | 875,155 | 0.097 | 0.211 |
| **Mean** | | **0.178** | **0.278** |

The chosen model beats the baseline on 21 of 25 gradual launches.
The three losses are early-access-contaminated launches (Diablo IV,
Zelda) where hour 1 was already at peak and the regression still
predicts a rise, and DOOM (a tie).

### If the target were simultaneous instead

On the 12 sharp-step pseudo-cases, the all-comparables regression
scores 0.657 vs the baseline's 2.287 — so the program's simultaneous
branch (train on all comparables) is far better than persistence
there too, even though it is not the branch GTA VI is expected to
take.

### Extrapolation check (GTA VI may exceed every comparable)

Trained only on gradual launches with hour 1 ≤ 200,000 and tested
on the 6 gradual launches with hour 1 > 200,000: regression only
0.196, regression + pre-day feature + smoothing 0.151. The pre-day
feature is what carries the extrapolation — it is kept, clipped, so
an out-of-range value cannot dominate.

## 4. Fitted shape (for a sanity read)

Per-hour regression on the 25 gradual launches (log viewers on
log hour-1): slope falls from 1.06 (hour 2) through 0.94 (hour 6),
0.82 (hour 9), 0.69 (hour 18) to 0.65 (hours 20–22); residual SD
is 0.19 at hour 2, ~0.33–0.37 from hour 6 on. In ratio terms, for a
mid-size hour 1 (~100k) the median forecast dips to ~0.8x at hours
3–5, recovers to ~1.0x by hours 6–12, and peaks at ~1.6–1.9x in
hours 19–21 — the US unlocks (05:00Z East, 08:00Z West) followed
by evening prime time in Europe/US. For a small hour 1 (~25k) the
same fit predicts a peak ratio near 2.8x; for a very large hour 1
(~500k) it predicts a peak ratio near 0.9x.

## 5. Practice runs (sanity only — answers are not in the kit)

```
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

- All 4 practice cases (p00 Wolverine, hour 1 = 23,357; p01 Red Dead
  Redemption 2, 116,550; p02 God of War Ragnarök, 269,517; p03
  Spider-Man 2, 119,087) produce valid output.
- Median forecasts dip in hours 2–5, rise to a peak at hours 20–21,
  then decline — e.g. p01 medians: 100,310 (h2) → 80,747 (h9) →
  191,319 (h20) → 161,480 (h24); p00: 17,777 (h2) → 68,147 (h20).
- Runtime: ~0.18 s for all four cases together (~0.05 s per case),
  far under the 8 s per-case cap, on 1 CPU.
- Determinism: two runs write byte-identical files (`diff -r`
  clean). No randomness, no clock, no network; inputs are sorted.
- Shape checks (automated on every output): exactly one `hours`
  key, 23 rows, hours 2–24 each once, each row exactly
  `{"hour", "q"}` with five finite numbers, each > 0 and ≤ 1e9,
  non-decreasing; file sizes 1,442–1,530 bytes, under the 65,536 cap.
- Edge cases tested with synthetic inputs: a single case file only;
  no comparables at all (falls back to the baseline, still valid);
  hour 1 = 100 and hour 1 = 2,000,000 (finite, non-decreasing,
  ≤ 1e9 quantiles); notes rewritten to a simultaneous unlock
  (takes the all-comparables branch, wider spread).

## 6. Risks and known weaknesses

- **Hour 1 outside the training range.** The largest gradual
  hour 1 in training is 875,155 (ELDEN RING). GTA VI could exceed
  that. The log-log fit extrapolates sub-linearly (slope ~0.65 late),
  which is the conservative direction, and the extrapolation check
  above supports it — but a record-breaking launch is, by definition,
  the case the comparables say least about. The spread (SD ~0.35 log,
  ×1.15) is the main protection; it is not widened further for
  extreme hour 1, which would be a judgement call, not a fitted one.
- **Hour 1 very low.** If hour 1 came in tiny (say < 1,000) the fit
  predicts a many-fold rise, as it should for a rolling launch whose
  early regions are small — but if the true cause were a delayed
  unlock or a recording gap, the forecast would overshoot badly.
  The clipped pre-day feature damps this only partly.
- **Missing comparables / sparse data.** If the final case shipped
  far fewer comparables, or with many missing hours, the fit degrades
  to the baseline fallback. The program requires ≥ 20 of 24 hours
  per curve and ≥ 10 gradual curves before trusting the gradual fit.
- **Unlock timing is taken from the case notes.** The regime branch
  reads "rolling"/"midnight local" vs "simultaneous"/"single global"
  in `notes`, backed by the data-driven pre-day-peak ratio
  (pre24max / hour 1 < 0.10 with no rolling wording ⇒ simultaneous).
  If the final case's notes described the unlock differently, the
  wrong branch could be taken; the data-driven ratio is the backstop.
- **Practice ≠ final.** The practice is scored on SullyGnome hourly
  averages; the final on minute-mean Twitch reads. The house notes
  the two measures are close but not the same, and the final's
  history is minute reads (denser) where the practice's is hourly.
  The pre-day feature is computed from `history_hourly` in both,
  so the measure is consistent within each case.
- **No PC launch.** GTA VI is console-only at release. Several
  comparables are PC-heavy with a simultaneous PC unlock; the
  gradual-only training set already excludes the sharp-step ones,
  but a few gradual curves still contain a PC component (e.g.
  Monster Hunter Wilds' 05:00Z Steam unlock, step 1.04 — excluded
  by the step filter). Residual risk remains that console-only
  launches rise less sharply at 05:00–08:00Z than the fit expects.
- **Early access.** GTA VI has no early access (per the rules).
  Early-access-contaminated comparables (Diablo IV, Hogwarts
  Legacy, Starfield) are in the training set and are the model's
  worst cases; they pull the late-day slope slightly up. They were
  kept because removing them on a text cue ("early access" in the
  unlock field) did not improve LOO on the gradual set.
- **Small validation set.** All selection above rests on 25 gradual
  launches (37 total). A single unusual launch moves the mean by
  ~0.004. The model is deliberately simple — three coefficients per
  hour, smoothed — to avoid fitting noise in that small sample.
