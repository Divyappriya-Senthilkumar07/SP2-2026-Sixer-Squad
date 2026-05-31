# Sona Power Predict – 2026

## Team Details
| Field | Details |
|-------|---------|
| **College Name** | Sona College of Technology |
| **Team Name** | Sixer Squad |
| **Team Lead** | Divyappriya S |

## Team Members
| Name | Year | Department |
|------|------|------------|
| Divyappriya S (Team Lead) | 2nd Year | Computer Science and Engineering |
| Vaishnavi K | 3rd Year | Computer Science and Engineering |
| Grace Crystel C | 3rd Year | Computer Science and Engineering |
| Mouriya V K | 3rd Year | Computer Science and Engineering |

## Libraries Used
- `pandas` — data loading, grouping, and feature engineering
- `numpy` — numerical computations, weighted averages, clipping
- `os` — file path checks for Docker container paths

## Model Approach

### Problem
Predict the IPL T20 powerplay score (first 6 overs) for both innings of a match, given the venue, batting team, bowling team, and player IDs.

### Strategy
Our model uses a **7-layer blended prediction pipeline**, where every value is learned dynamically from the training data — nothing is hardcoded.

---

### Layer 1 — Exponential Decay Team Averages
We compute a decay-weighted PP batting and bowling average for each team using:

```
weight = 0.65 ^ (2026 - year)
```

This means 2025 data is weighted ~4x more than 2022 data, capturing the recent scoring trends without discarding historical context.

---

### Layer 2 — Head-to-Head (H2H) Blend
When a batting team vs bowling team pair has ≥2 historical PP innings, we blend the H2H decay-weighted average with the team-combined average:

```
base = 0.45 × H2H_avg + 0.55 × team_combined
```

H2H is the single most predictive feature in our analysis.

---

### Layer 3 — Venue Adjustment
We merge the deliveries data with the matches data to compute per-venue PP averages. The venue average is blended at 10% weight:

```
base = 0.90 × base + 0.10 × venue_avg
```

---

### Layer 4 — Dynamic Season Uplift
Instead of a hardcoded value, we compute the uplift dynamically from the data:

```
uplift = (2024-25 avg − pre-2023 baseline) × 0.20
```

This captures the 2026 IPL scoring trend without overfitting.

---

### Layer 5 — Bowling Team Wicket Rate
Teams that take more PP wickets suppress the batting team's score. We compute each bowling team's PP wicket rate per innings (2022+) and adjust:

```
adj = -(wicket_rate − league_avg) × 5.0   [capped at ±4]
```

---

### Layer 6 — Player SR & Economy Adjustment
The test input provides batting and bowling player IDs. We:
1. Map numeric IDs → player names using the `ipl_players_uniqueid.csv` provided by Docker
2. Look up each player's PP strike rate (batsmen) or PP economy (bowlers) from training data
3. Average across all provided players and adjust:

```
SR adjustment  = (avg_SR − league_SR) × 0.54   [capped at ±8]
Econ adjustment = (league_econ − avg_econ) × 1.20  [capped at ±5]
```

Player ID mapping uses first-initial + last-name matching to bridge full names (CSV) to abbreviated names (training data).

---

### Layer 7 — Innings Correction
The 2nd innings historically scores ~1.57 more runs in PP than the 1st innings (computed dynamically from training data).

---

### Final Prediction
```
final = int(clip(base + player_adj + inn_adj, 10, 120))
```

All predictions are integers truncated to the range [10, 120].

---

### Key Design Decisions
- **Exponential decay** instead of simple recent averages — more stable across seasons
- **Dynamic uplift** instead of hardcoded correction — adapts to new data
- **Multi-player averaging** — handles comma-separated player ID lists
- **Embedded player ID map** as fallback — works even if player CSV is unavailable
- **Full try/except safety** — model never crashes, always returns a prediction
