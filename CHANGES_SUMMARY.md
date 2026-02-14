# Summary of Changes: hmb-navideh-jan → hmb-rs-feb13

## Quick Overview

This update represents a refactor of the HMB care pathway intervention, moving from individual-level iteration to vectorized array operations and adding more details to the care-seeking and treatment response modeling.

**Files modified:**
- `interventions.py` - HMBCarePathway class
- `menstruation.py` - Moved logistic function
- `utils.py` - **NEW**: Shared utility functions
- `test_run_intervention_pathway.py` - Simplified test setup

---

## Key Architectural Changes

### 1. Vectorization (Performance Improvement)
**Before:** Individual-level iteration
```python
for uid in eligible:
    if not self.tried_nsaid[uid]:
        started = self._offer_and_start(uid, 'nsaid')
        if started:
            continue
```

**After:** Array-based operations
```python
can_try_nsaid = uids & ~self.tried_nsaid
self._offer_and_start(can_try_nsaid, 'nsaid')
```

**Impact:** Much faster for large populations, cleaner code

---

### 2. Care-Seeking Behavior Model
**Before:** Simple Bernoulli probability
```python
prob_seek_care=ss.bernoulli(p=0.3)
```

**After:** Logistic regression with covariates
```python
care_behavior=sc.objdict(
    base = 0.5,      # Baseline 50% odds
    anemic = 1,      # OR ≈ 2.7 for anemia
    pain = 0.25,     # OR ≈ 1.3 for pain
)
```

Plus individual heterogeneity via `care_seeking_propensity` (normal distribution)

**Impact:** More realistic care-seeking that responds to disease severity

---

### 3. Treatment Response System (NEW)
Added treatment efficacy and responder states:
```python
efficacy=sc.objdict(
    nsaid=0.5,  # 50% of people respond
    txa=0.6,    # 60% respond
    pill=0.7,   # 70% respond
    hiud=0.8,   # 80% respond
)
```

Responder states: `nsaid_responder`, `txa_responder`, `pill_responder`, `hiud_responder`

**Impact:** Only responders receive biological effect in menstruation module

---

### 4. Treatment Duration Management
**Before:** Incrementally tracked duration
```python
self.treatment_duration[uid] += self.sim.t.dt_year * 12
if self.treatment_duration[uid] >= max_duration:
    self._stop_treatment(uid, success=True)
```

**After:** Time-based stopping
```python
self.dur_treatment[uids] = self.pars.dur_treatment[tx].rvs(uids)
self.ti_stop_treatment[uids] = self.ti + self.dur_treatment[uids]

# Later, in step():
self.stop_treatment()  # Auto-stops when ti_stop_treatment == ti
```

**Impact:** Simpler, more realistic, no need to track duration manually

---

## Module-by-Module Changes

### interventions.py - HMBCarePathway Class

#### Parameters Added/Changed
```python
# NEW: Care-seeking behavior model
care_behavior=sc.objdict(base=0.5, anemic=1, pain=0.25)

# NEW: Treatment efficacy
efficacy=sc.objdict(nsaid=0.5, txa=0.6, pill=0.7, hiud=0.8)

# CHANGED: Treatment duration
# Before: treatment_duration_months (fixed values)
# After: dur_treatment (distributions)
dur_treatment=sc.objdict(
    nsaid=ss.uniform(ss.months(10), ss.months(14)),
    txa=ss.uniform(ss.months(10), ss.months(14)),
)
```

#### States Added/Changed
```python
# CHANGED: Renamed for clarity
seeking_care → is_seeking_care
treatment_start_ti → ti_start_treatment
treatment_duration → dur_treatment

# NEW: Care-seeking propensity
care_seeking_propensity = ss.FloatArr(default=ss.normal(1, 1))

# NEW: Responder states
nsaid_responder, txa_responder, pill_responder, hiud_responder

# NEW: Treatment stop time
ti_stop_treatment

# REMOVED
did_not_seek_care
```

#### Methods Changed
| Method | Change | Details |
|--------|--------|---------|
| `determine_care_seeking()` | Major refactor | Now uses logistic regression with covariates |
| `offer_treatment()` | Vectorized | Parallel treatment evaluation instead of sequential |
| `_offer_and_start()` | Vectorized | Processes arrays instead of individuals |
| `_start_treatment()` | Vectorized + responders | Only responders get menstruation module effects |
| `assess_treatment_effectiveness()` | Vectorized | Array-based assessment |
| `check_adherence()` | Vectorized | Array-based adherence checking |
| `stop_treatment()` | Renamed + simplified | Was `_stop_treatment()`, now public, auto-called |
| `update_treatment_duration()` | **REMOVED** | No longer needed with time-based stopping |

#### Properties Added
```python
@property
def is_eligible(self):
    """Moved from __init__"""

@property
def anemic(self):
    """Shortcut to menstruation.anemic"""

@property
def pain(self):
    """Shortcut to menstruation.pain"""

@property
def tried_all(self):
    """Check if tried all treatments"""
```

---

### menstruation.py

#### Changes
- **Removed:** `_logistic()` method (moved to utils.py)
- **Added:** Import `from utils import logistic`
- **Updated:** Calls to `self._logistic()` → `logistic(self, ...)`

---

### utils.py (NEW FILE)

```python
def logistic(instance, uids, pars, intercept_scale=None):
    """
    Calculate logistic regression probabilities.

    Supports optional intercept_scale for individual heterogeneity.
    """
```

**Purpose:** Centralized logistic regression used across modules

---

### test_run_intervention_pathway.py

**Simplifications:**
- Removed `make_pars()` function
- Removed `births` demographic module
- Removed custom parameter overrides in pathway initialization
- Cleaner, minimal test setup

---

## Migration guide

If you have code using the old HMBCarePathway:

### 1. Update parameter names
```python
# Before
pathway = HMBCarePathway(
    prob_seek_care=ss.bernoulli(p=0.3),
    treatment_duration_months={'nsaid': 12, ...}
)

# After
pathway = HMBCarePathway(
    care_behavior={'base': 0.5, 'anemic': 1, 'pain': 0.25},
    dur_treatment={'nsaid': ss.uniform(10, 14), ...}
)
```

### 2. Update state names
```python
# Before
sim.interventions.hmb_care_pathway.seeking_care
sim.interventions.hmb_care_pathway.treatment_start_ti

# After
sim.interventions.hmb_care_pathway.is_seeking_care
sim.interventions.hmb_care_pathway.ti_start_treatment
```

### 3. Treatment efficacy now built-in
No need to manually handle responders - the intervention does this automatically.

### 4. Treatment stopping is automatic
No need to manually track duration - treatments stop automatically when `ti_stop_treatment` is reached.

---

## Testing Recommendations

1. **Verify care-seeking rates** respond appropriately to anemia/pain
2. **Check treatment responder rates** match efficacy parameters
3. **Validate treatment durations** follow expected distributions
4. **Ensure cascade progression** works with vectorized logic
5. **Test edge cases:**
   - All treatments tried
   - High/low care-seeking propensity
   - Fertility intent blocking pill/hIUD

---

## Questions or issues

See full details in `CHANGELOG.md` or review the git diff:
```bash
git diff hmb-navideh-jan..hmb-rs-feb13
```
