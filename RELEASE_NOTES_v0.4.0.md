# Release Notes: v0.4.0

**Release Date:** February 15, 2026

## Summary

Version 0.4.0 is a major release that includes critical bug fixes, architectural improvements, and enhanced analysis capabilities. Most importantly, this release fixes a critical bug that prevented interventions from properly reducing anemia in the population.

---

## Critical Bug Fixes

### 1. HMB Sequelae Timing Bug (CRITICAL)

**Issue:** Interventions appeared to have no impact on total anemia despite successfully reducing HMB.

**Root Cause:** HMB sequelae (anemia, pain, poor menstrual hygiene) were calculated in `Menstruation.step()` **before** interventions ran. The execution order was:
1. `Menstruation.step()` calculates sequelae based on pre-intervention HMB
2. `Interventions.step()` modifies HMB status
3. `update_results()` records the already-calculated sequelae

**Fix:** Moved sequelae calculation to `Menstruation.finish_step()`, which runs **after** interventions:
1. `Menstruation.step()` calculates HMB
2. `Interventions.step()` modifies HMB status
3. `Menstruation.finish_step()` calculates sequelae based on post-intervention HMB ✅
4. `update_results()` records the corrected sequelae

**Impact:** With this fix, interventions now properly reduce:
- HMB-related anemia: 98% reduction
- Total anemia: 35% reduction
- Overall HMB prevalence: 95% reduction (from 26% to 1.3%)

### 2. Treatment Continuation Bug

**Issue:** Women could not continue on treatments that worked for them. Once a treatment duration ended, women were forced to progress to the next treatment in the cascade, even if the previous treatment was effective.

**Root Cause:** Eligibility logic used `~tried_treatment`, which permanently excluded anyone who had previously tried a treatment.

**Fix:**
- Added `was_effective` state to persistently track successful treatments
- Modified eligibility: `(~tried_treatment | was_effective) & ~on_treatment`
- Women can now restart treatments that previously worked

**Impact:** Allows women to continue effective treatments instead of unnecessarily progressing through the cascade.

---

## Architectural Improvements

### Modular Intervention Architecture

**Previous (v0.3.0):** Monolithic `HMBCarePathway` class with all treatment logic intertwined.

**New (v0.4.0):** Modular architecture with:
- Individual treatment classes: `NSAIDTreatment`, `TXATreatment`, `PillTreatment`, `hIUDTreatment`
- Shared `HMBTreatmentBase` base class with common functionality
- `HMBCascade` orchestrator that coordinates treatment sequencing
- Flexible eligibility system using functions instead of hard-coded dependencies

**Benefits:**
- Easier to analyze individual treatment impacts
- Can run single-treatment scenarios for component analysis
- More maintainable and testable code
- Better separation of concerns

---

## New Features

### Component Analysis Tools

**Added:**
- `component_analysis.py` - Functions for analyzing individual treatment impacts
- `run_component_analysis.py` - Script for running component comparisons
- `analyze_cascade_impact.py` - Heuristic calculator for treatment cascade probabilities

**Enables:**
- Comparison of NSAID-only, TXA-only, Pill-only, and hIUD-only interventions
- Understanding of incremental benefit of each treatment
- Validation of cascade sequencing strategy

### Enhanced Cascade Analysis

**Added:**
- `run_cascade.py` - Comprehensive cascade analysis with integrated plotting
- `run_baseline.py` - Baseline simulation analysis
- `track_cascade()` analyzer - Detailed cascade depth and progression metrics

**New Metrics:**
- Treatment cascade depth (number of treatments tried: 0, 1, 2, 3, 4)
- Cascade dropoff points (offered vs accepted at each stage)
- Anemia prevalence by cascade depth
- Treatment-specific anemia outcomes

---

## Comparison: v0.3.0 vs v0.4.0

| Aspect | v0.3.0 (hmb-rs-feb13) | v0.4.0 (anemia-work) |
|--------|----------------------|----------------------|
| **Architecture** | Monolithic HMBCarePathway | Modular treatment classes |
| **Anemia Impact** | ❌ Bug: No impact on total anemia | ✅ Fixed: 35% reduction in total anemia |
| **Treatment Continuation** | ❌ Bug: Cannot restart effective treatments | ✅ Fixed: Can continue effective treatments |
| **Component Analysis** | Not available | ✅ Full component analysis tools |
| **Cascade Metrics** | Basic tracking | ✅ Comprehensive cascade depth analysis |
| **Code Organization** | Single large class | Separate treatment classes + orchestrator |
| **Testability** | Difficult to test components | ✅ Easy to test individual treatments |

---

## Migration Notes

### For Existing Code

If you have code using `HMBCarePathway` from v0.3.0, update to:

```python
# Old (v0.3.0)
from interventions import HMBCarePathway
pathway = HMBCarePathway()

# New (v0.4.0)
from interventions import HMBCascade
cascade = HMBCascade()
```

The interface is similar, but internal structure is different.

### For Analyzers

Analyzers that accessed intervention states directly may need updates:

```python
# Old (v0.3.0)
pathway.treatment_map
pathway.current_treatment

# New (v0.4.0)
cascade.treatments['nsaid'].on_treatment
cascade.treatments['txa'].on_treatment
# etc.
```

---

## Performance

No significant performance changes. The modular architecture maintains similar performance to the monolithic version while being more maintainable.

---

## Testing

All tests updated and passing:
- Modular treatment tests
- Cascade progression tests
- Care-seeking behavior tests
- Treatment effectiveness tests
- Adherence tests

---

## Known Issues

None at this time.

---

## Acknowledgments

Bug fixes and architecture improvements developed with assistance from Claude Sonnet 4.5.
