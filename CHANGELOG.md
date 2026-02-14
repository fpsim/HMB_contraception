# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [hmb-rs-feb13] - 2024-02-13

### Added
- **utils.py**: New utility module with shared `logistic()` function for logistic regression calculations
  - Supports optional `intercept_scale` parameter for scaling baseline probabilities
  - Used across multiple modules for consistent probability calculations

- **HMBCarePathway intervention**:
  - Treatment efficacy system with responder states (`nsaid_responder`, `txa_responder`, `pill_responder`, `hiud_responder`)
  - Efficacy parameters: NSAID (50%), TXA (60%), Pill (70%), hIUD (80%)
  - Care-seeking behavior model using logistic regression with anemia and pain as covariates
  - Individual care-seeking propensity (`care_seeking_propensity`) with heterogeneity
  - Treatment duration tracking via `ti_stop_treatment` for automated treatment cessation
  - Property shortcuts for accessing menstruation states (`anemic`, `pain`)
  - `tried_all` property to check if individual has tried all available treatments


### Changed

#### HMBCarePathway - Major Refactor
**Core Architecture**:
- Vectorized all operations: changed from individual-level (`uid`) to array-based (`uids`) processing
  - `_offer_and_start()`: now processes arrays of UIDs instead of single individuals
  - `_start_treatment()`: vectorized to handle multiple individuals simultaneously
  - `assess_treatment_effectiveness()`: vectorized assessment logic
  - `check_adherence()`: vectorized adherence checking
  - `stop_treatment()`: vectorized treatment stopping

**Care-seeking Behavior**:
- Replaced simple Bernoulli probability (`prob_seek_care`) with logistic regression model
- Care-seeking now influenced by:
  - Baseline probability (50% odds)
  - Anemia status (OR = exp(1) ≈ 2.7x increase)
  - Pain status (OR = exp(0.25) ≈ 1.3x increase)
  - Individual propensity (normally distributed, mean=1, sd=1)
- State renamed: `seeking_care` → `is_seeking_care`
- Removed `did_not_seek_care` state (no longer needed)

**Treatment Management**:
- State naming standardization:
  - `treatment_start_ti` → `ti_start_treatment`
  - `treatment_duration` → `dur_treatment`
  - Added `ti_stop_treatment` for explicit treatment end timing
- Treatment duration now set at initiation (not incrementally tracked)
  - NSAID/TXA: Uniform distribution between 10-14 months
  - Pill/hIUD: Duration set by FPsim contraception connector
- Removed `update_treatment_duration()` method (no longer needed with time-based stopping)

**Treatment Response System**:
- Only treatment responders have menstruation module states updated
- Non-responders marked as tried but don't receive biological effect
- Treatment effectiveness assessed by HMB status after `time_to_assess` period
- Failed treatments automatically stop at next timestep via `ti_stop_treatment`

**Adherence**:
- Moved from individual iteration to vectorized array operations
- Added `_p_adherent` Bernoulli distribution for consistency with other probability calculations
- Non-adherent individuals scheduled to stop at next timestep

**Treatment Stopping**:
- Changed from individual-based `_stop_treatment(uid)` to vectorized `stop_treatment(uids)`
- Automatic stopping based on `ti_stop_treatment` reaching current time
- Simplified logic: just reset states, no complex success/failure branching
- Treatment states reset to defaults (current_treatment=0, dur_treatment=nan)

**Treatment Offering**:
- Simplified cascade logic: all treatments checked in parallel rather than sequential iteration
- Changed from returning boolean to void (state changes indicate success)
- "Tried" status now set regardless of offer/accept outcome
- Removed postpartum check from pill/hIUD eligibility (kept fertility intent check)

**Code Organization**:
- Moved eligibility from `__init__` to property `is_eligible`
- Timing parameters reorganized:
  - `time_to_assess` now uses `ss.months(3)` for clarity
  - `treatment_duration_months` → `dur_treatment` with distributions
  - Removed pill/hIUD durations (handled by FPsim)
- Added property methods for cleaner code access

### Removed
- **HMBCarePathway**:
  - `update_treatment_duration()` method (replaced by time-based stopping)
  - `did_not_seek_care` state (no longer tracking this explicitly)
  - Fixed duration parameters for pill and hIUD (now handled by FPsim)
  - `success` parameter from `stop_treatment()` (simplified logic)

- **Menstruation**:
  - `_logistic()` method (moved to utils.py as standalone function)

- **Test file**:
  - `make_pars()` function (simplified test setup)
  - `births` demographic module (not needed for pathway testing)
  - Custom adherence parameters in pathway initialization

### Fixed
- Care-seeking behavior now properly accounts for disease severity (anemia, pain)
- Treatment duration properly initialized at treatment start (not tracked incrementally)
- Vectorized operations improve performance for large populations
- Treatment stopping now occurs at specified time rather than requiring manual checking
- Consistent use of Bernoulli distributions for all probability-based filtering

### Technical Improvements
- **Performance**: Vectorized operations significantly faster than individual iteration
- **Maintainability**:
  - Centralized logistic regression in utils.py
  - Clearer separation between treatment initiation and duration management
  - Simplified treatment stopping logic
- **Consistency**:
  - Standardized state naming conventions (ti_* for time indices)
  - Uniform use of distribution objects for probability calculations

### Notes
- Treatment responder system allows for heterogeneous treatment response
- Care-seeking propensity adds realistic variation in healthcare-seeking behavior
- Time-based treatment stopping is more realistic than duration tracking
- Commented out `no_care_prev` result tracking (may be re-enabled later)

---

## [0.2.0] - HMB Care Pathway

### Added
- HMBCarePathway intervention class
- Sequential treatment cascade: NSAID → TXA → Pill → hIUD
- Individual-level iteration through cascade
- Simple care-seeking probability model
- Treatment duration tracking
- Fixed treatment durations per treatment type
- Basic adherence checking

### Changed
- Extended menstruation module for HMB interventions
- Added treatment-specific states

---

## [0.1.0] - Initial Release

### Added
- Initial HMB contraception model implementation
- Menstruation module with HMB states
- Basic intervention framework
- Integration with FPsim for contraception modeling
