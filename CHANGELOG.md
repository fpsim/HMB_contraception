# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-02-15

### Added
- **New modules:**
  - `analyzers.py` with three specialized analyzers:
    - `track_care_seeking()` - monitors care-seeking rates stratified by anemia and pain status
    - `track_tx_eff()` - tracks treatment effectiveness rates at assessment points
    - `track_tx_dur()` - monitors actual treatment durations by type
  - `utils.py` with shared `logistic()` function for probability calculations across modules
- **Treatment efficacy system:**
  - Heterogeneous responder model with treatment-specific response rates:
    - NSAID: 50% effective
    - TXA: 60% effective
    - Pill: 70% effective
    - hIUD: 80% effective
  - Individual-level response determination at treatment initiation
  - Assessment after configurable `time_to_assess` period (default: 3 months)
- **Enhanced care-seeking model:**
  - Logistic regression model replacing simple probability
  - Baseline 50% probability with modifiers for:
    - Anemia status (odds multiplier ≈2.7x)
    - Menstrual pain (odds multiplier ≈1.3x)
    - Individual-level heterogeneity via `care_seeking_propensity` state
- **Treatment duration management:**
  - Automatic stopping at pre-specified durations
  - `ti_stop_treatment` state for time-based cessation
  - Re-entry to care pathway for persistent HMB after treatment stops
- **Comprehensive test suite:**
  - `tests/test_hmb_interventions.py` with 4 test categories covering:
    - Care-seeking response to anemia/pain
    - Treatment responder rates matching efficacy parameters
    - Treatment duration distributions
    - Treatment cascade progression
  - `tests/README.md` with test documentation
- **Documentation:**
  - Care pathway flowchart (Mermaid diagram) in README
  - Comprehensive docstrings throughout intervention code

### Changed
- **Performance optimization:**
  - Complete vectorization of HMBCarePathway from individual-level iteration to array-based operations
  - Parallel processing of treatment cascade eligibility
  - Vectorized state updates and transitions
- **State naming standardization:**
  - Prefix conventions: `ti_*` for time indices, `dur_*` for durations, `is_*` for boolean states
  - Renamed: `seeking_care` → `is_seeking_care`
- **Treatment cascade logic:**
  - Simplified parallel checking of all treatment options
  - Integrated adherence checking with effectiveness assessment
  - Treatment duration set at initiation with type-specific distributions:
    - NSAID/TXA: 10-14 months (uniform distribution)
    - Pill/hIUD: FPsim-determined durations
- **Code organization:**
  - Moved logistic regression utility from Menstruation to shared utils module
  - Consolidated treatment-related parameters in intervention class
  - Extracted analyzers into dedicated module

### Removed
- `did_not_seek_care` state (replaced by absence of `is_seeking_care`)
- `update_treatment_duration()` method (functionality absorbed into vectorized flow)
- Individual-iteration methods in HMBCarePathway (replaced by vectorized operations)
- `_logistic()` method from Menstruation class (moved to utils.py)

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
