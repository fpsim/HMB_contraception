# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-02-15

### Added
- **Modular intervention architecture:**
  - Refactored monolithic `HMBCarePathway` into modular treatment classes:
    - `NSAIDTreatment`, `TXATreatment`, `PillTreatment`, `hIUDTreatment`
    - Each treatment is a standalone intervention that can be used independently
    - `HMBCascade` orchestrator coordinates sequential treatment offering
  - `HMBTreatmentBase` base class with shared functionality
  - Flexible eligibility system using functions instead of hard-coded dependencies
- **Enhanced anemia tracking:**
  - `track_hmb_anemia()` analyzer for monitoring HMB-anemia relationships
  - Tracks anemia prevalence stratified by HMB status (with/without HMB)
  - Tracks HMB prevalence stratified by anemia status (with/without anemia)
  - Comprehensive anemia counts and prevalence metrics in menstruating non-pregnant women
- **Component analysis tools:**
  - `component_analysis.py` for analyzing individual treatment impacts
  - `run_component_analysis.py` for running component-level comparisons
  - `analyze_cascade_impact.py` with heuristic calculator for treatment success probabilities
- **Comprehensive plotting and analysis:**
  - `run_cascade.py` for full cascade intervention analysis with visualization
  - `run_baseline.py` for baseline simulation and characteristics plotting
  - Integrated plotting functions for intervention impact, cascade progression, and baseline dynamics
- **Enhanced analyzers:**
  - `track_cascade()` analyzer for detailed cascade metrics and treatment depth tracking
  - Expanded `track_care_seeking()` with care-seeking propensity stratification
  - All analyzers updated to work with modular architecture

### Changed
- **HMB sequelae calculation timing (CRITICAL BUG FIX):**
  - **Fixed execution order bug that prevented interventions from reducing anemia**
  - Moved sequelae (anemia, pain, poor menstrual hygiene) calculation from `step()` to `finish_step()`
  - Sequelae now calculated AFTER interventions run, ensuring they reflect post-intervention HMB status
  - Previously, sequelae were calculated before interventions, causing interventions to have no impact on total anemia
  - New execution order: Connector.step() → Intervention.step() → Connector.finish_step() → update_results()
- **Treatment continuation logic (BUG FIX):**
  - Added `was_effective` state to track successful treatments persistently
  - Women can now continue on treatments that worked for them instead of being forced to progress through cascade
  - Modified eligibility logic: `(~tried_treatment | was_effective) & ~on_treatment`
  - Dramatically improves intervention effectiveness by allowing continuation of successful treatments
- **HMB prediction model:**
  - Increased baseline HMB probability from 0.95 to 0.995 among prone individuals
  - Removed treatment effects from menstruation module (now handled via interventions)
  - Simplified prediction logic by centralizing treatment response in intervention
- **Code organization:**
  - Removed deprecated pill/hIUD/TXA states from menstruation module
  - Consolidated treatment effects in intervention module
  - Updated all tests to work with modular architecture
  - Reorganized analysis and plotting scripts

### Removed
- Treatment-specific states from menstruation module (`pill`, `hiud`, `txa`, `hiud_prone`)
- Treatment effect parameters from HMB and sequelae prediction in menstruation module
- Old analysis scripts (`plot_analysis.py`, `run_analysis.py`)
- Old example scripts (`run_kenya.py`, `run_kenya_package_extended.py`, `run_sensitivity_analysis.py`, `test_run.py`)
- Jupyter notebook (`HMB intervention package - explore parameter sweep.ipynb`)

---

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
