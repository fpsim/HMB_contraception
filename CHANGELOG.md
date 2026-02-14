# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [hmb-rs-feb13] - 2024-02-13

### Added
- `utils.py` module with shared `logistic()` function for probability calculations
- Treatment efficacy/responder system with heterogeneous response rates (NSAID 50%, TXA 60%, Pill 70%, hIUD 80%)
- Logistic regression model for care-seeking behavior (influenced by anemia, pain, and individual propensity)
- Time-based treatment stopping via `ti_stop_treatment` state

### Changed
- Vectorized all HMBCarePathway operations from individual-level to array-based processing
- Standardized state naming conventions (`ti_*` for time indices, `dur_*` for durations)
- Simplified treatment cascade logic with parallel checking
- Treatment duration set at initiation (NSAID/TXA: 10-14 months uniform; Pill/hIUD: FPsim-determined)

### Removed
- Individual-iteration methods replaced by vectorized operations
- `_logistic()` method from Menstruation (moved to utils.py)
- `did_not_seek_care` state (no longer needed)

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
