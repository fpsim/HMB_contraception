# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
