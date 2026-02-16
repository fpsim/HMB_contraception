# HMB test suite

This directory contains a comprehensive test suite for the HMB contraception intervention package.

## Test files

### 1. test_hmb.py
Tests core HMB functionality without interventions:
- HMB prevalence follows age-specific odds ratios
- HMB initialization and state transitions
- HMB sequelae (anemia, pain, poor menstrual hygiene)
- Menstrual state transitions (menarche, menopause)
- Hysterectomy prevalence

### 2. test_hmb_interventions.py
Tests the intervention care pathway:
- Care-seeking behavior responds appropriately to anemia and pain
- Treatment efficacy rates match parameters
- Treatment durations follow expected distributions
- Treatment cascade progresses correctly through stages

### 3. test_refactored_architecture.py
Tests the refactored treatment architecture:
- Individual treatment classes work standalone
- Cascade orchestration works correctly
- Component-specific simulations run successfully
- Treatment prerequisite ordering is enforced

## Running tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_hmb.py
pytest tests/test_hmb_interventions.py
pytest tests/test_refactored_architecture.py

# Run specific test function
pytest tests/test_hmb.py::test_hmb_prevalence_by_age
pytest tests/test_hmb_interventions.py::test_care_seeking
pytest tests/test_refactored_architecture.py::test_individual_treatments

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=.

# Run refactored architecture tests directly
python tests/test_refactored_architecture.py
```

## Test details

### test_hmb.py - Core HMB functionality
Tests the baseline HMB module behavior:
- **HMB prevalence by age**: Validates that HMB odds ratios match age-specific patterns
- **HMB sequelae**: Tests anemia, pain, and poor menstrual hygiene outcomes
- **Menstrual state transitions**: Validates menarche and menopause timing
- **Hysterectomy prevalence**: Tests hysterectomy rates by age

### test_hmb_interventions.py - Care pathway validation
Tests intervention components using pytest framework:

#### test_care_seeking
- Validates care-seeking rates respond to anemia and pain
- Tests baseline care-seeking rates match parameters
- Checks that symptoms increase care-seeking appropriately

#### test_tx_eff
- Validates treatment response rates match efficacy parameters
- Tests that responder status is assigned correctly
- Checks efficacy applies consistently across treatment episodes

#### test_tx_dur
- Validates treatment durations follow expected distributions
- Tests that mean and variance match specifications
- Checks minimum duration constraints are respected

#### test_cascade_stage_progression
- Validates women progress through cascade stages in correct order
- Tests that non-responders escalate to next treatment line
- Checks timing of transitions matches parameters

#### test_care_propensity_effects
- Validates care propensity affects cascade progression
- Tests dropout rates across different propensity quantiles
- Checks that higher propensity leads to better engagement

#### test_tx_hmb
- Integration test that validates full simulation runs without errors
- Tests that intervention reduces HMB prevalence compared to baseline

### test_refactored_architecture.py - Architecture validation
Tests the refactored treatment design (can run standalone):
- **Individual treatments**: Each treatment (NSAID, TXA, Pill, hIUD) works independently
- **Cascade orchestration**: Treatment prerequisites are enforced (e.g., TXA requires NSAID first)
- **Component-specific simulations**: Individual treatments produce distinct outcomes
- **Cascade depth**: Distribution of how many treatments women try

## Test fixtures and utilities

Each test file provides simulation configurations:

**test_hmb.py:**
- `base_sim()`: Basic simulation without interventions (for baseline HMB testing)

**test_hmb_interventions.py:**
- `base_sim()`: Basic simulation without interventions (for baseline comparisons)
- `intervention_sim()`: Simulation with HMB care pathway intervention and analyzers

**test_refactored_architecture.py:**
- `make_component_sim(component)`: Simulation with a single treatment component
- `make_cascade_sim()`: Simulation with full cascade orchestration

## Adding new tests

To add new tests:

1. Determine which test file is most appropriate:
   - `test_hmb.py` for core HMB module behavior
   - `test_hmb_interventions.py` for care pathway and treatment validation
   - `test_refactored_architecture.py` for architecture and component testing
2. Follow the naming convention: `test_descriptive_name`
3. Include a docstring explaining what is being tested
4. Add appropriate assertions
5. Use pytest fixtures (`sim`, `sim_base`, `sim_intv`) where appropriate

## Dependencies

Tests require:
- pytest
- starsim
- fpsim
- sciris
- numpy
- pandas

See `requirements.txt` or `pyproject.toml` for specific versions.
