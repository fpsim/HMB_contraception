# HMB intervention test suite

This directory contains a comprehensive test suite for the HMB contraception intervention package.

## Test structure

The test suite validates four key aspects of the intervention:

- **Care-seeking behavior**: Care-seeking rates respond appropriately to anemia and pain
- **Treatment efficacy**: Treatment responder rates match efficacy parameters
- **Treatment durations**: Treatment durations follow expected distributions
- **Treatment cascade**: Treatment cascade progresses correctly through stages

## Running tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_hmb_interventions.py

# Run specific test class
pytest tests/test_hmb_interventions.py::TestCareSeeking

# Run specific test
pytest tests/test_hmb_interventions.py::TestCareSeeking::test_anemia_increases_care_seeking

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=.
```

## Test implementation

All tests are fully implemented and validate:
- Care-seeking behavior responds to symptoms (anemia, pain)
- Treatment response rates match specified efficacy parameters
- Treatment durations follow expected distributions
- Treatment cascade progression follows correct sequence
- Integration of all components works as expected

## Test categories

### 1. Care-seeking tests (`TestCareSeeking`)
These tests verify that care-seeking behavior responds appropriately to symptoms:
- Anemia should increase care-seeking rates
- Pain should increase care-seeking rates
- Combined symptoms should have appropriate combined effects
- Baseline rates should match parameters

### 2. Treatment efficacy tests (`TestTreatmentEfficacy`)
These tests verify that treatment response rates match efficacy parameters:
- Response rates should match specified efficacy
- Different treatments should have different efficacy rates
- Responder status should be assigned correctly
- Efficacy should apply consistently across treatment episodes

### 3. Treatment duration tests (`TestTreatmentDurations`)
These tests verify that treatment durations follow expected distributions:
- Duration distribution should match specified type and parameters
- Mean and variance should match specifications
- Minimum duration constraints should be respected
- Different treatments should have distinct duration distributions

### 4. Treatment cascade tests (`TestTreatmentCascade`)
These tests verify that the care cascade progresses correctly:
- Women should progress through stages in correct order
- Non-responders should escalate to next treatment line
- Responders should complete cascade appropriately
- Timing of transitions should match parameters
- Dropout rates should match specifications
- Maximum treatment lines should be respected

### 5. Integration tests (`TestIntegration`)
These tests verify the complete system works together:
- Full simulation should run without errors
- Intervention should reduce HMB prevalence

## Test fixtures

The test suite provides two main simulation configurations:

- `base_sim()`: A basic simulation without interventions (for baseline comparisons)
- `intervention_sim()`: A simulation with the HMB care pathway intervention and all analyzers

## Adding new tests

To add new tests:

1. Determine which test class the test belongs to
2. Add the test method to that class
3. Follow the naming convention: `test_descriptive_name`
4. Include docstring explaining what is being tested
5. Add appropriate assertions

## Dependencies

Tests require:
- pytest
- starsim
- fpsim
- sciris
- numpy
- pandas

See `requirements.txt` or `pyproject.toml` for specific versions.
