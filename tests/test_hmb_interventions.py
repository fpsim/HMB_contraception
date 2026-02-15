"""
Test suite for HMB intervention package

Tests based on GitHub issues #29-33:
- Care-seeking rates respond to anemia/pain (#30)
- Treatment responder rates match efficacy parameters (#31)
- Treatment durations follow expected distributions (#32)
- Treatment cascade progresses correctly (#33)
"""

import sys
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import starsim as ss
import fpsim as fp
import sciris as sc

from menstruation import Menstruation
from education import Education
from interventions import HMBCarePathway
from analyzers import track_care_seeking, track_tx_eff, track_tx_dur


# ============================================================================
# Common things
# ============================================================================

def base_sim():
    """Create a basic simulation for testing"""
    mens = Menstruation()
    edu = Education()
    care_analyzer = track_care_seeking()

    sim = fp.Sim(
        start=2020,
        stop=2025,
        n_agents=1000,
        location='kenya',
        education_module=edu,
        connectors=[mens],
        analyzers=[care_analyzer],
        verbose=0,
    )
    return sim


def intervention_sim():
    """Create a simulation with HMB care pathway intervention"""
    mens = Menstruation()
    edu = Education()
    pathway = HMBCarePathway(
        year=2020,
        time_to_assess=2,
    )
    care_analyzer = track_care_seeking()
    tx_eff_analyzer = track_tx_eff()
    tx_dur_analyzer = track_tx_dur()

    sim = fp.Sim(
        start=2020,
        stop=2025,
        n_agents=1000,
        location='kenya',
        education_module=edu,
        connectors=[mens],
        interventions=[pathway],
        analyzers=[care_analyzer, tx_eff_analyzer, tx_dur_analyzer],
        verbose=0,
    )
    return sim


# ============================================================================
# Test that care-seeking rates respond to anemia/pain
# ============================================================================

def test_care_seeking(sim):
    """
    Test that women with anemia / pain have higher care-seeking rates

    Expected behavior:
    - Women with anemia / pain should be more likely to seek care
    """
    # Get analyzer results
    analyzer = sim.analyzers.track_care_seeking

    # Calculate mean care-seeking rates over the simulation period
    # Use data after first year to allow system to stabilize
    start_idx = 12  # Start after 1 year (12 months)

    mean_anemic = np.mean(analyzer.results.care_seeking_anemic[start_idx:])
    mean_not_anemic = np.mean(analyzer.results.care_seeking_not_anemic[start_idx:])
    mean_pain = np.mean(analyzer.results.care_seeking_pain[start_idx:])
    mean_no_pain = np.mean(analyzer.results.care_seeking_no_pain[start_idx:])

    # Print results
    print(f'  Mean care-seeking rate (anemic):     {mean_anemic:.3f}')
    print(f'  Mean care-seeking rate (not anemic): {mean_not_anemic:.3f}')
    print(f'  Mean care-seeking rate (pain):       {mean_pain:.3f}')
    print(f'  Mean care-seeking rate (no pain):    {mean_no_pain:.3f}')

    # Test that anemic women seek care more than non-anemic women
    assert mean_anemic > mean_not_anemic, \
        f"Anemic women should seek care more than non-anemic women ({mean_anemic:.3f} vs {mean_not_anemic:.3f})"

    # Test that women with pain seek care more than women without pain
    assert mean_pain > mean_no_pain, \
        f"Women with pain should seek care more than women without pain ({mean_pain:.3f} vs {mean_no_pain:.3f})"

    # Additional check: combined effects should be even higher
    mean_both = np.mean(analyzer.results.care_seeking_anemic_pain[start_idx:])
    print(f'  Mean care-seeking rate (anemic + pain): {mean_both:.3f}')

    assert mean_both > mean_anemic, \
        f"Women with anemia AND pain should seek care more than those with only anemia ({mean_both:.3f} vs {mean_anemic:.3f})"
    assert mean_both > mean_pain, \
        f"Women with anemia AND pain should seek care more than those with only pain ({mean_both:.3f} vs {mean_pain:.3f})"

    return


# ============================================================================
# Treatment responder rates match efficacy parameters
# ============================================================================

def test_tx_eff(sim):
    """
    Test that proportion of treatment responders matches efficacy parameters

    Expected behavior:
    - For NSAID/TXA: Actual response rate should match efficacy parameter
      (these treatments stop if ineffective)
    - For Pill/hIUD: Efficacy may not match responder rate exactly because
      women may continue using them for contraception even if ineffective for HMB
    - Response rates should be within statistical confidence intervals
    """
    from scipy import stats

    # Get analyzer and pathway intervention
    analyzer = sim.analyzers.track_tx_eff
    pathway = sim.interventions.hmb_care_pathway

    # Track test results
    all_passed = True

    # Test each treatment
    for tx_name in ['nsaid', 'txa', 'pill', 'hiud']:
        n_assessed = analyzer.n_assessed[tx_name]
        n_effective = analyzer.n_effective[tx_name]
        expected_eff = pathway.pars.efficacy[tx_name]

        print(f'\n  {tx_name.upper()}:')
        print(f'    Expected efficacy: {expected_eff:.1%}')

        if n_assessed == 0:
            print(f'    WARNING: No assessments recorded - skipping test')
            continue

        if n_assessed < 30:
            print(f'    WARNING: Only {n_assessed} assessments - small sample size')

        # Calculate actual efficacy
        actual_eff = n_effective / n_assessed
        print(f'    Actual efficacy:   {actual_eff:.1%} ({n_effective}/{n_assessed})')

        # Calculate 95% confidence interval for the actual rate
        # Using Wilson score interval (better for proportions)
        ci = stats.binom.interval(0.95, n_assessed, actual_eff)
        ci_lower = ci[0] / n_assessed
        ci_upper = ci[1] / n_assessed
        print(f'    95% CI:            [{ci_lower:.1%}, {ci_upper:.1%}]')

        # Perform binomial test: does observed data fit expected proportion?
        # Null hypothesis: true efficacy = expected efficacy
        result = stats.binomtest(n_effective, n_assessed, expected_eff, alternative='two-sided')
        p_value = result.pvalue
        print(f'    P-value:           {p_value:.3f}')

        # For pill/hIUD, efficacy measurement is confounded because women may continue
        # using them for contraception even if ineffective for HMB. Use more lenient threshold.
        if tx_name in ['pill', 'hiud']:
            threshold = 0.01  # More lenient for contraceptive methods
            note = ' (lenient threshold for contraceptive method)'
        else:
            threshold = 0.05
            note = ''

        # Test passes if p-value > threshold (can't reject null hypothesis)
        if p_value < threshold:
            print(f'    ✗ FAILED: Actual efficacy significantly different from expected{note}')
            all_passed = False
        else:
            print(f'    ✓ Passed: Actual efficacy consistent with expected{note}')

    # Overall assertion
    assert all_passed, "One or more treatments had efficacy rates significantly different from expected"

    return


# ============================================================================
# Treatment durations follow expected distributions
# ============================================================================

def test_tx_dur(sim):
    """
    Test that treatment durations follow expected distribution shape

    Expected behavior:
    - NSAID/TXA: Uniform distribution between 10-14 months
    - Pill/hIUD: Durations managed by FPsim (not tested here)
    - Mean and range should match distribution parameters
    """
    import numpy as np
    from scipy import stats

    analyzer = sim.analyzers.track_tx_dur

    # Get durations collected by analyzer
    all_passed = True

    for tx_name in ['nsaid', 'txa']:
        # Get durations from analyzer
        durations = analyzer.durations[tx_name]

        if len(durations) == 0:
            print(f'\n  {tx_name.upper()}: No durations recorded - skipping test')
            continue

        durations = np.array(durations)  # Convert list to array for analysis

        print(f'\n  {tx_name.upper()}:')
        print(f'    Sample size: {len(durations)}')
        print(f'    Mean duration: {np.mean(durations):.1f} months')
        print(f'    Min duration:  {np.min(durations):.1f} months')
        print(f'    Max duration:  {np.max(durations):.1f} months')

        # Expected: uniform(10, 14) months
        expected_min = 10
        expected_max = 14
        expected_mean = (expected_min + expected_max) / 2

        print(f'    Expected: uniform({expected_min}, {expected_max}) months')
        print(f'    Expected mean: {expected_mean:.1f} months')

        # Test 1: Check if mean is close to expected (12 months)
        mean_diff = abs(np.mean(durations) - expected_mean)
        mean_tolerance = 0.5  # Allow 0.5 month deviation from expected mean

        if mean_diff > mean_tolerance:
            print(f'    ✗ FAILED: Mean differs by {mean_diff:.2f} months (tolerance: {mean_tolerance})')
            all_passed = False
        else:
            print(f'    ✓ Passed: Mean within expected range')

        # Test 2: Check if values are within expected range
        within_range = (durations >= expected_min) & (durations <= expected_max)
        pct_within = np.mean(within_range) * 100

        print(f'    Values within [{expected_min}, {expected_max}]: {pct_within:.1f}%')

        if pct_within < 95:  # Allow 5% outliers due to floating point
            print(f'    ✗ FAILED: Only {pct_within:.1f}% within expected range')
            all_passed = False
        else:
            print(f'    ✓ Passed: Values within expected range')

        # Test 3: Kolmogorov-Smirnov test for uniform distribution
        # Normalize durations to [0, 1] range
        normalized = (durations - expected_min) / (expected_max - expected_min)
        ks_stat, ks_pvalue = stats.kstest(normalized, 'uniform')

        print(f'    K-S test p-value: {ks_pvalue:.3f}')

        if ks_pvalue < 0.05:
            print(f'    ✗ FAILED: Distribution significantly different from uniform')
            all_passed = False
        else:
            print(f'    ✓ Passed: Distribution consistent with uniform')

    # Note about pill/hIUD
    print(f'\n  PILL/HIUD: Durations managed by FPsim (not tested here)')

    # Overall assertion
    assert all_passed, "One or more treatment durations did not match expected distributions"

    return


# ============================================================================
# Treatment cascade progresses correctly
# ============================================================================

def test_cascade_stage_progression(intervention_sim):
    """
    Test that women progress through cascade stages in correct order

    Expected behavior:
    - Women should move through stages: NSAID → TXA → pill → IUD
    - No stage should be skipped
    """
    return


# ============================================================================
# Integration tests
# ============================================================================

def test_tx_hmb(sim_base, sim_intv):
    """
    Test that intervention reduces HMB prevalence compared to baseline

    Expected behavior:
    - HMB prevalence should be lower with intervention than without
    """
    # TODO: Implement test
    # 1. Run both baseline and intervention simulations
    # 2. Compare final HMB prevalence
    # 3. Assert intervention reduces HMB
    return


if __name__ == '__main__':

    # Create and run baseline simulation
    print('Creating baseline simulation...')
    sim_base = base_sim()
    print('Running baseline simulation...')
    sim_base.run()

    # Create and run intervention simulation
    print('Creating intervention simulation...')
    sim_intv = intervention_sim()
    print('Running intervention simulation...')
    sim_intv.run()

    # Run tests
    sc.heading('Running tests')

    print('Test 1: Care-seeking rates respond to anemia/pain')
    test_care_seeking(sim_intv)
    print('✓ Passed\n')

    print('Test 2: Treatment responder rates match efficacy parameters')
    test_tx_eff(sim_intv)
    print('✓ Passed\n')

    print('Test 3: Treatment durations follow expected distributions')
    test_tx_dur(sim_intv)
    print('✓ Passed\n')

    # print('Test 4: Treatment cascade progresses correctly')
    # test_cascade_stage_progression(sim_intv)
    # print('✓ Passed\n')

    # print('Test 5: Intervention reduces HMB prevalence')
    # test_tx_hmb(sim_base, sim_intv)
    # print('✓ Passed\n')

    # print('='*80)
    # print('All tests passed!')
    # print('='*80)

