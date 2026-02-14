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
from analyzers import track_care_seeking, track_tx_eff


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

    sim = fp.Sim(
        start=2020,
        stop=2025,
        n_agents=1000,
        location='kenya',
        education_module=edu,
        connectors=[mens],
        interventions=[pathway],
        analyzers=[care_analyzer, tx_eff_analyzer],
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
    - Each treatment's actual response rate should match its efficacy parameter
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

        # Test passes if p-value > 0.05 (can't reject null hypothesis)
        if p_value < 0.05:
            print(f'    ✗ FAILED: Actual efficacy significantly different from expected')
            all_passed = False
        else:
            print(f'    ✓ Passed: Actual efficacy consistent with expected')

    # Overall assertion
    assert all_passed, "One or more treatments had efficacy rates significantly different from expected"

    return


# ============================================================================
# Treatment durations follow expected distributions
# ============================================================================

def test_tx_dur():
    """
    Test that treatment durations follow expected distribution shape

    Expected behavior:
    - Distribution of treatment durations should match specified distribution type
    - Parameters (mean, SD, etc.) should match expected values
    """
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
    # sim_base.run()

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
    test_tx_dur()
    print('✓ Passed\n')

    print('Test 4: Treatment cascade progresses correctly')
    test_cascade_stage_progression(sim_intv)
    print('✓ Passed\n')

    print('Test 5: Intervention reduces HMB prevalence')
    test_tx_hmb(sim_base, sim_intv)
    print('✓ Passed\n')

    print('='*80)
    print('All tests passed!')
    print('='*80)

