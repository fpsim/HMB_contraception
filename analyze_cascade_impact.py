"""
Heuristic analysis of cascade treatment impact

This script calculates the expected probability that a woman with HMB
will be successfully treated at each stage of the cascade, helping
explain why the intervention has such a large impact.
"""
import numpy as np
import sciris as sc

def calculate_treatment_success_probability(pars):
    """
    Calculate the probability that a woman successfully gets treated
    at each level of the cascade.

    For a woman to be successfully treated, she needs to:
    1. Seek care
    2. Be offered treatment
    3. Accept treatment
    4. Be a responder (treatment is efficacious for her)
    5. Adhere to treatment

    Args:
        pars: Dictionary with treatment parameters (efficacy, adherence, prob_offer, prob_accept)
              and care_behavior parameters

    Returns:
        Dictionary with probabilities at each stage
    """
    results = {}

    # Assume base care-seeking probability (will be higher for anemic women)
    # From interventions.py: care_behavior = {base: 0.5, anemic: 1, pain: 0.25}
    # For anemic HMB women, logistic gives higher probability
    # Let's use a reasonable estimate of ~0.7 for anemic HMB women
    p_seek_care = 0.7

    results['p_seek_care'] = p_seek_care

    # For each treatment in the cascade
    treatments = ['nsaid', 'txa', 'pill', 'hiud']
    cumulative_success = 0  # Track cumulative probability of success

    for tx in treatments:
        tx_pars = pars[tx]

        # Probability of success at this treatment stage
        # (given that previous treatments failed or weren't tried)
        p_offer = tx_pars['prob_offer']
        p_accept = tx_pars['prob_accept']
        p_responder = tx_pars['efficacy']
        p_adherent = tx_pars['adherence']

        # Overall probability of success at this stage
        p_success_this_stage = (p_seek_care *
                                p_offer *
                                p_accept *
                                p_responder *
                                p_adherent)

        # Probability of failure (trying but not succeeding)
        p_try_but_fail = (p_seek_care *
                         p_offer *
                         p_accept *
                         (1 - p_responder * p_adherent))

        # Store results
        results[tx] = {
            'p_offer': p_offer,
            'p_accept': p_accept,
            'p_responder': p_responder,
            'p_adherent': p_adherent,
            'p_success_this_stage': p_success_this_stage,
            'p_try_but_fail': p_try_but_fail,
        }

        # Add to cumulative success (each stage is a chance to succeed)
        cumulative_success += p_success_this_stage * (1 - cumulative_success)

    results['cumulative_success'] = cumulative_success

    return results


def print_cascade_analysis(pars):
    """Print a detailed analysis of the cascade treatment probabilities."""
    results = calculate_treatment_success_probability(pars)

    print('\n' + '='*70)
    print('TREATMENT CASCADE PROBABILITY ANALYSIS')
    print('='*70)
    print(f'\nBase care-seeking probability (for anemic HMB women): {results["p_seek_care"]:.1%}')
    print('\n' + '-'*70)

    treatments = {
        'nsaid': 'NSAID',
        'txa': 'TXA',
        'pill': 'Pill',
        'hiud': 'hIUD'
    }

    for tx_key, tx_name in treatments.items():
        tx_results = results[tx_key]
        print(f'\n{tx_name}:')
        print(f'  Offered:         {tx_results["p_offer"]:.1%}')
        print(f'  Accepted:        {tx_results["p_accept"]:.1%}')
        print(f'  Responder:       {tx_results["p_responder"]:.1%}')
        print(f'  Adherent:        {tx_results["p_adherent"]:.1%}')
        print(f'  → Success:       {tx_results["p_success_this_stage"]:.2%}')
        print(f'  → Try but fail:  {tx_results["p_try_but_fail"]:.2%}')

    print('\n' + '-'*70)
    print(f'\nCumulative probability of successful treatment: {results["cumulative_success"]:.1%}')
    print('\nThis means:')
    print(f'  - {results["cumulative_success"]:.1%} of HMB women get successfully treated')
    print(f'  - {1-results["cumulative_success"]:.1%} remain with HMB (and higher anemia risk)')

    # Calculate expected impact on anemia
    print('\n' + '-'*70)
    print('EXPECTED IMPACT ON ANEMIA:')
    print('-'*70)

    # From menstruation.py: anemic = {base: 0.18, hmb: effect}
    # Effect of HMB: -np.log(1/0.35 - 1) + np.log(1/0.18 - 1) ≈ 1.0
    # So: P(anemia | no HMB) ≈ 0.18, P(anemia | HMB) ≈ 0.35
    p_anemia_no_hmb = 0.18
    p_anemia_hmb = 0.35

    # With intervention:
    p_treated = results["cumulative_success"]
    p_untreated = 1 - p_treated

    # Treated women effectively have no HMB (HMB=False for responders)
    p_anemia_with_intervention = p_treated * p_anemia_no_hmb + p_untreated * p_anemia_hmb

    print(f'\nAnemia prevalence among HMB women:')
    print(f'  Without intervention: {p_anemia_hmb:.1%}')
    print(f'  With intervention:    {p_anemia_with_intervention:.1%}')
    print(f'  Reduction:            {(p_anemia_hmb - p_anemia_with_intervention)/p_anemia_hmb:.1%}')

    print('\n' + '='*70 + '\n')

    return results


if __name__ == '__main__':
    # Load parameters from HMBCascade default values
    pars = {
        'nsaid': {
            'efficacy': 0.5,
            'adherence': 0.7,
            'prob_offer': 0.9,
            'prob_accept': 0.7,
        },
        'txa': {
            'efficacy': 0.6,
            'adherence': 0.6,
            'prob_offer': 0.9,
            'prob_accept': 0.6,
        },
        'pill': {
            'efficacy': 0.7,
            'adherence': 0.75,
            'prob_offer': 0.9,
            'prob_accept': 0.5,
        },
        'hiud': {
            'efficacy': 0.8,
            'adherence': 0.85,
            'prob_offer': 0.9,
            'prob_accept': 0.5,
        },
    }

    results = print_cascade_analysis(pars)
