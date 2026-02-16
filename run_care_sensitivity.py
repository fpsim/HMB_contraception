"""
Sensitivity analysis for care-seeking heterogeneity assumption

Compares:
1. Baseline: Heterogeneous care-seeking propensity (tied to severity)
2. Homogeneous: Everyone has equal care-seeking propensity

This analysis shows how important individual variation in care-seeking is
to the intervention's effectiveness and equity.
"""

import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
import starsim as ss
import fpsim as fp

from menstruation import Menstruation
from education import Education
from interventions import HMBCascade
from analyzers import track_care_seeking, track_tx_eff, track_tx_dur, track_hmb_anemia, track_cascade, track_anemia_duration


# ============================================================================
# Simulation creation functions
# ============================================================================

def make_severity_linked_sim(seed=0):
    """
    Create simulation where care-seeking is linked to severity (anemia)

    Anemic women are more likely to seek care (default: care_behavior['anemic'] = 1).
    This represents realistic care-seeking where sicker women access care more.
    """
    mens = Menstruation()
    edu = Education()
    cascade = HMBCascade(
        pars=dict(
            year=2025,
            time_to_assess=ss.months(3),
            # Default care_behavior has anemic=1 (increases care-seeking)
        )
    )
    care_analyzer = track_care_seeking()
    tx_eff_analyzer = track_tx_eff()
    tx_dur_analyzer = track_tx_dur()
    hmb_anemia_analyzer = track_hmb_anemia()
    cascade_analyzer = track_cascade()
    anemia_dur_analyzer = track_anemia_duration()

    sim = fp.Sim(
        start=2020,
        stop=2030,
        n_agents=5000,
        total_pop=55_000_000,
        location='kenya',
        education_module=edu,
        connectors=[mens],
        interventions=[cascade],
        analyzers=[care_analyzer, tx_eff_analyzer, tx_dur_analyzer, hmb_anemia_analyzer, cascade_analyzer, anemia_dur_analyzer],
        rand_seed=seed,
        verbose=0,
    )
    return sim


def make_severity_unlinked_sim(seed=0):
    """
    Create simulation where care-seeking is NOT linked to severity

    Anemic women are no more likely to seek care than non-anemic (care_behavior['anemic'] = 0).
    This represents a scenario where care access is independent of severity/need.
    """
    mens = Menstruation()
    edu = Education()

    # Create custom care_behavior with anemic=0
    care_behavior = sc.objdict(
        base=0.5,
        anemic=0,  # NO effect of anemia on care-seeking
        pain=0.25,
    )

    cascade = HMBCascade(
        pars=dict(
            year=2025,
            time_to_assess=ss.months(3),
            care_behavior=care_behavior,  # Override default
        )
    )
    care_analyzer = track_care_seeking()
    tx_eff_analyzer = track_tx_eff()
    tx_dur_analyzer = track_tx_dur()
    hmb_anemia_analyzer = track_hmb_anemia()
    cascade_analyzer = track_cascade()
    anemia_dur_analyzer = track_anemia_duration()

    sim = fp.Sim(
        start=2020,
        stop=2030,
        n_agents=5000,
        total_pop=55_000_000,
        location='kenya',
        education_module=edu,
        connectors=[mens],
        interventions=[cascade],
        analyzers=[care_analyzer, tx_eff_analyzer, tx_dur_analyzer, hmb_anemia_analyzer, cascade_analyzer, anemia_dur_analyzer],
        rand_seed=seed,
        verbose=0,
    )
    return sim


# ============================================================================
# Simulation execution
# ============================================================================

def run_care_sensitivity(n_runs=10, save_results=True):
    """
    Run severity-linked vs. severity-unlinked care-seeking scenarios

    Args:
        n_runs: Number of runs per scenario (default: 10)
        save_results: Whether to save results to disk (default: True)

    Returns:
        Dictionary with 'severity_linked' and 'severity_unlinked' MultiSim objects
    """
    print(f'Running care-seeking sensitivity analysis with {n_runs} runs per scenario...\n')

    # Severity-linked care-seeking (anemic=1, default)
    print('Running severity-linked scenario (anemic women more likely to seek care)...')
    linked_sims = [make_severity_linked_sim(seed=i) for i in range(n_runs)]
    msim_linked = ss.MultiSim(linked_sims)
    msim_linked.run()

    # Severity-unlinked care-seeking (anemic=0)
    print('Running severity-unlinked scenario (anemia does not affect care-seeking)...')
    unlinked_sims = [make_severity_unlinked_sim(seed=i) for i in range(n_runs)]
    msim_unlinked = ss.MultiSim(unlinked_sims)
    msim_unlinked.run()

    # Save results
    if save_results:
        print('Saving results...')
        sc.path('results').mkdir(exist_ok=True)
        sc.saveobj('results/severity_linked_msim.obj', msim_linked)
        sc.saveobj('results/severity_unlinked_msim.obj', msim_unlinked)
        print('Results saved to results/ directory')

    print('Done!')

    return {
        'severity_linked': msim_linked,
        'severity_unlinked': msim_unlinked,
    }


# ============================================================================
# Plotting functions
# ============================================================================

def plot_cascade_comparison(msim_linked, msim_unlinked, save_dir='figures'):
    """
    Compare cascade progression between severity-linked and unlinked scenarios

    Shows Panel A style plot (cascade progression by population) for both scenarios
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Cascade progression by care-seeking assumption', fontsize=16)

    for idx, (msim, title) in enumerate([
        (msim_linked, 'Severity-linked care-seeking\n(anemic women seek care more)'),
        (msim_unlinked, 'Severity-unlinked care-seeking\n(anemia does not affect care-seeking)')
    ]):
        ax = axes[idx]
        sim = msim.sims[0]  # Use first sim for visualization

        # Get states
        menstruation = sim.people.menstruation
        anemic = menstruation.anemic
        hmb = menstruation.hmb
        menstruating = menstruation.menstruating

        # Get cascade intervention
        cascade_intv = sim.interventions.hmb_cascade
        n_treatments = (
            np.array(cascade_intv.tried_nsaid, dtype=int) +
            np.array(cascade_intv.tried_txa, dtype=int) +
            np.array(cascade_intv.tried_pill, dtype=int) +
            np.array(cascade_intv.tried_hiud, dtype=int)
        )

        # Define populations
        pop_all = menstruating
        pop_anemic = anemic & menstruating
        pop_hmb_no_anemia = hmb & ~anemic & menstruating

        # Calculate distributions
        def get_cascade_dist(population):
            dist = []
            total = np.count_nonzero(population)
            if total == 0:
                return [0, 0, 0, 0, 0]
            for n in range(5):
                count = np.count_nonzero((n_treatments == n) & population)
                dist.append(100 * count / total)
            return dist

        dist_all = get_cascade_dist(pop_all)
        dist_anemic = get_cascade_dist(pop_anemic)
        dist_hmb_no_anemia = get_cascade_dist(pop_hmb_no_anemia)

        # Plot grouped bars
        treatments_tried = np.arange(5)
        width = 0.25
        x = treatments_tried

        ax.bar(x - width, dist_all, width, label='All women', color='steelblue', alpha=0.7)
        ax.bar(x, dist_anemic, width, label='Anemic', color='#d62728', alpha=0.7)
        ax.bar(x + width, dist_hmb_no_anemia, width, label='HMB, no anemia', color='#ff7f0e', alpha=0.7)

        ax.set_xlabel('Number of treatments tried')
        ax.set_ylabel('Percentage of population (%)')
        ax.set_title(title)
        ax.set_xticks(treatments_tried)
        ax.set_xticklabels(['0', '1', '2', '3', '4'])
        ax.legend(frameon=False, loc='upper right', fontsize=9)
        ax.set_ylim(0, max(max(dist_all), max(dist_anemic), max(dist_hmb_no_anemia)) * 1.15)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filename = f'{save_dir}/care_sensitivity_cascade.png'
    sc.savefig(filename, dpi=150)
    print(f'Saved cascade comparison to {filename}')

    return fig


def plot_impact_comparison(msim_linked, msim_unlinked, save_dir='figures'):
    """
    Compare intervention impact between severity-linked and unlinked scenarios

    Shows anemia burden reduction for both scenarios (similar to intervention_impact.png)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Intervention impact by care-seeking assumption', fontsize=16)

    # Get time vector from first sim
    tvec = msim_linked.sims[0].timevec
    years = np.array([t.year + (t.month - 1) / 12 for t in tvec])

    for idx, (msim, title) in enumerate([
        (msim_linked, 'Severity-linked (anemic=1)'),
        (msim_unlinked, 'Severity-unlinked (anemic=0)')
    ]):
        ax = axes[idx]

        # Extract HMB-related anemia cases
        hmb_anemia_cases = []
        total_anemia_cases = []

        for sim in msim.sims:
            hmb_anemia_cases.append(sim.results.track_hmb_anemia['n_anemia_with_hmb'])
            total_anemia_cases.append(sim.results.track_hmb_anemia['n_anemia_total'])

        # Calculate statistics
        hmb_array = np.array(hmb_anemia_cases)
        total_array = np.array(total_anemia_cases)

        hmb_mean = hmb_array.mean(axis=0)
        hmb_std = hmb_array.std(axis=0)
        total_mean = total_array.mean(axis=0)
        total_std = total_array.std(axis=0)

        # Plot total anemia
        ax.plot(years, total_mean, color='#ff7f0e', linewidth=2, label='Total anemia', linestyle='-')
        ax.fill_between(years, total_mean - total_std, total_mean + total_std,
                        color='#ff7f0e', alpha=0.15)

        # Plot HMB-related anemia
        ax.plot(years, hmb_mean, color='#d62728', linewidth=2, label='Anemia with HMB')
        ax.fill_between(years, hmb_mean - hmb_std, hmb_mean + hmb_std,
                        color='#d62728', alpha=0.2)

        # Add intervention start line
        ax.axvline(x=2025, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Intervention start')

        ax.set_xlabel('Year')
        ax.set_ylabel('Number of anemia cases')
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        sc.SIticks(ax=ax)
        ax.legend(frameon=False, loc='best')
        ax.grid(alpha=0.3)

        # Print summary stats
        print(f'\n{title}:')
        print(f'  Final year total anemia:           {total_mean[-1]/1e6:.2f}m ± {total_std[-1]/1e6:.2f}m')
        print(f'  Final year HMB-related anemia:     {hmb_mean[-1]/1e6:.2f}m ± {hmb_std[-1]/1e6:.2f}m')

        # Calculate reduction from pre-intervention (2024) to end (2029)
        pre_intv_idx = np.argmin(np.abs(years - 2024.5))
        hmb_reduction = hmb_mean[pre_intv_idx] - hmb_mean[-1]
        hmb_pct_reduction = 100 * hmb_reduction / hmb_mean[pre_intv_idx]
        print(f'  HMB-anemia reduction (2024→2030):  {hmb_reduction/1e6:.2f}m ({hmb_pct_reduction:.1f}%)')

    plt.tight_layout()
    filename = f'{save_dir}/care_sensitivity_impact.png'
    sc.savefig(filename, dpi=150)
    print(f'\nSaved impact comparison to {filename}')

    return fig


def plot_care_seeking_rates(msim_linked, msim_unlinked, save_dir='figures'):
    """
    Compare care-seeking rates over time between scenarios

    Shows how care-seeking rates differ by assumption, with stratification by anemia status
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Care-seeking rates by scenario', fontsize=16)

    # Get time vector
    tvec = msim_linked.sims[0].timevec
    years = np.array([t.year + (t.month - 1) / 12 for t in tvec])

    # Plot 1: Overall care-seeking rates
    ax = axes[0]
    for msim, label, color in [
        (msim_linked, 'Severity-linked (anemic=1)', 'steelblue'),
        (msim_unlinked, 'Severity-unlinked (anemic=0)', 'orange')
    ]:
        # Extract care-seeking rates
        care_rates = []
        for sim in msim.sims:
            care_rates.append(sim.results.track_care_seeking['care_seeking_prev'])

        care_array = np.array(care_rates)
        care_mean = care_array.mean(axis=0) * 100  # Convert to percentage
        care_std = care_array.std(axis=0) * 100

        ax.plot(years, care_mean, linewidth=2, label=label, color=color)
        ax.fill_between(years, care_mean - care_std, care_mean + care_std,
                        color=color, alpha=0.2)

    ax.axvline(x=2025, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Intervention start')
    ax.set_xlabel('Year')
    ax.set_ylabel('Care-seeking rate (%)')
    ax.set_title('Overall care-seeking rates')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    # Plot 2: Care-seeking rates stratified by anemia status
    ax = axes[1]
    for msim, label, color in [
        (msim_linked, 'Severity-linked', 'steelblue'),
        (msim_unlinked, 'Severity-unlinked', 'orange')
    ]:
        # Extract care-seeking rates by anemia status
        anemic_rates = []
        not_anemic_rates = []
        for sim in msim.sims:
            anemic_rates.append(sim.results.track_care_seeking['care_seeking_anemic'])
            not_anemic_rates.append(sim.results.track_care_seeking['care_seeking_not_anemic'])

        anemic_array = np.array(anemic_rates)
        not_anemic_array = np.array(not_anemic_rates)

        anemic_mean = anemic_array.mean(axis=0) * 100
        not_anemic_mean = not_anemic_array.mean(axis=0) * 100

        # Plot with different line styles for anemic vs not anemic
        ax.plot(years, anemic_mean, linewidth=2, label=f'{label} (anemic)',
                color=color, linestyle='-', alpha=0.8)
        ax.plot(years, not_anemic_mean, linewidth=2, label=f'{label} (not anemic)',
                color=color, linestyle='--', alpha=0.6)

    ax.axvline(x=2025, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Care-seeking rate (%)')
    ax.set_title('Care-seeking rates by anemia status')
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    filename = f'{save_dir}/care_sensitivity_rates.png'
    sc.savefig(filename, dpi=150)
    print(f'Saved care-seeking rates comparison to {filename}')

    return fig


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    # Configuration
    n_runs = 10

    # Check if results already exist
    try:
        print('Checking for existing results...')
        msim_linked = sc.loadobj('results/severity_linked_msim.obj')
        msim_unlinked = sc.loadobj('results/severity_unlinked_msim.obj')
        print('Found existing results! Loading from disk...\n')
    except:
        print('No existing results found. Running new simulations...\n')
        results = run_care_sensitivity(n_runs=n_runs, save_results=True)
        msim_linked = results['severity_linked']
        msim_unlinked = results['severity_unlinked']

    # Generate plots
    print('\nGenerating plots...')
    sc.path('figures').mkdir(exist_ok=True)

    plot_cascade_comparison(msim_linked, msim_unlinked)
    plot_impact_comparison(msim_linked, msim_unlinked)
    plot_care_seeking_rates(msim_linked, msim_unlinked)

    print('\nAll done! Figures saved to figures/ directory')
