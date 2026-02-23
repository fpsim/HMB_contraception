"""
Component analysis for HMB care pathway

Isolates the impact of each treatment component (NSAID, TXA, Pill, hIUD)
on anemia reduction by running separate scenarios with only one treatment
available at a time.

This avoids selection bias issues that arise from comparing treatment seekers
to non-seekers within the same simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
import starsim as ss
import fpsim as fp

from menstruation import Menstruation
from education import Education
from interventions import make_component_sim as _make_component_sim
from analyzers import track_hmb_anemia


def make_component_sim(component, seed=0):
    """
    Create simulation with only one treatment component available.

    Uses the new modular architecture where each treatment is independent.

    Args:
        component: Treatment to offer ('nsaid', 'txa', 'pill', or 'hiud')
        seed: Random seed

    Returns:
        Simulation object with only the specified treatment available
    """
    # Use the factory function from hmb_cascade
    return _make_component_sim(component, seed=seed)


def make_baseline_sim(seed=0):
    """Create baseline simulation without any intervention."""
    mens = Menstruation()
    edu = Education()
    hmb_anemia_analyzer = track_hmb_anemia()

    sim = fp.Sim(
        start=2020,
        stop=2030,
        n_agents=5000,
        total_pop=55_000_000,
        location='kenya',
        education_module=edu,
        connectors=[mens],
        analyzers=[hmb_anemia_analyzer],
        rand_seed=seed,
        verbose=0,
    )

    return sim


def make_full_package_sim(seed=0):
    """Create simulation with full care package using new cascade architecture."""
    from interventions import make_cascade_sim

    # Use the factory function from interventions
    return make_cascade_sim(seed=seed)


def run_component_analysis(n_runs=10, save_results=True):
    """
    Run component analysis by simulating each treatment in isolation.

    Compares:
    - Baseline (no treatment)
    - NSAID only
    - TXA only
    - Pill only
    - hIUD only
    - Full package (all treatments)

    Args:
        n_runs: Number of stochastic runs per scenario
        save_results: Whether to save results to disk

    Returns:
        Dictionary with MultiSim objects for each scenario
    """
    print(f'Running component analysis with {n_runs} runs per scenario...\n')

    # Baseline
    print('Running baseline (no intervention)...')
    baseline_sims = [make_baseline_sim(seed=i) for i in range(n_runs)]
    msim_baseline = ss.MultiSim(baseline_sims)
    msim_baseline.run()

    # NSAID only
    print('Running NSAID-only scenario...')
    nsaid_sims = [make_component_sim('nsaid', seed=i) for i in range(n_runs)]
    msim_nsaid = ss.MultiSim(nsaid_sims)
    msim_nsaid.run()

    # TXA only
    print('Running TXA-only scenario...')
    txa_sims = [make_component_sim('txa', seed=i) for i in range(n_runs)]
    msim_txa = ss.MultiSim(txa_sims)
    msim_txa.run()

    # Pill only
    print('Running pill-only scenario...')
    pill_sims = [make_component_sim('pill', seed=i) for i in range(n_runs)]
    msim_pill = ss.MultiSim(pill_sims)
    msim_pill.run()

    # hIUD only
    print('Running hIUD-only scenario...')
    hiud_sims = [make_component_sim('hiud', seed=i) for i in range(n_runs)]
    msim_hiud = ss.MultiSim(hiud_sims)
    msim_hiud.run()

    # Full package
    print('Running full package scenario...')
    full_sims = [make_full_package_sim(seed=i) for i in range(n_runs)]
    msim_full = ss.MultiSim(full_sims)
    msim_full.run()

    results = {
        'baseline': msim_baseline,
        'nsaid': msim_nsaid,
        'txa': msim_txa,
        'pill': msim_pill,
        'hiud': msim_hiud,
        'full': msim_full,
    }

    # Save results
    if save_results:
        print('\nSaving component analysis results...')
        sc.path('results').mkdir(exist_ok=True)
        for name, msim in results.items():
            sc.saveobj(f'results/component_{name}_msim.obj', msim)
        print('Results saved to results/component_*.obj')

    print('\nComponent analysis complete!')
    return results


def calculate_component_impacts(results):
    """
    Calculate anemia cases averted by each component.

    Compares each component-only scenario to baseline to isolate
    the marginal impact of that specific treatment.

    Args:
        results: Dictionary with MultiSim results from run_component_analysis

    Returns:
        Dictionary with impact metrics for each component
    """
    final_ti = -1

    # Extract baseline anemia (among HMB women)
    baseline_anemia = []
    for sim in results['baseline'].sims:
        baseline_anemia.append(sim.results.track_hmb_anemia.n_anemia_with_hmb[final_ti])
    baseline_mean = np.mean(baseline_anemia)
    baseline_std = np.std(baseline_anemia)

    # Calculate impact for each component
    impacts = {}

    for component in ['nsaid', 'txa', 'pill', 'hiud', 'full']:
        # Extract anemia cases with this component
        component_anemia = []
        for sim in results[component].sims:
            component_anemia.append(sim.results.track_hmb_anemia.n_anemia_with_hmb[final_ti])

        component_mean = np.mean(component_anemia)
        component_std = np.std(component_anemia)

        # Calculate cases averted
        averted = baseline_mean - component_mean
        # Propagate uncertainty (assuming independence)
        averted_std = np.sqrt(baseline_std**2 + component_std**2)

        # Calculate percent reduction
        pct_reduction = (averted / baseline_mean * 100) if baseline_mean > 0 else 0

        impacts[component] = {
            'anemia_cases': component_mean,
            'anemia_std': component_std,
            'cases_averted': averted,
            'averted_std': averted_std,
            'pct_reduction': pct_reduction,
        }

    # Add baseline for reference
    impacts['baseline'] = {
        'anemia_cases': baseline_mean,
        'anemia_std': baseline_std,
        'cases_averted': 0,
        'averted_std': 0,
        'pct_reduction': 0,
    }

    return impacts


def print_component_summary(impacts):
    """Print formatted summary of component impacts."""
    print("\n" + "="*80)
    print("COMPONENT ANALYSIS - ANEMIA REDUCTION ATTRIBUTION")
    print("="*80)
    print(f"\nBaseline anemia (HMB women):  {impacts['baseline']['anemia_cases']/1e6:.2f}m " +
          f"± {impacts['baseline']['anemia_std']/1e6:.2f}m cases\n")

    print("Impact by component (each treatment offered alone):")
    print("-" * 80)

    for component in ['nsaid', 'txa', 'pill', 'hiud']:
        data = impacts[component]
        print(f"\n{component.upper()}:")
        print(f"  Anemia cases:      {data['anemia_cases']/1e6:.2f}m ± {data['anemia_std']/1e6:.2f}m")
        print(f"  Cases averted:     {data['cases_averted']/1e6:.2f}m ± {data['averted_std']/1e6:.2f}m")
        print(f"  Percent reduction: {data['pct_reduction']:.1f}%")

    print("\n" + "-" * 80)
    data = impacts['full']
    print(f"\nFULL PACKAGE (all treatments):")
    print(f"  Anemia cases:      {data['anemia_cases']/1e6:.2f}m ± {data['anemia_std']/1e6:.2f}m")
    print(f"  Cases averted:     {data['cases_averted']/1e6:.2f}m ± {data['averted_std']/1e6:.2f}m")
    print(f"  Percent reduction: {data['pct_reduction']:.1f}%")

    # Compare sum of individual components to full package
    sum_individual = sum(impacts[c]['cases_averted'] for c in ['nsaid', 'txa', 'pill', 'hiud'])
    full_averted = impacts['full']['cases_averted']

    print("\n" + "="*80)
    print("INTERACTION EFFECTS:")
    print("-" * 80)
    print(f"Sum of individual components: {sum_individual/1e6:.2f}m cases averted")
    print(f"Full package impact:          {full_averted/1e6:.2f}m cases averted")
    print(f"Difference:                   {(full_averted - sum_individual)/1e6:.2f}m")

    if sum_individual > 0:
        interaction_pct = ((full_averted - sum_individual) / sum_individual * 100)
        print(f"                              ({interaction_pct:.1f}% {'positive' if interaction_pct > 0 else 'negative'} interaction)")

    print("\nNote: Individual component impacts may sum to more than full package impact")
    print("because in the full package, some women try one treatment and succeed, while")
    print("in component-only scenarios, everyone eligible receives that specific treatment.")
    print("="*80 + "\n")


def plot_component_impacts(impacts, save_dir='figures'):
    """
    Create visualization of component-specific impacts.

    Args:
        impacts: Dictionary from calculate_component_impacts
        save_dir: Directory to save figure
    """
    sc.options(dpi=150)
    sc.path(save_dir).mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Component-specific impact on anemia reduction', fontsize=16)

    components = ['nsaid', 'txa', 'pill', 'hiud']
    component_labels = ['NSAID', 'TXA', 'Pill', 'hIUD']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Panel 1: Cases averted by component
    ax = axes[0]
    averted = [impacts[c]['cases_averted']/1e6 for c in components]
    averted_std = [impacts[c]['averted_std']/1e6 for c in components]

    bars = ax.bar(component_labels, averted, yerr=averted_std,
                   capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Anemia cases averted (millions)')
    ax.set_title('Cases averted by each component\n(when offered alone)')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val, std in zip(bars, averted, averted_std):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{val:.2f}m\n±{std:.2f}m',
                ha='center', va='bottom', fontsize=9)

    # Panel 2: Percent reduction
    ax = axes[1]
    pct_reduction = [impacts[c]['pct_reduction'] for c in components]

    bars = ax.bar(component_labels, pct_reduction, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Percent reduction (%)')
    ax.set_title('Percent reduction in anemia\n(relative to baseline)')
    ax.set_ylim([0, max(pct_reduction) * 1.2])
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, pct_reduction):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel 3: Comparison with full package
    ax = axes[2]

    # Show baseline, individual components, and full package
    scenarios = ['Baseline'] + component_labels + ['Full\npackage']
    anemia_cases = [impacts['baseline']['anemia_cases']/1e6] + \
                   [impacts[c]['anemia_cases']/1e6 for c in components] + \
                   [impacts['full']['anemia_cases']/1e6]
    anemia_stds = [impacts['baseline']['anemia_std']/1e6] + \
                  [impacts[c]['anemia_std']/1e6 for c in components] + \
                  [impacts['full']['anemia_std']/1e6]

    bar_colors = ['gray'] + colors + ['purple']

    bars = ax.bar(scenarios, anemia_cases, yerr=anemia_stds,
                   capsize=5, color=bar_colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Anemia cases (millions)')
    ax.set_title('Anemia among HMB women\n(final year)')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    filename = f'{save_dir}/component_impacts.png'
    sc.savefig(filename, dpi=150)
    print(f'Saved component impacts figure to {filename}')

    return fig


def load_component_results():
    """Load saved component analysis results."""
    results = {}
    for name in ['baseline', 'nsaid', 'txa', 'pill', 'hiud', 'full']:
        results[name] = sc.loadobj(f'results/component_{name}_msim.obj')
    return results


if __name__ == '__main__':
    # Run component analysis
    print('Starting component analysis...\n')
    results = run_component_analysis(n_runs=10, save_results=True)

    # Calculate impacts
    print('\nCalculating component impacts...')
    impacts = calculate_component_impacts(results)

    # Print summary
    print_component_summary(impacts)

    # Create visualization
    print('Creating visualization...')
    plot_component_impacts(impacts)

    print('\nDone!')
