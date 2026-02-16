"""
Run and plot cascade analysis for HMB intervention

Compares baseline (no intervention) to full treatment package (NSAID→TXA→Pill→hIUD).
Creates simulations, runs them, and generates comprehensive plots of intervention impact.
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

def make_base_sim(seed=0):
    """
    Create baseline simulation without intervention

    Returns a simulation with HMB dynamics but no care pathway intervention
    """
    mens = Menstruation()
    edu = Education()
    care_analyzer = track_care_seeking()
    hmb_anemia_analyzer = track_hmb_anemia()

    sim = fp.Sim(
        start=2020,
        stop=2030,
        n_agents=5000,
        total_pop=55_000_000,  # Kenya's population for scaling
        location='kenya',
        education_module=edu,
        connectors=[mens],
        analyzers=[care_analyzer, hmb_anemia_analyzer],
        rand_seed=seed,
        verbose=0,
    )
    return sim


def make_intervention_sim(seed=0):
    """
    Create simulation with HMB care pathway intervention

    Returns a simulation with full care pathway including NSAID, TXA, Pill, and hIUD
    """
    mens = Menstruation()
    edu = Education()
    cascade = HMBCascade(
        pars=dict(
            year=2025,
            time_to_assess=ss.months(3),  # Assess treatment effectiveness after 3 months
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
        total_pop=55_000_000,  # Kenya's population for scaling
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

def run_cascade_comparison(n_runs=10, save_results=True):
    """
    Run baseline and intervention as MultiSim for statistical robustness

    Args:
        n_runs: Number of runs for each scenario (default: 10)
        save_results: Whether to save results to disk (default: True)

    Returns:
        Dictionary with 'baseline' and 'intervention' MultiSim objects
    """
    print(f'Running {n_runs} simulations for each scenario...')

    # Create baseline MultiSim
    print('Running baseline scenario...')
    base_sims = [make_base_sim(seed=i) for i in range(n_runs)]
    msim_base = ss.MultiSim(base_sims)
    msim_base.run()

    # Create intervention MultiSim
    print('Running full intervention scenario (NSAID→TXA→Pill→hIUD)...')
    intervention_sims = [make_intervention_sim(seed=i) for i in range(n_runs)]
    msim_intervention = ss.MultiSim(intervention_sims)
    msim_intervention.run()

    # Save results
    if save_results:
        print('Saving results...')
        sc.path('results').mkdir(exist_ok=True)
        sc.saveobj('results/baseline_msim.obj', msim_base)
        sc.saveobj('results/intervention_msim.obj', msim_intervention)
        print('Results saved to results/ directory')

    print('Done!')

    return {
        'baseline': msim_base,
        'intervention': msim_intervention,
    }


# ============================================================================
# Plotting functions
# ============================================================================

def plot_intervention_impact(msim_base, msim_intv, msim_iud=None, save_dir='figures'):
    """
    Compare baseline and intervention scenarios to show intervention impact on anemia cases.

    Shows:
    1. Total anemia burden and HMB-related anemia in baseline vs intervention(s)
    2. Averted anemia cases (difference between scenarios)
    3. Percent reduction over time

    Args:
        msim_base: MultiSim object with baseline results
        msim_intv: MultiSim object with full intervention results
        msim_iud: MultiSim object with IUD-only intervention results (optional)
        save_dir: Directory to save figures (default: 'figures')
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Intervention impact on anemia', fontsize=16)

    # Get time vector
    tvec = msim_base.sims[0].timevec
    years = np.array([t.year + (t.month - 1) / 12 for t in tvec])

    # Extract anemia cases among HMB women for each scenario
    base_hmb_cases = []
    intv_hmb_cases = []
    iud_hmb_cases = []

    # Extract total anemia cases for each scenario
    base_total_cases = []
    intv_total_cases = []
    iud_total_cases = []

    for sim in msim_base.sims:
        base_hmb_cases.append(sim.results.track_hmb_anemia['n_anemia_with_hmb'])
        base_total_cases.append(sim.results.track_hmb_anemia['n_anemia_total'])

    for sim in msim_intv.sims:
        intv_hmb_cases.append(sim.results.track_hmb_anemia['n_anemia_with_hmb'])
        intv_total_cases.append(sim.results.track_hmb_anemia['n_anemia_total'])

    if msim_iud is not None:
        for sim in msim_iud.sims:
            iud_hmb_cases.append(sim.results.track_hmb_anemia['n_anemia_with_hmb'])
            iud_total_cases.append(sim.results.track_hmb_anemia['n_anemia_total'])

    # Calculate statistics for HMB-related anemia
    base_hmb_array = np.array(base_hmb_cases)
    intv_hmb_array = np.array(intv_hmb_cases)
    averted_array = base_hmb_array - intv_hmb_array

    base_hmb_mean = base_hmb_array.mean(axis=0)
    base_hmb_std = base_hmb_array.std(axis=0)
    intv_hmb_mean = intv_hmb_array.mean(axis=0)
    intv_hmb_std = intv_hmb_array.std(axis=0)
    averted_mean = averted_array.mean(axis=0)
    averted_std = averted_array.std(axis=0)

    # Calculate statistics for total anemia
    base_total_array = np.array(base_total_cases)
    intv_total_array = np.array(intv_total_cases)

    base_total_mean = base_total_array.mean(axis=0)
    base_total_std = base_total_array.std(axis=0)
    intv_total_mean = intv_total_array.mean(axis=0)
    intv_total_std = intv_total_array.std(axis=0)

    if msim_iud is not None:
        iud_hmb_array = np.array(iud_hmb_cases)
        iud_averted_array = base_hmb_array - iud_hmb_array
        iud_hmb_mean = iud_hmb_array.mean(axis=0)
        iud_hmb_std = iud_hmb_array.std(axis=0)
        iud_averted_mean = iud_averted_array.mean(axis=0)
        iud_averted_std = iud_averted_array.std(axis=0)

        iud_total_array = np.array(iud_total_cases)
        iud_total_mean = iud_total_array.mean(axis=0)
        iud_total_std = iud_total_array.std(axis=0)

    # Plot 1: Total anemia and HMB-related anemia for baseline vs intervention(s)
    ax = axes[0]

    # Plot total anemia burden (baseline)
    ax.plot(years, base_total_mean, color='#ff7f0e', linewidth=2, label='Total anemia (baseline)', linestyle=':')
    ax.fill_between(years, base_total_mean - base_total_std, base_total_mean + base_total_std,
                     color='#ff7f0e', alpha=0.15)

    # Plot total anemia burden (intervention)
    ax.plot(years, intv_total_mean, color='#ff7f0e', linewidth=2, label='Total anemia (full package)', linestyle='-')
    ax.fill_between(years, intv_total_mean - intv_total_std, intv_total_mean + intv_total_std,
                     color='#ff7f0e', alpha=0.15)

    # Plot HMB-related anemia (baseline)
    ax.plot(years, base_hmb_mean, color='#d62728', linewidth=2, label='Anemia with HMB (baseline)', linestyle='--')
    ax.fill_between(years, base_hmb_mean - base_hmb_std, base_hmb_mean + base_hmb_std, color='#d62728', alpha=0.2)

    # Plot HMB-related anemia (intervention)
    ax.plot(years, intv_hmb_mean, color='#2ca02c', linewidth=2, label='Anemia with HMB (full package)')
    ax.fill_between(years, intv_hmb_mean - intv_hmb_std, intv_hmb_mean + intv_hmb_std, color='#2ca02c', alpha=0.2)

    if msim_iud is not None:
        ax.plot(years, iud_total_mean, color='#ff7f0e', linewidth=1.5, label='Total anemia (IUD only)', linestyle='-', alpha=0.7)
        ax.plot(years, iud_hmb_mean, color='#9467bd', linewidth=2, label='Anemia with HMB (IUD only)')
        ax.fill_between(years, iud_hmb_mean - iud_hmb_std, iud_hmb_mean + iud_hmb_std, color='#9467bd', alpha=0.2)

    ax.set_xlabel('Year')
    ax.set_ylabel('Number of anemia cases')
    ax.set_title('Total and HMB-related anemia burden')
    ax.set_ylim(bottom=0)
    sc.SIticks(ax=ax)
    ax.legend(frameon=False, loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    # Plot 2: Averted HMB-related anemia cases over time
    ax = axes[1]
    ax.plot(years, averted_mean, color='#2ca02c', linewidth=2, label='Full package')
    ax.fill_between(years, averted_mean - averted_std, averted_mean + averted_std,
                     color='#2ca02c', alpha=0.3)
    if msim_iud is not None:
        ax.plot(years, iud_averted_mean, color='#9467bd', linewidth=2, label='IUD only')
        ax.fill_between(years, iud_averted_mean - iud_averted_std, iud_averted_mean + iud_averted_std,
                         color='#9467bd', alpha=0.3)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of averted cases')
    ax.set_title('Anemia cases averted by intervention\n(among HMB women)')
    ax.set_ylim(bottom=0)
    sc.SIticks(ax=ax)
    ax.legend(frameon=False, loc='best')
    ax.grid(alpha=0.3)

    # Plot 3: Percent reduction in HMB-related anemia over time
    ax = axes[2]
    # Calculate percent reduction, avoiding division by zero
    valid_mask = base_hmb_mean > 0
    percent_reduction = np.zeros_like(averted_mean)
    percent_reduction[valid_mask] = 100 * averted_mean[valid_mask] / base_hmb_mean[valid_mask]

    ax.plot(years, percent_reduction, color='#2ca02c', linewidth=2, label='Full package')

    if msim_iud is not None:
        iud_percent_reduction = np.zeros_like(iud_averted_mean)
        iud_percent_reduction[valid_mask] = 100 * iud_averted_mean[valid_mask] / base_hmb_mean[valid_mask]
        ax.plot(years, iud_percent_reduction, color='#9467bd', linewidth=2, label='IUD only')

    ax.set_xlabel('Year')
    ax.set_ylabel('Percent reduction (%)')
    ax.set_title('Percent reduction in HMB-related anemia')
    ax.set_ylim(bottom=0, top=min(100, np.max(percent_reduction) * 1.1))
    ax.legend(frameon=False, loc='best')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    filename = f'{save_dir}/intervention_impact.png'
    sc.savefig(filename, dpi=150)
    print(f'Saved intervention impact figure to {filename}')

    # Print summary statistics
    print('\n' + '='*60)
    print('INTERVENTION IMPACT SUMMARY')
    print('='*60)
    print(f'Final year baseline:')
    print(f'  Total anemia cases:                {base_total_mean[-1]/1e6:.2f}m ± {base_total_std[-1]/1e6:.2f}m')
    print(f'  HMB-related anemia cases:          {base_hmb_mean[-1]/1e6:.2f}m ± {base_hmb_std[-1]/1e6:.2f}m')
    print(f'Final year with full package:')
    print(f'  Total anemia cases:                {intv_total_mean[-1]/1e6:.2f}m ± {intv_total_std[-1]/1e6:.2f}m')
    print(f'  HMB-related anemia cases:          {intv_hmb_mean[-1]/1e6:.2f}m ± {intv_hmb_std[-1]/1e6:.2f}m')
    print(f'  Averted HMB-related anemia:        {averted_mean[-1]/1e6:.2f}m ± {averted_std[-1]/1e6:.2f}m')
    print(f'  Percent reduction (HMB-related):   {percent_reduction[-1]:.1f}%')
    if msim_iud is not None:
        print(f'Final year with IUD only:')
        print(f'  Total anemia cases:                {iud_total_mean[-1]/1e6:.2f}m ± {iud_total_std[-1]/1e6:.2f}m')
        print(f'  HMB-related anemia cases:          {iud_hmb_mean[-1]/1e6:.2f}m ± {iud_hmb_std[-1]/1e6:.2f}m')
        print(f'  Averted HMB-related anemia:        {iud_averted_mean[-1]/1e6:.2f}m ± {iud_averted_std[-1]/1e6:.2f}m')
        print(f'  Percent reduction (HMB-related):   {iud_percent_reduction[-1]:.1f}%')
    print('='*60 + '\n')

    return fig


def plot_cascade_analysis(intervention_sim, save_dir='figures'):
    """
    Create visualization of treatment cascade progression

    Args:
        intervention_sim: Intervention simulation object (single sim from MultiSim)
        save_dir: Directory to save figures (default: 'figures')
    """
    sc.options(dpi=150)
    sc.path(save_dir).mkdir(exist_ok=True)

    # Get cascade analyzer (should be the last analyzer)
    cascade = None
    for analyzer in intervention_sim.analyzers.values():
        if hasattr(analyzer, 'results') and hasattr(analyzer.results, 'offered_nsaid'):
            cascade = analyzer
            break

    if cascade is None:
        print('Warning: Could not find cascade analyzer in intervention simulation')
        return None

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Treatment cascade analysis', fontsize=16, y=0.995)

    final_ti = -1

    # 1. Cascade progression by population
    ax = axes[0, 0]

    # Get states from simulation
    sim = intervention_sim
    menstruation = sim.people.menstruation
    anemic = menstruation.anemic
    hmb = menstruation.hmb
    menstruating = menstruation.menstruating

    # Get number of treatments tried per person
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

    # Calculate distributions for each population
    def get_cascade_dist(population):
        """Get distribution of treatments tried for a population"""
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
    ax.set_title('Cascade progression by population')
    ax.set_xticks(treatments_tried)
    ax.set_xticklabels(['0', '1', '2', '3', '4'])
    ax.legend(frameon=False, loc='upper right')
    ax.set_ylim(0, max(max(dist_all), max(dist_anemic), max(dist_hmb_no_anemia)) * 1.15)
    ax.grid(axis='y', alpha=0.3)

    # 2. Cascade dropoffs
    ax = axes[0, 1]
    treatments = ['nsaid', 'txa', 'pill', 'hiud']
    treatment_labels = ['NSAID', 'TXA', 'Pill', 'hIUD']

    offered = [getattr(cascade.results, f'offered_{tx}')[final_ti] for tx in treatments]
    accepted = [getattr(cascade.results, f'accepted_{tx}')[final_ti] for tx in treatments]

    x = np.arange(len(treatments))
    width = 0.35

    ax.bar(x - width/2, offered, width, label='Offered', alpha=0.7, color='lightcoral')
    ax.bar(x + width/2, accepted, width, label='Accepted', alpha=0.7, color='seagreen')

    ax.set_xlabel('Treatment')
    ax.set_ylabel('Number of women')
    ax.set_title('Treatment offers and acceptances')
    ax.set_xticks(x)
    ax.set_xticklabels(treatment_labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 3. Acceptance rates
    ax = axes[1, 0]
    acceptance_rates = [(accepted[i] / offered[i] * 100) if offered[i] > 0 else 0
                        for i in range(len(treatments))]
    colors = ['green' if r > 50 else 'orange' if r > 25 else 'red' for r in acceptance_rates]

    ax.bar(treatment_labels, acceptance_rates, color=colors, alpha=0.7)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='50% threshold')
    ax.set_xlabel('Treatment')
    ax.set_ylabel('Acceptance rate (%)')
    ax.set_title('Treatment acceptance rates')
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 4. Time with anemia by cascade depth
    ax = axes[1, 1]
    cascade_depths = [0, 1, 2, 3, 4]

    # Get anemia duration analyzer
    anemia_dur_analyzer = None
    for analyzer in intervention_sim.analyzers.values():
        if hasattr(analyzer, 'results') and hasattr(analyzer.results, 'mean_dur_anemia_tried_0'):
            anemia_dur_analyzer = analyzer
            break

    if anemia_dur_analyzer is not None:
        # Get average time with anemia at final timestep
        mean_durations = [
            anemia_dur_analyzer.results.mean_dur_anemia_tried_0[final_ti],
            anemia_dur_analyzer.results.mean_dur_anemia_tried_1[final_ti],
            anemia_dur_analyzer.results.mean_dur_anemia_tried_2[final_ti],
            anemia_dur_analyzer.results.mean_dur_anemia_tried_3[final_ti],
            anemia_dur_analyzer.results.mean_dur_anemia_tried_4[final_ti],
        ]

        ax.bar(cascade_depths, mean_durations, width=0.6, color='darkred', alpha=0.7)
        ax.set_xlabel('Number of treatments tried')
        ax.set_ylabel('Average time with anemia (months)')
        ax.set_title('Cumulative time with HMB-related anemia\nby cascade depth')
        ax.set_xticks(cascade_depths)
        ax.set_xticklabels(['0', '1', '2', '3', '4'])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, v in enumerate(mean_durations):
            if v > 0:
                ax.text(i, v + 0.2, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Anemia duration analyzer not found', ha='center', va='center',
                transform=ax.transAxes)

    plt.tight_layout()

    filename = f'{save_dir}/cascade_analysis.png'
    sc.savefig(filename, dpi=150)
    print(f'Saved cascade analysis figure to {filename}')

    return fig


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':
    # Configuration
    n_runs = 10  # Number of stochastic runs per scenario

    # Check if results already exist
    try:
        print('Checking for existing results...')
        msim_base = sc.loadobj('results/baseline_msim.obj')
        msim_intv = sc.loadobcj('results/intervention_msim.obj', )
        print('Found existing results! Loading from disk...\n')
    except:
        print('No existing results found. Running new simulations...\n')
        # Run simulations (this will take some time)
        results = run_cascade_comparison(n_runs=n_runs, save_results=True)
        msim_base = results['baseline']
        msim_intv = results['intervention']

    # Generate plots
    print('\nGenerating plots...')
    plot_intervention_impact(msim_base, msim_intv)

    # Plot cascade from first intervention sim
    plot_cascade_analysis(msim_intv.sims[0])

    print('\nAll done! Figures saved to figures/ directory')
