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
from analyzers import track_care_seeking, track_tx_eff, track_tx_dur, track_hmb_anemia, track_cascade


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
            year=2020,
            time_to_assess=ss.months(3),  # Assess treatment effectiveness after 3 months
        )
    )
    care_analyzer = track_care_seeking()
    tx_eff_analyzer = track_tx_eff()
    tx_dur_analyzer = track_tx_dur()
    hmb_anemia_analyzer = track_hmb_anemia()
    cascade_analyzer = track_cascade()

    sim = fp.Sim(
        start=2020,
        stop=2030,
        n_agents=5000,
        total_pop=55_000_000,  # Kenya's population for scaling
        location='kenya',
        education_module=edu,
        connectors=[mens],
        interventions=[cascade],
        analyzers=[care_analyzer, tx_eff_analyzer, tx_dur_analyzer, hmb_anemia_analyzer, cascade_analyzer],
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

def plot_baseline_characteristics(msim, save_dir='figures'):
    """
    Plot key baseline characteristics from simulation

    Args:
        msim: MultiSim object with baseline results
        save_dir: Directory to save figures (default: 'figures')
    """
    sc.options(dpi=150)
    sc.path(save_dir).mkdir(exist_ok=True)

    # Extract results
    sim = msim.sims[0]
    tvec = sim.timevec
    years = np.array([t.year + (t.month - 1) / 12 for t in tvec])

    # Get mean and std across all runs
    def get_stats(result_name):
        data = np.array([s.results.menstruation[result_name] for s in msim.sims])
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        return mean, std

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Baseline simulation characteristics (no intervention)', fontsize=16, y=0.995)

    # 1. HMB prevalence
    ax = axes[0, 0]
    mean, std = get_stats('hmb_prev')
    ax.plot(years, mean, color='#d62728', linewidth=2, label='Mean')
    ax.fill_between(years, mean - std, mean + std, color='#d62728', alpha=0.3, label='±1 SD')
    ax.set_xlabel('Year')
    ax.set_ylabel('Prevalence')
    ax.set_title('HMB prevalence')
    ax.set_ylim([0, 1])
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    # 2. Anemia prevalence
    ax = axes[0, 1]
    mean, std = get_stats('anemic_prev')
    ax.plot(years, mean, color='#ff7f0e', linewidth=2, label='Mean')
    ax.fill_between(years, mean - std, mean + std, color='#ff7f0e', alpha=0.3, label='±1 SD')
    ax.set_xlabel('Year')
    ax.set_ylabel('Prevalence')
    ax.set_title('Anemia prevalence')
    ax.set_ylim([0, 1])
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    # 3. Cumulative anemia cases
    ax = axes[0, 2]
    mean, std = get_stats('n_anemia')
    ax.plot(years, mean, color='#2ca02c', linewidth=2, label='Mean')
    ax.fill_between(years, mean - std, mean + std, color='#2ca02c', alpha=0.3, label='±1 SD')
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative cases')
    ax.set_title('Cumulative anemia cases')
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    # 4. Menstrual pain prevalence
    ax = axes[1, 0]
    mean, std = get_stats('pain_prev')
    ax.plot(years, mean, color='#9467bd', linewidth=2, label='Mean')
    ax.fill_between(years, mean - std, mean + std, color='#9467bd', alpha=0.3, label='±1 SD')
    ax.set_xlabel('Year')
    ax.set_ylabel('Prevalence')
    ax.set_title('Menstrual pain prevalence')
    ax.set_ylim([0, 1])
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    # 5. Poor menstrual hygiene prevalence
    ax = axes[1, 1]
    mean, std = get_stats('poor_mh_prev')
    ax.plot(years, mean, color='#8c564b', linewidth=2, label='Mean')
    ax.fill_between(years, mean - std, mean + std, color='#8c564b', alpha=0.3, label='±1 SD')
    ax.set_xlabel('Year')
    ax.set_ylabel('Prevalence')
    ax.set_title('Poor menstrual hygiene prevalence')
    ax.set_ylim([0, 1])
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    # 6. Hysterectomy prevalence
    ax = axes[1, 2]
    mean, std = get_stats('hyst_prev')
    ax.plot(years, mean, color='#e377c2', linewidth=2, label='Mean')
    ax.fill_between(years, mean - std, mean + std, color='#e377c2', alpha=0.3, label='±1 SD')
    ax.set_xlabel('Year')
    ax.set_ylabel('Prevalence')
    ax.set_title('Hysterectomy prevalence')
    ax.set_ylim([0, 0.1])
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    filename = f'{save_dir}/baseline_characteristics.png'
    sc.savefig(filename, dpi=150)
    print(f'Saved baseline characteristics figure to {filename}')

    return fig


def plot_intervention_impact(msim_base, msim_intv, save_dir='figures'):
    """
    Compare baseline and intervention scenarios to show intervention impact on anemia cases

    Shows:
    1. Anemia cases among HMB women in baseline vs intervention
    2. Averted anemia cases (difference between scenarios)
    3. Percent reduction over time

    Args:
        msim_base: MultiSim object with baseline results
        msim_intv: MultiSim object with intervention results
        save_dir: Directory to save figures (default: 'figures')
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Intervention impact on anemia among HMB women', fontsize=16)

    # Get time vector
    tvec = msim_base.sims[0].timevec
    years = np.array([t.year + (t.month - 1) / 12 for t in tvec])

    # Extract anemia cases among HMB women
    base_cases = []
    intv_cases = []

    for sim in msim_base.sims:
        base_cases.append(sim.results.track_hmb_anemia['n_anemia_with_hmb'])

    for sim in msim_intv.sims:
        intv_cases.append(sim.results.track_hmb_anemia['n_anemia_with_hmb'])

    # Calculate statistics
    base_array = np.array(base_cases)
    intv_array = np.array(intv_cases)
    averted_array = base_array - intv_array

    base_mean = base_array.mean(axis=0)
    base_std = base_array.std(axis=0)
    intv_mean = intv_array.mean(axis=0)
    intv_std = intv_array.std(axis=0)
    averted_mean = averted_array.mean(axis=0)
    averted_std = averted_array.std(axis=0)

    # Plot 1: Baseline vs intervention
    ax = axes[0]
    ax.plot(years, base_mean, color='#d62728', linewidth=2, label='Baseline', linestyle='--')
    ax.fill_between(years, base_mean - base_std, base_mean + base_std, color='#d62728', alpha=0.2)
    ax.plot(years, intv_mean, color='#2ca02c', linewidth=2, label='Full package')
    ax.fill_between(years, intv_mean - intv_std, intv_mean + intv_std, color='#2ca02c', alpha=0.2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of anemia cases')
    ax.set_title('Anemia among HMB women')
    ax.set_ylim(bottom=0)
    sc.SIticks(ax=ax)
    ax.legend(frameon=False, loc='best')
    ax.grid(alpha=0.3)

    # Plot 2: Averted cases over time
    ax = axes[1]
    ax.plot(years, averted_mean, color='#2ca02c', linewidth=2)
    ax.fill_between(years, averted_mean - averted_std, averted_mean + averted_std,
                     color='#2ca02c', alpha=0.3)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of averted cases')
    ax.set_title('Anemia cases averted by intervention')
    ax.set_ylim(bottom=0)
    sc.SIticks(ax=ax)
    ax.grid(alpha=0.3)

    # Plot 3: Percent reduction over time
    ax = axes[2]
    valid_mask = base_mean > 0
    percent_reduction = np.zeros_like(averted_mean)
    percent_reduction[valid_mask] = 100 * averted_mean[valid_mask] / base_mean[valid_mask]

    ax.plot(years, percent_reduction, color='#2ca02c', linewidth=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Percent reduction (%)')
    ax.set_title('Percent reduction in anemia cases')
    ax.set_ylim(bottom=0, top=min(100, np.max(percent_reduction) * 1.1))
    ax.grid(alpha=0.3)

    plt.tight_layout()
    filename = f'{save_dir}/intervention_impact.png'
    sc.savefig(filename, dpi=150)
    print(f'Saved intervention impact figure to {filename}')

    # Print summary statistics
    print('\n' + '='*60)
    print('INTERVENTION IMPACT SUMMARY')
    print('='*60)
    print(f'Final year baseline anemia cases:    {base_mean[-1]/1e6:.2f}m ± {base_std[-1]/1e6:.2f}m')
    print(f'Final year with full package:        {intv_mean[-1]/1e6:.2f}m ± {intv_std[-1]/1e6:.2f}m')
    print(f'  Averted anemia cases:              {averted_mean[-1]/1e6:.2f}m ± {averted_std[-1]/1e6:.2f}m')
    print(f'  Percent reduction:                 {percent_reduction[-1]:.1f}%')
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
    for analyzer in intervention_sim.analyzers:
        if hasattr(analyzer, 'results') and 'offered_nsaid' in analyzer.results:
            cascade = analyzer
            break

    if cascade is None:
        print('Warning: Could not find cascade analyzer in intervention simulation')
        return None

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Treatment cascade analysis', fontsize=16, y=0.995)

    final_ti = -1

    # 1. Number of treatments tried
    ax = axes[0, 0]
    n_tried = {
        0: cascade.results.n_tried_0_tx[final_ti],
        1: cascade.results.n_tried_1_tx[final_ti],
        2: cascade.results.n_tried_2_tx[final_ti],
        3: cascade.results.n_tried_3_tx[final_ti],
        4: cascade.results.n_tried_4_tx[final_ti],
    }
    prop_tried = {
        0: cascade.results.prop_tried_0_tx[final_ti],
        1: cascade.results.prop_tried_1_tx[final_ti],
        2: cascade.results.prop_tried_2_tx[final_ti],
        3: cascade.results.prop_tried_3_tx[final_ti],
        4: cascade.results.prop_tried_4_tx[final_ti],
    }

    treatments_tried = list(n_tried.keys())
    n_values = list(n_tried.values())

    ax.bar(treatments_tried, n_values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Number of treatments tried')
    ax.set_ylabel('Number of women')
    ax.set_title('Cascade depth distribution')
    ax.grid(axis='y', alpha=0.3)

    # Add percentages on bars
    for i, v in enumerate(n_values):
        if v > 0:
            pct = prop_tried[i] * 100
            ax.text(i, v, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 2. Cascade dropoffs
    ax = axes[0, 1]
    treatments = ['nsaid', 'txa', 'pill', 'hiud']
    treatment_labels = ['NSAID', 'TXA', 'Pill', 'hIUD']

    offered = [cascade.results[f'offered_{tx}'][final_ti] for tx in treatments]
    accepted = [cascade.results[f'accepted_{tx}'][final_ti] for tx in treatments]

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

    # 4. Anemia prevalence by cascade depth
    ax = axes[1, 1]
    cascade_depths = [0, 1, 2, 3, 4]
    anemia_prev = [
        cascade.results.anemia_tried_0[final_ti] * 100,
        cascade.results.anemia_tried_1[final_ti] * 100,
        cascade.results.anemia_tried_2[final_ti] * 100,
        cascade.results.anemia_tried_3[final_ti] * 100,
        cascade.results.anemia_tried_4[final_ti] * 100,
    ]

    ax.plot(cascade_depths, anemia_prev, marker='o', linewidth=2, markersize=10, color='darkred')
    ax.set_xlabel('Number of treatments tried')
    ax.set_ylabel('Anemia prevalence (%)')
    ax.set_title('Anemia by cascade depth')
    ax.grid(alpha=0.3)

    # Add value labels
    for i, v in enumerate(anemia_prev):
        if v > 0:
            ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

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

    # Run simulations
    results = run_cascade_comparison(n_runs=n_runs, save_results=True)
    msim_base = results['baseline']
    msim_intv = results['intervention']

    # Generate plots
    print('\nGenerating plots...')
    plot_baseline_characteristics(msim_base)
    plot_intervention_impact(msim_base, msim_intv)

    # Plot cascade from first intervention sim
    plot_cascade_analysis(msim_intv.sims[0])

    print('\nAll done! Figures saved to figures/ directory')
