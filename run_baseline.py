"""
Run and plot baseline HMB and anemia characteristics

Creates baseline simulations (no intervention) and generates plots showing:
- HMB and anemia prevalence over time
- Anemia prevalence stratified by HMB status
- Potentially avertable anemia cases
"""

import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
import starsim as ss
import fpsim as fp

from menstruation import Menstruation
from education import Education
from analyzers import track_hmb_anemia


# ============================================================================
# Simulation creation
# ============================================================================

def make_base_sim(seed=0):
    """
    Create baseline simulation without intervention

    Returns a simulation with HMB dynamics but no care pathway intervention
    """
    mens = Menstruation()
    edu = Education()
    hmb_anemia_analyzer = track_hmb_anemia()

    sim = fp.Sim(
        start=2020,
        stop=2030,
        n_agents=5000,
        total_pop=55_000_000,  # Kenya's population for scaling
        location='kenya',
        education_module=edu,
        connectors=[mens],
        analyzers=[hmb_anemia_analyzer],
        rand_seed=seed,
        verbose=0,
    )
    return sim


# ============================================================================
# Plotting functions
# ============================================================================

def plot_sim(msim, save_dir='figures'):
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


def plot_hmb_anemia_correlation(msim, save_dir='figures'):
    """
    Plot relationship between HMB and anemia

    Shows anemia prevalence stratified by HMB status, demonstrating that
    anemia is more common among women with HMB. Also shows potentially
    avertable anemia cases by targeting HMB.

    Args:
        msim: MultiSim object with baseline results
        save_dir: Directory to save figures (default: 'figures')
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('HMB and anemia relationship', fontsize=16)

    # Get data from all sims
    tvec = msim.sims[0].timevec
    years = np.array([t.year + (t.month - 1) / 12 for t in tvec])

    # Plot 1: Time series comparison - overall prevalence
    ax = axes[0]
    for result_name, color, label in [
        ('hmb_prev', '#d62728', 'HMB prevalence'),
        ('anemic_prev', '#ff7f0e', 'Overall anemia prevalence')
    ]:
        data = np.array([s.results.menstruation[result_name] for s in msim.sims])
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(years, mean, color=color, linewidth=2, label=label)
        ax.fill_between(years, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_xlabel('Year')
    ax.set_ylabel('Prevalence')
    ax.set_title('Overall prevalence over time')
    ax.set_ylim([0, 1])
    ax.legend(frameon=False, loc='best')
    ax.grid(alpha=0.3)

    # Plot 2: Anemia prevalence stratified by HMB status
    ax = axes[1]

    # Calculate stratified anemia prevalence for each sim
    anemia_with_hmb_all = []
    anemia_without_hmb_all = []

    for sim in msim.sims:
        mens = sim.connectors.menstruation

        # Get menstruating women at final timepoint
        menstruating = mens.menstruating.uids

        # Split by HMB status
        has_hmb = mens.hmb[menstruating]
        has_anemia = mens.anemic[menstruating]

        # Calculate prevalence in each group
        if np.sum(has_hmb) > 0:
            anemia_hmb = np.mean(has_anemia[has_hmb])
        else:
            anemia_hmb = 0

        if np.sum(~has_hmb) > 0:
            anemia_no_hmb = np.mean(has_anemia[~has_hmb])
        else:
            anemia_no_hmb = 0

        anemia_with_hmb_all.append(anemia_hmb)
        anemia_without_hmb_all.append(anemia_no_hmb)

    # Plot as bar chart with error bars
    x_pos = np.arange(2)
    means = [np.mean(anemia_without_hmb_all), np.mean(anemia_with_hmb_all)]
    stds = [np.std(anemia_without_hmb_all), np.std(anemia_with_hmb_all)]
    colors_bar = ['#7fc97f', '#d62728']

    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors_bar, alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['No HMB', 'HMB'])
    ax.set_ylabel('Anemia prevalence')
    ax.set_title('Anemia prevalence by HMB status\n(among menstruating women, final timepoint)')
    ax.set_ylim([0, max(means) * 1.3])
    ax.grid(alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.2%}\n±{std:.2%}',
                ha='center', va='bottom', fontsize=10)

    # Add statistical note
    fold_increase = means[1] / means[0] if means[0] > 0 else 0
    ax.text(0.5, 0.05, f'{fold_increase:.1f}x higher with HMB',
            transform=ax.transAxes, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 3: Anemia cases - total and potentially avertable
    ax = axes[2]

    # Calculate anemia cases among HMB women over time for each sim
    # These represent cases that could potentially be averted by targeting HMB
    avertable_cases_over_time = []
    total_anemia_cases_over_time = []

    for sim in msim.sims:
        # Get anemia cases among HMB women directly from analyzer
        avertable = sim.results.track_hmb_anemia['n_anemia_with_hmb']
        avertable_cases_over_time.append(avertable)

        # Get total anemia cases
        total_anemia = sim.results.track_hmb_anemia['n_anemia_total']
        total_anemia_cases_over_time.append(total_anemia)

    # Calculate mean and std across runs for avertable cases
    avertable_array = np.array(avertable_cases_over_time)
    mean_avertable = avertable_array.mean(axis=0)
    std_avertable = avertable_array.std(axis=0)

    # Calculate mean and std across runs for total anemia
    total_array = np.array(total_anemia_cases_over_time)
    mean_total = total_array.mean(axis=0)
    std_total = total_array.std(axis=0)

    # Plot total anemia cases
    ax.plot(years, mean_total, color='#ff7f0e', linewidth=2, label='Total anemia cases')
    ax.fill_between(years, mean_total - std_total, mean_total + std_total,
                     color='#ff7f0e', alpha=0.2)

    # Plot potentially avertable cases (anemia among HMB women)
    ax.plot(years, mean_avertable, color='#e377c2', linewidth=2, label='Anemia with HMB (potentially avertable)')
    ax.fill_between(years, mean_avertable - std_avertable, mean_avertable + std_avertable,
                     color='#e377c2', alpha=0.3)

    ax.set_xlabel('Year')
    ax.set_ylabel('Number of cases')
    ax.set_title('Anemia cases: total burden and potentially avertable')
    ax.set_ylim(bottom=0)  # Start y-axis at 0
    sc.SIticks(ax=ax)  # Use SI formatting (M for million, K for thousand)
    ax.legend(frameon=False, loc='best')
    ax.grid(alpha=0.3)

    # Add annotation with final year averages
    final_total = mean_total[-1]
    final_avert = mean_avertable[-1]
    pct_avertable = (final_avert / final_total * 100) if final_total > 0 else 0

    # Format in millions for readability
    total_m = final_total / 1e6
    avert_m = final_avert / 1e6

    ax.text(0.5, 0.95,
            f'Final year: {total_m:.2f}m total, {avert_m:.2f}m with HMB ({pct_avertable:.1f}%)',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    filename = f'{save_dir}/hmb_anemia_relationship.png'
    sc.savefig(filename, dpi=150)
    print(f'Saved HMB-anemia relationship figure to {filename}')

    return fig


# ============================================================================
# Main execution
# ============================================================================

if __name__ == '__main__':

    # Set plotting options
    sc.options(dpi=150)

    # Create figures directory if it doesn't exist
    sc.path('figures').mkdir(exist_ok=True)

    # Run baseline simulations
    print('Creating baseline simulations...')
    n_runs = 10  # Number of runs for uncertainty
    msim_base = ss.MultiSim(sims=[make_base_sim(seed=i) for i in range(n_runs)])

    print('Running baseline simulations...')
    msim_base.run()

    print('Generating baseline characteristics plot...')
    plot_sim(msim_base)

    print('Generating HMB-anemia correlation plots...')
    plot_hmb_anemia_correlation(msim_base)

    # Optionally save results
    print('Saving baseline results...')
    sc.path('results').mkdir(exist_ok=True)
    sc.saveobj('results/baseline_msim.obj', msim_base)

    print('Done! Generated:')
    print('  - figures/baseline_characteristics.png')
    print('  - figures/hmb_anemia_relationship.png')
