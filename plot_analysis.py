"""
Plot baseline simulation characteristics

Creates figures showing HMB dynamics, anemia prevalence, and related outcomes
in the baseline scenario (no intervention).
"""

import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
import starsim as ss


def load_results():
    """Load saved MultiSim results"""
    msim_base = sc.loadobj('results/baseline_msim.obj')
    return msim_base


def plot_baseline_characteristics(msim, save_dir='figures'):
    """
    Plot key baseline characteristics from simulation

    Args:
        msim: MultiSim object with baseline results
        save_dir: Directory to save figures (default: 'figures')
    """
    sc.options(dpi=150)

    # Create results directory if it doesn't exist
    sc.path(save_dir).mkdir(exist_ok=True)

    # Extract results from first sim (they all have same structure)
    sim = msim.sims[0]
    tvec = sim.timevec  # Time vector (in datetime format)

    # Convert to years for plotting
    years = np.array([t.year + (t.month - 1) / 12 for t in tvec])

    # Get mean and std across all runs for each metric
    def get_stats(result_name):
        """Helper to get mean and std across runs"""
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

    # Plot 3: Potentially avertable anemia cases over time
    ax = axes[2]

    # Calculate anemia cases among HMB women over time for each sim
    # These represent cases that could potentially be averted by targeting HMB
    avertable_cases_over_time = []

    for sim in msim.sims:
        # Get anemia cases among HMB women directly from analyzer
        avertable = sim.results.track_hmb_anemia['n_anemia_with_hmb']
        avertable_cases_over_time.append(avertable)

    # Calculate mean and std across runs
    avertable_array = np.array(avertable_cases_over_time)
    mean_avertable = avertable_array.mean(axis=0)
    std_avertable = avertable_array.std(axis=0)

    # Plot
    ax.plot(years, mean_avertable, color='#e377c2', linewidth=2, label='Mean')
    ax.fill_between(years, mean_avertable - std_avertable, mean_avertable + std_avertable,
                     color='#e377c2', alpha=0.3, label='±1 SD')
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of cases')
    ax.set_title('Potentially avertable anemia cases\n(anemia among HMB women)')
    ax.set_ylim(bottom=0)  # Start y-axis at 0
    sc.SIticks(ax=ax)  # Use SI formatting (M for million, K for thousand)
    ax.legend(frameon=False, loc='best')
    ax.grid(alpha=0.3)

    # Add annotation with final year average
    final_mean = mean_avertable[-1]
    final_std = std_avertable[-1]
    # Format in millions for readability
    mean_m = final_mean / 1e6
    std_m = final_std / 1e6
    ax.text(0.5, 0.95, f'Final year: {mean_m:.2f}m ± {std_m:.2f}m cases',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    filename = f'{save_dir}/hmb_anemia_relationship.png'
    sc.savefig(filename, dpi=150)
    print(f'Saved HMB-anemia relationship figure to {filename}')

    return fig


def plot_intervention_impact(msim_base, msim_intv, msim_iud=None, save_dir='figures'):
    """
    Compare baseline and intervention scenarios to show intervention impact on anemia cases.

    Shows:
    1. Anemia cases among HMB women in baseline vs intervention(s)
    2. Averted anemia cases (difference between scenarios)
    3. Percent reduction over time

    Args:
        msim_base: MultiSim object with baseline results
        msim_intv: MultiSim object with full intervention results
        msim_iud: MultiSim object with IUD-only intervention results (optional)
        save_dir: Directory to save figures (default: 'figures')
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Intervention impact on anemia among HMB women', fontsize=16)

    # Get time vector
    tvec = msim_base.sims[0].timevec
    years = np.array([t.year + (t.month - 1) / 12 for t in tvec])

    # Extract anemia cases among HMB women for each scenario
    base_cases = []
    intv_cases = []
    iud_cases = []

    for sim in msim_base.sims:
        base_cases.append(sim.results.track_hmb_anemia['n_anemia_with_hmb'])

    for sim in msim_intv.sims:
        intv_cases.append(sim.results.track_hmb_anemia['n_anemia_with_hmb'])

    if msim_iud is not None:
        for sim in msim_iud.sims:
            iud_cases.append(sim.results.track_hmb_anemia['n_anemia_with_hmb'])

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

    if msim_iud is not None:
        iud_array = np.array(iud_cases)
        iud_averted_array = base_array - iud_array
        iud_mean = iud_array.mean(axis=0)
        iud_std = iud_array.std(axis=0)
        iud_averted_mean = iud_averted_array.mean(axis=0)
        iud_averted_std = iud_averted_array.std(axis=0)

    # Plot 1: Baseline vs Intervention(s)
    ax = axes[0]
    ax.plot(years, base_mean, color='#d62728', linewidth=2, label='Baseline', linestyle='--')
    ax.fill_between(years, base_mean - base_std, base_mean + base_std, color='#d62728', alpha=0.2)
    ax.plot(years, intv_mean, color='#2ca02c', linewidth=2, label='Full package')
    ax.fill_between(years, intv_mean - intv_std, intv_mean + intv_std, color='#2ca02c', alpha=0.2)
    if msim_iud is not None:
        ax.plot(years, iud_mean, color='#9467bd', linewidth=2, label='IUD only')
        ax.fill_between(years, iud_mean - iud_std, iud_mean + iud_std, color='#9467bd', alpha=0.2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of anemia cases')
    ax.set_title('Anemia among HMB women')
    ax.set_ylim(bottom=0)
    sc.SIticks(ax=ax)
    ax.legend(frameon=False, loc='best')
    ax.grid(alpha=0.3)

    # Plot 2: Averted cases over time
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
    ax.set_title('Anemia cases averted by intervention')
    ax.set_ylim(bottom=0)
    sc.SIticks(ax=ax)
    ax.legend(frameon=False, loc='best')
    ax.grid(alpha=0.3)

    # Plot 3: Percent reduction over time
    ax = axes[2]
    # Calculate percent reduction, avoiding division by zero
    valid_mask = base_mean > 0
    percent_reduction = np.zeros_like(averted_mean)
    percent_reduction[valid_mask] = 100 * averted_mean[valid_mask] / base_mean[valid_mask]

    ax.plot(years, percent_reduction, color='#2ca02c', linewidth=2, label='Full package')

    if msim_iud is not None:
        iud_percent_reduction = np.zeros_like(iud_averted_mean)
        iud_percent_reduction[valid_mask] = 100 * iud_averted_mean[valid_mask] / base_mean[valid_mask]
        ax.plot(years, iud_percent_reduction, color='#9467bd', linewidth=2, label='IUD only')

    ax.set_xlabel('Year')
    ax.set_ylabel('Percent reduction (%)')
    ax.set_title('Percent reduction in anemia cases')
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
    print(f'Final year baseline anemia cases:    {base_mean[-1]/1e6:.2f}m ± {base_std[-1]/1e6:.2f}m')
    print(f'Final year with full package:        {intv_mean[-1]/1e6:.2f}m ± {intv_std[-1]/1e6:.2f}m')
    print(f'  Averted anemia cases:              {averted_mean[-1]/1e6:.2f}m ± {averted_std[-1]/1e6:.2f}m')
    print(f'  Percent reduction:                 {percent_reduction[-1]:.1f}%')
    if msim_iud is not None:
        print(f'Final year with IUD only:            {iud_mean[-1]/1e6:.2f}m ± {iud_std[-1]/1e6:.2f}m')
        print(f'  Averted anemia cases:              {iud_averted_mean[-1]/1e6:.2f}m ± {iud_averted_std[-1]/1e6:.2f}m')
        print(f'  Percent reduction:                 {iud_percent_reduction[-1]:.1f}%')
    print('='*60 + '\n')

    return fig


def print_summary_stats(msim):
    """Print summary statistics from baseline simulation"""
    print('\n' + '='*60)
    print('BASELINE SIMULATION SUMMARY STATISTICS')
    print('='*60)

    # Get final year values across all runs
    metrics = {
        'HMB prevalence': 'hmb_prev',
        'Anemia prevalence': 'anemic_prev',
        'Cumulative anemia cases': 'n_anemia',
        'Pain prevalence': 'pain_prev',
        'Poor MH prevalence': 'poor_mh_prev',
        'Hysterectomy prevalence': 'hyst_prev',
    }

    for label, result_name in metrics.items():
        values = [s.results.menstruation[result_name][-1] for s in msim.sims]
        mean = np.mean(values)
        std = np.std(values)
        print(f'{label:30s}: {mean:8.3f} ± {std:.3f}')

    print('='*60 + '\n')


if __name__ == '__main__':
    # Load results
    print('Loading baseline results...')
    msim_base = sc.loadobj('results/baseline_msim.obj')

    print('Loading intervention results...')
    msim_intv = sc.loadobj('results/intervention_msim.obj')

    print('Loading IUD-only results...')
    msim_iud = sc.loadobj('results/iud_only_msim.obj')

    # Generate plots
    print('Generating baseline characteristics plot...')
    plot_baseline_characteristics(msim_base)

    print('Generating HMB-anemia relationship plot...')
    plot_hmb_anemia_correlation(msim_base)

    print('Generating intervention impact plot (all scenarios)...')
    plot_intervention_impact(msim_base, msim_intv, msim_iud)

    # Print summary statistics
    print_summary_stats(msim_base)

    print('Done!')
