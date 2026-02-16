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


def load_cascade_results():
    """Load saved cascade analysis results"""
    baseline_sim = sc.loadobj('results/cascade_baseline_sim.obj')
    intervention_sim = sc.loadobj('results/cascade_intervention_sim.obj')
    return baseline_sim, intervention_sim


def analyze_cascade_depth(cascade_analyzer):
    """
    Analyze how many treatments women tried

    Returns dict with counts and proportions
    """
    final_ti = -1

    results = {
        'n_tried': {
            0: cascade_analyzer.results.n_tried_0_tx[final_ti],
            1: cascade_analyzer.results.n_tried_1_tx[final_ti],
            2: cascade_analyzer.results.n_tried_2_tx[final_ti],
            3: cascade_analyzer.results.n_tried_3_tx[final_ti],
            4: cascade_analyzer.results.n_tried_4_tx[final_ti],
        },
        'prop_tried': {
            0: cascade_analyzer.results.prop_tried_0_tx[final_ti],
            1: cascade_analyzer.results.prop_tried_1_tx[final_ti],
            2: cascade_analyzer.results.prop_tried_2_tx[final_ti],
            3: cascade_analyzer.results.prop_tried_3_tx[final_ti],
            4: cascade_analyzer.results.prop_tried_4_tx[final_ti],
        }
    }

    return results


def analyze_dropoffs(cascade_analyzer):
    """
    Analyze where women drop off in the cascade

    Returns dict with offer/accept counts and rates
    """
    final_ti = -1

    treatments = ['nsaid', 'txa', 'pill', 'hiud']
    dropoffs = {}

    for tx in treatments:
        offered = cascade_analyzer.results[f'offered_{tx}'][final_ti]
        accepted = cascade_analyzer.results[f'accepted_{tx}'][final_ti]

        dropoffs[tx] = {
            'offered': offered,
            'accepted': accepted,
            'declined': offered - accepted,
            'acceptance_rate': accepted / offered if offered > 0 else 0,
            'dropout_rate': (offered - accepted) / offered if offered > 0 else 0,
        }

    return dropoffs


def calculate_anemia_averted(baseline_sim, intervention_sim):
    """
    Calculate anemia cases averted by the intervention

    Returns dict with total and per-component estimates
    """
    # Get final anemia counts
    final_ti = -1

    # Use anemia among HMB women (the target population for the intervention)
    baseline_anemia = baseline_sim.results.track_hmb_anemia.n_anemia_with_hmb[final_ti]
    intervention_anemia = intervention_sim.results.track_hmb_anemia.n_anemia_with_hmb[final_ti]

    total_averted = baseline_anemia - intervention_anemia

    # Get the cascade analyzer from intervention sim
    cascade = intervention_sim.analyzers[0]

    component_averted = {}
    if hasattr(cascade, 'anemia_reduction') and cascade.anemia_reduction is not None:
        # Get number of people who tried each treatment
        n_nsaid = cascade.results.accepted_nsaid[final_ti]
        n_txa = cascade.results.accepted_txa[final_ti]
        n_pill = cascade.results.accepted_pill[final_ti]
        n_hiud = cascade.results.accepted_hiud[final_ti]

        # Estimate anemia cases averted by each treatment
        component_averted['nsaid'] = max(0, cascade.anemia_reduction['nsaid'] * n_nsaid)
        component_averted['txa'] = max(0, cascade.anemia_reduction['txa'] * n_txa)
        component_averted['pill'] = max(0, cascade.anemia_reduction['pill'] * n_pill)
        component_averted['hiud'] = max(0, cascade.anemia_reduction['hiud'] * n_hiud)
    else:
        component_averted = None

    return {
        'total_averted': total_averted,
        'baseline_anemia': baseline_anemia,
        'intervention_anemia': intervention_anemia,
        'component_averted': component_averted,
    }


def plot_cascade_analysis(baseline_sim, intervention_sim, save_dir='figures'):
    """
    Create comprehensive visualization of cascade analysis

    Args:
        baseline_sim: Baseline simulation object
        intervention_sim: Intervention simulation object with cascade analyzer
        save_dir: Directory to save figures (default: 'figures')
    """
    sc.options(dpi=150)
    sc.path(save_dir).mkdir(exist_ok=True)

    cascade = intervention_sim.analyzers[0]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Treatment cascade and component analysis', fontsize=16, y=0.995)

    # 1. Number of treatments tried
    ax = axes[0, 0]
    depth_results = analyze_cascade_depth(cascade)
    treatments_tried = list(depth_results['n_tried'].keys())
    n_tried = list(depth_results['n_tried'].values())

    ax.bar(treatments_tried, n_tried, color='steelblue', alpha=0.7)
    ax.set_xlabel('Number of treatments tried')
    ax.set_ylabel('Number of women')
    ax.set_title('How many women tried N treatments?')
    ax.grid(axis='y', alpha=0.3)

    # Add percentages on bars
    for i, v in enumerate(n_tried):
        if v > 0:
            pct = depth_results['prop_tried'][i] * 100
            ax.text(i, v, f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 2. Cascade dropoffs
    ax = axes[0, 1]
    dropoff_results = analyze_dropoffs(cascade)
    treatments = list(dropoff_results.keys())
    offered = [dropoff_results[tx]['offered'] for tx in treatments]
    accepted = [dropoff_results[tx]['accepted'] for tx in treatments]

    x = np.arange(len(treatments))
    width = 0.35

    ax.bar(x - width/2, offered, width, label='Offered', alpha=0.7, color='lightcoral')
    ax.bar(x + width/2, accepted, width, label='Accepted', alpha=0.7, color='seagreen')

    ax.set_xlabel('Treatment')
    ax.set_ylabel('Number of women')
    ax.set_title('Where do women drop off?')
    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in treatments])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add dropout percentages
    for i, tx in enumerate(treatments):
        dropout_pct = dropoff_results[tx]['dropout_rate'] * 100
        y_pos = offered[i] * 1.05
        ax.text(i - width/2, y_pos, f'{dropout_pct:.0f}%\ndropout',
                ha='center', va='bottom', fontsize=9, color='red')

    # 3. Acceptance rates
    ax = axes[0, 2]
    acceptance_rates = [dropoff_results[tx]['acceptance_rate'] * 100 for tx in treatments]
    colors = ['green' if r > 50 else 'orange' if r > 25 else 'red' for r in acceptance_rates]

    ax.bar(treatments, acceptance_rates, color=colors, alpha=0.7)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, label='50% threshold')
    ax.set_xlabel('Treatment')
    ax.set_ylabel('Acceptance rate (%)')
    ax.set_title('Treatment acceptance rates')
    ax.set_xticklabels([t.upper() for t in treatments])
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 4. Anemia prevalence by cascade depth
    ax = axes[1, 0]
    final_ti = -1
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

    # 5. Selection bias explanation
    ax = axes[1, 1]
    ax.axis('off')

    # Show the selection bias issue that prevents component attribution
    explanation_text = '''Why component attribution is not shown:

Women who seek treatment are sicker:
  • No treatment:    17.4% anemic
  • Tried NSAID:     31.9% anemic
  • Tried TXA:       31.9% anemic
  • Tried Pill:      32.7% anemic
  • Tried hIUD:      32.7% anemic

This is selection bias, not treatment failure!
Sicker women are more likely to seek care.

To attribute impact to specific treatments
would require tracking individuals over time
or using matched controls.'''

    ax.text(0.5, 0.5, explanation_text,
           ha='center', va='center', transform=ax.transAxes,
           fontsize=9, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

    # 6. Total impact summary (instead of flawed component attribution)
    ax = axes[1, 2]
    anemia_results = calculate_anemia_averted(baseline_sim, intervention_sim)

    # Create a summary text display instead of trying to attribute to components
    ax.axis('off')

    total_averted = anemia_results['total_averted']
    baseline = anemia_results['baseline_anemia']
    intervention = anemia_results['intervention_anemia']
    pct_reduction = (total_averted / baseline * 100) if baseline > 0 else 0

    # Format numbers in millions for readability
    summary_text = f'''Overall intervention impact

Baseline anemia (HMB women):
  {baseline/1e6:.2f} million cases

With intervention:
  {intervention/1e6:.2f} million cases

Cases averted:
  {total_averted/1e6:.2f} million
  ({pct_reduction:.1f}% reduction)

Note: Component attribution not shown due to
selection bias (sicker women seek treatment,
so naive comparisons are confounded).'''

    ax.text(0.5, 0.5, summary_text,
           ha='center', va='center', transform=ax.transAxes,
           fontsize=10, family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    filename = f'{save_dir}/cascade_component_analysis.png'
    sc.savefig(filename, dpi=150)
    print(f'Saved cascade analysis figure to {filename}')

    return fig


def print_cascade_summary(baseline_sim, intervention_sim):
    """
    Print text summary of cascade analysis
    """
    cascade = intervention_sim.analyzers[0]

    print("\n" + "="*80)
    print("TREATMENT CASCADE ANALYSIS - SUMMARY")
    print("="*80)

    # 1. Cascade depth
    print("\n1. HOW MANY WOMEN TRIED N TREATMENTS?")
    print("-" * 80)
    depth_results = analyze_cascade_depth(cascade)
    for n, count in depth_results['n_tried'].items():
        pct = depth_results['prop_tried'][n] * 100
        print(f"   Tried {n} treatment(s): {count:>6.0f} women ({pct:>5.1f}%)")

    # 2. Dropoffs
    print("\n2. WHERE DO WOMEN DROP OFF IN THE CASCADE?")
    print("-" * 80)
    dropoff_results = analyze_dropoffs(cascade)
    for tx, data in dropoff_results.items():
        print(f"   {tx.upper():<6}: {data['offered']:>6.0f} offered → "
              f"{data['accepted']:>6.0f} accepted ({data['acceptance_rate']*100:>5.1f}% acceptance) | "
              f"{data['declined']:>6.0f} declined ({data['dropout_rate']*100:>5.1f}% dropout)")

    # 3. Anemia outcomes
    print("\n3. ANEMIA OUTCOMES")
    print("-" * 80)
    anemia_results = calculate_anemia_averted(baseline_sim, intervention_sim)
    print(f"   Baseline anemia cases:     {anemia_results['baseline_anemia']:>8.0f}")
    print(f"   Intervention anemia cases: {anemia_results['intervention_anemia']:>8.0f}")
    print(f"   Total cases averted:       {anemia_results['total_averted']:>8.0f}")

    if anemia_results['component_averted']:
        print(f"\n   Component attribution (estimated):")
        total_component = sum(anemia_results['component_averted'].values())
        for tx, averted in anemia_results['component_averted'].items():
            pct = (averted / total_component * 100) if total_component > 0 else 0
            print(f"      {tx.upper():<6}: {averted:>6.0f} cases ({pct:>5.1f}% of component total)")

    # 4. Anemia by cascade depth
    print("\n4. ANEMIA PREVALENCE BY CASCADE DEPTH")
    print("-" * 80)
    final_ti = -1
    for i in range(5):
        anemia_prev = cascade.results[f'anemia_tried_{i}'][final_ti] * 100
        n_tried = depth_results['n_tried'][i]
        print(f"   Tried {i} treatment(s): {anemia_prev:>5.1f}% anemia (among {n_tried:.0f} women)")

    print("\n" + "="*80 + "\n")


def plot_component_attribution(save_dir='figures'):
    """
    Plot component-specific attribution if component analysis results exist.

    This function loads results from the component analysis (if available)
    and creates a comprehensive visualization showing the marginal impact
    of each treatment component.

    Args:
        save_dir: Directory to save figures

    Returns:
        Figure object, or None if component results not found
    """
    try:
        # Try to load component analysis results
        from component_analysis import load_component_results, calculate_component_impacts

        print('\nLoading component analysis results...')
        results = load_component_results()
        impacts = calculate_component_impacts(results)

        # Create visualization
        sc.options(dpi=150)
        sc.path(save_dir).mkdir(exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Component attribution: Anemia reduction by treatment type', fontsize=16, y=0.995)

        components = ['nsaid', 'txa', 'pill', 'hiud']
        component_labels = ['NSAID', 'TXA', 'Pill', 'hIUD']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

        # Panel 1: Cases averted (absolute)
        ax = axes[0, 0]
        averted = [impacts[c]['cases_averted']/1e6 for c in components]
        averted_std = [impacts[c]['averted_std']/1e6 for c in components]

        bars = ax.bar(component_labels, averted, yerr=averted_std,
                       capsize=5, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Anemia cases averted (millions)')
        ax.set_title('Anemia cases averted by each component')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val, std in zip(bars, averted, averted_std):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                    f'{val:.2f}m',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Panel 2: Percent reduction
        ax = axes[0, 1]
        pct_reduction = [impacts[c]['pct_reduction'] for c in components]

        bars = ax.bar(component_labels, pct_reduction, color=colors, alpha=0.7,
                       edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Percent reduction (%)')
        ax.set_title('Percent reduction relative to baseline')
        ax.set_ylim([0, max(pct_reduction) * 1.2])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bar, val in zip(bars, pct_reduction):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Panel 3: Comparison across scenarios
        ax = axes[1, 0]

        scenarios = ['Baseline', 'NSAID', 'TXA', 'Pill', 'hIUD', 'Full\npackage']
        anemia = [impacts['baseline']['anemia_cases']/1e6] + \
                 [impacts[c]['anemia_cases']/1e6 for c in components] + \
                 [impacts['full']['anemia_cases']/1e6]
        anemia_std = [impacts['baseline']['anemia_std']/1e6] + \
                     [impacts[c]['anemia_std']/1e6 for c in components] + \
                     [impacts['full']['anemia_std']/1e6]

        bar_colors = ['gray'] + colors + ['purple']

        bars = ax.bar(scenarios, anemia, yerr=anemia_std, capsize=5,
                       color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Anemia cases (millions)')
        ax.set_title('Anemia among HMB women by scenario')
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Panel 4: Impact decomposition with text summary
        ax = axes[1, 1]
        ax.axis('off')

        # Create summary text
        baseline_cases = impacts['baseline']['anemia_cases']
        full_averted = impacts['full']['cases_averted']

        summary_text = f'''Impact decomposition

Baseline anemia (HMB women):
  {baseline_cases/1e6:.2f} million cases

Individual component impacts:
  NSAID:  {impacts['nsaid']['cases_averted']/1e6:.2f}m averted ({impacts['nsaid']['pct_reduction']:.1f}%)
  TXA:    {impacts['txa']['cases_averted']/1e6:.2f}m averted ({impacts['txa']['pct_reduction']:.1f}%)
  Pill:   {impacts['pill']['cases_averted']/1e6:.2f}m averted ({impacts['pill']['pct_reduction']:.1f}%)
  hIUD:   {impacts['hiud']['cases_averted']/1e6:.2f}m averted ({impacts['hiud']['pct_reduction']:.1f}%)

Full package impact:
  {full_averted/1e6:.2f}m averted ({impacts['full']['pct_reduction']:.1f}%)

Note: Individual impacts measured when each
treatment is offered alone. Full package may
differ from sum due to cascade effects.'''

        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
                ha='center', va='center', fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.tight_layout()

        filename = f'{save_dir}/component_attribution.png'
        sc.savefig(filename, dpi=150)
        print(f'Saved component attribution figure to {filename}')

        # Print summary
        print('\n' + '='*80)
        print('COMPONENT ATTRIBUTION SUMMARY')
        print('='*80)
        print(f'Baseline anemia (HMB women):    {baseline_cases/1e6:.2f}m cases\n')
        print('Cases averted by component (when offered alone):')
        for c, label in zip(components, component_labels):
            averted = impacts[c]['cases_averted']
            pct = impacts[c]['pct_reduction']
            print(f'  {label:6s}: {averted/1e6:.2f}m cases ({pct:.1f}% reduction)')
        print(f'\nFull package: {full_averted/1e6:.2f}m cases ({impacts["full"]["pct_reduction"]:.1f}% reduction)')
        print('='*80 + '\n')

        return fig

    except Exception as e:
        print(f'\nComponent analysis results not found: {e}')
        print('To generate component attribution, run: python run_component_analysis.py\n')
        return None


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

    # Load and plot cascade analysis
    print('\nLoading cascade analysis results...')
    baseline_sim, intervention_sim = load_cascade_results()

    print('Generating cascade analysis plot...')
    plot_cascade_analysis(baseline_sim, intervention_sim)

    print_cascade_summary(baseline_sim, intervention_sim)

    # Try to plot component attribution if available
    plot_component_attribution()

    print('Done!')
