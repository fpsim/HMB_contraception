"""
Sensitivity analysis: base care-seeking rate

Varies the base care-seeking probability (10%, 20%, 35%) with anemia and pain
coefficients calibrated so the combined (anemia + pain) scenario reaches 46%
for all base rates, using a 70/30 budget split favoring anemia.

Interventions compared:
  - baseline  : no intervention
  - cascade   : full HMBCascade (NSAID -> TXA -> Pill -> hIUD)
"""

import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import fpsim as fp
import os
import gc
import matplotlib.pyplot as plt

from menstruation import Menstruation
from education import Education
from interventions import HMBCascade
from analyzers import (track_care_seeking, track_tx_eff, track_tx_dur,
                       track_hmb_anemia, track_cascade, track_anemia_duration)


# ── Output folders ─────────────────────────────────────────────────────────────
PLOTFOLDER = 'figures_care_sa/'
OUTFOLDER  = 'results_care_sa/'
for d in [PLOTFOLDER, OUTFOLDER]:
    os.makedirs(d, exist_ok=True)


# ── Settings ───────────────────────────────────────────────────────────────────
N_SEEDS   = 10
START     = 2020
STOP      = 2030
INTV_YEAR = 2026

# Care-seeking scenarios: combined (anemia + pain) reaches 46% for all
# CARE_SCENARIOS = {
#     '10%': sc.objdict(base=0.10, anemic=1.43, pain=0.61),
#     '20%': sc.objdict(base=0.20, anemic=0.86, pain=0.37),
#     '35%': sc.objdict(base=0.35, anemic=0.32, pain=0.14),
# }

# anemic and pain coefficients held fixed; only base rate varies
CARE_SCENARIOS = {
    '10%': sc.objdict(base=0.10, anemic=0.86, pain=0.37),
    '20%': sc.objdict(base=0.20, anemic=0.86, pain=0.37),
    '35%': sc.objdict(base=0.35, anemic=0.86, pain=0.37),
}

SCENARIO_LABELS = {
    '10%': 'Base 10%',
    '20%': 'Base 20%',
    '35%': 'Base 35%',
}
SCENARIO_COLORS = {
    '10%': '#d62728',   # red
    '20%': '#ff7f0e',   # orange
    '35%': '#2196F3',   # blue
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def _annualize(monthly_arr, how='sum'):
    """
    Convert monthly array to annual.

    Args:
        monthly_arr: Array of monthly values
        how: 'sum' for annual totals, 'eoy' for end-of-year snapshot

    Returns:
        Annual array
    """
    arr = np.asarray(monthly_arr)
    n_years = len(arr) // 12
    arr = arr[:12 * n_years].reshape(n_years, 12)
    if how == 'sum':
        return arr.sum(axis=1)
    elif how == 'eoy':
        return arr[:, -1]


# ── Simulation creation ───────────────────────────────────────────────────────
def make_sim(care_behavior=None, with_intervention=True, seed=0):
    """
    Build a simulation, optionally with the HMB cascade intervention.

    Args:
        care_behavior: sc.objdict with keys 'base', 'anemic', 'pain'.
                       Only used when with_intervention=True.
        with_intervention: Whether to include HMBCascade
        seed: Random seed

    Returns:
        fp.Sim instance
    """
    mens = Menstruation()
    edu  = Education()

    analyzers = [
        track_hmb_anemia(),
        track_anemia_duration(),
    ]

    sim_kwargs = dict(
        start=START,
        stop=STOP,
        n_agents=10000,
        total_pop=55_000_000,
        location='kenya',
        education_module=edu,
        connectors=[mens],
        analyzers=analyzers,
        rand_seed=seed,
        verbose=0,
    )

    if with_intervention:
        cascade = HMBCascade(
            pars=dict(
                year=INTV_YEAR,
                time_to_assess=ss.months(3),
                care_behavior=care_behavior,
            )
        )
        sim_kwargs['interventions'] = [cascade]
        # Add intervention-specific analyzers
        sim_kwargs['analyzers'].extend([
            track_care_seeking(),
            track_tx_eff(),
            track_tx_dur(),
            track_cascade(),
        ])

    return fp.Sim(**sim_kwargs)


# ── Run simulations ────────────────────────────────────────────────────────────
def run_care_sensitivity(force_rerun=True):
    """
    Run baseline and intervention simulations for each care-seeking scenario.

    For each scenario × seed, runs a matched pair:
      - baseline sim (no intervention, same seed)
      - cascade sim  (with HMBCascade, same seed)

    Averted cases are computed within each seed before aggregating.

    Returns:
        raw: dict with structure
             raw[scenario]['baseline']  = list of annual anemia arrays (one per seed)
             raw[scenario]['cascade']   = list of annual anemia arrays (one per seed)
             raw[scenario]['averted']   = list of (baseline - cascade) arrays
    """
    results_file = OUTFOLDER + 'care_sa_raw.obj'

    if not force_rerun and os.path.exists(results_file):
        print("Loading saved results...")
        return sc.loadobj(results_file)

    raw = {
        scenario: {
            'baseline': [], 'cascade': [], 'averted': [],
            'baseline_monthly': [], 'cascade_monthly': [],
            'baseline_total_anemia_monthly': [],
            'cascade_total_anemia_monthly': [],
            'care_seeking_prev': [],
            'care_seeking_anemic': [],
            'care_seeking_not_anemic': [],
            'cascade_depth': [],  # list of dicts per seed (end-of-sim snapshot)
            'hiud_uptake': [],  # per-seed hIUD uptake metrics
        }
        for scenario in CARE_SCENARIOS
    }

    for scenario, care_behavior in CARE_SCENARIOS.items():
        print(f"\n=== {SCENARIO_LABELS[scenario]} "
              f"(base={care_behavior.base}, anemic={care_behavior.anemic:.2f}, "
              f"pain={care_behavior.pain:.2f}) ===")

        for seed in range(N_SEEDS):
            print(f"  seed {seed}...", end=" ", flush=True)

            # Matched pair: same seed, same demographics
            sims = [
                make_sim(with_intervention=False, seed=seed),                    # baseline
                make_sim(care_behavior=care_behavior, with_intervention=True, seed=seed),  # cascade
            ]
            msim = ss.MultiSim(sims)
            msim.run()

            s_base    = msim.sims[0]
            s_cascade = msim.sims[1]

            # Extract and annualize HMB-related anemia
            base_monthly    = s_base.results.track_hmb_anemia['n_anemia_with_hmb']
            cascade_monthly = s_cascade.results.track_hmb_anemia['n_anemia_with_hmb']

            # Store monthly data
            raw[scenario]['baseline_monthly'].append(np.asarray(base_monthly))
            raw[scenario]['cascade_monthly'].append(np.asarray(cascade_monthly))

            # Store total anemia (for impact comparison plot)
            raw[scenario]['baseline_total_anemia_monthly'].append(
                np.asarray(s_base.results.track_hmb_anemia['n_anemia_total']))
            raw[scenario]['cascade_total_anemia_monthly'].append(
                np.asarray(s_cascade.results.track_hmb_anemia['n_anemia_total']))

            # Store care-seeking rates (from intervention sim only)
            raw[scenario]['care_seeking_prev'].append(
                np.asarray(s_cascade.results.track_care_seeking['care_seeking_prev']))
            raw[scenario]['care_seeking_anemic'].append(
                np.asarray(s_cascade.results.track_care_seeking['care_seeking_anemic']))
            raw[scenario]['care_seeking_not_anemic'].append(
                np.asarray(s_cascade.results.track_care_seeking['care_seeking_not_anemic']))

            # Store cascade depth distribution (end-of-sim snapshot)
            cascade_intv = s_cascade.interventions.hmb_cascade
            menstruating = s_cascade.people.menstruation.menstruating
            n_treatments = (
                np.array(cascade_intv.tried_nsaid, dtype=int) +
                np.array(cascade_intv.tried_txa, dtype=int) +
                np.array(cascade_intv.tried_pill, dtype=int) +
                np.array(cascade_intv.tried_hiud, dtype=int)
            )
            total = np.count_nonzero(menstruating)
            depth_dist = {}
            for n in range(5):
                count = np.count_nonzero((n_treatments == n) & menstruating)
                depth_dist[n] = 100 * count / total if total > 0 else 0
            raw[scenario]['cascade_depth'].append(depth_dist)
            
            # hIUD uptake among HMB women
            hmb = s_cascade.people.menstruation.hmb
            hmb_menstruating = hmb & menstruating
            n_hmb = np.count_nonzero(hmb_menstruating)

            # % of HMB women seeking care who initiate hIUD
            # "Seeking care" ≈ ever offered NSAID (first-line, so offered = sought care)
            ever_offered_nsaid = cascade_intv.treatments['nsaid'].offered & hmb_menstruating
            n_hmb_seekers = np.count_nonzero(ever_offered_nsaid)
            tried_hiud_among_seekers = cascade_intv.treatments['hiud'].tried_treatment & ever_offered_nsaid
            n_hiud_among_seekers = np.count_nonzero(tried_hiud_among_seekers)

            # % of women with HMB using hIUD (ever tried)
            tried_hiud_hmb = cascade_intv.treatments['hiud'].tried_treatment & hmb_menstruating
            n_hiud_hmb = np.count_nonzero(tried_hiud_hmb)

            hiud_uptake = {
                'pct_of_hmb_seekers': 100 * n_hiud_among_seekers / n_hmb_seekers if n_hmb_seekers > 0 else 0,
                'pct_of_hmb': 100 * n_hiud_hmb / n_hmb if n_hmb > 0 else 0,
                'n_hmb_seekers': n_hmb_seekers,
                'n_hiud_among_seekers': n_hiud_among_seekers,
                'n_hmb': n_hmb,
                'n_hiud_hmb': n_hiud_hmb,
            }
            raw[scenario]['hiud_uptake'].append(hiud_uptake)

            # Annualize
            base_annual    = _annualize(base_monthly)
            cascade_annual = _annualize(cascade_monthly)
            averted_annual = base_annual - cascade_annual

            raw[scenario]['baseline'].append(base_annual)
            raw[scenario]['cascade'].append(cascade_annual)
            raw[scenario]['averted'].append(averted_annual)

            del msim
            gc.collect()
            print("done")

    sc.saveobj(results_file, raw)
    print(f"\nSaved raw results: {results_file}")
    return raw


# ── Aggregate statistics ───────────────────────────────────────────────────────
def compute_stats(raw):
    """
    Compute % reduction in anemia cases, aggregated across seeds.

    % reduction = (baseline - cascade) / baseline * 100, computed per seed
    then summarised as mean ± std.

    Returns:
        stats: dict with structure
               stats[scenario] = {'mean', 'std'} arrays over years
    """
    stats = {}
    for scenario in CARE_SCENARIOS:
        base_arr    = np.array(raw[scenario]['baseline'])   # (n_seeds, n_years)
        averted_arr = np.array(raw[scenario]['averted'])

        pct = np.where(base_arr > 0, averted_arr / base_arr * 100, np.nan)

        stats[scenario] = {
            'mean': np.nanmean(pct, axis=0),
            'std':  np.nanstd(pct, axis=0),
        }
    return stats


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_annual_cases(raw, years):
    """
    Single panel: annual HMB-related anemia cases for baseline vs cascade,
    with care-seeking scenarios as separate colored lines.

    Baseline shown once (dashed black), intervention lines colored by scenario.
    Shaded bands = mean ± std across seeds.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Baseline: use any scenario (baseline is identical across scenarios for same seed)
    # Average across all scenarios' baselines for robustness
    all_baselines = []
    for scenario in CARE_SCENARIOS:
        all_baselines.extend(raw[scenario]['baseline'])
    base_arr = np.array(all_baselines)
    base_mean = base_arr.mean(axis=0)
    base_std  = base_arr.std(axis=0)

    ax.plot(years, base_mean, color='#6c757d', lw=2.5, ls='--', label='No intervention')
    ax.fill_between(years, base_mean - base_std, base_mean + base_std,
                    color='#6c757d', alpha=0.15)

    # Intervention: one line per scenario
    for scenario in CARE_SCENARIOS:
        casc_arr  = np.array(raw[scenario]['cascade'])
        casc_mean = casc_arr.mean(axis=0)
        casc_std  = casc_arr.std(axis=0)

        ax.plot(years, casc_mean, color=SCENARIO_COLORS[scenario], lw=2.5,
                label=f'Cascade ({SCENARIO_LABELS[scenario]})')
        ax.fill_between(years, casc_mean - casc_std, casc_mean + casc_std,
                        color=SCENARIO_COLORS[scenario], alpha=0.15)

    ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
    ylim = ax.get_ylim()
    ax.text(INTV_YEAR - 0.2, ylim[1] * 0.95, 'Start of\nintervention',
            ha='right', va='top', fontsize=9, color='#4d4d4d')

    ax.set_xlabel('Year')
    ax.set_ylabel('Annual HMB-related anemia cases')
    ax.set_title('Annual HMB-related anemia: baseline vs cascade\n'
                 'by base care-seeking probability')
    ax.set_xlim([START, STOP])
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    sc.SIticks(ax=ax)

    plt.tight_layout()
    outpath = PLOTFOLDER + 'care_sa_annual_cases.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


def plot_monthly_cases(raw, years_monthly):
    """
    Single panel: monthly HMB-related anemia cases for baseline vs cascade,
    with care-seeking scenarios as separate colored lines.

    Same as plot_annual_cases but at monthly resolution.
    Shaded bands = mean ± std across seeds.

    Args:
        raw: Output from run_care_sensitivity()
        years_monthly: Array of fractional years for each month
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Baseline: average across all scenarios' baselines
    all_baselines = []
    for scenario in CARE_SCENARIOS:
        for seed_data in raw[scenario]['baseline_monthly']:
            all_baselines.append(seed_data)
    base_arr = np.array(all_baselines)
    base_mean = base_arr.mean(axis=0)
    base_std  = base_arr.std(axis=0)

    ax.plot(years_monthly, base_mean, color='#6c757d', lw=1.5, ls='--',
            label='No intervention')
    ax.fill_between(years_monthly, base_mean - base_std, base_mean + base_std,
                    color='#6c757d', alpha=0.15)

    # Intervention: one line per scenario
    for scenario in CARE_SCENARIOS:
        casc_arr  = np.array(raw[scenario]['cascade_monthly'])
        casc_mean = casc_arr.mean(axis=0)
        casc_std  = casc_arr.std(axis=0)

        ax.plot(years_monthly, casc_mean, color=SCENARIO_COLORS[scenario], lw=1.5,
                label=f'Cascade ({SCENARIO_LABELS[scenario]})')
        ax.fill_between(years_monthly, casc_mean - casc_std, casc_mean + casc_std,
                        color=SCENARIO_COLORS[scenario], alpha=0.15)

    ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
    ylim = ax.get_ylim()
    ax.text(INTV_YEAR - 0.1, ylim[1] * 0.95, 'Start of\nintervention',
            ha='right', va='top', fontsize=9, color='#4d4d4d')

    ax.set_xlabel('Year')
    ax.set_ylabel('Monthly HMB-related anemia cases')
    ax.set_title('Monthly HMB-related anemia: baseline vs cascade\n'
                 'by base care-seeking probability')
    ax.set_xlim([START, STOP])
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    sc.SIticks(ax=ax)

    plt.tight_layout()
    outpath = PLOTFOLDER + 'care_sa_monthly_cases.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


def plot_pct_reduction_timeseries(stats, years):
    """
    Single panel: % reduction in anemia cases over time post-intervention,
    with separate colored lines for each care-seeking scenario.

    Shaded bands = mean ± std across seeds.
    """
    post_mask = years >= INTV_YEAR

    fig, ax = plt.subplots(figsize=(8, 5))

    for scenario in CARE_SCENARIOS:
        s = stats[scenario]

        mean  = np.where(post_mask, s['mean'], np.nan)
        upper = np.where(post_mask, s['mean'] + s['std'], np.nan)
        lower = np.where(post_mask, s['mean'] - s['std'], np.nan)

        ax.plot(years, mean, color=SCENARIO_COLORS[scenario], lw=2.5,
                label=SCENARIO_LABELS[scenario])
        ax.fill_between(years, lower, upper,
                        color=SCENARIO_COLORS[scenario], alpha=0.15)

    ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
    ylim = ax.get_ylim()
    ax.text(INTV_YEAR - 0.2, ylim[1] * 0.95, 'Start of\nintervention',
            ha='right', va='top', fontsize=9, color='#4d4d4d')

    ax.set_xlabel('Year')
    ax.set_ylabel('% reduction in HMB-related anemia cases')
    ax.set_title('Sensitivity: % reduction in HMB-related anemia\n'
                 'by base care-seeking probability')
    ax.set_xlim([START, STOP])
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=10)

    plt.tight_layout()
    outpath = PLOTFOLDER + 'care_sa_pct_reduction_timeseries.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


def plot_pct_reduction_barchart(raw, stats, years):
    """
    Bar chart: mean % reduction in HMB-related anemia (post-intervention average),
    with error bars showing ± std.

    Two panels: (1) % reduction, (2) absolute averted cases.
    """
    post_mask = years >= INTV_YEAR

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Anemia reduction vs. no-intervention baseline\n'
                 '(post-intervention average)', fontsize=14)

    labels = []
    pct_means = []
    pct_stds = []
    abs_means = []
    abs_stds = []
    bar_colors = []

    for scenario in CARE_SCENARIOS:
        s = stats[scenario]
        labels.append(SCENARIO_LABELS[scenario])
        bar_colors.append(SCENARIO_COLORS[scenario])

        # Average over post-intervention years
        pct_means.append(np.nanmean(s['mean'][post_mask]))
        pct_stds.append(np.nanmean(s['std'][post_mask]))

    # Panel 1: % reduction
    ax = axes[0]
    bars = ax.bar(labels, pct_means, yerr=pct_stds,
                  color=bar_colors, alpha=0.7, capsize=5,
                  edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Care-seeking scenario')
    ax.set_ylabel('% reduction in HMB-related anemia')
    ax.set_title('Percentage reduction')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val, err in zip(bars, pct_means, pct_stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + err + 0.3,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    # Panel 2: Absolute averted cases (annual average, post-intervention)
    ax = axes[1]
    for scenario in CARE_SCENARIOS:
        averted_arr = np.array(raw[scenario]['averted'])  # (n_seeds, n_years)
        # Annual averted, averaged over post-intervention years per seed
        post_averted = averted_arr[:, post_mask]
        abs_means.append(post_averted.mean())
        abs_stds.append(post_averted.std(axis=0).mean())  # avg std across years

    bars = ax.bar(labels, [v / 1e6 for v in abs_means],
                  yerr=[v / 1e6 for v in abs_stds],
                  color=bar_colors, alpha=0.7, capsize=5,
                  edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Care-seeking scenario')
    ax.set_ylabel('Annual averted anemia cases (millions)')
    ax.set_title('Absolute reduction')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val, err in zip(bars, abs_means, abs_stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + err / 1e6 + 0.01,
                f'{val/1e6:.2f}m', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    plt.tight_layout()
    outpath = PLOTFOLDER + 'care_sa_pct_reduction_barchart.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


def plot_cascade_comparison(raw):
    """
    Single grouped bar chart: cascade depth distribution across scenarios.

    Uses mean ± std across seeds for each depth level.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    scenario_labels_list = list(CARE_SCENARIOS.keys())
    n_scenarios = len(scenario_labels_list)
    treatments_tried = np.arange(5)
    total_width = 0.7
    bar_width = total_width / n_scenarios

    for i, scenario in enumerate(scenario_labels_list):
        # Collect depth distributions across seeds
        depth_arrays = {n: [] for n in range(5)}
        for depth_dist in raw[scenario]['cascade_depth']:
            for n in range(5):
                depth_arrays[n].append(depth_dist[n])

        means = [np.mean(depth_arrays[n]) for n in range(5)]
        stds  = [np.std(depth_arrays[n]) for n in range(5)]

        offset = (i - (n_scenarios - 1) / 2) * bar_width
        ax.bar(treatments_tried + offset, means, bar_width,
               yerr=stds, capsize=3,
               label=SCENARIO_LABELS[scenario],
               color=SCENARIO_COLORS[scenario], alpha=0.7,
               edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Number of treatments tried')
    ax.set_ylabel('Percentage of menstruating women (%)')
    ax.set_title('Cascade progression by base care-seeking probability')
    ax.set_xticks(treatments_tried)
    ax.set_xticklabels(['0', '1', '2', '3', '4'])
    ax.legend(frameon=False, fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    outpath = PLOTFOLDER + 'care_sa_cascade_depth.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


def plot_impact_comparison(raw, years_monthly):
    """
    Single panel: total anemia and HMB-related anemia over time,
    comparing baseline vs intervention scenarios at monthly resolution.

    Baseline shown once (dashed), each scenario as solid colored line.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Intervention impact on anemia by care-seeking scenario', fontsize=14)

    # Panel 1: Total anemia
    ax = axes[0]

    # Baseline total anemia (average across scenarios)
    all_base_total = []
    for scenario in CARE_SCENARIOS:
        all_base_total.extend(raw[scenario]['baseline_total_anemia_monthly'])
    base_arr = np.array(all_base_total)
    base_mean = base_arr.mean(axis=0)
    base_std  = base_arr.std(axis=0)

    ax.plot(years_monthly, base_mean, color='#6c757d', lw=1.5, ls='--',
            label='No intervention')
    ax.fill_between(years_monthly, base_mean - base_std, base_mean + base_std,
                    color='#6c757d', alpha=0.15)

    for scenario in CARE_SCENARIOS:
        casc_arr  = np.array(raw[scenario]['cascade_total_anemia_monthly'])
        casc_mean = casc_arr.mean(axis=0)
        casc_std  = casc_arr.std(axis=0)

        ax.plot(years_monthly, casc_mean, color=SCENARIO_COLORS[scenario], lw=1.5,
                label=f'Cascade ({SCENARIO_LABELS[scenario]})')
        ax.fill_between(years_monthly, casc_mean - casc_std, casc_mean + casc_std,
                        color=SCENARIO_COLORS[scenario], alpha=0.15)

    ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Total anemia cases')
    ax.set_title('Total anemia')
    ax.set_xlim([START, STOP])
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    sc.SIticks(ax=ax)

    # Panel 2: HMB-related anemia
    ax = axes[1]

    all_base_hmb = []
    for scenario in CARE_SCENARIOS:
        all_base_hmb.extend(raw[scenario]['baseline_monthly'])
    base_arr = np.array(all_base_hmb)
    base_mean = base_arr.mean(axis=0)
    base_std  = base_arr.std(axis=0)

    ax.plot(years_monthly, base_mean, color='#6c757d', lw=1.5, ls='--',
            label='No intervention')
    ax.fill_between(years_monthly, base_mean - base_std, base_mean + base_std,
                    color='#6c757d', alpha=0.15)

    for scenario in CARE_SCENARIOS:
        casc_arr  = np.array(raw[scenario]['cascade_monthly'])
        casc_mean = casc_arr.mean(axis=0)
        casc_std  = casc_arr.std(axis=0)

        ax.plot(years_monthly, casc_mean, color=SCENARIO_COLORS[scenario], lw=1.5,
                label=f'Cascade ({SCENARIO_LABELS[scenario]})')
        ax.fill_between(years_monthly, casc_mean - casc_std, casc_mean + casc_std,
                        color=SCENARIO_COLORS[scenario], alpha=0.15)

    ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('HMB-related anemia cases')
    ax.set_title('HMB-related anemia')
    ax.set_xlim([START, STOP])
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    sc.SIticks(ax=ax)

    plt.tight_layout()
    outpath = PLOTFOLDER + 'care_sa_impact_comparison.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


def plot_care_seeking_rates(raw, years_monthly):
    """
    Two panels: care-seeking rates over time across scenarios.

    Panel 1: Overall care-seeking prevalence
    Panel 2: Stratified by anemia status (solid = anemic, dashed = not anemic)

    Shaded bands = mean ± std across seeds.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Care-seeking rates by base probability scenario', fontsize=14)

    # Panel 1: Overall
    ax = axes[0]
    for scenario in CARE_SCENARIOS:
        care_arr  = np.array(raw[scenario]['care_seeking_prev'])
        care_mean = care_arr.mean(axis=0) * 100
        care_std  = care_arr.std(axis=0) * 100

        ax.plot(years_monthly, care_mean, lw=1.5,
                label=SCENARIO_LABELS[scenario],
                color=SCENARIO_COLORS[scenario])
        ax.fill_between(years_monthly, care_mean - care_std, care_mean + care_std,
                        color=SCENARIO_COLORS[scenario], alpha=0.15)

    ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Care-seeking rate (%)')
    ax.set_title('Overall care-seeking rates')
    ax.set_xlim([START, STOP])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(alpha=0.3)

    # Panel 2: By anemia status
    ax = axes[1]
    for scenario in CARE_SCENARIOS:
        color = SCENARIO_COLORS[scenario]

        anemic_arr     = np.array(raw[scenario]['care_seeking_anemic'])
        not_anemic_arr = np.array(raw[scenario]['care_seeking_not_anemic'])

        anemic_mean     = anemic_arr.mean(axis=0) * 100
        not_anemic_mean = not_anemic_arr.mean(axis=0) * 100

        ax.plot(years_monthly, anemic_mean, lw=1.5,
                label=f'{SCENARIO_LABELS[scenario]} (anemic)',
                color=color, linestyle='-', alpha=0.8)
        ax.plot(years_monthly, not_anemic_mean, lw=1.5,
                label=f'{SCENARIO_LABELS[scenario]} (not anemic)',
                color=color, linestyle='--', alpha=0.6)

    ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
    ax.set_xlabel('Year')
    ax.set_ylabel('Care-seeking rate (%)')
    ax.set_title('Care-seeking rates by anemia status')
    ax.set_xlim([START, STOP])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    outpath = PLOTFOLDER + 'care_sa_care_seeking_rates.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


def plot_hiud_uptake(raw):
    """
    Bar chart: hIUD uptake by care-seeking scenario.

    Panel 1: % of HMB women seeking care who initiate hIUD
    Panel 2: % of all HMB women who ever used hIUD
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('hIUD uptake among women with HMB', fontsize=14)

    scenario_list = list(CARE_SCENARIOS.keys())
    labels = [SCENARIO_LABELS[s] for s in scenario_list]
    bar_colors = [SCENARIO_COLORS[s] for s in scenario_list]

    # Panel 1: % of HMB care-seekers who initiate hIUD
    ax = axes[0]
    means = []
    stds = []
    for scenario in scenario_list:
        vals = [u['pct_of_hmb_seekers'] for u in raw[scenario]['hiud_uptake']]
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    bars = ax.bar(labels, means, yerr=stds,
                  color=bar_colors, alpha=0.7, capsize=5,
                  edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Care-seeking scenario')
    ax.set_ylabel('% initiating hIUD')
    ax.set_title('% of HMB women seeking care\nwho initiate hIUD')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val, err in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + err + 0.3,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    # Panel 2: % of all HMB women using hIUD
    ax = axes[1]
    means = []
    stds = []
    for scenario in scenario_list:
        vals = [u['pct_of_hmb'] for u in raw[scenario]['hiud_uptake']]
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    bars = ax.bar(labels, means, yerr=stds,
                  color=bar_colors, alpha=0.7, capsize=5,
                  edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Care-seeking scenario')
    ax.set_ylabel('% using hIUD')
    ax.set_title('% of women with HMB\nwho ever used hIUD')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, val, err in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + err + 0.3,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=11, fontweight='bold')

    plt.tight_layout()
    outpath = PLOTFOLDER + 'care_sa_hiud_uptake.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


def print_hiud_uptake_summary(raw):
    """Print hIUD uptake summary table."""
    print(f"\n{'─'*75}")
    print("hIUD uptake among women with HMB (end-of-sim, mean ± std)")
    print(f"{'─'*75}")

    print(f"\n  {'Metric':<40}", end="")
    for scenario in CARE_SCENARIOS:
        print(f"  {SCENARIO_LABELS[scenario]:>14}", end="")
    print()
    print(f"  {'─'*70}")

    # % of HMB care-seekers
    print(f"  {'% of HMB seekers initiating hIUD':<40}", end="")
    for scenario in CARE_SCENARIOS:
        vals = [u['pct_of_hmb_seekers'] for u in raw[scenario]['hiud_uptake']]
        print(f"  {np.mean(vals):>6.1f}% ± {np.std(vals):>4.1f}%", end="")
    print()

    # % of all HMB women
    print(f"  {'% of HMB women ever using hIUD':<40}", end="")
    for scenario in CARE_SCENARIOS:
        vals = [u['pct_of_hmb'] for u in raw[scenario]['hiud_uptake']]
        print(f"  {np.mean(vals):>6.1f}% ± {np.std(vals):>4.1f}%", end="")
    print()

    # Absolute numbers
    print(f"\n  {'N HMB women seeking care':<40}", end="")
    for scenario in CARE_SCENARIOS:
        vals = [u['n_hmb_seekers'] for u in raw[scenario]['hiud_uptake']]
        print(f"  {np.mean(vals):>14.0f}", end="")
    print()

    print(f"  {'N initiating hIUD (among seekers)':<40}", end="")
    for scenario in CARE_SCENARIOS:
        vals = [u['n_hiud_among_seekers'] for u in raw[scenario]['hiud_uptake']]
        print(f"  {np.mean(vals):>14.0f}", end="")
    print()

    print(f"  {'N HMB women total':<40}", end="")
    for scenario in CARE_SCENARIOS:
        vals = [u['n_hmb'] for u in raw[scenario]['hiud_uptake']]
        print(f"  {np.mean(vals):>14.0f}", end="")
    print()

    print(f"  {'N ever used hIUD (among all HMB)':<40}", end="")
    for scenario in CARE_SCENARIOS:
        vals = [u['n_hiud_hmb'] for u in raw[scenario]['hiud_uptake']]
        print(f"  {np.mean(vals):>14.0f}", end="")
    print()

    print(f"{'─'*75}\n")
    
    
# ── Summary table ──────────────────────────────────────────────────────────────
def print_summary(stats, years):
    """Print mean % reduction averaged over post-intervention years."""
    post_mask = years >= INTV_YEAR

    print(f"\n{'─'*65}")
    print(f"Mean % reduction in HMB-related anemia (post-intervention, "
          f"{INTV_YEAR}–{STOP})")
    print(f"{'─'*65}")
    print(f"  {'Scenario':<18}  {'Mean %':>8}  {'± Std'}")
    print(f"{'─'*65}")

    for scenario in CARE_SCENARIOS:
        s = stats[scenario]
        m   = np.nanmean(s['mean'][post_mask])
        std = np.nanmean(s['std'][post_mask])
        print(f"  {SCENARIO_LABELS[scenario]:<18}  {m:>7.1f}%  ± {std:.1f}%")

    print(f"{'─'*65}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    do_run = True   # Set False to load saved results

    # Run (or load) simulations
    raw = run_care_sensitivity(force_rerun=do_run)

    # Build year axis
    n_years = len(raw['10%']['baseline'][0])
    years = np.arange(START, START + n_years)

    # Build monthly time axis
    n_months = len(raw['10%']['baseline_monthly'][0])
    years_monthly = np.array([START + m / 12 for m in range(n_months)])

    # Compute % reduction statistics
    stats = compute_stats(raw)

    # Plots
    plot_annual_cases(raw, years)
    plot_monthly_cases(raw, years_monthly)
    plot_pct_reduction_timeseries(stats, years)
    plot_pct_reduction_barchart(raw, stats, years)
    plot_impact_comparison(raw, years_monthly)
    plot_care_seeking_rates(raw, years_monthly)
    plot_cascade_comparison(raw)
    plot_hiud_uptake(raw)
    print_hiud_uptake_summary(raw)

    # Summary table
    print_summary(stats, years)