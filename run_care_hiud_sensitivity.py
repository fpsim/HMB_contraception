# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 14:29:17 2026

@author: navidehno
"""

"""
Sensitivity analysis: care-seeking probability × hIUD uptake target

3 × 2 grid:
  Care-seeking base: 10%, 20%, 35%
  hIUD uptake target: ~5% (accept=0.50), ~10% (accept=0.85)

Interventions compared:
  - baseline  : no intervention
  - cascade   : full HMBCascade (NSAID → TXA → Pill → hIUD)
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
PLOTFOLDER = 'figures_care_hiud_sa/'
OUTFOLDER  = 'results_care_hiud_sa/'
for d in [PLOTFOLDER, OUTFOLDER]:
    os.makedirs(d, exist_ok=True)


# ── Settings ───────────────────────────────────────────────────────────────────
N_SEEDS   = 10
START     = 2020
STOP      = 2030
INTV_YEAR = 2026

# Care-seeking scenarios
CARE_SCENARIOS = {
    '10%': sc.objdict(base=0.10, anemic=1.43, pain=0.61),
    '20%': sc.objdict(base=0.20, anemic=0.86, pain=0.37),
    '35%': sc.objdict(base=0.35, anemic=0.32, pain=0.14),
}

# hIUD uptake scenarios (acceptance probabilities from calibration)
HIUD_SCENARIOS = {
    '5%':  0.50,   # produces ~5% of HMB women on hIUD
    '10%': 0.85,   # produces ~10% of HMB women on hIUD
}

# Labels and colors
CARE_LABELS = {
    '10%': 'Care 10%',
    '20%': 'Care 20%',
    '35%': 'Care 35%',
}
CARE_COLORS = {
    '10%': '#d62728',
    '20%': '#ff7f0e',
    '35%': '#2196F3',
}
HIUD_LINESTYLES = {
    '5%':  '-',
    '10%': '--',
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def _annualize(monthly_arr, how='sum'):
    arr = np.asarray(monthly_arr)
    n_years = len(arr) // 12
    arr = arr[:12 * n_years].reshape(n_years, 12)
    if how == 'sum':
        return arr.sum(axis=1)
    elif how == 'eoy':
        return arr[:, -1]


def scenario_key(care_label, hiud_label):
    """Create a unique key for each scenario combination."""
    return f'{care_label}_hiud{hiud_label}'


# ── Simulation creation ───────────────────────────────────────────────────────
def make_sim(care_behavior=None, hiud_accept=0.5, with_intervention=True, seed=0):
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
                hiud=sc.objdict(
                    efficacy=0.8,
                    adherence=0.85,
                    prob_offer=ss.bernoulli(p=0.9),
                    prob_accept=ss.bernoulli(p=hiud_accept),
                ),
            )
        )
        sim_kwargs['interventions'] = [cascade]
        sim_kwargs['analyzers'].extend([
            track_care_seeking(),
            track_tx_eff(),
            track_tx_dur(),
            track_cascade(),
        ])

    return fp.Sim(**sim_kwargs)


# ── Run simulations ────────────────────────────────────────────────────────────
def run_sa(force_rerun=True):
    results_file = OUTFOLDER + 'care_hiud_sa_raw.obj'

    if not force_rerun and os.path.exists(results_file):
        print("Loading saved results...")
        return sc.loadobj(results_file)

    raw = {}
    for care_label, care_behavior in CARE_SCENARIOS.items():
        for hiud_label, hiud_accept in HIUD_SCENARIOS.items():
            key = scenario_key(care_label, hiud_label)
            raw[key] = {
                'baseline': [], 'cascade': [], 'averted': [],
                'baseline_monthly': [], 'cascade_monthly': [],
                'baseline_total_anemia_monthly': [],
                'cascade_total_anemia_monthly': [],
                'care_seeking_prev': [],
                'care_seeking_anemic': [],
                'care_seeking_not_anemic': [],
                'cascade_depth': [],
                'hiud_uptake': [],
                'care_label': care_label,
                'hiud_label': hiud_label,
            }

    for care_label, care_behavior in CARE_SCENARIOS.items():
        for hiud_label, hiud_accept in HIUD_SCENARIOS.items():
            key = scenario_key(care_label, hiud_label)
            print(f"\n=== {CARE_LABELS[care_label]}, hIUD target={hiud_label} "
                  f"(accept={hiud_accept:.2f}) ===")

            for seed in range(N_SEEDS):
                print(f"  seed {seed}...", end=" ", flush=True)

                sims = [
                    make_sim(with_intervention=False, seed=seed),
                    make_sim(care_behavior=care_behavior, hiud_accept=hiud_accept,
                             with_intervention=True, seed=seed),
                ]
                msim = ss.MultiSim(sims)
                msim.run()

                s_base    = msim.sims[0]
                s_cascade = msim.sims[1]

                # HMB-related anemia
                base_monthly    = s_base.results.track_hmb_anemia['n_anemia_with_hmb']
                cascade_monthly = s_cascade.results.track_hmb_anemia['n_anemia_with_hmb']

                raw[key]['baseline_monthly'].append(np.asarray(base_monthly))
                raw[key]['cascade_monthly'].append(np.asarray(cascade_monthly))

                # Total anemia
                raw[key]['baseline_total_anemia_monthly'].append(
                    np.asarray(s_base.results.track_hmb_anemia['n_anemia_total']))
                raw[key]['cascade_total_anemia_monthly'].append(
                    np.asarray(s_cascade.results.track_hmb_anemia['n_anemia_total']))

                # Care-seeking rates
                raw[key]['care_seeking_prev'].append(
                    np.asarray(s_cascade.results.track_care_seeking['care_seeking_prev']))
                raw[key]['care_seeking_anemic'].append(
                    np.asarray(s_cascade.results.track_care_seeking['care_seeking_anemic']))
                raw[key]['care_seeking_not_anemic'].append(
                    np.asarray(s_cascade.results.track_care_seeking['care_seeking_not_anemic']))

                # Cascade depth
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
                raw[key]['cascade_depth'].append(depth_dist)

                # hIUD uptake — denominator includes women whose HMB is suppressed by treatment
                hmb = s_cascade.people.menstruation.hmb
                hmb_underlying = (hmb | cascade_intv.on_any_treatment) & menstruating
                n_hmb = np.count_nonzero(hmb_underlying)

                ever_offered_nsaid = cascade_intv.treatments['nsaid'].offered & hmb_underlying
                n_hmb_seekers = np.count_nonzero(ever_offered_nsaid)
                tried_hiud_seekers = cascade_intv.treatments['hiud'].tried_treatment & ever_offered_nsaid
                n_hiud_seekers = np.count_nonzero(tried_hiud_seekers)

                tried_hiud_hmb = cascade_intv.treatments['hiud'].tried_treatment & hmb_underlying
                n_hiud_hmb = np.count_nonzero(tried_hiud_hmb)

                hiud_uptake = {
                    'pct_of_hmb_seekers': 100 * n_hiud_seekers / n_hmb_seekers if n_hmb_seekers > 0 else 0,
                    'pct_of_hmb': 100 * n_hiud_hmb / n_hmb if n_hmb > 0 else 0,
                    'n_hmb_seekers': n_hmb_seekers,
                    'n_hiud_among_seekers': n_hiud_seekers,
                    'n_hmb': n_hmb,
                    'n_hiud_hmb': n_hiud_hmb,
                }
                raw[key]['hiud_uptake'].append(hiud_uptake)

                # Annualize
                base_annual    = _annualize(base_monthly)
                cascade_annual = _annualize(cascade_monthly)
                averted_annual = base_annual - cascade_annual

                raw[key]['baseline'].append(base_annual)
                raw[key]['cascade'].append(cascade_annual)
                raw[key]['averted'].append(averted_annual)

                del msim
                gc.collect()
                print("done")

    sc.saveobj(results_file, raw)
    print(f"\nSaved: {results_file}")
    return raw


# ── Aggregate statistics ───────────────────────────────────────────────────────
def compute_stats(raw):
    stats = {}
    for key in raw:
        if key.startswith('_'):
            continue
        base_arr    = np.array(raw[key]['baseline'])
        averted_arr = np.array(raw[key]['averted'])
        pct = np.where(base_arr > 0, averted_arr / base_arr * 100, np.nan)
        stats[key] = {
            'mean': np.nanmean(pct, axis=0),
            'std':  np.nanstd(pct, axis=0),
        }
    return stats


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_monthly_cases(raw, years_monthly):
    """Monthly HMB-related anemia: one panel per hIUD scenario."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Monthly HMB-related anemia by care-seeking and hIUD uptake', fontsize=14)

    for idx, hiud_label in enumerate(HIUD_SCENARIOS):
        ax = axes[idx]

        # Baseline (same across scenarios)
        all_baselines = []
        for care_label in CARE_SCENARIOS:
            key = scenario_key(care_label, hiud_label)
            all_baselines.extend(raw[key]['baseline_monthly'])
        base_arr = np.array(all_baselines)
        base_mean = base_arr.mean(axis=0)
        base_std  = base_arr.std(axis=0)

        ax.plot(years_monthly, base_mean, color='#6c757d', lw=1.5, ls='--',
                label='No intervention')
        ax.fill_between(years_monthly, base_mean - base_std, base_mean + base_std,
                        color='#6c757d', alpha=0.15)

        for care_label in CARE_SCENARIOS:
            key = scenario_key(care_label, hiud_label)
            casc_arr  = np.array(raw[key]['cascade_monthly'])
            casc_mean = casc_arr.mean(axis=0)
            casc_std  = casc_arr.std(axis=0)

            ax.plot(years_monthly, casc_mean, color=CARE_COLORS[care_label], lw=1.5,
                    label=f'{CARE_LABELS[care_label]}')
            ax.fill_between(years_monthly, casc_mean - casc_std, casc_mean + casc_std,
                            color=CARE_COLORS[care_label], alpha=0.15)

        ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
        ax.set_xlabel('Year')
        ax.set_ylabel('Monthly HMB-related anemia cases')
        ax.set_title(f'hIUD uptake target: {hiud_label}')
        ax.set_xlim([START, STOP])
        ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
        sc.SIticks(ax=ax)

    plt.tight_layout()
    outpath = PLOTFOLDER + 'sa_monthly_cases.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


def plot_impact_comparison(raw, years_monthly):
    """Total anemia and HMB-related anemia: one panel per hIUD scenario."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Intervention impact by care-seeking and hIUD uptake', fontsize=14)

    for col, hiud_label in enumerate(HIUD_SCENARIOS):
        # Row 0: Total anemia
        ax = axes[0, col]
        all_base = []
        for care_label in CARE_SCENARIOS:
            key = scenario_key(care_label, hiud_label)
            all_base.extend(raw[key]['baseline_total_anemia_monthly'])
        base_arr = np.array(all_base)
        base_mean = base_arr.mean(axis=0)
        base_std  = base_arr.std(axis=0)

        ax.plot(years_monthly, base_mean, color='#6c757d', lw=1.5, ls='--', label='No intervention')
        ax.fill_between(years_monthly, base_mean - base_std, base_mean + base_std,
                        color='#6c757d', alpha=0.15)

        for care_label in CARE_SCENARIOS:
            key = scenario_key(care_label, hiud_label)
            casc_arr = np.array(raw[key]['cascade_total_anemia_monthly'])
            casc_mean = casc_arr.mean(axis=0)
            casc_std  = casc_arr.std(axis=0)
            ax.plot(years_monthly, casc_mean, color=CARE_COLORS[care_label], lw=1.5,
                    label=CARE_LABELS[care_label])
            ax.fill_between(years_monthly, casc_mean - casc_std, casc_mean + casc_std,
                            color=CARE_COLORS[care_label], alpha=0.15)

        ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
        ax.set_title(f'Total anemia — hIUD target: {hiud_label}')
        ax.set_xlabel('Year')
        ax.set_ylabel('Total anemia cases')
        ax.set_xlim([START, STOP])
        ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        sc.SIticks(ax=ax)

        # Row 1: HMB-related anemia
        ax = axes[1, col]
        all_base_hmb = []
        for care_label in CARE_SCENARIOS:
            key = scenario_key(care_label, hiud_label)
            all_base_hmb.extend(raw[key]['baseline_monthly'])
        base_arr = np.array(all_base_hmb)
        base_mean = base_arr.mean(axis=0)
        base_std  = base_arr.std(axis=0)

        ax.plot(years_monthly, base_mean, color='#6c757d', lw=1.5, ls='--', label='No intervention')
        ax.fill_between(years_monthly, base_mean - base_std, base_mean + base_std,
                        color='#6c757d', alpha=0.15)

        for care_label in CARE_SCENARIOS:
            key = scenario_key(care_label, hiud_label)
            casc_arr = np.array(raw[key]['cascade_monthly'])
            casc_mean = casc_arr.mean(axis=0)
            casc_std  = casc_arr.std(axis=0)
            ax.plot(years_monthly, casc_mean, color=CARE_COLORS[care_label], lw=1.5,
                    label=CARE_LABELS[care_label])
            ax.fill_between(years_monthly, casc_mean - casc_std, casc_mean + casc_std,
                            color=CARE_COLORS[care_label], alpha=0.15)

        ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
        ax.set_title(f'HMB-related anemia — hIUD target: {hiud_label}')
        ax.set_xlabel('Year')
        ax.set_ylabel('HMB-related anemia cases')
        ax.set_xlim([START, STOP])
        ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        sc.SIticks(ax=ax)

    plt.tight_layout()
    outpath = PLOTFOLDER + 'sa_impact_comparison.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


def plot_pct_reduction_timeseries(stats, years):
    """% reduction over time: all 6 scenarios on one panel."""
    post_mask = years >= INTV_YEAR
    fig, ax = plt.subplots(figsize=(10, 6))

    for care_label in CARE_SCENARIOS:
        for hiud_label in HIUD_SCENARIOS:
            key = scenario_key(care_label, hiud_label)
            s = stats[key]

            mean  = np.where(post_mask, s['mean'], np.nan)
            upper = np.where(post_mask, s['mean'] + s['std'], np.nan)
            lower = np.where(post_mask, s['mean'] - s['std'], np.nan)

            ax.plot(years, mean, color=CARE_COLORS[care_label],
                    ls=HIUD_LINESTYLES[hiud_label], lw=2.5,
                    label=f'{CARE_LABELS[care_label]}, hIUD {hiud_label}')
            ax.fill_between(years, lower, upper,
                            color=CARE_COLORS[care_label], alpha=0.08)

    ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
    ylim = ax.get_ylim()
    ax.text(INTV_YEAR - 0.2, ylim[1] * 0.95, 'Start of\nintervention',
            ha='right', va='top', fontsize=9, color='#4d4d4d')

    ax.set_xlabel('Year')
    ax.set_ylabel('% reduction in HMB-related anemia cases')
    ax.set_title('% reduction in HMB-related anemia\nby care-seeking and hIUD uptake target')
    ax.set_xlim([START, STOP])
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=9, ncol=2)

    plt.tight_layout()
    outpath = PLOTFOLDER + 'sa_pct_reduction_timeseries.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


def plot_pct_reduction_barchart(raw, stats, years):
    """Grouped bar chart: % reduction and absolute averted, grouped by hIUD target."""
    post_mask = years >= INTV_YEAR

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Anemia reduction vs. no-intervention baseline\n'
                 '(post-intervention average)', fontsize=14)

    care_list = list(CARE_SCENARIOS.keys())
    hiud_list = list(HIUD_SCENARIOS.keys())
    n_care = len(care_list)
    n_hiud = len(hiud_list)
    n_groups = n_care
    n_bars = n_hiud

    x = np.arange(n_groups)
    total_width = 0.6
    bar_width = total_width / n_bars

    # Panel 1: % reduction
    ax = axes[0]
    for i, hiud_label in enumerate(hiud_list):
        means = []
        stds = []
        for care_label in care_list:
            key = scenario_key(care_label, hiud_label)
            s = stats[key]
            means.append(np.nanmean(s['mean'][post_mask]))
            stds.append(np.nanmean(s['std'][post_mask]))

        offset = (i - (n_bars - 1) / 2) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=stds, capsize=4,
                      label=f'hIUD {hiud_label}',
                      alpha=0.7 if i == 0 else 0.5,
                      edgecolor='black', linewidth=0.5,
                      color=[CARE_COLORS[c] for c in care_list])

        for bar, val, err in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + err + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax.set_xlabel('Care-seeking scenario')
    ax.set_ylabel('% reduction in HMB-related anemia')
    ax.set_title('Percentage reduction')
    ax.set_xticks(x)
    ax.set_xticklabels([CARE_LABELS[c] for c in care_list])
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 2: Absolute averted
    ax = axes[1]
    for i, hiud_label in enumerate(hiud_list):
        means = []
        stds = []
        for care_label in care_list:
            key = scenario_key(care_label, hiud_label)
            averted_arr = np.array(raw[key]['averted'])
            post_averted = averted_arr[:, post_mask]
            means.append(post_averted.mean())
            stds.append(post_averted.std(axis=0).mean())

        offset = (i - (n_bars - 1) / 2) * bar_width
        bars = ax.bar(x + offset, [v / 1e6 for v in means], bar_width,
                      yerr=[v / 1e6 for v in stds], capsize=4,
                      label=f'hIUD {hiud_label}',
                      alpha=0.7 if i == 0 else 0.5,
                      edgecolor='black', linewidth=0.5,
                      color=[CARE_COLORS[c] for c in care_list])

        for bar, val, err in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + err / 1e6 + 0.01,
                    f'{val/1e6:.2f}m', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax.set_xlabel('Care-seeking scenario')
    ax.set_ylabel('Annual averted anemia cases (millions)')
    ax.set_title('Absolute reduction')
    ax.set_xticks(x)
    ax.set_xticklabels([CARE_LABELS[c] for c in care_list])
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    outpath = PLOTFOLDER + 'sa_pct_reduction_barchart.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


def plot_hiud_uptake(raw):
    """hIUD uptake bar chart: grouped by care-seeking, colored by hIUD scenario."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('hIUD uptake among women with HMB', fontsize=14)

    care_list = list(CARE_SCENARIOS.keys())
    hiud_list = list(HIUD_SCENARIOS.keys())
    n_care = len(care_list)
    n_hiud = len(hiud_list)

    x = np.arange(n_care)
    total_width = 0.6
    bar_width = total_width / n_hiud

    hiud_bar_colors = {'5%': '#4CAF50', '10%': '#F44336'}

    # Panel 1: % of HMB seekers
    ax = axes[0]
    for i, hiud_label in enumerate(hiud_list):
        means = []
        stds = []
        for care_label in care_list:
            key = scenario_key(care_label, hiud_label)
            vals = [u['pct_of_hmb_seekers'] for u in raw[key]['hiud_uptake']]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        offset = (i - (n_hiud - 1) / 2) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=stds, capsize=4,
                      label=f'hIUD target {hiud_label}',
                      color=hiud_bar_colors[hiud_label], alpha=0.7,
                      edgecolor='black', linewidth=0.5)

        for bar, val, err in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + err + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax.set_xlabel('Care-seeking scenario')
    ax.set_ylabel('% initiating hIUD')
    ax.set_title('% of HMB women seeking care\nwho initiate hIUD')
    ax.set_xticks(x)
    ax.set_xticklabels([CARE_LABELS[c] for c in care_list])
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel 2: % of all HMB women
    ax = axes[1]
    for i, hiud_label in enumerate(hiud_list):
        means = []
        stds = []
        for care_label in care_list:
            key = scenario_key(care_label, hiud_label)
            vals = [u['pct_of_hmb'] for u in raw[key]['hiud_uptake']]
            means.append(np.mean(vals))
            stds.append(np.std(vals))

        offset = (i - (n_hiud - 1) / 2) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=stds, capsize=4,
                      label=f'hIUD target {hiud_label}',
                      color=hiud_bar_colors[hiud_label], alpha=0.7,
                      edgecolor='black', linewidth=0.5)

        for bar, val, err in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + err + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax.set_xlabel('Care-seeking scenario')
    ax.set_ylabel('% using hIUD')
    ax.set_title('% of women with HMB\nwho ever used hIUD')
    ax.set_xticks(x)
    ax.set_xticklabels([CARE_LABELS[c] for c in care_list])
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    outpath = PLOTFOLDER + 'sa_hiud_uptake.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


def plot_cascade_comparison(raw):
    """Cascade depth: one panel per hIUD scenario."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Cascade progression by care-seeking and hIUD uptake', fontsize=14)

    care_list = list(CARE_SCENARIOS.keys())
    n_care = len(care_list)
    treatments_tried = np.arange(5)
    total_width = 0.7
    bar_width = total_width / n_care

    for col, hiud_label in enumerate(HIUD_SCENARIOS):
        ax = axes[col]

        for i, care_label in enumerate(care_list):
            key = scenario_key(care_label, hiud_label)
            depth_arrays = {n: [] for n in range(5)}
            for depth_dist in raw[key]['cascade_depth']:
                for n in range(5):
                    depth_arrays[n].append(depth_dist[n])

            means = [np.mean(depth_arrays[n]) for n in range(5)]
            stds  = [np.std(depth_arrays[n]) for n in range(5)]

            offset = (i - (n_care - 1) / 2) * bar_width
            ax.bar(treatments_tried + offset, means, bar_width,
                   yerr=stds, capsize=3,
                   label=CARE_LABELS[care_label],
                   color=CARE_COLORS[care_label], alpha=0.7,
                   edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Number of treatments tried')
        ax.set_ylabel('Percentage of menstruating women (%)')
        ax.set_title(f'hIUD uptake target: {hiud_label}')
        ax.set_xticks(treatments_tried)
        ax.set_xticklabels(['0', '1', '2', '3', '4'])
        ax.legend(frameon=False, fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    outpath = PLOTFOLDER + 'sa_cascade_depth.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


# ── Summary tables ─────────────────────────────────────────────────────────────
def print_summary(raw, stats, years):
    post_mask = years >= INTV_YEAR

    print(f"\n{'═'*80}")
    print(f"  SENSITIVITY ANALYSIS: Care-seeking × hIUD uptake")
    print(f"  Post-intervention average ({INTV_YEAR}–{STOP})")
    print(f"{'═'*80}")

    # % reduction table
    print(f"\n  % Reduction in HMB-related anemia:")
    print(f"  {'':>12}", end="")
    for hiud_label in HIUD_SCENARIOS:
        print(f"  {'hIUD ' + hiud_label:>16}", end="")
    print()
    print(f"  {'─'*50}")

    for care_label in CARE_SCENARIOS:
        print(f"  {CARE_LABELS[care_label]:>12}", end="")
        for hiud_label in HIUD_SCENARIOS:
            key = scenario_key(care_label, hiud_label)
            s = stats[key]
            m = np.nanmean(s['mean'][post_mask])
            std = np.nanmean(s['std'][post_mask])
            print(f"  {m:>8.1f}% ± {std:.1f}%", end="")
        print()

    # hIUD uptake table
    print(f"\n  % of HMB women who ever used hIUD:")
    print(f"  {'':>12}", end="")
    for hiud_label in HIUD_SCENARIOS:
        print(f"  {'hIUD ' + hiud_label:>16}", end="")
    print()
    print(f"  {'─'*50}")

    for care_label in CARE_SCENARIOS:
        print(f"  {CARE_LABELS[care_label]:>12}", end="")
        for hiud_label in HIUD_SCENARIOS:
            key = scenario_key(care_label, hiud_label)
            vals = [u['pct_of_hmb'] for u in raw[key]['hiud_uptake']]
            print(f"  {np.mean(vals):>8.1f}% ± {np.std(vals):.1f}%", end="")
        print()

    # Absolute averted table
    print(f"\n  Annual averted HMB-related anemia cases (millions):")
    print(f"  {'':>12}", end="")
    for hiud_label in HIUD_SCENARIOS:
        print(f"  {'hIUD ' + hiud_label:>16}", end="")
    print()
    print(f"  {'─'*50}")

    for care_label in CARE_SCENARIOS:
        print(f"  {CARE_LABELS[care_label]:>12}", end="")
        for hiud_label in HIUD_SCENARIOS:
            key = scenario_key(care_label, hiud_label)
            averted_arr = np.array(raw[key]['averted'])
            post_averted = averted_arr[:, post_mask]
            m = post_averted.mean() / 1e6
            std = post_averted.std(axis=0).mean() / 1e6
            print(f"  {m:>8.2f}m ± {std:.2f}m", end="")
        print()

    print(f"{'═'*80}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    do_run = True

    raw = run_sa(force_rerun=do_run)

    # Build axes
    first_key = list(raw.keys())[0]
    n_years = len(raw[first_key]['baseline'][0])
    years = np.arange(START, START + n_years)

    n_months = len(raw[first_key]['baseline_monthly'][0])
    years_monthly = np.array([START + m / 12 for m in range(n_months)])

    stats = compute_stats(raw)

    # Plots
    plot_monthly_cases(raw, years_monthly)
    plot_impact_comparison(raw, years_monthly)
    plot_pct_reduction_timeseries(stats, years)
    plot_pct_reduction_barchart(raw, stats, years)
    plot_hiud_uptake(raw)
    plot_cascade_comparison(raw)

    # Summary
    print_summary(raw, stats, years)