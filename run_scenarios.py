# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:18:22 2026

@author: navidehno
"""

"""
Scenario comparison: Darcy's 3×3 intervention packages vs status quo.

Care-seeking levels (rows):
  Base:  10% ever-seek, 1/36 monthly first
  Mid:   20% ever-seek, 1/12 monthly first
  High:  35% ever-seek, 1/6 monthly first

hIUD uptake levels (columns):
  Low:   10% of seekers get hIUD, 73% offered treatment
  Mid:   25% of seekers get hIUD, 80% offered treatment
  High:  40% of seekers get hIUD, 90% offered treatment

tx_weights = % of those OFFERED treatment (sums to ~100).
prob_offer = global gate for % who get any treatment.
Together: prob_offer × tx_weight_i = % of seekers getting treatment i.

All use mid RR (1.73) throughout.
Pre-2026: all sims run status quo (10% care, 70% offer, no hIUD).
Post-2026: each sim switches to its scenario.

Outcomes: HMB prevalence and HMB-related anemia.
"""

import numpy as np
import sciris as sc
import starsim as ss
import fpsim as fp
import os
import gc
import matplotlib.pyplot as plt

from menstruation import Menstruation
from education import Education
from interventions_pool import HMBPool, HMBCounterfactual
from analyzers import track_hmb_anemia


# ── Output folders ─────────────────────────────────────────────────────────────
PLOTFOLDER = 'figures_scenarios/'
OUTFOLDER  = 'results_scenarios/'
for d in [PLOTFOLDER, OUTFOLDER]:
    os.makedirs(d, exist_ok=True)


# ── Settings ───────────────────────────────────────────────────────────────────
P_BASE      = 0.215
P_HMB_PRONE = 0.53
RR_MID      = 1.73
N_SEEDS     = 10
START       = 2020
STOP        = 2030
INTV_YEAR   = 2026


# ── Care-seeking levels ────────────────────────────────────────────────────────

CARE_PRE = sc.objdict(
    p_ever_seek=0.10,
    p_monthly_first=1/36,
    p_monthly_first_anemic=1/12,
    p_monthly_first_pain=1/24,
    p_monthly_repeat=1/6,
)

CARE_LEVELS = {
    'base': sc.objdict(
        label='Base care (10%)',
        care=sc.objdict(
            p_ever_seek=0.10,
            p_monthly_first=1/36,
            p_monthly_first_anemic=1/12,
            p_monthly_first_pain=1/24,
            p_monthly_repeat=1/6,
        ),
    ),
    'mid': sc.objdict(
        label='Mid care (20%)',
        care=sc.objdict(
            p_ever_seek=0.20,
            p_monthly_first=1/12,
            p_monthly_first_anemic=1/6,
            p_monthly_first_pain=1/12,
            p_monthly_repeat=1/6,
        ),
    ),
    'high': sc.objdict(
        label='High care (35%)',
        care=sc.objdict(
            p_ever_seek=0.35,
            p_monthly_first=1/6,
            p_monthly_first_anemic=1/6,
            p_monthly_first_pain=1/6,
            p_monthly_repeat=1/6,
        ),
    ),
}

# ── hIUD uptake levels ─────────────────────────────────────────────────────────
# tx_weights: % of those OFFERED treatment (sums to ~100)
# Derived from Darcy's table: raw % of seekers / prob_offer

HIUD_LEVELS = {
    'low_hiud': sc.objdict(
        label='Low hIUD (10%)',
        prob_offer=0.73,
        tx_weights=sc.objdict(nsaid=43.2, txa=21.6, pill=21.6, hiud=13.7),
    ),
    'mid_hiud': sc.objdict(
        label='Mid hIUD (25%)',
        prob_offer=0.80,
        tx_weights=sc.objdict(nsaid=34.4, txa=17.2, pill=17.2, hiud=31.3),
    ),
    'high_hiud': sc.objdict(
        label='High hIUD (40%)',
        prob_offer=0.90,
        tx_weights=sc.objdict(nsaid=27.8, txa=13.9, pill=13.9, hiud=44.4),
    ),
}

# ── Colors ─────────────────────────────────────────────────────────────────────
# 3×3 color matrix: rows=care, columns=hIUD, light→dark within each row
CARE_COLORS = {
    'base': {'low_hiud': '#BBDEFB', 'mid_hiud': '#64B5F6', 'high_hiud': '#1565C0'},
    'mid':  {'low_hiud': '#C8E6C9', 'mid_hiud': '#66BB6A', 'high_hiud': '#2E7D32'},
    'high': {'low_hiud': '#FFCDD2', 'mid_hiud': '#EF5350', 'high_hiud': '#B71C1C'},
}

# For plots paneled by hIUD level, color by care level
CARE_LINE_COLORS = {'base': '#2196F3', 'mid': '#4CAF50', 'high': '#F44336'}

# For plots paneled by care level, color by hIUD level
HIUD_LINE_COLORS = {'low_hiud': '#90CAF9', 'mid_hiud': '#4CAF50', 'high_hiud': '#F44336'}


# ── Helpers ────────────────────────────────────────────────────────────────────
def rr_to_logistic_coeff(rr, p_base=P_BASE):
    p_hmb = np.clip(p_base * rr, 1e-6, 1 - 1e-6)
    return (-np.log(1 / p_hmb - 1)) - (-np.log(1 / p_base - 1))


def make_menstruation():
    coeff = rr_to_logistic_coeff(RR_MID)
    return Menstruation(pars={
        'p_hmb_prone': ss.bernoulli(p=P_HMB_PRONE),
        'hmb_seq': sc.objdict(
            poor_mh=sc.objdict(base=0.4, hmb=1.0),
            anemic=sc.objdict(base=P_BASE, hmb=coeff),
            pain=sc.objdict(base=0.1, hmb=3.36),
        )
    })


def make_intervention(care_name, hiud_name):
    """
    Build intervention for a care × hIUD combination.
    Pre-2026: status quo. Post-2026: scenario-specific.
    """
    care_info = CARE_LEVELS[care_name]
    hiud_info = HIUD_LEVELS[hiud_name]

    return HMBPool(pars=dict(
        year=2020,
        intv_year=INTV_YEAR,

        care_behavior_pre=CARE_PRE,
        care_behavior_post=care_info.care,

        prob_offer_pre=0.70,
        prob_offer_post=hiud_info.prob_offer,

        tx_weights_pre=sc.objdict(nsaid=50.0, txa=25.0, pill=25.0, hiud=0.0),
        tx_weights_post=hiud_info.tx_weights,

        nsaid=sc.objdict(efficacy=0.33, adherence=0.80),
        txa=sc.objdict(efficacy=0.45, adherence=0.70),
        pill=sc.objdict(efficacy=0.59, adherence=0.80),
        hiud=sc.objdict(efficacy=0.88, adherence=1.00),
    ))


def make_counterfactual():
    return HMBCounterfactual(pars=dict(
        care_behavior_pre=CARE_PRE,
        care_behavior_post=CARE_PRE,
    ))


def make_sim(scenario, seed=0):
    """
    scenario: 'counterfactual' or tuple (care_name, hiud_name)
    """
    mens = make_menstruation()
    edu = Education()

    if scenario == 'counterfactual':
        intervention = make_counterfactual()
    else:
        care_name, hiud_name = scenario
        intervention = make_intervention(care_name, hiud_name)

    return fp.Sim(
        start=START, stop=STOP,
        n_agents=10000, total_pop=55_000_000,
        location='kenya',
        education_module=edu,
        connectors=[mens],
        interventions=[intervention],
        analyzers=[track_hmb_anemia()],
        rand_seed=seed, verbose=0,
    )


# ── Run ────────────────────────────────────────────────────────────────────────
def _annualize(monthly_arr, how='sum'):
    arr = np.asarray(monthly_arr)
    n_y = len(arr) // 12
    arr = arr[:12 * n_y].reshape(n_y, 12)
    return arr.sum(axis=1) if how == 'sum' else arr[:, -1]


def _annualize_mean(monthly_arr):
    arr = np.asarray(monthly_arr)
    n_y = len(arr) // 12
    arr = arr[:12 * n_y].reshape(n_y, 12)
    return arr.mean(axis=1)


def run_scenarios(force_rerun=True):
    results_file = OUTFOLDER + 'scenario_3x3_raw.obj'

    if not force_rerun and os.path.exists(results_file):
        print("Loading saved results...")
        return sc.loadobj(results_file)

    # Build scenario list: counterfactual + 3×3
    all_scenarios = ['counterfactual']
    for care_name in CARE_LEVELS:
        for hiud_name in HIUD_LEVELS:
            all_scenarios.append((care_name, hiud_name))

    raw = {}
    for scen in all_scenarios:
        key = scen if scen == 'counterfactual' else f'{scen[0]}_{scen[1]}'
        raw[key] = {
            'hmb_prev_monthly': [],
            'anemia_hmb_monthly': [],
            'anemia_total_monthly': [],
            'hmb_prev_annual': [],
            'anemia_hmb_annual': [],
            'anemia_total_annual': [],
        }

    for scen in all_scenarios:
        key = scen if scen == 'counterfactual' else f'{scen[0]}_{scen[1]}'
        if scen == 'counterfactual':
            label = 'Status quo (10% care, no hIUD)'
        else:
            label = f'{CARE_LEVELS[scen[0]].label} + {HIUD_LEVELS[scen[1]].label}'

        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")

        for seed in range(N_SEEDS):
            print(f"  seed {seed}...", end=" ", flush=True)
            sim = make_sim(scen, seed=seed)
            sim.run()

            hmb_prev = np.asarray(sim.results.menstruation['hmb_prev'])
            anemia_hmb = np.asarray(sim.results.track_hmb_anemia['n_anemia_with_hmb'])
            anemia_total = np.asarray(sim.results.track_hmb_anemia['n_anemia_total'])

            raw[key]['hmb_prev_monthly'].append(hmb_prev)
            raw[key]['anemia_hmb_monthly'].append(anemia_hmb)
            raw[key]['anemia_total_monthly'].append(anemia_total)
            raw[key]['hmb_prev_annual'].append(_annualize_mean(hmb_prev))
            raw[key]['anemia_hmb_annual'].append(_annualize(anemia_hmb))
            raw[key]['anemia_total_annual'].append(_annualize(anemia_total))

            del sim; gc.collect()
            print("done")

    sc.saveobj(results_file, raw)
    print(f"\nSaved: {results_file}")
    return raw


# ── Statistics ─────────────────────────────────────────────────────────────────
def compute_pct_reduction(raw, scen_key, outcome='anemia_hmb_annual'):
    cf = np.array(raw['counterfactual'][outcome])
    intv = np.array(raw[scen_key][outcome])
    averted = cf - intv
    pct = np.where(cf > 0, averted / cf * 100, np.nan)
    return {
        'mean':  np.nanmean(pct, axis=0),
        'lower': np.nanpercentile(pct, 2.5, axis=0),
        'upper': np.nanpercentile(pct, 97.5, axis=0),
    }


def compute_abs_reduction(raw, scen_key, outcome='anemia_hmb_annual'):
    cf = np.array(raw['counterfactual'][outcome])
    intv = np.array(raw[scen_key][outcome])
    diff = cf - intv
    return {
        'mean':  np.mean(diff, axis=0),
        'lower': np.percentile(diff, 2.5, axis=0),
        'upper': np.percentile(diff, 97.5, axis=0),
    }


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_panels_by_care(raw, years_monthly, outcome, ylabel, title_prefix, filename):
    """3 panels (base/mid/high care). Per panel: counterfactual + 3 hIUD lines."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(f'{title_prefix}\n'
                 f'Status quo before {INTV_YEAR}, intervention after. Mid RR ({RR_MID})',
                 fontsize=13)

    for idx, (care_name, care_info) in enumerate(CARE_LEVELS.items()):
        ax = axes[idx]

        cf_arr = np.array(raw['counterfactual'][outcome])
        ax.plot(years_monthly, cf_arr.mean(axis=0), color='#6c757d', ls='--', lw=2,
                label='Status quo')

        for hiud_name, hiud_info in HIUD_LEVELS.items():
            key = f'{care_name}_{hiud_name}'
            arr = np.array(raw[key][outcome])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            ax.plot(years_monthly, mean, color=CARE_COLORS[care_name][hiud_name],
                    lw=1.8, label=hiud_info.label)
            ax.fill_between(years_monthly, mean - std, mean + std,
                            color=CARE_COLORS[care_name][hiud_name], alpha=0.15)

        ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
        ax.set_xlabel('Year')
        if idx == 0:
            ax.set_ylabel(ylabel)
        ax.set_title(care_info.label)
        ax.set_xlim([START, STOP])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
        if 'anemia' in outcome:
            ax.set_ylim(bottom=0)
            sc.SIticks(ax=ax)

    plt.tight_layout()
    fig.savefig(PLOTFOLDER + filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTFOLDER}{filename}")
    return fig


def plot_pct_reduction_by_care(raw, years, outcome, ylabel, title_prefix, filename):
    """3 panels (base/mid/high care). Per panel: 3 hIUD lines showing % reduction."""
    post_mask = years >= INTV_YEAR

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(f'{title_prefix}\n'
                 f'% reduction vs status quo. Mid RR ({RR_MID})', fontsize=13)

    for idx, (care_name, care_info) in enumerate(CARE_LEVELS.items()):
        ax = axes[idx]

        for hiud_name, hiud_info in HIUD_LEVELS.items():
            key = f'{care_name}_{hiud_name}'
            s = compute_pct_reduction(raw, key, outcome=outcome)
            mean = np.where(post_mask, s['mean'], np.nan)
            lower = np.where(post_mask, s['lower'], np.nan)
            upper = np.where(post_mask, s['upper'], np.nan)

            ax.plot(years, mean, color=HIUD_LINE_COLORS[hiud_name], lw=2.5,
                    label=hiud_info.label)
            ax.fill_between(years, lower, upper,
                            color=HIUD_LINE_COLORS[hiud_name], alpha=0.15)

        ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
        ax.set_xlabel('Year')
        if idx == 0:
            ax.set_ylabel(ylabel)
        ax.set_title(care_info.label)
        ax.set_xlim([START, STOP]); ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    fig.savefig(PLOTFOLDER + filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTFOLDER}{filename}")
    return fig


def plot_pct_reduction_by_hiud(raw, years, outcome, ylabel, title_prefix, filename):
    """3 panels (low/mid/high hIUD). Per panel: 3 care lines."""
    post_mask = years >= INTV_YEAR

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle(f'{title_prefix}\n'
                 f'% reduction vs status quo. Mid RR ({RR_MID})', fontsize=13)

    for idx, (hiud_name, hiud_info) in enumerate(HIUD_LEVELS.items()):
        ax = axes[idx]

        for care_name, care_info in CARE_LEVELS.items():
            key = f'{care_name}_{hiud_name}'
            s = compute_pct_reduction(raw, key, outcome=outcome)
            mean = np.where(post_mask, s['mean'], np.nan)
            lower = np.where(post_mask, s['lower'], np.nan)
            upper = np.where(post_mask, s['upper'], np.nan)

            ax.plot(years, mean, color=CARE_LINE_COLORS[care_name], lw=2.5,
                    label=care_info.label)
            ax.fill_between(years, lower, upper,
                            color=CARE_LINE_COLORS[care_name], alpha=0.15)

        ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
        ax.set_xlabel('Year')
        if idx == 0:
            ax.set_ylabel(ylabel)
        ax.set_title(hiud_info.label)
        ax.set_xlim([START, STOP]); ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    fig.savefig(PLOTFOLDER + filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTFOLDER}{filename}")
    return fig


def plot_heatmap(raw, years, outcome, title, filename, fmt='.1f'):
    """3×3 heatmap: % reduction at end of sim."""
    post_mask = years >= INTV_YEAR
    care_names = list(CARE_LEVELS.keys())
    hiud_names = list(HIUD_LEVELS.keys())

    matrix = np.zeros((len(care_names), len(hiud_names)))
    for i, care_name in enumerate(care_names):
        for j, hiud_name in enumerate(hiud_names):
            key = f'{care_name}_{hiud_name}'
            s = compute_pct_reduction(raw, key, outcome=outcome)
            matrix[i, j] = s['mean'][-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')

    ax.set_xticks(range(len(hiud_names)))
    ax.set_xticklabels([HIUD_LEVELS[h].label for h in hiud_names], fontsize=10)
    ax.set_yticks(range(len(care_names)))
    ax.set_yticklabels([CARE_LEVELS[c].label for c in care_names], fontsize=10)

    for i in range(len(care_names)):
        for j in range(len(hiud_names)):
            ax.text(j, i, f'{matrix[i, j]:{fmt}}%',
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    color='white' if matrix[i, j] > matrix.max() * 0.5 else 'black')

    ax.set_xlabel('hIUD uptake level')
    ax.set_ylabel('Care-seeking level')
    ax.set_title(f'{title}\n% reduction vs status quo at {STOP} (mid RR = {RR_MID})')
    plt.colorbar(im, ax=ax, label='% reduction', shrink=0.8)
    plt.tight_layout()
    fig.savefig(PLOTFOLDER + filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTFOLDER}{filename}")
    return fig


def plot_combined_absolute(raw, years_monthly):
    """Side-by-side: HMB prevalence and HMB-related anemia, all 10 lines."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f'HMB prevalence and HMB-related anemia by scenario\n'
                 f'Mid RR ({RR_MID})', fontsize=14)

    panels = [
        ('hmb_prev_monthly', 'HMB prevalence', 'HMB prevalence'),
        ('anemia_hmb_monthly', 'HMB-related anemia', 'Monthly anemia cases (HMB women)'),
    ]

    for idx, (outcome, title, ylabel) in enumerate(panels):
        ax = axes[idx]

        cf = np.array(raw['counterfactual'][outcome])
        ax.plot(years_monthly, cf.mean(axis=0), color='#6c757d', ls='--', lw=2.5,
                label='Status quo')

        for care_name in CARE_LEVELS:
            for hiud_name in HIUD_LEVELS:
                key = f'{care_name}_{hiud_name}'
                arr = np.array(raw[key][outcome])
                label = f'{CARE_LEVELS[care_name].label} + {HIUD_LEVELS[hiud_name].label}'
                ax.plot(years_monthly, arr.mean(axis=0),
                        color=CARE_COLORS[care_name][hiud_name], lw=1.2,
                        label=label)

        ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
        ax.set_xlabel('Year'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_xlim([START, STOP])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if idx == 1:
            ax.set_ylim(bottom=0); sc.SIticks(ax=ax)
        ax.legend(frameon=False, fontsize=6, ncol=2)

    plt.tight_layout()
    fig.savefig(PLOTFOLDER + 'absolute_combined_all.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTFOLDER}absolute_combined_all.png")
    return fig


# ── Summary table ──────────────────────────────────────────────────────────────
def print_summary(raw, years):
    post_mask = years >= INTV_YEAR

    print(f"\n{'═'*100}")
    print(f"  Scenario comparison vs status quo (mid RR = {RR_MID})")
    print(f"  Post-intervention period: {INTV_YEAR}–{STOP}")
    print(f"{'═'*100}")

    # End-of-sim values
    print(f"\n  End-of-sim values:")
    print(f"  {'Scenario':<45}  {'HMB prev':>10}  {'HMB anemia (M)':>16}")
    print(f"  {'─'*75}")

    # Counterfactual
    hmb = np.array(raw['counterfactual']['hmb_prev_annual'])[:, -1]
    ane = np.array(raw['counterfactual']['anemia_hmb_annual'])[:, -1]
    print(f"  {'Status quo (10% care, no hIUD)':<45}  "
          f"{hmb.mean():.3f}±{hmb.std():.3f}  "
          f"{ane.mean()/1e6:>8.2f}±{ane.std()/1e6:.2f}M")

    for care_name, care_info in CARE_LEVELS.items():
        for hiud_name, hiud_info in HIUD_LEVELS.items():
            key = f'{care_name}_{hiud_name}'
            label = f'{care_info.label} + {hiud_info.label}'
            hmb = np.array(raw[key]['hmb_prev_annual'])[:, -1]
            ane = np.array(raw[key]['anemia_hmb_annual'])[:, -1]
            print(f"  {label:<45}  "
                  f"{hmb.mean():.3f}±{hmb.std():.3f}  "
                  f"{ane.mean()/1e6:>8.2f}±{ane.std()/1e6:.2f}M")

    # % reduction matrices
    for outcome, title in [('hmb_prev_annual', 'HMB prevalence'),
                           ('anemia_hmb_annual', 'HMB-related anemia')]:
        print(f"\n  % reduction in {title} vs status quo (mean over {INTV_YEAR}–{STOP}):")
        print(f"  {'':>20}", end="")
        for hiud_name, hiud_info in HIUD_LEVELS.items():
            print(f"  {hiud_info.label:>22}", end="")
        print()
        print(f"  {'─'*90}")

        for care_name, care_info in CARE_LEVELS.items():
            print(f"  {care_info.label:<20}", end="")
            for hiud_name in HIUD_LEVELS:
                key = f'{care_name}_{hiud_name}'
                s = compute_pct_reduction(raw, key, outcome=outcome)
                m = np.nanmean(s['mean'][post_mask])
                lo = np.nanmean(s['lower'][post_mask])
                hi = np.nanmean(s['upper'][post_mask])
                print(f"  {m:>5.1f}% ({lo:.1f}–{hi:.1f})", end="")
            print()

    # Absolute averted at end of sim
    print(f"\n  Averted HMB-anemia cases at end of sim (millions):")
    print(f"  {'':>20}", end="")
    for hiud_name, hiud_info in HIUD_LEVELS.items():
        print(f"  {hiud_info.label:>22}", end="")
    print()
    print(f"  {'─'*90}")

    for care_name, care_info in CARE_LEVELS.items():
        print(f"  {care_info.label:<20}", end="")
        for hiud_name in HIUD_LEVELS:
            key = f'{care_name}_{hiud_name}'
            s = compute_abs_reduction(raw, key, 'anemia_hmb_annual')
            m = s['mean'][-1] / 1e6
            print(f"  {m:>18.2f}M", end="")
        print()

    print(f"\n{'═'*100}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    do_run = True
    raw = run_scenarios(force_rerun=do_run)

    # Time axes
    n_years = len(raw['counterfactual']['hmb_prev_annual'][0])
    years_full = np.arange(START, START + n_years)

    n_months = len(raw['counterfactual']['hmb_prev_monthly'][0])
    years_monthly = np.array([START + m / 12 for m in range(n_months)])

    # ── Absolute plots by care level ──
    plot_panels_by_care(raw, years_monthly,
        outcome='hmb_prev_monthly', ylabel='HMB prevalence',
        title_prefix='HMB prevalence by scenario',
        filename='hmb_prev_by_care.png')

    plot_panels_by_care(raw, years_monthly,
        outcome='anemia_hmb_monthly', ylabel='Monthly anemia cases (HMB women)',
        title_prefix='HMB-related anemia by scenario',
        filename='anemia_hmb_by_care.png')

    # ── % reduction by care level ──
    plot_pct_reduction_by_care(raw, years_full,
        outcome='hmb_prev_annual', ylabel='% reduction in HMB prevalence',
        title_prefix='HMB prevalence reduction',
        filename='pct_reduction_hmb_by_care.png')

    plot_pct_reduction_by_care(raw, years_full,
        outcome='anemia_hmb_annual', ylabel='% reduction in HMB-related anemia',
        title_prefix='HMB-related anemia reduction',
        filename='pct_reduction_anemia_by_care.png')

    # ── % reduction by hIUD level ──
    plot_pct_reduction_by_hiud(raw, years_full,
        outcome='hmb_prev_annual', ylabel='% reduction in HMB prevalence',
        title_prefix='HMB prevalence reduction',
        filename='pct_reduction_hmb_by_hiud.png')

    plot_pct_reduction_by_hiud(raw, years_full,
        outcome='anemia_hmb_annual', ylabel='% reduction in HMB-related anemia',
        title_prefix='HMB-related anemia reduction',
        filename='pct_reduction_anemia_by_hiud.png')

    # ── Combined absolute ──
    plot_combined_absolute(raw, years_monthly)

    # ── Heatmaps ──
    plot_heatmap(raw, years_full,
        outcome='hmb_prev_annual', title='HMB prevalence',
        filename='heatmap_hmb_prev.png')

    plot_heatmap(raw, years_full,
        outcome='anemia_hmb_annual', title='HMB-related anemia',
        filename='heatmap_anemia_hmb.png')

    # ── Summary ──
    print_summary(raw, years_full)