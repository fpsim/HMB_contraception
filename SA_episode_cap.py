# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 09:38:33 2026

@author: navidehno
"""

"""
Sensitivity analysis: max care-seeking episodes.
Baseline: 3 episodes (current). Alternative: 6 episodes.

Scenario: Mid care (20%) + Mid hIUD (25%) vs status quo.
Mid RR (1.73) throughout.

Question: Does allowing more care-seeking attempts before give-up/hysterectomy
meaningfully increase treatment coverage and HMB/anemia reduction?
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
PLOTFOLDER = 'figures_episode_sa/'
OUTFOLDER  = 'results_episode_sa/'
for d in [PLOTFOLDER, OUTFOLDER]:
    os.makedirs(d, exist_ok=True)


# ── Settings ───────────────────────────────────────────────────────────────────
P_BASE      = 0.215
P_HMB_PRONE = 0.53
RR_MID      = 1.73
N_SEEDS     = 10
START       = 2017
STOP        = 2035
INTV_YEAR   = 2026
PLOT_START  = 2020

EPISODE_CAPS = [3, 6]

CARE_PRE = sc.objdict(
    p_ever_seek=0.10,
    p_monthly_first=1/36,
    p_monthly_first_anemic=1/12,
    p_monthly_first_pain=1/24,
    p_monthly_repeat=1/6,
)

CARE_POST = sc.objdict(
    p_ever_seek=0.20,
    p_monthly_first=1/12,
    p_monthly_first_anemic=1/6,
    p_monthly_first_pain=1/12,
    p_monthly_repeat=1/6,
)

HIUD_MID = sc.objdict(
    prob_offer=0.80,
    tx_weights=sc.objdict(nsaid=34.4, txa=17.2, pill=17.2, hiud=31.3),
)

# Colors
CAP_COLORS = {3: '#2196F3', 6: '#F44336'}
CAP_LABELS = {3: 'Max 3 episodes', 6: 'Max 6 episodes'}


# ── Helpers ────────────────────────────────────────────────────────────────────
def rr_to_logistic_coeff(rr, p_base=P_BASE):
    p_hmb = np.clip(p_base * rr, 1e-6, 1 - 1e-6)
    return (-np.log(1 / p_hmb - 1)) - (-np.log(1 / p_base - 1))


def _mask_pre_intv(years, values):
    """Set values before INTV_YEAR to NaN so only post-intervention is plotted."""
    masked = np.array(values, dtype=float).copy()
    masked[years < INTV_YEAR] = np.nan
    return masked


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


def make_intervention(max_episodes):
    return HMBPool(pars=dict(
        year=2017,
        intv_year=INTV_YEAR,
        care_behavior_pre=CARE_PRE,
        care_behavior_post=CARE_POST,
        prob_offer_pre=0.70,
        prob_offer_post=HIUD_MID.prob_offer,
        tx_weights_pre=sc.objdict(nsaid=50.0, txa=25.0, pill=25.0, hiud=0.0),
        tx_weights_post=HIUD_MID.tx_weights,
        nsaid=sc.objdict(efficacy=0.33, adherence=0.80),
        txa=sc.objdict(efficacy=0.45, adherence=0.70),
        pill=sc.objdict(efficacy=0.59, adherence=0.80),
        hiud=sc.objdict(efficacy=0.88, adherence=1.00),
        max_care_episodes=max_episodes,
    ))


def make_counterfactual(max_episodes):
    return HMBCounterfactual(pars=dict(
        year=2017,
        care_behavior_pre=CARE_PRE,
        care_behavior_post=CARE_PRE,
        max_care_episodes=max_episodes,
    ))


def make_sim(scenario, max_episodes, seed=0):
    mens = make_menstruation()
    edu = Education()

    if scenario == 'counterfactual':
        intervention = make_counterfactual(max_episodes)
    else:
        intervention = make_intervention(max_episodes)

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


def run_sa(force_rerun=True):
    results_file = OUTFOLDER + 'episode_cap_sa_raw.obj'

    if not force_rerun and os.path.exists(results_file):
        print("Loading saved results...")
        return sc.loadobj(results_file)

    raw = {}
    for cap in EPISODE_CAPS:
        for scen in ['counterfactual', 'intervention']:
            key = f'{scen}_cap{cap}'
            raw[key] = {
                'hmb_prev_monthly': [],
                'anemia_hmb_monthly': [],
                'anemia_total_monthly': [],
                'hmb_prev_annual': [],
                'anemia_hmb_annual': [],
                'anemia_total_annual': [],
            }

    for cap in EPISODE_CAPS:
        for scen in ['counterfactual', 'intervention']:
            key = f'{scen}_cap{cap}'
            label = f'{"Status quo" if scen == "counterfactual" else "Mid care + Mid hIUD"}, max {cap} episodes'

            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")

            for seed in range(N_SEEDS):
                print(f"  seed {seed}...", end=" ", flush=True)
                sim = make_sim(scen, cap, seed=seed)
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
def compute_pct_reduction(raw, cap, outcome='anemia_hmb_annual'):
    cf = np.array(raw[f'counterfactual_cap{cap}'][outcome])
    intv = np.array(raw[f'intervention_cap{cap}'][outcome])
    averted = cf - intv
    pct = np.where(cf > 0, averted / cf * 100, np.nan)
    return {
        'mean':  np.nanmean(pct, axis=0),
        'lower': np.nanpercentile(pct, 2.5, axis=0),
        'upper': np.nanpercentile(pct, 97.5, axis=0),
    }


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_hmb_prevalence(raw, years_monthly):
    """HMB prevalence: counterfactual + intervention for each cap.
    Only status quo shown before INTV_YEAR."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f'HMB prevalence by max care-seeking episodes\n'
                 f'Mid care (20%) + Mid hIUD (25%), mid RR ({RR_MID})', fontsize=13)

    # Left: absolute
    ax = axes[0]

    # Plot ONE status quo line (they overlap pre-2026; use cap=3 as reference)
    cf_ref = np.array(raw['counterfactual_cap3']['hmb_prev_monthly'])
    ax.plot(years_monthly, cf_ref.mean(axis=0), color='#6c757d', ls='--', lw=2,
            label='Status quo')

    # Intervention lines: post-intervention only
    for cap in EPISODE_CAPS:
        intv = np.array(raw[f'intervention_cap{cap}']['hmb_prev_monthly'])
        mean = intv.mean(axis=0)
        std = intv.std(axis=0)
        mean_masked = _mask_pre_intv(years_monthly, mean)
        std_masked = _mask_pre_intv(years_monthly, std)

        ax.plot(years_monthly, mean_masked, color=CAP_COLORS[cap], lw=2,
                label=f'Intervention ({CAP_LABELS[cap]})')
        ax.fill_between(years_monthly, mean_masked - std_masked,
                        mean_masked + std_masked,
                        color=CAP_COLORS[cap], alpha=0.12)

    ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
    ax.set_xlabel('Year'); ax.set_ylabel('HMB prevalence')
    ax.set_title('Absolute')
    ax.set_xlim([PLOT_START, STOP])
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=8)

    # Right: % reduction
    ax = axes[1]
    n_years = len(raw['counterfactual_cap3']['hmb_prev_annual'][0])
    years_full = np.arange(START, START + n_years)
    post_mask = years_full >= INTV_YEAR

    for cap in EPISODE_CAPS:
        s = compute_pct_reduction(raw, cap, outcome='hmb_prev_annual')
        mean = np.where(post_mask, s['mean'], np.nan)
        lower = np.where(post_mask, s['lower'], np.nan)
        upper = np.where(post_mask, s['upper'], np.nan)

        ax.plot(years_full, mean, color=CAP_COLORS[cap], lw=2.5, label=CAP_LABELS[cap])
        ax.fill_between(years_full, lower, upper, color=CAP_COLORS[cap], alpha=0.15)

    ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
    ax.set_xlabel('Year'); ax.set_ylabel('% reduction vs status quo')
    ax.set_title('% reduction in HMB prevalence')
    ax.set_xlim([PLOT_START, STOP]); ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    fig.savefig(PLOTFOLDER + 'hmb_prev_episode_sa.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTFOLDER}hmb_prev_episode_sa.png")
    return fig


def plot_anemia(raw, years_monthly):
    """Anemia among HMB women: counterfactual + intervention for each cap.
    Only status quo shown before INTV_YEAR."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(f'HMB-related anemia by max care-seeking episodes\n'
                 f'Mid care (20%) + Mid hIUD (25%), mid RR ({RR_MID})', fontsize=13)

    # Left: absolute
    ax = axes[0]

    # Plot ONE status quo line
    cf_ref = np.array(raw['counterfactual_cap3']['anemia_hmb_monthly'])
    ax.plot(years_monthly, cf_ref.mean(axis=0), color='#6c757d', ls='--', lw=2,
            label='Status quo')

    # Intervention lines: post-intervention only
    for cap in EPISODE_CAPS:
        intv = np.array(raw[f'intervention_cap{cap}']['anemia_hmb_monthly'])
        mean = intv.mean(axis=0)
        std = intv.std(axis=0)
        mean_masked = _mask_pre_intv(years_monthly, mean)
        std_masked = _mask_pre_intv(years_monthly, std)

        ax.plot(years_monthly, mean_masked, color=CAP_COLORS[cap], lw=2,
                label=f'Intervention ({CAP_LABELS[cap]})')
        ax.fill_between(years_monthly, mean_masked - std_masked,
                        mean_masked + std_masked,
                        color=CAP_COLORS[cap], alpha=0.12)

    ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
    ax.set_xlabel('Year'); ax.set_ylabel('Monthly anemia cases (HMB women)')
    ax.set_title('Absolute')
    ax.set_xlim([PLOT_START, STOP]); ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    sc.SIticks(ax=ax)

    # Right: % reduction
    ax = axes[1]
    n_years = len(raw['counterfactual_cap3']['anemia_hmb_annual'][0])
    years_full = np.arange(START, START + n_years)
    post_mask = years_full >= INTV_YEAR

    for cap in EPISODE_CAPS:
        s = compute_pct_reduction(raw, cap, outcome='anemia_hmb_annual')
        mean = np.where(post_mask, s['mean'], np.nan)
        lower = np.where(post_mask, s['lower'], np.nan)
        upper = np.where(post_mask, s['upper'], np.nan)

        ax.plot(years_full, mean, color=CAP_COLORS[cap], lw=2.5, label=CAP_LABELS[cap])
        ax.fill_between(years_full, lower, upper, color=CAP_COLORS[cap], alpha=0.15)

    ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
    ax.set_xlabel('Year'); ax.set_ylabel('% reduction vs status quo')
    ax.set_title('% reduction in HMB-related anemia')
    ax.set_xlim([PLOT_START, STOP]); ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    fig.savefig(PLOTFOLDER + 'anemia_episode_sa.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {PLOTFOLDER}anemia_episode_sa.png")
    return fig


# ── Summary ────────────────────────────────────────────────────────────────────
def print_summary(raw):
    n_years = len(raw['counterfactual_cap3']['hmb_prev_annual'][0])
    years = np.arange(START, START + n_years)
    post_mask = years >= INTV_YEAR

    print(f"\n{'═'*90}")
    print(f"  Episode Cap Sensitivity Analysis")
    print(f"  Mid care (20%) + Mid hIUD (25%), mid RR ({RR_MID})")
    print(f"{'═'*90}")

    print(f"\n  End-of-sim values:")
    print(f"  {'Scenario':<35}  {'HMB prev':>10}  {'HMB anemia (M)':>16}")
    print(f"  {'─'*65}")

    for cap in EPISODE_CAPS:
        for scen, label in [('counterfactual', 'Status quo'),
                            ('intervention', 'Mid care + Mid hIUD')]:
            key = f'{scen}_cap{cap}'
            hmb = np.array(raw[key]['hmb_prev_annual'])[:, -1]
            ane = np.array(raw[key]['anemia_hmb_annual'])[:, -1]
            print(f"  {f'{label} (max {cap})':<35}  "
                  f"{hmb.mean():.4f}±{hmb.std():.4f}  "
                  f"{ane.mean()/1e6:>8.2f}±{ane.std()/1e6:.2f}M")

    print(f"\n  % reduction vs status quo at {STOP}:")
    print(f"  {'Episode cap':<20}  {'HMB prev':>12}  {'HMB anemia':>12}")
    print(f"  {'─'*50}")

    for cap in EPISODE_CAPS:
        hmb_s = compute_pct_reduction(raw, cap, outcome='hmb_prev_annual')
        ane_s = compute_pct_reduction(raw, cap, outcome='anemia_hmb_annual')
        print(f"  {f'Max {cap} episodes':<20}  {hmb_s['mean'][-1]:>10.1f}%  {ane_s['mean'][-1]:>10.1f}%")

    # Difference
    hmb_3 = compute_pct_reduction(raw, 3, 'hmb_prev_annual')['mean'][-1]
    hmb_6 = compute_pct_reduction(raw, 6, 'hmb_prev_annual')['mean'][-1]
    ane_3 = compute_pct_reduction(raw, 3, 'anemia_hmb_annual')['mean'][-1]
    ane_6 = compute_pct_reduction(raw, 6, 'anemia_hmb_annual')['mean'][-1]

    print(f"\n  Additional gain from 6 vs 3 episodes:")
    print(f"    HMB prevalence: {hmb_6 - hmb_3:+.1f} percentage points")
    print(f"    HMB anemia:     {ane_6 - ane_3:+.1f} percentage points")

    print(f"\n{'═'*90}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    do_run = True
    raw = run_sa(force_rerun=do_run)

    # Time axes
    n_months = len(raw['counterfactual_cap3']['hmb_prev_monthly'][0])
    years_monthly = np.array([START + m / 12 for m in range(n_months)])

    # Plots
    plot_hmb_prevalence(raw, years_monthly)
    plot_anemia(raw, years_monthly)

    # Summary
    print_summary(raw)
    raw = sc.loadobj('results_episode_sa/episode_cap_sa_raw.obj')