# -*- coding: utf-8 -*-

"""
Sensitivity analysis: % reduction in anemia cases under low / mid / high RR
of anemia given HMB.

Pooled OR: 2.17 (1.09–4.31)  →  RR: 1.73 (1.07–2.50)

Interventions compared:
  - baseline  : no intervention
  - cascade   : full HMBCascade (NSAID → TXA → Pill → hIUD)

Architecture: new modular HMBCascade (v0.4.0)
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
from analyzers import track_hmb_anemia


# ── Output folders ─────────────────────────────────────────────────────────────
PLOTFOLDER = 'figures_anemia_sa/'
OUTFOLDER  = 'results_anemia_sa/'
for d in [PLOTFOLDER, OUTFOLDER]:
    os.makedirs(d, exist_ok=True)


# ── Settings ───────────────────────────────────────────────────────────────────
P_BASE   = 0.18   # baseline P(anemia | no HMB) – held fixed across scenarios
N_SEEDS  = 10     # stochastic runs per scenario
START    = 2020
STOP     = 2030
INTV_YEAR = 2025  # year intervention begins

# RR values derived from pooled OR 2.17 (95% CI: 1.09–4.31)
rr_values = {
    'low_rr':  1.07,
    'mid_rr':  1.73,
    'high_rr': 2.50,
}
rr_labels = {
    'low_rr':  'Low RR (1.07)',
    'mid_rr':  'Mid RR (1.79)',
    'high_rr': 'High RR (2.70)',
}
rr_colors = {
    'low_rr':  '#2196F3',   # blue
    'mid_rr':  '#4CAF50',   # green
    'high_rr': '#F44336',   # red
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def rr_to_logistic_coeff(rr, p_base=P_BASE):
    """
    Convert a risk ratio to a logistic regression coefficient.

    We want P(anemia | HMB=True) = p_base * rr.
    The logistic coefficient is the shift in log-odds needed to move
    from the baseline probability to the HMB probability.

    Args:
        rr: Risk ratio of anemia given HMB
        p_base: Baseline probability of anemia (no HMB)

    Returns:
        Logistic coefficient for HMB effect on anemia
    """
    p_hmb = p_base * rr
    # Clip to avoid log(0) or log of values >= 1
    p_hmb = np.clip(p_hmb, 1e-6, 1 - 1e-6)
    coeff = (-np.log(1 / p_hmb - 1)) - (-np.log(1 / p_base - 1))
    return coeff


def make_menstruation(rr):
    """
    Build a Menstruation connector with anemia risk set by the given RR.

    Only the hmb coefficient in hmb_seq.anemic changes across scenarios.
    All other Menstruation parameters use defaults.

    Args:
        rr: Risk ratio of anemia given HMB

    Returns:
        Menstruation connector instance
    """
    coeff = rr_to_logistic_coeff(rr)
    mens_pars = {
        'hmb_seq': sc.objdict(
            poor_mh=sc.objdict(base=0.4,  hmb=1.0),
            anemic =sc.objdict(base=P_BASE, hmb=coeff),   # <-- varies by RR
            pain   =sc.objdict(base=0.1,  hmb=1.5),
        )
    }
    return Menstruation(pars=mens_pars)


def make_cascade():
    """
    Build a fresh HMBCascade instance.

    Called fresh each time to avoid shared state across simulations.

    Returns:
        HMBCascade intervention instance
    """
    return HMBCascade(
        pars=dict(
            year=INTV_YEAR,
            time_to_assess=ss.months(3),
        )
    )


def make_sim(rr, with_intervention=False, seed=0):
    """
    Build a simulation with the specified anemia RR.

    Args:
        rr: Risk ratio of anemia given HMB
        with_intervention: Whether to include HMBCascade intervention
        seed: Random seed

    Returns:
        fp.Sim instance ready to run
    """
    mens = make_menstruation(rr)
    edu  = Education()
    hmb_anemia_analyzer = track_hmb_anemia()

    sim_kwargs = dict(
        start=START,
        stop=STOP,
        n_agents=5000,
        total_pop=55_000_000,
        location='kenya',
        education_module=edu,
        connectors=[mens],
        analyzers=[hmb_anemia_analyzer],
        rand_seed=seed,
        verbose=0,
    )

    if with_intervention:
        sim_kwargs['interventions'] = [make_cascade()]

    return fp.Sim(**sim_kwargs)


# ── Run simulations ────────────────────────────────────────────────────────────
def run_sensitivity(force_rerun=True):
    """
    Run baseline and intervention simulations for each RR value.

    For each RR x seed combination, runs a matched pair:
      - baseline sim (no intervention)
      - cascade sim  (with HMBCascade)

    Averted cases are computed within each seed before aggregating,
    which removes stochastic noise from the comparison.

    Returns:
        raw: dict with structure
             raw[rr_name]['baseline']    = list of annual anemia arrays (one per seed)
             raw[rr_name]['cascade']     = list of annual anemia arrays (one per seed)
             raw[rr_name]['averted']     = list of (baseline - cascade) arrays
    """
    results_file = OUTFOLDER + 'anemia_sa_rr_raw.obj'

    if not force_rerun and os.path.exists(results_file):
        print("Loading saved results...")
        return sc.loadobj(results_file)

    raw = {
        rr_name: {'baseline': [], 'cascade': [], 'averted': []}
        for rr_name in rr_values
    }

    for rr_name, rr in rr_values.items():
        p_hmb = P_BASE * rr
        print(f"\n=== {rr_labels[rr_name]}  (RR={rr:.2f}, "
              f"P(anemia|HMB)={p_hmb:.3f}) ===")

        for seed in range(N_SEEDS):
            print(f"  seed {seed}...", end=" ", flush=True)

            # Run matched pair with same seed and same RR
            sims = [
                make_sim(rr, with_intervention=False, seed=seed),  # baseline
                make_sim(rr, with_intervention=True,  seed=seed),  # cascade
            ]
            msim = ss.MultiSim(sims)
            msim.run()

            s_base    = msim.sims[0]
            s_cascade = msim.sims[1]

            # Extract annual total anemia from track_hmb_anemia analyzer
            # n_anemia_total is monthly; sum within each year
            base_monthly    = s_base.results.track_hmb_anemia['n_anemia_total']
            cascade_monthly = s_cascade.results.track_hmb_anemia['n_anemia_total']

            base_annual    = _annualize(base_monthly)
            cascade_annual = _annualize(cascade_monthly)
            averted_annual = base_annual - cascade_annual

            raw[rr_name]['baseline'].append(base_annual)
            raw[rr_name]['cascade'].append(cascade_annual)
            raw[rr_name]['averted'].append(averted_annual)

            del msim
            gc.collect()
            print("done")

    sc.saveobj(results_file, raw)
    print(f"\nSaved raw results: {results_file}")
    return raw


def _annualize(monthly_arr, how='sum'):
    """
    Convert monthly array to annual by summing (or taking end-of-year value).

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


# ── Aggregate statistics ───────────────────────────────────────────────────────
def compute_stats(raw):
    """
    Compute % reduction in anemia cases, aggregated across seeds.

    % reduction = (baseline - cascade) / baseline * 100, computed per seed
    then summarised as mean / 2.5th / 97.5th percentile.

    Args:
        raw: Output from run_sensitivity()

    Returns:
        stats: dict with structure
               stats[rr_name] = {'mean', 'lower', 'upper'} arrays over years
    """
    stats = {}
    for rr_name in rr_values:
        base_arr    = np.array(raw[rr_name]['baseline'])   # (n_seeds, n_years)
        averted_arr = np.array(raw[rr_name]['averted'])    # (n_seeds, n_years)

        # Compute % reduction per seed before aggregating
        pct = np.where(base_arr > 0, averted_arr / base_arr * 100, np.nan)

        stats[rr_name] = {
            'mean':  np.nanmean(pct,              axis=0),
            'lower': np.nanpercentile(pct,  2.5,  axis=0),
            'upper': np.nanpercentile(pct, 97.5,  axis=0),
        }
    return stats


# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_annual_cases(raw, years, intervention_year=INTV_YEAR):
    """
    Single panel: annual anemia cases for baseline and cascade intervention,
    with RR uncertainty shown as shaded band.

    Baseline: mean across seeds using mid RR (most likely estimate).
    Intervention: solid line = mean of low/high RR means;
                  shaded band spans low-RR mean to high-RR mean.
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    # Baseline: use mid RR as the representative estimate
    base_mid = np.array(raw['mid_rr']['baseline'])   # (n_seeds, n_years)
    base_mean = base_mid.mean(axis=0)
    base_lower = np.percentile(np.array(raw['low_rr']['baseline']),  2.5,  axis=0)
    base_upper = np.percentile(np.array(raw['high_rr']['baseline']), 97.5, axis=0)

    ax.plot(years, base_mean,  color='#6c757d', lw=2.5, label='Baseline (mid RR)')
    ax.fill_between(years, base_lower, base_upper, color='#6c757d', alpha=0.15,
                    label='Baseline RR uncertainty')

    # Cascade intervention: band spans low-RR mean to high-RR mean
    casc_low  = np.array(raw['low_rr']['cascade']).mean(axis=0)
    casc_mid  = np.array(raw['mid_rr']['cascade']).mean(axis=0)
    casc_high = np.array(raw['high_rr']['cascade']).mean(axis=0)

    ax.plot(years, casc_mid, color='#2ca02c', lw=2.5,
            label='HMB Cascade (mid RR)')
    ax.fill_between(years, casc_low, casc_high, color='#2ca02c', alpha=0.20,
                    label='Cascade RR uncertainty')

    ax.axvline(intervention_year, color='k', ls='--', lw=1.5)
    ylim = ax.get_ylim()
    ax.text(intervention_year - 0.2, ylim[1] * 0.95,
            'Start of\nintervention', ha='right', va='top',
            fontsize=9, color='#4d4d4d')

    ax.set_xlabel('Year')
    ax.set_ylabel('Annual anemia cases')
    ax.set_title('Annual anemia cases: baseline vs HMB cascade\n'
                 '(shaded band = RR uncertainty 1.07–2.50)')
    ax.set_xlim([START, STOP])
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    outpath = PLOTFOLDER + 'anemia_annual_cases.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.show()
    return fig, ax


def plot_pct_reduction(stats, years, intervention_year=INTV_YEAR):
    """
    Single panel: % reduction in anemia cases post-intervention,
    with separate lines for low / mid / high RR.

    Shaded bands show stochastic uncertainty (2.5th–97.5th percentile
    across seeds) for each RR value.
    """
    post_mask = years >= intervention_year

    fig, ax = plt.subplots(figsize=(7, 4))

    for rr_name in rr_values:
        s = stats[rr_name]

        # Mask pre-intervention period (set to NaN so lines start at intervention year)
        mean  = np.where(post_mask, s['mean'],  np.nan)
        lower = np.where(post_mask, s['lower'], np.nan)
        upper = np.where(post_mask, s['upper'], np.nan)

        ax.plot(years, mean, color=rr_colors[rr_name], lw=2.5,
                label=rr_labels[rr_name])
        ax.fill_between(years, lower, upper,
                        color=rr_colors[rr_name], alpha=0.15)

    ax.axvline(intervention_year, color='k', ls='--', lw=1.5)
    ylim = ax.get_ylim()
    ax.text(intervention_year - 0.2, ylim[1] * 0.95,
            'Start of\nintervention', ha='right', va='top',
            fontsize=9, color='#4d4d4d')

    ax.set_xlabel('Year')
    ax.set_ylabel('% reduction in annual anemia cases')
    ax.set_title('Sensitivity: % reduction in anemia cases\nby RR of anemia given HMB')
    ax.set_xlim([START, STOP])
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, fontsize=10)

    plt.tight_layout()
    outpath = PLOTFOLDER + 'anemia_pct_reduction_rr.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    plt.show()
    return fig, ax


# ── Summary table ──────────────────────────────────────────────────────────────
def print_summary(stats, years):
    """Print mean % reduction averaged over post-intervention years."""
    post_mask = years >= INTV_YEAR

    print(f"\n{'─'*60}")
    print("Mean % reduction in anemia cases (post-intervention, "
          f"{INTV_YEAR}–{STOP})")
    print(f"{'─'*60}")
    print(f"  {'RR scenario':<18}  {'Mean %':>8}  {'95% CI'}")
    print(f"{'─'*60}")

    for rr_name in rr_values:
        s = stats[rr_name]
        m  = np.nanmean(s['mean'][post_mask])
        lo = np.nanmean(s['lower'][post_mask])
        hi = np.nanmean(s['upper'][post_mask])
        print(f"  {rr_labels[rr_name]:<18}  {m:>7.1f}%  "
              f"({lo:.1f}%–{hi:.1f}%)")

    print(f"{'─'*60}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    do_run = True   # Set False to load saved results

    # Run (or load) simulations
    raw = run_sensitivity(force_rerun=do_run)

    # Build year axis from first result array
    n_years = len(raw['mid_rr']['baseline'][0])
    years_full = np.arange(START, START + n_years)

    # Compute % reduction statistics
    stats = compute_stats(raw)

    # Plots
    plot_annual_cases(raw, years_full)
    plot_pct_reduction(stats, years_full)

    # Summary table
    print_summary(stats, years_full)