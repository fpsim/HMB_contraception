"""
Sensitivity analysis: anemia reduction by hIUD uptake × RR of anemia given HMB.

Structure:
  3 panels (low / mid / high hIUD uptake)
  3 RR lines per panel + counterfactual (dashed) per RR

All intervention sims:
  2017–2025: status quo (10% care-seeking, NSAID/TXA/Pill, no hIUD)
  2026–2035: mid care-seeking (20%) + hIUD at specified uptake level

Counterfactual sims:
  2017–2035: status quo throughout (10% care-seeking, no hIUD)

RR varies from start of simulation (disease parameter).
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
PLOTFOLDER = 'figures_anemia_sa/'
OUTFOLDER  = 'results_anemia_sa/'
for d in [PLOTFOLDER, OUTFOLDER]:
    os.makedirs(d, exist_ok=True)


# ── Settings ───────────────────────────────────────────────────────────────────
P_BASE      = 0.215
P_HMB_PRONE = 0.53
N_SEEDS     = 10
START       = 2017
STOP        = 2035
INTV_YEAR   = 2026
PLOT_START  = 2020

# ── RR scenarios ──
rr_values = {
    'low_rr':  1.07,
    'mid_rr':  1.73,
    'high_rr': 2.50,
}
rr_labels = {
    'low_rr':  'Low RR (1.07)',
    'mid_rr':  'Mid RR (1.73)',
    'high_rr': 'High RR (2.50)',
}
rr_colors = {
    'low_rr':  '#2196F3',
    'mid_rr':  '#4CAF50',
    'high_rr': '#F44336',
}

# ── Care-seeking ──
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

# ── hIUD scenarios (from Darcy's table) ──
HIUD_SCENARIOS = {
    'low_hiud': sc.objdict(
        label='Low hIUD (10% of seekers)',
        tx_weights_post=sc.objdict(nsaid=31.5, txa=15.75, pill=15.75, hiud=10.0),
        prob_offer_post=0.73,
    ),
    'mid_hiud': sc.objdict(
        label='Mid hIUD (25% of seekers)',
        tx_weights_post=sc.objdict(nsaid=27.5, txa=13.75, pill=13.75, hiud=25.0),
        prob_offer_post=0.80,
    ),
    'high_hiud': sc.objdict(
        label='High hIUD (40% of seekers)',
        tx_weights_post=sc.objdict(nsaid=25.0, txa=12.5, pill=12.5, hiud=40.0),
        prob_offer_post=0.90,
    ),
}

# Colors for hIUD scenarios (used in % reduction plots)
HIUD_COLORS = {
    'low_hiud':  '#90CAF9',
    'mid_hiud':  '#4CAF50',
    'high_hiud': '#F44336',
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def rr_to_logistic_coeff(rr, p_base=P_BASE):
    p_hmb = np.clip(p_base * rr, 1e-6, 1 - 1e-6)
    return (-np.log(1 / p_hmb - 1)) - (-np.log(1 / p_base - 1))


def _mask_pre_intv(years, values):
    """Set values before INTV_YEAR to NaN so only post-intervention is plotted."""
    masked = np.array(values, dtype=float).copy()
    masked[years < INTV_YEAR] = np.nan
    return masked


def make_menstruation(rr):
    coeff = rr_to_logistic_coeff(rr)
    return Menstruation(pars={
        'p_hmb_prone': ss.bernoulli(p=P_HMB_PRONE),
        'hmb_seq': sc.objdict(
            poor_mh=sc.objdict(base=0.4, hmb=1.0),
            anemic=sc.objdict(base=P_BASE, hmb=coeff),
            pain=sc.objdict(base=0.1, hmb=3.36),
        )
    })


def make_counterfactual():
    return HMBPool(pars=dict(
        year=2017,
        intv_year=INTV_YEAR,

        care_behavior_pre=CARE_PRE,
        care_behavior_post=CARE_PRE,          # no change at 2026

        prob_offer_pre=0.70,
        prob_offer_post=0.70,                 # stays the same

        tx_weights_pre=sc.objdict(nsaid=50.0, txa=25.0, pill=25.0, hiud=0.0),
        tx_weights_post=sc.objdict(nsaid=50.0, txa=25.0, pill=25.0, hiud=0.0),

        nsaid=sc.objdict(efficacy=0.33, adherence=0.80),
        txa=sc.objdict(efficacy=0.45, adherence=0.70),
        pill=sc.objdict(efficacy=0.59, adherence=0.80),
        hiud=sc.objdict(efficacy=0.88, adherence=1.00),
    ))


def make_pool_intervention(hiud_scenario):
    """
    2017–2025: status quo (10% care, no hIUD, 70% receipt)
    2026+: mid care-seeking (20%) + hIUD at specified level
    """
    scen = HIUD_SCENARIOS[hiud_scenario]

    return HMBPool(pars=dict(
        year=2017,
        intv_year=INTV_YEAR,
        care_behavior_pre=CARE_PRE,
        care_behavior_post=CARE_POST,
        prob_offer_post=scen.prob_offer_post,
        tx_weights_pre=sc.objdict(nsaid=0.50, txa=0.25, pill=0.25, hiud=0.0),
        tx_weights_post=scen.tx_weights_post,
        nsaid=sc.objdict(efficacy=0.33, adherence=0.80),
        txa=sc.objdict(efficacy=0.45, adherence=0.70),
        pill=sc.objdict(efficacy=0.59, adherence=0.80),
        hiud=sc.objdict(efficacy=0.88, adherence=1.00),
    ))


def make_sim(rr, scenario, seed=0):
    """
    Build simulation.

    scenario: 'counterfactual' or one of HIUD_SCENARIOS keys
    """
    mens = make_menstruation(rr)
    edu = Education()

    if scenario == 'counterfactual':
        intervention = make_counterfactual()
    else:
        intervention = make_pool_intervention(scenario)

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
    n_years = len(arr) // 12
    arr = arr[:12 * n_years].reshape(n_years, 12)
    return arr.sum(axis=1) if how == 'sum' else arr[:, -1]


def run_sensitivity(force_rerun=True):
    results_file = OUTFOLDER + 'anemia_sa_hiud_rr_cf_raw.obj'

    if not force_rerun and os.path.exists(results_file):
        print("Loading saved results...")
        return sc.loadobj(results_file)

    # Structure: raw[rr_name][scenario] where scenario is 'counterfactual' or hiud name
    all_scenarios = ['counterfactual'] + list(HIUD_SCENARIOS.keys())
    raw = {}
    for rr_name in rr_values:
        raw[rr_name] = {}
        for scen in all_scenarios:
            raw[rr_name][scen] = {
                'monthly': [], 'hmb_monthly': [], 'annual': [],
            }

    for rr_name, rr in rr_values.items():
        print(f"\n{'='*60}")
        print(f"  {rr_labels[rr_name]}  (RR={rr:.2f})")
        print(f"{'='*60}")

        for seed in range(N_SEEDS):
            print(f"  seed {seed}:", end="", flush=True)

            for scen in all_scenarios:
                print(f" {scen}...", end="", flush=True)
                sim = make_sim(rr, scen, seed=seed)
                sim.run()

                monthly = np.asarray(sim.results.track_hmb_anemia['n_anemia_total'])
                hmb_monthly = np.asarray(sim.results.track_hmb_anemia['n_anemia_with_hmb'])

                raw[rr_name][scen]['monthly'].append(monthly)
                raw[rr_name][scen]['hmb_monthly'].append(hmb_monthly)
                raw[rr_name][scen]['annual'].append(_annualize(monthly))

                del sim; gc.collect()
            print(" done")

    sc.saveobj(results_file, raw)
    print(f"\nSaved: {results_file}")
    return raw


# ── Statistics ─────────────────────────────────────────────────────────────────
def compute_pct_reduction(raw, rr_name, hiud_name):
    """
    % reduction in annual anemia: intervention vs counterfactual at the same RR.
    """
    cf = np.array(raw[rr_name]['counterfactual']['annual'])
    intv = np.array(raw[rr_name][hiud_name]['annual'])
    averted = cf - intv
    pct = np.where(cf > 0, averted / cf * 100, np.nan)
    return {
        'mean':  np.nanmean(pct, axis=0),
        'lower': np.nanpercentile(pct, 2.5, axis=0),
        'upper': np.nanpercentile(pct, 97.5, axis=0),
    }


def compute_pct_reduction_hmb(raw, rr_name, hiud_name):
    """% reduction in HMB-specific anemia (where the signal lives)."""
    cf = np.array([_annualize(m) for m in raw[rr_name]['counterfactual']['hmb_monthly']])
    intv = np.array([_annualize(m) for m in raw[rr_name][hiud_name]['hmb_monthly']])
    averted = cf - intv
    pct = np.where(cf > 0, averted / cf * 100, np.nan)
    return {
        'mean':  np.nanmean(pct, axis=0),
        'lower': np.nanpercentile(pct, 2.5, axis=0),
        'upper': np.nanpercentile(pct, 97.5, axis=0),
    }


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_monthly_panels(raw, years_monthly):
    """
    3 panels (one per hIUD level). Per panel: 3 RR counterfactual lines (dashed, full)
    + 3 RR intervention lines (solid, post-2026 only).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle('Monthly anemia cases by hIUD uptake\n'
                 'Solid = intervention (post-2026), dashed = status quo',
                 fontsize=13)

    for idx, (hiud_name, scen) in enumerate(HIUD_SCENARIOS.items()):
        ax = axes[idx]

        for rr_name in rr_values:
            color = rr_colors[rr_name]

            # Counterfactual (dashed, full timeline)
            cf = np.array(raw[rr_name]['counterfactual']['monthly'])
            cf_mean = cf.mean(axis=0)
            ax.plot(years_monthly, cf_mean, color=color, ls='--', lw=1.2, alpha=0.7,
                    label=f'{rr_labels[rr_name]} (status quo)')

            # Intervention (solid, post-2026 only)
            intv = np.array(raw[rr_name][hiud_name]['monthly'])
            intv_mean = intv.mean(axis=0)
            intv_std = intv.std(axis=0)
            mean_masked = _mask_pre_intv(years_monthly, intv_mean)
            std_masked = _mask_pre_intv(years_monthly, intv_std)
            ax.plot(years_monthly, mean_masked, color=color, lw=1.5,
                    label=f'{rr_labels[rr_name]} (intv)')
            ax.fill_between(years_monthly, mean_masked - std_masked,
                            mean_masked + std_masked, color=color, alpha=0.12)

        ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
        ax.set_xlabel('Year')
        if idx == 0:
            ax.set_ylabel('Monthly anemia cases')
        ax.set_title(scen.label)
        ax.set_xlim([PLOT_START, STOP]); ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=7)
        sc.SIticks(ax=ax)

    plt.tight_layout()
    outpath = PLOTFOLDER + 'anemia_monthly_panels.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


def plot_annual_panels(raw, years):
    """
    3 panels (one per hIUD level). Per panel: 3 RR counterfactual lines (dashed, full)
    + 3 RR intervention lines (solid, post-2026 only).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle('Annual anemia cases by hIUD uptake\n'
                 'Solid = intervention (post-2026), dashed = status quo',
                 fontsize=13)

    post_mask = years >= INTV_YEAR

    for idx, (hiud_name, scen) in enumerate(HIUD_SCENARIOS.items()):
        ax = axes[idx]

        for rr_name in rr_values:
            color = rr_colors[rr_name]

            # Counterfactual (dashed, full timeline)
            cf = np.array(raw[rr_name]['counterfactual']['annual'])
            cf_mean = cf.mean(axis=0)
            ax.plot(years, cf_mean, color=color, ls='--', lw=1.5, alpha=0.7,
                    label=f'{rr_labels[rr_name]} (status quo)')

            # Intervention (solid, post-2026 only)
            intv = np.array(raw[rr_name][hiud_name]['annual'])
            intv_mean = intv.mean(axis=0)
            intv_std = intv.std(axis=0)
            mean_masked = np.where(post_mask, intv_mean, np.nan)
            std_masked = np.where(post_mask, intv_std, np.nan)
            ax.errorbar(years[post_mask], intv_mean[post_mask],
                        yerr=intv_std[post_mask], color=color,
                        lw=2, marker='o', capsize=3, markersize=4,
                        label=f'{rr_labels[rr_name]} (intv)')

        ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
        ax.set_xlabel('Year')
        if idx == 0:
            ax.set_ylabel('Annual anemia cases')
        ax.set_title(scen.label)
        ax.set_xlim([PLOT_START - 0.5, STOP + 0.5]); ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=7)
        sc.SIticks(ax=ax)

    plt.tight_layout()
    outpath = PLOTFOLDER + 'anemia_annual_panels.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


def plot_hmb_anemia_panels(raw, years_monthly):
    """3 panels: monthly anemia among HMB women, with counterfactual."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle('Anemia among HMB women by hIUD uptake\n'
                 'Solid = intervention (post-2026), dashed = status quo',
                 fontsize=13)

    for idx, (hiud_name, scen) in enumerate(HIUD_SCENARIOS.items()):
        ax = axes[idx]

        for rr_name in rr_values:
            color = rr_colors[rr_name]

            # Counterfactual (dashed, full timeline)
            cf = np.array(raw[rr_name]['counterfactual']['hmb_monthly'])
            cf_mean = cf.mean(axis=0)
            ax.plot(years_monthly, cf_mean, color=color, ls='--', lw=1.2, alpha=0.7,
                    label=f'{rr_labels[rr_name]} (status quo)')

            # Intervention (solid, post-2026 only)
            intv = np.array(raw[rr_name][hiud_name]['hmb_monthly'])
            intv_mean = intv.mean(axis=0)
            intv_std = intv.std(axis=0)
            mean_masked = _mask_pre_intv(years_monthly, intv_mean)
            std_masked = _mask_pre_intv(years_monthly, intv_std)
            ax.plot(years_monthly, mean_masked, color=color, lw=1.5,
                    label=f'{rr_labels[rr_name]} (intv)')
            ax.fill_between(years_monthly, mean_masked - std_masked,
                            mean_masked + std_masked, color=color, alpha=0.12)

        ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
        ax.set_xlabel('Year')
        if idx == 0:
            ax.set_ylabel('Monthly anemia cases (HMB women)')
        ax.set_title(scen.label)
        ax.set_xlim([PLOT_START, STOP]); ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=7)
        sc.SIticks(ax=ax)

    plt.tight_layout()
    outpath = PLOTFOLDER + 'anemia_hmb_monthly_panels.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


def plot_pct_reduction_by_hiud(raw, years):
    """
    3 panels (one per hIUD level): % reduction vs counterfactual, by RR.
    """
    post_mask = years >= INTV_YEAR

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle('% reduction in annual anemia vs counterfactual among HMB women\n'
                 'By hIUD uptake level and RR of anemia given HMB',
                 fontsize=13)

    for idx, (hiud_name, scen) in enumerate(HIUD_SCENARIOS.items()):
        ax = axes[idx]

        for rr_name in rr_values:
            s = compute_pct_reduction_hmb(raw, rr_name, hiud_name)

            mean = np.where(post_mask, s['mean'], np.nan)
            lower = np.where(post_mask, s['lower'], np.nan)
            upper = np.where(post_mask, s['upper'], np.nan)

            ax.plot(years, mean, color=rr_colors[rr_name], lw=2.5,
                    label=rr_labels[rr_name])
            ax.fill_between(years, lower, upper,
                            color=rr_colors[rr_name], alpha=0.15)

        ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
        ax.set_xlabel('Year')
        if idx == 0:
            ax.set_ylabel('% reduction vs counterfactual')
        ax.set_title(scen.label)
        ax.set_xlim([PLOT_START, STOP]); ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    outpath = PLOTFOLDER + 'anemia_pct_reduction_panels.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


def plot_pct_reduction_by_rr(raw, years):
    """
    3 panels (one per RR): % reduction vs counterfactual, by hIUD level.
    """
    post_mask = years >= INTV_YEAR

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle('% reduction in annual anemia vs counterfactual among HMB women\n'
                 'By RR assumption and hIUD uptake level',
                 fontsize=13)

    for idx, rr_name in enumerate(rr_values):
        ax = axes[idx]

        for hiud_name, scen in HIUD_SCENARIOS.items():
            s = compute_pct_reduction_hmb(raw, rr_name, hiud_name)

            mean = np.where(post_mask, s['mean'], np.nan)
            lower = np.where(post_mask, s['lower'], np.nan)
            upper = np.where(post_mask, s['upper'], np.nan)

            ax.plot(years, mean, color=HIUD_COLORS[hiud_name], lw=2.5,
                    label=scen.label)
            ax.fill_between(years, lower, upper,
                            color=HIUD_COLORS[hiud_name], alpha=0.15)

        ax.axvline(INTV_YEAR, color='k', ls='--', lw=1.5)
        ax.set_xlabel('Year')
        if idx == 0:
            ax.set_ylabel('% reduction vs counterfactual')
        ax.set_title(rr_labels[rr_name])
        ax.set_xlim([PLOT_START, STOP]); ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    outpath = PLOTFOLDER + 'anemia_pct_reduction_by_rr.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")
    return fig


# ── Summary table ──────────────────────────────────────────────────────────────
def print_summary(raw, years):
    post_mask = years >= INTV_YEAR

    print(f"\n{'═'*90}")
    print(f"  Anemia SA: % reduction vs counterfactual (no hIUD, 10% care)")
    print(f"  Intervention: mid care-seeking (20%) + hIUD, post-2026")
    print(f"{'═'*90}")

    # End-of-sim absolute cases
    print(f"\n  End-of-sim annual anemia cases (millions):")
    header = f"  {'':>18}  {'Counterfactual':>14}"
    for hiud_name, scen in HIUD_SCENARIOS.items():
        header += f"  {scen.label:>28}"
    print(header)
    print(f"  {'─'*120}")

    for rr_name in rr_values:
        cf_arr = np.array(raw[rr_name]['counterfactual']['annual'])
        cf_last = cf_arr[:, -1].mean() / 1e6
        line = f"  {rr_labels[rr_name]:<18}  {cf_last:>10.2f}M"

        for hiud_name in HIUD_SCENARIOS:
            arr = np.array(raw[rr_name][hiud_name]['annual'])
            m = arr[:, -1].mean() / 1e6
            line += f"  {m:>24.2f}M"
        print(line)

    # % reduction vs counterfactual
    print(f"\n  % reduction in anemia vs counterfactual")
    print(f"  (averaged over post-intervention years {INTV_YEAR}–{STOP})")

    for hiud_name, scen in HIUD_SCENARIOS.items():
        print(f"\n  {scen.label}")
        print(f"  {'RR scenario':<18}  {'Mean %':>8}  {'95% CI'}")
        print(f"  {'─'*50}")

        for rr_name in rr_values:
            s = compute_pct_reduction_hmb(raw, rr_name, hiud_name)
            m = np.nanmean(s['mean'][post_mask])
            lo = np.nanmean(s['lower'][post_mask])
            hi = np.nanmean(s['upper'][post_mask])
            print(f"  {rr_labels[rr_name]:<18}  {m:>7.1f}%  ({lo:.1f}%–{hi:.1f}%)")

    print(f"\n{'═'*90}\n")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':

    do_run = True
    raw = run_sensitivity(force_rerun=do_run)

    # Build time axes
    n_years = len(raw['mid_rr']['counterfactual']['annual'][0])
    years_full = np.arange(START, START + n_years)

    n_months = len(raw['mid_rr']['counterfactual']['monthly'][0])
    years_monthly = np.array([START + m / 12 for m in range(n_months)])

    # Plots
    plot_monthly_panels(raw, years_monthly)
    plot_annual_panels(raw, years_full)
    plot_hmb_anemia_panels(raw, years_monthly)
    plot_pct_reduction_by_hiud(raw, years_full)
    plot_pct_reduction_by_rr(raw, years_full)

    # Summary
    print_summary(raw, years_full)