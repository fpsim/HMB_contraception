"""
Calibration sweep: find hIUD acceptance probabilities that produce
~5%, 10%, 15% of HMB women ever using hIUD.

NSAID, TXA, and Pill acceptance all fixed at 50% or 25%.
Runs for 10%, 20%, and 35% base care-seeking.
"""

import numpy as np
import sciris as sc
import starsim as ss
import fpsim as fp
import os
import matplotlib.pyplot as plt

from menstruation import Menstruation
from education import Education
from interventions import HMBCascade
from analyzers import track_hmb_anemia

# ── Settings ───────────────────────────────────────────────────────────────────
START     = 2020
STOP      = 2030
INTV_YEAR = 2026
N_SEEDS   = 5

# Three care-seeking scenarios
CARE_SCENARIOS = {
    '10%': sc.objdict(base=0.10, anemic=1.43, pain=0.61),
    '20%': sc.objdict(base=0.20, anemic=0.86, pain=0.37),
    '35%': sc.objdict(base=0.35, anemic=0.32, pain=0.14),
}

CARE_COLORS = {
    '10%': '#d62728',
    '20%': '#ff7f0e',
    '35%': '#2196F3',
}

# Fixed acceptance for NSAID, TXA, Pill
FIXED_ACCEPT = 0.25

# Sweep hIUD acceptance from 0.1 to 1.0
HIUD_ACCEPT_VALUES = np.arange(0.1, 1.05, 0.1)

OUTFOLDER = 'results_calibration/'
os.makedirs(OUTFOLDER, exist_ok=True)


def make_sim(care_behavior, hiud_accept, seed=0):
    """Build sim with all treatment acceptance at 25% except hIUD which is varied."""
    mens = Menstruation()
    edu  = Education()
    cascade = HMBCascade(
        pars=dict(
            year=INTV_YEAR,
            time_to_assess=ss.months(3),
            care_behavior=care_behavior,
            nsaid=sc.objdict(
                efficacy=0.5,
                adherence=0.7,
                prob_offer=ss.bernoulli(p=0.9),
                prob_accept=ss.bernoulli(p=FIXED_ACCEPT),  # was 0.7
            ),
            txa=sc.objdict(
                efficacy=0.6,
                adherence=0.6,
                prob_offer=ss.bernoulli(p=0.9),
                prob_accept=ss.bernoulli(p=FIXED_ACCEPT),  # was 0.6
            ),
            pill=sc.objdict(
                efficacy=0.7,
                adherence=0.75,
                prob_offer=ss.bernoulli(p=0.9),
                prob_accept=ss.bernoulli(p=FIXED_ACCEPT),  # was 0.5 (unchanged)
            ),
            hiud=sc.objdict(
                efficacy=0.8,
                adherence=0.85,
                prob_offer=ss.bernoulli(p=0.9),
                prob_accept=ss.bernoulli(p=hiud_accept),   # swept
            ),
        )
    )

    sim = fp.Sim(
        start=START,
        stop=STOP,
        n_agents=10000,
        total_pop=55_000_000,
        location='kenya',
        education_module=edu,
        connectors=[mens],
        interventions=[cascade],
        analyzers=[track_hmb_anemia()],
        rand_seed=seed,
        verbose=0,
    )
    return sim


def run_calibration(force_rerun=True):
    """Sweep hIUD acceptance for each care-seeking scenario."""
    results_file = OUTFOLDER + 'hiud_calibration_accept25.obj'

    if not force_rerun and os.path.exists(results_file):
        print("Loading saved calibration...")
        return sc.loadobj(results_file)

    results = {}

    for care_label, care_behavior in CARE_SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"Care-seeking: {care_label}  |  NSAID/TXA/Pill accept = {FIXED_ACCEPT}")
        print(f"{'='*60}")

        results[care_label] = {}

        for accept_val in HIUD_ACCEPT_VALUES:
            accept_key = f'{accept_val:.1f}'
            print(f"\n  hIUD accept = {accept_val:.1f}")
            uptakes = []

            for seed in range(N_SEEDS):
                print(f"    seed {seed}...", end=" ", flush=True)
                sim = make_sim(care_behavior, hiud_accept=accept_val, seed=seed)
                sim.run()

                cascade_intv = sim.interventions.hmb_cascade
                menstruating = sim.people.menstruation.menstruating
                hmb = sim.people.menstruation.hmb

                n_treatments = (
                    np.array(cascade_intv.tried_nsaid, dtype=int) +
                    np.array(cascade_intv.tried_txa, dtype=int) +
                    np.array(cascade_intv.tried_pill, dtype=int) +
                    np.array(cascade_intv.tried_hiud, dtype=int)
                )

                # % of menstruating women with underlying HMB (including those on treatment)
                hmb_underlying = (hmb | cascade_intv.on_any_treatment) & menstruating
                hmb_menstruating = hmb_underlying
                
                n_hmb = np.count_nonzero(hmb_menstruating)
                tried_hiud = cascade_intv.treatments['hiud'].tried_treatment & hmb_menstruating
                n_hiud = np.count_nonzero(tried_hiud)
                pct_hmb = 100 * n_hiud / n_hmb if n_hmb > 0 else 0

                # % of those who tried any treatment who tried hIUD
                tried_any = (n_treatments >= 1) & menstruating
                n_tried_any = np.count_nonzero(tried_any)
                tried_hiud_any = cascade_intv.treatments['hiud'].tried_treatment & tried_any
                n_hiud_any = np.count_nonzero(tried_hiud_any)
                pct_treated = 100 * n_hiud_any / n_tried_any if n_tried_any > 0 else 0

                # % of HMB women who ever sought care (offered NSAID)
                ever_offered = cascade_intv.treatments['nsaid'].offered & hmb_menstruating
                n_seekers = np.count_nonzero(ever_offered)
                tried_hiud_seekers = cascade_intv.treatments['hiud'].tried_treatment & ever_offered
                n_hiud_seekers = np.count_nonzero(tried_hiud_seekers)
                pct_seekers = 100 * n_hiud_seekers / n_seekers if n_seekers > 0 else 0

                uptakes.append({
                    'pct_of_hmb': pct_hmb,
                    'pct_of_treated': pct_treated,
                    'pct_of_seekers': pct_seekers,
                    'n_hmb': n_hmb,
                    'n_hiud': n_hiud,
                    'n_tried_any': n_tried_any,
                    'n_seekers': n_seekers,
                })
                print("done")

            results[care_label][accept_key] = uptakes

    sc.saveobj(results_file, results)
    print(f"\nSaved: {results_file}")
    return results


def plot_calibration(results):
    """Plot calibration curves for all three care-seeking scenarios."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'hIUD acceptance calibration (NSAID/TXA/Pill accept = {int(FIXED_ACCEPT*100)}%)',
                 fontsize=14)

    panels = [
        ('pct_of_hmb', '% of HMB women\nwho ever tried hIUD', axes[0]),
        ('pct_of_seekers', '% of HMB care-seekers\nwho tried hIUD', axes[1]),
        ('pct_of_treated', '% of treated women\nwho tried hIUD', axes[2]),
    ]

    for metric_key, ylabel, ax in panels:
        for care_label in CARE_SCENARIOS:
            accept_vals = []
            means = []
            stds = []

            for accept_key in sorted(results[care_label].keys(), key=float):
                accept_vals.append(float(accept_key))
                vals = [u[metric_key] for u in results[care_label][accept_key]]
                means.append(np.mean(vals))
                stds.append(np.std(vals))

            ax.errorbar(accept_vals, means, yerr=stds,
                        marker='o', capsize=4, lw=2,
                        color=CARE_COLORS[care_label],
                        label=f'Base {care_label}')

        # Target lines
        for target, color, ls in [(5, '#2196F3', ':'), (10, '#4CAF50', '--'), (15, '#F44336', '-.')]:
            ax.axhline(target, ls=ls, color=color, lw=1, alpha=0.5,
                       label=f'Target: {target}%')

        ax.set_xlabel('hIUD acceptance probability')
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    outpath = OUTFOLDER + 'hiud_calibration_accept25.png'
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpath}")

    # Print lookup tables
    for care_label in CARE_SCENARIOS:
        print(f"\n{'─'*70}")
        print(f"  Care-seeking: {care_label}  |  NSAID/TXA/Pill accept = {int(FIXED_ACCEPT*100)}%")
        print(f"{'─'*70}")
        print(f"  {'Accept':>8}  {'% of HMB':>10}  {'% of seekers':>14}  {'% of treated':>14}")
        print(f"{'─'*70}")

        for accept_key in sorted(results[care_label].keys(), key=float):
            vals_hmb = [u['pct_of_hmb'] for u in results[care_label][accept_key]]
            vals_seek = [u['pct_of_seekers'] for u in results[care_label][accept_key]]
            vals_treat = [u['pct_of_treated'] for u in results[care_label][accept_key]]
            print(f"  {float(accept_key):>8.1f}  "
                  f"{np.mean(vals_hmb):>9.1f}%  "
                  f"{np.mean(vals_seek):>13.1f}%  "
                  f"{np.mean(vals_treat):>13.1f}%")

        print(f"{'─'*70}")

    return fig


if __name__ == '__main__':
    results = run_calibration(force_rerun=True)
    plot_calibration(results)