# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:58:52 2026

@author: navidehno
"""

"""
Calibrate p_hmb_prone so that with status quo treatment,
observed HMB prevalence ≈ 48%.
"""

import numpy as np
import sciris as sc
import starsim as ss
import fpsim as fp
import matplotlib.pyplot as plt

from menstruation import Menstruation
from education import Education
from interventions_pool import HMBCounterfactual
from analyzers import track_hmb_anemia

# ── Settings ──
START = 2020
STOP = 2030
N_SEEDS = 5
TARGET_PREV = 0.48

# Values to sweep
P_HMB_PRONE_VALUES = np.arange(0.48, 0.61, 0.01)


def make_sim(p_hmb_prone, with_treatment=True, seed=0):
    """Build sim with a specific p_hmb_prone value."""
    mens = Menstruation(pars=dict(
        p_hmb_prone=ss.bernoulli(p=p_hmb_prone),
    ))
    edu = Education()

    sim_kwargs = dict(
        start=START, stop=STOP,
        n_agents=10000, total_pop=55_000_000,
        location='kenya',
        education_module=edu,
        connectors=[mens],
        analyzers=[track_hmb_anemia()],
        rand_seed=seed, verbose=0,
    )

    if with_treatment:
        # Status quo: 10% ever-seek, no hIUD, runs from 2020
        counterfactual = HMBCounterfactual()
        sim_kwargs['interventions'] = [counterfactual]

    return fp.Sim(**sim_kwargs)


def run_calibration():
    results = {}

    for p_val in P_HMB_PRONE_VALUES:
        p_key = f'{p_val:.2f}'
        print(f"\np_hmb_prone = {p_val:.2f}")
        prevs_with_tx = []
        prevs_no_tx = []

        for seed in range(N_SEEDS):
            print(f"  seed {seed}...", end=" ", flush=True)

            # With status quo treatment
            sim_tx = make_sim(p_val, with_treatment=True, seed=seed)
            sim_tx.run()
            hmb_prev_tx = np.asarray(
                sim_tx.results.menstruation['hmb_prev']
            )
            # Take mean of last 12 months as the steady-state prevalence
            prevs_with_tx.append(hmb_prev_tx[-12:].mean())

            # Without any treatment (for comparison)
            sim_no = make_sim(p_val, with_treatment=False, seed=seed)
            sim_no.run()
            hmb_prev_no = np.asarray(
                sim_no.results.menstruation['hmb_prev']
            )
            prevs_no_tx.append(hmb_prev_no[-12:].mean())

            del sim_tx, sim_no
            print("done")

        results[p_key] = {
            'with_tx_mean': np.mean(prevs_with_tx),
            'with_tx_std': np.std(prevs_with_tx),
            'no_tx_mean': np.mean(prevs_no_tx),
            'no_tx_std': np.std(prevs_no_tx),
        }

        print(f"  With treatment: {results[p_key]['with_tx_mean']:.3f} "
              f"± {results[p_key]['with_tx_std']:.3f}")
        print(f"  No treatment:   {results[p_key]['no_tx_mean']:.3f} "
              f"± {results[p_key]['no_tx_std']:.3f}")

    return results


def plot_calibration(results):
    p_vals = [float(k) for k in results.keys()]
    with_tx_means = [results[k]['with_tx_mean'] for k in results.keys()]
    with_tx_stds = [results[k]['with_tx_std'] for k in results.keys()]
    no_tx_means = [results[k]['no_tx_mean'] for k in results.keys()]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(p_vals, with_tx_means, yerr=with_tx_stds,
                marker='o', capsize=4, lw=2, color='#d62728',
                label='With status quo treatment')
    ax.plot(p_vals, no_tx_means,
            marker='s', lw=2, color='#1f77b4', alpha=0.5,
            label='No treatment')

    ax.axhline(TARGET_PREV, color='black', ls='--', lw=1.5,
               label=f'Target: {TARGET_PREV:.0%}')

    # Find closest match
    diffs = [abs(m - TARGET_PREV) for m in with_tx_means]
    best_idx = np.argmin(diffs)
    best_p = p_vals[best_idx]
    best_prev = with_tx_means[best_idx]
    ax.axvline(best_p, color='green', ls=':', lw=1.5,
               label=f'Best fit: p={best_p:.2f} → {best_prev:.3f}')

    ax.set_xlabel('p_hmb_prone')
    ax.set_ylabel('Observed HMB prevalence')
    ax.set_title('Calibrating p_hmb_prone with status quo treatment')
    ax.legend(frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    fig.savefig('hmb_prone_calibration.png', dpi=300, bbox_inches='tight')
    print(f"\nBest fit: p_hmb_prone = {best_p:.2f} "
          f"→ observed prevalence = {best_prev:.3f}")
    return fig


if __name__ == '__main__':
    results = run_calibration()
    plot_calibration(results)