"""
Analysis script for HMB intervention impact on anemia prevention

Creates baseline and intervention simulations, runs them as MultiSim for
statistical robustness, and saves results for plotting.
"""

import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import fpsim as fp

from menstruation import Menstruation
from education import Education
from interventions import HMBCarePathway
from analyzers import track_care_seeking, track_tx_eff, track_tx_dur, track_hmb_anemia, track_cascade


def make_base_sim(seed=0):
    """
    Create baseline simulation without intervention

    Returns a simulation with HMB dynamics but no care pathway intervention
    """
    mens = Menstruation()
    edu = Education()
    care_analyzer = track_care_seeking()
    hmb_anemia_analyzer = track_hmb_anemia()

    sim = fp.Sim(
        start=2020,
        stop=2030,
        n_agents=5000,
        total_pop=55_000_000,  # Kenya's population for scaling
        location='kenya',
        education_module=edu,
        connectors=[mens],
        analyzers=[care_analyzer, hmb_anemia_analyzer],
        rand_seed=seed,
        verbose=0,
    )
    return sim


def make_intervention_sim(seed=0):
    """
    Create simulation with HMB care pathway intervention

    Returns a simulation with full care pathway including NSAID, TXA, Pill, and hIUD
    """
    mens = Menstruation()
    edu = Education()
    pathway = HMBCarePathway(
        year=2020,
        time_to_assess=3,  # Assess treatment effectiveness after 3 months
    )
    care_analyzer = track_care_seeking()
    tx_eff_analyzer = track_tx_eff()
    tx_dur_analyzer = track_tx_dur()
    hmb_anemia_analyzer = track_hmb_anemia()

    sim = fp.Sim(
        start=2020,
        stop=2030,
        n_agents=5000,
        total_pop=55_000_000,  # Kenya's population for scaling
        location='kenya',
        education_module=edu,
        connectors=[mens],
        interventions=[pathway],
        analyzers=[care_analyzer, tx_eff_analyzer, tx_dur_analyzer, hmb_anemia_analyzer],
        rand_seed=seed,
        verbose=0,
    )
    return sim


def make_iud_only_sim(seed=0):
    """
    Create simulation with IUD-only intervention

    Returns a simulation with care pathway offering only hIUD (no NSAID, TXA, or Pill)
    """
    mens = Menstruation()
    edu = Education()
    pathway = HMBCarePathway(
        year=2020,
        time_to_assess=3,  # Assess treatment effectiveness after 3 months
        prob_offer=sc.objdict(
            nsaid=ss.bernoulli(p=0.0),  # Don't offer NSAID
            txa=ss.bernoulli(p=0.0),    # Don't offer TXA
            pill=ss.bernoulli(p=0.0),   # Don't offer pill
            hiud=ss.bernoulli(p=0.9),   # Only offer hIUD
        ),
    )
    care_analyzer = track_care_seeking()
    tx_eff_analyzer = track_tx_eff()
    tx_dur_analyzer = track_tx_dur()
    hmb_anemia_analyzer = track_hmb_anemia()

    sim = fp.Sim(
        start=2020,
        stop=2030,
        n_agents=5000,
        total_pop=55_000_000,  # Kenya's population for scaling
        location='kenya',
        education_module=edu,
        connectors=[mens],
        interventions=[pathway],
        analyzers=[care_analyzer, tx_eff_analyzer, tx_dur_analyzer, hmb_anemia_analyzer],
        rand_seed=seed,
        verbose=0,
    )
    return sim


def run_scenarios(n_runs=10, save_results=True):
    """
    Run baseline, full intervention, and IUD-only intervention scenarios as MultiSim

    Args:
        n_runs: Number of runs for each scenario (default: 10)
        save_results: Whether to save results to disk (default: True)

    Returns:
        Dictionary with 'baseline', 'intervention', and 'iud_only' MultiSim objects
    """
    print(f'Running {n_runs} simulations for each scenario...')

    # Create baseline MultiSim
    print('Running baseline scenario...')
    base_sims = [make_base_sim(seed=i) for i in range(n_runs)]
    msim_base = ss.MultiSim(base_sims)
    msim_base.run()

    # Create full intervention MultiSim
    print('Running full intervention scenario (NSAID→TXA→Pill→hIUD)...')
    intervention_sims = [make_intervention_sim(seed=i) for i in range(n_runs)]
    msim_intervention = ss.MultiSim(intervention_sims)
    msim_intervention.run()

    # Create IUD-only intervention MultiSim
    print('Running IUD-only intervention scenario...')
    iud_only_sims = [make_iud_only_sim(seed=i) for i in range(n_runs)]
    msim_iud_only = ss.MultiSim(iud_only_sims)
    msim_iud_only.run()

    # Save results
    if save_results:
        print('Saving results...')
        sc.saveobj('results/baseline_msim.obj', msim_base)
        sc.saveobj('results/intervention_msim.obj', msim_intervention)
        sc.saveobj('results/iud_only_msim.obj', msim_iud_only)
        print('Results saved to results/ directory')

    print('Done!')

    return {
        'baseline': msim_base,
        'intervention': msim_intervention,
        'iud_only': msim_iud_only,
    }


def make_cascade_base_sim(seed=0):
    """
    Create baseline simulation for cascade analysis (no intervention)

    Returns a simulation without cascade analyzer since there's no treatment cascade
    """
    mens = Menstruation()
    edu = Education()
    hmb_anemia_analyzer = track_hmb_anemia()

    sim = fp.Sim(
        start=2020,
        stop=2030,
        n_agents=5000,
        total_pop=55_000_000,
        location='kenya',
        education_module=edu,
        connectors=[mens],
        analyzers=[hmb_anemia_analyzer],
        rand_seed=seed,
        verbose=0,
    )
    return sim


def make_cascade_intervention_sim(seed=0):
    """
    Create simulation with HMB care pathway and cascade analyzer

    Returns a simulation with full pathway and detailed cascade tracking
    """
    mens = Menstruation()
    edu = Education()
    pathway = HMBCarePathway(
        year=2020,
        time_to_assess=3,
    )
    cascade_analyzer = track_cascade()
    hmb_anemia_analyzer = track_hmb_anemia()

    sim = fp.Sim(
        start=2020,
        stop=2030,
        n_agents=5000,
        total_pop=55_000_000,
        location='kenya',
        education_module=edu,
        connectors=[mens],
        interventions=[pathway],
        analyzers=[cascade_analyzer, hmb_anemia_analyzer],
        rand_seed=seed,
        verbose=0,
    )
    return sim


def run_cascade_analysis(save_results=True):
    """
    Run baseline and intervention for cascade analysis

    Args:
        save_results: Whether to save results to disk (default: True)

    Returns:
        Dictionary with 'baseline' and 'intervention' Sim objects
    """
    print('Running cascade analysis...')

    # Run baseline
    print('Running baseline simulation...')
    baseline_sim = make_cascade_base_sim(seed=0)
    baseline_sim.run()

    # Run intervention with cascade tracking
    print('Running intervention simulation...')
    intervention_sim = make_cascade_intervention_sim(seed=0)
    intervention_sim.run()

    # Save results
    if save_results:
        print('Saving cascade analysis results...')
        sc.path('results').mkdir(exist_ok=True)
        sc.saveobj('results/cascade_baseline_sim.obj', baseline_sim)
        sc.saveobj('results/cascade_intervention_sim.obj', intervention_sim)
        print('Results saved to results/cascade_*.obj')

    print('Cascade analysis complete!')

    return {
        'baseline': baseline_sim,
        'intervention': intervention_sim,
    }


if __name__ == '__main__':
    # Configuration
    n_runs = 10  # Number of stochastic runs per scenario

    # Run scenarios
    results = run_scenarios(n_runs=n_runs, save_results=True)

    # Run cascade analysis
    cascade_results = run_cascade_analysis(save_results=True)
