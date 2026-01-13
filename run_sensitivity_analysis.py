# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 13:32:46 2025

@author: kirstinol
"""



import numpy as np
import pandas as pd
import pylab as pl
import sciris as sc
# starsim
import starsim as ss
# fpsim
import fpsim as fp
from fpsim import plotting as plt
# hmb
from menstruation import Menstruation
from education import Education
from interventions import hiud_hmb, txa, pill_hmb, hmb_package
# run functions
from run_kenya_package import make_pars, make_sim, set_font



# parameter ranges
# - probability of accepting each intervention (hIUD, TXA, pill)
# - effectiveness of interventions (relative effectiveness and multiples of all effectiveness)
# - probability of anemia with and without HMB 
# - ...



"""
Three separate sensitivity analyses:
1. Varying acceptance probabilities for interventions
2. Varying effectiveness of interventions
3. Varying probability of anemia with/without HMB
"""


plotfolder = 'figures_sensitivity/'
outfolder = 'results_sensitivity/'


def run_acceptance_sensitivity():
    """
    Sensitivity analysis 1: Vary acceptance probabilities for interventions
    """
    print("\n" + "="*60)
    print("RUNNING SENSITIVITY ANALYSIS 1: ACCEPTANCE PROBABILITIES")
    print("="*60)
    
    # Define acceptance probability combinations to test
    acceptance_scenarios = [
        {'name': 'low_all', 'hiud': 0.2, 'txa': 0.2, 'pill': 0.2},
        {'name': 'medium_all', 'hiud': 0.5, 'txa': 0.5, 'pill': 0.5},
        {'name': 'high_all', 'hiud': 0.8, 'txa': 0.8, 'pill': 0.8},
        {'name': 'high_hiud', 'hiud': 0.8, 'txa': 0.3, 'pill': 0.3},
        {'name': 'high_txa', 'hiud': 0.3, 'txa': 0.8, 'pill': 0.3},
        {'name': 'high_pill', 'hiud': 0.3, 'txa': 0.3, 'pill': 0.8},
        {'name': 'mixed_1', 'hiud': 0.7, 'txa': 0.5, 'pill': 0.3},
        {'name': 'mixed_2', 'hiud': 0.3, 'txa': 0.5, 'pill': 0.7},
    ]
    
    # Fixed parameters
    prob_offer = 0.4
    n_seeds = 5
    max_attempts = 50
    
    # Storage for results
    all_results = {}
    
    # Run baseline (no intervention)
    print("Running baseline scenario...")
    baseline_results = {'hiud': [], 'hmb': [], 'poor_mh': [], 'anemic': [], 'pain': []}
            
    successful_seeds = 0
    seed = 0
    while successful_seeds < n_seeds and seed < max_attempts:
        try:
            s_base = make_sim(stop=2032)
            s_base['pars']['rand_seed'] = seed
            s_base.run()
            
            # If successful, collect results
            for res in baseline_results.keys():
                baseline_results[res].append(s_base.results.menstruation[f'{res}_prev'][::12])
            
            successful_seeds += 1
            print(f"  Baseline seed {seed}: success ({successful_seeds}/{n_seeds} completed)")
            
        except ValueError as e:
            if "Postpartum women should not currently be using contraception" in str(e):
                print(f"  Baseline seed {seed}: skipped due to postpartum contraception error")
            else:
                raise e  # Re-raise if it's a different error
        
        seed += 1
    
    if successful_seeds < n_seeds:
        print(f"  Warning: Only completed {successful_seeds}/{n_seeds} seeds for baseline after {max_attempts} attempts")
    # Calculate statistics for baseline
    all_results['baseline'] = {}
    for res in baseline_results.keys():
        arr = np.array(baseline_results[res])
        all_results['baseline'][res] = {
            'mean': np.mean(arr, axis=0),
            'lower': np.percentile(arr, 2.5, axis=0),
            'upper': np.percentile(arr, 97.5, axis=0)
        }
    
    # Run each acceptance scenario
    for scenario in acceptance_scenarios:
        print(f"Running scenario: {scenario['name']}...")
        scenario_results = {'hiud': [], 'hmb': [], 'poor_mh': [], 'anemic': [], 'pain': []}
        
        for seed in range(n_seeds):
            s_int = make_sim(stop=2032)
            s_int['pars']['rand_seed'] = seed
            s_int['pars']['interventions'] = [
                hmb_package(
                    prob_offer=prob_offer,
                    prob_accept_hiud=scenario['hiud'],
                    prob_accept_txa=scenario['txa'],
                    prob_accept_pill=scenario['pill']
                )
            ]
            s_int.run()
            
            for res in scenario_results.keys():
                scenario_results[res].append(s_int.results.menstruation[f'{res}_prev'][::12])
        
        # Calculate statistics
        all_results[scenario['name']] = {}
        for res in scenario_results.keys():
            arr = np.array(scenario_results[res])
            all_results[scenario['name']][res] = {
                'mean': np.mean(arr, axis=0),
                'lower': np.percentile(arr, 2.5, axis=0),
                'upper': np.percentile(arr, 97.5, axis=0)
            }
    
    # Save results
    sc.saveobj(outfolder+'sensitivity_acceptance_results-stats.obj', all_results)
    
    # Plot results
    plot_acceptance_results(all_results)
    
    return all_results




def run_effectiveness_sensitivity():
    """
    Sensitivity analysis 2: Vary effectiveness of interventions
    """
    print("\n" + "="*60)
    print("RUNNING SENSITIVITY ANALYSIS 2: INTERVENTION EFFECTIVENESS")
    print("="*60)
    
    # Define effectiveness multipliers to test
    effectiveness_scenarios = [
        {'name': 'very_low', 'multiplier': 0.25},
        {'name': 'low', 'multiplier': 0.5},
        {'name': 'baseline', 'multiplier': 1.0},
        {'name': 'high', 'multiplier': 1.5},
        {'name': 'very_high', 'multiplier': 2.0},
    ]
    
    # Fixed parameters
    prob_offer = 0.4
    prob_accept = 0.5  # Fixed acceptance probability
    n_seeds = 5
    max_attempts = 50
    
    # Storage for results
    all_results = {}
    
    # Run baseline (no intervention)
    print("Running baseline scenario...")
    baseline_results = {'hiud': [], 'hmb': [], 'poor_mh': [], 'anemic': [], 'pain': []}
    
    for seed in range(n_seeds):
        s_base = make_sim(stop=2032)
        s_base['pars']['rand_seed'] = seed
        s_base.run()
        
        for res in baseline_results.keys():
            baseline_results[res].append(s_base.results.menstruation[f'{res}_prev'][::12])
    
    # Calculate statistics for baseline
    all_results['baseline'] = {}
    for res in baseline_results.keys():
        arr = np.array(baseline_results[res])
        all_results['baseline'][res] = {
            'mean': np.mean(arr, axis=0),
            'lower': np.percentile(arr, 2.5, axis=0),
            'upper': np.percentile(arr, 97.5, axis=0)
        }
    
    # Run each effectiveness scenario
    for scenario in effectiveness_scenarios:
        print(f"\nRunning scenario: {scenario['name']} (multiplier={scenario['multiplier']})...")
        scenario_results = {'hiud': [], 'hmb': [], 'poor_mh': [], 'anemic': [], 'pain': []}
        
        successful_seeds = 0
        seed = 0
        while successful_seeds < n_seeds and seed < max_attempts:
            try:
                # Create custom Menstruation connector with modified effectiveness
                mult = scenario['multiplier']
                mens = Menstruation()
                mens.pars.hmb_pred = sc.objdict(
                    base=0.5,
                    pill=-np.log(1/((1 - 0.25*0.312*mult) * 0.5) -1) - np.log(1/0.5 -1),
                    hiud=-np.log(1/((1 - 0.312*mult) * 0.5) -1) - np.log(1/0.5 -1),
                    txa=-np.log(1/((1 - 0.5*0.312*mult) * 0.5) -1) - np.log(1/0.5 -1),
                )
                
                # Get the base parameters from make_pars
                pars = make_pars()
                
                # Get education module
                objective_data = pd.read_csv("data/kenya_objective.csv")
                attainment_data = pd.read_csv("data/kenya_initialization.csv")
                edu = Education(objective_data=objective_data, attainment_data=attainment_data)
                
                # Create simulation with custom menstruation connector
                s_int = fp.Sim(
                    start=2000,
                    stop=2032,
                    n_agents=1000,
                    location='kenya',
                    pars=pars,
                    analyzers=[fp.cpr_by_age(), fp.method_mix_by_age()],
                    education_module=edu,
                    connectors=[mens],  # Use the custom menstruation connector
                )
                
                s_int['pars']['rand_seed'] = seed
                
                # Add intervention
                s_int['pars']['interventions'] = [
                    hmb_package(
                        prob_offer=prob_offer,
                        prob_accept_hiud=prob_accept,
                        prob_accept_txa=prob_accept,
                        prob_accept_pill=prob_accept
                    )
                ]
                
                s_int.run()
                
                for res in scenario_results.keys():
                    scenario_results[res].append(s_int.results.menstruation[f'{res}_prev'][::12])
                
                successful_seeds += 1
                print(f"  Seed {seed}: success ({successful_seeds}/{n_seeds} completed)")
                
            except ValueError as e:
                if "Postpartum women should not currently be using contraception" in str(e):
                    print(f"  Seed {seed}: skipped due to postpartum contraception error")
                else:
                    raise e
            
            seed += 1
        
        # Calculate statistics
        all_results[scenario['name']] = {}
        for res in scenario_results.keys():
            arr = np.array(scenario_results[res])
            all_results[scenario['name']][res] = {
                'mean': np.mean(arr, axis=0),
                'lower': np.percentile(arr, 2.5, axis=0),
                'upper': np.percentile(arr, 97.5, axis=0)
            }
    
    # Save results
    sc.saveobj(outfolder+'sensitivity_effectiveness_results.obj', all_results)
    
    # Plot results
    plot_effectiveness_results(all_results)
    
    return all_results


def run_anemia_sensitivity():
    """
    Sensitivity analysis 3: Vary probability of anemia with/without HMB
    """
    print("\n" + "="*60)
    print("RUNNING SENSITIVITY ANALYSIS 3: ANEMIA PROBABILITY")
    print("="*60)
    
    # Define anemia parameter scenarios using probabilities
    anemia_scenarios = [
        {'name': 'very_low', 'prob_base': 0.02, 'prob_hmb': 0.04},
        {'name': 'low_base_low_hmb', 'prob_base': 0.05, 'prob_hmb': 0.10},
        {'name': 'low_base_high_hmb', 'prob_base': 0.05, 'prob_hmb': 0.40},
        {'name': 'baseline', 'prob_base': 0.01, 'prob_hmb': 0.0433},
        {'name': 'moderate', 'prob_base': 0.10, 'prob_hmb': 0.25},
        {'name': 'high_base_low_hmb', 'prob_base': 0.20, 'prob_hmb': 0.30},
        {'name': 'high_base_high_hmb', 'prob_base': 0.20, 'prob_hmb': 0.50},
        {'name': 'very_high_base', 'prob_base': 0.30, 'prob_hmb': 0.45},
        {'name': 'very_high_hmb', 'prob_base': 0.15, 'prob_hmb': 0.60},
    ]
    
    # Fixed parameters
    prob_offer = 0.4
    prob_accept = 0.5
    n_seeds = 10
    max_attempts = 50
    
    # Storage for results
    all_results = {}
    
    # Get education module (load once to reuse)
    objective_data = pd.read_csv("data/kenya_objective.csv")
    attainment_data = pd.read_csv("data/kenya_initialization.csv")
    #edu_base = Education(objective_data=objective_data, attainment_data=attainment_data)
    
    # Run each anemia scenario (with and without intervention)
    for scenario in anemia_scenarios:
        print(f"\nRunning scenario: {scenario['name']} (base={scenario['prob_base']:.1%}, hmb={scenario['prob_hmb']:.1%})...")
        
        # Results storage for this scenario
        baseline_results = {'hiud': [], 'hmb': [], 'poor_mh': [], 'anemic': [], 'pain': []}
        intervention_results = {'hiud': [], 'hmb': [], 'poor_mh': [], 'anemic': [], 'pain': []}
        
        successful_seeds = 0
        seed = 0
        while successful_seeds < n_seeds and seed < max_attempts:
            try:
                # Get base parameters for baseline simulation
                pars_base = make_pars()
                
                # Create custom Menstruation connector with modified anemia parameters for baseline
                mens_base = Menstruation()
                prob_base = scenario['prob_base']
                prob_hmb = scenario['prob_hmb']
                
                mens_base.pars.hmb_seq.anemic = sc.objdict(
                    base = prob_base,
                    hmb = -np.log(1/prob_hmb - 1) + np.log(1/prob_base - 1),
                )
                
                # Create new education module for baseline (to avoid state conflicts)
                edu_baseline = Education(objective_data=objective_data, attainment_data=attainment_data)
                
                # Baseline simulation
                s_base = fp.Sim(
                    start=2000,
                    stop=2032,
                    n_agents=1000,
                    location='kenya',
                    pars=pars_base,
                    analyzers=[fp.cpr_by_age(), fp.method_mix_by_age()],
                    education_module=edu_baseline,
                    connectors=[mens_base],
                )
                s_base['pars']['rand_seed'] = seed
                s_base.run()
                
                # Get new parameters for intervention simulation
                pars_int = make_pars()
                
                # Create custom Menstruation connector for intervention
                mens_int = Menstruation()
                mens_int.pars.hmb_seq.anemic = sc.objdict(
                    base = prob_base,
                    hmb = -np.log(1/prob_hmb - 1) + np.log(1/prob_base - 1),
                )
                
                # Create new education module for intervention
                edu_intervention = Education(objective_data=objective_data, attainment_data=attainment_data)
                
                # Intervention simulation
                s_int = fp.Sim(
                    start=2000,
                    stop=2032,
                    n_agents=1000,
                    location='kenya',
                    pars=pars_int,
                    analyzers=[fp.cpr_by_age(), fp.method_mix_by_age()],
                    education_module=edu_intervention,
                    connectors=[mens_int],
                )
                s_int['pars']['rand_seed'] = seed
                
                # Add intervention
                s_int['pars']['interventions'] = [
                    hmb_package(
                        prob_offer=prob_offer,
                        prob_accept_hiud=prob_accept,
                        prob_accept_txa=prob_accept,
                        prob_accept_pill=prob_accept
                    )
                ]
                
                s_int.run()
                
                # Collect results
                for res in baseline_results.keys():
                    baseline_results[res].append(s_base.results.menstruation[f'{res}_prev'][::12])
                    intervention_results[res].append(s_int.results.menstruation[f'{res}_prev'][::12])
                
                successful_seeds += 1
                print(f"  Seed {seed}: success ({successful_seeds}/{n_seeds} completed)")
                
            except ValueError as e:
                if "Postpartum women should not currently be using contraception" in str(e):
                    print(f"  Seed {seed}: skipped due to postpartum contraception error")
                else:
                    raise e
            except Exception as e:
                print(f"  Seed {seed}: unexpected error: {e}")
                # Continue to next seed
            
            seed += 1
        
        if successful_seeds < n_seeds:
            print(f"  Warning: Only completed {successful_seeds}/{n_seeds} seeds for {scenario['name']} after {max_attempts} attempts")
        
        # Calculate statistics
        all_results[f"{scenario['name']}_baseline"] = {
            'prob_base': scenario['prob_base'],
            'prob_hmb': scenario['prob_hmb']
        }
        all_results[f"{scenario['name']}_intervention"] = {
            'prob_base': scenario['prob_base'],
            'prob_hmb': scenario['prob_hmb']
        }
        
        for res in baseline_results.keys():
            if len(baseline_results[res]) > 0:
                # Baseline statistics
                arr_base = np.array(baseline_results[res])
                all_results[f"{scenario['name']}_baseline"][res] = {
                    'mean': np.mean(arr_base, axis=0),
                    'lower': np.percentile(arr_base, 2.5, axis=0),
                    'upper': np.percentile(arr_base, 97.5, axis=0),
                    'n_seeds': len(baseline_results[res])
                }
                
                # Intervention statistics
                arr_int = np.array(intervention_results[res])
                all_results[f"{scenario['name']}_intervention"][res] = {
                    'mean': np.mean(arr_int, axis=0),
                    'lower': np.percentile(arr_int, 2.5, axis=0),
                    'upper': np.percentile(arr_int, 97.5, axis=0),
                    'n_seeds': len(intervention_results[res])
                }
        
        print(f"  Completed {scenario['name']} with {successful_seeds} successful seeds")
    
    # Save results
    sc.saveobj('results/sensitivity_anemia_results.obj', all_results)
    
    # Plot results
    plot_anemia_results(all_results)
    
    return all_results





def plot_acceptance_results(results):
    """Plot results from acceptance probability sensitivity analysis"""
    
    # Get time vector
    t = np.arange(2000, 2033)
    si = np.where(t >= 2020)[0][0]
    years = t[si:]
    
    set_font(20)
    
    # Create figure with subplots
    fig, axes = pl.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    res_to_plot = ['hiud', 'hmb', 'poor_mh', 'anemic', 'pain']
    labels = ['hIUD Usage', 'HMB', 'Poor MH', 'Anemic', 'Pain']
    
    # Define colors for different scenarios
    colors = {
        'baseline': '#6c757d',
        'low_all': '#6f1926',
        'medium_all': '#de324c',
        'high_all': '#f4895f',
        'high_hiud': '#95cf92',
        'high_txa': '#369acc',
        'high_pill': '#9656a2',
        'mixed_1': '#cbabd1',
        'mixed_2': '#f8e16f',
    }
    
    scenario_labels = {
        'baseline': 'Baseline',
        'low_all': 'Low all (20%)',
        'medium_all': 'Medium all (50%)',
        'high_all': 'High all (80%)',
        'high_hiud': 'High hIUD (80%)',
        'high_txa': 'High TXA (80%)',
        'high_pill': 'High Pill (80%)',
        'mixed_1': 'Mixed (hIUD 70%, TXA 50%, Pill 30%)',
        'mixed_2': 'Mixed (hIUD 30%, TXA 50%, Pill 70%)',
    }
    
    lw = 2.0
    
    for i, res in enumerate(res_to_plot):
        ax = axes[i]
        
        # Plot each scenario
        for scenario_name, color in colors.items():
            if scenario_name in results:
                mean = results[scenario_name][res]['mean'][si:] * 100
                lower = results[scenario_name][res]['lower'][si:] * 100
                upper = results[scenario_name][res]['upper'][si:] * 100
                
                ax.plot(years, mean, label=scenario_labels[scenario_name], 
                       color=color, linewidth=lw)
                ax.fill_between(years, lower, upper, color=color, alpha=0.1)
        
        # Add vertical line for intervention start
        ax.axvline(x=2026, color='k', ls='--', linewidth=1.5)
        if i == 0:
            ax.text(2025.5, ax.get_ylim()[1] * 0.9, 'Start of\nintervention', 
                   ha='right', va='top', fontsize=10, color='#4d4d4d')
        
        ax.set_title(labels[i])
        ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if i in [0, 3]:
            ax.set_ylabel('Prevalence (%)')
        if i >= 3:
            ax.set_xlabel('Year')
    
    # Add legend
    axes[5].axis('off')
    axes[5].legend(*axes[0].get_legend_handles_labels(), fontsize=12, frameon=False, loc='center')
    
    pl.suptitle('Sensitivity Analysis: Acceptance Probabilities', fontsize=22, y=1.02)
    sc.figlayout()
    sc.savefig(plotfolder+'sensitivity_acceptance_timeseries.png', dpi=150)


def plot_effectiveness_results(results):
    """Plot results from effectiveness sensitivity analysis"""
    
    # Get time vector
    t = np.arange(2000, 2033)
    si = np.where(t >= 2020)[0][0]
    years = t[si:]
    
    set_font(20)
    
    # Create figure with subplots
    fig, axes = pl.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()
    
    res_to_plot = ['hiud', 'hmb', 'poor_mh', 'anemic', 'pain']
    labels = ['hIUD Usage', 'HMB', 'Poor MH', 'Anemic', 'Pain']
    
    # Define colors for different effectiveness levels
    colors = {
        'baseline': '#6c757d',
        'very_low': '#fee5d9',
        'low': '#fcae91',
        'baseline_int': '#fb6a4a',
        'high': '#de2d26',
        'very_high': '#a50f15',
    }
    
    scenario_labels = {
        'baseline': 'No intervention',
        'very_low': 'Very low effectiveness (0.25x)',
        'low': 'Low effectiveness (0.5x)',
        'baseline_int': 'Baseline effectiveness (1.0x)',
        'high': 'High effectiveness (1.5x)',
        'very_high': 'Very high effectiveness (2.0x)',
    }
    
    lw = 2.0
    
    for i, res in enumerate(res_to_plot):
        ax = axes[i]
        
        # Plot baseline first
        if 'baseline' in results:
            mean = results['baseline'][res]['mean'][si:] * 100
            lower = results['baseline'][res]['lower'][si:] * 100
            upper = results['baseline'][res]['upper'][si:] * 100
            
            ax.plot(years, mean, label=scenario_labels['baseline'], 
                   color=colors['baseline'], linewidth=lw, linestyle='--')
            ax.fill_between(years, lower, upper, color=colors['baseline'], alpha=0.1)
        
        # Plot effectiveness scenarios
        for scenario_name in ['very_low', 'low', 'baseline', 'high', 'very_high']:
            if scenario_name in results:
                mean = results[scenario_name][res]['mean'][si:] * 100
                lower = results[scenario_name][res]['lower'][si:] * 100
                upper = results[scenario_name][res]['upper'][si:] * 100
                
                # Rename 'baseline' to 'baseline_int' for intervention scenarios
                label_key = 'baseline_int' if scenario_name == 'baseline' else scenario_name
                color_key = 'baseline_int' if scenario_name == 'baseline' else scenario_name
                
                ax.plot(years, mean, label=scenario_labels[label_key], 
                       color=colors[color_key], linewidth=lw)
                ax.fill_between(years, lower, upper, color=colors[color_key], alpha=0.1)
        
        # Add vertical line for intervention start
        ax.axvline(x=2026, color='k', ls='--', linewidth=1.5)
        if i == 0:
            ax.text(2025.5, ax.get_ylim()[1] * 0.9, 'Start of\nintervention', 
                   ha='right', va='top', fontsize=10, color='#4d4d4d')
        
        ax.set_title(labels[i])
        ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if i in [0, 3]:
            ax.set_ylabel('Prevalence (%)')
        if i >= 3:
            ax.set_xlabel('Year')
    
    # Add legend
    axes[5].axis('off')
    axes[5].legend(*axes[0].get_legend_handles_labels(), fontsize=12, frameon=False, loc='center')
    
    pl.suptitle('Sensitivity Analysis: Intervention Effectiveness', fontsize=22, y=1.02)
    sc.figlayout()
    sc.savefig(plotfolder+'sensitivity_effectiveness_timeseries.png', dpi=150)




def plot_anemia_results(results):
    """Plot results from anemia probability sensitivity analysis"""
    
    # Get time vector
    t = np.arange(2000, 2033)
    si = np.where(t >= 2020)[0][0]
    years = t[si:]
    
    set_font(20)
    
    # Extract scenario names from the results keys
    scenario_names = []
    for key in results.keys():
        scenario_name = key.replace('_baseline', '').replace('_intervention', '')
        if scenario_name not in scenario_names:
            scenario_names.append(scenario_name)
    
    # Focus on anemia outcome - limit to first 7 scenarios for display
    scenarios_to_plot = scenario_names[:7]
    
    fig, axes = pl.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    scenario_labels = {
        'very_low': f'Very low (2%, 4%)',
        'low_base_low_hmb': f'Low base/low HMB (5%, 10%)',
        'low_base_high_hmb': f'Low base/high HMB (5%, 40%)',
        'baseline': f'Baseline (1%, 4.3%)',
        'moderate': f'Moderate (10%, 25%)',
        'high_base_low_hmb': f'High base/low HMB (20%, 30%)',
        'high_base_high_hmb': f'High base/high HMB (20%, 50%)',
        'very_high_base': f'Very high base (30%, 45%)',
        'very_high_hmb': f'Very high HMB (15%, 60%)',
    }
    
    # Plot each scenario comparison
    for i, scenario in enumerate(scenarios_to_plot):
        ax = axes[i]
        
        # Plot baseline (no intervention)
        baseline_key = f"{scenario}_baseline"
        if baseline_key in results and 'anemic' in results[baseline_key]:
            mean_base = results[baseline_key]['anemic']['mean'][si:] * 100
            lower_base = results[baseline_key]['anemic']['lower'][si:] * 100
            upper_base = results[baseline_key]['anemic']['upper'][si:] * 100
            
            ax.plot(years, mean_base, label='No intervention', 
                   color='#6c757d', linewidth=2.5, linestyle='--')
            ax.fill_between(years, lower_base, upper_base, color='#6c757d', alpha=0.2)
        
        # Plot with intervention
        int_key = f"{scenario}_intervention"
        if int_key in results and 'anemic' in results[int_key]:
            mean_int = results[int_key]['anemic']['mean'][si:] * 100
            lower_int = results[int_key]['anemic']['lower'][si:] * 100
            upper_int = results[int_key]['anemic']['upper'][si:] * 100
            
            ax.plot(years, mean_int, label='With intervention', 
                   color='#ff6500', linewidth=2.5)
            ax.fill_between(years, lower_int, upper_int, color='#ff6500', alpha=0.2)
        
        # Add vertical line for intervention start
        ax.axvline(x=2026, color='k', ls='--', linewidth=1.5)
        
        # Get probabilities from results if available
        if baseline_key in results:
            prob_base = results[baseline_key].get('prob_base', 0) * 100
            prob_hmb = results[baseline_key].get('prob_hmb', 0) * 100
            title = f"{scenario_labels.get(scenario, scenario)}\n(base: {prob_base:.0f}%, HMB: {prob_hmb:.0f}%)"
        else:
            title = scenario_labels.get(scenario, scenario)
        
        ax.set_title(title, fontsize=11)
        ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if i % 4 == 0:
            ax.set_ylabel('Anemia Prevalence (%)')
        if i >= 4:
            ax.set_xlabel('Year')
        
        if i == 0:
            ax.legend(fontsize=10, frameon=False)
    
    # Remove last empty subplot
    axes[7].axis('off')
    
    pl.suptitle('Sensitivity Analysis: Anemia Probability Parameters', fontsize=22, y=1.02)
    sc.figlayout()
    sc.savefig('figures/sensitivity_anemia_timeseries.png', dpi=150)
    
    # Also create a summary plot showing all outcomes for selected scenarios
    fig2, axes2 = pl.subplots(2, 3, figsize=(18, 10))
    axes2 = axes2.ravel()
    
    res_to_plot = ['hiud', 'hmb', 'poor_mh', 'anemic', 'pain']
    labels = ['hIUD Usage', 'HMB', 'Poor MH', 'Anemic', 'Pain']
    
    # Select key scenarios to show
    key_scenarios = []
    for name in ['baseline', 'moderate', 'very_high_hmb']:
        if name in scenario_names:
            key_scenarios.append(name)
    
    # If not enough key scenarios, use first three available
    if len(key_scenarios) < 3:
        key_scenarios = scenario_names[:min(3, len(scenario_names))]
    
    colors = {
        'baseline': '#3c6e71',
        'moderate': '#ffa500',
        'very_high_hmb': '#8b008b',
    }
    # Add colors for any other scenarios that might be used
    default_colors = ['#ff6500', '#008080', '#800080']
    for i, scenario in enumerate(key_scenarios):
        if scenario not in colors:
            colors[scenario] = default_colors[i % len(default_colors)]
    
    for i, res in enumerate(res_to_plot):
        ax = axes2[i]
        
        for scenario in key_scenarios:
            # Plot with intervention
            int_key = f"{scenario}_intervention"
            if int_key in results and res in results[int_key]:
                mean = results[int_key][res]['mean'][si:] * 100
                lower = results[int_key][res]['lower'][si:] * 100
                upper = results[int_key][res]['upper'][si:] * 100
                
                # Get the probability values for the label
                prob_base = results[int_key].get('prob_base', 0) * 100
                prob_hmb = results[int_key].get('prob_hmb', 0) * 100
                label = f"{scenario_labels.get(scenario, scenario)}"
                
                ax.plot(years, mean, label=label, 
                       color=colors[scenario], linewidth=2.0)
                ax.fill_between(years, lower, upper, color=colors[scenario], alpha=0.15)
        
        # Add baseline without intervention (use first available baseline)
        if len(scenario_names) > 0:
            baseline_key = f"{scenario_names[0]}_baseline"
            if baseline_key in results and res in results[baseline_key]:
                mean_base = results[baseline_key][res]['mean'][si:] * 100
                ax.plot(years, mean_base, label='No intervention', 
                       color='#6c757d', linewidth=2.0, linestyle='--')
        
        # Add vertical line for intervention start
        ax.axvline(x=2026, color='k', ls='--', linewidth=1.5)
        if i == 0:
            ax.text(2025.5, ax.get_ylim()[1] * 0.9, 'Start of\nintervention', 
                   ha='right', va='top', fontsize=10, color='#4d4d4d')
        
        ax.set_title(labels[i])
        ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if i in [0, 3]:
            ax.set_ylabel('Prevalence (%)')
        if i >= 3:
            ax.set_xlabel('Year')
    
    # Add legend
    axes2[5].axis('off')
    axes2[5].legend(*axes2[0].get_legend_handles_labels(), fontsize=12, frameon=False, loc='center')
    
    pl.suptitle('Key Anemia Scenarios: All Outcomes', fontsize=22, y=1.02)
    sc.figlayout()
    sc.savefig(plotfolder+'sensitivity_anemia_all_outcomes.png', dpi=150)









if __name__ == '__main__':
    
    # Run all three sensitivity analyses
    print("\nStarting sensitivity analyses...")
    
    # 1. Acceptance probabilities
    acceptance_results = run_acceptance_sensitivity()
    
    # 2. Intervention effectiveness
    effectiveness_results = run_effectiveness_sensitivity()
    
    # 3. Anemia probability parameters
    anemia_results = run_anemia_sensitivity()























