
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 16:15:35 2025

@author: kirstinol
"""




import numpy as np
import pandas as pd
import pylab as pl
import sciris as sc
import os
import itertools
import gc
# viz
import seaborn as sns
# starsim
import starsim as ss
# fpsim
import fpsim as fp
from fpsim import plotting as plt
# hmb
from menstruation import Menstruation
from education import Education
from interventions import hiud_hmb, txa, pill_hmb, hmb_package, nsaid


# updates:
    # - 100k agents
    # - 100 sim iters


# set the output directories
plotfolder = 'figures_extended/'
outfolder = 'results_extended/'

plotfolder_stochastic = 'figures_stochastic_extended/'
outfolder_stochastic = 'results_stochastic_extended/'

for ff in [plotfolder, outfolder, plotfolder_stochastic, outfolder_stochastic]:
    if not os.path.exists(ff):
        os.makedirs(ff)

# Settings
country = 'kenya'
plt.Config.set_figs_directory(plotfolder)
plt.Config.do_save = True
plt.Config.do_show = False
plt.Config.show_rmse = False





def make_pars():
    pars = fp.make_fp_pars()  # For default pars
    pars.update_location('kenya')

    # Modify individual fecundity and exposure parameters
    # These adjust each woman's probability of conception and exposure to pregnancy.
    pars['exposure_factor'] = 2

    # Adjust contraceptive choice parameters
    pars['prob_use_intercept'] = -1  # Intercept for the probability of using contraception

    # Weights assigned to dictate preferences between methods:
    method_weights = dict(
        pill=.7,
        iud=.5,
        inj=2.2,
        cond=1.5,
        btl=1,
        wdraw=1,
        impl=3.5,
        othtrad=.15,
        othmod=.5,
    )
    pars['method_weights'] = np.array([*method_weights.values()])  # Weights for the methods in method_list in methods.py (excluding 'none', so starting with 'pill' and ending in 'othmod').

    return pars



def make_sim(pars=None, stop=2021):
    if pars is None:
        pars = make_pars()

    # Create modules
    mens = Menstruation()
    objective_data = pd.read_csv("data/kenya_objective.csv")
    attainment_data = pd.read_csv("data/kenya_initialization.csv")
    edu = Education(objective_data=objective_data, attainment_data=attainment_data)

    # Run the sim
    sim = fp.Sim(
        start=2000,
        stop=stop,
        n_agents=10000,
        location='kenya',
        pars=pars,
        analyzers=[fp.cpr_by_age(), fp.method_mix_by_age()],
        education_module=edu,
        connectors=[mens],
        verbose=0.1,
    )

    return sim



def plot_by_age(sim, do_save=True, figs_directory=plotfolder):

    fig, ax = pl.subplots()
    age_bins = [18, 20, 25, 35, 50]
    colors = sc.vectocolor(age_bins)
    cind = 0

    res = sim.analyzers.cpr_by_age.results
    for alabel, ares in res.items():
        if alabel not in ['timevec', 'total']:
            ax.plot(res.timevec, ares, label=alabel, color=colors[cind])
            cind += 1
    ax.legend(loc='best', frameon=False)
    ax.set_ylim([0, 1])
    ax.set_ylabel('CPR')
    ax.set_title('CPR')
    if do_save: sc.savefig(f'{figs_directory}cpr_by_age.png')

    return



def set_font(size=None, font='Arial'):
    sc.fonts(add='Helvetica', use='Helvetica')
    sc.options(font=font, fontsize=size)
    return



def plot_stochastic_results(stats, years, si, colors, scenarios_to_plot=None, 
                            res_to_plot=None, labels=None, label_map=None,
                            fixed_scale=False, filename=None, 
                            plotfolder=None):
    """
    Plot stochastic simulation results with uncertainty bands.
    
    Parameters
    ----------
    stats : dict
        Dictionary containing statistics for each scenario and result type.
        Structure: stats[scenario][result_type]['mean'/'lower'/'upper']
    years : array
        Array of years for x-axis (already sliced from si)
    si : int
        Start index for slicing results
    colors : dict
        Dictionary mapping scenario names to colors
    scenarios_to_plot : list, optional
        List of scenario names to include. If None, uses all scenarios in colors.
    res_to_plot : list, optional
        List of result types to plot. Default: ['hiud','pill', 'hmb', 'poor_mh', 'anemic', 'pain']
    labels : list, optional
        Labels for each result type. Default: ['hIUD Usage','pill Usage', 'HMB ', 'Poor MH', 'Anemic', 'Pain']
    label_map : dict, optional
        Dictionary mapping scenario names to display labels
    fixed_scale : bool
        If True, scale all y-axes to 0-100. If False, use variable scales.
    filename : str, optional
        Filename for saving. If None, auto-generates based on fixed_scale.
    plotfolder : str
        Directory to save the figure.
    
    Returns
    -------
    fig, axes : matplotlib figure and axes objects
    """
    
    # Set defaults
    if plotfolder is None:
        plotfolder = 'figures_stochastic/'
    if res_to_plot is None:
        res_to_plot = ['hiud', 'pill', 'hmb', 'poor_mh', 'anemic', 'pain']
    if labels is None:
        labels = ['hIUD Usage', 'pill Usage', 'HMB ', 'Poor MH', 'Anemic', 'Pain']
    if scenarios_to_plot is None:
        scenarios_to_plot = list(colors.keys())
    if label_map is None:
        label_map = {
            'baseline': 'Baseline',
            'hiud20': 'hIUD 20% coverage',
            'hiud40': 'hIUD 40% coverage',
            'txa20': 'TXA 20% coverage',
            'pill20': 'pill 20% coverage',
            'nsaid20': 'NSAID 20% coverage',
            'p20': 'Package 20% coverage',
            'p40': 'Package 40% coverage',
            'p60': 'Package 60% coverage'
        }
    
    set_font(20)
    
    fig, axes = pl.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    
    lw = 2.5  # line width
    
    for i, res in enumerate(res_to_plot):
        ax = axes[i]
        
        # Plot each scenario with uncertainty bands
        for scenario in scenarios_to_plot:
            if scenario not in stats:
                continue
            
            mean = stats[scenario][res]['mean'][si:] * 100
            lower = stats[scenario][res]['lower'][si:] * 100
            upper = stats[scenario][res]['upper'][si:] * 100
            
            # Plot mean line
            ax.plot(years, mean, label=label_map.get(scenario, scenario), 
                   color=colors[scenario], linewidth=lw)
            # Plot uncertainty band (95% CI)
            ax.fill_between(years, lower, upper, color=colors[scenario], alpha=0.2)
        
        # Add vertical line for intervention start
        ax.axvline(x=2026, color='k', ls='--', linewidth=1.5)
        
        # Add text label for start of intervention - positioning depends on scale type
        if i == 0:  # Add text label only to first panel
            if fixed_scale:
                # Fixed position for 0-100 scale
                label_height = 20
            else:
                # Dynamic position based on current y-axis limits
                ax.autoscale()
                label_height = ax.get_ylim()[1] * 0.9
            
            ax.text(2025.5, label_height, 'Start of\nintervention', 
                   ha='right', va='top', fontsize=10, color='#4d4d4d')
        
        ax.set_title(labels[i])
        
        # Set y-axis limits
        if fixed_scale:
            ax.set_ylim(bottom=0, top=100)
        else:
            ax.set_ylim(bottom=0)
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if i in [0, 3]:
            ax.set_ylabel('Prevalence (%)')
        
        if i >= 3:
            ax.set_xlabel('Year')
    
    sc.figlayout(fig=fig, tight=False)
    pl.subplots_adjust(right=0.75, hspace=0.35)  # Make room on the right
    
    # Add the legend
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, 
               loc='center left', 
               bbox_to_anchor=(0.75, 0.25),
               fontsize=14, 
               frameon=False)
    
    # Save figure
    if filename is None:
        scale_suffix = '_y-axis-scaled-0-100' if fixed_scale else ''
        filename = f'hmb_stochastic_results{scale_suffix}.png'
    
    # Construct full path
    full_path = os.path.join(plotfolder, filename)
    print(f"Saving figure to: {full_path}")  # Debug print
    
    # Use pl.savefig directly instead of sc.savefig
    fig.savefig(full_path, dpi=300, bbox_inches='tight')
    print("Figure saved successfully!")  # Debug print
    
    return fig, axes


def plot_parameter_sweep_heatmaps_seaborn(all_results, baseline_key='baseline',
                                           prob_offer_values=[0.25, 0.5, 0.75],
                                           prob_accept_values=[0.25, 0.5, 0.75],
                                           intervention_start_year=2026,
                                           res_to_plot=None, labels=None,
                                           plotfolder=None, filename=None):
    

    
    if res_to_plot is None:
        res_to_plot = ['hiud', 'pill', 'hmb', 'poor_mh', 'anemic', 'pain']
    if labels is None:
        labels = ['hIUD Usage', 'Pill Usage', 'HMB', 'Poor MH', 'Anemia', 'Pain']
    if plotfolder is None:
        plotfolder = 'figures/'
    if filename is None:
        filename = 'parameter_sweep_heatmaps_seaborn.png'
    
    years = np.arange(2000, 2033)
    intervention_idx = np.where(years >= intervention_start_year)[0][0]
    
    # Calculate changes
    change_matrices = {}
    baseline_post_avg = {}
    
    for res in res_to_plot:
        baseline_mean = all_results[baseline_key][res]['mean']
        baseline_post_avg[res] = np.mean(baseline_mean[intervention_idx:])
    
    for res in res_to_plot:
        matrix = np.zeros((len(prob_accept_values), len(prob_offer_values)))
        
        for i, prob_offer in enumerate(prob_offer_values):
            for j, prob_accept in enumerate(prob_accept_values):
                scenario_name = f'scenario_offer-{prob_offer*100}_accept-{prob_accept*100}'
                
                if scenario_name in all_results:
                    scenario_mean = all_results[scenario_name][res]['mean']
                    scenario_post_avg = np.mean(scenario_mean[intervention_idx:])
                    
                    if baseline_post_avg[res] != 0:
                        pct_change = ((scenario_post_avg - baseline_post_avg[res]) / 
                                      baseline_post_avg[res]) * 100
                    else:
                        pct_change = 0
                    
                    matrix[j, i] = pct_change  # Note: j,i for correct orientation
        
        change_matrices[res] = matrix
    
    # Create plot
    fig, axes = pl.subplots(2, 3, figsize=(14, 10))
    axes = axes.ravel()
    
    # Custom diverging palette
    cmap = sns.diverging_palette(220, 20, as_cmap=True)  # Blue to Orange
    
    # Find global limits
    all_values = np.concatenate([m.flatten() for m in change_matrices.values()])
    vmax = all_values.max()
    vmin = all_values.min()
    
    for idx, (res, label) in enumerate(zip(res_to_plot, labels)):
        ax = axes[idx]
        
        # Create DataFrame for seaborn
        df = pd.DataFrame(
            change_matrices[res],
            index=[f'{int(p*100)}%' for p in prob_accept_values],
            columns=[f'{int(p*100)}%' for p in prob_offer_values]
        )
        
        # Plot heatmap
        sns.heatmap(df, 
                    ax=ax,
                    cmap=cmap,
                    center=0,
                    vmin=vmin,
                    vmax=vmax,
                    annot=True,
                    fmt='.1f',
                    annot_kws={'size': 11, 'weight': 'bold'},
                    cbar=idx == 2,  # Only show colorbar for one panel
                    cbar_kws={'label': '% Change'} if idx == 2 else {},
                    linewidths=0.5,
                    linecolor='white')
        
        ax.set_title(label, fontsize=14, fontweight='bold')
        
        if idx >= 3:
            ax.set_xlabel('Probability of Offer', fontsize=12)
        else:
            ax.set_xlabel('')
            
        if idx in [0, 3]:
            ax.set_ylabel('Probability of Accept', fontsize=12)
        else:
            ax.set_ylabel('')
    
    fig.suptitle('Impact of Intervention by Coverage and Acceptability\n(% change from baseline)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    pl.tight_layout()
    
    full_path = os.path.join(plotfolder, filename)
    fig.savefig(full_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {full_path}")
    
    pl.show()
    
    return fig, axes, change_matrices





if __name__ == '__main__':

    to_run = [
         #'calib', # calibration
         #'plot_hmb',  # plot the calibration results
         'run_stochastic', # main analysis
         'run_coverage_sweep', # sensitivity analysis 
    ]
    do_run = True


    if 'calib' in to_run:
        if do_run:
            # Create simulation with parameters
            sim = make_sim()
            sim.run()
            sc.saveobj('results/kenya_calib.sim', sim)
        else:
            sim = sc.loadobj('results/kenya_calib.sim')

        # Set options for plotting
        set_font(20)
        plt.plot_calib(sim, single_fig=True)
        plot_by_age(sim)



    if 'plot_hmb' in to_run:
        sim = sc.loadobj(outfolder+'kenya_calib.sim')

        #import numpy as np
        t = sim.results.menstruation.timevec[::12]  # Take every 12th time point for speed
        years = np.array([y.year for y in t])
        si = sc.findfirst(years, 2010)
        years = years[si:]
        set_font(20)
        fig, axes = pl.subplots(2, 2, figsize=(15, 9))
        axes = axes.ravel()

        res_to_plot = ['hmb', 'poor_mh', 'anemic', 'pain'] # , 'hyst']
        labels = ['HMB prevalence\n(% of menstruating girls/women)', 'Poor MH prevalence\n(% of menstruating girls/women)', 'Anemia prevalence\n(% of menstruating girls/women)', 'Menstrual pain prevalence\n(% of menstruating girls/women)']  #, 'Hysterectomy']

        for i, res in enumerate(res_to_plot):
            ax = axes[i]
            r0 = sim.results.menstruation[f'{res}_prev']
            y0 = r0[::12][si:]
            ax.plot(years, y0*100)
            # ax.set_xticks(years[::2])
            ax.set_title(labels[i])
            ax.set_ylim(bottom=0, top=100)

        pl.legend(fontsize=16, frameon=False, loc='upper left')
        sc.figlayout()
        sc.savefig(plotfolder+'hmb_results.png', dpi=150)



    if 'run_stochastic' in to_run:
        # run the scenarios for multiple random seeds
        n_seeds = 50
        
        colors = {
            'hiud20':  '#372248',    # dark purple
            'hiud40':  '#3C427C',    # blue-ish purple
            'txa20': '#3c6e71',      # teal
            'pill20': '#8fbc8f',     # sage green
            'nsaid20': '#FF69B4',    # pink
            'p20': '#ffa500',        # orange
            'p40': '#ff8c00',        # darker orange
            'p60': '#ff6500',        # darkest orange
            'baseline': '#6c757d'   # dark gray
        }
                
        if do_run:
            for seed in np.arange(n_seeds):
                # --- set up simulations
                # baseline - no intervention
                s_base = make_sim(stop=2032)
                s_base['pars']['rand_seed'] = seed
                
                # only hIUD - 20% coverage
                s_hiud20 = make_sim(stop=2032)
                s_hiud20['pars']['interventions'] = [hiud_hmb(prob_offer=0.2, prob_accept=0.5)]
                s_hiud20['pars']['rand_seed'] = seed
                
                # only hIUD - 40% coverage
                s_hiud40 = make_sim(stop=2032)
                s_hiud40['pars']['interventions'] = [hiud_hmb(prob_offer=0.4, prob_accept=0.5)]
                s_hiud40['pars']['rand_seed'] = seed
                
                # only NSAID - 20% coverage
                s_nsaid20 = make_sim(stop=2032)
                s_nsaid20['pars']['interventions'] = [nsaid(prob_offer=0.2, prob_accept=0.5)]
                s_nsaid20['pars']['rand_seed'] = seed

                # only txa
                s_txa20 = make_sim(stop=2032)
                s_txa20['pars']['interventions'] = [txa(prob_offer=0.2, prob_accept=0.5)]
                s_txa20['pars']['rand_seed'] = seed
                
                # only pill
                s_pill20 = make_sim(stop=2032)
                s_pill20['pars']['interventions'] = [pill_hmb(prob_offer=0.2, prob_accept=0.5)]
                s_pill20['pars']['rand_seed'] = seed
                
                # full package - 20 % of eligible pop
                s_p20 = make_sim(stop=2032)
                s_p20['pars']['interventions'] = [hmb_package(prob_offer=0.2, 
                                                 prob_accept_nsaid=0.5,
                                                 prob_accept_txa=0.5, 
                                                 prob_accept_pill=0.5,
                                                 prob_accept_hiud=0.5)]
                s_p20['pars']['rand_seed'] = seed
    
                # full package - 40 % of eligible pop
                s_p40 = make_sim(stop=2032)
                s_p40['pars']['interventions'] = [hmb_package(prob_offer=0.4, 
                                                 prob_accept_nsaid=0.5,
                                                 prob_accept_txa=0.5, 
                                                 prob_accept_pill=0.5,
                                                 prob_accept_hiud=0.5)]
                s_p40['pars']['rand_seed'] = seed
                
                # full package - 60 % of eligible pop
                s_p60 = make_sim(stop=2032)
                s_p60['pars']['interventions'] = [hmb_package(prob_offer=0.6, 
                                                 prob_accept_nsaid=0.5,
                                                 prob_accept_txa=0.5, 
                                                 prob_accept_pill=0.5,
                                                 prob_accept_hiud=0.5)]
                s_p60['pars']['rand_seed'] = seed
                
                
                # --- run the simulations
                m = ss.parallel([s_base, s_hiud20, s_hiud40, s_txa20, s_pill20, s_nsaid20,
                                 s_p20, s_p40, s_p60], 
                                parallel=True)
                # replace sims with run versions
                s_base, s_hiud20, s_hiud40, s_txa20, s_pill20, s_nsaid20, s_p20, s_p40, s_p60 = m.sims[:]  
                
                # --- save results
                sc.saveobj(outfolder_stochastic+f'kenya_package_base_seed{seed}.sim', s_base)
                sc.saveobj(outfolder_stochastic+f'kenya_package_hiud-20_seed{seed}.sim', s_hiud20)
                sc.saveobj(outfolder_stochastic+f'kenya_package_hiud-40_seed{seed}.sim', s_hiud40)
                sc.saveobj(outfolder_stochastic+f'kenya_package_txa-20_seed{seed}.sim', s_txa20)
                sc.saveobj(outfolder_stochastic+f'kenya_package_pill-20_seed{seed}.sim', s_pill20)
                sc.saveobj(outfolder_stochastic+f'kenya_package_nsaid-20_seed{seed}.sim', s_nsaid20)
                sc.saveobj(outfolder_stochastic+f'kenya_package_package20_seed{seed}.sim', s_p20)
                sc.saveobj(outfolder_stochastic+f'kenya_package_package40_seed{seed}.sim', s_p40)
                sc.saveobj(outfolder_stochastic+f'kenya_package_package60_seed{seed}.sim', s_p60)
    
    
        
        # --- aggregate results
        
        # Initialize dictionaries to store results for each scenario
        scenarios = ['baseline', 'hiud20', 'hiud40', 'txa20', 'pill20','nsaid20' , 'p20', 'p40', 'p60']
        res_to_plot = ['hiud','pill', 'hmb', 'poor_mh', 'anemic', 'pain']
        labels = ['hIUD Usage','pill Usage', 'HMB ', 'Poor MH', 'Anemic', 'Pain']    
        
        # Dictionary to store all runs
        all_results = {scenario: {res: [] for res in res_to_plot} for scenario in scenarios}
    
        # load individual files
        for seed in range(n_seeds):
            s_base = sc.loadobj(outfolder_stochastic+f'kenya_package_base_seed{seed}.sim')
            s_hiud20 = sc.loadobj(outfolder_stochastic+f'kenya_package_hiud-20_seed{seed}.sim')
            s_hiud40 = sc.loadobj(outfolder_stochastic+f'kenya_package_hiud-40_seed{seed}.sim')
            s_txa20 = sc.loadobj(outfolder_stochastic+f'kenya_package_txa-20_seed{seed}.sim')
            s_pill20 = sc.loadobj(outfolder_stochastic+f'kenya_package_pill-20_seed{seed}.sim')
            s_nsaid20 = sc.loadobj(outfolder_stochastic+f'kenya_package_nsaid-20_seed{seed}.sim')
            s_p20 = sc.loadobj(outfolder_stochastic+f'kenya_package_package20_seed{seed}.sim')
            s_p40 = sc.loadobj(outfolder_stochastic+f'kenya_package_package40_seed{seed}.sim')
            s_p60 = sc.loadobj(outfolder_stochastic+f'kenya_package_package60_seed{seed}.sim')
            
            sims = {'baseline': s_base, 'hiud20': s_hiud20, 'hiud40': s_hiud40, 
                    'txa20': s_txa20, 'pill20': s_pill20, 'nsaid20': s_nsaid20, 
                    'p20': s_p20, 'p40': s_p40, 'p60': s_p60}
            
            for scenario, sim in sims.items():
                for res in res_to_plot:
                    result = sim.results.menstruation[f'{res}_prev'][::12]
                    all_results[scenario][res].append(result)
        
        # Convert to arrays and calculate statistics
        stats = {scenario: {res: {} for res in res_to_plot} for scenario in scenarios}
        
        for scenario in scenarios:
            for res in res_to_plot:
                arr = np.array(all_results[scenario][res])  # Shape: (n_seeds, time_points)
                stats[scenario][res]['mean'] = np.mean(arr, axis=0)
                stats[scenario][res]['median'] = np.median(arr, axis=0)
                stats[scenario][res]['lower'] = np.percentile(arr, 2.5, axis=0)
                stats[scenario][res]['upper'] = np.percentile(arr, 97.5, axis=0)
                stats[scenario][res]['q25'] = np.percentile(arr, 25, axis=0)
                stats[scenario][res]['q75'] = np.percentile(arr, 75, axis=0)
        
        # Get time vector
        t = s_base.results.menstruation.timevec[::12]
        years = np.array([y.year for y in t])
        si = sc.findfirst(years, 2020)
        years = years[si:]
        
        set_font(20)
        
        # ---- PLOT: All scenarios, variable scale
        plot_stochastic_results(
            stats=stats, years=years, si=si, colors=colors,
            fixed_scale=False,
            plotfolder=plotfolder_stochastic,
            filename='hmb_scenario-package_stochastic_results.png'
        )
        
        # ---- PLOT: All scenarios, fixed 0-100 scale
        plot_stochastic_results(
            stats=stats, years=years, si=si, colors=colors,
            fixed_scale=True,
            plotfolder=plotfolder_stochastic,
            filename='hmb_scenario-package_stochastic_results_y-axis-scaled-0-100.png'
        )
        
        # ---- PLOT: Subset of scenarios, variable scale
        # define the subset of scenarios
        scenarios_subset = ['baseline', 'hiud20', 'hiud40', 'p40']
        # make the plots
        plot_stochastic_results(
            stats=stats, years=years, si=si, colors=colors,
            scenarios_to_plot=scenarios_subset,
            fixed_scale=False,
            plotfolder=plotfolder_stochastic,
            filename='hmb_package_stochastic_results_subset-scenarios.png'
        )
        
        # ---- PLOT: Subset of scenarios, fixed 0-100 scale
        plot_stochastic_results(
            stats=stats, years=years, si=si, colors=colors,
            scenarios_to_plot=scenarios_subset,
            fixed_scale=True,
            plotfolder=plotfolder_stochastic,
            filename='hmb_package_stochastic_results_subset-scenarios_y-axis-scaled-0-100.png'
        )
        
        
        
    
    if 'run_coverage_sweep' in to_run:
                
        n_seeds = 20
        prob_offer_values = [0.25, 0.5, 0.75]
        prob_accept_values = [0.25, 0.5, 0.75]
        res_keys = ['hiud', 'pill', 'hmb', 'poor_mh', 'anemic', 'pain']
        
        def compute_and_save_scenario(scenario_name, sim_list):
            """Run sims, compute stats, save, and free memory."""
            msim = ss.MultiSim(sim_list)
            msim.run(n_cpus=4)
            
            scenario_results = {res: [] for res in res_keys}
            for sim in msim.sims:
                for res in res_keys:
                    scenario_results[res].append(sim.results.menstruation[f'{res}_prev'][::12])
            
            stats = {}
            for res in res_keys:
                arr = np.array(scenario_results[res])
                stats[res] = {
                    'mean': np.mean(arr, axis=0),
                    'lower': np.percentile(arr, 2.5, axis=0),
                    'upper': np.percentile(arr, 97.5, axis=0)
                }
            
            # Save intermediate result
            sc.saveobj(outfolder + f'uptake-sweep_{scenario_name}.obj', stats)
            
            # Free memory
            del msim, sim_list, scenario_results
            gc.collect()
            
            return stats
        
        all_results = {}
        
        # Baseline
        print("Running baseline...")
        baseline_sims = [make_sim(stop=2032) for _ in range(n_seeds)]
        for i, sim in enumerate(baseline_sims):
            sim['pars']['rand_seed'] = i
        all_results['baseline'] = compute_and_save_scenario('baseline', baseline_sims)
        
        # Intervention scenarios
        param_combinations = list(itertools.product(prob_offer_values, prob_accept_values))
        
        for prob_offer, prob_accept in param_combinations:
            scenario_name = f'scenario_offer-{prob_offer*100}_accept-{prob_accept*100}'
            print(f"Running {scenario_name}...")
            
            scenario_sims = []
            for seed in range(n_seeds):
                s_int = make_sim(stop=2032)
                s_int['pars']['rand_seed'] = seed
                s_int['pars']['interventions'] = [
                    hmb_package(
                        prob_offer=prob_offer,
                        prob_accept_hiud=prob_accept,
                        prob_accept_txa=prob_accept,
                        prob_accept_pill=prob_accept,
                        prob_accept_nsaid=prob_accept
                    )
                ]
                scenario_sims.append(s_int)
            
            all_results[scenario_name] = compute_and_save_scenario(scenario_name, scenario_sims)
        
        # Save combined results
        sc.saveobj(outfolder_stochastic + 'uptake-sweep_results-stats.obj', all_results)
        print("Complete!")
        
        
        # load
        all_results = sc.loadobj(outfolder + 'uptake-sweep_results-stats.obj')
        
        # Make heatmap
        fig, axes, change_matrices = plot_parameter_sweep_heatmaps_seaborn(
            all_results=all_results,
            baseline_key='baseline',
            prob_offer_values=prob_offer_values,
            prob_accept_values=prob_accept_values,
            intervention_start_year=2026,
            res_to_plot=[#'hiud', 'pill', 
                         'hmb', 'poor_mh', 'anemic', 'pain'],
            labels=[#'hIUD Usage', 'Pill Usage', 
                    'HMB', 'Poor MH', 'Anemia', 'Pain'],
            plotfolder=plotfolder_stochastic,
            filename='parameter_sweep_heatmaps.png'
        )
        
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
      
        
        
        
        
        