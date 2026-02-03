
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


# --- convert monthly time series to yearly ---
def annualize_monthly(arr, how="mean"):
    arr = np.asarray(arr)
    if len(arr) < 12:
        # Not enough months to form a year; return empty array for safety
        return np.array([])
    n_years = len(arr) // 12
    arr = arr[:12 * n_years].reshape(n_years, 12)
    if how == "mean":
        return arr.mean(axis=1)
    elif how == "eoy":
        return arr[:, -1]
    else:
        raise ValueError(how)


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
    births = ss.Births()
    edu = Education(objective_data=objective_data, attainment_data=attainment_data)

    # Run the sim
    sim = fp.Sim(
        start=2000,
        stop=stop,
        n_agents=1000,
        location='kenya',
        pars=pars,
        demographics=[births],
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



def plot_stochastic_results(stats, years_annual, years_monthly, si_annual, si_monthly, colors, scenarios_to_plot=None, 
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
    years_annual : array
        Array of years for x-axis for annual data 
    years_monthly : array
        Array of years for x-axis for monthly data 
    si_annual : int
        Start index for slicing annual results
    si_monthly : int
        Start index for slicing monthly results
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
            'hiud25': 'hIUD 25% coverage',
            'hiud50': 'hIUD 50% coverage',
            'p25': 'Package 25% coverage',
            'p50': 'Package 50% coverage',
            'p75': 'Package 75% coverage'
        }
    
    set_font(10)
    
    fig, axes = pl.subplots(3, 3, figsize=(15, 9))
    axes = axes.ravel()
    
    lw = 2.5  # line width
    
    for i, res in enumerate(res_to_plot):
        ax = axes[i]
        
        if res == 'prop_disrupted':
            years = years_monthly
            si = si_monthly
        else:
            years = years_annual
            si = si_annual

        # Plot each scenario with uncertainty bands
        for scenario in scenarios_to_plot:
            if scenario not in stats:
                continue
            
            # Ensure stats series align with the provided years array
            ylen = len(years)
            
            if res in ("n_disruptions", "n_anemia"):
                # compute "averted" relative to baseline for this year-range
                # baseline series (last ylen elements)
                base_mean = stats['baseline'][res]['mean'][-ylen:]
                base_lower = stats['baseline'][res]['lower'][-ylen:]
                base_upper = stats['baseline'][res]['upper'][-ylen:]

                scen_mean = stats[scenario][res]['mean'][-ylen:]
                scen_lower = stats[scenario][res]['lower'][-ylen:]
                scen_upper = stats[scenario][res]['upper'][-ylen:]

                # averted = baseline - scenario
                mean = base_mean - scen_mean
                # For CI on difference, use conservative bounds:
                lower = base_lower - scen_upper
                upper = base_upper - scen_lower

               # keep units as counts (do NOT multiply by 100)
            else:        
                
                mean = stats[scenario][res]['mean'][-ylen:] * 100
                lower = stats[scenario][res]['lower'][-ylen:] * 100
                upper = stats[scenario][res]['upper'][-ylen:] * 100

            # quick debug check (optional)
            if not (len(mean) == len(years) == len(lower) == len(upper)):
                print(f"DEBUG SHAPES mismatch for {scenario} {res}: years={len(years)}, mean={len(mean)}, lower={len(lower)}, upper={len(upper)}")
            
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
        
        if res == 'n_disruptions':
            ax.set_ylabel('Number of disruptions averted')   # counts
        elif res == 'prop_disrupted':
            ax.set_ylabel('% Disruption')                    # monthly percent
        elif res == 'n_anemia':
            ax.set_ylabel('Number of anemia cases averted')   # counts
        elif i in [0, 3]:
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
                    
                    if res in ("n_disruptions", "n_anemia"):
                        # Post-intervention total disruptions 
                        baseline_series = all_results[baseline_key][res]['mean']
                        baseline_post_total = sum(baseline_series[intervention_idx:])
                        scenario_post_total = sum(scenario_mean[intervention_idx:])
                        val  = baseline_post_total - scenario_post_total

                    else:
                        # Your existing post-intervention average approach
                        scenario_post_avg = np.mean(scenario_mean[intervention_idx:])
                        base = baseline_post_avg[res]

                        if base == 0:
                            # avoid divide-by-zero; set NaN or 0 depending on desired logic
                            val = np.nan
                        else:
                            val = ((scenario_post_avg - base) / base) * 100.0

                matrix[j, i] = val

                    
        
        change_matrices[res] = matrix
    
    # Create plot
    fig, axes = pl.subplots(3, 3, figsize=(18, 12))
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
        
        
        if res == 'n_disruptions':
            cb_label = '# disruptions averted'
            fmt = '.0f'
        elif res == 'n_anemia':
            cb_label = '# anemia cases averted'
            fmt = '.0f'
        else:
            cb_label = '% change'
            fmt = '.1f'

        
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
                    cbar=True,  # Only show colorbar for one panel
                    cbar_kws={'label': cb_label},
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
    
    fig.suptitle('Impact of Intervention by Coverage and Acceptability', 
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
        n_seeds = 2
        
        colors = {
            'hiud25':  '#372248',    # dark purple
            'hiud50':  '#3C427C',    # blue-ish purple
            'p25': '#ffa500',        # orange
            'p50': '#ff8c00',        # darker orange
            'p75': '#ff6500',        # darkest orange
            'baseline': '#6c757d'   # dark gray
        }
                
        if do_run:
            for seed in np.arange(n_seeds):
                # --- set up simulations
                # baseline - no intervention
                s_base = make_sim(stop=2032)
                s_base['pars']['rand_seed'] = seed
                
                # only hIUD - 25% coverage
                s_hiud25 = make_sim(stop=2032)
                s_hiud25['pars']['interventions'] = [hiud_hmb(prob_offer=0.25, prob_accept=0.5)]
                s_hiud25['pars']['rand_seed'] = seed
                
                # only hIUD - 50% coverage
                s_hiud50 = make_sim(stop=2032)
                s_hiud50['pars']['interventions'] = [hiud_hmb(prob_offer=0.5, prob_accept=0.5)]
                s_hiud50['pars']['rand_seed'] = seed
                
                # # only NSAID - 20% coverage
                # s_nsaid20 = make_sim(stop=2032)
                # s_nsaid20['pars']['interventions'] = [nsaid(prob_offer=0.2, prob_accept=0.5)]
                # s_nsaid20['pars']['rand_seed'] = seed

                # # only txa
                # s_txa20 = make_sim(stop=2032)
                # s_txa20['pars']['interventions'] = [txa(prob_offer=0.2, prob_accept=0.5)]
                # s_txa20['pars']['rand_seed'] = seed
                
                # # only pill
                # s_pill20 = make_sim(stop=2032)
                # s_pill20['pars']['interventions'] = [pill_hmb(prob_offer=0.2, prob_accept=0.5)]
                # s_pill20['pars']['rand_seed'] = seed
                
                # full package - 25 % of eligible pop
                s_p25 = make_sim(stop=2032)
                s_p25['pars']['interventions'] = [hmb_package(prob_offer=0.25, 
                                                 prob_accept_nsaid=0.5,
                                                 prob_accept_txa=0.5, 
                                                 prob_accept_pill=0.5,
                                                 prob_accept_hiud=0.5)]
                s_p25['pars']['rand_seed'] = seed
    
                # full package - 50 % of eligible pop
                s_p50 = make_sim(stop=2032)
                s_p50['pars']['interventions'] = [hmb_package(prob_offer=0.5, 
                                                 prob_accept_nsaid=0.5,
                                                 prob_accept_txa=0.5, 
                                                 prob_accept_pill=0.5,
                                                 prob_accept_hiud=0.5)]
                s_p50['pars']['rand_seed'] = seed
                
                # full package - 60 % of eligible pop
                s_p75 = make_sim(stop=2032)
                s_p75['pars']['interventions'] = [hmb_package(prob_offer=0.75, 
                                                 prob_accept_nsaid=0.5,
                                                 prob_accept_txa=0.5, 
                                                 prob_accept_pill=0.5,
                                                 prob_accept_hiud=0.5)]
                s_p75['pars']['rand_seed'] = seed
                
                
                # --- run the simulations
                m = ss.parallel([s_base, s_hiud25, s_hiud50, s_p25, s_p50, s_p75], 
                                parallel=True)
                # replace sims with run versions
                s_base, s_hiud25, s_hiud50, s_p25, s_p50, s_p75 = m.sims[:]  
                
                # --- save results
                sc.saveobj(outfolder_stochastic+f'kenya_package_base_seed{seed}.sim', s_base)
                sc.saveobj(outfolder_stochastic+f'kenya_package_hiud-25_seed{seed}.sim', s_hiud25)
                sc.saveobj(outfolder_stochastic+f'kenya_package_hiud-50_seed{seed}.sim', s_hiud50)
                # sc.saveobj(outfolder_stochastic+f'kenya_package_txa-20_seed{seed}.sim', s_txa20)
                # sc.saveobj(outfolder_stochastic+f'kenya_package_pill-20_seed{seed}.sim', s_pill20)
                # sc.saveobj(outfolder_stochastic+f'kenya_package_nsaid-20_seed{seed}.sim', s_nsaid20)
                sc.saveobj(outfolder_stochastic+f'kenya_package_package25_seed{seed}.sim', s_p25)
                sc.saveobj(outfolder_stochastic+f'kenya_package_package50_seed{seed}.sim', s_p50)
                sc.saveobj(outfolder_stochastic+f'kenya_package_package75_seed{seed}.sim', s_p75)
    
    
    

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

    # --- Save results to CSV ---
    print("Saving results to CSV...")

    # Save annual metrics together
    df_annual_data = {'year': years_annual}

    for res in res_to_plot:
        if res == 'prop_disrupted':
            continue  # monthly stored separately
        for scenario in scenarios:
            series_mean = stats[scenario][res]['mean'][-len(years_annual):]
            series_low  = stats[scenario][res]['lower'][-len(years_annual):]
            series_up   = stats[scenario][res]['upper'][-len(years_annual):]

            if res in ("n_disruptions", "n_anemia"):
                # counts: keep as-is (no *100)
                df_annual_data[f'{res}_{scenario}_mean'] = series_mean
                df_annual_data[f'{res}_{scenario}_lower'] = series_low
                df_annual_data[f'{res}_{scenario}_upper'] = series_up
            else:
                # proportions -> percent
                df_annual_data[f'{res}_{scenario}_mean'] = series_mean * 100
                df_annual_data[f'{res}_{scenario}_lower'] = series_low * 100
                df_annual_data[f'{res}_{scenario}_upper'] = series_up * 100

    df_annual = pd.DataFrame(df_annual_data)
    csv_filename_annual = os.path.join(outfolder_stochastic, 'results_annual_metrics.csv')
    df_annual.to_csv(csv_filename_annual, index=False)
    print(f"Saved {csv_filename_annual}")

    # Save monthly disruption data separately
    df_monthly_data = {'year': years_monthly}

    for scenario in scenarios:
        df_monthly_data[f'prop_disrupted_{scenario}_mean'] = stats[scenario]['prop_disrupted']['mean'][-len(years_monthly):] * 100
        df_monthly_data[f'prop_disrupted_{scenario}_lower'] = stats[scenario]['prop_disrupted']['lower'][-len(years_monthly):] * 100
        df_monthly_data[f'prop_disrupted_{scenario}_upper'] = stats[scenario]['prop_disrupted']['upper'][-len(years_monthly):] * 100

    df_monthly = pd.DataFrame(df_monthly_data)
    csv_filename_monthly = os.path.join(outfolder_stochastic, 'results_monthly_disruption.csv')
    df_monthly.to_csv(csv_filename_monthly, index=False)
    print(f"Saved {csv_filename_monthly}")

    print("All results saved to CSV!")    
        
        # --- aggregate results
        
        # Initialize dictionaries to store results for each scenario
        scenarios = ['baseline', 'hiud25', 'hiud50',  'p25', 'p50', 'p75']
        res_to_plot = ['hiud','pill', 'hmb', 'poor_mh', 'anemic','n_anemia', 'pain', 'prop_disrupted','n_disruptions']
        labels = ['hIUD Usage','pill Usage', 'HMB ', 'Poor MH', 'Anemic','Number of anemia cases averted', 'Pain', '% Disruption','Number of disruptions averted']    
        
        # Dictionary to store all runs
        all_results = {scenario: {res: [] for res in res_to_plot} for scenario in scenarios}
    
        # load individual files
        for seed in range(n_seeds):
            s_base = sc.loadobj(outfolder_stochastic+f'kenya_package_base_seed{seed}.sim')
            s_hiud25 = sc.loadobj(outfolder_stochastic+f'kenya_package_hiud-25_seed{seed}.sim')
            s_hiud50 = sc.loadobj(outfolder_stochastic+f'kenya_package_hiud-50_seed{seed}.sim')
            # s_txa20 = sc.loadobj(outfolder_stochastic+f'kenya_package_txa-20_seed{seed}.sim')
            # s_pill20 = sc.loadobj(outfolder_stochastic+f'kenya_package_pill-20_seed{seed}.sim')
            # s_nsaid20 = sc.loadobj(outfolder_stochastic+f'kenya_package_nsaid-20_seed{seed}.sim')
            s_p25 = sc.loadobj(outfolder_stochastic+f'kenya_package_package25_seed{seed}.sim')
            s_p50 = sc.loadobj(outfolder_stochastic+f'kenya_package_package50_seed{seed}.sim')
            s_p75 = sc.loadobj(outfolder_stochastic+f'kenya_package_package75_seed{seed}.sim')
            
            sims = {'baseline': s_base, 'hiud25': s_hiud25, 'hiud50': s_hiud50, 
                    'p25': s_p25, 'p50': s_p50, 'p75': s_p75}
            
            for scenario, sim in sims.items():
                for res in res_to_plot:
                    if res == 'prop_disrupted':
                        result = sim.results.edu[res]  # Keep monthly
                        # result = annualize_monthly(sim.results.edu[res], how="mean")   
                    elif res == 'n_disruptions':
                         result = annualize_monthly(sim.results.edu[res], how="eoy") 
                         
                    elif res == 'n_anemia':
                        result = annualize_monthly(sim.results.menstruation['n_anemia'], how="eoy")
                    else:
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
        # Get time vector - monthly for disruption, annual for others
        t_annual = s_base.results.menstruation.timevec[::12]
        years_annual = np.array([y.year for y in t_annual])
        si_annual = sc.findfirst(years_annual, 2020)
        years_annual = years_annual[si_annual:]

        # Monthly time vector for disruption
        t_monthly = s_base.results.edu.timevec
        years_monthly = np.array([y.year + y.month/12 for y in t_monthly])
        si_monthly = sc.findfirst(years_monthly >= 2020)
        years_monthly = years_monthly[si_monthly:]
        
        # t = s_base.results.menstruation.timevec[::12]
        # years = np.array([y.year for y in t])
        # si = sc.findfirst(years, 2020)
        # years = years[si:]
        
        import matplotlib.pyplot as plt

        plt.rcParams['font.family'] = 'Arial'   # or 'Calibri', 'Times New Roman'
        plt.rcParams['font.size'] = 10   
        set_font(10)
     
        # ---- PLOT: All scenarios, variable scale
        plot_stochastic_results(
            stats=stats, years_annual=years_annual, 
            years_monthly=years_monthly,
            si_annual=si_annual,
            si_monthly=si_monthly, colors=colors,
            fixed_scale=False,
            plotfolder=plotfolder_stochastic,
            res_to_plot=['hiud', 'pill', 'hmb', 'poor_mh', 'anemic', 'n_anemia','pain', 'prop_disrupted','n_disruptions'],  
            labels=['hIUD Usage', 'pill Usage', 'HMB', 'Poor MH', 'Anemic','Number of anemia cases averted',  'Pain', 'Disruption','Disruption'],  
            filename='hmb_scenario-package_stochastic_results.png'
        )
        
        # ---- PLOT: All scenarios, fixed 0-100 scale
        plot_stochastic_results(
            stats=stats, years_annual=years_annual, 
            years_monthly=years_monthly,
            si_annual=si_annual,
            si_monthly=si_monthly, colors=colors,
            fixed_scale=True,
            plotfolder=plotfolder_stochastic,
            res_to_plot=['hiud', 'pill', 'hmb', 'poor_mh', 'anemic','n_anemia', 'pain', 'prop_disrupted','n_disruptions'],  
            labels=['hIUD Usage', 'pill Usage', 'HMB', 'Poor MH', 'Anemic', 'Number of anemia cases averted', 'Pain', 'Disruption','Disruption'],  
            filename='hmb_scenario-package_stochastic_results_y-axis-scaled-0-100.png'
        )
        
        # ---- PLOT: Subset of scenarios, variable scale
        # define the subset of scenarios
        scenarios_subset = ['baseline', 'hiud25', 'hiud50', 'p50']
        # make the plots
        plot_stochastic_results(
            stats=stats, years_annual=years_annual, 
            years_monthly=years_monthly,
            si_annual=si_annual,
            si_monthly=si_monthly,colors=colors,
            scenarios_to_plot=scenarios_subset,
            fixed_scale=False,
            plotfolder=plotfolder_stochastic,
            res_to_plot=['hiud', 'pill', 'hmb', 'poor_mh', 'anemic', 'n_anemia','pain', 'prop_disrupted','n_disruptions'],  
            labels=['hIUD Usage', 'pill Usage', 'HMB', 'Poor MH', 'Anemic','Number of anemia cases averted',  'Pain', 'Disruption','Disruption'],    
            filename='hmb_package_stochastic_results_subset-scenarios.png'
        )
        
        # ---- PLOT: Subset of scenarios, fixed 0-100 scale
        plot_stochastic_results(
            stats=stats, years_annual=years_annual, 
            years_monthly=years_monthly,
            si_annual=si_annual,
            si_monthly=si_monthly, colors=colors,
            scenarios_to_plot=scenarios_subset,
            fixed_scale=True,
            plotfolder=plotfolder_stochastic,
            res_to_plot=['hiud', 'pill', 'hmb', 'poor_mh', 'anemic', 'n_anemia','pain', 'prop_disrupted','n_disruptions'],  
            labels=['hIUD Usage', 'pill Usage', 'HMB', 'Poor MH', 'Anemic','Number of anemia cases averted',  'Pain', 'Disruption','Disruption'],  
            filename='hmb_package_stochastic_results_subset-scenarios_y-axis-scaled-0-100.png'
        )
        

    
    if 'run_coverage_sweep' in to_run:
                
        n_seeds = 2
        prob_offer_values = [0.25, 0.5, 0.75]
        prob_accept_values = [0.25, 0.5, 0.75]
        res_keys = ['hiud', 'pill', 'hmb', 'poor_mh', 'anemic','n_anemia', 'pain', 'prop_disrupted', 'n_disruptions']
        
        def compute_and_save_scenario(scenario_name, sim_list):
            """Run sims, compute stats, save, and free memory."""
            msim = ss.MultiSim(sim_list)
            msim.run(n_cpus=4)
            
            scenario_results = {res: [] for res in res_keys}
            for sim in msim.sims:
                for res in res_keys:
                    if res == 'prop_disrupted':
                        scenario_results[res].append(annualize_monthly(sim.results.edu[res], how="mean"))
                    elif res == 'n_disruptions':
                        scenario_results[res].append(annualize_monthly(sim.results.edu[res], how="eoy"))
                    elif res == 'n_anemia':
                        scenario_results[res].append(annualize_monthly(sim.results.menstruation['n_anemia'], how="eoy"))
                    else:
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
        all_results = sc.loadobj(outfolder_stochastic + 'uptake-sweep_results-stats.obj')
        
        # Make heatmap
        fig, axes, change_matrices = plot_parameter_sweep_heatmaps_seaborn(
            all_results=all_results,
            baseline_key='baseline',
            prob_offer_values=prob_offer_values,
            prob_accept_values=prob_accept_values,
            intervention_start_year=2026,
            res_to_plot=[#'hiud', 'pill', 
                         'hmb', 'poor_mh', 'anemic', 'n_anemia','pain','prop_disrupted','n_disruptions'],
            labels=[#'hIUD Usage', 'Pill Usage', 
                    'HMB', 'Poor MH', 'Anemia', 'Total anemia cases averted (post-2026)', 'Pain', 'Disruption','Total disruptions averted (post-2026)'],
            plotfolder=plotfolder_stochastic,
            filename='parameter_sweep_heatmaps.png'
        )
        
            
            
            
            
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
      
        
        
        
        
        