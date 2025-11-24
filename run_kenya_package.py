# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 10:26:40 2025

@author: kirstinol
"""



import numpy as np
import pandas as pd
import pylab as pl
import sciris as sc
import os
# starsim
import starsim as ss
# fpsim
import fpsim as fp
from fpsim import plotting as plt
# hmb
from menstruation import Menstruation
from education import Education
from interventions import hiud_hmb, txa, pill_hmb, hmb_package


# set the output directories
plotfolder = 'figures/'
outfolder = 'results/'

plotfolder_stochastic = 'figures_stochastic/'
outfolder_stochastic = 'results_stochastic/'

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
        n_agents=1000,
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

    # fig, ax = pl.subplots()
    # df = pd.DataFrame(sim.analyzers.method_mix_by_age.results)
    # df['method'] = sim.connectors.contraception.methods.keys()
    # df_plot = df.melt(id_vars='method')
    # sns.barplot(x=df_plot['method'], y=df_plot['value'], ax=ax, hue=df_plot['variable'], palette="viridis")
    # pl.show()
    # if do_save: sc.savefig(f'{figs_directory}/method_mix_by_age.png')

    return


def set_font(size=None, font='Arial'):
    sc.fonts(add='Helvetica', use='Helvetica')
    sc.options(font=font, fontsize=size)
    return




if __name__ == '__main__':

    to_run = [
         #'calib',
         #'plot_hmb',  # plot the HMB results
         'run_scenario',  # run a scenario with interventions
         'run_stochastic', # run multiple iterations of scenarios
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


    if 'run_scenario' in to_run:
        # run the scenarios
        if do_run:
            # --- set up simulations
            # baseline - no intervention
            s_base = make_sim(stop=2032)
            # only hIUD
            s_hiud30 = make_sim(stop=2032)
            s_hiud30['pars']['interventions'] = [hiud_hmb(prob_offer=0.3, prob_accept=0.5)]
            # only txa
            s_txa30 = make_sim(stop=2032)
            s_txa30['pars']['interventions'] = [txa(prob_offer=0.3, prob_accept=0.5)]
            # only pill
            s_pill30 = make_sim(stop=2032)
            s_pill30['pars']['interventions'] = [pill_hmb(prob_offer=0.3, prob_accept=0.5)]
            # full package - 20 % of eligible pop
            s_p20 = make_sim(stop=2032)
            s_p20['pars']['interventions'] = [hmb_package(prob_offer=0.2, 
                                             prob_accept_hiud=0.5, 
                                             prob_accept_txa=0.5, 
                                             prob_accept_pill=0.5)]
            # full package - 40 % of eligible pop
            s_p40 = make_sim(stop=2032)
            s_p40['pars']['interventions'] = [hmb_package(prob_offer=0.4, 
                                             prob_accept_hiud=0.5, 
                                             prob_accept_txa=0.5, 
                                             prob_accept_pill=0.5)]
            # full package - 60 % of eligible pop
            s_p60 = make_sim(stop=2032)
            s_p60['pars']['interventions'] = [hmb_package(prob_offer=0.6, 
                                             prob_accept_hiud=0.5, 
                                             prob_accept_txa=0.5, 
                                             prob_accept_pill=0.5)]
            
            # --- run the simulations
            m = ss.parallel([s_base, s_hiud30, s_txa30, s_pill30, 
                             s_p20, s_p40, s_p60], 
                            parallel=False, verbose=0)
            # replace sims with run versions
            s_base, s_hiud30, s_txa30, s_pill30, s_p20, s_p40, s_p60 = m.sims[:]  
            
            # --- save results
            sc.saveobj(outfolder+'kenya_package_base.sim', s_base)
            sc.saveobj(outfolder+'kenya_package_hiud30.sim', s_hiud30)
            sc.saveobj(outfolder+'kenya_package_txa30.sim', s_txa30)
            sc.saveobj(outfolder+'kenya_package_pill30.sim', s_pill30)
            sc.saveobj(outfolder+'kenya_package_package20.sim', s_p20)
            sc.saveobj(outfolder+'kenya_package_package40.sim', s_p40)
            sc.saveobj(outfolder+'kenya_package_package60.sim', s_p60)

        else:
            s_base = sc.loadobj(outfolder+'kenya_package_base.sim')
            s_hiud30 = sc.loadobj(outfolder+'kenya_package_hiud30.sim')
            s_txa30 = sc.loadobj(outfolder+'kenya_package_txa30.sim')
            s_pill30 = sc.loadobj(outfolder+'kenya_package_pill30.sim')
            s_p20 = sc.loadobj(outfolder+'kenya_package_package20.sim')
            s_p40 = sc.loadobj(outfolder+'kenya_package_package40.sim')
            s_p60 = sc.loadobj(outfolder+'kenya_package_package60.sim')



        ##### ---- Plot ----
        t = s_base.results.menstruation.timevec[::12]  # Take every 12th time point for speed
        years = np.array([y.year for y in t])
        si = sc.findfirst(years, 2020)
        years = years[si:]
        set_font(20)
        


        # ---- PLOT: subpanels have their own y-axis scale
        
        fig, axes = pl.subplots(2, 3, figsize=(15, 9))
        axes = axes.ravel()
        
        res_to_plot = ['hiud','pill', 'hmb', 'poor_mh', 'anemic', 'pain']
        labels = ['hIUD Usage','pill Usage', 'HMB ', 'Poor MH', 'Anemic', 'Pain']
        
        # Define colors
        colors = {
            'baseline': '#6c757d',  # dark gray
            'hiud30':  '#372248',    # dark purple
            'txa30': '#3c6e71',     # teal
            'pill30': '#8fbc8f',    # sage green
            'p20': '#ffa500',       # orange
            'p40': '#ff8c00',       # darker orange
            'p60': '#ff6500'        # darkest orange
        }
        
        lw = 2.5  # line width
        
        for i, res in enumerate(res_to_plot):
            ax = axes[i]
            r0 = s_base.results.menstruation[f'{res}_prev']
            y0 = r0[::12][si:]
            yhiud30 = s_hiud30.results.menstruation[f'{res}_prev'][::12][si:]
            ytxa30 = s_txa30.results.menstruation[f'{res}_prev'][::12][si:]
            ypill30 = s_pill30.results.menstruation[f'{res}_prev'][::12][si:]
            yp20 = s_p20.results.menstruation[f'{res}_prev'][::12][si:]
            yp40 = s_p40.results.menstruation[f'{res}_prev'][::12][si:]
            yp60 = s_p60.results.menstruation[f'{res}_prev'][::12][si:]
            
            ax.plot(years, yhiud30*100, label='hIUD 30% uptake', color=colors['hiud30'], linewidth=lw)
            ax.plot(years, ytxa30*100, label='TXA 30% uptake', color=colors['txa30'], linewidth=lw)
            ax.plot(years, ypill30*100, label='pill 30% uptake', color=colors['pill30'], linewidth=lw)
            ax.plot(years, yp20*100, label='package 20% uptake', color=colors['p20'], linewidth=lw)
            ax.plot(years, yp40*100, label='package 40% uptake', color=colors['p40'], linewidth=lw)
            ax.plot(years, yp60*100, label='package 60% uptake', color=colors['p60'], linewidth=lw)
            ax.plot(years, y0*100, label='Baseline', color=colors['baseline'], linewidth=lw)

            # Add vertical line with label
            ax.axvline(x=2026, color='k', ls='--', linewidth=1.5)
            # add a label for the start of the intervention
            # label height is slightly above max value before start of intervention
            label_height = 1.5 *  y0[years < 2026].max() * 100 
            if i == 0:  # Add text label only to first panel
                ax.text(2025.5, label_height, 'Start of\nintervention', ha='right', va='top', fontsize=10, color='#4d4d4d')
            
            ax.set_title(labels[i])
            ax.set_ylim(bottom=0)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if i in [0, 3]:
                ax.set_ylabel('Prevalence (%)')
        
        sc.figlayout(fig=fig, tight=False)
        pl.subplots_adjust(right=0.85, hspace=0.35)  # Make room on the right
        
        # THEN add the legend
        handles, labels_legend = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_legend, 
                   loc='center left', 
                   bbox_to_anchor=(0.87, 0.25),
                   fontsize=14, 
                   frameon=False)

        sc.savefig(plotfolder+'hmb_scenario-package_results_3-interventions.png', dpi=150)
        
        
        
        
        # ---- PLOT: all axes scaled to 0-100
        fig, axes = pl.subplots(2, 3, figsize=(15, 9))
        axes = axes.ravel()
        
        res_to_plot = ['hiud', 'hmb', 'poor_mh', 'anemic', 'pain']
        labels = ['hIUD Usage', 'HMB ', 'Poor MH', 'Anemic', 'Pain']
        
        # Define colors
        colors = {
            'baseline': '#6c757d',  # dark gray
            'hiud30':  '#372248',    # dark purple
            'txa30': '#3c6e71',     # teal
            'pill30': '#8fbc8f',    # sage green
            'p20': '#ffa500',       # orange
            'p40': '#ff8c00',       # darker orange
            'p60': '#ff6500'        # darkest orange
        }
        
        lw = 2.5  # line width
        
        for i, res in enumerate(res_to_plot):
            ax = axes[i]
            r0 = s_base.results.menstruation[f'{res}_prev']
            y0 = r0[::12][si:]
            yhiud30 = s_hiud30.results.menstruation[f'{res}_prev'][::12][si:]
            ytxa30 = s_txa30.results.menstruation[f'{res}_prev'][::12][si:]
            ypill30 = s_pill30.results.menstruation[f'{res}_prev'][::12][si:]
            yp20 = s_p20.results.menstruation[f'{res}_prev'][::12][si:]
            yp40 = s_p40.results.menstruation[f'{res}_prev'][::12][si:]
            yp60 = s_p60.results.menstruation[f'{res}_prev'][::12][si:]
            
            ax.plot(years, yhiud30*100, label='hIUD 30% uptake', color=colors['hiud30'], linewidth=lw)
            ax.plot(years, ytxa30*100, label='TXA 30% uptake', color=colors['txa30'], linewidth=lw)
            ax.plot(years, ypill30*100, label='pill 30% uptake', color=colors['pill30'], linewidth=lw)
            ax.plot(years, yp20*100, label='package 20% uptake', color=colors['p20'], linewidth=lw)
            ax.plot(years, yp40*100, label='package 40% uptake', color=colors['p40'], linewidth=lw)
            ax.plot(years, yp60*100, label='package 60% uptake', color=colors['p60'], linewidth=lw)
            ax.plot(years, y0*100, label='Baseline', color=colors['baseline'], linewidth=lw)

            # Add vertical line with label
            ax.axvline(x=2026, color='k', ls='--', linewidth=1.5)
            # add a label for the start of the intervention
            # label height is slightly above max value before start of intervention
            label_height = 1.5 *  y0[years < 2026].max() * 100 
            if i == 0:  # Add text label only to first panel
                ax.text(2025.5, 20, 'Start of\nintervention', ha='right', va='top', fontsize=10, color='#4d4d4d')
            
            ax.set_title(labels[i])
            ax.set_ylim(bottom=0, top=100)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if i in [0, 3]:
                ax.set_ylabel('Prevalence (%)')
        
        sc.figlayout(fig=fig, tight=False)
        pl.subplots_adjust(right=0.85, hspace=0.35)  # Make room on the right
        
        # THEN add the legend
        handles, labels_legend = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_legend, 
                   loc='center left', 
                   bbox_to_anchor=(0.87, 0.25),
                   fontsize=14, 
                   frameon=False)
        
        sc.savefig(plotfolder+'hmb_scenario-package_results_3-interventions_y-axis-scaled-0-100.png', dpi=150)





    if 'run_stochastic' in to_run:
        # run the scenarios for multiple random seeds
        n_seeds = 50 
        
        if do_run:
            for seed in np.arange(n_seeds):
                # --- set up simulations
                # baseline - no intervention
                s_base = make_sim(stop=2032)
                s_base['pars']['rand_seed'] = seed
                
                # only hIUD
                s_hiud20 = make_sim(stop=2032)
                s_hiud20['pars']['interventions'] = [hiud_hmb(prob_offer=0.2, prob_accept=0.5)]
                s_hiud20['pars']['rand_seed'] = seed
                
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
                                                 prob_accept_hiud=0.5, 
                                                 prob_accept_txa=0.5, 
                                                 prob_accept_pill=0.5)]
                s_p20['pars']['rand_seed'] = seed
    
                # full package - 40 % of eligible pop
                s_p40 = make_sim(stop=2032)
                s_p40['pars']['interventions'] = [hmb_package(prob_offer=0.4, 
                                                 prob_accept_hiud=0.5, 
                                                 prob_accept_txa=0.5, 
                                                 prob_accept_pill=0.5)]
                s_p40['pars']['rand_seed'] = seed
                # full package - 60 % of eligible pop
                s_p60 = make_sim(stop=2032)
                s_p60['pars']['interventions'] = [hmb_package(prob_offer=0.6, 
                                                 prob_accept_hiud=0.5, 
                                                 prob_accept_txa=0.5, 
                                                 prob_accept_pill=0.5)]
                s_p60['pars']['rand_seed'] = seed
                
                # --- run the simulations
                m = ss.parallel([s_base, s_hiud20, s_txa20, s_pill20,
                                 s_p20, s_p40, s_p60], 
                                parallel=False)
                # replace sims with run versions
                s_base, s_hiud20, s_txa20, s_pill20, s_p20, s_p40, s_p60 = m.sims[:]  
                
                # --- save results
                sc.saveobj(outfolder_stochastic+f'kenya_package_base_seed{seed}.sim', s_base)
                sc.saveobj(outfolder_stochastic+f'kenya_package_hiud-20_seed{seed}.sim', s_hiud20)
                sc.saveobj(outfolder_stochastic+f'kenya_package_txa-20_seed{seed}.sim', s_txa20)
                sc.saveobj(outfolder_stochastic+f'kenya_package_pill-20_seed{seed}.sim', s_pill20)
                sc.saveobj(outfolder_stochastic+f'kenya_package_package20_seed{seed}.sim', s_p20)
                sc.saveobj(outfolder_stochastic+f'kenya_package_package40_seed{seed}.sim', s_p40)
                sc.saveobj(outfolder_stochastic+f'kenya_package_package60_seed{seed}.sim', s_p60)
    
           
    
        
        # --- aggregate results
        
        # Initialize dictionaries to store results for each scenario
        scenarios = ['base', 'hiud20', 'txa20', 'pill20', 'p20', 'p40', 'p60']
        res_to_plot = ['hiud', 'hmb', 'poor_mh', 'anemic', 'pain']
        
        # Dictionary to store all runs
        all_results = {scenario: {res: [] for res in res_to_plot} for scenario in scenarios}
    
        # load individual files
        for seed in range(n_seeds):
            s_base = sc.loadobj(outfolder_stochastic+f'kenya_package_base_seed{seed}.sim')
            s_hiud20 = sc.loadobj(outfolder_stochastic+f'kenya_package_hiud-20_seed{seed}.sim')
            s_txa20 = sc.loadobj(outfolder_stochastic+f'kenya_package_txa-20_seed{seed}.sim')
            s_pill20 = sc.loadobj(outfolder_stochastic+f'kenya_package_pill-20_seed{seed}.sim')
            s_p20 = sc.loadobj(outfolder_stochastic+f'kenya_package_package20_seed{seed}.sim')
            s_p40 = sc.loadobj(outfolder_stochastic+f'kenya_package_package40_seed{seed}.sim')
            s_p60 = sc.loadobj(outfolder_stochastic+f'kenya_package_package60_seed{seed}.sim')
            
            sims = {'base': s_base, 'hiud20': s_hiud20, 'txa20': s_txa20, 'pill20': s_pill20, 
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
        
        # ---- PLOT: Stochastic results with uncertainty bands
        fig, axes = pl.subplots(2, 3, figsize=(15, 9))
        axes = axes.ravel()
        
        labels = ['hIUD Usage', 'HMB', 'Poor MH', 'Anemic', 'Pain']
        
        # Define colors
        colors = {
            'p20': '#ffa500',    # orange
            'hiud20':  '#372248',    # dark purple
            'txa20': '#3c6e71',     # teal
            'pill20': '#8fbc8f',    # sage green
            'p40': '#ff8c00',    # darker orange
            'p60': '#ff6500',    # darkest orange
            'base': '#6c757d',   # dark gray
        }
        
        lw = 2.5  # line width
        
        for i, res in enumerate(res_to_plot):
            ax = axes[i]
            
            # Plot each scenario with uncertainty bands
            for scenario, color in colors.items():
                mean = stats[scenario][res]['mean'][si:] * 100
                lower = stats[scenario][res]['lower'][si:] * 100
                upper = stats[scenario][res]['upper'][si:] * 100
                
                label_map = {
                    'base': 'Baseline',
                    'hiud20':  'hIUD 20% uptake',  
                    'txa20': 'TXA 20% uptake',   
                    'pill20': 'pill 20% uptake', 
                    'p20': 'Package 20% uptake',
                    'p40': 'Package 40% uptake',
                    'p60': 'Package 60% uptake'
                }
                
                # Plot mean line
                ax.plot(years, mean, label=label_map[scenario], color=color, linewidth=lw)
                # Plot uncertainty band (95% CI)
                ax.fill_between(years, lower, upper, color=color, alpha=0.2)
            
            # Add vertical line with label
            ax.axvline(x=2026, color='k', ls='--', linewidth=1.5)
            if i == 0:  # Add text label only to first panel
                ax.text(2025.5, ax.get_ylim()[1] * 0.9, 'Start of\nintervention', 
                       ha='right', va='top', fontsize=10, color='#4d4d4d')
            
            ax.set_title(labels[i])
            ax.set_ylim(bottom=0)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if i in [0, 3]:
                ax.set_ylabel('Prevalence (%)')
            
            if i >= 3:
                ax.set_xlabel('Year')
        
        sc.figlayout(fig=fig, tight=False)
        pl.subplots_adjust(right=0.85, hspace=0.35)  # Make room on the right
        
        # THEN add the legend
        handles, labels_legend = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_legend, 
                   loc='center left', 
                   bbox_to_anchor=(0.87, 0.25),
                   fontsize=14, 
                   frameon=False)
       
        sc.savefig(plotfolder_stochastic+'hmb_scenario-package_stochastic_results.png', dpi=150)
        
        
        
        
        # ---- PLOT: Scaled 0-100 version
        fig, axes = pl.subplots(2, 3, figsize=(15, 9))
        axes = axes.ravel()
        
        for i, res in enumerate(res_to_plot):
            ax = axes[i]
            
            # Plot each scenario with uncertainty bands
            for scenario, color in colors.items():
                mean = stats[scenario][res]['mean'][si:] * 100
                lower = stats[scenario][res]['lower'][si:] * 100
                upper = stats[scenario][res]['upper'][si:] * 100
                
                label_map = {
                    'base': 'Baseline',
                    'hiud20':  'hIUD 20% uptake',  
                    'txa20': 'TXA 20% uptake',   
                    'pill20': 'pill 20% uptake', 
                    'p20': 'Package 20% uptake',
                    'p40': 'Package 40% uptake',
                    'p60': 'Package 60% uptake'
                }
                
                # Plot mean line
                ax.plot(years, mean, label=label_map[scenario], color=color, linewidth=lw)
                # Plot uncertainty band (95% CI)
                ax.fill_between(years, lower, upper, color=color, alpha=0.2)
            
            # Add vertical line with label
            ax.axvline(x=2026, color='k', ls='--', linewidth=1.5)
            if i == 0:  # Add text label only to first panel
                ax.text(2025.5, 20, 'Start of\nintervention', 
                       ha='right', va='top', fontsize=10, color='#4d4d4d')
            
            ax.set_title(labels[i])
            ax.set_ylim(bottom=0, top=100)
            
            # Remove top and right spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            if i in [0, 3]:
                ax.set_ylabel('Prevalence (%)')
            
            if i >= 3:
                ax.set_xlabel('Year')
        
        sc.figlayout(fig=fig, tight=False)
        pl.subplots_adjust(right=0.85, hspace=0.35)  # Make room on the right
        
        # THEN add the legend
        handles, labels_legend = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_legend, 
                   loc='center left', 
                   bbox_to_anchor=(0.87, 0.25),
                   fontsize=14, 
                   frameon=False)
        
        sc.savefig(plotfolder_stochastic+'hmb_scenario-package_stochastic_results_y-axis-scaled-0-100.png', dpi=150)
        
        
        
        # Save aggregated statistics
        sc.saveobj(outfolder_stochastic+'aggregated_stats.obj', stats)








