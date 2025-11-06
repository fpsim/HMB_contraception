"""
Script to create a model of Kenya, modify some parameters, run the model and generate plots showing the
discrepancies between the model and data.
"""

import numpy as np
import fpsim as fp
from fpsim import plotting as plt
import pandas as pd
from menstruation import Menstruation
from education import Education
import starsim as ss
import pylab as pl
import sciris as sc

# Settings
country = 'kenya'
plt.Config.set_figs_directory('figures/')
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
    objective_data = pd.read_csv(f"data/kenya_objective.csv")
    attainment_data = pd.read_csv(f"data/kenya_initialization.csv")
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
    )

    return sim


def plot_by_age(sim, do_save=True, figs_directory='figures'):

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
    if do_save: sc.savefig(f'{figs_directory}/cpr_by_age.png')

    # fig, ax = pl.subplots()
    # df = pd.DataFrame(sim.analyzers.method_mix_by_age.results)
    # df['method'] = sim.connectors.contraception.methods.keys()
    # df_plot = df.melt(id_vars='method')
    # sns.barplot(x=df_plot['method'], y=df_plot['value'], ax=ax, hue=df_plot['variable'], palette="viridis")
    # pl.show()
    # if do_save: sc.savefig(f'{figs_directory}/method_mix_by_age.png')

    return


def set_font(size=None, font='Libertinus Sans'):
    sc.fonts(add=sc.thisdir(aspath=True) / 'assets' / 'LibertinusSans-Regular.otf')
    sc.options(font=font, fontsize=size)
    return




if __name__ == '__main__':

    to_run = [
         #'calib',
         #'plot_hmb',  # Plot the HMB results
         'run_scenario',  # Run a scenario with interventions
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
        sim = sc.loadobj('results/kenya_calib.sim')

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
        sc.savefig('figures/hmb_results.png', dpi=150)


    if 'run_scenario' in to_run:
        # run the scenarios
        if do_run:
            # baseline - no intervention
            s_base = make_sim(stop=2032)
            # full package: low prob for each intervention
            s_i10 = make_sim(stop=2032)
            # full package: high prob for each intervention
            s_i30 = make_sim(stop=2032)
            # only hIUD
            s_hiud30 = make_sim(stop=2032)
            # only txa
            s_txa30 = make_sim(stop=2032)
            # only pill
            s_pill30 = make_sim(stop=2032)
            
            from interventions import contra_hmb, txa, pill_hmb
            s_i10['pars']['interventions'] = [contra_hmb(prob=0.1), txa(prob=0.1), pill_hmb(prob=0.1)]
            s_i30['pars']['interventions'] = [contra_hmb(prob=0.3), txa(prob=0.3), pill_hmb(prob=0.3)]
            s_hiud30['pars']['interventions'] = [contra_hmb(prob=0.3), txa(prob=0.), pill_hmb(prob=0.)]
            s_txa30['pars']['interventions'] = [contra_hmb(prob=0.), txa(prob=0.3), pill_hmb(prob=0.)]
            s_pill30['pars']['interventions'] = [contra_hmb(prob=0.0), txa(prob=0.), pill_hmb(prob=0.3)]

            m = ss.parallel([s_base, s_i10, s_i30, s_hiud30, s_txa30, s_pill30], 
                            parallel=False)
            s_base, s_i10, s_i30, s_hiud30, s_txa30, s_pill30 = m.sims[:]  # Replace with run versions
            # Save results
            sc.saveobj('results/kenya_base.sim', s_base)
            sc.saveobj('results/kenya_i10.sim', s_i10)
            sc.saveobj('results/kenya_i30.sim', s_i30)
            sc.saveobj('results/kenya_hiud30.sim', s_hiud30)
            sc.saveobj('results/kenya_txa30.sim', s_txa30)
            sc.saveobj('results/kenya_pill30.sim', s_pill30)

        else:
            s_base = sc.loadobj('results/kenya_base.sim')
            s_i10 = sc.loadobj('results/kenya_i10.sim')
            s_i30 = sc.loadobj('results/kenya_i30.sim')
            s_hiud30 = sc.loadobj('results/kenya_hiud30.sim')
            s_txa30 = sc.loadobj('results/kenya_txa30.sim')
            s_pill30 = sc.loadobj('results/kenya_pill30.sim')

        # Plot
        #import numpy as np
        t = s_base.results.menstruation.timevec[::12]  # Take every 12th time point for speed
        years = np.array([y.year for y in t])
        si = sc.findfirst(years, 2020)
        years = years[si:]
        set_font(20)
        
        # --- make plot
        fig, axes = pl.subplots(2, 3, figsize=(15, 9))
        axes = axes.ravel()

        res_to_plot = ['hiud', 'hmb', 'poor_mh', 'anemic', 'pain']
        labels = ['hIUD Usage', 'HMB ', 'Poor MH', 'Anemic', 'Pain']

        for i, res in enumerate(res_to_plot):
            ax = axes[i]
            r0 = s_base.results.menstruation[f'{res}_prev']
            y0 = r0[::12][si:]
            y10 = s_i10.results.menstruation[f'{res}_prev'][::12][si:]
            y30 = s_i30.results.menstruation[f'{res}_prev'][::12][si:]
            yhiud30 = s_hiud30.results.menstruation[f'{res}_prev'][::12][si:]
            ytxa30 = s_txa30.results.menstruation[f'{res}_prev'][::12][si:]
            ypill30 = s_pill30.results.menstruation[f'{res}_prev'][::12][si:]
            ax.plot(years, y0*100, label='Baseline')
            ax.plot(years, y10*100, label='hIUD+TXA+pill, 10% uptake each')
            ax.plot(years, y30*100, label='hIUD+TXA+pill 30% uptake each')
            ax.plot(years, yhiud30*100, label='hIUD 30% uptake')
            ax.plot(years, ytxa30*100, label='TXA 30% uptake')
            ax.plot(years, ypill30*100, label='pill 30% uptake')
            # ax.set_xticks(years[::2])
            ax.axvline(x=2026, color='k', ls='--')
            ax.set_title(labels[i])
            ax.set_ylim(bottom=0)
            if i in [0, 3]:
                ax.set_ylabel('Prevalence (%)')
        # Make an empty final axis
        #ax = axes[5]
        axes[5].axis('off')
        axes[0].legend(fontsize=16, frameon=False, loc='upper left')
        sc.figlayout()
        sc.savefig('figures/hmb_scenario_results_3-interventions.png', dpi=150)
        
        
        # --- same plot but with y-axes 0-100
        fig, axes = pl.subplots(2, 3, figsize=(15, 9))
        axes = axes.ravel()

        res_to_plot = ['hiud', 'hmb', 'poor_mh', 'anemic', 'pain']
        labels = ['hIUD Usage', 'HMB ', 'Poor MH', 'Anemic', 'Pain']

        for i, res in enumerate(res_to_plot):
            ax = axes[i]
            r0 = s_base.results.menstruation[f'{res}_prev']
            y0 = r0[::12][si:]
            y10 = s_i10.results.menstruation[f'{res}_prev'][::12][si:]
            y30 = s_i30.results.menstruation[f'{res}_prev'][::12][si:]
            yhiud30 = s_hiud30.results.menstruation[f'{res}_prev'][::12][si:]
            ytxa30 = s_txa30.results.menstruation[f'{res}_prev'][::12][si:]
            ypill30 = s_pill30.results.menstruation[f'{res}_prev'][::12][si:]
            ax.plot(years, y0*100, label='Baseline')
            ax.plot(years, y10*100, label='hIUD+TXA+pill, 10% uptake each')
            ax.plot(years, y30*100, label='hIUD+TXA+pill 30% uptake each')
            ax.plot(years, yhiud30*100, label='hIUD 30% uptake')
            ax.plot(years, ytxa30*100, label='TXA 30% uptake')
            ax.plot(years, ypill30*100, label='pill 30% uptake')
            # ax.set_xticks(years[::2])
            ax.axvline(x=2026, color='k', ls='--')
            ax.set_title(labels[i])
            ax.set_ylim(bottom=0, top=100)
            if i in [0, 3]:
                ax.set_ylabel('Prevalence (%)')

        # Make an empty final axis
        #ax = axes[5]
        axes[5].axis('off')

        axes[0].legend(fontsize=16, frameon=False, loc='upper left')
        sc.figlayout()
        sc.savefig('figures/hmb_scenario_results_3-interventions_y-axis-scaled-0-100.png', dpi=150)

