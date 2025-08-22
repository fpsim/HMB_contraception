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
import seaborn as sns
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

    # # Modify individual fecundity and exposure parameters
    # # These adjust each woman's probability of conception and exposure to pregnancy.
    # pars['fecundity_var_low'] = .8
    # pars['fecundity_var_high'] = 3.25
    # pars['exposure_factor'] = 3.5

    # # Last free parameter, postpartum sexual activity correction or 'birth spacing preference'.
    # # Pulls values from {location}/data/birth_spacing_pref.csv by default
    # # Set all to 1 to reset.
    # pars['spacing_pref']['preference'][:3] =  1  # Spacing of 0-6 months
    # pars['spacing_pref']['preference'][3:6] = 1  # Spacing of 9-15 months
    # pars['spacing_pref']['preference'][6:9] = 1  # Spacing of 18-24 months
    # pars['spacing_pref']['preference'][9:] =  1  # Spacing of 27-36 months

    # Adjust contraceptive choice parameters
    pars['prob_use_year'] = 2020,         # Time trend intercept
    pars['prob_use_trend_par'] = 0.03,    # Time trend parameter
    pars['force_choose'] = False,         # Whether to force non-users to choose a method ('False' by default)
    # Weights assigned to dictate preferences between methods:
    method_weights = dict(
        pill=.7,
        iud=.5,
        inj=2,
        cond=.2,
        btl=1,
        wdraw=1,
        impl=1.2,
        othtrad=1,
        othmod=1,
    )
    pars['method_weights'] = np.array([*method_weights.values()])  # Weights for the methods in method_list in methods.py (excluding 'none', so starting with 'pill' and ending in 'othmod').

    return pars


def make_sim(pars=None):
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


if __name__ == '__main__':
    # Run the simulation
    sim = make_sim()
    sim.run()
    # sim = sc.loadobj('kenya.sim')

    # Set options for plotting
    # plt.plot_cpr(sim)
    plt.plot_calib(sim)
    plot_by_age(sim)

