"""
Test HMB module
"""

import pylab as pl
import starsim as ss
import fpsim as fp
import sciris as sc
import pandas as pd
from menstruation import Menstruation
from education import Education


class IUD(ss.Intervention):
    """
    Example intervention for IUD usage
    """
    def __init__(self, start=2027, new_val=0.1, **kwargs):
        super().__init__(**kwargs)
        self.start = ss.date(start)
        self.new_val = new_val
        return

    def step(self):
        if self.t.now() >= self.start:
            self.sim.diseases.menstruation.pars.p_iud.set(self.new_val)
        return


class contra_hmb(ss.Intervention):
    def __init__(self, pars=None, eligibility=None, **kwargs):
        super().__init__(name='contra_hmb', eligibility=eligibility)
        self.define_pars(
            year=2026,  # When to apply the intervention
            prob=ss.bernoulli(p=0.5),  # Proportion of HMB-prone non-users who will accept
        )
        self.update_pars(pars, **kwargs)
        if eligibility is None:
            self.eligibility = lambda sim: (
                    sim.people.menstruation.hmb_prone &
                    sim.people.menstruation.menstruating &
                    # ~sim.people.on_contra &
                    ~sim.people.pregnant)
        self.define_states(
            ss.State('intervention_applied', label="Received IUD through intervention"),
        )
        return

    @property
    def iud_idx(self):
        """ Get the index of the IUD method """
        return self.sim.connectors.contraception.get_method_by_label('IUDs').idx

    def step(self):
        sim = self.sim
        if sim.t.now() == self.pars.year:
            # Print message
            print(f'Changing IUDs!')

            # Get women who accept the intervention
            elig_uids = self.check_eligibility()
            accept_uids = self.pars.prob.filter(elig_uids)

            # Adjust their contraception
            sim.people.method[accept_uids] = self.iud_idx
            sim.people.on_contra[accept_uids] = True
            sim.people.ever_used_contra[accept_uids] = True
            method_dur = sim.connectors.contraception.set_dur_method(accept_uids)
            sim.people.ti_contra[accept_uids] = self.ti + method_dur

            self.intervention_applied[accept_uids] = True
        return


if __name__ == '__main__':

    # What to run
    to_run = [
        # 'run_single',  # Run a single simulation
        'run_scenario',  # Run a scenario with interventions
    ]
    sc.fonts(add=sc.thisdir(aspath=True) / 'assets' / 'LibertinusSans-Regular.otf')
    sc.options(font='Libertinus Sans', fontsize=24)

    # Creat modules
    mens = Menstruation()
    objective_data = pd.read_csv(f"data/kenya_objective.csv")
    attainment_data = pd.read_csv(f"data/kenya_initialization.csv")
    edu = Education(objective_data=objective_data, attainment_data=attainment_data)

    # Create sim
    pars = dict(
        n_agents=1000,
        start=2000,
        stop=2032,
        location='kenya',
    )

    if 'run_single' in to_run:
        sim = fp.Sim(education_module=edu, connectors=[mens], **pars)
        sim.run(verbose=1/12)

        # Plot education
        import pylab as pl
        t = sim.results.edu.timevec
        fig, axes = pl.subplots(2, 3, figsize=(20, 12))
        axes = axes.ravel()

        res_to_plot = ['mean_attainment', 'mean_objective', 'prop_completed', 'prop_in_school', 'prop_dropped']
        sc.options(fontsize=16)

        for i, res in enumerate(res_to_plot):
            ax = axes[i]
            r0 = sim.results.edu[res]
            ax.plot(t, r0)
            ax.set_title(res)

        all_props = [sim.results.edu.prop_in_school,
                     sim.results.edu.prop_completed,
                     sim.results.edu.prop_dropped]

        ax = axes[-1]
        ax.stackplot(t, all_props, labels=['In school', 'Completed', 'Dropped'], alpha=0.8)
        ax.set_title('All AGYW')
        ax.legend()

        sc.figlayout()
        pl.show()

    if 'run_scenario' in to_run:

        s_base = fp.Sim(education_module=edu, connectors=[mens], **pars)
        edu = Education(objective_data=objective_data, attainment_data=attainment_data)
        mens = Menstruation()
        s_intv = fp.Sim(education_module=edu, connectors=[mens], interventions=contra_hmb(), **pars)

        s_base.run()
        s_intv.run()

        # Plot
        import numpy as np
        t = s_base.results.menstruation.timevec[::12]  # Take every 12th time point for speed
        years = np.array([y.year for y in t])
        si = sc.findfirst(years, 2020)
        years = years[si:]
        fig, axes = pl.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        res_to_plot = ['hiud', 'hmb', 'poor_mh', 'anemic', 'pain', 'hyst']
        labels = ['hIUD Usage', 'HMB ', 'Poor MH', 'Anemic', 'Pain', 'Hysterectomy']

        for i, res in enumerate(res_to_plot):
            ax = axes[i]
            r0 = s_base.results.menstruation[f'{res}_prev']
            y0 = r0[::12][si:]
            y1 = s_intv.results.menstruation[f'{res}_prev'][::12][si:]
            ax.plot(years, y0, label='Baseline')
            ax.plot(years, y1, label='Increased IUD usage')
            # ax.set_xticks(years[::2])
            ax.axvline(x=2026, color='k', ls='--')
            ax.set_title(labels[i])

        pl.legend(fontsize=16, frameon=False, loc='upper left')
        sc.figlayout()
        sc.savefig('hmb_scenario_results.png', dpi=150)

        # Plot education
        fig, axes = pl.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        res_to_plot = ['mean_attainment', 'mean_objective']

        for i, res in enumerate(res_to_plot):
            ax = axes[i]
            r0 = s_base.results.edu[res]
            y0 = r0[::12][si:]
            ax.plot(years, y0)
            ax.set_title(res)
            ax.set_title(res.split('_')[1].capitalize())

        all_props = [s_base.results.edu.prop_in_school[::12][si:],
                     s_base.results.edu.prop_completed[::12][si:],
                     s_base.results.edu.prop_dropped[::12][si:]]

        ax = axes[2]
        ax.stackplot(years, all_props, labels=['In school', 'Completed', 'Dropped'], alpha=0.8)
        ax.set_title('All AGYW')
        ax.legend(fontsize=16)

        edu_res = ['prop_in_school', 'prop_completed', 'prop_dropped']
        for i, res in enumerate(edu_res):
            ax = axes[i+3]
            r0 = s_base.results.edu[res]
            y0 = r0[::12][si:]
            y1 = s_intv.results.edu[res][::12][si:]
            ax.plot(years, y0, label='Baseline')
            ax.plot(years, y1, label='Increased IUD usage')
            # ax.set_xticks(years[::2])
            ax.axvline(x=2026, color='k', ls='--')
            ax.set_title(res.strip('prop_').replace('_',' ').capitalize())

        pl.legend(fontsize=16, frameon=False, loc='upper left')
        sc.figlayout()
        sc.savefig('iud_edu_results.png', dpi=150)

        # pl.show()

        # # Cumulative impact
        # base_hmb = s_base.results.menstruation.n_hmb[-1]
        # intv_hmb = s_intv.results.menstruation.n_hmb[-1]
        # n_averted = (base_hmb - intv_hmb)
        # print(f"Reduction in number of women with HMB: {n_averted/1e6}M, or {n_averted/base_hmb:.2%} of baseline")
        #
        # base_hyst = s_base.results.menstruation.n_hyst[-1]
        # intv_hyst = s_intv.results.menstruation.n_hyst[-1]
        # n_averted = (base_hyst - intv_hyst)
        # print(f"Reduction in number of women having had hysterectomies: {n_averted/1e6}M, or {n_averted/base_hyst:.2%} of baseline")
