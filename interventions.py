"""
HMB module interventions
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
                    ~sim.people.fp.pregnant)
        self.define_states(
            ss.State('intervention_applied', label="Received hIUD through intervention"),
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
            print(f'Offering hIUD for HMB!')

            # Get women who accept the intervention package
            elig_uids = self.check_eligibility()
            accept_uids = self.pars.prob.filter(elig_uids)

            # Adjust their contraception method
            sim.people.fp.method[accept_uids] = self.iud_idx
            sim.people.fp.on_contra[accept_uids] = True
            sim.people.fp.ever_used_contra[accept_uids] = True
            # adjust method duration
            method_dur = sim.connectors.contraception.set_dur_method(accept_uids)
            sim.people.fp.ti_contra[accept_uids] = self.ti + method_dur
            sim.people.menstruation.hiud_prone[accept_uids] = 1

            self.intervention_applied[accept_uids] = True
        return


class txa(ss.Intervention):
    def __init__(self, pars=None, eligibility=None, **kwargs):
        super().__init__(name='txa', eligibility=eligibility)
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
                    ~sim.people.fp.pregnant)
        self.define_states(
            ss.State('intervention_applied', label="Received TXA through intervention"),
        )
        return

    def step(self):
        sim = self.sim
        if sim.t.now() == self.pars.year:
            # Print message
            print(f'Offering TXA for HMB')

            # Get women who accept the intervention package
            elig_uids = self.check_eligibility()
            accept_uids = self.pars.prob.filter(elig_uids)
            
            # update txa status
            sim.diseases.menstruation.txa[accept_uids] = True
            self.intervention_applied[accept_uids] = True
        return



class pill_hmb(ss.Intervention):
    def __init__(self, pars=None, eligibility=None, **kwargs):
        super().__init__(name='pill_hmb', eligibility=eligibility)
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
                    ~sim.people.fp.pregnant)
        self.define_states(
            ss.State('intervention_applied', label="Received pill through intervention"),
        )
        return

    @property
    def pill_idx(self):
        """ Get the index of the pill method """
        return self.sim.connectors.contraception.get_method_by_label('pill').idx

    def step(self):
        sim = self.sim
        if sim.t.now() == self.pars.year:
            # Print message
            print(f'Offering pill for HMB!')

            # Get women who accept the intervention package
            elig_uids = self.check_eligibility()
            accept_uids = self.pars.prob.filter(elig_uids)

            # Adjust their contraception method
            sim.people.fp.method[accept_uids] = self.pill_idx
            sim.people.fp.on_contra[accept_uids] = True
            sim.people.fp.ever_used_contra[accept_uids] = True
            # adjust method duration
            method_dur = sim.connectors.contraception.set_dur_method(accept_uids)
            sim.people.fp.ti_contra[accept_uids] = self.ti + method_dur

            self.intervention_applied[accept_uids] = True
        return
