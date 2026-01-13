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
import numpy as np




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
  
class hiud_hmb(ss.Intervention):
    def __init__(self, pars=None, eligibility=None, **kwargs):
        super().__init__(name='hiud_hmb', eligibility=eligibility)
        self.define_pars(
            year=2026,  # When to apply the intervention
            prob_offer=ss.bernoulli(p=0.1),  # Proportion of eligible people who are offered
            prob_accept=ss.bernoulli(p=0.5),  # Proportion of those offered who accept
        )
        self.update_pars(pars, **kwargs)
        if eligibility is None:
            self.eligibility = lambda sim: (
                    sim.people.menstruation.hmb_prone &
                    sim.people.menstruation.menstruating &
                    ~sim.people.fp.pregnant &
                    ~sim.people.fp.postpartum)
        self.define_states(
            ss.BoolState('intervention_applied', label="Received hIUD through intervention"),
            ss.BoolState('hiud_offered', label="Was offered hIUD"),
            ss.BoolState('hiud_accepted', label="Accepted hIUD"),
        )
        return
    
    @property
    def iud_idx(self):
        """ Get the index of the IUD method """
        return self.sim.connectors.contraception.get_method_by_label('IUDs').idx
    
    def step(self):
        sim = self.sim
        if sim.t.now() == self.pars.year:
            print('Offering hIUD for HMB!')

            # Step 1: Get eligible people
            elig_uids = self.check_eligibility()
            print(f"Eligible for intervention: {len(elig_uids)}")
            
            # Step 2: Randomly select 10% to be offered the intervention
            offered_uids = self.pars.prob_offer.filter(elig_uids)
            self.hiud_offered[offered_uids] = True
            print(f"Offered intervention: {len(offered_uids)}")
            
            # Step 3: Of those offered, 50% accept
            accept_uids = self.pars.prob_accept.filter(offered_uids)
            self.hiud_accepted[accept_uids] = True
            print(f"Accepted intervention: {len(accept_uids)}")

            # Step 4: Apply the intervention to those who accepted
            # adjust contraceptive method
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
            prob_offer=ss.bernoulli(p=0.1),  # Proportion of eligible people who are offered
            prob_accept=ss.bernoulli(p=0.5),  # Proportion of those offered who accept        
            )
        self.update_pars(pars, **kwargs)
        if eligibility is None:
            self.eligibility = lambda sim: (
                    sim.people.menstruation.hmb_prone &
                    sim.people.menstruation.menstruating &
                    # ~sim.people.on_contra &
                    ~sim.people.fp.pregnant &
                    ~sim.people.fp.postpartum)
        self.define_states(
            ss.BoolState('intervention_applied', label="Received TXA through intervention"),
            ss.BoolState('txa_offered', label="Was offered TXA"),
            ss.BoolState('txa_accepted', label="Accepted TXA"),
        )
        return

    def step(self):
        sim = self.sim
        if sim.t.now() == self.pars.year:
            # Print message
            print('Offering TXA for HMB')

            # Step 1: Get eligible people
            elig_uids = self.check_eligibility()
            print(f"Eligible for intervention: {len(elig_uids)}")
            
            # Step 2: Randomly select 10% to be offered the intervention
            offered_uids = self.pars.prob_offer.filter(elig_uids)
            self.txa_offered[offered_uids] = True
            print(f"Offered intervention: {len(offered_uids)}")
            
            # Step 3: Of those offered, 50% accept
            accept_uids = self.pars.prob_accept.filter(offered_uids)
            self.txa_accepted[accept_uids] = True
            print(f"Accepted intervention: {len(accept_uids)}")
        
            # Step 4: Apply the intervention to those who accepted            
            sim.people.menstruation.txa[accept_uids] = True
            self.intervention_applied[accept_uids] = True
            
            # todo: change contraceptive method and/or eligibility requires on non-hormonal contraceptive method
        return



class pill_hmb(ss.Intervention):
    def __init__(self, pars=None, eligibility=None, **kwargs):
        super().__init__(name='pill_hmb', eligibility=eligibility)
        self.define_pars(
            year=2026,  # When to apply the intervention
            prob_offer=ss.bernoulli(p=0.1),  # Proportion of eligible people who are offered
            prob_accept=ss.bernoulli(p=0.5),  # Proportion of those offered who accept        
            )
        self.update_pars(pars, **kwargs)
        if eligibility is None:
            self.eligibility = lambda sim: (
                    sim.people.menstruation.hmb_prone &
                    sim.people.menstruation.menstruating &
                    # ~sim.people.on_contra &
                    ~sim.people.fp.pregnant &
                    ~sim.people.fp.postpartum)
        self.define_states(
            ss.BoolState('intervention_applied', label="Received pill through intervention"),
            ss.BoolState('pill_offered', label="Was offered pill"),
            ss.BoolState('pill_accepted', label="Accepted pill"),
        )
        return

    @property
    def pill_idx(self):
        """ Get the index of the pill method """
        return self.sim.connectors.contraception.get_method_by_label('Pill').idx

    def step(self):
        sim = self.sim
        if sim.t.now() == self.pars.year:
            # Print message
            print('Offering pill for HMB!')
            
            # Step 1: Get eligible people
            elig_uids = self.check_eligibility()
            print(f"Eligible for intervention: {len(elig_uids)}")
            
            # Step 2: Randomly select 10% to be offered the intervention
            offered_uids = self.pars.prob_offer.filter(elig_uids)
            self.pill_offered[offered_uids] = True
            print(f"Offered intervention: {len(offered_uids)}")
            
            # Step 3: Of those offered, 50% accept
            accept_uids = self.pars.prob_accept.filter(offered_uids)
            self.pill_accepted[accept_uids] = True
            print(f"Accepted intervention: {len(accept_uids)}")
        
            # Step 4: Apply the intervention to those who accepted              
            # adjust contraception method
            sim.people.fp.method[accept_uids] = self.pill_idx
            sim.people.fp.on_contra[accept_uids] = True
            sim.people.fp.ever_used_contra[accept_uids] = True
            # adjust method duration
            method_dur = sim.connectors.contraception.set_dur_method(accept_uids)
            sim.people.fp.ti_contra[accept_uids] = self.ti + method_dur

            self.intervention_applied[accept_uids] = True
        return

    


class hmb_package(ss.Intervention):
    """
    Intervention package that offers hIUD, TXA, and pill sequentially
    to a proportion of the eligible population
    """
    def __init__(self, pars=None, eligibility=None, **kwargs):
        super().__init__(name='hmb_package', eligibility=eligibility)
        self.define_pars(
            year=2026,
            prob_offer=ss.bernoulli(p=0.2),  # 20% offered the package
            prob_accept_hiud=ss.bernoulli(p=0.5),  # 50% accept contraception
            prob_accept_txa=ss.bernoulli(p=0.5),  # 50% accept TXA
            prob_accept_pill=ss.bernoulli(p=0.5),  # 50% accept pill
        )
        self.update_pars(pars, **kwargs)
        if eligibility is None:
            self.eligibility = lambda sim: (
                    sim.people.menstruation.hmb_prone &
                    sim.people.menstruation.menstruating &
                    ~sim.people.fp.pregnant &
                    #~sim.people.fp.postpartum & 
                    (sim.connectors.fp.ti_delivery != sim.connectors.fp.ti)
                    )
        self.define_states(
            ss.BoolState('package_offered', label="Was offered HMB package"),
            ss.BoolState('hiud_offered', label="Was offered hIUD"),
            ss.BoolState('hiud_accepted', label="Accepted hIUD"),
            ss.BoolState('txa_offered', label="Was offered TXA"),
            ss.BoolState('txa_accepted', label="Accepted TXA"),
            ss.BoolState('pill_offered', label="Was offered pill"),
            ss.BoolState('pill_accepted', label="Accepted pill"),
        )
        return
    
    @property
    def iud_idx(self):
        return self.sim.connectors.contraception.get_method_by_label('IUDs').idx
    
    @property
    def pill_idx(self):
        return self.sim.connectors.contraception.get_method_by_label('Pill').idx
    
    def step(self):
        sim = self.sim
        if sim.t.now() == self.pars.year:
            
            # Step 1: Get eligible people
            elig_uids = self.check_eligibility()
            
            # Step 2: Select 20% to offer the package
            package_offered_uids = self.pars.prob_offer.filter(elig_uids)
            self.package_offered[package_offered_uids] = True
            
            # Step 3: Offer the three interventions in the package
            # 3.1 TXA
            self.txa_offered[package_offered_uids] = True
            txa_accept_uids = self.pars.prob_accept_txa.filter(package_offered_uids)
            self.txa_accepted[txa_accept_uids] = True
            
            # 3.2 pill
            txa_declined_uids = np.setdiff1d(package_offered_uids, txa_accept_uids)
            self.pill_offered[txa_declined_uids] = True
            pill_accept_uids = self.pars.prob_accept_pill.filter(txa_declined_uids)
            self.pill_accepted[pill_accept_uids] = True
            # apply contraception
            sim.people.fp.method[pill_accept_uids] = self.pill_idx
            sim.people.fp.on_contra[pill_accept_uids] = True
            sim.people.fp.ever_used_contra[pill_accept_uids] = True
            method_dur = sim.connectors.contraception.set_dur_method(pill_accept_uids)
            sim.people.fp.ti_contra[pill_accept_uids] = self.ti + method_dur
            
            # 3.3 hIUD
            pill_declined_uids = np.setdiff1d(txa_declined_uids, pill_accept_uids)
            self.hiud_offered[pill_declined_uids] = True
            hiud_accept_uids = self.pars.prob_accept_hiud.filter(pill_declined_uids)
            self.hiud_accepted[hiud_accept_uids] = True
            # apply contraception
            sim.people.fp.method[hiud_accept_uids] = self.iud_idx
            sim.people.fp.on_contra[hiud_accept_uids] = True
            sim.people.fp.ever_used_contra[hiud_accept_uids] = True
            method_dur = sim.connectors.contraception.set_dur_method(hiud_accept_uids)
            sim.people.fp.ti_contra[hiud_accept_uids] = self.ti + method_dur
            sim.people.menstruation.hiud_prone[hiud_accept_uids] = 1
            
            # Summary
            total_accepted = len(hiud_accept_uids) + len(txa_accept_uids) + len(pill_accept_uids)
            print(f"Total who accepted any intervention: {total_accepted}")
            print(f"  hIUD: {len(hiud_accept_uids)}")
            print(f"  TXA: {len(txa_accept_uids)}")
            print(f"  Pill: {len(pill_accept_uids)}")
            print(f"  None: {len(package_offered_uids) - total_accepted}")
            
        return



                
            




















