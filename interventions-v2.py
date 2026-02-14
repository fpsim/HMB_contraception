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
        if sim.t.now() >= self.pars.year: #sustained intervention
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

class nsaid(ss.Intervention):
    """
    Standalone NSAID intervention for HMB
    """
    def __init__(self, pars=None, eligibility=None, **kwargs):
        super().__init__(name='nsaid', eligibility=eligibility)
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
            ss.BoolState('intervention_applied', label="Received NSAID through intervention"),
            ss.BoolState('nsaid_offered', label="Was offered NSAID"),
            ss.BoolState('nsaid_accepted', label="Accepted NSAID"),
        )
        return

    def step(self):
        sim = self.sim
        if sim.t.now() >= self.pars.year: #sustained intervention
            # Print message
            print('Offering NSAID for HMB')

            # Step 1: Get eligible people
            elig_uids = self.check_eligibility()
            print(f"Eligible for intervention: {len(elig_uids)}")
            
            # Step 2: Randomly select to be offered the intervention
            offered_uids = self.pars.prob_offer.filter(elig_uids)
            self.nsaid_offered[offered_uids] = True
            print(f"Offered intervention: {len(offered_uids)}")
            
            # Step 3: Of those offered, some accept
            accept_uids = self.pars.prob_accept.filter(offered_uids)
            self.nsaid_accepted[accept_uids] = True
            print(f"Accepted intervention: {len(accept_uids)}")
        
            # Step 4: Apply the intervention to those who accepted            
            sim.people.menstruation.nsaid[accept_uids] = True
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
        if sim.t.now() >= self.pars.year: #sustained intervention
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
        if sim.t.now() >= self.pars.year: #sustained intervention
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
            
            prob_offer=sc.objdict(
                nsaid=ss.bernoulli(p=0.2),
                txa=ss.bernoulli(p=0.2),
                pill=ss.bernoulli(p=0.2),
                hiud=ss.bernoulli(p=0.2),
            ),
            
            prob_accept_nsaid=ss.bernoulli(p=0.5), # 50% accept NSAID
            prob_accept_txa=ss.bernoulli(p=0.5),  # 50% accept TXA
            prob_accept_pill=ss.bernoulli(p=0.5),  # 50% accept pill
            prob_accept_hiud=ss.bernoulli(p=0.5),  # 50% accept contraception
        )
        self.update_pars(pars, **kwargs)
        if eligibility is None:
            self.eligibility = lambda sim: (
                    sim.people.menstruation.hmb_prone &
                    sim.people.menstruation.menstruating &
                    ~sim.people.fp.pregnant &
                   # ~sim.people.fp.postpartum & 
                    (sim.connectors.fp.ti_delivery != sim.connectors.fp.ti)
                    )
        self.define_states(
            ss.BoolState('package_offered', label="Was offered HMB package"),
            ss.BoolState('nsaid_offered', label="Was offered NSAID"),
            ss.BoolState('nsaid_accepted', label="Accepted NSAID"),
            ss.BoolState('txa_offered', label="Was offered TXA"),
            ss.BoolState('txa_accepted', label="Accepted TXA"),
            ss.BoolState('pill_offered', label="Was offered pill"),
            ss.BoolState('pill_accepted', label="Accepted pill"),
            ss.BoolState('hiud_offered', label="Was offered hIUD"),
            ss.BoolState('hiud_accepted', label="Accepted hIUD"),
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
        if sim.t.now() >= self.pars.year: #sustained intervention
            
            # Step 1: Get eligible people
            elig_uids = self.check_eligibility()
            
            # Step 2: Select 20% to offer the package
            package_offered_uids = self.pars.prob_offer_package.filter(elig_uids)
            self.package_offered[package_offered_uids] = True
            
            # Step 3: Offer the four interventions in the package in order: NSAID → TXA → Pill → hIUD

            # 3.1 NSAID 
            nsaid_offered_uids = self.pars.prob_offer.nsaid.filter(package_offered_uids)
            self.nsaid_offered[nsaid_offered_uids] = True
            nsaid_accept_uids = self.pars.prob_accept_nsaid.filter(nsaid_offered_uids)
            self.nsaid_accepted[nsaid_accept_uids] = True
            sim.people.menstruation.nsaid[nsaid_accept_uids] = True
            
            # 3.2 TXA
            nsaid_declined_uids = np.setdiff1d(package_offered_uids, nsaid_accept_uids)
            txa_offered_uids = self.pars.prob_offer.txa.filter(nsaid_declined_uids)
            self.txa_offered[txa_offered_uids] = True
            txa_accept_uids = self.pars.prob_accept_txa.filter(txa_offered_uids)
            self.txa_accepted[txa_accept_uids] = True
            sim.people.menstruation.txa[txa_accept_uids] = True
            
            # 3.3 pill
            txa_declined_uids = np.setdiff1d(nsaid_declined_uids, txa_accept_uids)
            pill_offered_uids = self.pars.prob_offer.pill.filter(txa_declined_uids)
            self.pill_offered[pill_offered_uids] = True
            pill_accept_uids = self.pars.prob_accept_pill.filter(pill_offered_uids)
            self.pill_accepted[pill_accept_uids] = True
            # apply contraception
            sim.people.fp.method[pill_accept_uids] = self.pill_idx
            sim.people.fp.on_contra[pill_accept_uids] = True
            sim.people.fp.ever_used_contra[pill_accept_uids] = True
            method_dur = sim.connectors.contraception.set_dur_method(pill_accept_uids)
            sim.people.fp.ti_contra[pill_accept_uids] = self.ti + method_dur
            
            # 3.4 hIUD (only offer to those who declined/weren't offered pill)
            pill_declined_uids = np.setdiff1d(txa_declined_uids, pill_accept_uids)
            hiud_offered_uids = self.pars.prob_offer.hiud.filter(pill_declined_uids)
            self.hiud_offered[hiud_offered_uids] = True
            hiud_accept_uids = self.pars.prob_accept_hiud.filter(hiud_offered_uids)
            self.hiud_accepted[hiud_accept_uids] = True
            
            # apply contraception
            sim.people.fp.method[hiud_accept_uids] = self.iud_idx
            sim.people.fp.on_contra[hiud_accept_uids] = True
            sim.people.fp.ever_used_contra[hiud_accept_uids] = True
            method_dur = sim.connectors.contraception.set_dur_method(hiud_accept_uids)
            sim.people.fp.ti_contra[hiud_accept_uids] = self.ti + method_dur
            sim.people.menstruation.hiud_prone[hiud_accept_uids] = 1
            
            # Summary
            total_accepted = len(nsaid_accept_uids) + len(hiud_accept_uids) + len(txa_accept_uids) + len(pill_accept_uids)
            print(f"Total who accepted any intervention: {total_accepted}")
            print(f"  NSAID: {len(nsaid_accept_uids)}")
            print(f"  TXA: {len(txa_accept_uids)}")
            print(f"  Pill: {len(pill_accept_uids)}")
            print(f"  hIUD: {len(hiud_accept_uids)}")
            print(f"  None: {len(package_offered_uids) - total_accepted}")
            
        return
    
    
class HMBCarePathway(ss.Intervention):
        """
        Sequential HMB treatment cascade following the intervention pathway.
        Implements: HMB → Care-seeking → NSAID → TXA → Pill → hIUD
    
        Each individual progresses through treatments based on:
            - Care-seeking behavior
            - Treatment effectiveness
            - Adherence
            - Fertility intent 
            """
        # Treatment encoding mapping
        treatment_map = {
            'none': 0,
            'nsaid': 1,
            'txa': 2,
            'pill': 3,
            'hiud': 4,
        }
    
        treatment_reverse_map = {v: k for k, v in treatment_map.items()}
    
        def __init__(self, pars=None, eligibility=None, **kwargs):
            super().__init__(name='hmb_care_pathway', eligibility=eligibility)
        
            self.define_pars(
                year=2026,  # When intervention starts
            
                # Care-seeking behavior
                prob_seek_care=ss.bernoulli(p=0.3),
                        
                # Adherence (probability of continuing if treatment works)
                adherence=sc.objdict(
                    nsaid=0.7,
                    txa=0.6,
                    pill=0.75,
                    hiud=0.85,
                ),
            
                # Timing parameters
                time_to_assess=3,  # Months before assessing effectiveness
                treatment_duration_months=sc.objdict(
                    nsaid=12,
                    txa=12,
                    pill=24,
                    hiud=60,  # 5 years
                ),
                
                # Offer + accept at each treatment node
                prob_offer=sc.objdict(
                    nsaid=ss.bernoulli(p=0.9),
                    txa=ss.bernoulli(p=0.9),
                    pill=ss.bernoulli(p=0.9),
                    hiud=ss.bernoulli(p=0.9),
                ),
                prob_accept=sc.objdict(
                    nsaid=ss.bernoulli(p=0.7),
                    txa=ss.bernoulli(p=0.6),
                    pill=ss.bernoulli(p=0.5),
                    hiud=ss.bernoulli(p=0.5),
                ), 
            )
        
            self.update_pars(pars, **kwargs)
        
            # Default eligibility: menstruating, non-pregnant women experiencing HMB
            if eligibility is None:
                self.eligibility = lambda sim: (
                    sim.people.menstruation.hmb &  # Currently experiencing HMB
                    sim.people.menstruation.menstruating &
                    ~sim.people.fp.pregnant &
                    ~sim.people.fp.postpartum
                )
        
            # Define states for tracking cascade
            self.define_states(
                # Treatment history
                ss.BoolState('seeking_care', default=False),
                ss.BoolState('tried_nsaid', default=False),
                ss.BoolState('tried_txa', default=False),
                ss.BoolState('tried_pill', default=False),
                ss.BoolState('tried_hiud', default=False),
                ss.BoolState('did_not_seek_care', default=False),  # Track "Seeking care?" -> No
            
                # Current treatment status
                ss.FloatArr('current_treatment', default=0), # 0=none, 1=nsaid, 2=txa, 3=pill, 4=hiud
                ss.BoolState('on_treatment', default=False),
                ss.FloatArr('treatment_start_ti', default=-1),
                ss.FloatArr('treatment_duration', default=0),
            
                # Treatment outcomes
                ss.BoolState('treatment_effective', default=False),
                ss.BoolState('treatment_assessed', default=False),
                ss.BoolState('adherent', default=False),
                
                ss.BoolState('nsaid_offered', default=False),
                ss.BoolState('nsaid_accepted', default=False),

                ss.BoolState('txa_offered', default=False),
                ss.BoolState('txa_accepted', default=False),

                ss.BoolState('pill_offered', default=False),
                ss.BoolState('pill_accepted', default=False),

                ss.BoolState('hiud_offered', default=False),
                ss.BoolState('hiud_accepted', default=False),
            )
        
            return
    
        def init_results(self):
            """Initialize results for cascade tracking"""
            super().init_results()
            results = [
                ss.Result('seeking_care_prev', scale=False, label="Proportion seeking care"),
                ss.Result('on_nsaid_prev', scale=False, label="Proportion on NSAID"),
                ss.Result('on_txa_prev', scale=False, label="Proportion on TXA"),
                ss.Result('on_pill_hmb_prev', scale=False, label="Proportion on pill for HMB"),
                ss.Result('on_hiud_hmb_prev', scale=False, label="Proportion on hIUD for HMB"),
                ss.Result('on_any_treatment_prev', scale=False, label="Proportion on any treatment"),
                ss.Result('treatment_success_rate', scale=False, label="Treatment success rate"),
                ss.Result('adherence_rate', scale=False, label="Adherence rate"),
                ss.Result('cascade_step_0', scale=False, label="No treatment tried"),
                ss.Result('cascade_step_1', scale=False, label="Tried NSAID"),
                ss.Result('cascade_step_2', scale=False, label="Tried TXA"),
                ss.Result('cascade_step_3', scale=False, label="Tried Pill"),
                ss.Result('cascade_step_4', scale=False, label="Tried hIUD"),
                ss.Result('no_care_prev', scale=False, label="Proportion with HMB not seeking care"),
            ]
            self.define_results(*results)
            return
    
        @property
        def pill_idx(self):
            """Get pill method index"""
            return self.sim.connectors.contraception.get_method_by_label('Pill').idx
    
        @property
        def iud_idx(self):
            """Get IUD method index"""
            return self.sim.connectors.contraception.get_method_by_label('IUDs').idx
    
        def step(self):
            """Execute cascade at each timestep"""
            # Only run if intervention has started
            if self.sim.t.now() < self.pars.year:
                return
        
            # 1. Determine who seeks care
            self.determine_care_seeking()
        
            # 2. Offer next treatment in cascade
            self.offer_treatment()
        
            # 3. Update treatment duration
            self.update_treatment_duration()
        
            # 4. Assess treatment effectiveness
            self.assess_treatment_effectiveness()
        
            # 5. Check adherence
            self.check_adherence()
        
            return
    
        def determine_care_seeking(self):
            """
            Determine who seeks care for HMB this timestep.
            Following the pathway: HMB → Seeking care?
            """
            # Eligible: currently have HMB, not already seeking care, not on treatment
            eligible = (
                self.sim.people.menstruation.hmb &
                ~self.seeking_care &
                ~self.on_treatment &
                self.sim.people.menstruation.menstruating &
                ~self.sim.people.fp.pregnant
            ).uids
        
            if len(eligible) == 0:
                return
            
            # Reset the "no care" marker for these eligible this timestep
            self.did_not_seek_care[eligible] = False
    
            # Randomly select who seeks care
            seek_care_uids = self.pars.prob_seek_care.filter(eligible)
            self.seeking_care[seek_care_uids] = True
            
            # Everyone else does NOT seek care this timestep → no effective treatment; no change to HMB status
            no_care_uids = np.setdiff1d(eligible, seek_care_uids)
            self.did_not_seek_care[no_care_uids] = True
        
            return
    
        def offer_treatment(self):
            """
            Offer next treatment in cascade: NSAID → TXA → Pill → hIUD
            Skip pill/hIUD if fertility_intent = True
            """
            # Eligible: seeking care, not currently on treatment
            eligible = (self.seeking_care & ~self.on_treatment).uids
        
            if len(eligible) == 0:
                return
        
            # Get fertility intent from FPsim
            fertility_intent = self.sim.people.fp.fertility_intent
        
            for uid in eligible:
                # NSAID node
                if not self.tried_nsaid[uid]:
                    started = self._offer_and_start(uid, 'nsaid')
                    if started:
                        continue  # they are now on treatment

                # TXA node
                if not self.tried_txa[uid]:
                    started = self._offer_and_start(uid, 'txa')
                    if started:
                        continue

                # Pill node (only if no fertility intent)
                if (not fertility_intent[uid]) and (not self.tried_pill[uid]):
                    started = self._offer_and_start(uid, 'pill')
                    if started:
                        continue

                # hIUD node (only if no fertility intent)
                if (not fertility_intent[uid]) and (not self.tried_hiud[uid]):
                    started = self._offer_and_start(uid, 'hiud')
                    if started:
                        continue

                # Exhausted or blocked → exit care pathway this timestep
                self.seeking_care[uid] = False
            return
    
    
        def _offer_and_start(self, uid, treatment_type):
            """
            Offer and accept logic for a treatment node.
            Returns True if treatment started, False otherwise.
            """
            # offer
            offered = self.pars.prob_offer[treatment_type].filter(np.array([uid]))
            if len(offered):
                getattr(self, f"{treatment_type}_offered")[uid] = True
            else:
                # Not offered counts as "tried/considered" for cascade progression
                getattr(self, f"tried_{treatment_type}")[uid] = True
                return False
            
            accepted = self.pars.prob_accept[treatment_type].filter(np.array([uid]))
            if len(accepted):
                getattr(self, f"{treatment_type}_accepted")[uid] = True
                self._start_treatment(uid, treatment_type)  # this sets tried_*, on_treatment, and module flags
                return True
            else:
                # Declined counts as "tried/considered" so you can advance next timestep
                getattr(self, f"tried_{treatment_type}")[uid] = True
                return False


        def _start_treatment(self, uid, treatment_type):
            """
            Start a specific treatment for an individual.
            Updates both intervention states AND menstruation module states.
            """
            # Mark as tried
            getattr(self, f"tried_{treatment_type}")[uid] = True
            
            # Set current treatment tracking
            self.current_treatment[uid] = self.treatment_map[treatment_type]
            self.on_treatment[uid] = True
            self.treatment_start_ti[uid] = self.ti
            self.treatment_duration[uid] = 0
            self.treatment_assessed[uid] = False
        
            # Update menstruation module states (these affect HMB probability)
            mens = self.sim.people.menstruation
        
            if treatment_type == 'nsaid':
                mens.nsaid[uid] = True
            
            elif treatment_type == 'txa':
                mens.txa[uid] = True
            
            elif treatment_type == 'pill':
                # Set as contraceptive method
                self.sim.people.fp.method[uid] = self.pill_idx
                self.sim.people.fp.on_contra[uid] = True
                self.sim.people.fp.ever_used_contra[uid] = True
                # Set method duration
                method_dur = self.sim.connectors.contraception.set_dur_method(np.array([uid]))
                self.sim.people.fp.ti_contra[uid] = self.ti + method_dur[0]
                # Menstruation module will detect pill usage in step_states()
            
            elif treatment_type == 'hiud':
                # Set as contraceptive method
                self.sim.people.fp.method[uid] = self.iud_idx
                self.sim.people.fp.on_contra[uid] = True
                self.sim.people.fp.ever_used_contra[uid] = True
                mens.hiud_prone[uid] = True  # Mark as hormonal IUD user
                # Set method duration
                method_dur = self.sim.connectors.contraception.set_dur_method(np.array([uid]))
                self.sim.people.fp.ti_contra[uid] = self.ti + method_dur[0]
                # Menstruation module will detect hIUD usage in step_states()
        
            return
    
        def assess_treatment_effectiveness(self):
            """
            After time_to_assess months, check if HMB has improved.
            
            Treatment is considered effective if the person's HMB status has resolved.
            The effect is determined by hmb_pred coefficients in menstruation.py
            """
            
            # Find those ready to assess
            on_treatment_uids = (self.on_treatment & ~self.treatment_assessed).uids
        
            if len(on_treatment_uids) == 0:
                return
        
            # Check if enough time has passed
            time_on_treatment = self.ti - self.treatment_start_ti[on_treatment_uids]
            ready_to_assess = on_treatment_uids[time_on_treatment >= self.pars.time_to_assess]
        
            if len(ready_to_assess) == 0:
                return
        
            # Assess each person
            for uid in ready_to_assess:
                has_hmb = self.sim.people.menstruation.hmb[uid]
            
                if not has_hmb:
                    # HMB resolved - 
                    self.treatment_effective[uid] = True
                    # Person continues to adherence check
                else:
                    # HMB still present - treatment not effective for this person
                    self.treatment_effective[uid] = False
                    # Stop treatment and try next option in cascade
                    self._stop_treatment(uid, success=False)
        
                self.treatment_assessed[uid] = True
    
            return
    
        def check_adherence(self):
            """
            For those with effective treatment, check adherence.
            Following pathway: Does she adhere? → Yes/No
            """
            # Eligible: on treatment, treatment effective, not yet checked adherence
            eligible = (
                self.on_treatment &
                self.treatment_effective &
                ~self.adherent
            ).uids
        
            if len(eligible) == 0:
                return
        
            for uid in eligible:
                treatment = self.treatment_reverse_map[self.current_treatment[uid]]
                adherence_prob = self.pars.adherence[treatment]
            
                # Randomly determine adherence
                if np.random.random() < adherence_prob:
                    self.adherent[uid] = True
                    # Continue treatment
                else:
                    self.adherent[uid] = False
                    # Non-adherent - stop and try next option
                    self._stop_treatment(uid, success=False)
        
            return
    
        def _stop_treatment(self, uid, success=False):
            """
            Stop current treatment.
            Updates both intervention states AND menstruation module states.
            """
            treatment = self.treatment_reverse_map[self.current_treatment[uid]]
            mens = self.sim.people.menstruation
        
            # Reset menstruation module treatment states
            if treatment == 'nsaid':
                mens.nsaid[uid] = False
            
            elif treatment == 'txa':
                mens.txa[uid] = False
            
            elif treatment in ['pill', 'hiud']:
                # Keep contraceptive method (they might continue for contraception)
                # Or optionally reset:
                # self.sim.people.fp.method[uid] = 0
                # self.sim.people.fp.on_contra[uid] = False
                pass
        
            # Reset tracking states
            self.current_treatment[uid] = 0
            self.on_treatment[uid] = False
            self.treatment_duration[uid] = 0
            self.treatment_effective[uid] = False
            self.treatment_assessed[uid] = False
        
            if not success:
               # Treatment failed/discontinued - can try next option
               # seeking_care stays True
               pass
            else:
                # Successfully completed treatment duration
                self.seeking_care[uid] = False
                self.adherent[uid] = False
        
            return
    
        def update_treatment_duration(self):
            """
            Increment duration for those on treatment.
            Check if treatment duration is complete.
            """
            on_treatment_uids = self.on_treatment.uids
        
            if len(on_treatment_uids) == 0:
                return
        
            # Increment duration (convert to months)
            self.treatment_duration[on_treatment_uids] += self.sim.t.dt_year * 12
        
            # Check if anyone completed their duration
            for uid in on_treatment_uids:
               if not self.adherent[uid]:
                   continue
            
               treatment = self.treatment_reverse_map[self.current_treatment[uid]]
               max_duration = self.pars.treatment_duration_months[treatment]
            
               if self.treatment_duration[uid] >= max_duration:
                    # Completed successfully
                    self._stop_treatment(uid, success=True)
        
            return
    
        def update_results(self):
            """Track cascade metrics"""
            super().update_results()
            ti = self.ti
        
            # Get menstruating women
            mens = self.sim.people.menstruation.menstruating
            hmb_cases = self.sim.people.menstruation.hmb & mens
        
            # Care-seeking
            if np.count_nonzero(hmb_cases) > 0:
                self.results.seeking_care_prev[ti] = np.count_nonzero(self.seeking_care & hmb_cases) / np.count_nonzero(hmb_cases)
            else:
                self.results.seeking_care_prev[ti] = 0
                
            # No-care prevalence among HMB cases (diagram branch: Seeking care? -> No)
            if np.count_nonzero(hmb_cases) > 0:
                self.results.no_care_prev[ti] = np.count_nonzero(self.did_not_seek_care & hmb_cases) / np.count_nonzero(hmb_cases)
            else:
                self.results.no_care_prev[ti] = 0 
        
            # Treatment prevalence
            if np.count_nonzero(mens) > 0:
                self.results.on_nsaid_prev[ti] = np.count_nonzero((self.current_treatment == 1) & mens) / np.count_nonzero(mens)
                self.results.on_txa_prev[ti] = np.count_nonzero((self.current_treatment == 2) & mens) / np.count_nonzero(mens)
                self.results.on_pill_hmb_prev[ti] = np.count_nonzero((self.current_treatment == 3) & mens) / np.count_nonzero(mens)
                self.results.on_hiud_hmb_prev[ti] = np.count_nonzero((self.current_treatment == 4) & mens) / np.count_nonzero(mens)
                self.results.on_any_treatment_prev[ti] = np.count_nonzero(self.on_treatment & mens) / np.count_nonzero(mens)
            else:
                self.results.on_nsaid_prev[ti] = 0
                self.results.on_txa_prev[ti] = 0
                self.results.on_pill_hmb_prev[ti] = 0
                self.results.on_hiud_hmb_prev[ti] = 0
                self.results.on_any_treatment_prev[ti] = 0
        
            # Success metrics
            assessed = self.treatment_assessed & mens
            if np.count_nonzero(assessed) > 0:
                self.results.treatment_success_rate[ti] = np.count_nonzero(self.treatment_effective & assessed) / np.count_nonzero(assessed)
            else:
                self.results.treatment_success_rate[ti] = 0
        
            effective_treatment = self.treatment_effective & mens
            if np.count_nonzero(effective_treatment) > 0:
                self.results.adherence_rate[ti] = np.count_nonzero(self.adherent & effective_treatment) / np.count_nonzero(effective_treatment)
            else:
                self.results.adherence_rate[ti] = 0
        
            # Cascade progression
            if np.count_nonzero(hmb_cases) > 0:
                self.results.cascade_step_0[ti] = np.count_nonzero(~self.tried_nsaid & hmb_cases) / np.count_nonzero(hmb_cases)
                self.results.cascade_step_1[ti] = np.count_nonzero(self.tried_nsaid & hmb_cases) / np.count_nonzero(hmb_cases)
                self.results.cascade_step_2[ti] = np.count_nonzero(self.tried_txa & hmb_cases) / np.count_nonzero(hmb_cases)
                self.results.cascade_step_3[ti] = np.count_nonzero(self.tried_pill & hmb_cases) / np.count_nonzero(hmb_cases)
                self.results.cascade_step_4[ti] = np.count_nonzero(self.tried_hiud & hmb_cases) / np.count_nonzero(hmb_cases)
            else:
                self.results.cascade_step_0[ti] = 0
                self.results.cascade_step_1[ti] = 0
                self.results.cascade_step_2[ti] = 0
                self.results.cascade_step_3[ti] = 0
                self.results.cascade_step_4[ti] = 0
        
            return

                
            




















