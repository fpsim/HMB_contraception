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
from utils import logistic




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
            package_offered_uids = self.pars.prob_offer.filter(elig_uids)
            self.package_offered[package_offered_uids] = True
            
            # Step 3: Offer the four interventions in the package in order: NSAID → TXA → Pill → hIUD

            # 3.1 NSAID 
            self.nsaid_offered[package_offered_uids] = True
            nsaid_accept_uids = self.pars.prob_accept_nsaid.filter(package_offered_uids)
            self.nsaid_accepted[nsaid_accept_uids] = True
            sim.people.menstruation.nsaid[nsaid_accept_uids] = True
            
            # 3.2 TXA
            nsaid_declined_uids = np.setdiff1d(package_offered_uids, nsaid_accept_uids)
            self.txa_offered[nsaid_declined_uids] = True
            txa_accept_uids = self.pars.prob_accept_txa.filter(nsaid_declined_uids)
            self.txa_accepted[txa_accept_uids] = True
            sim.people.menstruation.txa[txa_accept_uids] = True

            
            # 3.3 pill
            txa_declined_uids = np.setdiff1d(nsaid_declined_uids, txa_accept_uids)
            self.pill_offered[txa_declined_uids] = True
            pill_accept_uids = self.pars.prob_accept_pill.filter(txa_declined_uids)
            self.pill_accepted[pill_accept_uids] = True
            # apply contraception
            sim.people.fp.method[pill_accept_uids] = self.pill_idx
            sim.people.fp.on_contra[pill_accept_uids] = True
            sim.people.fp.ever_used_contra[pill_accept_uids] = True
            method_dur = sim.connectors.contraception.set_dur_method(pill_accept_uids)
            sim.people.fp.ti_contra[pill_accept_uids] = self.ti + method_dur
            
            # 3.4 hIUD
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

        Pathway: HMB → Care-seeking → NSAID → TXA → Pill → hIUD

        Each individual progresses through treatments based on:
        - **Care-seeking behavior**: Logistic model with anemia/pain covariates
          and individual propensity
        - **Treatment efficacy**: Heterogeneous response; only responders
          receive biological effect
        - **Treatment effectiveness**: Assessed after 3 months based on HMB status
        - **Adherence**: Stochastic with treatment-specific probabilities
        - **Fertility intent**: Blocks access to pill/hIUD if planning pregnancy
        - **Time-based duration**: Treatments automatically stop after set duration

        Key features:
        - Vectorized operations for performance with large populations
        - Treatment responder system for realistic heterogeneous response
        - Automatic treatment stopping via ti_stop_treatment
        - Parallel treatment evaluation (not strictly sequential)
        - Integration with FPsim for contraceptive methods (pill/hIUD)

        States track:
        - Treatment history (tried_*, *_offered, *_accepted)
        - Current treatment status (on_treatment, current_treatment)
        - Treatment outcomes (treatment_effective, adherent)
        - Responder status (nsaid_responder, etc.)
        - Care-seeking behavior (seeking_care, care_seeking_propensity)
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
            
                # Care-seeking behavior: logistic regression
                care_behavior=sc.objdict(  # Parameters for poor menstrual hygiene
                    base = 0.5,  # Baseline 50% odds of seeking care
                    anemic = 1,  # Increased care-seeking behavior for those with anemia - placeholder
                    pain = 0.25, # Increased care-seeking behavior for those with pain - placeholder
                ),

                # Treatment efficacy (probability of being a responder)
                efficacy=sc.objdict(
                    nsaid=0.5,  # 50% of people respond to NSAID
                    txa=0.6,    # 60% of people respond to TXA
                    pill=0.7,   # 70% of people respond to pill
                    hiud=0.8,   # 80% of people respond to hIUD
                ),

                # Adherence (probability of continuing if treatment works)
                adherence=sc.objdict(
                    nsaid=0.7,
                    txa=0.6,
                    pill=0.75,
                    hiud=0.85,
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
                            
                # Timing parameters
                time_to_assess=ss.months(3),  # Months before assessing effectiveness
                dur_treatment=sc.objdict(
                    nsaid=ss.uniform(ss.months(10), ss.months(14)),
                    txa=ss.uniform(ss.months(10), ss.months(14)),
                    # Note that pill and hIUD are not included because FPsim assigns their duration
                ),
                # --- Prob of hormonal IUD
                # The probability of IUD usage is set within FPsim, so this parameter just
                # determines whether each woman is a hormonal or non-hormonal IUD user
                p_hiud=ss.bernoulli(p=0.17),
                                 
            )
        
            self.update_pars(pars, **kwargs)

            # Store probabilities that will be calculated within the module
            self._p_care = ss.bernoulli(p=0)
            self._p_accept = ss.bernoulli(p=0)
            self._p_adherent = ss.bernoulli(p=0)

            # Define states for tracking cascade
            self.define_states(
                # Care seeking
                ss.BoolState('seeking_care'),
                ss.FloatArr('care_seeking_propensity', default=ss.normal(1, 1)),

                # Treatment 
                ss.BoolState('nsaid'),
                ss.BoolState('txa'),
                ss.BoolState('hiud'),
                ss.BoolState('hiud_prone', label="Prone to use hormonal IUD, if using IUD"),

                # Treatment history
                ss.BoolState('tried_nsaid'),
                ss.BoolState('tried_txa'),
                ss.BoolState('tried_pill'),
                ss.BoolState('tried_hiud'),

                # States representing treatment responsiveness
                # TODO, does this make sense, or better to assess another way?
                ss.BoolState('nsaid_responder', default=ss.bernoulli(p=self.pars.efficacy['nsaid'])),
                ss.BoolState('txa_responder', default=ss.bernoulli(p=self.pars.efficacy['txa'])),
                ss.BoolState('pill_responder', default=ss.bernoulli(p=self.pars.efficacy['pill'])),
                ss.BoolState('hiud_responder', default=ss.bernoulli(p=self.pars.efficacy['hiud'])),

                # Current treatment status
                ss.FloatArr('current_treatment', default=0), # 0=none, 1=nsaid, 2=txa, 3=pill, 4=hiud
                ss.BoolState('on_treatment'),
                ss.FloatArr('ti_start_treatment'),
                ss.FloatArr('dur_treatment'),
                ss.FloatArr('ti_stop_treatment'),

                # Treatment outcomes
                ss.BoolState('treatment_effective'),
                ss.BoolState('treatment_assessed'),
                ss.FloatArr('assessed_treatment', default=0), # Store which treatment was assessed
                ss.BoolState('adherent'),
                
                ss.BoolState('nsaid_offered'),
                ss.BoolState('nsaid_accepted'),

                ss.BoolState('txa_offered'),
                ss.BoolState('txa_accepted'),

                ss.BoolState('pill_offered'),
                ss.BoolState('pill_accepted'),

                ss.BoolState('hiud_offered'),
                ss.BoolState('hiud_accepted'),
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

        def _calc_individualized_prob(self, uids, base_prob):
            """
            Calculate individualized probabilities incorporating care_seeking_propensity.

            Uses logistic transformation to map base probability and individual
            care-seeking propensity to a valid probability in [0,1].

            Args:
                uids: Array of UIDs
                base_prob: Base probability (scalar between 0 and 1)

            Returns:
                Array of individualized probabilities for each UID
            """
            # Use logistic approach with care_seeking_propensity as intercept_scale
            pars = sc.objdict(base=base_prob)
            p = logistic(self, uids, pars, intercept_scale=self.care_seeking_propensity[uids])
            return p
    
        @property
        def pill_idx(self):
            """Get pill method index"""
            return self.sim.connectors.contraception.get_method_by_label('Pill').idx
    
        @property
        def iud_idx(self):
            """Get IUD method index"""
            return self.sim.connectors.contraception.get_method_by_label('IUDs').idx
    
        @property
        def pill(self):
            """Get pill users"""
            return self.sim.people.fp.method == self.pill_idx
    
        @property
        def iud(self):
            """Get IUD users"""
            return self.sim.people.fp.method == self.iud_idx

        @property
        def is_eligible(self):
            # Default eligibility: menstruating, non-pregnant women experiencing HMB
            ppl = self.sim.people
            is_el = (ppl.menstruation.hmb &  # Currently experiencing HMB
                    ppl.menstruation.menstruating &
                    ~ppl.fp.pregnant &
                    ~ppl.fp.postpartum
                    )
            return is_el

        # Shortcut to states from menstruation module
        @property
        def anemic(self):
            mens = self.sim.people.menstruation
            return mens.anemic

        @property
        def pain(self):
            mens = self.sim.people.menstruation
            return mens.pain

        @property
        def tried_all(self):
            """ Tried all available treatments """
            return self.tried_nsaid & self.tried_txa & self.tried_pill & self.tried_hiud

        def set_states(self):
            uids = ss.uids(self.hiud_prone.isnan)
            self.hiud_prone[uids] = self.pars.p_hiud.rvs(uids)
            self.hiud[:] = self.hiud_prone & self.iud
            return

        def step(self):
            """
            Execute cascade at each timestep.

            Order of operations:
            1. Stop treatment for those whose ti_stop_treatment == ti
            2. Determine who seeks care (reset each timestep)
            3. Offer treatments to care-seekers (parallel evaluation)
            4. Assess effectiveness for those past time_to_assess
            5. Check adherence for those on treatment

            Note: Treatments are stopped first to allow individuals whose
            treatment just ended to immediately re-enter care-seeking if
            their HMB persists.
            """
            # Only run if intervention has started
            if self.sim.t.now() < self.pars.year:
                return
            self.set_states()

            # 1. Figure out who's stopping treatment
            self.stop_treatment()

            # 2. Determine who seeks care
            self.determine_care_seeking()
        
            # 3. Offer next treatment in cascade - this also triggers treatment initiation and duration
            self.offer_treatment()
        
            # 4. Assess treatment effectiveness
            self.assess_treatment_effectiveness()
        
            # 5. Check adherence
            self.check_adherence()
        
            return
    
        def determine_care_seeking(self, uids=None):
            """
            Determine who seeks care for HMB this timestep using logistic regression.

            Care-seeking probability is influenced by:
            - Baseline odds (50%)
            - Anemia status (increases odds by factor exp(1) ≈ 2.7)
            - Pain status (increases odds by factor exp(0.25) ≈ 1.3)
            - Individual propensity (normally distributed multiplicative factor)

            Following the pathway: HMB → Seeking care?

            Args:
                uids: UIDs to check. If None, uses eligible individuals not on treatment.

            Updates:
                seeking_care: Set to True for those who seek care this timestep
            """
            self.seeking_care[:] = False
            if uids is None: uids = (self.is_eligible & ~self.on_treatment).uids

            # Main logic
            p_care = logistic(self, uids, self.pars.care_behavior, intercept_scale=self.care_seeking_propensity[uids])
            self._p_care.set(0)
            self._p_care.set(p_care)
            seeks_care = self._p_care.filter(uids)
            self.seeking_care[seeks_care] = True

            return
    
        def offer_treatment(self, uids=None):
            """
            Offer treatments in cascade: NSAID → TXA → Pill → hIUD.

            All treatment options are evaluated in parallel for each individual.
            Individuals can only be on one treatment at a time, so pill/hIUD
            checks also verify not already on treatment.

            Cascade logic:
            - NSAID: Available to all seeking care who haven't tried it
            - TXA: Available to all seeking care who haven't tried it
            - Pill: Available if no fertility intent AND not on treatment
            - hIUD: Available if no fertility intent AND not on treatment

            Note: Treatments are marked as "tried" regardless of whether
            they are offered or accepted, to prevent repeated attempts.

            Args:
                uids: UIDs eligible for treatment. If None, uses all seeking care.

            Updates:
                tried_*: Marks treatments as attempted
                *_offered: Tracks which treatments were offered
                *_accepted: Tracks which treatments were accepted
                on_treatment: Set via _start_treatment for accepted treatments
            """
            # Treatment can be offered to those seeking care
            if uids is None: uids = self.seeking_care.uids
        
            if len(uids) == 0:
                return
        
            # Get fertility intent from FPsim
            fertility_intent = self.sim.people.fp.fertility_intent

            # NSAID node - first line treatment
            can_try_nsaid = uids & ~self.tried_nsaid & ~self.on_treatment
            self._offer_and_start(can_try_nsaid, 'nsaid')

            # TXA node - only if tried NSAID
            can_try_txa = uids & ~self.tried_txa & ~self.on_treatment & self.tried_nsaid
            self._offer_and_start(can_try_txa, 'txa')

            # Pill node - only if tried NSAID and TXA, and no fertility intent
            can_try_pill = uids & ~self.tried_pill & ~self.on_treatment & self.tried_nsaid & self.tried_txa & ~fertility_intent
            self._offer_and_start(can_try_pill, 'pill')

            # hIUD node - only if tried NSAID, TXA, and Pill, and no fertility intent
            can_try_hiud = uids & ~self.tried_hiud & ~self.on_treatment & self.tried_nsaid & self.tried_txa & self.tried_pill & ~fertility_intent
            self._offer_and_start(can_try_hiud, 'hiud')

            # Exhausted or blocked → exit care pathway permanently?
            # exiters = self.tried_all[uids] & ~self.on_treatment[uids] 
            # self.care_seeking_propensity[exiters] = 0

            return
    
        def _offer_and_start(self, uids, treatment_type):
            """
            Offer treatment to individuals and start for those who accept.

            Process:
            1. Randomly select who is offered based on prob_offer
            2. Randomly select who accepts from those offered based on prob_accept
               (incorporating care_seeking_propensity)
            3. Mark acceptors as "tried" (was previously marking ALL as tried)
            4. Start treatment for those who accept

            Args:
                uids: Array of UIDs eligible for this treatment
                treatment_type: One of 'nsaid', 'txa', 'pill', 'hiud'

            Updates:
                tried_{treatment_type}: Only for acceptors (FIXED: was marking all)
                {treatment_type}_offered: Subset who were offered
                {treatment_type}_accepted: Subset who accepted
                Plus all states updated by _start_treatment for acceptors
            """
            # See who's offered the treatment
            offered = self.pars.prob_offer[treatment_type].filter(uids)
            if len(offered):
                getattr(self, f"{treatment_type}_offered")[offered] = True

            # Calculate individualized acceptance probabilities incorporating care_seeking_propensity
            if len(offered):
                base_accept_prob = self.pars.prob_accept[treatment_type].pars['p']
                p_accept = self._calc_individualized_prob(offered, base_accept_prob)

                # Use distribution defined in __init__ with individualized probabilities
                self._p_accept.set(0)
                self._p_accept.set(p_accept)
                accepted = self._p_accept.filter(offered)

                if len(accepted):
                    getattr(self, f"{treatment_type}_accepted")[accepted] = True
                    # Mark as tried ONLY for those who accepted (FIXED: was marking all UIDs as tried)
                    getattr(self, f"tried_{treatment_type}")[accepted] = True
                    self._start_treatment(accepted, treatment_type)  # this sets on_treatment and module flags

            return

        def _start_treatment(self, uids, tx):
            """
            Start treatment for individuals who accepted.

            Updates both intervention tracking states AND menstruation module states.
            Only treatment responders have their menstruation module states updated,
            implementing heterogeneous treatment response.

            Treatment-specific logic:
            - NSAID/TXA:
              * Only responders get menstruation module state set
              * Duration drawn from uniform distribution (10-14 months)
              * Stop time calculated as ti + duration
            - Pill/hIUD:
              * Set as contraceptive method via FPsim
              * Duration and efficacy managed by FPsim contraception connector
              * Stop time set based on FPsim-assigned duration

            Args:
                uids: Array of UIDs who accepted treatment
                tx: Treatment type ('nsaid', 'txa', 'pill', 'hiud')

            Updates:
                current_treatment: Sets to treatment code (1-4)
                on_treatment: Set to True
                ti_start_treatment: Records start time
                treatment_assessed: Reset to False
                dur_treatment: Treatment duration
                ti_stop_treatment: Scheduled stop time

                For responders only:
                    menstruation.{tx}: Set to True (NSAID/TXA)

                For pill/hIUD:
                    fp.method: Set to pill/IUD index
                    fp.on_contra: Set to True
                    fp.ever_used_contra: Set to True
                    fp.ti_contra: Set to stop time

            Note: tried_{tx} is now marked in _offer_and_start, not here
            """
            # Set current treatment tracking
            self.current_treatment[uids] = self.treatment_map[tx]
            self.on_treatment[uids] = True
            self.ti_start_treatment[uids] = self.ti
            self.treatment_assessed[uids] = False
        
            # For each treatment, first pull out the responders. We only adjust the
            # states within the menstruation module for responders.
            if tx == 'nsaid':
                nsaid_responders = uids & self.nsaid_responder
                self.nsaid[nsaid_responders] = True
                self.dur_treatment[uids] = self.pars.dur_treatment[tx].rvs(uids)
                self.ti_stop_treatment[uids] = self.ti + self.dur_treatment[uids]

            elif tx == 'txa':
                txa_responders = uids & self.txa_responder
                self.txa[txa_responders] = True
                self.dur_treatment[uids] = self.pars.dur_treatment[tx].rvs(uids)
                self.ti_stop_treatment[uids] = self.ti + self.dur_treatment[uids]
            
            elif tx in ['pill', 'hiud']:
                # Set as contraceptive method
                method_idx = self.pill_idx if tx == 'pill' else self.iud_idx

                self.sim.people.fp.method[uids] = method_idx
                self.sim.people.fp.on_contra[uids] = True
                self.sim.people.fp.ever_used_contra[uids] = True

                # Set method duration
                method_dur = self.sim.connectors.contraception.set_dur_method(uids)
                self.sim.people.fp.ti_contra[uids] = self.ti + method_dur
                self.dur_treatment[uids] = method_dur
                self.ti_stop_treatment[uids] = self.ti + method_dur

            return
    
        def assess_treatment_effectiveness(self):
            """
            Assess treatment effectiveness after time_to_assess period (default: 3 months).

            Treatment responder status determines the biological effect: responders have
            their HMB suppressed, non-responders do not benefit.

            Process:
            1. Apply treatment effects: remove HMB from responders currently on treatment
            2. Find individuals on treatment who haven't been assessed yet
            3. Check if time_to_assess has elapsed since treatment start
            4. Check current HMB status
            5. If HMB resolved: mark as effective, continue to adherence check
            6. If HMB persists: mark as ineffective, schedule stop for next timestep

            Updates:
                menstruation.hmb: Set to False for treatment responders
                treatment_assessed: Marks individuals as assessed
                treatment_effective: True if HMB resolved, False otherwise
                ti_stop_treatment: Set to ti+1 for ineffective treatments

            Note: Failed treatments are automatically stopped at the next timestep,
            allowing individuals to try the next option in the cascade.
            """

            # Apply treatment effects: remove HMB from responders currently on treatment
            mens = self.sim.people.menstruation

            # NSAID responders
            on_nsaid = self.current_treatment == self.treatment_map['nsaid']
            nsaid_responders = on_nsaid & self.nsaid_responder
            mens.hmb[nsaid_responders.uids] = False

            # TXA responders
            on_txa = self.current_treatment == self.treatment_map['txa']
            txa_responders = on_txa & self.txa_responder
            mens.hmb[txa_responders.uids] = False

            # Pill responders
            on_pill = self.current_treatment == self.treatment_map['pill']
            pill_responders = on_pill & self.pill_responder
            mens.hmb[pill_responders.uids] = False

            # hIUD responders
            on_hiud = self.current_treatment == self.treatment_map['hiud']
            hiud_responders = on_hiud & self.hiud_responder
            mens.hmb[hiud_responders.uids] = False

            # Find those ready to assess
            on_treatment_uids = (self.on_treatment & ~self.treatment_assessed).uids

            if len(on_treatment_uids) == 0:
                return

            # Check if enough time has passed to assess them
            time_on_treatment = self.ti - self.ti_start_treatment[on_treatment_uids]
            ready_to_assess = on_treatment_uids[time_on_treatment >= self.pars.time_to_assess]
            self.treatment_assessed[ready_to_assess] = True
            # Store which treatment was assessed (before it gets reset to NaN when stopped)
            self.assessed_treatment[ready_to_assess] = self.current_treatment[ready_to_assess]

            if len(ready_to_assess) == 0:
                return

            # Assess whether treatment has helped HMB
            has_hmb = ready_to_assess & mens.hmb
            no_hmb = ready_to_assess & ~mens.hmb
            self.treatment_effective[no_hmb] = True
            self.treatment_effective[has_hmb] = False
            self.ti_stop_treatment[has_hmb] = self.ti + 1  # Stop next time step

            return
    
        def check_adherence(self, uids=None):
            """
            Check adherence for individuals currently on treatment.
            Adherence probabilities are treatment-specific and incorporate
            care_seeking_propensity.

            Process:
            1. Get adherence probability for each individual's current treatment
            2. Adjust probability based on care_seeking_propensity
            3. Randomly determine adherence based on these individualized probabilities
            4. Non-adherent individuals scheduled to stop at next timestep

            Following pathway: Treatment effective? → Does she adhere? → Yes/No

            Args:
                uids: UIDs to check. If None, uses all individuals on treatment.

            Updates:
                adherent: Set to True for adherent individuals
                ti_stop_treatment: Set to ti+1 for non-adherent individuals

            Note: Adherence is only checked for those on treatment, regardless
            of whether treatment has been assessed as effective yet.
            """
            # Check adherence for those who are on treatment
            if uids is None: uids = self.on_treatment.uids

            if len(uids) == 0:
                return

            # Calculate individualized adherence probability for each treatment type
            # Loop through each treatment type to handle them separately
            all_adherent = []

            for tx_name, tx_code in self.treatment_map.items():
                if tx_name == 'none':
                    continue

                # Find people on this treatment
                on_this_tx = uids & (self.current_treatment == tx_code)
                if len(on_this_tx) == 0:
                    continue

                # Get base adherence probability for this treatment
                base_adherence = self.pars.adherence[tx_name]

                # Calculate individualized probabilities incorporating care_seeking_propensity
                p_adherent = self._calc_individualized_prob(on_this_tx, base_adherence)

                # Use distribution defined in __init__ with individualized probabilities
                self._p_adherent.set(0)
                self._p_adherent.set(p_adherent)
                is_adherent = self._p_adherent.filter(on_this_tx)
                self.adherent[is_adherent] = True
                all_adherent.append(is_adherent)

            # Non-adherent - stop
            stoppers = uids & ~self.adherent
            self.ti_stop_treatment[stoppers] = self.ti + 1

            return
    
        def stop_treatment(self, uids=None, success=False):
            """
            Stop treatment for individuals whose ti_stop_treatment equals current time.

            This is called at the beginning of each step to stop treatments that have
            reached their scheduled end time (either from duration completion,
            ineffectiveness, or non-adherence).

            Process:
            1. Identify individuals whose ti_stop_treatment == current ti
            2. Reset menstruation module states for NSAID/TXA
            3. Reset all intervention tracking states

            Args:
                uids: UIDs to stop treatment for. If None, finds all with ti_stop_treatment==ti
                success: Unused parameter (kept for compatibility)

            Updates:
                menstruation.nsaid: Set to False for NSAID stoppers
                menstruation.txa: Set to False for TXA stoppers
                current_treatment: Reset to 0
                on_treatment: Set to False
                dur_treatment: Set to NaN
                treatment_effective: Reset to False
                treatment_assessed: Reset to False

            Note: Pill/hIUD contraceptive states are not reset here as they are
            managed by FPsim. Individuals remain on contraception for contraceptive
            purposes even if HMB treatment has ended.
            """
            if uids is None: uids = (self.ti_stop_treatment == self.ti).uids
            if len(uids) is None:
                return

            # Reset menstruation module treatment states for NSAID and TXA. Don't need to 
            # do this for pill or IUD because FPsim handles them.
            nsaid_idx = self.treatment_map['nsaid']
            nsaid_stoppers = uids & (self.current_treatment == nsaid_idx)
            self.nsaid[nsaid_stoppers] = False
            txa_idx = self.treatment_map['txa']
            txa_stoppers = uids & (self.current_treatment == txa_idx)
            self.txa[txa_stoppers] = False
                    
            # Reset tracking states
            self.current_treatment[uids] = np.nan
            self.on_treatment[uids] = False
            self.dur_treatment[uids] = np.nan
            self.treatment_effective[uids] = False
            self.treatment_assessed[uids] = False

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
                
            # # No-care prevalence among HMB cases (diagram branch: Seeking care? -> No)
            # if np.count_nonzero(hmb_cases) > 0:
            #     self.results.no_care_prev[ti] = np.count_nonzero(self.did_not_seek_care & hmb_cases) / np.count_nonzero(hmb_cases)
            # else:
            #     self.results.no_care_prev[ti] = 0 
        
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

                
            




















