"""
Modular HMB treatment interventions

Each treatment (NSAID, TXA, Pill, hIUD) is implemented as a standalone
intervention that can be used independently or combined in a cascade.

This architecture allows:
- Component-specific impact analysis
- Flexible intervention configurations
- Better testing and maintainability

The HMBCascade orchestrator coordinates sequential offering of treatments
using Starsim's eligibility system without hard-coding dependencies in
individual treatment classes.
"""

import numpy as np
import starsim as ss
import sciris as sc
from utils import logistic


# ============================================================================
# Base treatment class
# ============================================================================

class HMBTreatmentBase(ss.Intervention):
    """
    Base class for HMB treatments.

    Provides common functionality for care-seeking, treatment response,
    effectiveness assessment, and adherence checking.
    """

    def __init__(self, name, pars=None, eligibility=None, **kwargs):
        super().__init__(name=name, eligibility=eligibility)

        # Store probabilities calculated within the module
        self._p_care = ss.bernoulli(p=0)
        self._p_accept = ss.bernoulli(p=0)
        self._p_adherent = ss.bernoulli(p=0)
        self._p_discontinue = ss.bernoulli(p=0)

        # Handle parameters
        self.define_pars(
            care_seeking_dist=ss.normal(1, 1),  # Default distribution
            use_at_will=False,  # Whether treatment is use-at-will (NSAIDs/TXA) vs continuous (Pill/hIUD)
            p_discontinue_nonadherent=0.1,  # Probability of discontinuation per timestep when non-adherent (for use-at-will)
        )
        self.update_pars(pars, **kwargs)

        return 

    def _define_common_states(self):
        """
        Define states common to all HMB treatments.

        Call this in subclass __init__ after define_pars() so that
        self.pars.efficacy is available for the responder state.
        """
        self.define_states(
            # Care seeking
            ss.BoolState('seeking_care'),
            ss.FloatArr('care_seeking_propensity', default=self.pars.care_seeking_dist),

            # Treatment status
            ss.BoolState('on_treatment'),
            ss.BoolState('tried_treatment'),
            ss.BoolState('offered'),
            ss.BoolState('accepted'),
            ss.FloatArr('accept_propensity', default=float('nan')),  # drawn once at first offer, nan = never offered

            # Treatment response
            ss.BoolState('responder', default=ss.bernoulli(p=self.pars.efficacy)),
            ss.BoolState('treatment_effective'),
            ss.BoolState('treatment_assessed'),
            ss.BoolState('adherent'),
            ss.BoolState('was_effective', default=False),  # Track if treatment was ever effective (persists after stopping)

            # Timing
            ss.FloatArr('ti_start_treatment'),
            ss.FloatArr('dur_treatment'),
            ss.FloatArr('ti_stop_treatment'),
            ss.FloatArr('ti_nonadherent'),  # Track when non-adherence started (for gradual discontinuation)
        )

    def _calc_individualized_prob(self, uids, base_prob):
        """
        Calculate individualized probabilities incorporating care_seeking_propensity.

        Args:
            uids: Array of UIDs
            base_prob: Base probability (scalar between 0 and 1)

        Returns:
            Array of individualized probabilities for each UID
        """
        pars = sc.objdict(base=base_prob)
        p = logistic(self, uids, pars, intercept_scale=self.care_seeking_propensity[uids])
        return p

    def determine_care_seeking(self, uids=None):
        """
        Determine who seeks care using logistic regression.

        Args:
            uids: UIDs to check. If None, uses eligible individuals not on treatment.
        """
        self.seeking_care[:] = False
        if uids is None:
            # Eligible: menstruating, non-pregnant HMB women not currently on this treatment
            ppl = self.sim.people
            eligible = (ppl.menstruation.hmb &
                       ppl.menstruation.menstruating &
                       ~ppl.fp.pregnant &
                       ~ppl.fp.postpartum &
                       ~self.on_treatment)
            uids = eligible.uids

        if len(uids) == 0:
            return

        # Calculate care-seeking probability
        p_care = logistic(self, uids, self.pars.care_behavior,
                         intercept_scale=self.care_seeking_propensity[uids])
        self._p_care.set(0)
        self._p_care.set(p_care)
        seeks_care = self._p_care.filter(uids)
        self.seeking_care[seeks_care] = True

        return

    def assess_treatment_effectiveness(self):
        """
        Assess whether treatment has resolved HMB after time_to_assess.

        This method is identical across all treatments:
        - Apply treatment effect for responders
        - Check if enough time has passed
        - Assess HMB status
        - Stop ineffective treatment
        """
        # Apply treatment effect for responders currently on treatment
        on_treatment_responders = self.on_treatment & self.responder
        self.sim.people.menstruation.hmb[on_treatment_responders.uids] = False

        # Find those ready to assess
        on_treatment_uids = (self.on_treatment & ~self.treatment_assessed).uids
        if len(on_treatment_uids) == 0:
            return

        # Check if enough time has passed
        time_on_treatment = self.ti - self.ti_start_treatment[on_treatment_uids]
        ready_to_assess = on_treatment_uids[time_on_treatment >= self.pars.time_to_assess]
        self.treatment_assessed[ready_to_assess] = True

        if len(ready_to_assess) == 0:
            return

        # Assess HMB status
        hmb = self.sim.people.menstruation.hmb
        has_hmb = ready_to_assess & hmb
        no_hmb = ready_to_assess & ~hmb

        self.treatment_effective[no_hmb] = True
        self.was_effective[no_hmb] = True  # Mark as having been effective (persists after stopping)
        self.treatment_effective[has_hmb] = False
        self.ti_stop_treatment[has_hmb] = self.ti + 1  # Stop ineffective treatment

    def check_adherence(self):
        """
        Check adherence for those on treatment.

        Behavior depends on treatment type:
        - Use-at-will (NSAIDs/TXA): Probabilistic discontinuation when non-adherent
        - Continuous (Pill/hIUD): Immediate cessation when non-adherent
        """
        on_treatment_uids = self.on_treatment.uids
        if len(on_treatment_uids) == 0:
            return

        # Calculate individualized adherence probability
        p_adherent = self._calc_individualized_prob(on_treatment_uids, self.pars.adherence)

        self._p_adherent.set(0)
        self._p_adherent.set(p_adherent)
        is_adherent = self._p_adherent.filter(on_treatment_uids)
        self.adherent[is_adherent] = True

        # Get non-adherent individuals
        nonadherent_uids = on_treatment_uids & ~self.adherent

        if len(nonadherent_uids) == 0:
            return

        if self.pars.use_at_will:
            # Use-at-will treatments: probabilistic discontinuation over time
            # Mark when non-adherence started
            newly_nonadherent = nonadherent_uids & np.isnan(self.ti_nonadherent[nonadherent_uids])
            self.ti_nonadherent[newly_nonadherent] = self.ti

            # Apply discontinuation probability each timestep
            self._p_discontinue.set(0)
            self._p_discontinue.set(self.pars.p_discontinue_nonadherent)
            discontinue = self._p_discontinue.filter(nonadherent_uids)
            self.ti_stop_treatment[discontinue] = self.ti + 1
        else:
            # Continuous treatments: immediate cessation
            self.ti_stop_treatment[nonadherent_uids] = self.ti + 1

    def stop_treatment(self):
        """
        Stop treatment for those whose ti_stop_treatment equals current time.

        This method is identical across all treatments, though some treatments
        (Pill/hIUD) may need to preserve FPsim state for continued contraceptive use.
        """
        stoppers = (self.ti_stop_treatment == self.ti).uids
        if len(stoppers) == 0:
            return

        # Reset tracking states
        self.on_treatment[stoppers] = False
        self.dur_treatment[stoppers] = np.nan
        self.treatment_effective[stoppers] = False
        self.treatment_assessed[stoppers] = False
        self.adherent[stoppers] = False
        self.ti_nonadherent[stoppers] = np.nan

    @property
    def anemic(self):
        """Return anemia state from menstruation module."""
        return self.sim.people.menstruation.anemic

    @property
    def pain(self):
        """Return pain state from menstruation module."""
        return self.sim.people.menstruation.pain

    def step(self):
        """
        Execute intervention at each timestep.

        This method follows a standard pattern across all treatments.
        Subclasses can override _pre_step_hook() for treatment-specific setup.
        """
        if self.sim.t.now() < self.pars.year:
            return

        self._pre_step_hook()
        self.stop_treatment()
        self.determine_care_seeking()
        self.offer_treatment()
        self.assess_treatment_effectiveness()
        self.check_adherence()
        self.check_continuation()

    def _pre_step_hook(self):
        """
        Hook for subclass-specific pre-step behavior.

        Override this method in subclasses if you need to execute code
        before the standard step sequence (e.g., hIUD's set_states()).
        """
        pass

    def offer_treatment(self):
        """
        Offer treatment to eligible care seekers.

        This method follows a standard pattern:
        1. Get care seekers
        2. Filter by eligibility (override _get_eligible_for_treatment for custom logic)
        3. Offer treatment (probabilistic)
        4. Accept treatment (Each person's accept_propensity is compared against
                             the treatment's prob_accept. If propensity < prob_accept, they accept.
        5. Start treatment

        Subclasses can override _get_eligible_for_treatment() to customize eligibility.
        """
        care_seekers = self.seeking_care.uids
        if len(care_seekers) == 0:
            return

        # Get eligible candidates (subclasses can customize via _get_eligible_for_treatment)
        can_try = self._get_eligible_for_treatment(care_seekers)

        # Offer treatment
        offered = self.pars.prob_offer.filter(can_try)
        if len(offered) > 0:
            self.offered[offered] = True

            # Calculate individualized acceptance probability
            # --- Initialize propensity for anyone who doesn't have one yet ---
            unset = np.isnan(self.accept_propensity[offered])
            if np.any(unset):
                unset_uids = offered[unset]
                self.accept_propensity[unset_uids] = np.random.uniform(
                    0, 1, len(unset_uids)
                )

            # --- Accept if propensity < prob_accept threshold ---
            threshold = self.pars.prob_accept.pars['p']
            accepts = self.accept_propensity[offered] < threshold
            accepted = offered[accepts]
            
            # base_accept_prob = self.pars.prob_accept.pars['p']
            # p_accept = self._calc_individualized_prob(offered, base_accept_prob)

            # self._p_accept.set(0)
            # self._p_accept.set(p_accept)
            # accepted = self._p_accept.filter(offered)

            if len(accepted) > 0:
                self.accepted[accepted] = True
                self.tried_treatment[accepted] = True
                self._start_treatment(accepted)

    def _get_eligible_for_treatment(self, care_seekers):
        """
        Determine who is eligible for treatment.

        Default implementation: eligible if:
        - Seeking care, AND
        - (Haven't tried this treatment OR treatment was previously effective), AND
        - Not currently on this treatment

        This allows women to continue with treatments that worked for them,
        rather than being forced to progress through the cascade.

        Override this method to add treatment-specific eligibility criteria
        (e.g., Pill/hIUD check fertility intent).

        Args:
            care_seekers: UIDs of people seeking care

        Returns:
            UIDs of people eligible to try this treatment
        """
        return care_seekers & (~self.tried_treatment | self.was_effective) & ~self.on_treatment

    def check_continuation(self):
        """
        Hook for cycle-by-cycle continuation checks.
    
        Default is no-op. Overridden by NSAID and TXA.
        Pill and hIUD do not use this (they follow FPsim duration).
        """
        pass

# ============================================================================
# Individual treatment classes
# ============================================================================

class NSAIDTreatment(HMBTreatmentBase):
    """
    NSAID treatment for HMB.

    NSAIDs reduce menstrual blood loss through prostaglandin inhibition.
    Taken during menstruation only.
    """

    def __init__(self, pars=None, eligibility=None, **kwargs):
        super().__init__(name='nsaid_treatment', eligibility=eligibility)

        self.define_pars(
            year=2020,

            # Care-seeking behavior
            care_behavior=sc.objdict(
                base=0.5,
                anemic=1,
                pain=0.25,
            ),
            care_seeking_dist = ss.normal(1, 1),

            # Treatment parameters
            efficacy=0.5,  # 50% responder rate
            adherence=0.7,
            prob_offer=ss.bernoulli(p=0.9),
            prob_accept=ss.bernoulli(p=0.7),

            # Timing
            time_to_assess=ss.months(3),
            dur_treatment=ss.uniform(ss.months(10), ss.months(14)),

            # NSAIDs are use-at-will (taken during menstruation only)
            use_at_will=True,
            p_discontinue_nonadherent=0.1,  # 10% chance per timestep of discontinuing when non-adherent
            
            # Continuation parameters
            p_continue_first_cycle=0.6,    # No history yet
            p_continue_if_resolved=0.9,    # HMB resolved last cycle
            p_continue_if_persists=0.2,    # HMB persisted last cycle

        )

        self.update_pars(pars, **kwargs)

        # Define common states from base class
        self._define_common_states()
        
        #State for cycle-by-cycle continuation tracking
        self.define_states(
            ss.FloatArr('hmb_last_cycle', default=float('nan')),
        )

    def _start_treatment(self, uids):
        """Start NSAID treatment for accepted individuals."""
        # Set treatment status
        self.on_treatment[uids] = True
        self.ti_start_treatment[uids] = self.ti
        self.treatment_assessed[uids] = False

        # Set duration and stop time
        self.dur_treatment[uids] = self.pars.dur_treatment.rvs(uids)
        self.ti_stop_treatment[uids] = self.ti + self.dur_treatment[uids]
        
    def check_continuation(self):
        """
        Cycle-by-cycle continuation check for NSAIDs.

        Each cycle, women on treatment decide whether to continue based on
        whether HMB was resolved last cycle:
          - First cycle (no history): p_continue_first_cycle
          - HMB resolved last cycle:  p_continue_if_resolved
          - HMB persisted last cycle: p_continue_if_persists

        Those who discontinue have tried_treatment=True and become eligible
        for the next treatment in the cascade via care-seeking
        """
        on_treatment_uids = self.on_treatment.uids
        if len(on_treatment_uids) == 0:
            return

        hmb_this_cycle = self.sim.people.menstruation.hmb[on_treatment_uids]
        last_cycle = self.hmb_last_cycle[on_treatment_uids]

        p_continue = np.where(
            np.isnan(last_cycle),
            self.pars.p_continue_first_cycle,
            np.where(
                last_cycle == 1.0,
                self.pars.p_continue_if_resolved,
                self.pars.p_continue_if_persists,
            )
        )

        continues = np.random.uniform(0, 1, len(on_treatment_uids)) < p_continue
        stops = on_treatment_uids[~continues]

        # Update last cycle tracking: 1.0 = resolved, 0.0 = persisted
        self.hmb_last_cycle[on_treatment_uids] = (~hmb_this_cycle).astype(float)

        if len(stops) > 0:
            self.ti_stop_treatment[stops] = self.ti + 1
            self.hmb_last_cycle[stops] = np.nan
        
        
class TXATreatment(HMBTreatmentBase):
    """
    Tranexamic acid (TXA) treatment for HMB.

    TXA reduces menstrual blood loss through antifibrinolytic action.
    Taken during menstruation only.
    """

    def __init__(self, pars=None, eligibility=None, **kwargs):
        super().__init__(name='txa_treatment', eligibility=eligibility)

        self.define_pars(
            year=2020,

            # Care-seeking behavior
            care_behavior=sc.objdict(
                base=0.5,
                anemic=1,
                pain=0.25,
            ),
            care_seeking_dist = ss.normal(1, 1),

            # Treatment parameters
            efficacy=0.6,  # 60% responder rate
            adherence=0.6,
            prob_offer=ss.bernoulli(p=0.9),
            prob_accept=ss.bernoulli(p=0.6),

            # Timing
            time_to_assess=ss.months(3),
            dur_treatment=ss.uniform(ss.months(10), ss.months(14)),

            # TXA is use-at-will (taken during menstruation only)
            use_at_will=True,
            p_discontinue_nonadherent=0.1,  # 10% chance per timestep of discontinuing when non-adherent
            
            # NEW: Continuation parameters
            p_continue_first_cycle=0.6,
            p_continue_if_resolved=0.9,
            p_continue_if_persists=0.2,

        )

        self.update_pars(pars, **kwargs)

        # Define common states from base class
        self._define_common_states()
        
        # NEW: State for continuation tracking
        self.define_states(
            ss.FloatArr('hmb_last_cycle', default=float('nan')),
            )


    def _start_treatment(self, uids):
        """Start TXA treatment."""
        self.on_treatment[uids] = True
        self.ti_start_treatment[uids] = self.ti
        self.treatment_assessed[uids] = False

        self.dur_treatment[uids] = self.pars.dur_treatment.rvs(uids)
        self.ti_stop_treatment[uids] = self.ti + self.dur_treatment[uids]

    def check_continuation(self):
        """
        Cycle-by-cycle continuation check for TXA.
        same logic as NSAIDTreatment.check_continuation().
        """
        on_treatment_uids = self.on_treatment.uids
        if len(on_treatment_uids) == 0:
            return

        hmb_this_cycle = self.sim.people.menstruation.hmb[on_treatment_uids]
        last_cycle = self.hmb_last_cycle[on_treatment_uids]

        p_continue = np.where(
            np.isnan(last_cycle),
            self.pars.p_continue_first_cycle,
            np.where(
                last_cycle == 1.0,
                self.pars.p_continue_if_resolved,
                self.pars.p_continue_if_persists,
            )
        )

        continues = np.random.uniform(0, 1, len(on_treatment_uids)) < p_continue
        stops = on_treatment_uids[~continues]

        self.hmb_last_cycle[on_treatment_uids] = (~hmb_this_cycle).astype(float)

        if len(stops) > 0:
            self.ti_stop_treatment[stops] = self.ti + 1
            self.hmb_last_cycle[stops] = np.nan



class PillTreatment(HMBTreatmentBase):
    """
    Combined oral contraceptive pill for HMB.

    Hormonal treatment that reduces menstrual blood loss.
    Integrates with FPsim for contraceptive management.
    """

    def __init__(self, pars=None, eligibility=None, **kwargs):
        super().__init__(name='pill_treatment', eligibility=eligibility)

        self.define_pars(
            year=2020,

            care_behavior=sc.objdict(
                base=0.5,
                anemic=1,
                pain=0.25,
            ),
            care_seeking_dist = ss.normal(1, 1),

            efficacy=0.7,
            adherence=0.75,
            prob_offer=ss.bernoulli(p=0.9),
            prob_accept=ss.bernoulli(p=0.5),

            time_to_assess=ss.months(3),
        )

        self.update_pars(pars, **kwargs)

        # Define common states from base class
        self._define_common_states()

    @property
    def pill_idx(self):
        """Get pill method index from FPsim."""
        return self.sim.connectors.contraception.get_method_by_label('Pill').idx

    def _get_eligible_for_treatment(self, care_seekers):
        """Pill requires no fertility intent (contraceptive)."""
        base_eligible = super()._get_eligible_for_treatment(care_seekers)
        fertility_intent = self.sim.people.fp.fertility_intent
        return base_eligible & ~fertility_intent

    def _start_treatment(self, uids):
        """Start pill treatment via FPsim."""
        self.on_treatment[uids] = True
        self.ti_start_treatment[uids] = self.ti
        self.treatment_assessed[uids] = False

        # Set as contraceptive method via FPsim
        self.sim.people.fp.method[uids] = self.pill_idx
        self.sim.people.fp.on_contra[uids] = True
        self.sim.people.fp.ever_used_contra[uids] = True

        # Get duration from FPsim
        method_dur = self.sim.connectors.contraception.set_dur_method(uids)
        self.sim.people.fp.ti_contra[uids] = self.ti + method_dur
        self.dur_treatment[uids] = method_dur
        self.ti_stop_treatment[uids] = self.ti + method_dur

        # Apply treatment effect for responders
        responders = uids & self.responder
        # Pill effect is handled through FPsim contraception, which affects HMB


class hIUDTreatment(HMBTreatmentBase):
    """
    Hormonal IUD treatment for HMB.

    Long-acting hormonal treatment that reduces menstrual blood loss.
    Integrates with FPsim for contraceptive management.
    """

    def __init__(self, pars=None, eligibility=None, **kwargs):
        super().__init__(name='hiud_treatment', eligibility=eligibility)

        self.define_pars(
            year=2020,

            care_behavior=sc.objdict(
                base=0.5,
                anemic=1,
                pain=0.25,
            ),
            care_seeking_dist = ss.normal(1, 1),

            efficacy=0.8,
            adherence=0.85,
            prob_offer=ss.bernoulli(p=0.9),
            prob_accept=ss.bernoulli(p=0.5),

            time_to_assess=ss.months(3),
            p_hiud=ss.bernoulli(p=0.17),  # Proportion of IUD users who get hormonal IUD
        )

        self.update_pars(pars, **kwargs)

        # Define common states from base class
        self._define_common_states()

        # Define hIUD-specific state
        self.define_states(
            ss.BoolState('hiud_prone', label="Prone to use hormonal IUD if using IUD"),
        )

    @property
    def iud_idx(self):
        """Get IUD method index from FPsim."""
        return self.sim.connectors.contraception.get_method_by_label('IUDs').idx

    def set_states(self):
        """Set hormonal IUD propensity."""
        uids = ss.uids(self.hiud_prone.isnan)
        self.hiud_prone[uids] = self.pars.p_hiud.rvs(uids)

    def _pre_step_hook(self):
        """Execute hIUD-specific state setting before standard step."""
        self.set_states()

    def _get_eligible_for_treatment(self, care_seekers):
        """hIUD requires no fertility intent (contraceptive)."""
        base_eligible = super()._get_eligible_for_treatment(care_seekers)
        fertility_intent = self.sim.people.fp.fertility_intent
        return base_eligible & ~fertility_intent

    def _start_treatment(self, uids):
        """Start hIUD treatment via FPsim."""
        self.on_treatment[uids] = True
        self.ti_start_treatment[uids] = self.ti
        self.treatment_assessed[uids] = False

        # Set as contraceptive method via FPsim
        self.sim.people.fp.method[uids] = self.iud_idx
        self.sim.people.fp.on_contra[uids] = True
        self.sim.people.fp.ever_used_contra[uids] = True

        # Get duration from FPsim
        method_dur = self.sim.connectors.contraception.set_dur_method(uids)
        self.sim.people.fp.ti_contra[uids] = self.ti + method_dur
        self.dur_treatment[uids] = method_dur
        self.ti_stop_treatment[uids] = self.ti + method_dur

        # Apply treatment effect for responders
        responders = uids & self.responder
        # hIUD effect is handled through FPsim contraception


# ============================================================================
# Treatment cascade orchestrator
# ============================================================================

class HMBCascade(ss.Intervention):
    """
    Orchestrates HMB treatment cascade.

    Uses eligibility functions to ensure proper sequencing:
    - NSAID: First-line treatment for all HMB
    - TXA: Eligible if NSAID was not offered, declined, or tried NSAID
    - Pill: Eligible if NSAID and TXA were each not offered, declined, or tried NSAID and TXA
    - hIUD: Eligible if NSAID, TXA, and Pill were each not offered, declined, or tried NSAID and TXA and Pill

    Each treatment can also be used independently for component analysis.
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__(name='hmb_cascade', **kwargs)

        self.define_pars(
            year=2020,
            time_to_assess=ss.months(3),

            # Treatment-specific parameters
            nsaid=sc.objdict(
                efficacy=0.5,
                adherence=0.7,
                prob_offer=ss.bernoulli(p=0.9),
                prob_accept=ss.bernoulli(p=0.7),
            ),
            txa=sc.objdict(
                efficacy=0.6,
                adherence=0.6,
                prob_offer=ss.bernoulli(p=0.9),
                prob_accept=ss.bernoulli(p=0.6),
            ),
            pill=sc.objdict(
                efficacy=0.7,
                adherence=0.75,
                prob_offer=ss.bernoulli(p=0.9),
                prob_accept=ss.bernoulli(p=0.5),
            ),
            hiud=sc.objdict(
                efficacy=0.8,
                adherence=0.85,
                prob_offer=ss.bernoulli(p=0.9),
                prob_accept=ss.bernoulli(p=0.5),
            ),

            # Care-seeking behavior (shared across treatments)
            care_behavior=sc.objdict(
                base=0.5,
                anemic=1,
                pain=0.25,
            ),
            care_seeking_dist = ss.normal(1, 1),
        )
        self.update_pars(pars, **kwargs)

        # Treatment components will be added during initialization
        self.treatments = {}
        return

    def init_pre(self, sim):
        """Initialize treatment components with eligibility functions."""
        super().init_pre(sim)

        # Create NSAID treatment (first-line, no prerequisites)
        nsaid = NSAIDTreatment(
            pars=dict(
                year=self.pars.year,
                time_to_assess=self.pars.time_to_assess,
                care_behavior=self.pars.care_behavior,
                efficacy=self.pars.nsaid.efficacy,
                adherence=self.pars.nsaid.adherence,
                prob_offer=self.pars.nsaid.prob_offer,
                prob_accept=self.pars.nsaid.prob_accept,
                care_seeking_dist=self.pars.care_seeking_dist,
            ),
        )

        # Create TXA treatment (requires tried NSAID)
        def txa_eligibility(sim):
            # Eligible if tried NSAID or offered but refused 
            return nsaid.tried_treatment | (nsaid.offered & ~nsaid.on_treatment) 

        txa = TXATreatment(
            pars=dict(
                year=self.pars.year,
                time_to_assess=self.pars.time_to_assess,
                care_behavior=self.pars.care_behavior,
                efficacy=self.pars.txa.efficacy,
                adherence=self.pars.txa.adherence,
                prob_offer=self.pars.txa.prob_offer,
                prob_accept=self.pars.txa.prob_accept,
                care_seeking_dist=self.pars.care_seeking_dist,
            ),
            eligibility=txa_eligibility,
        )

        # Create pill treatment (requires tried NSAID and TXA)      
        def pill_eligibility(sim):
            # Eligible if tried TXA OR offered but refused/didn't work OR tried NSAID but not continuing and not offered TXA
            return txa.tried_treatment | (txa.offered & ~txa.on_treatment) | (nsaid.tried_treatment & ~nsaid.on_treatment & ~txa.offered)

        pill = PillTreatment(
            pars=dict(
                year=self.pars.year,
                time_to_assess=self.pars.time_to_assess,
                care_behavior=self.pars.care_behavior,
                efficacy=self.pars.pill.efficacy,
                adherence=self.pars.pill.adherence,
                prob_offer=self.pars.pill.prob_offer,
                prob_accept=self.pars.pill.prob_accept,
                care_seeking_dist=self.pars.care_seeking_dist,                
            ),
            eligibility=pill_eligibility,
        )

        # Create hIUD treatment (requires tried all previous treatments)
        def hiud_eligibility(sim):      
            # Eligible if tried Pill OR offered but refused/didn't work OR tried NSAID but not continuing and not offered TXA 
            # OR tried TXA but not continuing and not offered Pill
            return pill.tried_treatment | (pill.offered & ~pill.on_treatment) | (nsaid.tried_treatment & ~nsaid.on_treatment & ~txa.offered) | (txa.tried_treatment & ~txa.on_treatment & ~pill.offered) 


        hiud = hIUDTreatment(
            pars=dict(
                year=self.pars.year,
                time_to_assess=self.pars.time_to_assess,
                care_behavior=self.pars.care_behavior,
                efficacy=self.pars.hiud.efficacy,
                adherence=self.pars.hiud.adherence,
                prob_offer=self.pars.hiud.prob_offer,
                prob_accept=self.pars.hiud.prob_accept,
                care_seeking_dist=self.pars.care_seeking_dist,
            ),
            eligibility=hiud_eligibility,
        )

        # Store treatments
        self.treatments = {
            'nsaid': nsaid,
            'txa': txa,
            'pill': pill,
            'hiud': hiud,
        }

        # Initialize each treatment
        for treatment in self.treatments.values():
            treatment.init_pre(sim)

    def init_post(self):
        """Complete initialization after sim is set up."""
        super().init_post()
        for treatment in self.treatments.values():
            treatment.init_post()

    def step(self):
        """
        Execute cascade at each timestep.

        Each treatment's eligibility function ensures proper sequencing.
        """
        for treatment in self.treatments.values():
            treatment.step()

    def finalize(self):
        """Finalize all treatments."""
        super().finalize()
        for treatment in self.treatments.values():
            treatment.finalize()

    # Convenience properties for accessing cascade metrics
    @property
    def tried_nsaid(self):
        return self.treatments['nsaid'].tried_treatment

    @property
    def tried_txa(self):
        return self.treatments['txa'].tried_treatment

    @property
    def tried_pill(self):
        return self.treatments['pill'].tried_treatment

    @property
    def tried_hiud(self):
        return self.treatments['hiud'].tried_treatment

    @property
    def on_any_treatment(self):
        """Boolean array of who is on any treatment."""
        on_any = self.treatments['nsaid'].on_treatment.copy()
        for tx_name in ['txa', 'pill', 'hiud']:
            on_any |= self.treatments[tx_name].on_treatment
        return on_any

    def get_cascade_depth(self):
        """
        Calculate number of treatments tried for each person.

        Returns:
            Array of integers (0-4) indicating cascade depth
        """
        depth = (
            self.tried_nsaid.astype(int) +
            self.tried_txa.astype(int) +
            self.tried_pill.astype(int) +
            self.tried_hiud.astype(int)
        )
        return depth


# ============================================================================
# Factory functions
# ============================================================================

def make_cascade_sim(seed=0, **cascade_kwargs):
    """
    Convenience function to create simulation with HMB cascade.

    Args:
        seed: Random seed
        **cascade_kwargs: Parameters to pass to HMBCascade

    Returns:
        Simulation object with cascade configured
    """
    import fpsim as fp
    from menstruation import Menstruation
    from education import Education
    from analyzers import track_hmb_anemia, track_cascade

    mens = Menstruation()
    edu = Education()
    cascade = HMBCascade(**cascade_kwargs)

    # Note: track_cascade will need updating to work with new architecture
    hmb_anemia_analyzer = track_hmb_anemia()

    sim = fp.Sim(
        start=2020,
        stop=2030,
        n_agents=5000,
        total_pop=55_000_000,
        location='kenya',
        education_module=edu,
        connectors=[mens],
        interventions=[cascade],
        analyzers=[hmb_anemia_analyzer],
        rand_seed=seed,
        verbose=0,
    )

    return sim


def make_component_sim(component, seed=0, **component_kwargs):
    """
    Create simulation with only one treatment component.

    Args:
        component: One of 'nsaid', 'txa', 'pill', 'hiud'
        seed: Random seed
        **component_kwargs: Parameters to pass to treatment

    Returns:
        Simulation object with single component configured
    """
    import fpsim as fp
    from menstruation import Menstruation
    from education import Education
    from analyzers import track_hmb_anemia

    mens = Menstruation()
    edu = Education()

    # Create the appropriate treatment
    treatment_classes = {
        'nsaid': NSAIDTreatment,
        'txa': TXATreatment,
        'pill': PillTreatment,
        'hiud': hIUDTreatment,
    }

    treatment = treatment_classes[component](**component_kwargs)
    hmb_anemia_analyzer = track_hmb_anemia()

    sim = fp.Sim(
        start=2020,
        stop=2030,
        n_agents=5000,
        total_pop=55_000_000,
        location='kenya',
        education_module=edu,
        connectors=[mens],
        interventions=[treatment],
        analyzers=[hmb_anemia_analyzer],
        rand_seed=seed,
        verbose=0,
    )

    return sim

