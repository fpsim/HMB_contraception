# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 09:49:40 2026

@author: navidehno

=============================================================================
FILE OVERVIEW
=============================================================================

This single file contains everything needed for the HMB treatment model:

  1. HMBTreatmentBase  — shared logic for all four treatments
  2. NSAIDTreatment, TXATreatment, PillTreatment, hIUDTreatment
  3. HMBPool           — pool-based treatment assignment 
  4. HMBCounterfactual — status-quo pool (no hIUD, runs from 2020)
  5. HMBCascade        — sequential cascade (optional, kept for comparison)
  6. Factory functions  — convenience sim builders

=============================================================================
CARE-SEEKING: 
=============================================================================

Layer 1 — Ever-seeker (one-time draw at HMB onset):
    Bernoulli(p_ever_seek).  Women who get False NEVER enter the care
    system.  This controls the "% who will ever seek care" (10–35%).

Layer 2 — Monthly arrival (memoryless Bernoulli each month):
    Among ever-seekers, a fresh draw each eligible month determines
    whether she shows up at a clinic.  Different rates for first vs
    repeat visits, and condition-specific rates for anemia/pain:

      First visit:
        Base:              p_monthly_first        (default 1/36 ≈ 3yr lag)
        With pain:         p_monthly_first_pain   (default 1/24 ≈ 2yr lag)
        With anemia:       p_monthly_first_anemic (default 1/12 ≈ 1yr lag)
        Both anemia+pain:  takes the faster rate (anemia)

      Repeat visits:       p_monthly_repeat       (default 1/6 ≈ 6mo lag)
        Always the same regardless of condition.

Layer 3 — Repeat cap (max 3 care-seeking episodes):
    After 3 episodes, the next re-entry attempt routes:
      90% → give up (permanently stop seeking)
      10% → hysterectomy (HMB removed, permanently stop seeking)

=============================================================================
ADHERENCE MODEL
=============================================================================

Adherence is a RETENTION mechanism, not a responder-rate modifier.
The chain is:
  efficacy → responder draw (at treatment start, unchanged)
  adherence → determines if she keeps taking it each month
  continuation → for use-at-will treatments, voluntary stop based on HMB

Behavior by treatment type:
  Use-at-will (NSAID, TXA):
    Non-adherent woman has 10% monthly probability of discontinuing.
    She may remain non-adherent for months before actually stopping.
  Continuous (Pill, hIUD):
    Non-adherent → immediate cessation next timestep.
    (Missing a daily pill or removing IUD ends the treatment.)

Adherence values (from Darcy):
  NSAID: 80%  |  TXA: 70%  |  Pill: 80%  |  hIUD: 100%

=============================================================================
POOL vs CASCADE
=============================================================================

Pool (primary):
  At each care-seeking visit, one treatment is randomly sampled from a
  weighted distribution (e.g., 50% NSAID, 25% TXA, 25% Pill).
  Women who tried-and-failed a treatment cannot be re-assigned it.
  Women who tried-and-it-worked CAN be re-assigned it.

Cascade (optional, for comparison):
  Sequential: NSAID → TXA → Pill → hIUD.
  Each tier requires failure/refusal of the previous tier.

=============================================================================
"""

import numpy as np
import starsim as ss
import sciris as sc


# ============================================================================
# SECTION 1: BASE TREATMENT CLASS
# ============================================================================
# This is the foundation that all four treatments inherit from.
# It provides:
#   - State definitions (seeking_care, on_treatment, responder, etc.)
#   - Three-layer care-seeking logic
#   - Treatment effectiveness assessment
#   - Adherence checking (retention model)
#   - Continuation logic (for use-at-will treatments)
#   - Two-phase stepping (for use inside orchestrators)
# ============================================================================

class HMBTreatmentBase(ss.Intervention):
    """
    Base class for all HMB treatments.

    Not used directly — subclassed by NSAIDTreatment, TXATreatment, etc.
    """

    def __init__(self, name, pars=None, eligibility=None, **kwargs):
        super().__init__(name=name, eligibility=eligibility)

        # ── Starsim distributions (reused each timestep) ──
        self._p_care = ss.bernoulli(p=0)        # Layer 2: monthly arrival draw
        self._p_accept = ss.bernoulli(p=0)
        self._p_continue = ss.bernoulli(p=0)     # Continuation check
        self._p_ever_seek = ss.bernoulli(p=0)    # Layer 1: ever-seeker draw
        self._p_adherent = ss.bernoulli(p=0)     # Adherence check
        self._p_discontinue = ss.bernoulli(p=0)  # Discontinuation for non-adherent

        # ── Default parameters (overridden by subclasses) ──
        self.define_pars(
            use_at_will=False,
            p_discontinue_nonadherent=0.10,  # 10% monthly chance of stopping when non-adherent
        )
        self.update_pars(pars, **kwargs)

        return

    # ------------------------------------------------------------------
    # State definitions
    # ------------------------------------------------------------------

    def _define_common_states(self):
        """
        Define states shared by all treatments.

        Called in each subclass __init__() AFTER define_pars(), so that
        self.pars.efficacy is available for the responder Bernoulli.

        Note: responder rate = efficacy ONLY (adherence is separate).
        """
        self.define_states(
            # ── Care seeking ──
            ss.BoolState('seeking_care'),          # Is she seeking care THIS month?
            ss.BoolState('ever_sought_care'),       # Has she EVER sought care?
            ss.BoolState('ever_seeker'),            # Layer 1: will she ever seek care?
            ss.BoolState('ever_seeker_assigned'),   # Has the Layer 1 draw been made?

            # ── Treatment status ──
            ss.BoolState('on_treatment'),           # Currently receiving this treatment?
            ss.BoolState('tried_treatment'),         # Has she ever tried this treatment?
            ss.BoolState('offered'),                 # Was this treatment ever offered?
            ss.BoolState('accepted'),                # Did she ever accept this treatment?
            ss.FloatArr('accept_propensity', default=ss.uniform()),  # Fixed U(0,1) for acceptance

            # ── Treatment response ──
            # Responder rate = efficacy only. Adherence is handled separately.
            ss.BoolState('responder', default=ss.bernoulli(p=self.pars.efficacy)),
            ss.BoolState('treatment_effective'),     # Did treatment resolve HMB?
            ss.BoolState('treatment_assessed'),      # Has 3-month assessment happened?
            ss.BoolState('was_effective', default=False),  # Ever effective (persists after stopping)

            # ── Adherence ──
            ss.BoolState('adherent'),               # Is she adherent this month?
            ss.FloatArr('ti_nonadherent'),          # When did non-adherence start?

            # ── Timing ──
            ss.FloatArr('ti_start_treatment'),
            ss.FloatArr('dur_treatment'),
            ss.FloatArr('ti_stop_treatment'),
        )

        # Use-at-will treatments track HMB status for continuation decisions
        if self.pars.use_at_will:
            self.define_states(
                ss.FloatArr('hmb_last_cycle', default=float('nan')),
            )

    # ------------------------------------------------------------------
    # Helper: find the orchestrator (pool, counterfactual, or cascade)
    # ------------------------------------------------------------------

    def _get_hmb_intervention(self):
        """
        Return the orchestrator intervention (pool/counterfactual/cascade)
        that this treatment belongs to, or None if standalone.

        This lets the treatment class reference shared states (ever_seeker,
        on_any_treatment, gave_up, had_hysterectomy) without hardcoding
        which orchestrator type it's inside.
        """
        for name in ['hmb_pool', 'hmb_counterfactual', 'hmb_cascade']:
            if hasattr(self.sim.interventions, name):
                return getattr(self.sim.interventions, name)
        return None

    # ------------------------------------------------------------------
    # Layer 1: Ever-seeker assignment (standalone mode only)
    # ------------------------------------------------------------------

    def _assign_ever_seekers_standalone(self):
        """
        Assign ever_seeker for newly-HMB women when running standalone.

        When this treatment is inside an orchestrator (pool/cascade),
        the orchestrator handles assignment centrally. This method only
        runs for standalone single-treatment sims.
        """
        if self._get_hmb_intervention() is not None:
            return  # Orchestrator handles it

        ppl = self.sim.people
        new_hmb = (ppl.menstruation.hmb &
                   ppl.menstruation.menstruating &
                   ~self.ever_seeker_assigned).uids

        if len(new_hmb) == 0:
            return

        self.ever_seeker_assigned[new_hmb] = True
        self._p_ever_seek.set(self.pars.care_behavior.p_ever_seek)
        assigned_true = self._p_ever_seek.filter(new_hmb)
        self.ever_seeker[assigned_true] = True

    # ------------------------------------------------------------------
    # Layer 2: Monthly arrival (memoryless process)
    # ------------------------------------------------------------------

    def determine_care_seeking(self, uids=None):
        """
        Determine who seeks care this month.

        Layer 1 (ever-seeker) is already assigned before this runs.
        This method implements Layer 2: among eligible ever-seekers,
        draw Bernoulli(p) where p depends on:
          - First visit vs repeat visit
          - Anemia/pain status (first visits only)

        First-visit rates (condition-specific):
          Base:    p_monthly_first          (default 1/36)
          Pain:    p_monthly_first_pain     (default 1/24)
          Anemia:  p_monthly_first_anemic   (default 1/12)
          Both:    takes the fastest (anemia)

        Repeat-visit rate:
          Always p_monthly_repeat (default 1/6), regardless of condition.
        """
        self.seeking_care[:] = False

        if uids is None:
            ppl = self.sim.people
            orch = self._get_hmb_intervention()

            # Use orchestrator-level states if available
            if orch is not None:
                on_any = orch.on_any_treatment
                is_ever_seeker = orch.ever_seeker
            else:
                on_any = self.on_treatment
                is_ever_seeker = self.ever_seeker

            # Build eligibility mask
            eligible = (is_ever_seeker &
                       ppl.menstruation.hmb &
                       ppl.menstruation.menstruating &
                       ~ppl.fp.pregnant &
                       ~ppl.fp.postpartum &
                       ~on_any)

            # Exclude women who gave up or had hysterectomy
            if orch is not None:
                eligible = eligible & ~orch.gave_up & ~orch.had_hysterectomy

            uids = eligible.uids

        if len(uids) == 0:
            return

        # ── Compute per-person monthly probability ──
        cb = self.pars.care_behavior

        # Start with base first-visit rate for everyone
        p_monthly = np.full(len(uids), cb.p_monthly_first, dtype=float)

        # Determine who is a first-time seeker vs repeat
        ever_sought = np.asarray(self.ever_sought_care[uids])
        first_time = ~ever_sought

        # For first-time seekers: apply condition-specific rates
        # Pain gives a faster rate than base
        pain_mask = first_time & np.asarray(self.sim.people.menstruation.pain[uids])
        p_monthly[pain_mask] = np.maximum(
            p_monthly[pain_mask], cb.p_monthly_first_pain)

        # Anemia gives an even faster rate (applied second so it wins over pain)
        anemic_mask = first_time & np.asarray(self.sim.people.menstruation.anemic[uids])
        p_monthly[anemic_mask] = np.maximum(
            p_monthly[anemic_mask], cb.p_monthly_first_anemic)

        # Repeat seekers: flat rate regardless of condition
        p_monthly[~first_time] = cb.p_monthly_repeat

        # Cap at 1.0
        np.minimum(p_monthly, 1.0, out=p_monthly)

        # ── Fresh Bernoulli draw (memoryless) ──
        self._p_care.set(p_monthly)
        seeks_care_uids = self._p_care.filter(uids)

        self.seeking_care[seeks_care_uids] = True
        self.ever_sought_care[seeks_care_uids] = True

        return

    # ------------------------------------------------------------------
    # Treatment effectiveness assessment
    # ------------------------------------------------------------------

    def assess_treatment_effectiveness(self):
        """
        Assess whether treatment resolved HMB after time_to_assess (3 months).

        For responders currently on treatment: set HMB = False.
        After 3 months, check if HMB is still resolved:
          - If resolved: mark treatment_effective and was_effective
          - If not: stop treatment (she re-enters care-seeking pool)
        """
        # Apply treatment effect for responders
        on_treatment_responders = self.on_treatment & self.responder
        self.sim.people.menstruation.hmb[on_treatment_responders.uids] = False

        # Find those ready to assess
        on_treatment_uids = (self.on_treatment & ~self.treatment_assessed).uids
        if len(on_treatment_uids) == 0:
            return

        time_on = self.ti - self.ti_start_treatment[on_treatment_uids]
        ready = on_treatment_uids[time_on >= self.pars.time_to_assess]
        self.treatment_assessed[ready] = True

        if len(ready) == 0:
            return

        hmb = self.sim.people.menstruation.hmb
        has_hmb = ready & hmb
        no_hmb = ready & ~hmb

        self.treatment_effective[no_hmb] = True
        self.was_effective[no_hmb] = True
        self.treatment_effective[has_hmb] = False
        self.ti_stop_treatment[has_hmb] = self.ti + 1  # Stop next timestep

    # ------------------------------------------------------------------
    # Adherence checking (retention model)
    # ------------------------------------------------------------------

    def check_adherence(self):
        """
        Check adherence for women currently on treatment.

        Each month, each woman on treatment draws Bernoulli(adherence).
        If non-adherent:
          Use-at-will (NSAID/TXA):
            10% monthly probability of actually discontinuing.
            She may stay non-adherent for months before stopping.
          Continuous (Pill/hIUD):
            Immediate cessation — stops next timestep.

        This is SEPARATE from efficacy. A responder who is non-adherent
        may stop a treatment that was working for her.
        """
        on_treatment_uids = self.on_treatment.uids
        if len(on_treatment_uids) == 0:
            return

        # Draw adherence for everyone on treatment
        adherence_rate = self.pars.get('adherence', 1.0)
        self._p_adherent.set(adherence_rate)
        is_adherent = self._p_adherent.filter(on_treatment_uids)
        self.adherent[:] = False
        self.adherent[is_adherent] = True

        # Get non-adherent individuals
        nonadherent = on_treatment_uids - is_adherent
        if len(nonadherent) == 0:
            return

        if self.pars.use_at_will:
            # Gradual: 10% monthly probability of discontinuing
            self._p_discontinue.set(self.pars.p_discontinue_nonadherent)
            discontinue = self._p_discontinue.filter(nonadherent)
            if len(discontinue) > 0:
                self.ti_stop_treatment[discontinue] = self.ti + 1
        else:
            # Immediate: stop next timestep
            self.ti_stop_treatment[nonadherent] = self.ti + 1

    # ------------------------------------------------------------------
    # Stop treatment
    # ------------------------------------------------------------------

    def stop_treatment(self):
        """Stop treatment for women whose ti_stop_treatment == current time."""
        stoppers = (self.ti_stop_treatment == self.ti).uids
        if len(stoppers) == 0:
            return

        self.on_treatment[stoppers] = False
        self.dur_treatment[stoppers] = np.nan
        self.treatment_effective[stoppers] = False
        self.treatment_assessed[stoppers] = False
        self.adherent[stoppers] = False
        self.ti_nonadherent[stoppers] = np.nan

    # ------------------------------------------------------------------
    # Continuation (use-at-will only)
    # ------------------------------------------------------------------

    def check_continuation(self):
        """
        Cycle-by-cycle voluntary continuation for use-at-will treatments.

        Each month, women on NSAID/TXA decide whether to continue:
          First cycle (no history):  60% continue
          HMB resolved last cycle:   90% continue
          HMB persisted last cycle:  20% continue

        Not used for Pill/hIUD — those follow FPsim duration.
        """
        if not self.pars.use_at_will:
            return

        on_treatment_uids = self.on_treatment.uids
        if len(on_treatment_uids) == 0:
            return

        hmb_this = self.sim.people.menstruation.hmb[on_treatment_uids]
        last = self.hmb_last_cycle[on_treatment_uids]

        p_continue = np.where(
            np.isnan(last),
            self.pars.p_continue_first_cycle,
            np.where(last == 1.0,
                     self.pars.p_continue_if_resolved,
                     self.pars.p_continue_if_persists))

        self._p_continue.set(p_continue)
        continuers = self._p_continue.filter(on_treatment_uids)
        stops = on_treatment_uids - continuers

        self.hmb_last_cycle[on_treatment_uids] = (~hmb_this).astype(float)

        if len(stops) > 0:
            self.ti_stop_treatment[stops] = self.ti + 1
            self.hmb_last_cycle[stops] = np.nan

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def anemic(self):
        return self.sim.people.menstruation.anemic

    @property
    def pain(self):
        return self.sim.people.menstruation.pain

    # ------------------------------------------------------------------
    # Two-phase stepping (used by orchestrators)
    # ------------------------------------------------------------------

    def step_seek(self):
        """Phase 1: housekeeping + care-seeking. Called by orchestrator."""
        if self.sim.t.now() < self.pars.year:
            return
        self._pre_step_hook()
        self.stop_treatment()
        self.determine_care_seeking()

    def step_treat(self):
        """Phase 2: offer/assess/adherence/continuation. Called by orchestrator."""
        if self.sim.t.now() < self.pars.year:
            return
        self.offer_treatment()
        self.assess_treatment_effectiveness()
        self.check_adherence()
        self.check_continuation()

    def step(self):
        """Full step for standalone use (not inside orchestrator)."""
        if self.sim.t.now() < self.pars.year:
            return
        self._pre_step_hook()
        self.stop_treatment()
        self._assign_ever_seekers_standalone()
        self.determine_care_seeking()
        self.offer_treatment()
        self.assess_treatment_effectiveness()
        self.check_adherence()
        self.check_continuation()

    def _pre_step_hook(self):
        """Override in subclasses for pre-step setup (e.g., hIUD set_states)."""
        pass

    # ------------------------------------------------------------------
    # Treatment offering (used by cascade; pool overrides this)
    # ------------------------------------------------------------------

    def offer_treatment(self):
        """
        Offer this treatment to eligible care seekers.

        Used by the cascade model. The pool model bypasses this entirely
        and uses its own _assign_treatments() instead.
        """
        care_seekers = self.seeking_care.uids
        if len(care_seekers) == 0:
            return

        can_try = self._get_eligible_for_treatment(care_seekers)

        offered = self.pars.prob_offer.filter(can_try)
        if len(offered) > 0:
            self.offered[offered] = True

            threshold = self.pars.prob_accept.pars['p']
            accepts = self.accept_propensity[offered] < threshold
            accepted = offered[accepts]

            if len(accepted) > 0:
                self.accepted[accepted] = True
                self.tried_treatment[accepted] = True
                self._start_treatment(accepted)

    def _get_eligible_for_treatment(self, care_seekers):
        """
        Eligibility: seeking care AND (never tried OR was effective) AND not on treatment.
        """
        return care_seekers & (~self.tried_treatment | self.was_effective) & ~self.on_treatment


# ============================================================================
# SECTION 2: INDIVIDUAL TREATMENT CLASSES
# ============================================================================
# Each subclass sets treatment-specific defaults and implements
# _start_treatment().  Pill and hIUD also integrate with FPsim.
# ============================================================================

class NSAIDTreatment(HMBTreatmentBase):
    """
    NSAID treatment for HMB.

    Prostaglandin inhibition. Taken during menstruation only (use-at-will).
    Efficacy: 33% responder rate.  Adherence: 80%.
    """

    def __init__(self, pars=None, eligibility=None, **kwargs):
        super().__init__(name='nsaid_treatment', eligibility=eligibility)

        self.define_pars(
            year=2020,

            # Care-seeking (defaults; orchestrator overrides these)
            care_behavior=sc.objdict(
                p_ever_seek=0.10,
                p_monthly_first=1/36,
                p_monthly_first_anemic=1/12,
                p_monthly_first_pain=1/24,
                p_monthly_repeat=1/6,
            ),

            # Treatment
            efficacy=0.33,
            adherence=0.80,
            prob_offer=ss.bernoulli(p=0.7),
            prob_accept=ss.bernoulli(p=0.5),

            # Timing
            time_to_assess=ss.months(3),
            dur_treatment=ss.uniform(ss.months(10), ss.months(14)),

            # Use-at-will
            use_at_will=True,
            p_discontinue_nonadherent=0.10,

            # Continuation
            p_continue_first_cycle=0.6,
            p_continue_if_resolved=0.9,
            p_continue_if_persists=0.2,
        )
        self.update_pars(pars, **kwargs)
        self._define_common_states()

    def _start_treatment(self, uids):
        self.on_treatment[uids] = True
        self.ti_start_treatment[uids] = self.ti
        self.treatment_assessed[uids] = False
        self.dur_treatment[uids] = self.pars.dur_treatment.rvs(uids)
        self.ti_stop_treatment[uids] = self.ti + self.dur_treatment[uids]


class TXATreatment(HMBTreatmentBase):
    """
    Tranexamic acid (TXA) treatment for HMB.

    Antifibrinolytic. Taken during menstruation only (use-at-will).
    Efficacy: 45% responder rate.  Adherence: 70% (higher pill burden).
    """

    def __init__(self, pars=None, eligibility=None, **kwargs):
        super().__init__(name='txa_treatment', eligibility=eligibility)

        self.define_pars(
            year=2020,
            care_behavior=sc.objdict(
                p_ever_seek=0.10,
                p_monthly_first=1/36,
                p_monthly_first_anemic=1/12,
                p_monthly_first_pain=1/24,
                p_monthly_repeat=1/6,
            ),
            efficacy=0.45,
            adherence=0.70,
            prob_offer=ss.bernoulli(p=0.7),
            prob_accept=ss.bernoulli(p=0.5),
            time_to_assess=ss.months(3),
            dur_treatment=ss.uniform(ss.months(10), ss.months(14)),
            use_at_will=True,
            p_discontinue_nonadherent=0.10,
            p_continue_first_cycle=0.6,
            p_continue_if_resolved=0.9,
            p_continue_if_persists=0.2,
        )
        self.update_pars(pars, **kwargs)
        self._define_common_states()

    def _start_treatment(self, uids):
        self.on_treatment[uids] = True
        self.ti_start_treatment[uids] = self.ti
        self.treatment_assessed[uids] = False
        self.dur_treatment[uids] = self.pars.dur_treatment.rvs(uids)
        self.ti_stop_treatment[uids] = self.ti + self.dur_treatment[uids]


class PillTreatment(HMBTreatmentBase):
    """
    Combined oral contraceptive pill for HMB.

    Hormonal. Continuous use (not use-at-will). Integrates with FPsim.
    Efficacy: 59% responder rate.  Adherence: 80%.
    Requires no fertility intent (it's a contraceptive).
    """

    def __init__(self, pars=None, eligibility=None, **kwargs):
        super().__init__(name='pill_treatment', eligibility=eligibility)

        self.define_pars(
            year=2020,
            care_behavior=sc.objdict(
                p_ever_seek=0.10,
                p_monthly_first=1/36,
                p_monthly_first_anemic=1/12,
                p_monthly_first_pain=1/24,
                p_monthly_repeat=1/6,
            ),
            efficacy=0.59,
            adherence=0.80,
            prob_offer=ss.bernoulli(p=0.7),
            prob_accept=ss.bernoulli(p=0.5),
            time_to_assess=ss.months(3),
            # use_at_will defaults to False (continuous)
        )
        self.update_pars(pars, **kwargs)
        self._define_common_states()

    @property
    def pill_idx(self):
        return self.sim.connectors.contraception.get_method_by_label('Pill').idx

    def _get_eligible_for_treatment(self, care_seekers):
        """Pill requires no fertility intent."""
        base = super()._get_eligible_for_treatment(care_seekers)
        return base & ~self.sim.people.fp.fertility_intent

    def _start_treatment(self, uids):
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


class hIUDTreatment(HMBTreatmentBase):
    """
    Hormonal IUD treatment for HMB.

    Long-acting hormonal. Continuous. Integrates with FPsim.
    Efficacy: 88% responder rate.  Adherence: 100% (until removal).
    Requires no fertility intent.
    """

    def __init__(self, pars=None, eligibility=None, **kwargs):
        super().__init__(name='hiud_treatment', eligibility=eligibility)

        self.define_pars(
            year=2020,
            care_behavior=sc.objdict(
                p_ever_seek=0.10,
                p_monthly_first=1/36,
                p_monthly_first_anemic=1/12,
                p_monthly_first_pain=1/24,
                p_monthly_repeat=1/6,
            ),
            efficacy=0.88,
            adherence=1.00,
            prob_offer=ss.bernoulli(p=0.7),
            prob_accept=ss.bernoulli(p=0.5),
            time_to_assess=ss.months(3),
            p_hiud=ss.bernoulli(p=0.17),
        )
        self.update_pars(pars, **kwargs)
        self._define_common_states()

        self.define_states(
            ss.BoolState('hiud_prone', label="Prone to use hormonal IUD"),
        )

    @property
    def iud_idx(self):
        return self.sim.connectors.contraception.get_method_by_label('IUDs').idx

    def set_states(self):
        uids = ss.uids(self.hiud_prone.isnan)
        self.hiud_prone[uids] = self.pars.p_hiud.rvs(uids)

    def _pre_step_hook(self):
        self.set_states()

    def _get_eligible_for_treatment(self, care_seekers):
        """hIUD requires no fertility intent."""
        base = super()._get_eligible_for_treatment(care_seekers)
        return base & ~self.sim.people.fp.fertility_intent

    def _start_treatment(self, uids):
        self.on_treatment[uids] = True
        self.ti_start_treatment[uids] = self.ti
        self.treatment_assessed[uids] = False

        self.sim.people.fp.method[uids] = self.iud_idx
        self.sim.people.fp.on_contra[uids] = True
        self.sim.people.fp.ever_used_contra[uids] = True

        method_dur = self.sim.connectors.contraception.set_dur_method(uids)
        self.sim.people.fp.ti_contra[uids] = self.ti + method_dur
        self.dur_treatment[uids] = method_dur
        self.ti_stop_treatment[uids] = self.ti + method_dur


# ============================================================================
# SECTION 3: POOL-BASED TREATMENT ASSIGNMENT (PRIMARY MODEL)
# ============================================================================
# When a woman seeks care, she draws from a weighted distribution to get
# one treatment. This replaces the sequential cascade as the default.
#
# Supports pre/post treatment weights for status quo → intervention shift.
# ============================================================================

class HMBPool(ss.Intervention):
    """
    Pool-based HMB treatment assignment with three-layer care-seeking.

    At each care-seeking visit, one treatment is randomly sampled from
    a weighted distribution. Failed treatments are blocked from re-draw.

    Supports two sets of weights (tx_weights_pre / tx_weights_post) to
    model an immediate shift at intv_year (e.g., introducing hIUD in 2026).
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__(name='hmb_pool', **kwargs)

        self.define_pars(
            year=2020,          # When this intervention starts running
            intv_year=2026,     # When tx_weights shift from pre to post
            time_to_assess=ss.months(3),

            # Treatment weights BEFORE intervention year (status quo)
            # These are relative weights among treated women.
            # "Not treated" is handled by prob_offer (70% = 30% not treated).
            tx_weights_pre=sc.objdict(
                nsaid=0.50,
                txa=0.25,
                pill=0.25,
                hiud=0.0,    # No hIUD in status quo
            ),

            # Treatment weights AFTER intervention year
            # Override this for different hIUD scenarios.
            tx_weights_post=sc.objdict(
                nsaid=0.50,
                txa=0.25,
                pill=0.25,
                hiud=0.0,    # Override to introduce hIUD
            ),

            # Per-treatment parameters
            nsaid=sc.objdict(
                efficacy=0.33,
                adherence=0.80,
            ),
            txa=sc.objdict(
                efficacy=0.45,
                adherence=0.70,
            ),
            pill=sc.objdict(
                efficacy=0.59,
                adherence=0.80,
            ),
            hiud=sc.objdict(
                efficacy=0.88,
                adherence=1.00,
            ),
            
           prob_offer_pre=0.70,    # Status quo: 30% not treated
           prob_offer_post=0.70,   # Override per scenario 

            # Shared acceptance
            prob_accept=1.00,

            # Care-seeking BEFORE intervention year (status quo)
            care_behavior_pre=sc.objdict(
                p_ever_seek=0.10,
                p_monthly_first=1/36,
                p_monthly_first_anemic=1/12,
                p_monthly_first_pain=1/24,
                p_monthly_repeat=1/6,
            ),

            # Care-seeking AFTER intervention year (demand creation)
            care_behavior_post=sc.objdict(
                p_ever_seek=0.10,
                p_monthly_first=1/36,
                p_monthly_first_anemic=1/12,
                p_monthly_first_pain=1/24,
                p_monthly_repeat=1/6,
            ),

            # Repeat cap (Layer 3)
            max_care_episodes=3,
            p_hysterectomy_at_cap=0.10,
        )
        self.update_pars(pars, **kwargs)

        # ── Orchestrator-level states ──
        self.define_states(
            # Layer 1
            ss.BoolState('ever_seeker'),
            ss.BoolState('ever_seeker_assigned'),
            # Layer 3
            ss.FloatArr('care_episodes', default=0),
            ss.BoolState('gave_up'),
            ss.BoolState('had_hysterectomy'),
        )

        # Distributions
        self._p_ever_seek = ss.bernoulli(p=0)
        self._p_hysterectomy = ss.bernoulli(p=0)
        self._p_offer = ss.bernoulli(p=0)

        self.treatments = {}
        return

    # ------------------------------------------------------------------ init

    def init_pre(self, sim):
        super().init_pre(sim)

        shared = dict(
            year=self.pars.year,
            time_to_assess=self.pars.time_to_assess,
            care_behavior=self.pars.care_behavior_pre,  # Start with pre-intervention
        )
        accept_p = self.pars.prob_accept

        self.treatments['nsaid'] = NSAIDTreatment(pars=dict(
            **shared,
            efficacy=self.pars.nsaid.efficacy,
            adherence=self.pars.nsaid.adherence,
            prob_accept=ss.bernoulli(p=accept_p),
        ))
        self.treatments['txa'] = TXATreatment(pars=dict(
            **shared,
            efficacy=self.pars.txa.efficacy,
            adherence=self.pars.txa.adherence,
            prob_accept=ss.bernoulli(p=accept_p),
        ))
        self.treatments['pill'] = PillTreatment(pars=dict(
            **shared,
            efficacy=self.pars.pill.efficacy,
            adherence=self.pars.pill.adherence,
            prob_accept=ss.bernoulli(p=accept_p),
        ))
        self.treatments['hiud'] = hIUDTreatment(pars=dict(
            **shared,
            efficacy=self.pars.hiud.efficacy,
            adherence=self.pars.hiud.adherence,
            prob_accept=ss.bernoulli(p=accept_p),
        ))

        for tx in self.treatments.values():
            tx.init_pre(sim)

        # Pre-compute weight arrays
        self._tx_names = list(self.treatments.keys())
        self._n_tx = len(self._tx_names)
        self._tx_probs_pre = self._normalize_weights(self.pars.tx_weights_pre)
        self._tx_probs_post = self._normalize_weights(self.pars.tx_weights_post)

    def _normalize_weights(self, weights):
        """Normalize weight dict to probability array. Zero-sum returns uniform."""
        raw = np.array([weights[k] for k in self._tx_names], dtype=float)
        total = raw.sum()
        if total == 0:
            # All weights zero — no treatment can be offered
            return np.zeros(self._n_tx)
        return raw / total

    def init_post(self):
        super().init_post()
        for tx in self.treatments.values():
            tx.init_post()
        # For episode detection (Layer 3)
        self._prev_any_seeking_uids = set()

    # ------------------------------------------------------------------ step

    def step(self):
        """
        One timestep of the pool model.

        Order:
          1. Pre-step hooks (e.g., hIUD set_states)
          2. Stop expired treatments
          3. Layer 1: assign ever-seekers to new HMB women
          4. Layer 2: centralised care-seeking (one determination, shared)
          5. Layer 3: enforce episode cap
          6. Random treatment assignment from pool
          7. Assess treatment effectiveness
          8. Check adherence
          9. Continuation checks (use-at-will)
        """
        if self.sim.t.now() < self.pars.year:
            return
        
        # 0. Select time-appropriate care behavior and push to child treatments
        if self.sim.t.now() >= self.pars.intv_year:
            active_cb = self.pars.care_behavior_post
        else:
            active_cb = self.pars.care_behavior_pre

        for tx in self.treatments.values():
            tx.pars.care_behavior = active_cb

        # 1. Pre-step hooks
        for tx in self.treatments.values():
            tx._pre_step_hook()

        # 2. Stop expired treatments
        for tx in self.treatments.values():
            tx.stop_treatment()

        # 3. Layer 1: ever-seeker assignment
        self._assign_ever_seekers()

        # 4. Layer 2: centralised care-seeking
        #    Use NSAID's logic as the shared determination, then copy to all.
        ref_tx = self.treatments['nsaid']
        ref_tx.determine_care_seeking()
        seeking_uids = ref_tx.seeking_care.uids

        for name, tx in self.treatments.items():
            if name != 'nsaid':
                tx.seeking_care[:] = False
                tx.seeking_care[seeking_uids] = True
                tx.ever_sought_care[seeking_uids] = True

        # 5. Layer 3: episode cap
        self._enforce_episode_cap()

        # 6. Treatment assignment (re-read seeking after cap zeroed some out)
        seeking_uids = ref_tx.seeking_care.uids
        self._assign_treatments(seeking_uids)

        # 7–9. Assess, adherence, continuation
        for tx in self.treatments.values():
            tx.assess_treatment_effectiveness()
        for tx in self.treatments.values():
            tx.check_adherence()
        for tx in self.treatments.values():
            tx.check_continuation()

    # ------------------------------------------------ Layer 1: ever-seeker

    def _assign_ever_seekers(self):
        """
        One-time draw for new HMB women: will she ever seek care?

        Bernoulli(p_ever_seek). Women who get False never enter care.
        Runs each step to catch new HMB cases (onset, births aging in).
        """
        ppl = self.sim.people
        new_hmb = (ppl.menstruation.hmb &
                   ppl.menstruation.menstruating &
                   ~self.ever_seeker_assigned).uids

        if len(new_hmb) == 0:
            return

        self.ever_seeker_assigned[new_hmb] = True
        
        if self.sim.t.now() >= self.pars.intv_year:
            p_es = self.pars.care_behavior_post.p_ever_seek
        else:
            p_es = self.pars.care_behavior_pre.p_ever_seek

        self._p_ever_seek.set(p_es)
        
        assigned_true = self._p_ever_seek.filter(new_hmb)
        self.ever_seeker[assigned_true] = True

    # ------------------------------------------------ Layer 3: episode cap

    def _get_anyone_seeking_uids(self):
        """Set of UIDs seeking care in any treatment."""
        uids = set()
        all_uids = np.asarray(self.sim.people.uid)
        for tx in self.treatments.values():
            seeking = np.asarray(tx.seeking_care)
            uids.update(all_uids[seeking])
        return uids

    def _enforce_episode_cap(self):
        """
        Detect new care-seeking episodes and enforce repeat cap.

        New episode = transition from not-seeking to seeking.
        After max_care_episodes:
          90% give up permanently
          10% get hysterectomy (HMB removed permanently)
        """
        now_seeking = self._get_anyone_seeking_uids()

        newly_seeking = np.array(
            list(now_seeking - self._prev_any_seeking_uids), dtype=int
        )
        self._prev_any_seeking_uids = set(now_seeking)

        if len(newly_seeking) == 0:
            return

        self.care_episodes[newly_seeking] += 1

        over_cap = newly_seeking[
            self.care_episodes[newly_seeking] > self.pars.max_care_episodes
        ]
        if len(over_cap) == 0:
            return

        # Route: hysterectomy vs give up
        self._p_hysterectomy.set(self.pars.p_hysterectomy_at_cap)
        hyst_uids = self._p_hysterectomy.filter(over_cap)
        giveup_uids = np.setdiff1d(over_cap, hyst_uids)

        self.had_hysterectomy[hyst_uids] = True
        self.gave_up[giveup_uids] = True

        # Zero out seeking so they don't enter treatment
        for tx in self.treatments.values():
            tx.seeking_care[over_cap] = False
        self._prev_any_seeking_uids -= set(over_cap)

        # Hysterectomy removes HMB
        self.sim.people.menstruation.hmb[hyst_uids] = False

    # ------------------------------------------------ Treatment assignment
    
    def _assign_treatments(self, seeking_uids):
        """
        For each care-seeker not on treatment:
            1. Global prob_offer gate
            2. Among those offered, random draw from tx_weights
            3. If assigned treatment is ineligible, redraw from remaining
            4. Accept based on propensity
        """
        if len(seeking_uids) == 0:
                return

        on_any = self.on_any_treatment
        candidates = seeking_uids[~on_any[seeking_uids]]
        if len(candidates) == 0:
            return

        # ── Global prob_offer gate ──
        if self.sim.t.now() >= self.pars.intv_year:
            offer_p = self.pars.prob_offer_post
        else:
            offer_p = self.pars.prob_offer_pre

        self._p_offer.set(offer_p)
        offered_any = self._p_offer.filter(candidates)
        if len(offered_any) == 0:
            return

        # Select weight set
        if self.sim.t.now() >= self.pars.intv_year:
            tx_probs = self._tx_probs_post
        else:
            tx_probs = self._tx_probs_pre

        if tx_probs.sum() == 0:
            return

        # ── Per-person eligible weights and draw ──
        # Build eligibility mask per treatment per person
        n = len(offered_any)
        eligible_mask = np.ones((n, self._n_tx), dtype=bool)

        for tx_idx in range(self._n_tx):
            tx_name = self._tx_names[tx_idx]
            tx = self.treatments[tx_name]

            # Tried-and-failed → ineligible
            tried = np.asarray(tx.tried_treatment[offered_any])
            was_eff = np.asarray(tx.was_effective[offered_any])
            tried_and_failed = tried & ~was_eff
            eligible_mask[tried_and_failed, tx_idx] = False

            # Pill/hIUD need no fertility intent
            if tx_name in ('pill', 'hiud'):
                fert = np.asarray(self.sim.people.fp.fertility_intent[offered_any])
                eligible_mask[fert, tx_idx] = False

        # Build per-person probability vectors
        # Start with global weights, zero out ineligible, renormalize
        person_probs = np.tile(tx_probs, (n, 1))  # (n, n_tx)
        person_probs[~eligible_mask] = 0.0

        # Renormalize per person
        row_sums = person_probs.sum(axis=1, keepdims=True)
        has_options = (row_sums > 0).flatten()

        if not has_options.any():
            return

        # Only draw for people who have at least one eligible treatment
        draw_uids = offered_any[has_options]
        draw_probs = person_probs[has_options]
        draw_probs = draw_probs / draw_probs.sum(axis=1, keepdims=True)

        # Draw one treatment per person
        draws = np.array([
            np.random.choice(self._n_tx, p=p) for p in draw_probs
        ])

        # ── Route to treatments ──
        for tx_idx in range(self._n_tx):
            tx_name = self._tx_names[tx_idx]
            tx = self.treatments[tx_name]
            assigned_mask = draws == tx_idx
            assigned_uids = draw_uids[assigned_mask]

            if len(assigned_uids) == 0:
                continue

            tx.offered[assigned_uids] = True

            # Accept (propensity < threshold)
            threshold = tx.pars.prob_accept.pars['p']
            accepts = tx.accept_propensity[assigned_uids] < threshold
            accepted = assigned_uids[accepts]
            if len(accepted) == 0:
                continue

            tx.accepted[accepted] = True
            tx.tried_treatment[accepted] = True
            tx._start_treatment(accepted)
        
    # ------------------------------------------------ Properties

    @property
    def on_any_treatment(self):
        on_any = self.treatments['nsaid'].on_treatment.copy()
        for name in ['txa', 'pill', 'hiud']:
            on_any |= self.treatments[name].on_treatment
        return on_any

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
    def n_gave_up(self):
        return np.count_nonzero(self.gave_up)

    @property
    def n_hysterectomy(self):
        return np.count_nonzero(self.had_hysterectomy)

    def finalize(self):
        super().finalize()
        for tx in self.treatments.values():
            tx.finalize()


# ============================================================================
# SECTION 4: COUNTERFACTUAL (STATUS QUO, NO hIUD)
# ============================================================================
# Runs from 2020. Same pool model but with hIUD weight = 0 always.
# This is the baseline that intervention scenarios are compared against.
# ============================================================================

class HMBCounterfactual(HMBPool):
    """
    Status quo: pool model with no hIUD, running from start of sim.

    Uses Darcy's baseline parameters:
      10% ever-seek, 1/36 first-visit rate, 70% receipt,
      50/25/25 NSAID/TXA/Pill split, no hIUD.
    """

    def __init__(self, pars=None, **kwargs):
        default_pars = dict(
            year=2020,
            intv_year=9999,
            prob_offer_pre=0.70,
            prob_offer_post=0.70,
            care_behavior_pre=sc.objdict(
                p_ever_seek=0.10,
                p_monthly_first=1/36,
                p_monthly_first_anemic=1/12,
                p_monthly_first_pain=1/24,
                p_monthly_repeat=1/6,
            ),
            care_behavior_post=sc.objdict(  # Same as pre — never changes
                p_ever_seek=0.10,
                p_monthly_first=1/36,
                p_monthly_first_anemic=1/12,
                p_monthly_first_pain=1/24,
                p_monthly_repeat=1/6,
            ),
            tx_weights_pre=sc.objdict(nsaid=0.50, txa=0.25, pill=0.25, hiud=0.0),
            tx_weights_post=sc.objdict(nsaid=0.50, txa=0.25, pill=0.25, hiud=0.0),
        )
        merged = sc.mergedicts(default_pars, pars)
        merged.setdefault('tx_weights_pre', {})['hiud'] = 0.0
        merged.setdefault('tx_weights_post', {})['hiud'] = 0.0
        super().__init__(pars=merged, **kwargs)
        self.name = 'hmb_counterfactual'


# ============================================================================
# SECTION 5: CASCADE MODEL (OPTIONAL, FOR COMPARISON)
# ============================================================================
# Sequential: NSAID → TXA → Pill → hIUD.
# Each tier requires failure/refusal of the previous.
# Uses same three-layer care-seeking and adherence as the pool.
# ============================================================================

# class HMBCascade(ss.Intervention):
#     """
#     Sequential cascade with three-layer care-seeking.

#     Treatment order: NSAID → TXA → Pill → hIUD.
#     Each tier's eligibility function requires trying/refusing the previous.
#     """

#     def __init__(self, pars=None, **kwargs):
#         super().__init__(name='hmb_cascade', **kwargs)

#         self.define_pars(
#             year=2020,
#             time_to_assess=ss.months(3),

#             nsaid=sc.objdict(efficacy=0.33, adherence=0.80,
#                              prob_offer=ss.bernoulli(p=0.7),
#                              prob_accept=ss.bernoulli(p=0.7)),
#             txa=sc.objdict(efficacy=0.45, adherence=0.70,
#                            prob_offer=ss.bernoulli(p=0.7),
#                            prob_accept=ss.bernoulli(p=0.6)),
#             pill=sc.objdict(efficacy=0.59, adherence=0.80,
#                             prob_offer=ss.bernoulli(p=0.7),
#                             prob_accept=ss.bernoulli(p=0.5)),
#             hiud=sc.objdict(efficacy=0.88, adherence=1.00,
#                             prob_offer=ss.bernoulli(p=0.7),
#                             prob_accept=ss.bernoulli(p=0.5)),

#             care_behavior=sc.objdict(
#                 p_ever_seek=0.10,
#                 p_monthly_first=1/36,
#                 p_monthly_first_anemic=1/12,
#                 p_monthly_first_pain=1/24,
#                 p_monthly_repeat=1/6,
#             ),

#             max_care_episodes=3,
#             p_hysterectomy_at_cap=0.10,
#         )
#         self.update_pars(pars, **kwargs)

#         # Orchestrator-level states (same as pool)
#         self.define_states(
#             ss.BoolState('ever_seeker'),
#             ss.BoolState('ever_seeker_assigned'),
#             ss.FloatArr('care_episodes', default=0),
#             ss.BoolState('gave_up'),
#             ss.BoolState('had_hysterectomy'),
#         )

#         self._p_ever_seek = ss.bernoulli(p=0)
#         self._p_hysterectomy = ss.bernoulli(p=0)
#         self.treatments = {}
#         return

#     def init_pre(self, sim):
#         super().init_pre(sim)

#         shared = dict(
#             year=self.pars.year,
#             time_to_assess=self.pars.time_to_assess,
#             care_behavior=self.pars.care_behavior,
#         )

#         nsaid = NSAIDTreatment(pars=dict(**shared,
#             efficacy=self.pars.nsaid.efficacy,
#             adherence=self.pars.nsaid.adherence,
#             prob_offer=self.pars.nsaid.prob_offer,
#             prob_accept=self.pars.nsaid.prob_accept))

#         def txa_elig(sim):
#             return nsaid.tried_treatment | (nsaid.offered & ~nsaid.on_treatment)

#         txa = TXATreatment(pars=dict(**shared,
#             efficacy=self.pars.txa.efficacy,
#             adherence=self.pars.txa.adherence,
#             prob_offer=self.pars.txa.prob_offer,
#             prob_accept=self.pars.txa.prob_accept),
#             eligibility=txa_elig)

#         def pill_elig(sim):
#             return (txa.tried_treatment |
#                     (txa.offered & ~txa.on_treatment) |
#                     (nsaid.tried_treatment & ~nsaid.on_treatment & ~txa.offered))

#         pill = PillTreatment(pars=dict(**shared,
#             efficacy=self.pars.pill.efficacy,
#             adherence=self.pars.pill.adherence,
#             prob_offer=self.pars.pill.prob_offer,
#             prob_accept=self.pars.pill.prob_accept),
#             eligibility=pill_elig)

#         def hiud_elig(sim):
#             return (pill.tried_treatment |
#                     (pill.offered & ~pill.on_treatment) |
#                     (nsaid.tried_treatment & ~nsaid.on_treatment & ~txa.offered) |
#                     (txa.tried_treatment & ~txa.on_treatment & ~pill.offered))

#         hiud = hIUDTreatment(pars=dict(**shared,
#             efficacy=self.pars.hiud.efficacy,
#             adherence=self.pars.hiud.adherence,
#             prob_offer=self.pars.hiud.prob_offer,
#             prob_accept=self.pars.hiud.prob_accept),
#             eligibility=hiud_elig)

#         self.treatments = {'nsaid': nsaid, 'txa': txa, 'pill': pill, 'hiud': hiud}
#         for tx in self.treatments.values():
#             tx.init_pre(sim)

#     def init_post(self):
#         super().init_post()
#         for tx in self.treatments.values():
#             tx.init_post()
#         self._prev_any_seeking_uids = set()

#     # Reuse the same Layer 1 and Layer 3 logic as pool
#     def _assign_ever_seekers(self):
#         ppl = self.sim.people
#         new_hmb = (ppl.menstruation.hmb &
#                    ppl.menstruation.menstruating &
#                    ~self.ever_seeker_assigned).uids
#         if len(new_hmb) == 0:
#             return
#         self.ever_seeker_assigned[new_hmb] = True
#         self._p_ever_seek.set(self.pars.care_behavior.p_ever_seek)
#         assigned_true = self._p_ever_seek.filter(new_hmb)
#         self.ever_seeker[assigned_true] = True

#     def _get_anyone_seeking_uids(self):
#         uids = set()
#         all_uids = np.asarray(self.sim.people.uid)
#         for tx in self.treatments.values():
#             seeking = np.asarray(tx.seeking_care)
#             uids.update(all_uids[seeking])
#         return uids

#     def _enforce_episode_cap(self):
#         now_seeking = self._get_anyone_seeking_uids()
#         newly_seeking = np.array(
#             list(now_seeking - self._prev_any_seeking_uids), dtype=int)
#         self._prev_any_seeking_uids = set(now_seeking)

#         if len(newly_seeking) == 0:
#             return
#         self.care_episodes[newly_seeking] += 1

#         over_cap = newly_seeking[
#             self.care_episodes[newly_seeking] > self.pars.max_care_episodes]
#         if len(over_cap) == 0:
#             return

#         self._p_hysterectomy.set(self.pars.p_hysterectomy_at_cap)
#         hyst_uids = self._p_hysterectomy.filter(over_cap)
#         giveup_uids = np.setdiff1d(over_cap, hyst_uids)

#         self.had_hysterectomy[hyst_uids] = True
#         self.gave_up[giveup_uids] = True

#         for tx in self.treatments.values():
#             tx.seeking_care[over_cap] = False
#         self._prev_any_seeking_uids -= set(over_cap)
#         self.sim.people.menstruation.hmb[hyst_uids] = False

#     def step(self):
#         """Cascade step: Layer 1 → seek → cap → offer/assess/adherence/continue."""
#         self._assign_ever_seekers()

#         for tx in self.treatments.values():
#             tx.step_seek()

#         self._enforce_episode_cap()

#         for tx in self.treatments.values():
#             tx.step_treat()

#     def finalize(self):
#         super().finalize()
#         for tx in self.treatments.values():
#             tx.finalize()

#     @property
#     def on_any_treatment(self):
#         on_any = self.treatments['nsaid'].on_treatment.copy()
#         for name in ['txa', 'pill', 'hiud']:
#             on_any |= self.treatments[name].on_treatment
#         return on_any

#     @property
#     def n_gave_up(self):
#         return np.count_nonzero(self.gave_up)

#     @property
#     def n_hysterectomy(self):
#         return np.count_nonzero(self.had_hysterectomy)


# ============================================================================
# SECTION 6: FACTORY FUNCTIONS
# ============================================================================

def make_pool_sim(seed=0, pool_pars=None, counterfactual=False):
    """
    Create a simulation with the pool (or counterfactual) intervention.

    Args:
        seed: Random seed
        pool_pars: Dict of overrides for HMBPool/HMBCounterfactual
        counterfactual: If True, use HMBCounterfactual (no hIUD)
    """
    import fpsim as fp
    from menstruation import Menstruation
    from education import Education
    from analyzers import track_hmb_anemia

    mens = Menstruation()
    edu = Education()

    if counterfactual:
        intervention = HMBCounterfactual(pars=pool_pars)
    else:
        intervention = HMBPool(pars=pool_pars)

    sim = fp.Sim(
        start=2020, stop=2030,
        n_agents=5000, total_pop=55_000_000,
        location='kenya',
        education_module=edu,
        connectors=[mens],
        interventions=[intervention],
        analyzers=[track_hmb_anemia()],
        rand_seed=seed, verbose=0,
    )
    return sim


# def make_cascade_sim(seed=0, **cascade_kwargs):
#     """Create a simulation with the cascade intervention."""
#     import fpsim as fp
#     from menstruation import Menstruation
#     from education import Education
#     from analyzers import track_hmb_anemia

#     mens = Menstruation()
#     edu = Education()
#     cascade = HMBCascade(**cascade_kwargs)

#     sim = fp.Sim(
#         start=2020, stop=2030,
#         n_agents=5000, total_pop=55_000_000,
#         location='kenya',
#         education_module=edu,
#         connectors=[mens],
#         interventions=[cascade],
#         analyzers=[track_hmb_anemia()],
#         rand_seed=seed, verbose=0,
#     )
#     return sim